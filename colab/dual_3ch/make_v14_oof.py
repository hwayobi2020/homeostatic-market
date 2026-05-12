"""v14 MTL 5-fold TimeSeriesSplit OOF (Out-Of-Fold) predictions on train CSV.

Paper-grade OOF integrity for Conditional Flow cond:
  v14 (Deep MTL Transformer, base_mtl_metab) 의 train 부분 prediction 도 OOS-strength signal
  로 만들기 위해 5-fold TimeSeriesSplit. cond Flow 학습 시 train cond 와 test cond 가
  signal strength 일관성 (둘 다 OOS).

구조 (train_vol_pilot_3m_mtl.py 와 정확히 동일 — 동일 model / loss / hyperparam):
  - Model       : CausalTransformerVolMTL (shared encoder + vol head + metab aux head)
  - cond cols   : v14 = [tbill_wr, m2_13w_cum_lag, ads_lag, cpi_13w_cum_lag, sp_std_13w,
                          wti_wr, sp_log_std_13w]
  - aux  col    : metab_13w
  - λ_aux       : 1.0
  - PAST_LEN    : 52,  FUTURE_LEN: 13
  - epochs/batch/lr: 60 / 32 / 1e-4  (paper main 동일)
  - early stop  : val MSE patience=30 (TimeSeriesSplit 의 val = held-out fold)

Method (expanding-window TimeSeriesSplit):
  origins = train.csv 의 모든 origin (n_w = n − L + 1)
  for k in 1..5:
    tr_origins = origins[0 : split_k]   (확장)
    va_origins = origins[split_k : split_{k+1}]
    임시 train_part.csv  ←  csv 의 (origin 0 의 첫 row) ... (origin tr 의 last L row)
    임시 val_part.csv    ←  va 의 first L row .. last row
    학습 (1 seed) → va prediction 저장

  결과: 첫 chunk 를 제외한 모든 train origin 에 대한 OOS prediction.

Output:
  result/v14_{fold}_train_oof.csv   (date, target_log_std, pred_log_std_oof)

Usage:
  python colab/dual_3ch/make_v14_oof.py --fold F1 --variant 14 --seed 42

Note (paper limitation):
  • 1 seed OOF (vs main pipeline 의 5 seed ensemble). OOF 는 individual model variance 더 큼.
  • TimeSeriesSplit fold 1 은 짧은 train (n_origin / 5) — fold k 별 prediction 신뢰도 다름.
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

# Reuse v14 components (no redefinition — exact paper main compatibility)
from train_vol_pilot_3m_mtl import (   # noqa: E402
    PAST_LEN, FUTURE_LEN, L,
    D_MODEL, N_HEADS, N_LAYERS,
    LR, BATCH, MAX_EPOCHS, PATIENCE,
    VARIANTS, build_spec,
    CausalTransformerVolMTL,
    load_windows_mtl, mask_future_channels,
    pearson_corr,
)


def fit_one_split(train_csv_part, val_csv_part, spec, seed, device,
                  max_epochs=MAX_EPOCHS, patience=PATIENCE, batch=BATCH, lr=LR,
                  lambda_aux=1.0, verbose_every=15):
    """v14 MTL 학습 1 회 + val prediction 반환.

    Returns:
        v_val_best : (n_val,) val OOF prediction (best val mse 시점)
        meta       : dict
    """
    torch.manual_seed(seed); np.random.seed(seed)

    cols_cond = spec["cols_cond"]
    cols_target = spec["cols_target"]
    aux_col = spec["aux_col"]
    mask_future = spec["mask_future_ch"]

    Xtr_p, Ctr, ytr, atr, sc, sa = load_windows_mtl(
        train_csv_part, cols_cond, cols_target, aux_col, L=L)
    Xv_p, Cv, yv, av, _, _ = load_windows_mtl(
        val_csv_part, cols_cond, cols_target, aux_col, L=L,
        cond_stats=sc, aux_stats=sa)

    if Xv_p is None or Xv_p.shape[0] == 0:
        return None, dict(error="no_val_windows", n_val=0)

    Ctr_m = mask_future_channels(Ctr, PAST_LEN, mask_future)
    Cv_m  = mask_future_channels(Cv,  PAST_LEN, mask_future)

    model = CausalTransformerVolMTL(
        d_cond=len(cols_cond), d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        past_len=PAST_LEN, future_len=FUTURE_LEN,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    train_ds = TensorDataset(Xtr_p, Ctr_m, ytr, atr)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)

    def eval_val():
        model.eval()
        with torch.no_grad():
            v, _ = model(Cv_m.to(device), Xv_p.to(device))
            v = v.cpu().numpy()
        mse = float(np.mean((yv.cpu().numpy() - v) ** 2))
        return v, mse

    best_val_mse = float("inf")
    best_val_pred = None
    best_ep = -1
    pat = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, cb, yb, ab in train_dl:
            xb = xb.to(device); cb = cb.to(device); yb = yb.to(device); ab = ab.to(device)
            v_pred, a_pred = model(cb, xb)
            loss_v = ((v_pred - yb) ** 2).mean()
            loss_a = ((a_pred - ab) ** 2).mean()
            loss = loss_v + lambda_aux * loss_a
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v_pred, val_mse = eval_val()
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_pred = v_pred.copy()
            best_ep = epoch
            pat = 0
        else:
            pat += 1
        if epoch == 1 or epoch % verbose_every == 0 or epoch == max_epochs or pat == 0 and epoch <= 5:
            print(f"      ep{epoch:>3d}  val_mse={val_mse:.4f}  best={best_val_mse:.4f} @ep{best_ep}")
        if pat >= patience:
            print(f"      [early stop] ep{epoch} (patience {patience} from best ep{best_ep})")
            break

    p_val = pearson_corr(yv.cpu().numpy(), best_val_pred)
    return best_val_pred, dict(n_val=len(best_val_pred), best_val_mse=best_val_mse,
                               best_ep=best_ep, pearson=p_val)


def main():
    ap = argparse.ArgumentParser(description="v14 MTL 5-fold TimeSeriesSplit OOF on train")
    ap.add_argument("--fold", required=True, help="F1 / F2 / F3")
    ap.add_argument("--variant", type=int, default=14,
                    help="VARIANTS key (default 14 = paper main mtl_base)")
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--out-dir",   default=os.path.join(HERE, "result"))
    ap.add_argument("--n-splits",  type=int, default=5)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--max-epochs",type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience",  type=int, default=PATIENCE)
    args = ap.parse_args()

    train_csv = os.path.join(args.folds_dir, f"{args.fold}_train.csv")
    if not os.path.exists(train_csv):
        sys.exit(f"[FATAL] {train_csv} not found")
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"v14_{args.fold}_train_oof.csv")

    spec = build_spec(args.variant)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f" v14 MTL 5-fold TimeSeriesSplit OOF | fold={args.fold} | variant={args.variant} ({spec['name']})")
    print(f"  cond  = {spec['cols_cond']}")
    print(f"  aux   = {spec['aux_col']}")
    print(f"  seed={args.seed}, max_epochs={args.max_epochs}, patience={args.patience}, device={device}")
    print("=" * 78)

    # Load full train CSV → enumerate origins
    df = pd.read_csv(train_csv, parse_dates=["date"])
    n = len(df)
    n_w = n - L + 1
    origin_indices = np.arange(n_w)
    origin_dates = pd.to_datetime(
        df["date"].iloc[(PAST_LEN - 1) : (PAST_LEN - 1 + n_w)].values)
    print(f"  csv rows = {n},  n_origin = {n_w}  (PAST_LEN={PAST_LEN}, FUTURE_LEN={FUTURE_LEN})")

    # Target for OOF reporting (log std future 13w)
    sp = df["sp_return"].values
    y_log_std_all = np.full(n_w, np.nan, dtype=np.float64)
    for t in range(n_w):
        fut = sp[t + PAST_LEN : t + PAST_LEN + FUTURE_LEN]
        if np.any(np.isnan(fut)):
            continue
        y_log_std_all[t] = float(np.log(max(np.std(fut, ddof=1), 1e-8)))

    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    oof_pred = np.full(n_w, np.nan, dtype=np.float64)
    fold_stats = []

    for k, (tr_orig_idx, va_orig_idx) in enumerate(tscv.split(origin_indices), start=1):
        print(f"\n  [Fold {k}/{args.n_splits}] tr_origin={len(tr_orig_idx)}, va_origin={len(va_orig_idx)}")
        # csv row 범위 산출:
        #   train_part covers rows  [0  .. tr_last + L - 1)  so that all tr origins fit
        #   val_part   covers rows  [va_first .. va_last + L - 1)
        tr_last = int(tr_orig_idx.max())
        va_first = int(va_orig_idx.min())
        va_last  = int(va_orig_idx.max())
        train_rows = slice(0, tr_last + L)            # exclusive end
        val_rows   = slice(va_first, va_last + L)
        tr_part_df = df.iloc[train_rows].reset_index(drop=True)
        va_part_df = df.iloc[val_rows].reset_index(drop=True)
        print(f"    train_part csv rows: 0 .. {tr_last + L - 1}  (n_csv={len(tr_part_df)})")
        print(f"    val_part   csv rows: {va_first} .. {va_last + L - 1}  (n_csv={len(va_part_df)})")
        # Save temp CSVs (load_windows_mtl is path-based)
        with tempfile.TemporaryDirectory() as tmpd:
            tr_path = os.path.join(tmpd, f"tr_k{k}.csv")
            va_path = os.path.join(tmpd, f"va_k{k}.csv")
            tr_part_df.to_csv(tr_path, index=False)
            va_part_df.to_csv(va_path, index=False)
            val_pred, meta = fit_one_split(tr_path, va_path, spec, args.seed, device,
                                           max_epochs=args.max_epochs, patience=args.patience)
        if val_pred is None:
            print(f"    [FAIL] {meta.get('error')}")
            continue
        oof_pred[va_orig_idx] = val_pred
        fold_stats.append((k, len(tr_orig_idx), len(va_orig_idx),
                           meta["best_val_mse"], meta["best_ep"], meta["pearson"]))
        print(f"    fold {k}: best_val_mse={meta['best_val_mse']:.4f} @ep{meta['best_ep']}, "
              f"Pearson={meta['pearson']:+.3f}")

    # Aggregate OOF metrics
    mask = ~np.isnan(oof_pred) & ~np.isnan(y_log_std_all)
    n_oof = int(mask.sum())
    if n_oof == 0:
        sys.exit("[FATAL] no OOF predictions produced")
    y_all = y_log_std_all[mask]; y_hat_all = oof_pred[mask]
    mse_all = float(np.mean((y_all - y_hat_all) ** 2))
    ss_res = float(np.sum((y_all - y_hat_all) ** 2))
    ss_tot = float(np.sum((y_all - y_all.mean()) ** 2))
    r2_all = 1.0 - ss_res / max(ss_tot, 1e-12)
    a = y_all - y_all.mean(); b = y_hat_all - y_hat_all.mean()
    denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    p_all = float((a * b).sum() / denom) if denom > 1e-12 else float("nan")
    print(f"\n  OOF aggregate (excl. warm-up chunk): n_oof = {n_oof}/{n_w}")
    print(f"    MSE={mse_all:.4f}  R²={r2_all:+.3f}  Pearson={p_all:+.3f}")

    # Save OOF CSV
    out_df = pd.DataFrame({
        "date": origin_dates,
        "target_log_std": y_log_std_all,
        "pred_log_std_oof": oof_pred,
    }).dropna(subset=["pred_log_std_oof"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n  saved v14 OOF: {out_path}  (n={len(out_df)})")


if __name__ == "__main__":
    main()
