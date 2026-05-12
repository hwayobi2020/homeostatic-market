"""Residual Learning (Gradient Boosting 패턴) — HAR-RV baseline + v14 MTL.

Sequential boosting:
  Step 1: HAR-RV (OLS, sp_log_std_13w + sp_std_13w) fit on train.
          → predict train/val/test 모두 (train_har_rv.py 에 저장됨).
  Step 2: v14 MTL 학습 — main target = residual r = y_actual - y_HAR.
          aux target = metab_13w (그대로).
  Step 3: Final test prediction = HAR_test_pred + v14_residual_pred.

이론적 정합:
  HAR-RV 가 vol clustering (autoregression mean) 잡음.
  v14 가 OOD/위기 시기의 macro 갭만 학습 → distribution shift 영향 작음.
  Pearson + MSE 둘 다 향상 기대.

Citations:
  Kim & Won (2018) "Forecasting the volatility of stock price index: A hybrid model
                   integrating LSTM with multiple GARCH-type models." ESWA 103:25-37.
  Bucci (2020) "Realized Volatility Forecasting with Neural Networks." JFE 18(3): 502-531.

Output prefix: vol_pilot_3m_mtl_residual_msel_v{id}_{name}_{fold}_seed{N}_*
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_vol_pilot_3m_mtl import (   # noqa: E402
    CausalTransformerVolMTL, mask_future_channels,
    pearson_corr, spearman_corr,
    VARIANTS, build_spec,
    PAST_LEN, FUTURE_LEN, L, D_MODEL, N_HEADS, N_LAYERS, LR, BATCH,
    MAX_EPOCHS, PATIENCE, GRAD_CLIP, MIN_EPOCH,
)

ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def load_har_predictions(fold, result_dir):
    """Load HAR-RV train/val/test predictions (saved by train_har_rv.py)."""
    out = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(result_dir, f"har_rv_{fold}_{split}_predictions.csv")
        if not os.path.exists(csv_path):
            sys.exit(f"[FATAL] HAR-RV {split} preds 없음: {csv_path}\n"
                     "  → 먼저 train_har_rv.py --fold {fold} 실행 필요.")
        df = pd.read_csv(csv_path, parse_dates=["date"])
        out[split] = df
    return out


def load_windows_residual(csv_path, cols_cond, cols_target, aux_col, har_pred_df, L=65,
                          cond_stats=None, aux_stats=None):
    """Load windows + residual target (y_actual - y_HAR).

    HAR pred 는 origin t 의 target row (csv index t+L-1) 의 timestamp 에 대응.
    csv 의 origin 순서 (i = 0..n_w-1) 에 맞춰 HAR pred 정렬.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        return None, None, None, None, None, None, None

    # Build HAR prediction lookup by date
    har_pred_lookup = dict(zip(har_pred_df["date"].astype(str), har_pred_df["pred_log_std"].values))

    X = np.zeros((n_w, L, len(cols_target)), dtype=np.float32)
    C = np.zeros((n_w, L, len(cols_cond)), dtype=np.float32)
    aux = np.zeros((n_w,), dtype=np.float32)
    y_har = np.zeros((n_w,), dtype=np.float32)
    valid_mask = np.ones((n_w,), dtype=bool)

    aux_vals = df[aux_col].values.astype(np.float32)
    for i in range(n_w):
        X[i] = df[cols_target].iloc[i : i + L].values
        C[i] = df[cols_cond].iloc[i : i + L].values
        aux[i] = aux_vals[i + L - 1]
        # HAR pred 는 origin t 의 last past row (csv index t+PAST_LEN-1) 의 date 에 대응
        origin_date = str(pd.Timestamp(df["date"].iloc[i + PAST_LEN - 1]).date())
        if origin_date in har_pred_lookup:
            y_har[i] = har_pred_lookup[origin_date]
        else:
            valid_mask[i] = False

    # Drop invalid (HAR pred 없는 origin)
    if not valid_mask.all():
        n_invalid = int((~valid_mask).sum())
        print(f"    [warn] dropping {n_invalid}/{n_w} windows (no HAR pred at origin date)")
        X = X[valid_mask]
        C = C[valid_mask]
        aux = aux[valid_mask]
        y_har = y_har[valid_mask]
        n_w = int(valid_mask.sum())

    # cond stats
    if cond_stats is None:
        cmu = C.reshape(-1, len(cols_cond)).mean(axis=0)
        csd = C.reshape(-1, len(cols_cond)).std(axis=0) + 1e-8
        cond_stats_out = {"mean": cmu.tolist(), "std": csd.tolist()}
    else:
        cmu = np.asarray(cond_stats["mean"], dtype=np.float32)
        csd = np.asarray(cond_stats["std"], dtype=np.float32)
        cond_stats_out = cond_stats
    C = (C - cmu) / csd

    # aux stats
    if aux_stats is None:
        amu = float(aux.mean()); asd = float(aux.std() + 1e-8)
        aux_stats_out = {"mean": amu, "std": asd}
    else:
        amu = aux_stats["mean"]; asd = aux_stats["std"]
        aux_stats_out = aux_stats
    aux_std = (aux - amu) / asd

    # Compute y_actual (log_std of future 13w sp_return)
    sp_idx = cols_target.index("sp_return")
    future_sp = X[:, PAST_LEN:, sp_idx]
    actual_std = future_sp.std(axis=1, ddof=1)
    actual_std = np.maximum(actual_std, 1e-8)
    y_log_std = np.log(actual_std).astype(np.float32)

    # Residual = y_actual - y_HAR
    residual = (y_log_std - y_har).astype(np.float32)

    X_past = X[:, :PAST_LEN, :]

    return (torch.from_numpy(X_past),
            torch.from_numpy(C),
            torch.from_numpy(residual),  # main target = residual
            torch.from_numpy(aux_std.astype(np.float32)),
            torch.from_numpy(y_log_std),   # for evaluation
            torch.from_numpy(y_har),       # for final prediction
            cond_stats_out, aux_stats_out)


def run_residual(spec, train_csv, val_csv, test_csv, save_dir, seed, fold_tag, lambda_aux=1.0):
    cols_cond = spec["cols_cond"]
    cols_target = spec["cols_target"]
    aux_col = spec["aux_col"]
    mask_future = spec["mask_future_ch"]

    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = f"vol_pilot_3m_mtl_residual_msel_v{spec['variant_id']}_{spec['name']}_{fold_tag}_seed{seed}"
    ckpt_path = os.path.join(save_dir, f"{tag}_best.pt")
    pred_path = os.path.join(save_dir, f"{tag}_test_preds.npz")
    summary_path = os.path.join(save_dir, f"{tag}_summary.json")

    if os.path.exists(ckpt_path) and os.path.exists(pred_path) and os.path.exists(summary_path):
        print(f"[SKIP] {tag} — residual learning 결과 이미 존재")
        return

    print(f"\n{'='*72}")
    print(f"[Variant {spec['variant_id']} = {spec['name']}_{fold_tag}] seed={seed}  RESIDUAL LEARNING")
    print(f"  Step 1: HAR-RV pred load (from train_har_rv.py)")
    print(f"  Step 2: v14 MTL target = residual (y_actual - y_HAR)")
    print(f"  Step 3: Final = HAR + v14_residual")
    print(f"{'='*72}")

    # 1) Load HAR predictions
    har_preds = load_har_predictions(fold_tag, save_dir)

    # 2) Load windows with residual target
    Xtr_past, Ctr, rtr, atr, ytr_actual, ytr_har, stats_c, stats_a = load_windows_residual(
        train_csv, cols_cond, cols_target, aux_col, har_preds["train"], L=L)
    Xv_past, Cv, rv, av, yv_actual, yv_har, _, _ = load_windows_residual(
        val_csv, cols_cond, cols_target, aux_col, har_preds["val"], L=L,
        cond_stats=stats_c, aux_stats=stats_a)
    Xte_past, Cte, rte, ate, yte_actual, yte_har, _, _ = load_windows_residual(
        test_csv, cols_cond, cols_target, aux_col, har_preds["test"], L=L,
        cond_stats=stats_c, aux_stats=stats_a)

    if Xv_past is None or Xv_past.shape[0] == 0:
        print(f"[FAIL] not enough val windows: {val_csv}")
        return

    print(f"  train_w={Xtr_past.shape[0]}  val_w={Xv_past.shape[0]}  test_w={Xte_past.shape[0]}")
    print(f"  residual stats — train: mean={rtr.mean():.4f}, std={rtr.std():.4f}")
    print(f"  residual stats — val  : mean={rv.mean():.4f}, std={rv.std():.4f}")
    print(f"  residual stats — test : mean={rte.mean():.4f}, std={rte.std():.4f}")

    Ctr_m = mask_future_channels(Ctr, PAST_LEN, mask_future)
    Cv_m  = mask_future_channels(Cv,  PAST_LEN, mask_future)
    Cte_m = mask_future_channels(Cte, PAST_LEN, mask_future)

    model = CausalTransformerVolMTL(
        d_cond=len(cols_cond), d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        past_len=PAST_LEN, future_len=FUTURE_LEN
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    train_ds = TensorDataset(Xtr_past, Ctr_m, rtr, atr)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)

    baseline_test_mse = float(((yte_actual - yte_actual.mean()) ** 2).mean())

    def evaluate(Xp, C, r_target, a_target, y_actual, y_har):
        model.eval()
        with torch.no_grad():
            r_pred, _ = model(C.to(device), Xp.to(device))
            r_pred = r_pred.cpu().numpy()
        # Final prediction = HAR + residual_pred
        y_final = y_har.numpy() + r_pred
        y_act_np = y_actual.numpy()
        mse_final = float(np.mean((y_act_np - y_final) ** 2))
        ss_res = float(np.sum((y_act_np - y_final) ** 2))
        ss_tot = float(np.sum((y_act_np - y_act_np.mean()) ** 2))
        r2_final = 1.0 - ss_res / max(ss_tot, 1e-12)
        p_final = pearson_corr(y_act_np, y_final)
        s_final = spearman_corr(y_act_np, y_final)
        # Residual learning loss (model 의 MSE on residual)
        r_target_np = r_target.numpy()
        mse_res = float(np.mean((r_target_np - r_pred) ** 2))
        return r_pred, y_final, mse_final, r2_final, p_final, s_final, mse_res

    best_val_mse = float("inf")
    best_state = None
    best_epoch = -1
    best_test_final = None
    best_test_actual = None
    best_test_mse = None
    best_test_r2 = None
    best_test_p = None
    best_test_s = None
    best_val_p = None
    pat = 0
    log = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, cb, rb, ab in train_dl:
            xb = xb.to(device); cb = cb.to(device); rb = rb.to(device); ab = ab.to(device)
            r_pred, a_pred = model(cb, xb)
            loss_v = ((r_pred - rb) ** 2).mean()  # target = residual
            loss_a = ((a_pred - ab) ** 2).mean()
            loss = loss_v + lambda_aux * loss_a
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        r_v, y_v_final, val_mse_final, val_r2, val_p, val_s, val_mse_res = evaluate(
            Xv_past, Cv_m, rv, av, yv_actual, yv_har)
        r_t, y_t_final, test_mse_final, test_r2, test_p, test_s, test_mse_res = evaluate(
            Xte_past, Cte_m, rte, ate, yte_actual, yte_har)

        log.append(dict(epoch=epoch, val_mse_res=val_mse_res, val_mse_final=val_mse_final,
                        val_p=val_p, test_mse_final=test_mse_final, test_r2=test_r2, test_p=test_p))

        # Selection: val_mse_final (on actual, not residual) for HAR + residual combo
        improved = (epoch >= MIN_EPOCH) and (val_mse_final < best_val_mse - 1e-6)
        if improved:
            best_val_mse = val_mse_final
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test_final = y_t_final
            best_test_actual = yte_actual.numpy()
            best_test_mse = test_mse_final
            best_test_r2 = test_r2
            best_test_p = test_p
            best_test_s = test_s
            best_val_p = val_p
            best_epoch = epoch
            pat = 0
        else:
            pat += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep {epoch:3d}: val_mse(final)={val_mse_final:.5f}  val_p={val_p:+.3f}  "
                  f"test_mse(final)={test_mse_final:.5f}  test_r2={test_r2:+.3f}  test_p={test_p:+.3f}")
        if pat >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val_mse_final={best_val_mse:.5f}, val_p={best_val_p:+.4f}")
    print(f"    test MSE  (final = HAR + v14_residual) = {best_test_mse:.5f}")
    print(f"    test R²    = {best_test_r2:+.4f}")
    print(f"    test Pearson  = {best_test_p:+.4f}")
    print(f"    test Spearman = {best_test_s:+.4f}")
    print(f"    baseline test MSE = {baseline_test_mse:.5f}")

    if best_state is not None:
        torch.save({
            "model_state": best_state,
            "model_type": "causal_transformer_vol_mtl_3m_residual",
            "cond_cols": cols_cond,
            "target_cols": cols_target,
            "aux_col": aux_col,
            "lambda_aux": lambda_aux,
            "stats_cond": stats_c,
            "stats_aux": stats_a,
            "mask_future_ch": mask_future,
            "config": dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                           d_cond=len(cols_cond), past_len=PAST_LEN,
                           future_len=FUTURE_LEN, total_len=L),
            "variant": dict(id=spec["variant_id"], name=spec["name"]),
        }, ckpt_path)
        np.savez(pred_path,
                 y_pred_log_std=best_test_final,
                 y_actual_log_std=best_test_actual,
                 y_pred_std=np.exp(best_test_final),
                 y_actual_std=np.exp(best_test_actual),
                 baseline_test_mse=baseline_test_mse)
        print(f"  saved: {ckpt_path}")
        print(f"  saved: {pred_path}")

    summary = dict(
        model_type="causal_transformer_vol_mtl_3m_residual",
        variant_id=spec["variant_id"],
        variant_name=spec["name"],
        fold=fold_tag,
        seed=seed,
        cond_cols=cols_cond,
        target_cols=cols_target,
        aux_col=aux_col,
        lambda_aux=lambda_aux,
        mask_future_ch=mask_future,
        future_len=FUTURE_LEN,
        selection_criterion="val_mse_final_min (HAR + v14_residual)",
        best_epoch=best_epoch,
        val_mse=best_val_mse,
        val_pearson_at_best=best_val_p,
        test_mse=best_test_mse,
        test_r2=best_test_r2,
        test_pearson_std=best_test_p,
        test_spearman_std=best_test_s,
        baseline_test_mse=baseline_test_mse,
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Residual Learning (HAR-RV + v14 MTL boosting)")
    ap.add_argument("--variant", type=int, default=14, choices=list(VARIANTS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--fold", default="F1")
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--lambda-aux", type=float, default=1.0)
    args = ap.parse_args()

    spec = build_spec(args.variant)
    train_csv = os.path.join(args.folds_dir, f"{args.fold}_train.csv")
    val_csv   = os.path.join(args.folds_dir, f"{args.fold}_val.csv")
    test_csv  = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    for p in [train_csv, val_csv, test_csv]:
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing {p}")

    print(f"\n[Residual Learning | --fold {args.fold} --variant {args.variant}]")
    for seed in args.seeds:
        run_residual(spec, train_csv, val_csv, test_csv, args.out_dir, seed,
                     fold_tag=args.fold, lambda_aux=args.lambda_aux)


if __name__ == "__main__":
    main()
