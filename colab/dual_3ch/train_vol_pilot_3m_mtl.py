"""MTL vol forecasting — main: log(σ_future_13w), aux: metab_13w (single fold F1).

ESWA paper 패턴 (sp+stockpp/bondpp MTL) 재현:
  Shared Transformer encoder + 2 heads.
    main head:  log( std(future 13w sp_return) )                                   — paper main
    aux  head:  metab_13w at future window 끝 row (t+L-1)                          — backbone macro-aware

  loss = MSE_vol + λ_aux · MSE_metab     (λ_aux = 1.0 default)

Variant 14: base_mtl_metab
  cond (6채널, metab_13w 빠지고 그 자리는 aux target):
    [tbill_wr, m2_13w_cum_lag, ads_lag, cpi_13w_cum_lag, wti_wr, sp_log_std_13w]

비교 (DM test):
  v13 (cond 7ch, metab in cond)  vs  v14 (cond 6ch, metab as aux MTL)
  HAR-RV (sp_log_std_13w only)   vs  v14

Output:
  result/vol_pilot_3m_mtl_msel_v14_base_mtl_metab_F1_seed{N}_best.pt
  result/vol_pilot_3m_mtl_msel_v14_base_mtl_metab_F1_seed{N}_test_preds.npz
  result/vol_pilot_3m_mtl_msel_v14_base_mtl_metab_F1_seed{N}_summary.json
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
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

PAST_LEN   = 52
FUTURE_LEN = 13
L          = PAST_LEN + FUTURE_LEN  # 65

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
LR = 1e-4
BATCH = 32
MAX_EPOCHS = 60
PATIENCE = 30
GRAD_CLIP = 1.0
MIN_EPOCH = 20

# Note: cond 에 sp_std_13w (raw past 13w std) 추가 — magnitude direct read-out 보조
COLS_COND_BASE = ["tbill_wr", "m2_13w_cum_lag", "ads_lag", "cpi_13w_cum_lag", "sp_std_13w"]

VARIANTS = {
    14: dict(name="base_mtl_metab",
             cond_cols=COLS_COND_BASE + ["wti_wr", "sp_log_std_13w"],
             aux_col="metab_13w"),
    # v15: permutation importance 로 HARMFUL 판정된 3개 (ads_lag, wti_wr, sp_log_std_13w) 제거
    # 단 sp_std_13w (raw) 는 유지 — magnitude reference. paper logic: macro + raw vol level
    15: dict(name="base_clean_mtl",
             cond_cols=["tbill_wr", "m2_13w_cum_lag", "cpi_13w_cum_lag", "sp_std_13w"],
             aux_col="metab_13w"),
}


def build_spec(variant_id):
    v = VARIANTS[variant_id]
    cols_cond = v["cond_cols"]
    cols_target = ["sp_return"]
    aux_col = v["aux_col"]
    mask_future_ch = list(range(1, len(cols_cond)))
    return dict(variant_id=variant_id, name=v["name"],
                cols_cond=cols_cond, cols_target=cols_target,
                aux_col=aux_col, mask_future_ch=mask_future_ch)


# =====================================================================
# Model — shared encoder + 2 heads (vol, metab)
# =====================================================================

class CausalTransformerVolMTL(nn.Module):
    def __init__(self, d_cond, d_model, n_heads, n_layers,
                 past_len, future_len, dropout=0.1):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.total_len = past_len + future_len

        self.input_dim = d_cond + 1
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.pos_emb = nn.Embedding(self.total_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        causal_mask = torch.triu(
            torch.ones(self.total_len, self.total_len), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        def head():
            return nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        self.head_vol = head()
        self.head_aux = head()

    def forward(self, cond, target_past):
        B = cond.shape[0]
        device = cond.device

        target_full = torch.zeros(B, self.total_len, 1, dtype=cond.dtype, device=device)
        target_full[:, :self.past_len, :] = target_past

        x = torch.cat([cond, target_full], dim=-1)
        x = self.input_proj(x)

        pos = torch.arange(self.total_len, device=device)
        x = x + self.pos_emb(pos).unsqueeze(0)

        h = self.encoder(x, mask=self.causal_mask, is_causal=True)
        h_future = h[:, self.past_len:, :].mean(dim=1)
        vol_pred = self.head_vol(h_future).squeeze(-1)
        aux_pred = self.head_aux(h_future).squeeze(-1)
        return vol_pred, aux_pred


# =====================================================================
# Data — main + aux target
# =====================================================================

def load_windows_mtl(csv_path, cols_cond, cols_target, aux_col, L=65,
                     cond_stats=None, aux_stats=None):
    """Return (X_past_target, C_cond, y_log_std, y_aux, cond_stats, aux_stats).

    y_aux = aux_col at the last future row (origin t 에서 t+L-1 시점의 metab_13w 값).
    """
    df = pd.read_csv(csv_path)
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        return None, None, None, None, None, None

    X = np.zeros((n_w, L, len(cols_target)), dtype=np.float32)
    C = np.zeros((n_w, L, len(cols_cond)), dtype=np.float32)
    aux = np.zeros((n_w,), dtype=np.float32)

    aux_vals = df[aux_col].values.astype(np.float32)
    for i in range(n_w):
        X[i] = df[cols_target].iloc[i : i + L].values
        C[i] = df[cols_cond].iloc[i : i + L].values
        aux[i] = aux_vals[i + L - 1]   # last future row

    # cond standardization
    if cond_stats is None:
        cmu = C.reshape(-1, len(cols_cond)).mean(axis=0)
        csd = C.reshape(-1, len(cols_cond)).std(axis=0) + 1e-8
        cond_stats_out = {"mean": cmu.tolist(), "std": csd.tolist()}
    else:
        cmu = np.asarray(cond_stats["mean"], dtype=np.float32)
        csd = np.asarray(cond_stats["std"], dtype=np.float32)
        cond_stats_out = cond_stats
    C = (C - cmu) / csd

    # aux standardization (train 기준)
    if aux_stats is None:
        amu = float(aux.mean()); asd = float(aux.std() + 1e-8)
        aux_stats_out = {"mean": amu, "std": asd}
    else:
        amu = aux_stats["mean"]; asd = aux_stats["std"]
        aux_stats_out = aux_stats
    aux_std = (aux - amu) / asd

    sp_idx = cols_target.index("sp_return")
    future_sp = X[:, PAST_LEN:, sp_idx]
    actual_std = future_sp.std(axis=1, ddof=1)
    actual_std = np.maximum(actual_std, 1e-8)
    y_log_std = np.log(actual_std).astype(np.float32)

    X_past = X[:, :PAST_LEN, :]

    return (torch.from_numpy(X_past),
            torch.from_numpy(C),
            torch.from_numpy(y_log_std),
            torch.from_numpy(aux_std.astype(np.float32)),
            cond_stats_out,
            aux_stats_out)


def mask_future_channels(C, past_len, mask_channels):
    C = C.clone()
    for ch in mask_channels:
        C[:, past_len:, ch] = 0.0
    return C


def pearson_corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def spearman_corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    ra = pd.Series(a).rank().values - pd.Series(a).rank().mean()
    rb = pd.Series(b).rank().values - pd.Series(b).rank().mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


# =====================================================================
# Train
# =====================================================================

def run(spec, train_csv, val_csv, test_csv, save_dir, seed,
        max_epochs, patience, batch, lr, fold_tag, lambda_aux=1.0):
    cols_cond = spec["cols_cond"]
    cols_target = spec["cols_target"]
    aux_col = spec["aux_col"]
    mask_future = spec["mask_future_ch"]

    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = f"vol_pilot_3m_mtl_msel_v{spec['variant_id']}_{spec['name']}_{fold_tag}_seed{seed}"
    ckpt_path = os.path.join(save_dir, f"{tag}_best.pt")
    pred_path = os.path.join(save_dir, f"{tag}_test_preds.npz")
    summary_path = os.path.join(save_dir, f"{tag}_summary.json")
    log_path = os.path.join(save_dir, f"{tag}_log.csv")

    if os.path.exists(ckpt_path) and os.path.exists(pred_path) and os.path.exists(summary_path):
        print(f"[SKIP] {tag} — ckpt + pred + summary 이미 존재")
        return

    print(f"\n{'='*72}")
    print(f"[Variant {spec['variant_id']} = {spec['name']}_{fold_tag}] seed={seed}  MTL (vol+metab)")
    print(f"  cond   = {cols_cond}  (D_COND={len(cols_cond)})")
    print(f"  target_main = log( std(future 13w sp_return) )")
    print(f"  target_aux  = {aux_col} at t+L-1  (λ_aux={lambda_aux})")
    print(f"  mask_future_ch = {mask_future}")
    print(f"{'='*72}")

    Xtr_past, Ctr, ytr, atr, stats_c, stats_a = load_windows_mtl(
        train_csv, cols_cond, cols_target, aux_col, L=L)
    Xte_past, Cte, yte, ate, _, _ = load_windows_mtl(
        test_csv, cols_cond, cols_target, aux_col, L=L,
        cond_stats=stats_c, aux_stats=stats_a)
    Xv_past, Cv, yv, av, _, _ = load_windows_mtl(
        val_csv, cols_cond, cols_target, aux_col, L=L,
        cond_stats=stats_c, aux_stats=stats_a)

    if Xv_past is None or Xv_past.shape[0] == 0:
        print(f"[FAIL] not enough val windows: {val_csv}")
        return

    print(f"  train_w={Xtr_past.shape[0]}  val_w={Xv_past.shape[0]}  test_w={Xte_past.shape[0]}")

    Ctr_m = mask_future_channels(Ctr, PAST_LEN, mask_future)
    Cv_m  = mask_future_channels(Cv,  PAST_LEN, mask_future)
    Cte_m = mask_future_channels(Cte, PAST_LEN, mask_future)

    model = CausalTransformerVolMTL(
        d_cond=len(cols_cond), d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        past_len=PAST_LEN, future_len=FUTURE_LEN
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    train_ds = TensorDataset(Xtr_past, Ctr_m, ytr, atr)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)

    y_train_mean = float(ytr.mean())
    baseline_test_mse = float(((yte - y_train_mean) ** 2).mean())

    def evaluate(Xp, C, y, a):
        model.eval()
        with torch.no_grad():
            v, ax = model(C.to(device), Xp.to(device))
            v = v.cpu().numpy(); ax = ax.cpu().numpy()
        y_np = y.cpu().numpy(); a_np = a.cpu().numpy()
        mse_v = float(np.mean((y_np - v) ** 2))
        mse_a = float(np.mean((a_np - ax) ** 2))
        p_v = pearson_corr(y_np, v)
        s_v = spearman_corr(y_np, v)
        return v, ax, mse_v, mse_a, p_v, s_v

    best_val_mse = float("inf")
    best_state = None
    best_epoch = -1
    best_test_pred = None
    best_test_actual = None
    best_test_mse = None
    best_test_pearson = None
    best_test_spearman = None
    best_test_r2 = None
    best_val_pearson = None
    pat = 0
    log = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, cb, yb, ab in train_dl:
            xb = xb.to(device); cb = cb.to(device); yb = yb.to(device); ab = ab.to(device)
            v_pred, a_pred = model(cb, xb)
            loss_v = ((v_pred - yb) ** 2).mean()
            loss_a = ((a_pred - ab) ** 2).mean()
            loss = loss_v + lambda_aux * loss_a
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        v_v, a_v, val_mse_v, val_mse_a, val_p, val_s = evaluate(Xv_past, Cv_m, yv, av)
        v_t, a_t, test_mse, test_mse_a, test_p, test_s = evaluate(Xte_past, Cte_m, yte, ate)
        yte_np = yte.cpu().numpy()
        ss_res = float(np.sum((yte_np - v_t) ** 2))
        ss_tot = float(np.sum((yte_np - yte_np.mean()) ** 2))
        test_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        log.append(dict(epoch=epoch,
                        val_mse_vol=val_mse_v, val_mse_aux=val_mse_a, val_pearson=val_p,
                        test_mse=test_mse, test_r2=test_r2, test_pearson=test_p))

        improved = (epoch >= MIN_EPOCH) and (val_mse_v < best_val_mse - 1e-6)
        if improved:
            best_val_mse = val_mse_v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test_mse = test_mse
            best_test_r2 = test_r2
            best_test_pearson = test_p
            best_test_spearman = test_s
            best_test_pred = v_t
            best_test_actual = yte_np
            best_val_pearson = val_p
            best_epoch = epoch
            pat = 0
        else:
            pat += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep {epoch:3d}: val_mse={val_mse_v:.5f}  val_p={val_p:+.3f}  "
                  f"val_mse_aux={val_mse_a:.4f}  test_mse={test_mse:.4f}  test_p={test_p:+.3f}  "
                  f"test_r2={test_r2:+.3f}")
        if pat >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val_mse={best_val_mse:.5f}, val_pearson={best_val_pearson:+.4f}")
    print(f"    test MSE  = {best_test_mse:.5f}")
    print(f"    test R²   = {best_test_r2:+.4f}")
    print(f"    test Pearson  = {best_test_pearson:+.4f}")
    print(f"    test Spearman = {best_test_spearman:+.4f}")
    print(f"    baseline test MSE = {baseline_test_mse:.5f}")

    os.makedirs(save_dir, exist_ok=True)
    if best_state is not None:
        torch.save({
            "model_state": best_state,
            "model_type": "causal_transformer_vol_mtl_3m",
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
                 y_pred_log_std=best_test_pred,
                 y_actual_log_std=best_test_actual,
                 y_pred_std=np.exp(best_test_pred),
                 y_actual_std=np.exp(best_test_actual),
                 baseline_test_mse=baseline_test_mse)
        print(f"  saved test preds: {pred_path}")
        print(f"  saved: {ckpt_path}")

    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        model_type="causal_transformer_vol_mtl_3m",
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
        selection_criterion="val_mse_min",
        best_epoch=best_epoch,
        val_mse=best_val_mse,
        val_pearson_at_best=best_val_pearson,
        test_mse=best_test_mse,
        test_r2=best_test_r2,
        test_pearson_std=best_test_pearson,
        test_spearman_std=best_test_spearman,
        baseline_test_mse=baseline_test_mse,
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="MTL vol forecasting (main: log_std, aux: metab_13w)")
    ap.add_argument("--variant", type=int, default=14, choices=list(VARIANTS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--fold", default="F1")
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--lambda-aux", type=float, default=1.0,
                    help="aux loss weight (default 1.0)")
    args = ap.parse_args()

    spec = build_spec(args.variant)
    train_csv = os.path.join(args.folds_dir, f"{args.fold}_train.csv")
    val_csv   = os.path.join(args.folds_dir, f"{args.fold}_val.csv")
    test_csv  = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    for p in [train_csv, val_csv, test_csv]:
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing {p}")

    print(f"[--fold {args.fold}]  ({args.folds_dir})")

    for seed in args.seeds:
        run(spec, train_csv, val_csv, test_csv, args.out_dir, seed,
            args.max_epochs, args.patience, args.batch, args.lr,
            fold_tag=args.fold, lambda_aux=args.lambda_aux)


if __name__ == "__main__":
    main()
