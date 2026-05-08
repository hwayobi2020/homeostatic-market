"""Vol pilot MTL — sp_return vol + excess_liq vol 동시 학습.

Base 4ch cond (tbill + m2/gdp/cpi) 위에 excess_liq 를 MTL aux target 으로 추가.
shared Causal Transformer representation 으로 main task (sp vol) generalization 향상 의도.

Targets:
  main: y_main = log( std(sp_return future 52w) )
  aux:  y_aux  = log( std(excess_liq_yoy_lag future 52w) )

Loss:
  L = MSE(pred_main, y_main) + λ × MSE(pred_aux, y_aux)
  λ = 0.1 (기존 standard)

Architecture:
  Causal Transformer encoder (vol_pilot_v2 동일 backbone) — base 4ch cond + sp past
  Future portion (52 step) mean pool
  분리 head 2개 (sp_head, aux_head) — 각각 1 scalar output

Best ckpt: sp_return main val_mse 기준 (raw)
ckpt prefix: vol_pilot_mtl_*
"""
import torch  # MUST be first

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch.nn as nn

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

L = 104
PAST_LEN = 52
FUTURE_LEN = 52
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
LR = 1e-4
BATCH = 32
MAX_EPOCHS = 60
PATIENCE = 30
GRAD_CLIP = 1.0
LAMBDA_AUX = 0.1

# Base 4ch cond
COLS_COND = ["tbill_wr", "m2_yoy_lag", "gdp_yoy_lag", "cpi_yoy_lag"]
# Multi-target — main + aux
COLS_TARGET = ["sp_return", "excess_liq_yoy_lag"]


# =====================================================================
# Model — shared encoder + 분리 head 2개
# =====================================================================

class CausalTransformerVolMTL(nn.Module):
    """Shared Causal Transformer encoder + future mean pool + 분리 head per target.

    Input:
      cond:        (B, L=104, d_cond=4)
      target_past: (B, past_len=52, d_target=2)  past sp_return + past excess_liq_yoy_lag

    Output:
      log_std_pred: dict {col: (B,)}  per target channel
    """
    def __init__(self, d_cond, target_cols, d_model, n_heads, n_layers,
                 past_len, future_len, dropout=0.1):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.target_cols = target_cols
        self.total_len = past_len + future_len  # 104

        self.input_dim = d_cond + len(target_cols)
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

        # 분리 head — channel 별 독립
        self.heads = nn.ModuleDict({
            col: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            for col in target_cols
        })

    def forward(self, cond, target_past):
        B = cond.shape[0]
        device = cond.device

        target_full = torch.zeros(
            B, self.total_len, len(self.target_cols),
            dtype=cond.dtype, device=device
        )
        target_full[:, :self.past_len, :] = target_past

        x = torch.cat([cond, target_full], dim=-1)
        x = self.input_proj(x)

        pos = torch.arange(self.total_len, device=device)
        x = x + self.pos_emb(pos).unsqueeze(0)

        h = self.encoder(x, mask=self.causal_mask, is_causal=True)
        h_future = h[:, self.past_len:, :].mean(dim=1)               # (B, d_model)

        outputs = {}
        for col in self.target_cols:
            outputs[col] = self.heads[col](h_future).squeeze(-1)     # (B,)
        return outputs


# =====================================================================
# Data
# =====================================================================

def load_windows(csv_path, cols_cond, cols_target, L=104, cond_stats=None):
    df = pd.read_csv(csv_path)
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        return None, None, None, None
    X = np.zeros((n_w, L, len(cols_target)), dtype=np.float32)
    C = np.zeros((n_w, L, len(cols_cond)),   dtype=np.float32)
    for i in range(n_w):
        X[i] = df[cols_target].iloc[i : i + L].values
        C[i] = df[cols_cond].iloc[i : i + L].values

    if cond_stats is None:
        cmu = C.reshape(-1, len(cols_cond)).mean(axis=0)
        csd = C.reshape(-1, len(cols_cond)).std(axis=0) + 1e-8
        cond_stats_out = {"mean": cmu.tolist(), "std": csd.tolist()}
    else:
        cmu = np.asarray(cond_stats["mean"], dtype=np.float32)
        csd = np.asarray(cond_stats["std"],  dtype=np.float32)
        cond_stats_out = cond_stats
    C = (C - cmu) / csd

    # target = log( std(future 52w of each target) ) — multi-target
    y_log_std = np.zeros((n_w, len(cols_target)), dtype=np.float32)
    for i, col in enumerate(cols_target):
        future_t = X[:, PAST_LEN:, i]                                # (n_w, 52)
        actual_std = future_t.std(axis=1, ddof=1)                    # (n_w,)
        actual_std = np.maximum(actual_std, 1e-8)
        y_log_std[:, i] = np.log(actual_std).astype(np.float32)

    X_past = X[:, :PAST_LEN, :]                                      # (n_w, 52, d_target)

    return (torch.from_numpy(X_past),
            torch.from_numpy(C),
            torch.from_numpy(y_log_std),
            cond_stats_out)


def mask_future_channels(C, past_len, mask_channels):
    C = C.clone()
    for ch in mask_channels:
        C[:, past_len:, ch] = 0.0
    return C


# =====================================================================
# Eval helpers
# =====================================================================

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


# =====================================================================
# Train
# =====================================================================

def run(train_csv, val_csv, test_csv, save_dir, seed,
        max_epochs, patience, batch, lr, fold_tag, lambda_aux):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cols_cond   = COLS_COND
    cols_target = COLS_TARGET
    mask_future = list(range(1, len(cols_cond)))  # tbill (idx 0) 만 future 활성

    fold_str = f"_{fold_tag}" if fold_tag else ""
    tag_full = f"vol_pilot_mtl_excessliq{fold_str}_seed{seed}"
    ckpt_path    = os.path.join(save_dir, f"{tag_full}_best.pt")
    summary_path = os.path.join(save_dir, f"{tag_full}_summary.json")
    log_path     = os.path.join(save_dir, f"{tag_full}_trainlog.csv")
    pred_path    = os.path.join(save_dir, f"{tag_full}_test_preds.npz")

    if os.path.exists(ckpt_path) and os.path.exists(summary_path):
        print(f"[SKIP] {tag_full} — ckpt + summary 이미 존재")
        with open(summary_path) as f:
            return json.load(f)

    print(f"\n{'='*72}")
    print(f"[Vol MTL excess_liq aux{fold_str}] seed={seed}")
    print(f"  cond   = {cols_cond}  (D_COND={len(cols_cond)})")
    print(f"  target = {cols_target}  (main + aux)")
    print(f"  mask_future_ch = {mask_future}")
    print(f"  λ_aux = {lambda_aux},  L={L} (past={PAST_LEN} + future={FUTURE_LEN})")
    print(f"{'='*72}")

    Xtr_past, Ctr, ytr, stats_c = load_windows(train_csv, cols_cond, cols_target, L=L)
    if Xtr_past is None:
        print(f"[FAIL] not enough train windows: {train_csv}")
        return None
    Xte_past, Cte, yte, _ = load_windows(test_csv, cols_cond, cols_target, L=L, cond_stats=stats_c)
    if Xte_past is None:
        print(f"[FAIL] not enough test windows: {test_csv}")
        return None

    Ctr = mask_future_channels(Ctr, PAST_LEN, mask_future)
    Cte = mask_future_channels(Cte, PAST_LEN, mask_future)

    if val_csv is not None and os.path.exists(val_csv):
        Xv_past, Cv, yv, _ = load_windows(val_csv, cols_cond, cols_target, L=L, cond_stats=stats_c)
        if Xv_past is None:
            print(f"[FAIL] not enough val windows: {val_csv}")
            return None
        Cv = mask_future_channels(Cv, PAST_LEN, mask_future)
        Xtr_past_, Ctr_, ytr_ = Xtr_past, Ctr, ytr
        print(f"  train_w={Xtr_past_.shape[0]} (full), val_w={Xv_past.shape[0]} (val_csv), "
              f"test_w={Xte_past.shape[0]}")
    else:
        n_w_tr = Xtr_past.shape[0]
        n_val = max(int(n_w_tr * 0.15), 1)
        Xtr_past_, Ctr_, ytr_ = Xtr_past[:-n_val], Ctr[:-n_val], ytr[:-n_val]
        Xv_past,   Cv,   yv   = Xtr_past[-n_val:], Ctr[-n_val:], ytr[-n_val:]
        print(f"  train_w={Xtr_past_.shape[0]}, val_w={Xv_past.shape[0]} (auto 15%), "
              f"test_w={Xte_past.shape[0]}")

    sp_idx = cols_target.index("sp_return")

    # main baseline (sp_return vol)
    y_train_mean_sp = float(ytr_[:, sp_idx].mean())
    baseline_test_mse_sp = float(((yte.numpy()[:, sp_idx] - y_train_mean_sp) ** 2).mean())
    print(f"  log_std(sp_return) baseline (train mean) = {y_train_mean_sp:+.4f}")
    print(f"  baseline test MSE (sp constant)         = {baseline_test_mse_sp:.6f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformerVolMTL(
        d_cond=len(cols_cond), target_cols=cols_target,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        past_len=PAST_LEN, future_len=FUTURE_LEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    best_test_mse = None
    best_test_r2 = None
    best_test_pearson = None
    best_test_spearman = None
    best_test_pred = None
    best_test_actual = None
    best_test_aux_mse = None
    pat = 0
    log = []

    def eval_split(X_past, C, y):
        model.eval()
        with torch.no_grad():
            outputs = model(C.to(device), X_past.to(device))
            preds = {col: outputs[col].cpu().numpy() for col in cols_target}
            actual = y.numpy()
            mse_per_col = {}
            for i, col in enumerate(cols_target):
                mse_per_col[col] = float(((preds[col] - actual[:, i]) ** 2).mean())
            sp_pred = preds["sp_return"]
            sp_act  = actual[:, sp_idx]
            pearson = pearson_corr(np.exp(sp_pred), np.exp(sp_act))
            spearman = spearman_corr(np.exp(sp_pred), np.exp(sp_act))
        return mse_per_col, pearson, spearman, preds, actual

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(Xtr_past_.shape[0])
        losses = []
        losses_main = []
        losses_aux = []
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            xp = Xtr_past_[idx].to(device)
            cb = Ctr_[idx].to(device)
            yb = ytr_[idx].to(device)
            outputs = model(cb, xp)
            loss_main = ((outputs["sp_return"] - yb[:, sp_idx]) ** 2).mean()
            loss_aux = ((outputs["excess_liq_yoy_lag"]
                         - yb[:, cols_target.index("excess_liq_yoy_lag")]) ** 2).mean()
            loss = loss_main + lambda_aux * loss_aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            opt.step()
            losses.append(loss.item())
            losses_main.append(loss_main.item())
            losses_aux.append(loss_aux.item())

        train_loss      = float(np.mean(losses))
        train_loss_main = float(np.mean(losses_main))
        train_loss_aux  = float(np.mean(losses_aux))
        val_mse,  val_p,  val_s,  _, _ = eval_split(Xv_past,  Cv,  yv)
        test_mse, test_p, test_s, test_preds, test_actual = eval_split(Xte_past, Cte, yte)

        val_sp  = val_mse["sp_return"]
        test_sp = test_mse["sp_return"]
        val_aux  = val_mse["excess_liq_yoy_lag"]
        test_aux = test_mse["excess_liq_yoy_lag"]
        val_r2  = 1.0 - val_sp  / float(((yv.numpy()[:, sp_idx]  - y_train_mean_sp) ** 2).mean() + 1e-12)
        test_r2 = 1.0 - test_sp / baseline_test_mse_sp if baseline_test_mse_sp > 0 else float("nan")

        print(f"  ep{epoch:>3d}  total={train_loss:.5f} (main={train_loss_main:.5f} aux={train_loss_aux:.5f})  "
              f"val_sp={val_sp:.5f}  test_sp={test_sp:.5f} (R²={test_r2:+.3f})  "
              f"test_pearson={test_p:+.3f}  test_aux_mse={test_aux:.4f}")

        log.append(dict(epoch=epoch, train_loss=train_loss,
                        train_main=train_loss_main, train_aux=train_loss_aux,
                        val_sp=val_sp, val_aux=val_aux, val_r2=val_r2,
                        test_sp=test_sp, test_aux=test_aux, test_r2=test_r2,
                        test_pearson=test_p, test_spearman=test_s))

        if not np.isfinite(val_sp):
            pat += 1
        elif val_sp < best_val - 1e-6:
            best_val = val_sp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test_mse = test_sp
            best_test_r2 = test_r2
            best_test_pearson = test_p
            best_test_spearman = test_s
            best_test_pred = test_preds
            best_test_actual = test_actual
            best_test_aux_mse = test_aux
            best_epoch = epoch
            pat = 0
        else:
            pat += 1

        if (not np.isfinite(test_sp)) or test_sp > 100.0:
            print(f"  ⚠ divergence (test_sp={test_sp:.2f}) — early termination")
            break
        if pat >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val_sp_mse={best_val:.5f}")
    print(f"    test sp MSE   = {best_test_mse:.5f}")
    print(f"    test sp R²    = {best_test_r2:+.4f}")
    print(f"    test Pearson  (sp std)  = {best_test_pearson:+.4f}")
    print(f"    test Spearman (sp std)  = {best_test_spearman:+.4f}")
    print(f"    test aux MSE  (excess_liq vol)  = {best_test_aux_mse:.5f}")
    print(f"    baseline test MSE (sp)         = {baseline_test_mse_sp:.5f}")

    os.makedirs(save_dir, exist_ok=True)
    if best_state is not None:
        torch.save({
            "model_state":     best_state,
            "model_type":      "causal_transformer_vol_mtl",
            "cond_cols":       cols_cond,
            "target_cols":     cols_target,
            "stats_cond":      stats_c,
            "mask_future_ch":  mask_future,
            "lambda_aux":      lambda_aux,
            "config": dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                           d_cond=len(cols_cond), past_len=PAST_LEN,
                           future_len=FUTURE_LEN, total_len=L),
        }, ckpt_path)
        np.savez(pred_path,
                 y_pred_log_std_sp =best_test_pred["sp_return"],
                 y_actual_log_std_sp=best_test_actual[:, sp_idx],
                 y_pred_std_sp     =np.exp(best_test_pred["sp_return"]),
                 y_actual_std_sp   =np.exp(best_test_actual[:, sp_idx]),
                 y_pred_log_std_aux=best_test_pred["excess_liq_yoy_lag"],
                 y_actual_log_std_aux=best_test_actual[:, cols_target.index("excess_liq_yoy_lag")],
                 y_train_log_std_mean_sp=y_train_mean_sp,
                 baseline_test_mse_sp=baseline_test_mse_sp)
        print(f"  saved test preds: {pred_path}")

    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        model_type="causal_transformer_vol_mtl",
        fold=fold_tag,
        seed=seed,
        cond_cols=cols_cond,
        target_cols=cols_target,
        mask_future_ch=mask_future,
        lambda_aux=lambda_aux,
        best_epoch=best_epoch,
        val_sp_mse=best_val,
        test_sp_mse=best_test_mse,
        test_sp_r2=best_test_r2,
        test_aux_mse=best_test_aux_mse,
        test_pearson_std_sp=best_test_pearson,
        test_spearman_std_sp=best_test_spearman,
        y_train_log_std_mean_sp=y_train_mean_sp,
        baseline_test_mse_sp=baseline_test_mse_sp,
        n_params=n_params,
        device=str(device),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {ckpt_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Vol pilot MTL (excess_liq aux)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--fold", default=None)
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--val-csv",   default=None)
    ap.add_argument("--test-csv",  default=None)
    ap.add_argument("--out-dir",   default=os.path.join(HERE, "result"))
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience",   type=int, default=PATIENCE)
    ap.add_argument("--batch",      type=int, default=BATCH)
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--lambda-aux", type=float, default=LAMBDA_AUX)
    ap.add_argument("--pilot-split", action="store_true")
    args = ap.parse_args()

    train_csv = val_csv = test_csv = None
    if args.pilot_split:
        repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
        split_dir = os.path.join(repo_root, "data", "pilot_split")
        train_csv = os.path.join(split_dir, "train.csv")
        val_csv   = os.path.join(split_dir, "val.csv")
        test_csv  = os.path.join(split_dir, "test.csv")
        print(f"[--pilot-split]")
    elif args.fold is not None:
        repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
        folds_dir = os.path.join(repo_root, "data", "folds_v33")
        train_csv = os.path.join(folds_dir, f"{args.fold}_train.csv")
        val_csv   = os.path.join(folds_dir, f"{args.fold}_val.csv")
        test_csv  = os.path.join(folds_dir, f"{args.fold}_test.csv")
        print(f"[--fold {args.fold}]")

    if args.train_csv: train_csv = args.train_csv
    if args.val_csv:   val_csv   = args.val_csv
    if args.test_csv:  test_csv  = args.test_csv

    if train_csv is None or test_csv is None:
        ap.error("Provide --pilot-split or --fold or explicit --train-csv/--test-csv")

    os.makedirs(args.out_dir, exist_ok=True)

    fold_tag = "pilot" if args.pilot_split else args.fold

    for seed in args.seeds:
        run(train_csv, val_csv, test_csv, args.out_dir, seed,
            max_epochs=args.max_epochs, patience=args.patience,
            batch=args.batch, lr=args.lr, fold_tag=fold_tag,
            lambda_aux=args.lambda_aux)


if __name__ == "__main__":
    main()
