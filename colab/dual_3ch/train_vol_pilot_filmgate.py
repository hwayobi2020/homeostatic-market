"""Vol pilot FiLM — Two-Stream Macro-Modulated Volatility Forecasting.

vol_pilot_v2 (단일 stream) 의 cond 평탄 처리에서 추가 변수가 noise 였던 것을
Two-Stream FiLM 으로 명시적 분리하여 우회.

Architecture:
  Market backbone: x_market = [sp_return past, tbill_full]
    - L=104 (past 52 + future 52)
    - tbill 만 future 활성 (시나리오 input)
    - Causal Transformer (D_MODEL=128, n_heads=4, n_layers=3)
  Macro modulator: x_macro = [m2_yoy_lag, gdp_yoy_lag, cpi_yoy_lag,
                              bondpp_13w_lag, stockpp_13w_lag, excess_liq_yoy_lag]
    - 모두 lag 변수 → future mask=0
    - point-wise MLP → bottleneck → γ, β
    - γ, β projection layers Zero-init (학습 초기 발산 방지)
  FiLM fusion:
    h_modulated = (1 + γ) * h_market + β
  Output:
    h_modulated 의 future portion (52 step) mean pool → Linear → 1 scalar (log_std)

Target:
  y = log( std(future 52w sp_return) )  per origin (vol_pilot_v2 동일)

Loss:
  MSE on log_std (vol_pilot_v2 동일)

Eval:
  test MSE / R² / Pearson + Spearman (predicted std vs actual std)

ckpt prefix: vol_pilot_film_*
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

# Market backbone 변수 — base 4ch 모두 포함 (단순 cond 처리에서 ranking 학습 입증)
# (tbill: future 활성, m2/gdp/cpi: lag 변수 future mask=0, sp_return: target_past 로 처리)
COLS_MARKET = ["tbill_wr", "m2_yoy_lag", "gdp_yoy_lag", "cpi_yoy_lag"]  # cond idx 0..3

# Macro modulator 변수 — 화폐절하 지표만 (FiLM 으로 base 위에 modulation)
COLS_MACRO  = ["bondpp_13w_lag", "stockpp_13w_lag", "excess_liq_yoy_lag"]  # cond idx 4..6

# 전체 cond 순서: market + macro
COLS_COND = COLS_MARKET + COLS_MACRO  # 4 + 3 = 7 ch
COLS_TARGET = ["sp_return"]


# =====================================================================
# Model — Macro-Gated Two-Stream
# =====================================================================

class MacroGatedVolScalar(nn.Module):
    """Two-Stream FiLM-modulated Causal Transformer for vol scalar regression.

    Input:
      cond:        (B, L=104, d_cond=7)   [tbill, m2, gdp, cpi, bondpp, stockpp, excess_liq]
                                          future 부분 idx 1..6 = mask=0 (외부 mask_future_ch)
      target_past: (B, past_len=52, 1)    sp_return past
    Output:
      log_std_pred: (B,)
    """
    MARKET_DIM = 5  # sp_return (target_past) + 4 market cond (tbill, m2_lag, gdp_lag, cpi_lag)
    MACRO_DIM  = 3  # bondpp_13w_lag, stockpp_13w_lag, excess_liq_yoy_lag (화폐절하)

    def __init__(self, d_cond, d_model, n_heads, n_layers,
                 past_len, future_len, dropout=0.1):
        super().__init__()
        assert d_cond == 7, f"FilmGate vol expects d_cond=7 (4 market + 3 macro), got {d_cond}"

        self.past_len = past_len
        self.future_len = future_len
        self.total_len = past_len + future_len  # 104

        # === Market Backbone (Causal Transformer) ===
        self.market_proj = nn.Linear(self.MARKET_DIM, d_model)
        self.market_pos_emb = nn.Embedding(self.total_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.market_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        causal_mask = torch.triu(
            torch.ones(self.total_len, self.total_len), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # === Macro Modulator (MLP, bottleneck = d_model // 4) ===
        bottleneck = d_model // 4   # 32 when d_model=128
        self.macro_mlp = nn.Sequential(
            nn.Linear(self.MACRO_DIM, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
        )
        # γ, β projection — Zero-init (Hard rule: 학습 초기 안정)
        self.macro_gamma = nn.Linear(bottleneck, d_model)
        self.macro_beta  = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.macro_gamma.weight)
        nn.init.zeros_(self.macro_gamma.bias)
        nn.init.zeros_(self.macro_beta.weight)
        nn.init.zeros_(self.macro_beta.bias)

        # === Output Head: future mean pool → scalar ===
        self.scalar_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, cond, target_past):
        B = cond.shape[0]
        device = cond.device

        # === 슬라이싱 ===
        market_cond = cond[:, :, 0:4]                               # (B, 104, 4) — tbill + m2 + gdp + cpi
        macro_full  = cond[:, :, 4:7]                               # (B, 104, 3) — bondpp + stockpp + excess_liq

        # target_full sp: past 52 + future 52 (future = 0)
        target_full = torch.zeros(B, self.total_len, 1, dtype=cond.dtype, device=device)
        target_full[:, :self.past_len, :] = target_past             # (B, 52, 1)

        x_market = torch.cat([target_full, market_cond], dim=-1)    # (B, 104, 5) — sp + 4 base cond
        x_macro  = macro_full                                       # (B, 104, 3)

        # === Market Backbone ===
        h = self.market_proj(x_market)                              # (B, 104, d_model)
        pos = torch.arange(self.total_len, device=device)
        h = h + self.market_pos_emb(pos).unsqueeze(0)
        h_market = self.market_encoder(h, mask=self.causal_mask, is_causal=True)

        # === Macro Modulator ===
        m_hidden = self.macro_mlp(x_macro)                          # (B, 104, bottleneck)
        gamma = self.macro_gamma(m_hidden)                          # (B, 104, d_model)
        beta  = self.macro_beta(m_hidden)                           # (B, 104, d_model)

        # === FiLM Fusion ===
        h_modulated = (1.0 + gamma) * h_market + beta               # (B, 104, d_model)

        # === Future portion mean pool → scalar ===
        h_future = h_modulated[:, self.past_len:, :].mean(dim=1)    # (B, d_model)
        log_std_pred = self.scalar_head(h_future).squeeze(-1)       # (B,)
        return log_std_pred


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

    sp_idx = cols_target.index("sp_return")
    future_sp = X[:, PAST_LEN:, sp_idx]
    actual_std = future_sp.std(axis=1, ddof=1)
    actual_std = np.maximum(actual_std, 1e-8)
    y_log_std = np.log(actual_std).astype(np.float32)

    X_past = X[:, :PAST_LEN, :]

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
        max_epochs, patience, batch, lr, fold_tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cols_cond   = COLS_COND
    cols_target = COLS_TARGET
    mask_future = list(range(1, len(cols_cond)))  # tbill (idx 0) 만 future 활성

    fold_str = f"_{fold_tag}" if fold_tag else ""
    tag_full = f"vol_pilot_film_filmgate{fold_str}_seed{seed}"
    ckpt_path    = os.path.join(save_dir, f"{tag_full}_best.pt")
    summary_path = os.path.join(save_dir, f"{tag_full}_summary.json")
    log_path     = os.path.join(save_dir, f"{tag_full}_trainlog.csv")
    pred_path    = os.path.join(save_dir, f"{tag_full}_test_preds.npz")

    if os.path.exists(ckpt_path) and os.path.exists(summary_path):
        print(f"[SKIP] {tag_full} — ckpt + summary 이미 존재")
        with open(summary_path) as f:
            return json.load(f)

    print(f"\n{'='*72}")
    print(f"[FilmGate vol{fold_str}] seed={seed}")
    print(f"  Market    = {COLS_MARKET} + sp_return  (Causal Transformer)")
    print(f"  Macro     = {COLS_MACRO}  (FiLM modulator → γ, β)")
    print(f"  d_cond    = {len(cols_cond)}  (1 market + 6 macro)")
    print(f"  mask_future_ch = {mask_future}  (tbill 만 future 활성)")
    print(f"  L={L} (past={PAST_LEN} + future={FUTURE_LEN})")
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

    y_train_mean = float(ytr_.mean())
    baseline_test_mse = float(((yte.numpy() - y_train_mean) ** 2).mean())
    print(f"  log_std baseline (train mean) = {y_train_mean:+.4f}")
    print(f"  baseline test MSE (constant pred) = {baseline_test_mse:.6f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MacroGatedVolScalar(
        d_cond=len(cols_cond),
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
    pat = 0
    log = []

    def eval_split(X_past, C, y):
        model.eval()
        with torch.no_grad():
            pred = model(C.to(device), X_past.to(device)).cpu().numpy()
            actual = y.numpy()
            mse = float(((pred - actual) ** 2).mean())
            pred_std = np.exp(pred)
            actual_std = np.exp(actual)
            pearson = pearson_corr(pred_std, actual_std)
            spearman = spearman_corr(pred_std, actual_std)
        return mse, pearson, spearman, pred, actual

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(Xtr_past_.shape[0])
        losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            xp = Xtr_past_[idx].to(device)
            cb = Ctr_[idx].to(device)
            yb = ytr_[idx].to(device)
            pred = model(cb, xp)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_mse,  val_p,  val_s,  _, _ = eval_split(Xv_past,  Cv,  yv)
        test_mse, test_p, test_s, test_pred, test_actual = eval_split(Xte_past, Cte, yte)
        val_r2  = 1.0 - val_mse  / float(((yv.numpy()  - y_train_mean) ** 2).mean() + 1e-12)
        test_r2 = 1.0 - test_mse / baseline_test_mse if baseline_test_mse > 0 else float("nan")

        print(f"  ep{epoch:>3d}  train_mse={train_loss:.5f}  "
              f"val_mse={val_mse:.5f} (R²={val_r2:+.3f})  "
              f"test_mse={test_mse:.5f} (R²={test_r2:+.3f})  "
              f"test_pearson={test_p:+.3f}  test_spearman={test_s:+.3f}")

        log.append(dict(epoch=epoch, train_mse=train_loss,
                        val_mse=val_mse, val_r2=val_r2,
                        test_mse=test_mse, test_r2=test_r2,
                        test_pearson=test_p, test_spearman=test_s))

        if not np.isfinite(val_mse):
            pat += 1
        elif val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test_mse = test_mse
            best_test_r2 = test_r2
            best_test_pearson = test_p
            best_test_spearman = test_s
            best_test_pred = test_pred
            best_test_actual = test_actual
            best_epoch = epoch
            pat = 0
        else:
            pat += 1

        if (not np.isfinite(test_mse)) or test_mse > 100.0:
            print(f"  ⚠ divergence (test_mse={test_mse:.2f}) — early termination")
            break
        if pat >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val_mse={best_val:.5f}")
    print(f"    test MSE   = {best_test_mse:.5f}")
    print(f"    test R²    = {best_test_r2:+.4f}")
    print(f"    test Pearson  (std)  = {best_test_pearson:+.4f}")
    print(f"    test Spearman (std)  = {best_test_spearman:+.4f}")
    print(f"    baseline test MSE   = {baseline_test_mse:.5f}")

    os.makedirs(save_dir, exist_ok=True)
    if best_state is not None:
        torch.save({
            "model_state":     best_state,
            "model_type":      "macro_gated_vol_scalar",
            "cond_cols":       cols_cond,
            "target_cols":     cols_target,
            "stats_cond":      stats_c,
            "mask_future_ch":  mask_future,
            "config": dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                           d_cond=len(cols_cond), past_len=PAST_LEN,
                           future_len=FUTURE_LEN, total_len=L),
        }, ckpt_path)
        np.savez(pred_path,
                 y_pred_log_std=best_test_pred,
                 y_actual_log_std=best_test_actual,
                 y_pred_std=np.exp(best_test_pred),
                 y_actual_std=np.exp(best_test_actual),
                 y_train_log_std_mean=y_train_mean,
                 baseline_test_mse=baseline_test_mse)
        print(f"  saved test preds: {pred_path}")

    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        model_type="macro_gated_vol_scalar",
        fold=fold_tag,
        seed=seed,
        cond_cols=cols_cond,
        target_cols=cols_target,
        market_cols=COLS_MARKET + ["sp_return"],
        macro_cols=COLS_MACRO,
        mask_future_ch=mask_future,
        best_epoch=best_epoch,
        val_mse=best_val,
        test_mse=best_test_mse,
        test_r2=best_test_r2,
        test_pearson_std=best_test_pearson,
        test_spearman_std=best_test_spearman,
        y_train_log_std_mean=y_train_mean,
        baseline_test_mse=baseline_test_mse,
        n_params=n_params,
        device=str(device),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {ckpt_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Vol pilot FiLM-modulated trainer")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--fold", default=None,
                    help="F1/F2/F3 (folds_v33) — pilot-split 사용 시 None")
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--val-csv",   default=None)
    ap.add_argument("--test-csv",  default=None)
    ap.add_argument("--out-dir",   default=os.path.join(HERE, "result"))
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience",   type=int, default=PATIENCE)
    ap.add_argument("--batch",      type=int, default=BATCH)
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--pilot-split", action="store_true",
                    help="data/pilot_split/{train,val,test}.csv 사용")
    args = ap.parse_args()

    train_csv = val_csv = test_csv = None
    if args.pilot_split:
        repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
        split_dir = os.path.join(repo_root, "data", "pilot_split")
        train_csv = os.path.join(split_dir, "train.csv")
        val_csv   = os.path.join(split_dir, "val.csv")
        test_csv  = os.path.join(split_dir, "test.csv")
        print(f"[--pilot-split]")
        print(f"  train = {train_csv}")
        print(f"  val   = {val_csv}")
        print(f"  test  = {test_csv}")
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
            batch=args.batch, lr=args.lr, fold_tag=fold_tag)


if __name__ == "__main__":
    main()
