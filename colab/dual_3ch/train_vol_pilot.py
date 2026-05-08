"""Volatility-only pilot — μ=0 고정, σ_t (52주 path) 만 학습.

paper 새 main thesis 후보 = "Macro-Conditional Volatility Forecasting":
  Model A (이 파일): tbill 시나리오 + 매크로 lag → sp_return 의 σ_t path 예측
  Model B (별도, 추후): 표준화 잔차 ε_t = r_t / σ_t 의 fat-tail 분포
  결합 시나리오: r_t = σ_t × ε_t

본 파일럿은 Model A 의 학습 가능성 검증만 (variant 1 base, 1 fold × 1 seed).

Architecture:
  - Causal Transformer (1-step / multi-step Tr 매트릭스와 동일 backbone)
  - Head: log_var 1 출력 (μ_t = 0 고정)
  - Window L=104 (past 52 + future 52)

Loss (μ=0 고정 Gaussian NLL):
  L_t = 0.5 * (log 2π + log σ_t² + r_t² / σ_t²)
  total = mean over (B, future_len=52)

Baseline (정규 marginal, σ = train σ_sp constant):
  H_marginal = 0.5 * log(2π e σ²) — 1-step 매트릭스와 동일 baseline
  Path 평가도 같은 단위 (per-step mean NLL)

평가 metric:
  (1) test mean per-step NLL  vs  H_marginal
  (2) Spearman corr (σ_t, |r_t|) on test future
  (3) σ_t path npz 저장 (시각화용)

ckpt prefix: vol_pilot_v* — 1-step / mtxf / matrix prefix 모두와 격리.
SKIP 로직 동일.
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

LOG2PI = math.log(2 * math.pi)

COLS_COND_BASE = ["tbill_wr", "m2_yoy_lag", "gdp_yoy_lag", "cpi_yoy_lag"]

# 파일럿은 variant 1 (base) 만. 매트릭스 확장은 본격 단계에서.
VARIANTS = {
    1: dict(name="base",       cond_extra=[]),
    2: dict(name="base_bp",    cond_extra=["bondpp_13w_lag"]),
    3: dict(name="base_sp",    cond_extra=["stockpp_13w_lag"]),
    4: dict(name="base_bp_sp", cond_extra=["bondpp_13w_lag", "stockpp_13w_lag"]),
}


def build_spec(variant_id):
    if variant_id not in VARIANTS:
        raise ValueError(f"variant must be in {list(VARIANTS)}, got {variant_id}")
    v = VARIANTS[variant_id]
    cols_cond = COLS_COND_BASE + v["cond_extra"]
    cols_target = ["sp_return"]
    mask_future_ch = list(range(1, len(cols_cond)))  # tbill (idx 0) 만 future 활성
    return dict(
        variant_id=variant_id,
        name=v["name"],
        cols_cond=cols_cond,
        cols_target=cols_target,
        mask_future_ch=mask_future_ch,
    )


# =====================================================================
# Model — μ=0 고정, σ_t (log_var) 만 출력
# =====================================================================

class CausalTransformerVolOnly(nn.Module):
    """μ_t = 0 fixed, log σ_t² 만 학습.

    Input:
      cond:        (B, L=104, d_cond)             past 52 + future 52 (tbill 활성)
      target_past: (B, past_len=52, 1)            sp_return past

    Output:
      log_var: (B, future_len=52)                 σ_t² log
    """
    def __init__(self, d_cond, d_model, n_heads, n_layers,
                 past_len, future_len, dropout=0.1):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.total_len = past_len + future_len  # 104

        self.input_dim = d_cond + 1  # +1 for sp_return past
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

        # Single head: log_var only (μ=0 고정)
        self.log_var_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),  # log σ²
        )

    def forward(self, cond, target_past):
        B = cond.shape[0]
        device = cond.device

        target_full = torch.zeros(B, self.total_len, 1, dtype=cond.dtype, device=device)
        target_full[:, :self.past_len, :] = target_past

        x = torch.cat([cond, target_full], dim=-1)  # (B, L, d_cond+1)
        x = self.input_proj(x)

        pos = torch.arange(self.total_len, device=device)
        x = x + self.pos_emb(pos).unsqueeze(0)

        h = self.encoder(x, mask=self.causal_mask, is_causal=True)  # (B, L, d_model)
        h_future = h[:, self.past_len:, :]                           # (B, future_len, d_model)

        log_var = self.log_var_head(h_future).squeeze(-1)            # (B, future_len)
        log_var = log_var.clamp(min=-10.0, max=10.0)
        return log_var


# =====================================================================
# Loss — Gaussian NLL with μ=0
# =====================================================================

def gaussian_nll_mu0(target, log_var):
    """μ=0 고정 Gaussian NLL.
       target: (B, future_len, 1)   sp_return future raw
       log_var: (B, future_len)     log σ_t²
       returns: (B, future_len) per-step NLL
    """
    r = target.squeeze(-1)                          # (B, future_len)
    return 0.5 * (LOG2PI + log_var + r.pow(2) * torch.exp(-log_var))


# =====================================================================
# Data
# =====================================================================

def load_windows(csv_path, cols_cond, cols_target, L=104, cond_stats=None):
    df = pd.read_csv(csv_path)
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        return None, None, None
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

    return torch.from_numpy(X), torch.from_numpy(C), cond_stats_out


def mask_future_channels(C, past_len, mask_channels):
    C = C.clone()
    for ch in mask_channels:
        C[:, past_len:, ch] = 0.0
    return C


# =====================================================================
# Spearman correlation (no scipy 의존)
# =====================================================================

def spearman_corr(a, b):
    """Spearman rank correlation between 1D arrays a and b."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


# =====================================================================
# Train
# =====================================================================

def run(spec, train_csv, val_csv, test_csv, save_dir, seed,
        max_epochs, patience, batch, lr, fold_tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cols_cond   = spec["cols_cond"]
    cols_target = spec["cols_target"]
    mask_future = spec["mask_future_ch"]

    fold_str = f"_{fold_tag}" if fold_tag else ""
    tag_full = f"vol_pilot_v{spec['variant_id']}_{spec['name']}{fold_str}_seed{seed}"
    ckpt_path    = os.path.join(save_dir, f"{tag_full}_best.pt")
    summary_path = os.path.join(save_dir, f"{tag_full}_summary.json")
    log_path     = os.path.join(save_dir, f"{tag_full}_trainlog.csv")
    sigma_path   = os.path.join(save_dir, f"{tag_full}_sigma_test.npz")

    if os.path.exists(ckpt_path) and os.path.exists(summary_path):
        print(f"[SKIP] {tag_full} — ckpt + summary 이미 존재")
        with open(summary_path) as f:
            return json.load(f)

    print(f"\n{'='*72}")
    print(f"[Variant {spec['variant_id']} = {spec['name']}{fold_str}] seed={seed}  "
          f"(Vol-only Causal Transformer, μ=0 fixed)")
    print(f"  cond   = {cols_cond}  (D_COND={len(cols_cond)})")
    print(f"  target = {cols_target}  (μ=0, log σ² 만 학습)")
    print(f"  mask_future_ch = {mask_future}  (tbill 만 future 활성)")
    print(f"  L={L} (past={PAST_LEN} + future={FUTURE_LEN})")
    print(f"{'='*72}")

    Xtr, Ctr, stats_c = load_windows(train_csv, cols_cond, cols_target, L=L)
    if Xtr is None:
        print(f"[FAIL] not enough train windows: {train_csv}")
        return None
    Xte, Cte, _ = load_windows(test_csv, cols_cond, cols_target, L=L, cond_stats=stats_c)
    if Xte is None:
        print(f"[FAIL] not enough test windows: {test_csv}")
        return None

    Ctr = mask_future_channels(Ctr, PAST_LEN, mask_future)
    Cte = mask_future_channels(Cte, PAST_LEN, mask_future)

    if val_csv is not None and os.path.exists(val_csv):
        Xv, Cv, _ = load_windows(val_csv, cols_cond, cols_target, L=L, cond_stats=stats_c)
        Cv = mask_future_channels(Cv, PAST_LEN, mask_future)
        Xtr_, Ctr_ = Xtr, Ctr
        print(f"  train_w={Xtr_.shape[0]} (full), val_w={Xv.shape[0]} (val_csv), test_w={Xte.shape[0]}")
    else:
        n_w_tr = Xtr.shape[0]
        n_val = max(int(n_w_tr * 0.15), 1)
        Xtr_, Ctr_ = Xtr[:-n_val], Ctr[:-n_val]
        Xv,  Cv    = Xtr[-n_val:], Ctr[-n_val:]
        print(f"  train_w={Xtr_.shape[0]}, val_w={Xv.shape[0]} (auto 15%), test_w={Xte.shape[0]}")

    # Marginal baseline — train sp σ
    sp_train_full = pd.read_csv(train_csv)["sp_return"].values
    sigma_marginal = float(sp_train_full.std())
    H_marginal_per_step = 0.5 * (math.log(2 * math.pi * math.e) + 2.0 * math.log(sigma_marginal))
    print(f"  sigma_marginal (train sp std)   = {sigma_marginal:.6f}")
    print(f"  H_marginal per step (baseline) = {H_marginal_per_step:+.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformerVolOnly(
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
    best_test_nll = None
    best_test_spearman = None
    best_test_sigma = None
    best_test_r = None
    pat = 0
    log = []

    def eval_split(X, C):
        model.eval()
        with torch.no_grad():
            target_past = X[:, :PAST_LEN, :].to(device)
            target_future = X[:, PAST_LEN:, :].to(device)  # (B, 52, 1)
            log_var = model(C.to(device), target_past)      # (B, 52)
            nll = gaussian_nll_mu0(target_future, log_var)  # (B, 52)
            mean_nll = float(nll.mean())
            sigma = torch.exp(0.5 * log_var).cpu().numpy()  # (B, 52)
            r = target_future.squeeze(-1).cpu().numpy()      # (B, 52)
            sp = spearman_corr(sigma.ravel(), np.abs(r).ravel())
        return mean_nll, sp, sigma, r

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(Xtr_.shape[0])
        losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            xb = Xtr_[idx].to(device)
            cb = Ctr_[idx].to(device)
            target_past = xb[:, :PAST_LEN, :]
            target_future = xb[:, PAST_LEN:, :]
            log_var = model(cb, target_past)
            nll = gaussian_nll_mu0(target_future, log_var)
            loss = nll.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_nll, val_sp, _, _ = eval_split(Xv, Cv)
        test_nll, test_sp, test_sigma, test_r = eval_split(Xte, Cte)

        print(f"  ep{epoch:>3d}  train={train_loss:+.4f}  "
              f"val_nll={val_nll:+.4f}  test_nll={test_nll:+.4f}  "
              f"val_spearman={val_sp:+.3f}  test_spearman={test_sp:+.3f}")

        log.append(dict(epoch=epoch, train_loss=train_loss,
                        val_nll=val_nll, test_nll=test_nll,
                        val_spearman=val_sp, test_spearman=test_sp))

        if not np.isfinite(val_nll):
            print(f"  ⚠ val_nll non-finite — best ckpt 갱신 차단")
            pat += 1
        elif val_nll < best_val - 1e-4:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test_nll = test_nll
            best_test_spearman = test_sp
            best_test_sigma = test_sigma
            best_test_r = test_r
            best_epoch = epoch
            pat = 0
        else:
            pat += 1

        if (not np.isfinite(test_nll)) or test_nll > 100.0:
            print(f"  ⚠ divergence (test_nll={test_nll:+.2f}) — early termination")
            break
        if pat >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val_nll={best_val:+.4f}, "
          f"test_nll={best_test_nll:+.4f}, test_spearman={best_test_spearman:+.3f}")
    print(f"  Δ(test_nll - H_marginal) = {best_test_nll - H_marginal_per_step:+.4f} nat/step")

    os.makedirs(save_dir, exist_ok=True)
    if best_state is not None:
        torch.save({
            "model_state":     best_state,
            "model_type":      "causal_transformer_vol_only",
            "cond_cols":       cols_cond,
            "target_cols":     cols_target,
            "stats_cond":      stats_c,
            "mask_future_ch":  mask_future,
            "config": dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                           d_cond=len(cols_cond), past_len=PAST_LEN,
                           future_len=FUTURE_LEN, total_len=L),
            "variant": dict(id=spec["variant_id"], name=spec["name"]),
        }, ckpt_path)
        np.savez(sigma_path,
                 sigma=best_test_sigma, r=best_test_r,
                 sigma_marginal=sigma_marginal,
                 H_marginal_per_step=H_marginal_per_step)
        print(f"  saved sigma path npz: {sigma_path}")

    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        model_type="causal_transformer_vol_only",
        variant_id=spec["variant_id"],
        variant_name=spec["name"],
        fold=fold_tag,
        seed=seed,
        cond_cols=cols_cond,
        target_cols=cols_target,
        mask_future_ch=mask_future,
        best_epoch=best_epoch,
        val_nll=best_val,
        test_nll=best_test_nll,
        test_spearman=best_test_spearman,
        sigma_marginal=sigma_marginal,
        H_marginal_per_step=H_marginal_per_step,
        delta_vs_marginal=(best_test_nll - H_marginal_per_step) if best_test_nll is not None else None,
        n_params=n_params,
        device=str(device),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {ckpt_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Volatility-only pilot (μ=0 fixed)")
    ap.add_argument("--variant", type=int, default=1, choices=list(VARIANTS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--fold", default="F1", choices=["F1", "F2", "F3"])
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--val-csv",   default=None)
    ap.add_argument("--test-csv",  default=None)
    ap.add_argument("--out-dir",   default=os.path.join(HERE, "result"))
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience",   type=int, default=PATIENCE)
    ap.add_argument("--batch",      type=int, default=BATCH)
    ap.add_argument("--lr",         type=float, default=LR)
    args = ap.parse_args()

    spec = build_spec(args.variant)

    if args.fold is not None:
        repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
        folds_dir = os.path.join(repo_root, "data", "folds_v33")
        train_csv = os.path.join(folds_dir, f"{args.fold}_train.csv")
        val_csv   = os.path.join(folds_dir, f"{args.fold}_val.csv")
        test_csv  = os.path.join(folds_dir, f"{args.fold}_test.csv")
        print(f"[--fold {args.fold}]")

    if args.train_csv: train_csv = args.train_csv
    if args.val_csv:   val_csv   = args.val_csv
    if args.test_csv:  test_csv  = args.test_csv

    os.makedirs(args.out_dir, exist_ok=True)

    for seed in args.seeds:
        run(spec, train_csv, val_csv, test_csv, args.out_dir, seed,
            max_epochs=args.max_epochs, patience=args.patience,
            batch=args.batch, lr=args.lr, fold_tag=args.fold)


if __name__ == "__main__":
    main()
