"""Sequence-conditioned Flow — Conditional independence across 13 weeks.

User design 의도 (chat/215ace1f.jsonl L47, L101, 2026-05-10):
  - in_cond(past 52w):   sp_return + macro 7 = 8 channel sequence
  - in_cond(future 13w): tbill_wr 만 unmask (다른 macro mask)
  - Output: 13-step sp_return path, **13 step 간 conditional i.i.d. given encoder embedding**

Architecture (2026-05-13 변경 — joint AR 폐기, conditional i.i.d. 로 복귀):
  Encoder = TransformerEncoder (v14 CausalTransformerVolMTL encoder 구조 차용)
           input  (B, 65, 8)  ← past 52 + future 13, future 의 tbill_wr 만 unmask
           output (B, d_model) ← pooled embedding C_t
  Decoder = 1D Conditional NSF (features=1)
           MaskedPiecewiseRationalQuadraticAutoregressiveTransform × 6
           features=1, context_features=d_model
  Loss   = -mean over (origin, week) of log p(ε_w | C_t)
           각 origin 의 13 ε samples 가 same C_t 받음 → conditional i.i.d.

Output (per fold):
  result/flow_seq_{fold}_best.pt
  result/flow_seq_{fold}_log.csv

Usage:
  python colab/dual_3ch/train_flow_seq.py --fold F_long
"""
import argparse
import json
import math
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
sys.path.insert(0, HERE)

try:
    from nflows.flows.base import Flow
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms import (
        CompositeTransform,
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
        ReversePermutation,
    )
except ImportError:
    sys.exit("FATAL: nflows required.  pip install nflows")

# =====================================================================
# Constants
# =====================================================================

PAST_LEN   = 52
FUTURE_LEN = 13
L          = PAST_LEN + FUTURE_LEN   # 65

# Cond sequence channel definition (8 channels)
#   index 0: sp_return         (past only, future mask)
#   index 1: tbill_wr          (past + future BOTH unmask — 사용자 강조)
#   index 2..7: macro 6        (past only, future mask)
COND_COLS = [
    "sp_return",          # 0  past only
    "tbill_wr",           # 1  past + FUTURE unmask  ← 사용자 강조
    "m2_13w_cum_lag",     # 2  past only
    "ads_lag",            # 3  past only  ← 경기순환 cond (COVID robust, INDPRO 와 상관 0.5 = 고유정보)
    "cpi_13w_cum_lag",    # 4  past only
    "indpro_13w_pct_lag", # 5  past only  ← 실물생산 (metab=m2-INDPRO-cpi 핵심 분모). ADS 와 역할 분리: ADS=경기 cond / INDPRO=화폐가치절하 실물축.
    "sp_std_13w",         # 6  past only
    "wti_wr",             # 7  past only
    "sp_log_std_13w",     # 8  past only
]
N_CHANNELS = len(COND_COLS)
TBILL_CH = 1                                       # 미래 unmask 인 channel
MASK_FUTURE_CH = [i for i in range(N_CHANNELS) if i != TBILL_CH]   # 0, 2..7

# Architecture defaults (v14 와 일관)
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 3
N_FLOW_LAYERS    = 6
N_FLOW_HIDDEN    = 64
N_FLOW_BLOCKS    = 2
N_FLOW_BINS      = 16
FLOW_TAIL_BOUND  = 10.0

LR        = 1e-4
BATCH     = 32
MAX_EPOCH = 60
PATIENCE  = 30
GRAD_CLIP = 1.0


# =====================================================================
# Model
# =====================================================================

class SequenceEncoder(nn.Module):
    """Transformer encoder for cond sequence (B, L, N_CHANNELS) → (B, d_model).

    Future 13w 의 mask 채널 (MASK_FUTURE_CH) 은 0 으로 zero-out 처리하여 정보 차단.
    Tbill (channel 1) 의 future 값은 그대로 유지 → 사용자 강조 미래 금리 시나리오 input.
    """
    def __init__(self, d_input, d_model, n_heads, n_layers, past_len, future_len):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.past_len = past_len
        self.future_len = future_len
        self.L = past_len + future_len
        self.input_proj = nn.Linear(d_input, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(self.L, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool_proj = nn.Linear(d_model, d_model)

    def forward(self, x_cond):
        # x_cond: (B, L, N_CHANNELS) — already future-masked at MASK_FUTURE_CH
        h = self.input_proj(x_cond) + self.pos_emb.unsqueeze(0)
        h = self.encoder(h)
        # Pool: take last token (last future step) — captures both past + future info
        pooled = h[:, -1, :]
        return self.pool_proj(pooled)


def build_1d_cond_flow(context_features, num_layers, hidden_features,
                       num_blocks, num_bins, tail_bound):
    """1D Conditional NSF (features=1).

    매 week 별로 same flow 호출 → 13 step conditional i.i.d. given encoder embedding.
    base = StandardNormal(shape=[1])
    transforms = MaskedPiecewiseRationalQuadraticAutoregressiveTransform × num_layers
                 (features=1 이면 ReversePermutation 의미 없음 — 생략)
    """
    base = StandardNormal(shape=[1])
    transforms = []
    for _ in range(num_layers):
        transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=1,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
        ))
    return Flow(CompositeTransform(transforms), base)


# Backward-compat alias (for old ckpts loading sensitivity_flow_seq.py)
build_joint_flow = build_1d_cond_flow   # deprecated, will be removed


class SequenceCondFlow(nn.Module):
    """Encoder + 1D Conditional NSF wrapper (13 step 간 conditional i.i.d.)."""
    def __init__(self, encoder, flow, future_len=FUTURE_LEN):
        super().__init__()
        self.encoder = encoder
        self.flow = flow
        self.future_len = future_len

    def log_prob(self, target_path, x_cond):
        """target_path: (B, future_len), x_cond: (B, L, N_CHANNELS).

        Returns per-origin log p = sum over weeks of log p(ε_w | C_t).
        For batch loss: caller takes .mean() / future_len for per-(origin, week) average.
        """
        B, T = target_path.shape
        ctx = self.encoder(x_cond)                       # (B, d_model)
        # Flatten (B, T) → (B*T, 1), repeat ctx
        y_flat = target_path.reshape(B * T, 1)           # (B*T, 1)
        ctx_rep = ctx.repeat_interleave(T, dim=0)        # (B*T, d_model)
        log_p_flat = self.flow.log_prob(inputs=y_flat, context=ctx_rep)   # (B*T,)
        # Reshape back and sum over weeks
        log_p = log_p_flat.reshape(B, T).sum(dim=1)      # (B,)
        return log_p

    @torch.no_grad()
    def sample(self, n_sim, x_cond):
        """Sample n_sim × future_len conditional i.i.d. samples per origin.

        Returns (B, n_sim, future_len).
        """
        B = x_cond.shape[0]
        T = self.future_len
        ctx = self.encoder(x_cond)                       # (B, d_model)
        # 13 i.i.d. samples per (origin, path)
        samples = self.flow.sample(n_sim * T, context=ctx)   # (B, n_sim*T, 1)
        samples = samples.reshape(B, n_sim, T)
        return samples


# =====================================================================
# Data loader
# =====================================================================

def load_windows_seq(csv_path, target_col="sp_return", past_len=PAST_LEN, future_len=FUTURE_LEN,
                    cond_stats=None, target_stats=None):
    """Build (X_cond, Y_target) windows.

    X_cond: (n_w, L, N_CHANNELS) — full cond sequence (future not yet masked here)
    Y_target: (n_w, future_len) — sp_return path standardized

    Standardization:
      - cond features (per channel) z-score on train (sp_return 도 동일)
      - target sp_return standardize separately (target_stats: mean, std)
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in COND_COLS if c not in df.columns]
    if missing:
        sys.exit(f"[FATAL] missing cond cols in {csv_path}: {missing}")

    # Drop rows with any NaN in cond
    sub = df[COND_COLS].copy()
    sub_dropna = sub.dropna()
    if len(sub_dropna) < len(sub):
        # Keep alignment with original index — drop only at start
        pass
    cond_arr = sub.values.astype(np.float64)
    target_arr = df[target_col].values.astype(np.float64)
    n = len(df)
    n_w = n - past_len - future_len + 1
    if n_w <= 0:
        sys.exit(f"[FATAL] csv too short: n={n}, need ≥ {past_len + future_len}")

    # Standardize cond per channel using train stats
    if cond_stats is None:
        cmu = np.nanmean(cond_arr, axis=0)
        csd = np.nanstd(cond_arr, axis=0, ddof=1) + 1e-8
        cond_stats_out = {"mean": cmu.tolist(), "std": csd.tolist()}
    else:
        cmu = np.asarray(cond_stats["mean"], dtype=np.float64)
        csd = np.asarray(cond_stats["std"],  dtype=np.float64)
        cond_stats_out = cond_stats
    cond_z = (cond_arr - cmu) / csd     # NaN propagates

    # Standardize target sp_return using train stats
    if target_stats is None:
        tmu = float(np.nanmean(target_arr))
        tsd = float(np.nanstd(target_arr, ddof=1) + 1e-8)
        target_stats_out = {"mean": tmu, "std": tsd}
    else:
        tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
        target_stats_out = target_stats
    target_z = (target_arr - tmu) / tsd

    X_cond = np.zeros((n_w, past_len + future_len, N_CHANNELS), dtype=np.float32)
    Y_target = np.zeros((n_w, future_len), dtype=np.float32)
    valid = np.ones(n_w, dtype=bool)
    for t in range(n_w):
        win_cond = cond_z[t : t + past_len + future_len]   # (L, N_CHANNELS)
        win_tgt  = target_z[t + past_len : t + past_len + future_len]  # (future_len,)
        if np.any(np.isnan(win_cond)) or np.any(np.isnan(win_tgt)):
            valid[t] = False
            continue
        X_cond[t] = win_cond.astype(np.float32)
        Y_target[t] = win_tgt.astype(np.float32)

    X_cond = X_cond[valid]
    Y_target = Y_target[valid]
    return (
        torch.from_numpy(X_cond),
        torch.from_numpy(Y_target),
        cond_stats_out,
        target_stats_out,
        int(valid.sum()),
    )


def mask_future_channels(x_cond, past_len, mask_channels):
    """Zero out future 13w 의 specified channels (in-place 아닌 새 tensor 반환)."""
    out = x_cond.clone()
    for ch in mask_channels:
        out[:, past_len:, ch] = 0.0
    return out


# =====================================================================
# Train
# =====================================================================

def train(fold, train_csv, val_csv, save_path, log_path, summary_path,
          max_epoch=MAX_EPOCH, patience=PATIENCE, batch=BATCH, lr=LR,
          device="cuda", seed=2026):
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"\n[1] Load train + val")
    Xtr, Ytr, cond_stats, target_stats, n_tr = load_windows_seq(train_csv)
    print(f"    train windows = {n_tr}, X.shape={Xtr.shape}, Y.shape={Ytr.shape}")
    Xv, Yv, _, _, n_v = load_windows_seq(val_csv, cond_stats=cond_stats, target_stats=target_stats)
    print(f"    val windows   = {n_v}")
    if n_v == 0:
        sys.exit(f"[FATAL] no val windows in {val_csv}")

    # Apply future mask to BOTH train and val (사용자 design: tbill 만 future unmask)
    Xtr_m = mask_future_channels(Xtr, PAST_LEN, MASK_FUTURE_CH)
    Xv_m  = mask_future_channels(Xv,  PAST_LEN, MASK_FUTURE_CH)
    print(f"    future mask applied: only ch{TBILL_CH} (tbill_wr) visible in future 13w")

    # Build model
    encoder = SequenceEncoder(N_CHANNELS, D_MODEL, N_HEADS, N_LAYERS, PAST_LEN, FUTURE_LEN)
    flow = build_1d_cond_flow(
        context_features=D_MODEL,
        num_layers=N_FLOW_LAYERS,
        hidden_features=N_FLOW_HIDDEN,
        num_blocks=N_FLOW_BLOCKS,
        num_bins=N_FLOW_BINS,
        tail_bound=FLOW_TAIL_BOUND,
    )
    model = SequenceCondFlow(encoder, flow).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    model params = {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_ds = TensorDataset(Xtr_m, Ytr)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)
    Xv_dev = Xv_m.to(device); Yv_dev = Yv.to(device)

    best_val_nll = float("inf"); best_epoch = -1
    best_state = None
    log = []
    pat = 0

    print(f"\n[2] Train (max_epoch={max_epoch}, patience={patience}, batch={batch}, lr={lr})")
    for ep in range(1, max_epoch + 1):
        model.train()
        losses = []
        for xb, yb in train_dl:
            xb = xb.to(device); yb = yb.to(device)
            loss = -model.log_prob(yb, xb).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())
        train_nll = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            val_nll = float(-model.log_prob(Yv_dev, Xv_dev).mean().item())

        log.append(dict(epoch=ep, train_nll=train_nll, val_nll=val_nll))
        improved = val_nll < best_val_nll
        if improved:
            best_val_nll = val_nll; best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if ep == 1 or ep % 5 == 0 or improved or ep == max_epoch:
            mark = " ★" if improved else ""
            # per-origin = sum over 13 weeks 의 NLL.  per-week = nll / 13.
            print(f"    ep{ep:>3d}: train_nll={train_nll:+.4f} (per-week {train_nll/FUTURE_LEN:+.3f})"
                  f"  val_nll={val_nll:+.4f} (per-week {val_nll/FUTURE_LEN:+.3f})"
                  f"  (best {best_val_nll:+.4f} @ep{best_epoch}){mark}")
        if pat >= patience:
            print(f"    [early stop] ep{ep}  (patience {patience} from ep{best_epoch})")
            break

    # Save best ckpt
    torch.save({
        "model_state": best_state,
        "meta": dict(
            fold=fold,
            cond_cols=COND_COLS, tbill_ch=TBILL_CH, mask_future_ch=MASK_FUTURE_CH,
            past_len=PAST_LEN, future_len=FUTURE_LEN,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            n_flow_layers=N_FLOW_LAYERS, n_flow_hidden=N_FLOW_HIDDEN,
            n_flow_blocks=N_FLOW_BLOCKS, n_flow_bins=N_FLOW_BINS,
            flow_tail_bound=FLOW_TAIL_BOUND,
            cond_stats=cond_stats, target_stats=target_stats,
            best_epoch=best_epoch, best_val_nll=best_val_nll,
            n_train=int(n_tr), n_val=int(n_v),
            n_params=int(n_params),
            seed=seed,
        ),
    }, save_path)
    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        fold=fold, best_epoch=best_epoch, best_val_nll=best_val_nll,
        n_train=int(n_tr), n_val=int(n_v), n_params=int(n_params),
        cond_cols=COND_COLS, mask_future_ch=MASK_FUTURE_CH,
        target_stats=target_stats, cond_stats=cond_stats,
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  saved ckpt: {save_path}")
    print(f"  saved log : {log_path}")
    print(f"  saved summary: {summary_path}")
    print(f"\n  best val NLL = {best_val_nll:+.4f} @ep{best_epoch}")


def main():
    ap = argparse.ArgumentParser(description="Sequence-conditioned Flow (past 52w + future 13w tbill → 13-step sp_return path)")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--out-dir",   default=os.path.join(HERE, "result"))
    ap.add_argument("--max-epoch", type=int, default=MAX_EPOCH)
    ap.add_argument("--patience",  type=int, default=PATIENCE)
    ap.add_argument("--batch",     type=int, default=BATCH)
    ap.add_argument("--lr",        type=float, default=LR)
    ap.add_argument("--seed",      type=int, default=2026)
    args = ap.parse_args()

    train_csv = os.path.join(args.folds_dir, f"{args.fold}_train.csv")
    val_csv   = os.path.join(args.folds_dir, f"{args.fold}_val.csv")
    for p in [train_csv, val_csv]:
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing {p}")
    os.makedirs(args.out_dir, exist_ok=True)
    save_path    = os.path.join(args.out_dir, f"flow_seq_{args.fold}_best.pt")
    log_path     = os.path.join(args.out_dir, f"flow_seq_{args.fold}_log.csv")
    summary_path = os.path.join(args.out_dir, f"flow_seq_{args.fold}_summary.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(f" Sequence-cond Flow training — fold={args.fold}")
    print(f"  past 52w channels: all 8 unmask  ({COND_COLS})")
    print(f"  future 13w channel: only ch{TBILL_CH} ({COND_COLS[TBILL_CH]}) unmask")
    print(f"  output: 13-step sp_return — conditional i.i.d. given encoder embedding")
    print(f"  flow: 1D Cond NSF (features=1), 13 step 간 shared NSF, same context")
    print(f"  device: {device}")
    print("=" * 78)

    train(args.fold, train_csv, val_csv, save_path, log_path, summary_path,
          max_epoch=args.max_epoch, patience=args.patience,
          batch=args.batch, lr=args.lr, device=device, seed=args.seed)


if __name__ == "__main__":
    main()
