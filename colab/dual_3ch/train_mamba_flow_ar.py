"""Mamba AR + Conditional Flow head -- End-to-End sp_return distribution model.

Paper thesis (post Gemini critique 2026-05-13):
  Traditional finance models split (predictable mean mu) and (uncontrollable
  residual epsilon) -- e.g. HAR-Ridge baseline.  Under structural crisis this
  decoupling causes fatal tail-risk underestimation.  This model combines
  Mamba (sequence state encoder) with a Conditional Flow head and maps the
  ENTIRE return distribution End-to-End, without ever splitting mean from
  residual.  Flow density learns directly on sp_return z-score conditioned
  on the per-step Mamba hidden state.

Architecture:
  Input  : (B, L=65, 8) z-score sequence
           [sp_return  : 1-step shifted (input[tau].sp = sp[tau-1]),
                         teacher-forced in training / AR-sampled in inference,
            tbill_wr   : real future scenario, UNMASKED for tau >= PAST_LEN,
            macro 6    : future positions [PAST_LEN, L) zero-masked]
  Encoder: Linear(8 -> d_model) + learned pos embedding + Mamba-SSM stack
           (mamba-ssm pkg, d_model=128, n_layers=3, d_state=16, d_conv=4,
           expand=2; pre-norm + residual block, final LayerNorm).
           Mamba is causal SSM -> hidden h_tau depends only on input[0..tau],
           so combined with the 1-step shift on the sp channel the target
           sp_return[tau] never leaks.
  Head   : 1D Conditional NSF (features=1, context_features=d_model, same
           spec as train_flow_seq.py).  For tau in [PAST_LEN, L):
              log p(sp_z[tau] | h_tau).
           NO mean/residual split -- Flow learns the full conditional density.

Training (teacher forcing):
  Loss = -mean over (origin, tau in [PAST_LEN, L)) of log p(sp_z[tau] | h_tau).
  Optimizer: AdamW lr=1e-4, batch=32, max_epoch=60, patience=30
             (matched to train_flow_seq.py for fair comparison).

Inference (AR rollout):
  For tau = 0 .. FUTURE_LEN-1:
    Append input[PAST_LEN + tau] = (last_sp (real last past sp at tau=0,
                                              sampled at tau>0),
                                    future_tbill_z[tau],
                                    macro 6 zero-mask).
    Forward Mamba on the growing sequence -> h_{PAST_LEN+tau}.
    Sample sp_z[tau] ~ Flow( . | h_{PAST_LEN+tau} ).
    Set last_sp = sp_z[tau] -- feedback for next iteration.
  n_sim independent paths per test origin.  Sampled paths converted to raw
  sp_return units for scenario metrics (CRPS / EMD / VaR / CVaR / coverage).

Output (per fold):
  result/mamba_flow_ar_{fold}_best.pt          (state + meta)
  result/mamba_flow_ar_{fold}_log.csv          (epoch train/val NLL)
  result/mamba_flow_ar_{fold}_summary.json     (train summary + test eval)
  result/mamba_flow_ar_{fold}_predictions.csv  (origin, per-step actual vs sim)
  result/mamba_flow_ar_{fold}_fanchart.png
  result/mamba_flow_ar_{fold}_histogram.png

Usage:
  python colab/dual_3ch/train_mamba_flow_ar.py --fold F_long
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
    )
except ImportError:
    sys.exit("FATAL: nflows required.  pip install nflows")

try:
    from mamba_ssm import Mamba
except ImportError:
    sys.exit("FATAL: mamba-ssm required.  "
             "pip install mamba-ssm causal-conv1d --no-build-isolation  "
             "(GPU / CUDA only)")

# Reuse channel + window constants from train_flow_seq.py for full consistency
from train_flow_seq import COND_COLS, TBILL_CH, PAST_LEN, FUTURE_LEN

N_CHANNELS = len(COND_COLS)                                  # 8
SP_CH      = COND_COLS.index("sp_return")                    # 0
MACRO_CH   = [c for c in range(N_CHANNELS) if c not in (SP_CH, TBILL_CH)]   # 6
L          = PAST_LEN + FUTURE_LEN                           # 65

# Architecture defaults (matched to train_flow_seq.py)
D_MODEL          = 128
N_MAMBA_LAYERS   = 3
MAMBA_D_STATE    = 16    # Tri Dao default
MAMBA_D_CONV     = 4     # Tri Dao default
MAMBA_EXPAND     = 2     # Tri Dao default
N_FLOW_LAYERS    = 6
N_FLOW_HIDDEN    = 64
N_FLOW_BLOCKS    = 2
N_FLOW_BINS      = 16
FLOW_TAIL_BOUND  = 10.0

# Train recipe (matched to train_flow_seq.py)
LR        = 1e-4
BATCH     = 32
MAX_EPOCH = 60
PATIENCE  = 30
GRAD_CLIP = 1.0


# =====================================================================
# Data loader (z-score, shifted sp, masked future macro)
# =====================================================================

# Module-level dataset cache for in-process sweeps -- avoids re-reading the
# same train/val/test csv across many (spec, fold) combinations.  Key is
# (csv_path, fingerprint of cond_stats, fingerprint of target_stats).
_DATA_CACHE = {}


def _cache_key(csv_path, cond_stats, target_stats):
    cs_key = None if cond_stats is None else (
        tuple(cond_stats["mean"]), tuple(cond_stats["std"]))
    ts_key = None if target_stats is None else (
        float(target_stats["mean"]), float(target_stats["std"]))
    return (csv_path, cs_key, ts_key)


def cached_load_windows_seq(csv_path, past_len=PAST_LEN, future_len=FUTURE_LEN,
                            cond_stats=None, target_stats=None):
    """Memoized wrapper around load_windows_seq.  Identical signature."""
    key = _cache_key(csv_path, cond_stats, target_stats)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    out = load_windows_seq(csv_path, past_len=past_len, future_len=future_len,
                           cond_stats=cond_stats, target_stats=target_stats)
    _DATA_CACHE[key] = out
    return out


def load_extra_context(csv_path, extra_cols, past_len=PAST_LEN, future_len=FUTURE_LEN,
                       extra_stats=None):
    """Load per-origin extra context features (origin-frozen).

    For each origin t in 0 .. n_w-1, extract the z-scored values of `extra_cols`
    at row (t + past_len - 1) -- the last past row, i.e. origin row.  Origin
    frozen policy mirrors macro-6 masking (운영 시 미래 derived 모름 가정).

    Returns (X_extra (n_w, len(extra_cols)), extra_stats, n_valid_origins,
             valid_mask).  Caller must align with load_windows_seq's valid mask
    by intersecting the two masks.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in extra_cols if c not in df.columns]
    if missing:
        sys.exit(f"[FATAL] missing extra context cols in {csv_path}: {missing}")
    arr = df[extra_cols].values.astype(np.float64)
    if extra_stats is None:
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1) + 1e-8
        extra_stats_out = {"mean": mu.tolist(), "std": sd.tolist(),
                           "cols": list(extra_cols)}
    else:
        mu = np.asarray(extra_stats["mean"], dtype=np.float64)
        sd = np.asarray(extra_stats["std"],  dtype=np.float64)
        extra_stats_out = extra_stats
    arr_z = (arr - mu) / sd
    n = len(df)
    n_w = n - past_len - future_len + 1
    out = np.zeros((n_w, len(extra_cols)), dtype=np.float32)
    valid = np.ones(n_w, dtype=bool)
    for t in range(n_w):
        row = arr_z[t + past_len - 1]    # origin row (last past)
        if np.any(np.isnan(row)):
            valid[t] = False
            continue
        out[t] = row.astype(np.float32)
    return torch.from_numpy(out[valid]), extra_stats_out, int(valid.sum()), valid


def load_windows_seq(csv_path, past_len=PAST_LEN, future_len=FUTURE_LEN,
                     cond_stats=None, target_stats=None):
    """Build (X_input, Y_target) windows with shifted sp + future macro mask.

    For each origin t in 0 .. n_w - 1 with n_w = n - past_len - future_len + 1:
      Skip if any NaN in the cond window cond_z[t : t + L] (this also covers
      target NaN since target is the sp_return channel of cond).
      Build sequence x of shape (L, N_CHANNELS):
        * z-score all 8 cond channels using train stats
        * shift the sp channel by 1: x[tau].sp = sp_z[tau - 1] for tau >= 1,
          x[0].sp = 0 (BOS padding -- training loss starts at tau=PAST_LEN so
          this padding has no effect on the loss).
        * zero-mask macro channels for positions [PAST_LEN, L):
          x[PAST_LEN:, c] = 0  for c in MACRO_CH.
        * tbill_wr stays untouched (unmask future scenario channel).
      Target y of shape (FUTURE_LEN,) = sp_return z at positions [PAST_LEN, L).

    Returns (X (n_v, L, N_CH), Y (n_v, FUTURE_LEN), cond_stats, target_stats,
    n_valid_origins).
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in COND_COLS if c not in df.columns]
    if missing:
        sys.exit(f"[FATAL] missing cond cols in {csv_path}: {missing}")

    cond_arr   = df[COND_COLS].values.astype(np.float64)
    target_arr = df["sp_return"].values.astype(np.float64)
    n = len(df)
    n_w = n - past_len - future_len + 1
    if n_w <= 0:
        sys.exit(f"[FATAL] csv too short: n={n}, need >= {past_len + future_len}")

    if cond_stats is None:
        cmu = np.nanmean(cond_arr, axis=0)
        csd = np.nanstd(cond_arr, axis=0, ddof=1) + 1e-8
        cond_stats_out = {"mean": cmu.tolist(), "std": csd.tolist()}
    else:
        cmu = np.asarray(cond_stats["mean"], dtype=np.float64)
        csd = np.asarray(cond_stats["std"],  dtype=np.float64)
        cond_stats_out = cond_stats

    if target_stats is None:
        tmu = float(np.nanmean(target_arr))
        tsd = float(np.nanstd(target_arr, ddof=1) + 1e-8)
        target_stats_out = {"mean": tmu, "std": tsd}
    else:
        tmu = float(target_stats["mean"])
        tsd = float(target_stats["std"])
        target_stats_out = target_stats

    cond_z   = (cond_arr   - cmu) / csd
    target_z = (target_arr - tmu) / tsd

    X_input  = np.zeros((n_w, past_len + future_len, N_CHANNELS), dtype=np.float32)
    Y_target = np.zeros((n_w, future_len), dtype=np.float32)
    valid    = np.ones(n_w, dtype=bool)

    for t in range(n_w):
        win_cond = cond_z[t : t + past_len + future_len]                   # (L, N_CH)
        win_tgt  = target_z[t + past_len : t + past_len + future_len]       # (FUTURE_LEN,)
        if np.any(np.isnan(win_cond)) or np.any(np.isnan(win_tgt)):
            valid[t] = False
            continue

        x = win_cond.astype(np.float32).copy()
        # Shift sp channel by 1: input[tau].sp = sp_z[tau - 1].
        sp_col = x[:, SP_CH].copy()
        x[1:, SP_CH] = sp_col[:-1]
        x[0,  SP_CH] = 0.0       # BOS padding (loss masks tau < PAST_LEN)
        # Zero-mask macro channels in the future portion [PAST_LEN, L).
        for ch in MACRO_CH:
            x[past_len:, ch] = 0.0
        # NOTE: For training, x[past_len:, SP_CH] currently holds shifted
        # real sp values (teacher forcing).  At inference these positions
        # are overwritten with sampled values (see MambaFlowAR.ar_sample).

        X_input[t]  = x
        Y_target[t] = win_tgt.astype(np.float32)

    X_input  = X_input[valid]
    Y_target = Y_target[valid]
    return (torch.from_numpy(X_input), torch.from_numpy(Y_target),
            cond_stats_out, target_stats_out, int(valid.sum()))


# ---- Derived (skip-connection) feature helpers ----------------------------
# Derived channels are NOT fed into the Mamba encoder; they are concatenated
# directly into the Flow head's context.  Extracted at the ORIGIN ROW
# (z-scored, frozen) and broadcast across all future steps.

def compute_derived_stats(csv_path, derived_cols):
    """Fit z-score stats for derived columns on the given csv (typically train)."""
    if not derived_cols:
        return None
    df = pd.read_csv(csv_path)
    missing = [c for c in derived_cols if c not in df.columns]
    if missing:
        sys.exit(f"[FATAL] missing derived cols in {csv_path}: {missing}")
    arr = df[derived_cols].values.astype(np.float64)
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1) + 1e-8
    return {"mean": mu.tolist(), "std": sd.tolist(), "cols": list(derived_cols)}


def extract_derived_origin(csv_path, derived_cols, derived_stats, valid_mask,
                           past_len=PAST_LEN, future_len=FUTURE_LEN):
    """For each VALID origin t, return z-scored derived values at the origin
    row (t + past_len - 1).  Returns torch.FloatTensor (n_valid, n_d) or None.
    """
    if not derived_cols:
        return None
    df = pd.read_csv(csv_path)
    arr = df[derived_cols].values.astype(np.float64)
    mu = np.asarray(derived_stats["mean"], dtype=np.float64)
    sd = np.asarray(derived_stats["std"],  dtype=np.float64)
    z = (arr - mu) / sd
    n = arr.shape[0]
    n_w = n - past_len - future_len + 1
    origin_row_idx = np.arange(n_w) + past_len - 1
    derived_all = z[origin_row_idx].astype(np.float32)
    return torch.from_numpy(derived_all[valid_mask])


def cond_valid_mask(csv_path, cond_stats, past_len=PAST_LEN, future_len=FUTURE_LEN):
    """Recompute valid-origin mask consistently with load_windows_seq.
    Used to align derived_origin extraction with (X, Y) from load_windows_seq.
    """
    df = pd.read_csv(csv_path)
    arr = df[COND_COLS].values.astype(np.float64)
    cmu = np.asarray(cond_stats["mean"], dtype=np.float64)
    csd = np.asarray(cond_stats["std"],  dtype=np.float64)
    z = (arr - cmu) / csd
    n = z.shape[0]
    n_w = n - past_len - future_len + 1
    L_full = past_len + future_len
    return np.array([not np.any(np.isnan(z[t : t + L_full])) for t in range(n_w)],
                    dtype=bool)


def compute_valid_mask(csv_path, cond_stats):
    """Recompute the valid origin mask consistently with load_windows_seq.
    Returns (valid_mask (n_w,) bool, full_z (n, N_CH)).
    """
    df = pd.read_csv(csv_path)
    arr = df[COND_COLS].values.astype(np.float64)
    cmu = np.asarray(cond_stats["mean"], dtype=np.float64)
    csd = np.asarray(cond_stats["std"],  dtype=np.float64)
    z = (arr - cmu) / csd
    n = z.shape[0]
    n_w = n - PAST_LEN - FUTURE_LEN + 1
    valid = np.array([not np.any(np.isnan(z[t : t + L])) for t in range(n_w)],
                     dtype=bool)
    return valid, z.astype(np.float32), df


# =====================================================================
# Model
# =====================================================================

class _MambaResBlock(nn.Module):
    """Pre-norm + residual wrapper around a single mamba-ssm Mamba block.

    Standard SSM block layout (Tri Dao 2023, Mamba paper):
      out = x + dropout(Mamba(LN(x))).
    """
    def __init__(self, d_model, d_state=MAMBA_D_STATE, d_conv=MAMBA_D_CONV,
                 expand=MAMBA_EXPAND, dropout=0.0):
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.mamba   = Mamba(d_model=d_model, d_state=d_state,
                             d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaEncoder(nn.Module):
    """Stack of pre-norm + residual Mamba blocks (mamba-ssm) with final LN.

    Replaces mambapy.Mamba(MambaConfig(...)) which wrapped multi-layer +
    norm internally.  Here we use the Tri Dao official mamba-ssm package
    (single block) and build the residual stack explicitly.

    Mamba blocks are causal SSMs -- hidden state at position tau depends
    only on inputs at positions [0..tau], so combined with the 1-step
    shift on the sp channel there is no target leakage.
    """
    def __init__(self, d_model, n_layers, d_state=MAMBA_D_STATE,
                 d_conv=MAMBA_D_CONV, expand=MAMBA_EXPAND, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _MambaResBlock(d_model, d_state=d_state, d_conv=d_conv,
                           expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class MambaFlowAR(nn.Module):
    """Mamba encoder + 1D Conditional NSF head, End-to-End sp_return density.

    log_prob:  teacher-forced sequence -> per-step log p(sp_z | h_tau).
    ar_sample: AR rollout, overwrites future sp positions with Flow.sample
               outputs iteratively and re-runs Mamba on the growing sequence.
    """
    def __init__(self, d_input=N_CHANNELS, d_model=D_MODEL,
                 n_mamba_layers=N_MAMBA_LAYERS,
                 n_flow_layers=N_FLOW_LAYERS, n_flow_hidden=N_FLOW_HIDDEN,
                 n_flow_blocks=N_FLOW_BLOCKS, n_flow_bins=N_FLOW_BINS,
                 flow_tail_bound=FLOW_TAIL_BOUND, dropout=0.0,
                 extra_context_dim=0,
                 past_len=PAST_LEN, future_len=FUTURE_LEN):
        super().__init__()
        self.d_input           = d_input
        self.d_model           = d_model
        self.extra_context_dim = extra_context_dim
        self.flow_context_dim  = d_model + extra_context_dim
        self.past_len          = past_len
        self.future_len        = future_len
        self.L                 = past_len + future_len

        self.input_proj = nn.Linear(d_input, d_model)
        self.pos_emb    = nn.Parameter(torch.zeros(self.L, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # mamba-ssm Mamba block expects (B, L, d_model) and returns the same
        # shape; it is intrinsically causal (selective scan SSM).  We wrap
        # it in a pre-norm + residual stack to get a multi-layer encoder.
        self.mamba = MambaEncoder(
            d_model=d_model, n_layers=n_mamba_layers,
            d_state=MAMBA_D_STATE, d_conv=MAMBA_D_CONV, expand=MAMBA_EXPAND,
            dropout=dropout,
        )

        # 1D Conditional NSF (same spec as train_flow_seq.build_1d_cond_flow).
        base = StandardNormal(shape=[1])
        transforms = []
        for _ in range(n_flow_layers):
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=1,
                hidden_features=n_flow_hidden,
                context_features=self.flow_context_dim,
                num_blocks=n_flow_blocks,
                num_bins=n_flow_bins,
                tails="linear",
                tail_bound=flow_tail_bound,
                dropout_probability=dropout,
            ))
        self.flow = Flow(CompositeTransform(transforms), base)

    def encode(self, x):
        """x: (B, L_cur, N_CHANNELS) -> h: (B, L_cur, d_model).

        Causal SSM: h[:, tau, :] depends only on x[:, 0..tau, :].  Together
        with the 1-step shift on the sp channel this guarantees that
        log p(sp_z[tau] | h_tau) never sees sp_z[tau] itself.

        The positional embedding is sliced to match the current length so
        that AR rollout (which grows the sequence step by step) reuses the
        same trained position-conditioned representation.
        """
        L_cur = x.shape[1]
        h = self.input_proj(x) + self.pos_emb[:L_cur].unsqueeze(0)
        h = self.mamba(h)
        return h

    def log_prob(self, x_input, target_path, extra_context=None):
        """Teacher-forced per-origin log probability summed over future steps.

        x_input      : (B, L, N_CHANNELS)
        target_path  : (B, FUTURE_LEN)
        extra_context: (B, extra_context_dim) or None  -- origin-frozen extra
                       signals (e.g. bondpp/stockpp/metab origin row z-score)
                       concatenated to h_tau as Flow context, NOT into Mamba.

        Returns (B,) -- sum over future steps of log p(sp_z[tau] | h_tau, extra).
        """
        B = x_input.shape[0]
        T = target_path.shape[1]
        h = self.encode(x_input)                                # (B, L, d_model)
        h_future = h[:, self.past_len:, :]                      # (B, T, d_model)
        if self.extra_context_dim > 0:
            if extra_context is None:
                raise ValueError("extra_context required when "
                                 "extra_context_dim > 0")
            extra_expanded = extra_context.unsqueeze(1).expand(-1, T, -1)
            ctx_seq = torch.cat([h_future, extra_expanded], dim=-1)    # (B, T, flow_ctx)
        else:
            ctx_seq = h_future
        y_flat   = target_path.reshape(B * T, 1)
        ctx_flat = ctx_seq.reshape(B * T, self.flow_context_dim)
        log_p_flat = self.flow.log_prob(inputs=y_flat, context=ctx_flat)
        log_p = log_p_flat.reshape(B, T).sum(dim=1)             # (B,)
        return log_p

    @torch.no_grad()
    def ar_sample(self, x_past, future_tbill_z, last_past_sp_z, n_sim,
                  extra_context=None):
        """AR rollout sampling.

        x_past        : (B, PAST_LEN, N_CHANNELS) -- input past portion
                        (sp channel already shifted-by-1 inside the past block
                        as produced by load_windows_seq).
        future_tbill_z: (B, FUTURE_LEN)           -- z-score future tbill scenario.
        last_past_sp_z: (B,)                      -- unshifted sp_z at row
                        (origin_idx + PAST_LEN - 1).  Seeds the sp channel of
                        the first appended future token (input[PAST_LEN].sp =
                        sp[PAST_LEN - 1]).
        n_sim         : int -- independent rollout paths per origin.

        Returns: sampled_paths (B, n_sim, FUTURE_LEN) -- sp_z samples.
        """
        device = x_past.device
        B = x_past.shape[0]
        N_CH = x_past.shape[-1]   # support arbitrary channel count after extra
        # Replicate the past block for each sim path.
        seq = x_past.unsqueeze(1).expand(B, n_sim, -1, -1).contiguous()
        seq = seq.view(B * n_sim, PAST_LEN, N_CH)
        tbill_fut = future_tbill_z.unsqueeze(1).expand(B, n_sim, -1).contiguous()
        tbill_fut = tbill_fut.view(B * n_sim, FUTURE_LEN)
        last_sp = last_past_sp_z.unsqueeze(1).expand(B, n_sim).contiguous().view(-1)
        # Extra context: replicate origin-frozen vector across sim paths.
        if self.extra_context_dim > 0:
            if extra_context is None:
                raise ValueError("extra_context required when "
                                 "extra_context_dim > 0")
            extra_rep = extra_context.unsqueeze(1).expand(
                B, n_sim, -1
            ).contiguous().view(B * n_sim, self.extra_context_dim)
        else:
            extra_rep = None

        sampled = []
        for tau in range(FUTURE_LEN):
            next_input = torch.zeros(B * n_sim, 1, N_CHANNELS,
                                     device=device, dtype=seq.dtype)
            next_input[:, 0, SP_CH]    = last_sp                # shifted sp
            next_input[:, 0, TBILL_CH] = tbill_fut[:, tau]      # future tbill scenario
            # Macro 6 channels remain 0 (future mask) by zero-init.

            seq = torch.cat([seq, next_input], dim=1)            # (BN, L_cur, 8)
            h_seq = self.encode(seq)                              # (BN, L_cur, d_model)
            h_tau = h_seq[:, -1, :]                               # (BN, d_model)
            if self.extra_context_dim > 0:
                ctx_tau = torch.cat([h_tau, extra_rep], dim=-1)   # (BN, flow_ctx)
            else:
                ctx_tau = h_tau
            # Flow.sample(num_samples=1, context=ctx) -> (ctx_batch, 1, 1).
            sp_z = self.flow.sample(1, context=ctx_tau).squeeze(-1).squeeze(-1)
            sampled.append(sp_z)
            last_sp = sp_z                                        # AR feedback

        sampled = torch.stack(sampled, dim=1)                     # (BN, FUTURE_LEN)
        sampled = sampled.view(B, n_sim, FUTURE_LEN)
        return sampled


# =====================================================================
# Metrics (kept self-contained -- match baseline_har_ar.py / sensitivity_flow_seq.py)
# =====================================================================

def crps_ensemble_sample(sim, y):
    n = len(sim); sim_sorted = np.sort(sim)
    term1 = np.mean(np.abs(sim - y))
    i = np.arange(n)
    term2 = (2.0 / (n * n)) * np.sum((2 * i + 1 - n) * sim_sorted)
    return float(term1 - 0.5 * term2)


def crps_pooled(sim_paths, actual_paths):
    n_origin, n_sim, T = sim_paths.shape
    vals = np.empty(n_origin * T, dtype=np.float64); k = 0
    for t in range(n_origin):
        for w in range(T):
            vals[k] = crps_ensemble_sample(sim_paths[t, :, w], actual_paths[t, w])
            k += 1
    return float(np.mean(vals)), float(np.std(vals))


def compute_emd_1d(samples_a, samples_b, n_bins=200):
    lo = float(min(samples_a.min(), samples_b.min()))
    hi = float(max(samples_a.max(), samples_b.max()))
    if hi - lo < 1e-12: return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    a_hist, _ = np.histogram(samples_a, bins=bins, density=False)
    b_hist, _ = np.histogram(samples_b, bins=bins, density=False)
    a_cum = np.cumsum(a_hist) / max(a_hist.sum(), 1)
    b_cum = np.cumsum(b_hist) / max(b_hist.sum(), 1)
    return float(np.mean(np.abs(a_cum - b_cum)) * (hi - lo))


def compute_var(returns_flat, alpha=0.05):
    return float(np.quantile(returns_flat, alpha))


def compute_cvar(returns_flat, alpha=0.05):
    sorted_r = np.sort(returns_flat)
    n_tail = max(1, int(alpha * len(sorted_r)))
    return float(sorted_r[:n_tail].mean())


# =====================================================================
# Plots
# =====================================================================

def plot_fanchart(actual_paths, sim_paths, title, save_path):
    import matplotlib.pyplot as plt
    h = sim_paths.shape[2]
    sim_cum    = np.cumsum(sim_paths, axis=2).reshape(-1, h)
    actual_cum = np.cumsum(actual_paths, axis=1)
    p05, p25, p50, p75, p95 = np.percentile(sim_cum, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, h + 1)
    ax.fill_between(x, p05, p95, alpha=0.18, color="C2", label="Mamba+Flow 90% CI")
    ax.fill_between(x, p25, p75, alpha=0.30, color="C2", label="Mamba+Flow 50% CI")
    ax.plot(x, p50, color="C2", lw=2, label="Mamba+Flow median")
    ax.plot(x, np.median(actual_cum, axis=0), color="red", lw=2, ls="--",
            label="Actual median")
    ax.set_xlabel("week (future)")
    ax.set_ylabel("cumulative sp_return")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    saved fanchart : {save_path}")


def plot_histogram(actual_flat, sim_flat, title, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    lo = float(min(actual_flat.min(), np.quantile(sim_flat, 0.001)))
    hi = float(max(actual_flat.max(), np.quantile(sim_flat, 0.999)))
    bins = np.linspace(lo, hi, 100)
    ax.hist(sim_flat,    bins=bins, density=True, alpha=0.4, color="C2",
            label="Mamba+Flow sim")
    ax.hist(actual_flat, bins=bins, density=True, alpha=0.5, color="red",
            label="Actual")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("sp_return (weekly)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    saved histogram: {save_path}")


# =====================================================================
# Train
# =====================================================================

def train(fold, train_csv, val_csv, save_path, log_path, summary_path,
          d_model=D_MODEL, n_mamba_layers=N_MAMBA_LAYERS,
          n_flow_layers=N_FLOW_LAYERS, n_flow_hidden=N_FLOW_HIDDEN,
          weight_decay=0.01, dropout=0.0,
          extra_cond_cols=None,
          max_epoch=MAX_EPOCH, patience=PATIENCE, batch=BATCH, lr=LR,
          device="cuda", seed=2026):
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"\n[1] Load train + val (teacher-forced sequences)")
    Xtr, Ytr, cond_stats, target_stats, n_tr = cached_load_windows_seq(train_csv)
    print(f"    train windows = {n_tr}, X.shape = {tuple(Xtr.shape)}, "
          f"Y.shape = {tuple(Ytr.shape)}")
    Xv, Yv, _, _, n_v = cached_load_windows_seq(
        val_csv, cond_stats=cond_stats, target_stats=target_stats
    )
    print(f"    val windows   = {n_v}")
    if n_v == 0:
        sys.exit(f"[FATAL] no val windows in {val_csv}")

    # Extra context (Flow-only, NOT Mamba input).  Origin-frozen z-score values.
    extra_stats = None
    Xtr_extra = None; Xv_extra = None
    extra_context_dim = 0
    if extra_cond_cols:
        Xtr_extra, extra_stats, n_extra_tr, _ = load_extra_context(
            train_csv, extra_cond_cols)
        Xv_extra, _, n_extra_v, _ = load_extra_context(
            val_csv, extra_cond_cols, extra_stats=extra_stats)
        if n_extra_tr != n_tr or n_extra_v != n_v:
            sys.exit(f"[FATAL] extra context valid count mismatch: "
                     f"train {n_extra_tr} vs {n_tr}, val {n_extra_v} vs {n_v}")
        extra_context_dim = len(extra_cond_cols)
        print(f"    extra context : {list(extra_cond_cols)} "
              f"(dim={extra_context_dim}, origin-frozen, Flow-only)")

    model = MambaFlowAR(
        d_model=d_model, n_mamba_layers=n_mamba_layers,
        n_flow_layers=n_flow_layers, n_flow_hidden=n_flow_hidden,
        dropout=dropout, extra_context_dim=extra_context_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    model params  = {n_params:,}  "
          f"(flow_context_dim={model.flow_context_dim})")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if Xtr_extra is not None:
        train_ds = TensorDataset(Xtr, Ytr, Xtr_extra)
    else:
        train_ds = TensorDataset(Xtr, Ytr)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)
    Xv_dev = Xv.to(device); Yv_dev = Yv.to(device)
    Xv_extra_dev = Xv_extra.to(device) if Xv_extra is not None else None

    best_val_nll = float("inf"); best_epoch = -1
    best_state = None
    log_rows = []
    pat = 0

    print(f"\n[2] Train  (max_epoch={max_epoch}, patience={patience}, "
          f"batch={batch}, lr={lr})")
    for ep in range(1, max_epoch + 1):
        model.train()
        losses = []
        for batch_tuple in train_dl:
            if Xtr_extra is not None:
                xb, yb, eb = batch_tuple
                xb = xb.to(device); yb = yb.to(device); eb = eb.to(device)
                loss = -model.log_prob(xb, yb, extra_context=eb).mean()
            else:
                xb, yb = batch_tuple
                xb = xb.to(device); yb = yb.to(device)
                loss = -model.log_prob(xb, yb).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())
        train_nll = float(np.mean(losses))

        model.eval()
        with torch.no_grad():
            val_nll = float(-model.log_prob(
                Xv_dev, Yv_dev, extra_context=Xv_extra_dev,
            ).mean().item())

        log_rows.append(dict(epoch=ep, train_nll=train_nll, val_nll=val_nll))
        improved = val_nll < best_val_nll
        if improved:
            best_val_nll = val_nll
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if ep == 1 or ep % 5 == 0 or improved or ep == max_epoch:
            mark = " *" if improved else ""
            print(f"    ep{ep:>3d}: train_nll = {train_nll:+.4f} "
                  f"(per-week {train_nll / FUTURE_LEN:+.3f})   "
                  f"val_nll = {val_nll:+.4f} "
                  f"(per-week {val_nll / FUTURE_LEN:+.3f})   "
                  f"(best {best_val_nll:+.4f} @ep{best_epoch}){mark}")
        if pat >= patience:
            print(f"    [early stop] ep{ep}  (patience {patience} from "
                  f"ep{best_epoch})")
            break

    # Save best checkpoint and training-time summary.
    torch.save({
        "model_state": best_state,
        "meta": dict(
            fold=fold, cond_cols=COND_COLS,
            past_len=PAST_LEN, future_len=FUTURE_LEN,
            d_model=d_model, n_mamba_layers=n_mamba_layers,
            mamba_d_state=MAMBA_D_STATE, mamba_d_conv=MAMBA_D_CONV,
            mamba_expand=MAMBA_EXPAND,
            n_flow_layers=n_flow_layers, n_flow_hidden=n_flow_hidden,
            weight_decay=weight_decay, dropout=dropout,
            n_flow_blocks=N_FLOW_BLOCKS, n_flow_bins=N_FLOW_BINS,
            flow_tail_bound=FLOW_TAIL_BOUND,
            cond_stats=cond_stats, target_stats=target_stats,
            extra_cond_cols=(list(extra_cond_cols) if extra_cond_cols else []),
            extra_context_dim=int(extra_context_dim),
            extra_stats=extra_stats,
            best_epoch=best_epoch, best_val_nll=best_val_nll,
            n_train=int(n_tr), n_val=int(n_v),
            n_params=int(n_params), seed=seed,
        ),
    }, save_path)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    summary = dict(
        fold=fold,
        model="Mamba AR + Conditional Flow head (End-to-End sp_return density)",
        cond_cols=list(COND_COLS),
        mask_future_macro_ch=list(MACRO_CH),
        past_len=int(PAST_LEN), future_len=int(FUTURE_LEN),
        d_model=int(d_model), n_mamba_layers=int(n_mamba_layers),
        mamba_d_state=int(MAMBA_D_STATE), mamba_d_conv=int(MAMBA_D_CONV),
        mamba_expand=int(MAMBA_EXPAND),
        n_flow_layers=int(n_flow_layers), n_flow_hidden=int(n_flow_hidden),
        weight_decay=float(weight_decay), dropout=float(dropout),
        n_flow_blocks=int(N_FLOW_BLOCKS), n_flow_bins=int(N_FLOW_BINS),
        flow_tail_bound=float(FLOW_TAIL_BOUND),
        best_epoch=int(best_epoch), best_val_nll=float(best_val_nll),
        best_val_nll_per_week=float(best_val_nll / FUTURE_LEN),
        n_train=int(n_tr), n_val=int(n_v), n_params=int(n_params),
        seed=int(seed),
        target_stats=target_stats, cond_stats=cond_stats,
        extra_cond_cols=(list(extra_cond_cols) if extra_cond_cols else []),
        extra_context_dim=int(extra_context_dim),
        extra_stats=extra_stats,
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  saved ckpt    : {save_path}")
    print(f"  saved log     : {log_path}")
    print(f"  saved summary : {summary_path}")
    print(f"\n  best val NLL = {best_val_nll:+.4f} (per-week "
          f"{best_val_nll / FUTURE_LEN:+.4f}) @ep{best_epoch}")
    return model, best_state, cond_stats, target_stats, extra_stats


# =====================================================================
# Evaluate on test (closed-form NLL + AR rollout scenario metrics)
# =====================================================================

@torch.no_grad()
def evaluate_test(model, best_state, test_csv, cond_stats, target_stats,
                  result_prefix, n_sim, seed, device, chunk_origins=8,
                  extra_cond_cols=None, extra_stats=None):
    model.load_state_dict(best_state)
    model.eval()

    print(f"\n[5] Load test + closed-form NLL (teacher-forced)")
    Xte, Yte, _, _, n_te = cached_load_windows_seq(
        test_csv, cond_stats=cond_stats, target_stats=target_stats
    )
    print(f"    test windows = {n_te}")
    if n_te == 0:
        sys.exit(f"[FATAL] no test windows in {test_csv}")

    Xte_extra_dev = None
    if extra_cond_cols:
        Xte_extra, _, n_extra_te, _ = load_extra_context(
            test_csv, extra_cond_cols, extra_stats=extra_stats)
        if n_extra_te != n_te:
            sys.exit(f"[FATAL] test extra valid {n_extra_te} != main {n_te}")
        Xte_extra_dev = Xte_extra.to(device)
        print(f"    extra context : {list(extra_cond_cols)} "
              f"(dim={len(extra_cond_cols)}, origin-frozen)")

    Xte_dev = Xte.to(device); Yte_dev = Yte.to(device)
    nll_test = -model.log_prob(Xte_dev, Yte_dev,
                                extra_context=Xte_extra_dev)              # (n_te,)
    per_origin_nll_z = float(nll_test.mean().item())
    per_week_nll_z   = float(per_origin_nll_z / FUTURE_LEN)
    print(f"    per-week NLL (z-score)              = {per_week_nll_z:+.4f}")
    print(f"    per-origin NLL (sum over 13 weeks)  = {per_origin_nll_z:+.4f}")

    # Recover last_past_sp_z for each valid origin (seed for AR rollout).
    valid_mask, z_te, df_te = compute_valid_mask(test_csv, cond_stats)
    n = z_te.shape[0]
    n_w = n - PAST_LEN - FUTURE_LEN + 1
    last_past_sp_full = z_te[PAST_LEN - 1 : PAST_LEN - 1 + n_w, SP_CH]
    last_past_sp_valid = last_past_sp_full[valid_mask].astype(np.float32)
    assert len(last_past_sp_valid) == Xte.shape[0], (
        f"valid mask mismatch: {len(last_past_sp_valid)} vs {Xte.shape[0]}"
    )

    print(f"\n[6] AR rollout sampling  (n_sim={n_sim}, chunk={chunk_origins})")
    torch.manual_seed(seed)
    last_sp_dev = torch.from_numpy(last_past_sp_valid).to(device)
    n_origins = Xte.shape[0]

    sim_paths_z_chunks = []
    for s in range(0, n_origins, chunk_origins):
        e = min(n_origins, s + chunk_origins)
        x_past_chunk = Xte_dev[s:e, :PAST_LEN, :]                  # (k, PAST_LEN, 8)
        future_tbill_chunk = Xte_dev[s:e, PAST_LEN:, TBILL_CH]     # (k, FUTURE_LEN)
        last_sp_chunk = last_sp_dev[s:e]
        extra_chunk = Xte_extra_dev[s:e] if Xte_extra_dev is not None else None
        sim_chunk = model.ar_sample(
            x_past_chunk, future_tbill_chunk, last_sp_chunk, n_sim,
            extra_context=extra_chunk,
        )                                                           # (k, n_sim, T)
        sim_paths_z_chunks.append(sim_chunk.cpu())
    sim_paths_z = torch.cat(sim_paths_z_chunks, dim=0).numpy()      # (n_v, n_sim, T)

    actual_z      = Yte.numpy()                                     # (n_v, T)
    tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
    sim_paths_raw = sim_paths_z * tsd + tmu
    actual_raw    = actual_z   * tsd + tmu

    # Scenario metrics (raw units, matched to baseline_har_ar.py).
    print(f"\n[7] Scenario metrics (raw sp_return units)")
    actual_flat = actual_raw.ravel()
    sim_flat    = sim_paths_raw.ravel()
    var5_a = compute_var(actual_flat, 0.05); var5_s = compute_var(sim_flat, 0.05)
    var1_a = compute_var(actual_flat, 0.01); var1_s = compute_var(sim_flat, 0.01)
    cv5_a  = compute_cvar(actual_flat, 0.05); cv5_s  = compute_cvar(sim_flat, 0.05)
    cv1_a  = compute_cvar(actual_flat, 0.01); cv1_s  = compute_cvar(sim_flat, 0.01)
    emd    = compute_emd_1d(actual_flat, sim_flat, n_bins=200)
    crps_m, crps_s = crps_pooled(sim_paths_raw, actual_raw)
    std_a = float(actual_raw.std(ddof=1))
    std_s = float(sim_paths_raw.std(ddof=1))
    std_ratio = std_s / std_a if std_a > 1e-12 else float("nan")
    lo95, hi95 = np.percentile(sim_flat, [2.5, 97.5])
    cov95 = float(((actual_flat >= lo95) & (actual_flat <= hi95)).mean())
    lo80, hi80 = np.percentile(sim_flat, [10.0, 90.0])
    cov80 = float(((actual_flat >= lo80) & (actual_flat <= hi80)).mean())
    lo50, hi50 = np.percentile(sim_flat, [25.0, 75.0])
    cov50 = float(((actual_flat >= lo50) & (actual_flat <= hi50)).mean())

    print(f"    CRPS pooled       = {crps_m:.5f}  (std {crps_s:.5f})")
    print(f"    EMD               = {emd:.6f}")
    print(f"    std act/sim/ratio = {std_a:.5f} / {std_s:.5f} / {std_ratio:.3f}")
    print(f"    VaR1   act/sim/D  = {var1_a:+.5f} / {var1_s:+.5f} / "
          f"{var1_s - var1_a:+.5f}")
    print(f"    CVaR1  act/sim/D  = {cv1_a:+.5f} / {cv1_s:+.5f} / "
          f"{cv1_s - cv1_a:+.5f}")
    print(f"    cov 50 / 80 / 95  = {cov50:.3f} / {cov80:.3f} / {cov95:.3f}")

    # Plots
    print(f"\n[8] Plots")
    fold = os.path.basename(result_prefix).replace("mamba_flow_ar_", "")
    title = (f"Mamba AR + Conditional Flow head -- fold {fold}\n"
             f"(n_origin = {n_origins}, n_sim = {n_sim})  "
             f"End-to-End sp_return density")
    plot_fanchart(actual_raw, sim_paths_raw, title + "\nfan chart",
                  f"{result_prefix}_fanchart.png")
    plot_histogram(actual_flat, sim_flat,
                   title + "\nweekly sp_return histogram",
                   f"{result_prefix}_histogram.png")

    # Predictions CSV
    print(f"\n[9] Save predictions CSV")
    date_col = df_te["date"].values if "date" in df_te.columns else None
    orig_idx_array = np.where(valid_mask)[0]
    rows = []
    sim_mean_z = sim_paths_z.mean(axis=1)                                # (n_v, T)
    sim_std_z  = sim_paths_z.std(axis=1, ddof=1)
    for ii in range(n_origins):
        t = int(orig_idx_array[ii])
        origin_date = (str(date_col[t + PAST_LEN - 1])
                       if date_col is not None else f"origin_{t}")
        for tau in range(FUTURE_LEN):
            rows.append(dict(
                origin_idx   = t,
                origin_date  = origin_date,
                step         = int(tau),
                actual_z     = float(actual_z[ii, tau]),
                actual_raw   = float(actual_raw[ii, tau]),
                sim_mean_z   = float(sim_mean_z[ii, tau]),
                sim_std_z    = float(sim_std_z[ii, tau]),
                sim_mean_raw = float(sim_paths_raw[ii, :, tau].mean()),
            ))
    pred_path = f"{result_prefix}_predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    print(f"    saved predictions: {pred_path}")

    eval_metrics = dict(
        n_test_origins   = int(n_origins),
        n_sim_per_origin = int(n_sim),
        per_week_nll_z   = per_week_nll_z,
        per_origin_nll_z = per_origin_nll_z,
        crps_pooled      = crps_m,
        crps_std         = crps_s,
        emd              = emd,
        std_actual       = std_a,
        std_sim          = std_s,
        std_ratio        = std_ratio,
        var_5pct_actual  = var5_a, var_5pct_sim = var5_s,
        var_5pct_diff    = var5_s - var5_a,
        var_1pct_actual  = var1_a, var_1pct_sim = var1_s,
        var_1pct_diff    = var1_s - var1_a,
        cvar_5pct_actual = cv5_a,  cvar_5pct_sim = cv5_s,
        cvar_5pct_diff   = cv5_s - cv5_a,
        cvar_1pct_actual = cv1_a,  cvar_1pct_sim = cv1_s,
        cvar_1pct_diff   = cv1_s - cv1_a,
        coverage_50      = cov50,
        coverage_80      = cov80,
        coverage_95      = cov95,
    )
    return eval_metrics


# =====================================================================
# Main
# =====================================================================

def main_worker(args):
    """In-process entry point for one (fold, spec) train + evaluate.

    Accepts argparse.Namespace OR a dict of overrides.  Reuses the module-level
    _DATA_CACHE so repeated calls on the same fold avoid re-reading csv files.
    At the end frees CUDA memory (Gemini critique: VRAM leak prevention for
    long sweeps).

    Returns the test eval metrics dict.
    """
    import gc

    if isinstance(args, dict):
        defaults = dict(
            folds_dir=os.path.join(ROOT, "data", "folds_v33_vix_expanding"),
            out_dir=os.path.join(HERE, "result"),
            max_epoch=MAX_EPOCH, patience=PATIENCE,
            batch=BATCH, lr=LR,
            n_sim=1000, chunk_origins=8, tag="",
            d_model=D_MODEL, n_mamba_layers=N_MAMBA_LAYERS,
            n_flow_layers=N_FLOW_LAYERS, n_flow_hidden=N_FLOW_HIDDEN,
            weight_decay=0.01, dropout=0.0, seed=2026,
            extra_context_channels="",
        )
        merged = {**defaults, **args}
        if "fold" not in merged:
            raise ValueError("args dict must include 'fold'")
        args = argparse.Namespace(**merged)

    train_csv = os.path.join(args.folds_dir, f"{args.fold}_train.csv")
    val_csv   = os.path.join(args.folds_dir, f"{args.fold}_val.csv")
    test_csv  = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    for p in (train_csv, val_csv, test_csv):
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing csv: {p}")
    os.makedirs(args.out_dir, exist_ok=True)
    prefix_base = (f"mamba_flow_ar_{args.tag}_{args.fold}" if args.tag
                   else f"mamba_flow_ar_{args.fold}")
    result_prefix = os.path.join(args.out_dir, prefix_base)
    save_path    = f"{result_prefix}_best.pt"
    log_path     = f"{result_prefix}_log.csv"
    summary_path = f"{result_prefix}_summary.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print(f" Mamba AR + Conditional Flow head  -  fold = {args.fold}")
    print(f"  Mamba-SSM      : d_model = {args.d_model}, "
          f"n_layers = {args.n_mamba_layers}, "
          f"d_state = {MAMBA_D_STATE}, d_conv = {MAMBA_D_CONV}, "
          f"expand = {MAMBA_EXPAND}")
    print(f"  Flow head      : 1D Cond NSF features = 1, "
          f"layers = {args.n_flow_layers}, hidden = {args.n_flow_hidden}, "
          f"blocks = {N_FLOW_BLOCKS}, bins = {N_FLOW_BINS}")
    print(f"  train          : AdamW lr = {args.lr}, batch = {args.batch}, "
          f"weight_decay = {args.weight_decay}, dropout = {args.dropout}, "
          f"max_epoch = {args.max_epoch}, patience = {args.patience}")
    print(f"  output prefix  : {prefix_base}")
    print(f"  device         : {device}")
    print("=" * 78)

    extra_cond_cols = (
        [c.strip() for c in args.extra_context_channels.split(",") if c.strip()]
        if getattr(args, "extra_context_channels", "") else None
    )

    model, best_state, cond_stats, target_stats, extra_stats = train(
        args.fold, train_csv, val_csv, save_path, log_path, summary_path,
        d_model=args.d_model, n_mamba_layers=args.n_mamba_layers,
        n_flow_layers=args.n_flow_layers, n_flow_hidden=args.n_flow_hidden,
        weight_decay=args.weight_decay, dropout=args.dropout,
        extra_cond_cols=extra_cond_cols,
        max_epoch=args.max_epoch, patience=args.patience,
        batch=args.batch, lr=args.lr, device=device, seed=args.seed,
    )
    eval_metrics = evaluate_test(
        model, best_state, test_csv, cond_stats, target_stats,
        result_prefix, args.n_sim, args.seed, device,
        chunk_origins=args.chunk_origins,
        extra_cond_cols=extra_cond_cols, extra_stats=extra_stats,
    )

    with open(summary_path, "r") as f:
        existing = json.load(f)
    existing.update(dict(test_eval=eval_metrics))
    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\n  updated summary with test eval : {summary_path}")

    # VRAM cleanup (Gemini: VRAM leak prevention in long sweeps)
    del model
    if best_state is not None:
        del best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[done]\n")
    return eval_metrics


def main():
    ap = argparse.ArgumentParser(
        description="Mamba AR + Conditional Flow head -- End-to-End sp_return "
                    "distribution model (no mean/residual split)."
    )
    ap.add_argument("--fold", required=True,
                    help="fold id (e.g. F_long, F_long_A, F_long_B)")
    ap.add_argument("--folds-dir",
                    default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--max-epoch", type=int, default=MAX_EPOCH)
    ap.add_argument("--patience",  type=int, default=PATIENCE)
    ap.add_argument("--batch",     type=int, default=BATCH)
    ap.add_argument("--lr",        type=float, default=LR)
    ap.add_argument("--n-sim",     type=int, default=1000)
    ap.add_argument("--chunk-origins", type=int, default=8,
                    help="AR rollout origin chunk size (GPU memory control)")
    ap.add_argument("--tag", default="",
                    help="output prefix suffix (e.g. 'small') -- preserves "
                         "previous results when sweeping hyperparams")
    ap.add_argument("--d-model",         type=int, default=D_MODEL)
    ap.add_argument("--n-mamba-layers",  type=int, default=N_MAMBA_LAYERS)
    ap.add_argument("--n-flow-layers",   type=int, default=N_FLOW_LAYERS)
    ap.add_argument("--n-flow-hidden",   type=int, default=N_FLOW_HIDDEN)
    ap.add_argument("--extra-context-channels", default="",
                    help="comma-separated extra channels passed ONLY to Flow "
                         "context (concat to h_tau), NOT into Mamba input. "
                         "e.g. 'bondpp_13w_lag,stockpp_13w_lag,metab_13w'. "
                         "Origin-frozen z-score (운영 시 미래 derived 모름 가정).")
    ap.add_argument("--weight-decay",    type=float, default=0.01,
                    help="AdamW weight_decay (L2 regularization)")
    ap.add_argument("--dropout",         type=float, default=0.0,
                    help="dropout for Mamba residual + Flow head MLPs")
    ap.add_argument("--seed",      type=int, default=2026)
    args = ap.parse_args()
    main_worker(args)


if __name__ == "__main__":
    main()
