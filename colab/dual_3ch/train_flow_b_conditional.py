"""Conditional 1D Neural Spline Flow — Model B with macro context.

기존 generate_3m_scenario.py 의 unconditional 1D NSF 를 macro-conditional 로 확장.

설계:
  - ε_{t..t+12} = r_{t..t+12} / std(r_{t..t+12})    (standardized residual, 기존과 동일 정의)
  - context = macro at origin t  (cond_cols 의 마지막 past row 값, Model A v13 의 stats_cond 로 정규화)
       = 5D: [tbill_wr, m2_13w_cum_lag, gdp_13w_proxy_lag, cpi_13w_cum_lag, excess_liq_yoy_lag]
  - 같은 origin 의 13 개 ε 는 동일 context 공유 (per-origin conditioning)

Architecture:
  - MaskedPiecewiseRationalQuadraticAutoregressiveTransform with context_features=5
  - 4 layers, num_bins=8, tail_bound=5.0, hidden_features=32

Training:
  - 학습 데이터: data/pilot_split/train.csv 의 모든 origin 추출
  - Context 정규화: Model A v13 ckpt 의 stats_cond 재사용 (inference 일관성)
  - Loss: NLL of flow.log_prob(ε, context)
  - 200 epochs, batch 256

Output:
  result/scenario_3m_flow_1d_cond.pt   (conditional flow state_dict + meta)

Usage:
  python colab/dual_3ch/train_flow_b_conditional.py --pilot-split
"""
import torch  # MUST be first

import argparse
import glob
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_vol_pilot_3m import PAST_LEN, FUTURE_LEN, L

try:
    from nflows.flows.base import Flow
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms import (
        CompositeTransform,
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    )
except ImportError:
    print("FATAL: nflows required.  pip install nflows")
    sys.exit(1)


# =====================================================================
# Build conditional flow
# =====================================================================

def build_conditional_flow_1d(context_features=5, num_layers=4, num_bins=8,
                              tail_bound=5.0, hidden_features=32):
    """1D Conditional Neural Spline Flow.

    For 1D 'autoregressive' is trivial — the transform is a context-conditioned
    spline whose parameters are produced by an MLP from context.
    """
    base = StandardNormal(shape=[1])
    transforms = []
    for _ in range(num_layers):
        transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=1,
            hidden_features=hidden_features,
            context_features=context_features,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
        ))
    return Flow(CompositeTransform(transforms), base)


# =====================================================================
# Extract (ε, context) training pairs
# =====================================================================

def extract_eps_with_context(train_csv, cols_cond, stats_cond,
                              past_len=PAST_LEN, future_len=FUTURE_LEN):
    """Build (ε, context) training pairs from train CSV.

    For each origin t in train (window [t : t+L]):
      ε_{0..future_len-1} = r_{t+past_len .. t+L-1} / std(r_{t+past_len .. t+L-1})
      context              = cond_cols values at row (t + past_len - 1)
                             (마지막 past row — Model A 도 같은 시점 macro 를 본다고 가정)
      → 13 (ε, context) pairs per origin, all with same context.

    Context is then z-score normalized using Model A's stats_cond (train stats).

    Returns:
      eps:  torch.FloatTensor (N, 1)         — N = n_origins × future_len
      ctx:  torch.FloatTensor (N, d_cond)    — normalized
      meta: dict
    """
    df = pd.read_csv(train_csv)
    sp = df["sp_return"].values.astype(np.float64)
    n = len(df)
    n_w = n - (past_len + future_len) + 1
    if n_w <= 0:
        raise RuntimeError(f"Not enough train data: n={n}")

    # Verify required cond columns exist
    missing = [c for c in cols_cond if c not in df.columns]
    if missing:
        raise KeyError(f"missing cond cols in train CSV: {missing}")

    cond_arr = df[cols_cond].values.astype(np.float32)   # (n, d_cond)

    eps_list = []
    ctx_list = []
    n_kept = 0
    for t in range(n_w):
        future = sp[t + past_len : t + past_len + future_len]   # 13 step
        std = float(future.std(ddof=1))
        if std < 1e-8:
            continue
        eps = (future / std).astype(np.float32)             # (13,)
        ctx_row = cond_arr[t + past_len - 1]                # (d_cond,)
        eps_list.append(eps)
        ctx_list.append(np.tile(ctx_row, (future_len, 1)))  # (13, d_cond)
        n_kept += 1

    eps_flat = np.concatenate(eps_list, axis=0)             # (n_kept * 13,)
    ctx_flat = np.concatenate(ctx_list, axis=0)             # (n_kept * 13, d_cond)

    # Normalize context with Model A's train stats (consistent w/ inference time)
    cmu = np.asarray(stats_cond["mean"], dtype=np.float32)
    csd = np.asarray(stats_cond["std"],  dtype=np.float32)
    ctx_norm = (ctx_flat - cmu) / csd

    print(f"  ε samples extracted: {len(eps_flat):,}  "
          f"(from {n_kept} train origins × {future_len} steps)")
    print(f"    eps mean={eps_flat.mean():+.4f}, std={eps_flat.std():.4f}, "
          f"min/max={eps_flat.min():+.3f}/{eps_flat.max():+.3f}")
    print(f"  context dim={ctx_norm.shape[1]} (= len(cond_cols)), normalized via Model A stats")
    print(f"    cond_cols: {cols_cond}")

    eps_t = torch.from_numpy(eps_flat.astype(np.float32)).unsqueeze(-1)   # (N, 1)
    ctx_t = torch.from_numpy(ctx_norm.astype(np.float32))                 # (N, d_cond)
    return eps_t, ctx_t, dict(n_origins=n_kept, future_len=future_len,
                              cond_cols=list(cols_cond))


# =====================================================================
# Load v13 Model A cond_cols + stats_cond (reuse for context spec)
# =====================================================================

def load_v13_meta(result_dir, fold_tag="pilot"):
    """Load any one v13 ckpt to get cond_cols + stats_cond for context build."""
    pattern = os.path.join(result_dir, f"vol_pilot_3m_psel_v13_*_{fold_tag}_seed*_best.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No v13 ckpt at {pattern} — 먼저 v13 학습 필요")
    ckpt = torch.load(paths[0], map_location="cpu", weights_only=False)
    return dict(cond_cols=ckpt["cond_cols"],
                stats_cond=ckpt["stats_cond"],
                ref_ckpt=paths[0])


# =====================================================================
# Train conditional flow
# =====================================================================

def train_conditional_flow(eps_train, ctx_train, save_path,
                            num_layers=4, num_bins=8, tail_bound=5.0,
                            hidden_features=32, epochs=200, batch=256,
                            lr=5e-4, device="cuda", log_every=20):
    """Train conditional 1D NSF on (ε, context) pairs.

    If save_path already exists, load and return.
    """
    context_features = ctx_train.shape[1]

    if os.path.exists(save_path):
        print(f"  [SKIP] conditional flow ckpt 이미 존재: {save_path}")
        flow = build_conditional_flow_1d(
            context_features=context_features,
            num_layers=num_layers, num_bins=num_bins, tail_bound=tail_bound,
            hidden_features=hidden_features,
        ).to(device)
        state = torch.load(save_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            flow.load_state_dict(state["model_state"])
        else:
            flow.load_state_dict(state)
        return flow

    flow = build_conditional_flow_1d(
        context_features=context_features,
        num_layers=num_layers, num_bins=num_bins, tail_bound=tail_bound,
        hidden_features=hidden_features,
    ).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"  Conditional Flow 1D NSF: layers={num_layers}, bins={num_bins}, "
          f"tail={tail_bound}, hidden={hidden_features}, ctx_features={context_features}, "
          f"params={n_params:,}")

    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    eps_train = eps_train.to(device)
    ctx_train = ctx_train.to(device)
    N = eps_train.shape[0]
    print(f"  N (ε, context) pairs = {N:,},  batch={batch},  epochs={epochs},  lr={lr}")

    losses_history = []
    best_nll = float("inf")
    for ep in range(1, epochs + 1):
        flow.train()
        perm = torch.randperm(N)
        losses = []
        for i in range(0, N, batch):
            idx = perm[i : i + batch]
            xb = eps_train[idx]                  # (b, 1)
            cb = ctx_train[idx]                  # (b, d_cond)
            loss = -flow.log_prob(inputs=xb, context=cb).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
            opt.step()
            losses.append(loss.item())
        ep_loss = float(np.mean(losses))
        losses_history.append(ep_loss)
        if ep_loss < best_nll:
            best_nll = ep_loss
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            print(f"    flow ep{ep:>3d}: nll = {ep_loss:+.4f}   (best so far {best_nll:+.4f})")

    # Save
    torch.save({
        "model_state":    flow.state_dict(),
        "model_type":     "conditional_1d_nsf",
        "context_features": context_features,
        "num_layers":     num_layers,
        "num_bins":       num_bins,
        "tail_bound":     tail_bound,
        "hidden_features": hidden_features,
        "final_train_nll": ep_loss,
        "best_train_nll":  best_nll,
    }, save_path)
    print(f"  saved: {save_path}   (final train NLL = {ep_loss:+.4f})")
    return flow


# =====================================================================
# Sanity check — sample 100 ε per regime, compare moments
# =====================================================================

def sanity_check_conditional_shape(flow, ctx_train, eps_train, device, n_sample=1000):
    """Pick 2 contexts (one with smallest first-feature value, one largest)
    and sample n_sample ε each. Compare moments to verify conditioning works.
    """
    flow.eval()
    # rank by first feature (tbill_wr) just as a regime proxy
    first_col = ctx_train[:, 0].cpu().numpy()
    idx_low  = int(np.argmin(first_col))
    idx_high = int(np.argmax(first_col))

    ctx_low  = ctx_train[idx_low : idx_low + 1].to(device)    # (1, d)
    ctx_high = ctx_train[idx_high : idx_high + 1].to(device)  # (1, d)

    with torch.no_grad():
        eps_low  = flow.sample(n_sample, context=ctx_low).cpu().numpy().reshape(-1)
        eps_high = flow.sample(n_sample, context=ctx_high).cpu().numpy().reshape(-1)
        # Also unconditional-style baseline: random context from training pool (mean)
        ctx_mean = ctx_train.mean(dim=0, keepdim=True).to(device)
        eps_mean = flow.sample(n_sample, context=ctx_mean).cpu().numpy().reshape(-1)

    def mom(x):
        x = np.asarray(x, dtype=np.float64)
        m, s = float(x.mean()), float(x.std(ddof=1))
        if s < 1e-12:
            return m, s, float("nan"), float("nan")
        sk = float(((x - m) ** 3).mean() / (s ** 3))
        kt = float(((x - m) ** 4).mean() / (s ** 4) - 3.0)
        return m, s, sk, kt

    print("  Sanity check — conditional sampling moments (n={}):".format(n_sample))
    print(f"    {'context':16s} {'mean':>+10s} {'std':>8s} {'skew':>+8s} {'ex_kurt':>+8s}")
    for label, eps in [("first_col MIN", eps_low),
                       ("ctx mean   ", eps_mean),
                       ("first_col MAX", eps_high)]:
        m, s, sk, kt = mom(eps)
        print(f"    {label:16s} {m:+10.4f} {s:8.4f} {sk:+8.3f} {kt:+8.3f}")
    # Also the empirical train ε for reference
    m, s, sk, kt = mom(eps_train.cpu().numpy().reshape(-1))
    print(f"    {'train ε emp.':16s} {m:+10.4f} {s:8.4f} {sk:+8.3f} {kt:+8.3f}")


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Train Conditional 1D NSF — Model B with macro context")
    ap.add_argument("--pilot-split", action="store_true", default=True)
    ap.add_argument("--train-csv",   default=None,
                    help="override train CSV path (default: data/pilot_split/train.csv)")
    ap.add_argument("--result-dir",  default=os.path.join(HERE, "result"))
    ap.add_argument("--save-name",   default="scenario_3m_flow_1d_cond.pt")
    ap.add_argument("--num-layers",  type=int,   default=4)
    ap.add_argument("--num-bins",    type=int,   default=8)
    ap.add_argument("--tail-bound",  type=float, default=5.0)
    ap.add_argument("--hidden",      type=int,   default=32)
    ap.add_argument("--epochs",      type=int,   default=200)
    ap.add_argument("--batch",       type=int,   default=256)
    ap.add_argument("--lr",          type=float, default=5e-4)
    ap.add_argument("--seed",        type=int,   default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
    train_csv = args.train_csv or os.path.join(repo_root, "data", "pilot_split", "train.csv")
    if not os.path.exists(train_csv):
        sys.exit(f"[FATAL] missing train CSV: {train_csv}")
    os.makedirs(args.result_dir, exist_ok=True)
    save_path = os.path.join(args.result_dir, args.save_name)

    print("=" * 78)
    print(" Train Conditional 1D NSF — Model B with macro context")
    print("=" * 78)
    print(f"  device     : {device}")
    print(f"  train_csv  : {train_csv}")
    print(f"  save_path  : {save_path}")

    # 1. Load Model A v13 metadata (cond_cols + stats_cond)
    print("\n[1] Load Model A v13 metadata (cond_cols + stats_cond reused as context spec)")
    meta = load_v13_meta(args.result_dir, fold_tag="pilot")
    print(f"    ref ckpt   : {meta['ref_ckpt']}")
    print(f"    cond_cols  : {meta['cond_cols']}")

    # 2. Extract (ε, context) pairs
    print("\n[2] Extract (ε, context) training pairs from train CSV")
    eps_train, ctx_train, ext_meta = extract_eps_with_context(
        train_csv,
        cols_cond=meta["cond_cols"],
        stats_cond=meta["stats_cond"],
    )

    # 3. Train conditional flow
    print("\n[3] Train conditional 1D NSF")
    flow = train_conditional_flow(
        eps_train, ctx_train, save_path,
        num_layers=args.num_layers, num_bins=args.num_bins,
        tail_bound=args.tail_bound, hidden_features=args.hidden,
        epochs=args.epochs, batch=args.batch, lr=args.lr, device=device,
    )

    # 4. Sanity check — moments at min/max of first context feature
    print("\n[4] Sanity check — sample moments at extreme contexts")
    sanity_check_conditional_shape(flow, ctx_train, eps_train, device, n_sample=1000)

    # 5. Write meta json
    meta_json_path = save_path.replace(".pt", "_meta.json")
    meta_json = dict(
        ref_v13_ckpt=meta["ref_ckpt"],
        cond_cols=meta["cond_cols"],
        n_train_origins=ext_meta["n_origins"],
        future_len=ext_meta["future_len"],
        num_layers=args.num_layers, num_bins=args.num_bins,
        tail_bound=args.tail_bound, hidden_features=args.hidden,
        epochs=args.epochs, batch=args.batch, lr=args.lr,
        stats_cond=meta["stats_cond"],  # echo so sensitivity_v13 can pick up if needed
    )
    with open(meta_json_path, "w") as f:
        json.dump(meta_json, f, indent=2)
    print(f"\n  meta saved: {meta_json_path}")

    print("\nDone — use this ckpt via:\n"
          f"  python colab/dual_3ch/sensitivity_v13.py --flow-mode cond")


if __name__ == "__main__":
    main()
