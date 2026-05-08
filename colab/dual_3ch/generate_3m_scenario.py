"""3M scenario generation — Two-Model framework.

Model A: vol_pilot_3m_psel_v13 (base + excess_liq) → σ_pred per origin (13w vol scalar)
Model B: 1D Neural Spline Flow on standardized residual ε (fat-tail innovation)
Combine: r_path = σ_pred × ε_path  (13 step scenario, N=100 sim per origin)

Steps:
  1. ε training data 추출 — 각 train origin t 마다 (r_{t}, ..., r_{t+12}) / std(window)
  2. 1D Neural Spline Flow 학습 (nflows, RQS spline, ε 의 fat-tail 모델링)
  3. variant 13 ckpt 5개 (seed 42-46) load → σ_pred per (origin, seed)
  4. Scenario generation: σ_pred × ε_sample → r_path
  5. Evaluation: 13w cumulative return EMD, CVaR_5%, 80%/95% interval calibration

Output:
  result/scenario_3m.npz   (scenarios, actual, metadata)
  result/scenario_3m_metrics.csv

Requirements:
  pip install nflows
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
import torch.nn as nn

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# variant 13 model class (재사용)
from train_vol_pilot_3m import (
    CausalTransformerVolScalar, build_spec, load_windows, mask_future_channels,
    PAST_LEN, FUTURE_LEN, L, D_MODEL, N_HEADS, N_LAYERS,
)

# nflows
try:
    from nflows.flows.base import Flow
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms import CompositeTransform, PiecewiseRationalQuadraticCDF
except ImportError:
    print("FATAL: nflows required. pip install nflows")
    sys.exit(1)

try:
    from scipy.stats import wasserstein_distance
except ImportError:
    print("FATAL: scipy required.")
    sys.exit(1)


# =====================================================================
# 1. ε training data 추출
# =====================================================================

def extract_eps_train(train_csv, future_len=FUTURE_LEN):
    """Train csv 의 각 origin t 마다 r_{t..t+future_len-1} / std(window) → ε samples.

    Returns:
        eps_flat: (n_train_origins × future_len,) — 1D tensor of standardized residuals
        meta: dict {n_origins, future_len}
    """
    df = pd.read_csv(train_csv)
    sp = df["sp_return"].values.astype(np.float32)
    n = len(sp)
    n_origins = n - future_len + 1
    if n_origins <= 0:
        raise RuntimeError(f"Not enough train data: {n} < {future_len}")
    eps_list = []
    for t in range(n_origins):
        window = sp[t : t + future_len]
        std = window.std(ddof=1)
        if std < 1e-8:
            continue
        eps = window / std  # (future_len,)
        eps_list.append(eps)
    eps_flat = np.concatenate(eps_list).astype(np.float32)  # (~n_origins * 13,)
    print(f"  ε samples extracted: {len(eps_flat):,}  "
          f"(from {n_origins} train origins × {future_len} steps)")
    print(f"    mean = {eps_flat.mean():+.4f}, std = {eps_flat.std():.4f}, "
          f"min/max = {eps_flat.min():+.3f}/{eps_flat.max():+.3f}")
    return torch.from_numpy(eps_flat), dict(n_origins=n_origins, future_len=future_len)


# =====================================================================
# 2. 1D Neural Spline Flow (nflows RQS)
# =====================================================================

def build_flow_1d(num_layers=4, num_bins=8, tail_bound=5.0):
    """1D Neural Spline Flow — base = StandardNormal, transform = stacked RQS spline."""
    base = StandardNormal(shape=[1])
    transforms = []
    for _ in range(num_layers):
        transforms.append(PiecewiseRationalQuadraticCDF(
            shape=[1], num_bins=num_bins, tail_bound=tail_bound, tails="linear",
        ))
    flow = Flow(CompositeTransform(transforms), base)
    return flow


def train_flow(eps_train, save_path, num_layers=4, num_bins=8, tail_bound=5.0,
               epochs=200, batch=256, lr=5e-4, device="cuda"):
    """Train 1D NSF on ε."""
    if os.path.exists(save_path):
        print(f"  [SKIP] flow ckpt 이미 존재: {save_path}")
        flow = build_flow_1d(num_layers, num_bins, tail_bound).to(device)
        flow.load_state_dict(torch.load(save_path, map_location=device))
        return flow

    flow = build_flow_1d(num_layers, num_bins, tail_bound).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"  Flow 1D NSF: layers={num_layers}, bins={num_bins}, tail={tail_bound}, "
          f"params={n_params:,}")

    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    eps_train = eps_train.unsqueeze(-1).to(device)  # (N, 1)
    n = eps_train.shape[0]

    for ep in range(1, epochs + 1):
        flow.train()
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, batch):
            xb = eps_train[perm[i : i + batch]]
            loss = -flow.log_prob(inputs=xb).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
            opt.step()
            losses.append(loss.item())
        if ep % 20 == 0 or ep == 1:
            print(f"    flow ep{ep:>3d}: nll = {np.mean(losses):+.4f}")

    torch.save(flow.state_dict(), save_path)
    print(f"  flow saved: {save_path}")
    return flow


# =====================================================================
# 3. Variant 13 Model A — σ_pred per (origin, seed)
# =====================================================================

def load_variant13_models(result_dir, fold_tag="pilot", device="cuda"):
    """vol_pilot_3m_psel_v13_*_seed{42..46}_best.pt 5개 로드."""
    pattern = os.path.join(result_dir, f"vol_pilot_3m_psel_v13_*_{fold_tag}_seed*_best.pt")
    paths = sorted(glob.glob(pattern))
    print(f"  variant 13 ckpt: {len(paths)} found  (expected 5)")
    if not paths:
        raise FileNotFoundError(f"No ckpt at {pattern}")

    models = []
    seeds = []
    for p in paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        cols_cond = ckpt["cond_cols"]
        cfg = ckpt["config"]
        model = CausalTransformerVolScalar(
            d_cond=cfg["d_cond"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            past_len=cfg["past_len"], future_len=cfg["future_len"],
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        seed = int(p.split("_seed")[1].split("_")[0])
        models.append(dict(model=model, cond_cols=cols_cond,
                           stats_cond=ckpt["stats_cond"],
                           mask_future_ch=ckpt["mask_future_ch"], seed=seed))
        seeds.append(seed)
    print(f"    seeds: {seeds}")
    return models


def predict_sigma(models, test_csv, device="cuda"):
    """Variant 13 5개 model 각각 test origin 별 σ_pred (log_std → exp).

    Returns:
        sigmas: (n_test_origins, n_seeds=5)  predicted σ
        actual_cum: (n_test_origins,)        actual 13w cumulative return
        actual_std: (n_test_origins,)        actual 13w realized std
    """
    n_seeds = len(models)
    sigmas_per_seed = []
    actual_cum = None
    actual_std = None
    for m in models:
        cols_cond = m["cond_cols"]
        stats_c   = m["stats_cond"]
        mask_fut  = m["mask_future_ch"]

        Xte_past, Cte, yte, _ = load_windows(
            test_csv, cols_cond, ["sp_return"], L=L, cond_stats=stats_c)
        if Xte_past is None:
            raise RuntimeError(f"test windows fail: {test_csv}")
        Cte = mask_future_channels(Cte, PAST_LEN, mask_fut)

        with torch.no_grad():
            log_std_pred = m["model"](Cte.to(device), Xte_past.to(device)).cpu().numpy()
        sigmas_per_seed.append(np.exp(log_std_pred))

        if actual_cum is None:
            # actual 13w cumulative + std (origin 별)
            df = pd.read_csv(test_csv)
            sp = df["sp_return"].values.astype(np.float32)
            n = len(sp); n_w = n - L + 1
            cum_list = []
            std_list = []
            for i in range(n_w):
                future = sp[i + PAST_LEN : i + L]   # 13 step
                cum_list.append(future.sum())
                std_list.append(future.std(ddof=1))
            actual_cum = np.array(cum_list, dtype=np.float32)
            actual_std = np.array(std_list, dtype=np.float32)

    sigmas = np.stack(sigmas_per_seed, axis=1)  # (n_origins, 5)
    print(f"  σ_pred shape: {sigmas.shape}")
    print(f"    σ_pred range: [{sigmas.min():.4f}, {sigmas.max():.4f}]")
    print(f"    actual_std range: [{actual_std.min():.4f}, {actual_std.max():.4f}]")
    return sigmas, actual_cum, actual_std


# =====================================================================
# 4. Scenario generation
# =====================================================================

def generate_scenarios(sigmas, flow, n_sim=100, future_len=FUTURE_LEN, device="cuda"):
    """r_path = σ_pred × ε_sample.

    Args:
        sigmas: (n_origins, n_seeds)
        flow:   trained 1D NSF
        n_sim:  per (origin, seed) sim count
    Returns:
        scenarios: (n_origins, n_seeds, n_sim, future_len)
    """
    n_origins, n_seeds = sigmas.shape
    flow.eval()
    n_total = n_origins * n_seeds * n_sim * future_len
    with torch.no_grad():
        eps_flat = flow.sample(n_total).cpu().numpy().squeeze()  # (n_total,)
    # reshape
    eps_arr = eps_flat.reshape(n_origins, n_seeds, n_sim, future_len)
    sig_b = sigmas[:, :, np.newaxis, np.newaxis]  # broadcast
    scenarios = sig_b * eps_arr  # (n_origins, n_seeds, n_sim, future_len)
    print(f"  scenarios shape: {scenarios.shape}")
    return scenarios


# =====================================================================
# 5. Evaluation
# =====================================================================

def evaluate(scenarios, actual_cum, actual_std):
    """13w cumulative return distribution metrics."""
    n_origins, n_seeds, n_sim, future_len = scenarios.shape

    # cumulative sum over 13 steps per (origin, seed, sim)
    cum_returns = scenarios.sum(axis=3)  # (n_origins, n_seeds, n_sim)

    # === EMD: pooled (all origins × seeds × sims) vs actual (n_origins) ===
    pooled_cum = cum_returns.flatten()
    emd = wasserstein_distance(pooled_cum, actual_cum)

    # === CVaR_5% pooled ===
    n_tail = max(int(len(pooled_cum) * 0.05), 1)
    cvar_pooled = float(np.mean(np.sort(pooled_cum)[:n_tail]))
    n_tail_act = max(int(len(actual_cum) * 0.05), 1)
    cvar_actual = float(np.mean(np.sort(actual_cum)[:n_tail_act]))

    # === Calibration: 80%, 95% interval per origin ===
    cov80_list = []
    cov95_list = []
    for i in range(n_origins):
        cum_i = cum_returns[i].flatten()  # (n_seeds * n_sim,)
        for cov_list, alpha in [(cov80_list, 0.10), (cov95_list, 0.025)]:
            lo = np.quantile(cum_i, alpha)
            hi = np.quantile(cum_i, 1 - alpha)
            cov_list.append(int(lo <= actual_cum[i] <= hi))
    cov80 = float(np.mean(cov80_list))
    cov95 = float(np.mean(cov95_list))

    # === Mean shift (cumulative) ===
    gen_cum_mean = float(pooled_cum.mean())
    act_cum_mean = float(actual_cum.mean())
    cum_shift = gen_cum_mean - act_cum_mean

    metrics = dict(
        n_origins=n_origins, n_seeds=n_seeds, n_sim=n_sim, future_len=future_len,
        emd_cum=float(emd),
        cvar5_pooled=cvar_pooled,
        cvar5_actual=cvar_actual,
        gen_cum_mean=gen_cum_mean,
        act_cum_mean=act_cum_mean,
        cum_shift=cum_shift,
        cov80_mean=cov80,
        cov95_mean=cov95,
    )
    return metrics


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="3M scenario generation (Model A × Flow Model B)")
    ap.add_argument("--pilot-split", action="store_true",
                    help="data/pilot_split 사용 (default)")
    ap.add_argument("--result-dir",  default=os.path.join(HERE, "result"))
    ap.add_argument("--n-sim",       type=int, default=100)
    ap.add_argument("--flow-epochs", type=int, default=200)
    ap.add_argument("--flow-layers", type=int, default=4)
    ap.add_argument("--flow-bins",   type=int, default=8)
    ap.add_argument("--seed",        type=int, default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
    if args.pilot_split:
        train_csv = os.path.join(repo_root, "data", "pilot_split", "train.csv")
        test_csv  = os.path.join(repo_root, "data", "pilot_split", "test.csv")
        fold_tag = "pilot"
    else:
        ap.error("--pilot-split required for now")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("=" * 72)
    print("3M Scenario Generation — Model A (variant 13) × Flow Model B (1D NSF)")
    print("=" * 72)
    print(f"  device: {device}")
    print(f"  train: {train_csv}")
    print(f"  test : {test_csv}")
    print(f"  N sim: {args.n_sim}, flow epochs: {args.flow_epochs}")

    # === Step 1: ε training data ===
    print("\n[1] ε training data 추출")
    eps_train, eps_meta = extract_eps_train(train_csv)

    # === Step 2: Flow learning ===
    print("\n[2] Flow 1D NSF 학습")
    flow_path = os.path.join(args.result_dir, "scenario_3m_flow_1d.pt")
    flow = train_flow(eps_train, flow_path,
                      num_layers=args.flow_layers, num_bins=args.flow_bins,
                      epochs=args.flow_epochs, device=device)

    # === Step 3: Model A (variant 13) inference ===
    print("\n[3] Model A (variant 13) σ_pred")
    models = load_variant13_models(args.result_dir, fold_tag=fold_tag, device=device)
    sigmas, actual_cum, actual_std = predict_sigma(models, test_csv, device=device)

    # === Step 4: Scenario generation ===
    print("\n[4] Scenario generation")
    scenarios = generate_scenarios(sigmas, flow, n_sim=args.n_sim, device=device)

    # === Step 5: Eval ===
    print("\n[5] Evaluation")
    metrics = evaluate(scenarios, actual_cum, actual_std)
    print("\n" + "=" * 72)
    print("Metrics")
    print("=" * 72)
    for k, v in metrics.items():
        print(f"  {k:<20s} = {v}")

    # === Save ===
    npz_path = os.path.join(args.result_dir, "scenario_3m.npz")
    np.savez(npz_path,
             scenarios=scenarios.astype(np.float32),
             actual_cum=actual_cum.astype(np.float32),
             actual_std=actual_std.astype(np.float32),
             sigmas=sigmas.astype(np.float32),
             **{k: v for k, v in metrics.items() if not isinstance(v, str)})
    print(f"\n  saved: {npz_path}")

    csv_path = os.path.join(args.result_dir, "scenario_3m_metrics.csv")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    print(f"  saved: {csv_path}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
