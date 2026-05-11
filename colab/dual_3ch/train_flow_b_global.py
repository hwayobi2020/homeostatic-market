"""Unconditional 1D Neural Spline Flow on globally-standardized ε.

기존 generate_3m_scenario.py 의 ε 정의 (`r / std(window)`, within-window 정규화) 는
window 단위 정규화로 fat-tail/skew 를 평탄화시켜 ε ≈ N(0,1) 로 만들어 Flow vs Gaussian
이 통계적으로 구분 안 되는 결과를 낳음.

이 스크립트는 ε 정의를 변경:

  ε = (r - r̄_train) / σ_train       (global, 단일 scalar 정규화)

  - r̄_train, σ_train 은 train CSV 전체 sp_return 의 mean/std (단일 scalar)
  - 결과 ε 의 skew/kurt 가 train return 의 실제 fat-tail (skew −0.86, kurt +6.85)
    을 그대로 보존
  - σ̂ × ε 결합 시 std = σ̂ (Model A 의미 변함 없음), shape 만 fat-tail 로 갱신

Model B 는 **unconditional** 1D NSF — conditional 은 macro extrapolation 문제로 폐기.

Output:
  result/scenario_3m_flow_1d_global.pt          (state_dict + meta)
  result/scenario_3m_flow_1d_global_meta.json   (training stats + sanity check)

Usage:
  python colab/dual_3ch/train_flow_b_global.py --pilot-split
  python colab/dual_3ch/sensitivity_v13.py --flow-mode global
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

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse the unconditional 1D NSF builder + training loop from existing module.
from generate_3m_scenario import build_flow_1d, train_flow


# =====================================================================
# Extract ε with global standardization
# =====================================================================

def extract_eps_global(train_csv):
    """Global standardization of sp_return.

      r̄ = mean of all train sp_return (single scalar)
      σ  = std of all train sp_return (single scalar, ddof=1)
      ε  = (r - r̄) / σ                  per row

    Returns:
      eps_tensor: torch.FloatTensor (N,)   — N = len(train sp_return)
      meta: dict with mean, std, n, raw skew/kurt, eps skew/kurt (should match)
    """
    df = pd.read_csv(train_csv)
    if "sp_return" not in df.columns:
        raise KeyError("train CSV missing 'sp_return' column")
    r = df["sp_return"].values.astype(np.float64)
    n = len(r)
    if n < 30:
        raise RuntimeError(f"too few train rows: {n}")
    r_mean = float(r.mean())
    r_std  = float(r.std(ddof=1))
    if r_std < 1e-12:
        raise RuntimeError(f"degenerate r_std={r_std}")
    eps = ((r - r_mean) / r_std).astype(np.float32)

    # Empirical moments — raw vs ε (should match in skew/kurt, scale-invariant)
    def moments(x):
        x = np.asarray(x, dtype=np.float64)
        m, s = float(x.mean()), float(x.std(ddof=1))
        if s < 1e-12:
            return m, s, float("nan"), float("nan")
        sk = float(((x - m) ** 3).mean() / (s ** 3))
        kt = float(((x - m) ** 4).mean() / (s ** 4) - 3.0)
        return m, s, sk, kt

    rm, rs, rsk, rkt = moments(r)
    em, es, esk, ekt = moments(eps)

    print(f"  sp_return:  n={n}, mean={rm:+.6f}, std={rs:.6f}, "
          f"skew={rsk:+.3f}, ex_kurt={rkt:+.3f}, min/max=[{r.min():+.4f},{r.max():+.4f}]")
    print(f"  ε (global): mean={em:+.4f}, std={es:.4f}, "
          f"skew={esk:+.3f}, ex_kurt={ekt:+.3f}, min/max=[{eps.min():+.3f},{eps.max():+.3f}]")
    print(f"    (skew/kurt are scale-invariant — ε vs raw should match)")

    return torch.from_numpy(eps), dict(
        n=n, r_mean=rm, r_std=rs,
        raw_skew=rsk, raw_ex_kurt=rkt,
        eps_skew=esk, eps_ex_kurt=ekt,
        eps_min=float(eps.min()), eps_max=float(eps.max()),
    )


# =====================================================================
# Sanity — generated sample moments should match train ε empirical
# =====================================================================

def sanity_check_unconditional(flow, train_meta, device, n_sample=5000, seed=42):
    """Sample n_sample ε from trained Flow, compare moments to train ε empirical."""
    flow.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        eps_gen = flow.sample(n_sample).cpu().numpy().reshape(-1)

    def moments(x):
        m, s = float(np.mean(x)), float(np.std(x, ddof=1))
        if s < 1e-12:
            return m, s, float("nan"), float("nan")
        sk = float(np.mean((x - m) ** 3) / (s ** 3))
        kt = float(np.mean((x - m) ** 4) / (s ** 4) - 3.0)
        return m, s, sk, kt

    m, s, sk, kt = moments(eps_gen)
    print(f"\n  Sanity — generated ε moments (n={n_sample}):")
    print(f"    {'source':22s}  {'mean':>10s}  {'std':>8s}  {'skew':>8s}  {'ex_kurt':>10s}")
    print(f"    {'train ε empirical':22s}  {train_meta['eps_skew']:>10}".replace(
        f"{train_meta['eps_skew']:>10}",
        f"{0.0:+10.4f}  {1.0:>8.4f}  {train_meta['eps_skew']:+8.3f}  {train_meta['eps_ex_kurt']:+10.3f}"
    ))
    print(f"    {'Flow generated':22s}  {m:+10.4f}  {s:>8.4f}  {sk:+8.3f}  {kt:+10.3f}")

    # SE estimates for sample-size n
    se_skew = math.sqrt(6.0 / n_sample)
    se_kurt = math.sqrt(24.0 / n_sample)
    print(f"    SE(skew)≈{se_skew:.3f},  SE(ex_kurt)≈{se_kurt:.3f}")

    # Distance metrics
    d_sk = sk - train_meta['eps_skew']
    d_kt = kt - train_meta['eps_ex_kurt']
    print(f"    Δskew (gen − emp)    = {d_sk:+.3f}   "
          f"({'OK' if abs(d_sk) < 3*se_skew else 'MISMATCH'} at 3·SE)")
    print(f"    Δex_kurt (gen − emp) = {d_kt:+.3f}   "
          f"({'OK' if abs(d_kt) < 3*se_kurt else 'MISMATCH'} at 3·SE)")

    return dict(
        gen_mean=m, gen_std=s, gen_skew=sk, gen_ex_kurt=kt,
        delta_skew=d_sk, delta_ex_kurt=d_kt,
        se_skew=se_skew, se_kurt=se_kurt,
    )


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Train UNCONDITIONAL 1D NSF on globally-standardized ε "
                    "(preserves train return fat-tail)")
    ap.add_argument("--pilot-split", action="store_true", default=True)
    ap.add_argument("--train-csv",  default=None,
                    help="override train CSV path (default: data/pilot_split/train.csv)")
    ap.add_argument("--result-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--save-name",  default="scenario_3m_flow_1d_global.pt")
    ap.add_argument("--num-layers", type=int,   default=4)
    ap.add_argument("--num-bins",   type=int,   default=8)
    ap.add_argument("--tail-bound", type=float, default=5.0)
    ap.add_argument("--epochs",     type=int,   default=200)
    ap.add_argument("--batch",      type=int,   default=256)
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--seed",       type=int,   default=2026)
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
    print(" Train UNCONDITIONAL 1D NSF — globally-standardized ε (fat-tail preserved)")
    print("=" * 78)
    print(f"  device     : {device}")
    print(f"  train_csv  : {train_csv}")
    print(f"  save_path  : {save_path}")
    print(f"  ε def      : ε = (r - r̄_train) / σ_train    (global scalar normalization)")

    # 1. Extract ε
    print("\n[1] Extract ε with global standardization")
    eps, train_meta = extract_eps_global(train_csv)

    # 2. Train unconditional 1D NSF (reuse build_flow_1d / train_flow from generate_3m_scenario)
    print("\n[2] Train unconditional 1D NSF")
    flow = train_flow(eps, save_path,
                      num_layers=args.num_layers, num_bins=args.num_bins,
                      tail_bound=args.tail_bound, epochs=args.epochs,
                      batch=args.batch, lr=args.lr, device=device)

    # 3. Sanity — generated moments should match empirical
    print("\n[3] Sanity check — generated vs empirical moments")
    sanity = sanity_check_unconditional(flow, train_meta, device, n_sample=5000)

    # 4. Meta json
    meta_json_path = save_path.replace(".pt", "_meta.json")
    meta_json = dict(
        model_type="unconditional_1d_nsf_global_eps",
        eps_definition="global: (r - r_mean_train) / r_std_train",
        train_csv=os.path.abspath(train_csv),
        r_mean_train=train_meta["r_mean"],
        r_std_train=train_meta["r_std"],
        n_train=train_meta["n"],
        train_eps_skew=train_meta["eps_skew"],
        train_eps_ex_kurt=train_meta["eps_ex_kurt"],
        train_eps_min=train_meta["eps_min"],
        train_eps_max=train_meta["eps_max"],
        sanity=sanity,
        flow_config=dict(
            num_layers=args.num_layers, num_bins=args.num_bins,
            tail_bound=args.tail_bound, epochs=args.epochs,
            batch=args.batch, lr=args.lr,
        ),
    )
    with open(meta_json_path, "w") as f:
        json.dump(meta_json, f, indent=2)
    print(f"\n  meta saved: {meta_json_path}")

    # 5. Summary
    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    print(f"  Train return:    skew={train_meta['raw_skew']:+.3f}, ex_kurt={train_meta['raw_ex_kurt']:+.3f}")
    print(f"  Train ε (=raw):  skew={train_meta['eps_skew']:+.3f}, ex_kurt={train_meta['eps_ex_kurt']:+.3f}")
    print(f"  Flow generated:  skew={sanity['gen_skew']:+.3f}, ex_kurt={sanity['gen_ex_kurt']:+.3f}")
    print(f"  → Δskew={sanity['delta_skew']:+.3f}  Δex_kurt={sanity['delta_ex_kurt']:+.3f}")
    print(f"  → 500-path SE: skew ≈ ±{math.sqrt(6/500):.3f}, ex_kurt ≈ ±{math.sqrt(24/500):.3f}")
    print(f"\n  Next: python colab/dual_3ch/sensitivity_v13.py --flow-mode global")


if __name__ == "__main__":
    main()
