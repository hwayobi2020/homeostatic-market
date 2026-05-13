"""Sensitivity for Sequence-conditioned Flow — scenario generation + tail metrics.

Architecture (train_flow_seq.py):
  Encoder (Transformer) + Joint 13-dim Cond NSF
  Input  cond sequence (past 52w full + future 13w with only tbill_wr unmask)
  Output 13-step sp_return path joint distribution

Workflow:
  1. Load ckpt (flow_seq_{fold}_best.pt)
  2. Build test windows (X_cond, Y_actual) — same standardization as train
  3. Sample 1000 paths per test origin (joint 13-dim sample)
  4. Unstandardize sp_return path → r̂_path
  5. Compute pooled tail metrics: CRPS, EMD, cov95, std ratio, VaR1/CVaR1 Δ
  6. Save fan chart + histogram + summary JSON

Usage:
  python colab/dual_3ch/sensitivity_flow_seq.py --fold F_long
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from train_flow_seq import (
    SequenceEncoder, SequenceCondFlow, build_joint_flow,
    load_windows_seq, mask_future_channels,
    PAST_LEN, FUTURE_LEN, COND_COLS, N_CHANNELS, TBILL_CH, MASK_FUTURE_CH,
)


# =====================================================================
# Metrics
# =====================================================================

def crps_ensemble_sample(sim, y):
    n = len(sim)
    sim_sorted = np.sort(sim)
    term1 = np.mean(np.abs(sim - y))
    i = np.arange(n)
    term2 = (2.0 / (n * n)) * np.sum((2 * i + 1 - n) * sim_sorted)
    return float(term1 - 0.5 * term2)


def crps_pooled(sim_paths, actual_paths):
    n_origin, n_sim, T = sim_paths.shape
    vals = np.empty(n_origin * T, dtype=np.float64)
    k = 0
    for t in range(n_origin):
        for w in range(T):
            vals[k] = crps_ensemble_sample(sim_paths[t, :, w], actual_paths[t, w])
            k += 1
    return float(np.mean(vals)), float(np.std(vals))


def compute_emd_1d(samples_a, samples_b, n_bins=200):
    lo = float(min(samples_a.min(), samples_b.min()))
    hi = float(max(samples_a.max(), samples_b.max()))
    if hi - lo < 1e-12:
        return 0.0
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
    n_origin, n_sim, h = sim_paths.shape
    sim_cum = np.cumsum(sim_paths, axis=2)
    actual_cum = np.cumsum(actual_paths, axis=1)
    sim_flat = sim_cum.reshape(-1, h)
    p05, p25, p50, p75, p95 = np.percentile(sim_flat, [5, 25, 50, 75, 95], axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, h + 1)
    ax.fill_between(x, p05, p95, alpha=0.18, color="C0", label="Flow 90% CI")
    ax.fill_between(x, p25, p75, alpha=0.30, color="C0", label="Flow 50% CI")
    ax.plot(x, p50, color="C0", lw=2, label="Flow median")
    ax.plot(x, np.median(actual_cum, axis=0), color="red", lw=2, ls="--", label="Actual median")
    ax.plot(x, np.percentile(actual_cum, 10, axis=0), color="red", lw=1, ls=":")
    ax.plot(x, np.percentile(actual_cum, 90, axis=0), color="red", lw=1, ls=":")
    ax.set_xlabel("week (future)"); ax.set_ylabel("cumulative sp_return")
    ax.set_title(title); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()
    print(f"    saved fanchart: {save_path}")


def plot_histogram(actual_flat, sim_flat, title, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    lo = float(min(actual_flat.min(), np.quantile(sim_flat, 0.001)))
    hi = float(max(actual_flat.max(), np.quantile(sim_flat, 0.999)))
    bins = np.linspace(lo, hi, 100)
    ax.hist(sim_flat, bins=bins, density=True, alpha=0.4, color="C0",
            label=f"Flow sim (n={len(sim_flat):,})")
    ax.hist(actual_flat, bins=bins, density=True, alpha=0.5, color="red",
            label=f"Actual (n={len(actual_flat):,})")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("sp_return (weekly)"); ax.set_ylabel("density")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=100, bbox_inches="tight"); plt.close()
    print(f"    saved histogram: {save_path}")


# =====================================================================
# Main
# =====================================================================

def load_model_from_ckpt(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = state["meta"]
    encoder = SequenceEncoder(
        d_input=len(meta["cond_cols"]),
        d_model=meta["d_model"], n_heads=meta["n_heads"], n_layers=meta["n_layers"],
        past_len=meta["past_len"], future_len=meta["future_len"],
    )
    flow = build_joint_flow(
        features=meta["future_len"], context_features=meta["d_model"],
        num_layers=meta["n_flow_layers"], hidden_features=meta["n_flow_hidden"],
        num_blocks=meta["n_flow_blocks"], num_bins=meta["n_flow_bins"],
        tail_bound=meta["flow_tail_bound"],
    )
    model = SequenceCondFlow(encoder, flow).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--result-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--n-sim", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_csv = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    if not os.path.exists(test_csv):
        sys.exit(f"[FATAL] missing {test_csv}")
    ckpt_path = os.path.join(args.result_dir, f"flow_seq_{args.fold}_best.pt")
    if not os.path.exists(ckpt_path):
        sys.exit(f"[FATAL] missing ckpt: {ckpt_path}\n  → train_flow_seq.py --fold {args.fold} 먼저 실행")

    print("=" * 78)
    print(f" Sensitivity — sequence-cond Flow  fold={args.fold}  n_sim={args.n_sim}")
    print("=" * 78)

    print(f"\n[1] Load ckpt + model")
    model, meta = load_model_from_ckpt(ckpt_path, device)
    cond_stats = meta["cond_stats"]; target_stats = meta["target_stats"]
    print(f"    best_epoch={meta['best_epoch']}, best_val_nll={meta['best_val_nll']:+.4f}")

    print(f"\n[2] Load test windows")
    Xte, Yte, _, _, n_te = load_windows_seq(test_csv, cond_stats=cond_stats, target_stats=target_stats)
    print(f"    test windows = {n_te}")
    if n_te == 0:
        sys.exit("[FATAL] no test windows.")

    Xte_m = mask_future_channels(Xte, PAST_LEN, MASK_FUTURE_CH).to(device)
    Yte_dev = Yte.to(device)

    # Val NLL on test
    with torch.no_grad():
        test_nll = float(-model.log_prob(Yte_dev, Xte_m).mean().item())
    print(f"    test NLL = {test_nll:+.4f}")

    print(f"\n[3] Sample {args.n_sim} paths per test origin (joint 13-dim)")
    sim_paths_std = np.zeros((n_te, args.n_sim, FUTURE_LEN), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_te, args.chunk):
            e = min(s + args.chunk, n_te)
            sub = Xte_m[s:e]
            samples = model.sample(args.n_sim, sub)
            # samples shape: (B, n_sim, future_len)
            sim_paths_std[s:e] = samples.cpu().numpy().astype(np.float32)
    print(f"    sim_paths shape = {sim_paths_std.shape}")

    # Unstandardize sp_return (target_stats)
    tmu = target_stats["mean"]; tsd = target_stats["std"]
    sim_paths = sim_paths_std * tsd + tmu
    actual_paths = Yte.cpu().numpy() * tsd + tmu

    # Pooled metrics
    actual_flat = actual_paths.ravel()
    sim_flat = sim_paths.ravel()

    var5_act = compute_var(actual_flat, 0.05); var5_sim = compute_var(sim_flat, 0.05)
    var1_act = compute_var(actual_flat, 0.01); var1_sim = compute_var(sim_flat, 0.01)
    cvar5_act = compute_cvar(actual_flat, 0.05); cvar5_sim = compute_cvar(sim_flat, 0.05)
    cvar1_act = compute_cvar(actual_flat, 0.01); cvar1_sim = compute_cvar(sim_flat, 0.01)
    emd = compute_emd_1d(actual_flat, sim_flat, n_bins=200)
    crps_m, crps_s = crps_pooled(sim_paths, actual_paths)

    lo95, hi95 = np.percentile(sim_flat, [2.5, 97.5])
    cov95 = float(((actual_flat >= lo95) & (actual_flat <= hi95)).mean())
    lo80, hi80 = np.percentile(sim_flat, [10.0, 90.0])
    cov80 = float(((actual_flat >= lo80) & (actual_flat <= hi80)).mean())
    lo50, hi50 = np.percentile(sim_flat, [25.0, 75.0])
    cov50 = float(((actual_flat >= lo50) & (actual_flat <= hi50)).mean())

    std_actual = float(actual_paths.std(ddof=1))
    std_sim = float(sim_paths.std(ddof=1))

    print(f"\n[4] Metrics")
    print(f"    test_NLL     = {test_nll:+.4f}")
    print(f"    CRPS pooled  = {crps_m:.5f}  (std {crps_s:.5f})")
    print(f"    EMD          = {emd:.6f}")
    print(f"    std act/sim  = {std_actual:.5f} / {std_sim:.5f}  (ratio {std_sim/std_actual:.3f})")
    print(f"    VaR1   act/sim/Δ  = {var1_act:+.5f} / {var1_sim:+.5f} / {var1_sim-var1_act:+.5f}")
    print(f"    CVaR1  act/sim/Δ  = {cvar1_act:+.5f} / {cvar1_sim:+.5f} / {cvar1_sim-cvar1_act:+.5f}")
    print(f"    cov50 / cov80 / cov95 = {cov50:.3f} / {cov80:.3f} / {cov95:.3f}")

    # Plots
    print(f"\n[5] Plots")
    title = (f"Sequence-cond Flow — fold {args.fold} (n_origin={n_te}, n_sim={args.n_sim})\n"
             f"future tbill input only; joint 13-step sp_return path")
    out_prefix = f"sensitivity_flow_seq_{args.fold}"
    plot_fanchart(actual_paths, sim_paths,
                  title + "\nfan chart (cumulative)",
                  os.path.join(args.result_dir, f"{out_prefix}_fanchart.png"))
    plot_histogram(actual_flat, sim_flat,
                   title + "\nweekly sp_return histogram",
                   os.path.join(args.result_dir, f"{out_prefix}_histogram.png"))

    # Save summary
    summary = dict(
        fold=args.fold,
        model="Sequence-cond Flow (Transformer encoder + Joint 13D Cond NSF)",
        n_origin=int(n_te), n_sim_per_origin=int(args.n_sim),
        test_nll=test_nll,
        crps_pooled=crps_m, crps_std=crps_s,
        emd=emd,
        std_actual=std_actual, std_sim=std_sim,
        std_ratio=std_sim / std_actual,
        var_5pct_actual=var5_act, var_5pct_sim=var5_sim, var_5pct_diff=var5_sim - var5_act,
        var_1pct_actual=var1_act, var_1pct_sim=var1_sim, var_1pct_diff=var1_sim - var1_act,
        cvar_5pct_actual=cvar5_act, cvar_5pct_sim=cvar5_sim, cvar_5pct_diff=cvar5_sim - cvar5_act,
        cvar_1pct_actual=cvar1_act, cvar_1pct_sim=cvar1_sim, cvar_1pct_diff=cvar1_sim - cvar1_act,
        coverage_50=cov50, coverage_80=cov80, coverage_95=cov95,
        cond_cols=COND_COLS, mask_future_ch=MASK_FUTURE_CH, tbill_ch=TBILL_CH,
        ckpt_meta_best_epoch=meta["best_epoch"], ckpt_meta_best_val_nll=meta["best_val_nll"],
    )
    summary_path = os.path.join(args.result_dir, f"{out_prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    saved summary: {summary_path}")


if __name__ == "__main__":
    main()
