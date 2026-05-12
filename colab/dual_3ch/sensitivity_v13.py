"""Sensitivity analysis + Fan Chart for Variant 13 (Model A: base + metab_13w cond).

Variant 13 (code, `base_metab_wti_har_13`):
  cond = [tbill_wr, m2_13w_cum_lag, gdp_13w_proxy_lag, cpi_13w_cum_lag,
          metab_13w, wti_wr, sp_log_std_13w]
  target_past = [sp_return]
  output = log( std(future 13w sp_return) )  → σ̂ = exp(log_std_pred)

생성하는 Figure (모두 result/ 에 저장):

  Figure 1 — Sensitivity curves (2×2 grid):
      행: (calm, stress) origin   |   열: (tbill, metab_13w) 변수
      X: Δ (level shift, raw unit) — annual %p 도 보조축으로 표기
      Y: σ̂ (5-seed mean) with ±1 std band
      Δ=0 baseline vertical guide + 실측 future-13w realized std horizontal guide.

  Figure 2 — Fan charts (2×3 grid):
      행: (calm, stress) origin   |   열: (tbill_baseline, +1.0%p, +2.0%p annual)
      Flow scenarios (500 path) — 5-95%, 10-90% shaded bands + median (solid) + mean (dotted)
      Gaussian baseline (500 path, same σ̂ × N(0,1)) — 5-95% dashed outline (no fill)
      Actual realized 13w cumulative path overlay (green solid).

  Figure 3 — Week-13 cumulative return distribution (2×3 grid):
      Same panel grid as Figure 2.
      Flow distribution (orange filled histogram + KDE) vs
      Gaussian distribution (blue filled histogram + KDE, semi-transparent overlay).
      Actual realized cumulative (green vertical line).
      Annotation per panel: mean, median, skewness, excess kurtosis,
                            + bootstrap 95% CI on skew/kurt.

Two-Model framework:
  σ̂ : Variant 13, 5 seed (42-46) mean of predicted log_std → exp.
  ε  : Model B = 1D Neural Spline Flow on train ε (sp_return / 13w std).
  r_path = σ̂ × ε,  cumulative_t = Σ_{k=1..t} r_k.
  Gaussian baseline : same σ̂ × N(0,1) (same n_sim, separate sampling).

Output:
  result/sensitivity_v13_curves.png      (Figure 1)
  result/sensitivity_v13_fanchart.png    (Figure 2)
  result/sensitivity_v13_histogram.png   (Figure 3)
  result/sensitivity_v13_summary.csv     (모든 정량 지표)

Usage (Colab, after running paper table 학습 + Model B):
  !python colab/dual_3ch/sensitivity_v13.py --pilot-split

Pre-requisites:
  - v13 학습 완료 (vol_pilot_3m_psel_v13_*_pilot_seed{42..46}_best.pt 5 ckpt)
  - data/pilot_split/{train,test}.csv
  - Model B (1D NSF): result/scenario_3m_flow_1d.pt — 없으면 자동 학습 (train ε 추출 후).
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Variant 13 model class + window utilities
from train_vol_pilot_3m import (
    CausalTransformerVolScalar, build_spec, load_windows, mask_future_channels,
    PAST_LEN, FUTURE_LEN, L, D_MODEL, N_HEADS, N_LAYERS,
)

# Flow 1D NSF — Model B (unconditional)
from generate_3m_scenario import (
    extract_eps_train, build_flow_1d, train_flow,
)

# Flow 1D NSF — Model B (conditional, macro-context)
from train_flow_b_conditional import (
    build_conditional_flow_1d, extract_eps_with_context,
    train_conditional_flow,
)

# Flow 1D NSF — Model B (global ε, with selectable base: normal / student_t / skew_t)
from train_flow_b_global import (
    build_flow as build_flow_global,
    StudentTBase, SkewTBase,
)


# =====================================================================
# Constants / defaults
# =====================================================================

V13_VARIANT_ID = 13   # code variant id (paper row #2)
SEEDS_DEFAULT = [42, 43, 44, 45, 46]

# Perturbation grids (raw units, level shift over entire 65w window)
# tbill_wr is weekly decimal:  annual %p / 52 = weekly Δ
TBILL_ANNUAL_PP_RANGE = 2.0          # ±2.0%p annual
TBILL_ANNUAL_PP_STEP  = 0.25
METAB_FRACTION_RANGE  = 0.02         # ±2.0%p 13w cum fraction (metab_13w std ≈ 0.015)
METAB_FRACTION_STEP   = 0.004

# Fan chart tbill scenarios (annual %p shifts on top of baseline)
FAN_TBILL_ANNUAL_PP = [0.0, 1.0, 2.0]

# Scenario generation
N_SIM_DEFAULT  = 100   # per (origin, seed) → total 100 × 5 seeds = 500 paths per scenario
FLOW_LAYERS    = 4     # for original uncond mode (within-window ε, scenario_3m_flow_1d.pt)
FLOW_BINS      = 8
FLOW_TAIL      = 5.0
FLOW_EPOCHS    = 200
# Global-ε mode (v2 ckpt) — capacity expanded to learn train return fat-tail.
GLOBAL_FLOW_LAYERS = 6
GLOBAL_FLOW_BINS   = 16
GLOBAL_FLOW_TAIL   = 10.0

# Bootstrap
BOOTSTRAP_B    = 2000


# =====================================================================
# Origin selection — calm + stress from test set
# =====================================================================

def select_origins(test_csv, n_calm=1, n_stress=1):
    """Pick (n_calm + n_stress) origins from test set.

    Calm   = future-13w realized std 하위 (lowest realized vol).
    Stress = future-13w realized std 상위 (highest realized vol).

    Returns:
      list of dicts: {kind, origin_idx, date, future_realized_std, future_cum_return}
    """
    df = pd.read_csv(test_csv, parse_dates=["date"])
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        raise RuntimeError(f"Not enough test data: {n} < L={L}")
    sp = df["sp_return"].values.astype(np.float64)
    dates = df["date"].values

    rec = []
    for t in range(n_w):
        fut = sp[t + PAST_LEN : t + L]   # 13 step
        rstd = float(fut.std(ddof=1))
        rcum = float(fut.sum())
        # origin_date = 마지막 past 시점 (origin index t 의 마지막 past row)
        origin_date = pd.Timestamp(dates[t + PAST_LEN - 1])
        rec.append(dict(origin_idx=t, origin_date=origin_date,
                        future_realized_std=rstd, future_cum_return=rcum))
    rec_df = pd.DataFrame(rec).sort_values("future_realized_std").reset_index(drop=True)

    chosen = []
    for i in range(n_calm):
        r = rec_df.iloc[i]
        chosen.append(dict(kind=f"calm{i+1}" if n_calm > 1 else "calm",
                           origin_idx=int(r["origin_idx"]),
                           origin_date=r["origin_date"],
                           future_realized_std=float(r["future_realized_std"]),
                           future_cum_return=float(r["future_cum_return"])))
    for i in range(n_stress):
        r = rec_df.iloc[-(i + 1)]
        chosen.append(dict(kind=f"stress{i+1}" if n_stress > 1 else "stress",
                           origin_idx=int(r["origin_idx"]),
                           origin_date=r["origin_date"],
                           future_realized_std=float(r["future_realized_std"]),
                           future_cum_return=float(r["future_cum_return"])))
    return chosen


# =====================================================================
# Load V13 models (5 seeds)
# =====================================================================

def load_v13_models(result_dir, fold_tag="pilot", device="cuda", sel_filter="*sel*"):
    """Load all vol_pilot_3m_{sel_filter}_v13_*_{fold}_seed{seed}_best.pt ckpts.

    sel_filter: glob fragment to filter selection mode.
      "*sel*" — both msel + isel + psel
      "_msel" — only MSE-selected ckpts
      "_isel" — only IC-selected ckpts
    """
    pattern = os.path.join(result_dir, f"vol_pilot_3m_{sel_filter}_v13_*_{fold_tag}_seed*_best.pt")
    paths = sorted(glob.glob(pattern))
    print(f"  v13 ckpts found: {len(paths)}")
    if not paths:
        raise FileNotFoundError(f"No v13 ckpt at {pattern}\n"
                                f"  → 먼저 run_paper_table_3m.py 로 v13 (paper #2) 학습 필요.")

    models = []
    for p in paths:
        ckpt = torch.load(p, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = CausalTransformerVolScalar(
            d_cond=cfg["d_cond"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            past_len=cfg["past_len"], future_len=cfg["future_len"],
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        seed = int(p.split("_seed")[1].split("_")[0])
        models.append(dict(model=model, seed=seed,
                           cond_cols=ckpt["cond_cols"],
                           target_cols=ckpt["target_cols"],
                           stats_cond=ckpt["stats_cond"],
                           mask_future_ch=ckpt["mask_future_ch"]))
    seeds_loaded = [m["seed"] for m in models]
    print(f"    seeds: {seeds_loaded}")
    return models


# =====================================================================
# Window builder (raw → normalized → masked) with perturbation
# =====================================================================

def build_window_tensors(df_test, origin_idx, cols_cond, cols_target,
                         stats_cond, mask_future_ch,
                         perturb_col=None, perturb_delta=0.0):
    """Build (cond_norm_masked, target_past_raw) tensors for one origin.

    Perturbation: level shift on raw value of perturb_col across entire window (L rows).
    Normalization: applied with train stats_cond.
    Masking: future portion of mask_future_ch channels set to 0 (per trainer convention).

    Returns:
      cond_t:  (1, L, d_cond) float32 tensor — normalized + masked
      tgt_t:   (1, PAST_LEN, d_target) float32 tensor — raw target_past
    """
    window = df_test.iloc[origin_idx : origin_idx + L].copy()

    if perturb_col is not None and perturb_delta != 0.0:
        if perturb_col not in window.columns:
            raise KeyError(f"perturb_col={perturb_col} not in window")
        window[perturb_col] = window[perturb_col].astype(np.float64) + float(perturb_delta)

    C_raw = window[cols_cond].values.astype(np.float32)             # (L, d_cond)
    X     = window[cols_target].values.astype(np.float32)           # (L, d_target)

    cmu = np.asarray(stats_cond["mean"], dtype=np.float32)
    csd = np.asarray(stats_cond["std"],  dtype=np.float32)
    C   = (C_raw - cmu) / csd                                       # normalize

    for ch in mask_future_ch:
        C[PAST_LEN:, ch] = 0.0                                      # mask future

    X_past = X[:PAST_LEN, :]
    return (torch.from_numpy(C).unsqueeze(0),
            torch.from_numpy(X_past).unsqueeze(0))


def build_batch_tensors(df_test, origin_idx, perturbations, cols_cond, cols_target,
                        stats_cond, mask_future_ch, perturb_col):
    """Build batched (cond, target_past) tensors for one origin × many perturbations.

    Args:
      perturbations: list of float Δ values (raw units of perturb_col)
      perturb_col: column to shift

    Returns:
      cond_batch:  (n_pert, L, d_cond)
      tgt_batch:   (n_pert, PAST_LEN, d_target)
    """
    cs = []
    ts = []
    for delta in perturbations:
        c, t = build_window_tensors(df_test, origin_idx, cols_cond, cols_target,
                                     stats_cond, mask_future_ch,
                                     perturb_col=perturb_col, perturb_delta=delta)
        cs.append(c)
        ts.append(t)
    return torch.cat(cs, dim=0), torch.cat(ts, dim=0)


def predict_sigma_5seed(models, cond_batch, tgt_batch, device):
    """Run Model A inference for each of 5 seeds on a perturbation batch.

    Returns:
      sigmas: (n_pert, n_seeds) — σ̂ = exp(log_std_pred) per (perturbation, seed)
    """
    out = []
    for m in models:
        with torch.no_grad():
            log_std = m["model"](cond_batch.to(device), tgt_batch.to(device)).cpu().numpy()
        out.append(np.exp(log_std))
    return np.stack(out, axis=1)   # (n_pert, n_seeds)


# =====================================================================
# Sensitivity scan
# =====================================================================

def sensitivity_scan(models, df_test, origin_idx, perturb_col, deltas,
                     cols_cond, cols_target, stats_cond, mask_future_ch, device):
    """Scan one variable over delta grid for one origin.

    Returns:
      sigmas: (n_deltas, n_seeds)
    """
    cond_batch, tgt_batch = build_batch_tensors(
        df_test, origin_idx, deltas, cols_cond, cols_target,
        stats_cond, mask_future_ch, perturb_col=perturb_col,
    )
    return predict_sigma_5seed(models, cond_batch, tgt_batch, device)


# =====================================================================
# Statistics — skew, excess kurtosis, bootstrap CI
# =====================================================================

def sample_skew(x):
    """Bias-corrected sample skewness (Fisher-Pearson)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=1)
    if s < 1e-12:
        return float("nan")
    g1 = ((x - m) ** 3).sum() / n / (s ** 3)
    # adjusted Fisher–Pearson
    G1 = math.sqrt(n * (n - 1)) / (n - 2) * g1
    return float(G1)


def sample_excess_kurtosis(x):
    """Bias-corrected sample excess kurtosis (Fisher's g2)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=1)
    if s < 1e-12:
        return float("nan")
    g2_raw = ((x - m) ** 4).sum() / n / (s ** 4) - 3.0
    # adjusted (Joanes & Gill 1998)
    G2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2_raw + 6.0)
    return float(G2)


def bootstrap_ci(x, stat_fn, B=BOOTSTRAP_B, alpha=0.05, seed=0):
    """Bootstrap percentile CI for stat_fn applied to x."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 4:
        return (float("nan"), float("nan"), float("nan"))
    boots = np.zeros(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = stat_fn(x[idx])
    est = float(stat_fn(x))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return (est, lo, hi)


# =====================================================================
# Scenario generation (Flow + Gaussian) for one σ̂
# =====================================================================

def generate_paths(sigma_scalar, flow, n_sim, future_len, device,
                   rng_seed=0, use_gaussian=False, context_vec=None):
    """Generate n_sim paths of length future_len for a single σ̂.

    r_t = sigma_scalar × ε_t,  ε ~ Flow (or N(0,1)).

    If use_gaussian: ε ~ N(0, 1) (independent of flow / context).
    Elif context_vec is not None: conditional Flow — ε ~ flow | context_vec.
    Else: unconditional Flow — ε ~ flow.

    Returns:
      paths: (n_sim, future_len) np.float32
    """
    n_total = n_sim * future_len
    if use_gaussian:
        rng = np.random.default_rng(rng_seed)
        eps = rng.standard_normal(n_total).astype(np.float32)
    else:
        flow.eval()
        with torch.no_grad():
            torch.manual_seed(rng_seed)
            if context_vec is not None:
                # Conditional sample — nflows returns (B=1, n_total, 1) for context (1, d)
                ctx = torch.from_numpy(np.asarray(context_vec, dtype=np.float32)).unsqueeze(0).to(device)
                eps = flow.sample(n_total, context=ctx).cpu().numpy().reshape(-1).astype(np.float32)
            else:
                # Unconditional sample — returns (n_total, 1)
                eps = flow.sample(n_total).cpu().numpy().squeeze().astype(np.float32)
    eps = eps.reshape(n_sim, future_len)
    return sigma_scalar * eps


def generate_paths_5seed(sigmas_5seed, flow, n_sim_per_seed, future_len, device,
                          use_gaussian=False, rng_base=0, context_vec=None):
    """Generate paths across 5 seeds → 5×n_sim_per_seed = total paths.

    sigmas_5seed: (n_seeds,) array of σ̂ per seed.
    context_vec : (d_cond,) np array, or None.  Passed through for conditional flow.

    Returns:
      paths_all: (n_seeds * n_sim_per_seed, future_len)
    """
    parts = []
    for si, sig in enumerate(sigmas_5seed):
        p = generate_paths(float(sig), flow, n_sim_per_seed, future_len, device,
                            rng_seed=rng_base + si, use_gaussian=use_gaussian,
                            context_vec=context_vec)
        parts.append(p)
    return np.concatenate(parts, axis=0)


# =====================================================================
# Context vector helper (for conditional flow)
# =====================================================================

def extract_context_vec(df_test, origin_idx, cols_cond, stats_cond,
                         perturb_col=None, perturb_delta=0.0):
    """Build normalized context vector at origin (last past row index = origin_idx + PAST_LEN - 1).

    Conditional Flow B 의 context 정의와 정확히 일치해야 함 (train_flow_b_conditional.py 와
    같은 row + 같은 stats_cond 정규화).

    perturb_col 가 cond_cols 에 있으면 perturb_delta 만큼 raw 값에 더한 후 정규화.

    Returns:
      ctx_norm: (d_cond,) np.float32
    """
    row = df_test.iloc[origin_idx + PAST_LEN - 1].copy()
    if perturb_col is not None and float(perturb_delta) != 0.0:
        if perturb_col in row.index:
            row[perturb_col] = float(row[perturb_col]) + float(perturb_delta)
    ctx_raw = row[cols_cond].values.astype(np.float32)
    cmu = np.asarray(stats_cond["mean"], dtype=np.float32)
    csd = np.asarray(stats_cond["std"],  dtype=np.float32)
    return ((ctx_raw - cmu) / csd).astype(np.float32)


# =====================================================================
# Figure 1 — Sensitivity curves
# =====================================================================

def make_sensitivity_figure(origins, scan_results, baseline_sigmas, save_path):
    """2×2 grid: (calm, stress) × (tbill, metab_13w).

    scan_results: dict keyed by (origin_kind, perturb_col) → dict
        deltas (n_d,), sigmas (n_d, n_seeds), seeds (list)
    baseline_sigmas: dict keyed by origin_kind → np.array (n_seeds,)
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    var_specs = [
        ("tbill_wr", "tbill (weekly decimal)", "tbill: annual %p shift", 52.0,
                                   TBILL_ANNUAL_PP_RANGE, "C0"),
        ("metab_13w", "metab_13w (13w cum)", "metab_13w: %p (13w cum) shift", 100.0,
                                   METAB_FRACTION_RANGE, "C1"),
    ]
    kinds = [o["kind"] for o in origins]
    n_pairs = min(2, len(kinds))  # only 2 origins expected (calm, stress)
    for r_i, ok in enumerate(kinds[:n_pairs]):
        for c_i, (col, _, sec_label, sec_mult, _drange, color) in enumerate(var_specs):
            ax = axes[r_i, c_i]
            entry = scan_results[(ok, col)]
            deltas  = entry["deltas"]
            sigmas  = entry["sigmas"]                  # (n_d, n_seeds)
            mean_s  = sigmas.mean(axis=1)
            std_s   = sigmas.std(axis=1, ddof=1)
            base_s  = baseline_sigmas[ok].mean()       # baseline (Δ=0)
            actual_std = entry["actual_realized_std"]

            ax.plot(deltas, mean_s, color=color, lw=2,
                    label="σ̂ (5-seed mean)")
            ax.fill_between(deltas, mean_s - std_s, mean_s + std_s,
                            color=color, alpha=0.18, label="±1σ across seeds")
            ax.axhline(actual_std, color="k", linestyle=":", lw=1,
                       label=f"actual realized std = {actual_std:.4f}")
            ax.axvline(0.0, color="gray", linestyle="--", lw=0.8)
            ax.scatter([0.0], [base_s], color="red", s=40, zorder=5,
                       label=f"baseline σ̂ = {base_s:.4f}")

            origin_date = entry["origin_date"]
            ax.set_title(f"[{ok.upper()}] {col}\n"
                         f"origin = {origin_date.date()}  realized_std={actual_std:.4f}",
                         fontsize=11)
            ax.set_xlabel(f"Δ ({col}, raw unit)")
            if c_i == 0:
                ax.set_ylabel("σ̂ (predicted future-13w std)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

            # secondary x-axis with human-readable unit
            secax = ax.secondary_xaxis(
                "top", functions=(lambda x, m=sec_mult: x * m, lambda x, m=sec_mult: x / m))
            secax.set_xlabel(sec_label, fontsize=9)

    fig.suptitle("Sensitivity Curves — Variant 13 (base + metab_13w cond)\n"
                 "(OAT level shift; row=origin regime, col=variable)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


# =====================================================================
# Figure 2 — Fan chart (Flow + Gaussian overlay + actual)
# =====================================================================

def make_fanchart_figure(panels, save_path):
    """2×3 grid: (calm, stress) × (baseline, +1pp, +2pp).

    panels: dict keyed by (origin_kind, tbill_pp_label) → dict
       flow_paths:    (n_total_sim, 13)
       gauss_paths:   (n_total_sim, 13)
       actual_path:   (13,)
       sigma_used:    scalar σ̂ (5-seed mean used for cum_eps × σ̂)
       sigmas_5seed:  (5,)
       origin_date
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey="row")
    kinds = ["calm", "stress"]
    pp_labels = ["+0.0pp", "+1.0pp", "+2.0pp"]

    weeks = np.arange(1, FUTURE_LEN + 1)
    for r_i, ok in enumerate(kinds):
        for c_i, pp in enumerate(pp_labels):
            ax = axes[r_i, c_i]
            P = panels[(ok, pp)]
            flow_paths  = P["flow_paths"]                # (N, 13)
            gauss_paths = P["gauss_paths"]               # (N, 13)
            actual      = P["actual_path"]               # (13,)
            sig_mean    = P["sigma_used"]

            flow_cum  = np.cumsum(flow_paths,  axis=1)   # (N, 13)
            gauss_cum = np.cumsum(gauss_paths, axis=1)
            actual_cum = np.cumsum(actual)

            # Flow bands (filled)
            flow_q05 = np.quantile(flow_cum, 0.05, axis=0)
            flow_q10 = np.quantile(flow_cum, 0.10, axis=0)
            flow_q50 = np.quantile(flow_cum, 0.50, axis=0)
            flow_q90 = np.quantile(flow_cum, 0.90, axis=0)
            flow_q95 = np.quantile(flow_cum, 0.95, axis=0)
            flow_mean = flow_cum.mean(axis=0)

            ax.fill_between(weeks, flow_q05, flow_q95, color="C1", alpha=0.18,
                            label="Flow 5–95%")
            ax.fill_between(weeks, flow_q10, flow_q90, color="C1", alpha=0.30,
                            label="Flow 10–90%")
            ax.plot(weeks, flow_q50, color="C1", lw=2.0, label="Flow median")
            ax.plot(weeks, flow_mean, color="C1", lw=1.2, linestyle=":",
                    label="Flow mean")

            # Gaussian outline overlay (dashed)
            g_q05 = np.quantile(gauss_cum, 0.05, axis=0)
            g_q95 = np.quantile(gauss_cum, 0.95, axis=0)
            g_q50 = np.quantile(gauss_cum, 0.50, axis=0)
            ax.plot(weeks, g_q05, color="C0", linestyle="--", lw=1.3,
                    label="Gauss 5/95%")
            ax.plot(weeks, g_q95, color="C0", linestyle="--", lw=1.3)
            ax.plot(weeks, g_q50, color="C0", lw=1.2, alpha=0.8,
                    label="Gauss median")

            # Actual
            ax.plot(weeks, actual_cum, color="green", lw=2.2, marker="o",
                    markersize=4, label="actual realized")

            ax.axhline(0.0, color="gray", lw=0.5)
            ax.set_title(f"[{ok.upper()}] tbill {pp}   σ̂={sig_mean:.4f}",
                         fontsize=10)
            ax.set_xlabel("Week ahead")
            if c_i == 0:
                ax.set_ylabel("Cumulative log-return")
            ax.grid(True, alpha=0.3)
            if r_i == 0 and c_i == 0:
                ax.legend(loc="lower left", fontsize=7, ncol=2)

    fig.suptitle("Fan Chart — Variant 13 Model A × Flow Model B  (Gaussian overlay)\n"
                 "(row=origin regime, col=tbill annual %p shift; 500 paths per panel)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


# =====================================================================
# Figure 3 — Week-13 cumulative return histogram
# =====================================================================

def make_histogram_figure(panels, save_path):
    """2×3 grid: distribution of cum_return at week 13 (terminal).

    Flow histogram (filled orange) + Gaussian histogram (filled blue, transparent overlay).
    Actual realized cum_return (green vertical line).
    Annotation: skew, kurt, mean, median + bootstrap 95% CI per distribution.
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey="row")
    kinds = ["calm", "stress"]
    pp_labels = ["+0.0pp", "+1.0pp", "+2.0pp"]

    summary_rows = []

    for r_i, ok in enumerate(kinds):
        for c_i, pp in enumerate(pp_labels):
            ax = axes[r_i, c_i]
            P = panels[(ok, pp)]

            flow_cum  = np.cumsum(P["flow_paths"],  axis=1)[:, -1]
            gauss_cum = np.cumsum(P["gauss_paths"], axis=1)[:, -1]
            actual_cum_terminal = float(np.cumsum(P["actual_path"])[-1])

            # x-range — common across both for fair compare
            lo = min(flow_cum.min(), gauss_cum.min(), actual_cum_terminal)
            hi = max(flow_cum.max(), gauss_cum.max(), actual_cum_terminal)
            pad = 0.08 * (hi - lo + 1e-8)
            bins = np.linspace(lo - pad, hi + pad, 51)

            ax.hist(gauss_cum, bins=bins, density=True, color="C0", alpha=0.50,
                    label="Gaussian", edgecolor="C0", linewidth=0.4)
            ax.hist(flow_cum,  bins=bins, density=True, color="C1", alpha=0.65,
                    label="Flow",      edgecolor="C1", linewidth=0.4)

            # Vertical lines
            ax.axvline(actual_cum_terminal, color="green", lw=2.0,
                       label=f"actual = {actual_cum_terminal:+.3f}")
            ax.axvline(np.median(flow_cum),  color="C1", lw=1.5, linestyle="--",
                       alpha=0.85, label=f"Flow med={np.median(flow_cum):+.3f}")
            ax.axvline(np.median(gauss_cum), color="C0", lw=1.5, linestyle="--",
                       alpha=0.85, label=f"Gauss med={np.median(gauss_cum):+.3f}")

            # Stats + bootstrap CI
            f_mean = float(flow_cum.mean())
            f_med  = float(np.median(flow_cum))
            f_sk, f_sk_lo, f_sk_hi = bootstrap_ci(flow_cum, sample_skew, seed=42)
            f_kt, f_kt_lo, f_kt_hi = bootstrap_ci(flow_cum, sample_excess_kurtosis, seed=43)

            g_mean = float(gauss_cum.mean())
            g_med  = float(np.median(gauss_cum))
            g_sk, g_sk_lo, g_sk_hi = bootstrap_ci(gauss_cum, sample_skew, seed=44)
            g_kt, g_kt_lo, g_kt_hi = bootstrap_ci(gauss_cum, sample_excess_kurtosis, seed=45)

            txt = (
                f"Flow:\n"
                f"  mean={f_mean:+.3f}, med={f_med:+.3f}\n"
                f"  skew={f_sk:+.3f} [{f_sk_lo:+.2f},{f_sk_hi:+.2f}]\n"
                f"  ex_kurt={f_kt:+.2f} [{f_kt_lo:+.2f},{f_kt_hi:+.2f}]\n"
                f"Gauss:\n"
                f"  mean={g_mean:+.3f}, med={g_med:+.3f}\n"
                f"  skew={g_sk:+.3f} [{g_sk_lo:+.2f},{g_sk_hi:+.2f}]\n"
                f"  ex_kurt={g_kt:+.2f} [{g_kt_lo:+.2f},{g_kt_hi:+.2f}]"
            )
            ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=7.5,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                              edgecolor="gray", linewidth=0.5))

            ax.set_title(f"[{ok.upper()}] tbill {pp}   (week-13 cum return)",
                         fontsize=10)
            ax.set_xlabel("13w cumulative log-return")
            if c_i == 0:
                ax.set_ylabel("density")
            ax.grid(True, alpha=0.3)
            if r_i == 0 and c_i == 0:
                ax.legend(loc="upper right", fontsize=7)

            summary_rows.append(dict(
                origin_kind=ok, tbill_shift=pp,
                actual_cum=actual_cum_terminal,
                flow_mean=f_mean, flow_median=f_med,
                flow_skew=f_sk, flow_skew_lo=f_sk_lo, flow_skew_hi=f_sk_hi,
                flow_kurt=f_kt, flow_kurt_lo=f_kt_lo, flow_kurt_hi=f_kt_hi,
                gauss_mean=g_mean, gauss_median=g_med,
                gauss_skew=g_sk, gauss_skew_lo=g_sk_lo, gauss_skew_hi=g_sk_hi,
                gauss_kurt=g_kt, gauss_kurt_lo=g_kt_lo, gauss_kurt_hi=g_kt_hi,
            ))

    fig.suptitle("Week-13 Cumulative Return Distribution — Flow vs Gaussian (+ actual)\n"
                 "(Annotations: mean, median, skewness, excess kurtosis, bootstrap 95% CI)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")
    return pd.DataFrame(summary_rows)


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Sensitivity + Fan Chart for Variant 13")
    ap.add_argument("--pilot-split", action="store_true", default=True)
    ap.add_argument("--result-dir",  default=os.path.join(HERE, "result"))
    ap.add_argument("--n-sim-per-seed", type=int, default=N_SIM_DEFAULT,
                    help="Paths per (origin, seed) for fan/histogram — 5 seeds × this = total")
    ap.add_argument("--flow-epochs", type=int, default=FLOW_EPOCHS)
    ap.add_argument("--flow-mode",   type=str, default="global",
                    choices=["uncond", "cond", "global"],
                    help="Flow B mode: "
                         "'uncond' = original 1D NSF on within-window-standardized ε pool, "
                         "'cond'   = macro-conditional 1D NSF (context = 5D macro), "
                         "'global' = UNCONDITIONAL 1D NSF on globally-standardized ε "
                         "           (preserves train return fat-tail, default).")
    ap.add_argument("--cond-flow-ckpt", type=str, default="scenario_3m_flow_1d_cond.pt",
                    help="filename for conditional flow ckpt (under result-dir)")
    ap.add_argument("--global-flow-ckpt", type=str,
                    default="scenario_3m_flow_1d_global_skewt_df5.pt",
                    help="filename for global-ε unconditional flow ckpt. "
                         "Default = Skew-t(α learnable, df=5) base. "
                         "Other options: '..._global_student_df5.pt' (Student-t), "
                         "'..._global_v2.pt' (Normal).")
    ap.add_argument("--fold", type=str, default="pilot",
                    choices=["pilot", "F1", "F2", "F3"],
                    help="Evaluation fold: 'pilot' (legacy single split) or "
                         "'F1'..'F3' (folds_v33_vix_expanding 3-fold expanding-train, 3mo gap, "
                         "test = 인플레 사이클 3단계: F1 시작 21.1-22.6, F2 정점 22.10-24.3, F3 해소 24.7-25.12).")
    ap.add_argument("--loss-mode", choices=["any", "mse", "ic"], default="any",
                    help="Which loss-mode ckpts to load. 'any' matches *sel* (mse+ic+psel). "
                         "'mse' → _msel only; 'ic' → _isel only.")
    ap.add_argument("--seed",        type=int, default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
    fold = args.fold
    if fold == "pilot":
        test_csv  = os.path.join(repo_root, "data", "pilot_split", "test.csv")
        train_csv = os.path.join(repo_root, "data", "pilot_split", "train.csv")
        v13_fold_tag = "pilot"
        if not os.path.exists(test_csv) or not os.path.exists(train_csv):
            sys.exit(f"[FATAL] pilot_split missing — 먼저 build_pilot_split.py 실행")
    else:
        fold_dir = os.path.join(repo_root, "data", "folds_v33_vix_expanding")
        test_csv  = os.path.join(fold_dir, f"{fold}_test.csv")
        train_csv = os.path.join(fold_dir, f"{fold}_train.csv")
        v13_fold_tag = fold
        if not os.path.exists(test_csv) or not os.path.exists(train_csv):
            sys.exit(f"[FATAL] folds_v33_vix_expanding/{fold}_* missing")

    os.makedirs(args.result_dir, exist_ok=True)

    flow_mode = args.flow_mode
    # Figure suffix: include fold name when not pilot, plus flow_mode suffix.
    mode_suffix = "" if flow_mode == "uncond" else f"_{flow_mode}"
    fold_suffix = "" if fold == "pilot" else f"_{fold}"
    fig_suffix = f"{fold_suffix}{mode_suffix}"

    print("=" * 78)
    print(" Sensitivity Analysis — Variant 13 (paper #2: base + metab_13w cond)")
    print("=" * 78)
    print(f"  device      : {device}")
    print(f"  fold        : {fold}   (v13 fold_tag = '{v13_fold_tag}')")
    print(f"  test_csv    : {test_csv}")
    print(f"  result_dir  : {args.result_dir}")
    print(f"  flow_mode   : {flow_mode}    (figures saved with suffix '{fig_suffix}')")
    print(f"  n_sim/seed  : {args.n_sim_per_seed}  → 5 seeds × {args.n_sim_per_seed}"
          f" = {5 * args.n_sim_per_seed} paths/panel")

    # --- 1. Origin selection ---
    print("\n[1] Select calm + stress origins from test set...")
    origins = select_origins(test_csv, n_calm=1, n_stress=1)
    for o in origins:
        print(f"    {o['kind']:8s}  origin_date={o['origin_date'].date()}  "
              f"future_realized_std={o['future_realized_std']:.4f}  "
              f"future_cum_return={o['future_cum_return']:+.4f}")

    # --- 2. Load v13 models ---
    # sel_filter: ckpt tag 의 sel 부분 매칭. trainer 가 sel_tag='_msel' (leading _ 포함)
    # 으로 저장하므로 우리 pattern 의 '3m_{sel_filter}_v13' 에선 leading _ 빼야 함.
    sel_filter = {"any": "*sel*", "mse": "msel", "ic": "isel"}[args.loss_mode]
    print(f"\n[2] Load Variant 13 (paper #2) — 5 seed ckpts...  (loss-mode filter: {args.loss_mode}, glob: {sel_filter})")
    models = load_v13_models(args.result_dir, fold_tag=v13_fold_tag, device=device,
                              sel_filter=sel_filter)
    cols_cond       = models[0]["cond_cols"]
    cols_target     = models[0]["target_cols"]
    stats_cond      = models[0]["stats_cond"]
    mask_future_ch  = models[0]["mask_future_ch"]
    print(f"    cond_cols      : {cols_cond}")
    print(f"    target_cols    : {cols_target}")
    print(f"    mask_future_ch : {mask_future_ch}  (tbill_wr unmasked, others = 0 in future)")

    df_test = pd.read_csv(test_csv, parse_dates=["date"])

    # --- 3. Sensitivity scan ---
    print("\n[3] Sensitivity scan (OAT level shift on tbill_wr and metab_13w)...")
    tbill_weekly_step  = TBILL_ANNUAL_PP_STEP / 100.0 / 52.0     # annual %p → weekly decimal
    tbill_weekly_range = TBILL_ANNUAL_PP_RANGE / 100.0 / 52.0
    tbill_deltas = np.arange(-tbill_weekly_range,
                              tbill_weekly_range + tbill_weekly_step / 2,
                              tbill_weekly_step).astype(np.float64)
    metab_deltas = np.arange(-METAB_FRACTION_RANGE,
                              METAB_FRACTION_RANGE + METAB_FRACTION_STEP / 2,
                              METAB_FRACTION_STEP).astype(np.float64)

    print(f"    tbill_wr   deltas: n={len(tbill_deltas)},  "
          f"range = [{tbill_deltas.min():.6f}, {tbill_deltas.max():.6f}] weekly  "
          f"(= ±{TBILL_ANNUAL_PP_RANGE:.1f}%p annual, step {TBILL_ANNUAL_PP_STEP:.2f}%p)")
    print(f"    metab_13w  deltas: n={len(metab_deltas)},  "
          f"range = [{metab_deltas.min():+.4f}, {metab_deltas.max():+.4f}]  "
          f"(= ±{METAB_FRACTION_RANGE*100:.1f}%p 13w cum, step {METAB_FRACTION_STEP*100:.2f}%p)")

    scan_results     = {}
    baseline_sigmas  = {}
    actual_realized  = {}

    for o in origins:
        ok = o["kind"]
        # baseline σ̂ (Δ=0)
        c, t = build_window_tensors(df_test, o["origin_idx"], cols_cond, cols_target,
                                     stats_cond, mask_future_ch)
        base_s = predict_sigma_5seed(models, c, t, device)[0]   # (n_seeds,)
        baseline_sigmas[ok] = base_s
        actual_realized[ok] = o["future_realized_std"]
        print(f"    [{ok}] baseline σ̂ (5-seed) = {base_s} (mean {base_s.mean():.4f}); "
              f"actual realized std = {o['future_realized_std']:.4f}")

        for col, deltas in [("tbill_wr", tbill_deltas),
                            ("metab_13w", metab_deltas)]:
            sigmas = sensitivity_scan(models, df_test, o["origin_idx"], col, deltas,
                                       cols_cond, cols_target, stats_cond, mask_future_ch,
                                       device)
            scan_results[(ok, col)] = dict(
                deltas=deltas, sigmas=sigmas,
                origin_date=o["origin_date"],
                actual_realized_std=o["future_realized_std"],
            )
            # positive-sensitivity check at endpoint
            s_lo = sigmas[0].mean()
            s_hi = sigmas[-1].mean()
            slope = (s_hi - s_lo) / (deltas[-1] - deltas[0] + 1e-12)
            print(f"      [{ok}] {col}: σ̂(Δmin)={s_lo:.4f}  σ̂(Δmax)={s_hi:.4f}  "
                  f"slope={slope:+.3f} / unit-Δ")

    print("\n[4] Figure 1 — Sensitivity curves")
    fig1_path = os.path.join(args.result_dir, f"sensitivity_v13_curves{fig_suffix}.png")
    make_sensitivity_figure(origins, scan_results, baseline_sigmas, fig1_path)

    # --- 5. Model B (Flow 1D NSF) load / train ---
    if flow_mode == "uncond":
        print("\n[5] Model B — UNCONDITIONAL 1D Neural Spline Flow on within-window ε")
        eps_train, _ = extract_eps_train(train_csv)
        flow_path = os.path.join(args.result_dir, "scenario_3m_flow_1d.pt")
        flow = train_flow(eps_train, flow_path,
                          num_layers=FLOW_LAYERS, num_bins=FLOW_BINS,
                          tail_bound=FLOW_TAIL, epochs=args.flow_epochs, device=device)
    elif flow_mode == "global":
        print("\n[5] Model B — UNCONDITIONAL 1D NSF on globally-standardized ε")
        flow_path = os.path.join(args.result_dir, args.global_flow_ckpt)
        if not os.path.exists(flow_path):
            sys.exit(f"[FATAL] global flow ckpt not found: {flow_path}\n"
                     f"  → 먼저 python colab/dual_3ch/train_flow_b_global.py 실행")
        print(f"    [load] {flow_path}")
        state = torch.load(flow_path, map_location=device, weights_only=False)

        if isinstance(state, dict) and "model_state" in state and "meta" in state:
            # New format with metadata — build flow exactly per saved architecture
            meta = state["meta"]
            base_kind  = meta.get("base_kind",  "normal")
            df         = meta.get("df",         5.0)
            num_layers = meta.get("num_layers", GLOBAL_FLOW_LAYERS)
            num_bins   = meta.get("num_bins",   GLOBAL_FLOW_BINS)
            tail_bound = meta.get("tail_bound", GLOBAL_FLOW_TAIL)
            alpha_init = meta.get("alpha_init", -5.0) if base_kind == "skew_t" else -5.0
            flow = build_flow_global(base_kind, df, num_layers, num_bins, tail_bound,
                                      alpha_init=alpha_init).to(device)
            flow.load_state_dict(state["model_state"])
            flow.eval()
            if base_kind == "skew_t":
                a_final = meta.get("alpha_final", "?")
                base_desc = f"Skew-t(α_final={a_final}, df={df})"
            elif base_kind == "student_t":
                base_desc = f"Student-t(df={df})"
            else:
                base_desc = "Normal"
            print(f"    loaded: base={base_desc}, layers={num_layers}, bins={num_bins}, "
                  f"tail={tail_bound}")
        else:
            # Legacy plain state_dict — assume Normal-base v2 architecture (layers=6/bins=16/tail=10)
            flow = build_flow_1d(num_layers=GLOBAL_FLOW_LAYERS, num_bins=GLOBAL_FLOW_BINS,
                                  tail_bound=GLOBAL_FLOW_TAIL).to(device)
            sd = state["model_state"] if (isinstance(state, dict) and "model_state" in state) else state
            flow.load_state_dict(sd)
            flow.eval()
            print(f"    loaded (legacy plain state_dict): Normal-base v2 — "
                  f"layers={GLOBAL_FLOW_LAYERS}, bins={GLOBAL_FLOW_BINS}, tail={GLOBAL_FLOW_TAIL}")
    else:  # cond
        print("\n[5] Model B — CONDITIONAL 1D Neural Spline Flow on (ε, macro context) pairs")
        flow_path = os.path.join(args.result_dir, args.cond_flow_ckpt)
        if os.path.exists(flow_path):
            print(f"    [load] {flow_path}")
            state = torch.load(flow_path, map_location=device, weights_only=False)
            d_ctx = state.get("context_features", len(cols_cond)) if isinstance(state, dict) else len(cols_cond)
            num_layers     = state.get("num_layers", FLOW_LAYERS)     if isinstance(state, dict) else FLOW_LAYERS
            num_bins       = state.get("num_bins",   FLOW_BINS)       if isinstance(state, dict) else FLOW_BINS
            tail_bound     = state.get("tail_bound", FLOW_TAIL)       if isinstance(state, dict) else FLOW_TAIL
            hidden_features = state.get("hidden_features", 32)         if isinstance(state, dict) else 32
            flow = build_conditional_flow_1d(
                context_features=d_ctx, num_layers=num_layers, num_bins=num_bins,
                tail_bound=tail_bound, hidden_features=hidden_features,
            ).to(device)
            sd = state["model_state"] if (isinstance(state, dict) and "model_state" in state) else state
            flow.load_state_dict(sd)
            flow.eval()
            print(f"    loaded: layers={num_layers}, bins={num_bins}, tail={tail_bound}, "
                  f"hidden={hidden_features}, ctx_features={d_ctx}")
        else:
            print(f"    [train] no ckpt at {flow_path} — training now (epochs={args.flow_epochs})")
            eps_t, ctx_t, _ = extract_eps_with_context(
                train_csv, cols_cond=cols_cond, stats_cond=stats_cond,
            )
            flow = train_conditional_flow(
                eps_t, ctx_t, flow_path,
                num_layers=FLOW_LAYERS, num_bins=FLOW_BINS, tail_bound=FLOW_TAIL,
                hidden_features=32, epochs=args.flow_epochs, batch=256, lr=5e-4,
                device=device,
            )

    # --- 6. Fan chart + histogram preparation ---
    print(f"\n[6] Fan-chart scenario generation — n_sim_per_seed={args.n_sim_per_seed} × 5 "
          f"= {5 * args.n_sim_per_seed} paths/panel")
    panels = {}
    for o in origins:
        ok = o["kind"]
        # actual realized 13-step path (raw sp_return)
        idx = o["origin_idx"]
        future_actual = df_test["sp_return"].iloc[idx + PAST_LEN : idx + L].values.astype(np.float32)

        for pp in FAN_TBILL_ANNUAL_PP:
            pp_label = f"+{pp:.1f}pp"
            delta_weekly = pp / 100.0 / 52.0
            # σ̂ at this perturbation
            c, t = build_window_tensors(df_test, idx, cols_cond, cols_target,
                                         stats_cond, mask_future_ch,
                                         perturb_col="tbill_wr",
                                         perturb_delta=delta_weekly)
            sig_5seed = predict_sigma_5seed(models, c, t, device)[0]   # (5,)
            sig_mean = float(sig_5seed.mean())

            # Context vector for conditional flow — perturbed tbill at origin's last past row.
            # For uncond / global modes this is unused (context_vec=None passed below).
            if flow_mode == "cond":
                ctx_vec = extract_context_vec(df_test, idx, cols_cond, stats_cond,
                                              perturb_col="tbill_wr",
                                              perturb_delta=delta_weekly)
            else:
                ctx_vec = None

            flow_paths  = generate_paths_5seed(sig_5seed, flow,
                                               n_sim_per_seed=args.n_sim_per_seed,
                                               future_len=FUTURE_LEN, device=device,
                                               use_gaussian=False, rng_base=1000,
                                               context_vec=ctx_vec)
            gauss_paths = generate_paths_5seed(sig_5seed, flow,
                                               n_sim_per_seed=args.n_sim_per_seed,
                                               future_len=FUTURE_LEN, device=device,
                                               use_gaussian=True, rng_base=2000,
                                               context_vec=None)
            panels[(ok, pp_label)] = dict(
                flow_paths=flow_paths,
                gauss_paths=gauss_paths,
                actual_path=future_actual,
                sigma_used=sig_mean,
                sigmas_5seed=sig_5seed,
                origin_date=o["origin_date"],
                context_vec=ctx_vec,
            )
            print(f"    [{ok}] tbill {pp_label}  σ̂(mean)={sig_mean:.4f}  "
                  f"flow_path_shape={flow_paths.shape}  "
                  f"ctx={'(none, uncond)' if ctx_vec is None else f'{ctx_vec.tolist()}'}")

    # --- 7. Figure 2 — Fan chart ---
    print(f"\n[7] Figure 2 — Fan chart (Flow [{flow_mode}] + Gaussian overlay + actual)")
    fig2_path = os.path.join(args.result_dir, f"sensitivity_v13_fanchart{fig_suffix}.png")
    make_fanchart_figure(panels, fig2_path)

    # --- 8. Figure 3 — Histogram of week-13 cumulative ---
    print(f"\n[8] Figure 3 — Week-13 cumulative return histogram (Flow [{flow_mode}] vs Gaussian vs actual)")
    fig3_path = os.path.join(args.result_dir, f"sensitivity_v13_histogram{fig_suffix}.png")
    hist_summary_df = make_histogram_figure(panels, fig3_path)

    # --- 9. Summary CSV (sensitivity + histogram stats unified) ---
    summary_rows = []
    for o in origins:
        ok = o["kind"]
        base_s = baseline_sigmas[ok]
        actual_s = actual_realized[ok]
        for col, deltas_key in [("tbill_wr", "tbill_wr"),
                                ("metab_13w", "metab_13w")]:
            entry = scan_results[(ok, col)]
            deltas = entry["deltas"]
            sigmas = entry["sigmas"]
            s_lo = sigmas[0].mean()
            s_hi = sigmas[-1].mean()
            slope = (s_hi - s_lo) / (deltas[-1] - deltas[0] + 1e-12)
            sens_idx = (s_hi - s_lo) / (base_s.mean() + 1e-12)
            summary_rows.append(dict(
                origin_kind=ok,
                origin_date=str(o["origin_date"].date()),
                actual_realized_std=actual_s,
                baseline_sigma_5seed_mean=float(base_s.mean()),
                perturb_var=col,
                delta_min=float(deltas.min()),
                delta_max=float(deltas.max()),
                sigma_at_delta_min=float(s_lo),
                sigma_at_delta_max=float(s_hi),
                slope_per_unit=float(slope),
                sensitivity_index_rel=float(sens_idx),
                positive_sensitivity=bool(s_hi > s_lo),
            ))
    sens_df = pd.DataFrame(summary_rows)
    sens_csv_path = os.path.join(args.result_dir, f"sensitivity_v13_summary{fig_suffix}.csv")
    sens_df.to_csv(sens_csv_path, index=False)
    print(f"\n  saved sensitivity summary: {sens_csv_path}")

    hist_csv_path = os.path.join(args.result_dir, f"sensitivity_v13_histogram_stats{fig_suffix}.csv")
    hist_summary_df.to_csv(hist_csv_path, index=False)
    print(f"  saved histogram stats: {hist_csv_path}")

    # --- 10. Console final summary ---
    print(f"\n{'='*78}")
    print(" POSITIVE SENSITIVITY VERIFICATION (slope of σ̂ over Δ range)")
    print(f"{'='*78}")
    print(f"  {'origin':8s}  {'var':22s}  "
          f"{'σ̂(Δmin)':>10s}  {'σ̂(Δmax)':>10s}  {'slope':>14s}  {'positive?':>10s}")
    print("-" * 78)
    for r in summary_rows:
        print(f"  {r['origin_kind']:8s}  {r['perturb_var']:22s}  "
              f"{r['sigma_at_delta_min']:>10.4f}  {r['sigma_at_delta_max']:>10.4f}  "
              f"{r['slope_per_unit']:>+14.3f}  "
              f"{'✓' if r['positive_sensitivity'] else '✗':>10s}")
    print("=" * 78)
    print(f"\n  Figures: \n"
          f"    {fig1_path}\n"
          f"    {fig2_path}\n"
          f"    {fig3_path}")


if __name__ == "__main__":
    main()
