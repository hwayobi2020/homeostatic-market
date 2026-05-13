"""HAR-RV σ̂ × Cond Flow ε ~ Skew-t | C_t — scenario generation + fan chart + EMD/CVaR.

Paper main thesis (regime-conditional tail):
  σ̂(t)        = HAR-RV OLS prediction (=>volume dial, magnitude estimator)
  ε_{p,w}|C_t ~ Conditional NSF (Skew-t base, α learnable, K=7 cond features)
  r(t,p,w)    = r̄_train + σ̂(t) × ε_{p,w}     (Azzalini-Capitanio + HAR composite)

C_t = origin-row macro vector, z-scored on TRAIN cond_mean/std (fold-isolated):
  [tbill_wr, m2_13w_cum_lag, ads_lag, cpi_13w_cum_lag, sp_std_13w,
   wti_wr, sp_log_std_13w]

Required inputs (must run BEFORE this script):
  result/har_rv_{fold}_test_predictions.csv   ← train_har_rv.py --fold {fold}
  result/scenario_3m_flow_1d_cond_{fold}_skewt_df5.pt
      ← train_flow_cond.py --train-csv data/folds_v33_vix_expanding/{fold}_train.csv
                            --save-name scenario_3m_flow_1d_cond_{fold}_skewt_df5.pt

Output (per fold):
  result/sensitivity_har_rv_cond_{fold}_fanchart.png
  result/sensitivity_har_rv_cond_{fold}_histogram.png
  result/sensitivity_har_rv_cond_{fold}_summary.json   (EMD, CVaR, moments)

Usage:
  python sensitivity_har_rv.py --fold F1
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
sys.path.insert(0, HERE)

# Cond Flow build (재현) — 같은 패키지 모듈
from train_flow_cond import build_cond_flow, COND_FEATURES_MACRO   # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RESULT = os.path.join(HERE, "result")
FUTURE_LEN = 13
PAST_LEN   = 52     # train_har_rv.py / train_vol_pilot_3m_mtl.py 와 동일 origin alignment


# =====================================================================
# Load conditional flow + meta
# =====================================================================

def load_cond_flow(fold, device, ckpt_name=None):
    if ckpt_name is None:
        ckpt_name = f"scenario_3m_flow_1d_cond_{fold}_skewt_df5.pt"
    ckpt_path = os.path.join(RESULT, ckpt_name)
    if not os.path.exists(ckpt_path):
        sys.exit(f"[FATAL] Cond Flow ckpt 없음: {ckpt_path}\n"
                 f"  → 먼저: python train_flow_cond.py "
                 f"--train-csv data/folds_v33_vix_expanding/{fold}_train.csv "
                 f"--save-name {ckpt_name}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = state["meta"]
    # K = source-of-truth = len(cond_features). 일부 ckpt 는 meta["context_features"] 가
    # 잘못 저장됐을 수 있음 (학습 코드 버그) → cond_features list 의 길이를 신뢰.
    K_meta = len(meta.get("cond_features", [])) or meta["context_features"]
    flow = build_cond_flow(
        base_kind=meta["base_kind"],
        df=meta["df"],
        num_layers=meta["num_layers"],
        num_bins=meta["num_bins"],
        tail_bound=meta["tail_bound"],
        hidden_features=meta["hidden_features"],
        num_blocks=meta.get("num_blocks", 2),
        context_features=K_meta,
        alpha_init=meta.get("alpha_init", -5.0),
    ).to(device)
    flow.load_state_dict(state["model_state"])
    flow.eval()
    a_str = ""
    if meta["base_kind"] == "skew_t":
        a_str = f", α_final={meta.get('alpha_final', 'N/A')}"
    print(f"  Cond Flow loaded: base={meta['base_kind']}{a_str}, "
          f"K_cond={meta['context_features']}, layers={meta['num_layers']}")
    print(f"    cond_features = {meta['cond_features']}")
    print(f"    r_mean_train={meta['r_mean_train']:+.6f}, r_std_train={meta['r_std_train']:.6f}")
    return flow, meta


# =====================================================================
# Build origin-aligned cond + actual future paths from test CSV
# =====================================================================

def build_origin_table(test_csv, har_df, cond_features, cond_mean, cond_std,
                       oof_lookups=None):
    """For each HAR test prediction origin date, locate the matching test row
    and extract (normalized_cond_vector, future_sp_return_path).

    Origin in test CSV: HAR date == last past row date == csv row index t+PAST_LEN-1.
    Future path:        test_csv["sp_return"].iloc[t+PAST_LEN : t+L]    (FUTURE_LEN steps)

    cond_features 가 9-dim 일 경우 (paper main + OOF):
        macro 7  : test.csv 에서 직접 (origin row)
        OOF cond : oof_lookups[col_name] = {date → value}    에서 매칭

    oof_lookups : dict {col_name: {Timestamp: value}}  (test 원천 OOS prediction)
    """
    df = pd.read_csv(test_csv, parse_dates=["date"])
    test_dates = df["date"].values
    n_csv = len(df)
    L = PAST_LEN + FUTURE_LEN
    if oof_lookups is None:
        oof_lookups = {}

    # macro vs oof split
    macro_cols = [c for c in cond_features if c not in oof_lookups]
    oof_cols   = [c for c in cond_features if c in oof_lookups]
    cond_mean = np.asarray(cond_mean, dtype=np.float64)
    cond_std  = np.asarray(cond_std,  dtype=np.float64)
    cond_std_safe = np.where(cond_std < 1e-12, 1.0, cond_std)
    # Index map for cond_features → (cond_mean / cond_std) order matches cond_features order
    feat_idx = {c: i for i, c in enumerate(cond_features)}

    cond_arr = []      # (n_origin, K)  z-scored, in cond_features order
    actual_paths = []  # (n_origin, FUTURE_LEN)
    valid = []

    har_dates = har_df["date"].values
    for origin_date in har_dates:
        idx = np.where(test_dates == np.datetime64(origin_date))[0]
        bad_row = (len(idx) == 0)
        if bad_row:
            cond_arr.append(np.full(len(cond_features), np.nan))
            actual_paths.append(np.full(FUTURE_LEN, np.nan))
            valid.append(False)
            continue
        last_past = int(idx[0])
        t = last_past - (PAST_LEN - 1)
        # Build raw cond in cond_features order
        c_raw = np.zeros(len(cond_features), dtype=np.float64)
        ok = True
        for c in macro_cols:
            v = df[c].iloc[last_past]
            if pd.isna(v):
                ok = False; break
            c_raw[feat_idx[c]] = float(v)
        if ok:
            origin_ts = pd.Timestamp(origin_date)
            for c in oof_cols:
                v = oof_lookups[c].get(origin_ts)
                if v is None or pd.isna(v):
                    ok = False; break
                c_raw[feat_idx[c]] = float(v)
        c_norm = (c_raw - cond_mean) / cond_std_safe
        # Future path
        if not ok:
            cond_arr.append(c_norm)
            actual_paths.append(np.full(FUTURE_LEN, np.nan))
            valid.append(False)
            continue
        fut_idx_end = t + L
        if fut_idx_end > n_csv:
            cond_arr.append(c_norm)
            actual_paths.append(np.full(FUTURE_LEN, np.nan))
            valid.append(False)
            continue
        fut = df["sp_return"].iloc[t + PAST_LEN : fut_idx_end].values
        if np.any(np.isnan(fut)):
            cond_arr.append(c_norm)
            actual_paths.append(np.full(FUTURE_LEN, np.nan))
            valid.append(False)
        else:
            cond_arr.append(c_norm)
            actual_paths.append(fut)
            valid.append(True)

    return np.stack(cond_arr), np.stack(actual_paths), np.array(valid, dtype=bool)


# =====================================================================
# Generate conditional scenarios
# =====================================================================

@torch.no_grad()
def generate_cond_scenarios(sigma_log, cond_norm, flow, r_mean_train,
                            n_sim, device, chunk=64):
    """For each origin t (n_origin total), sample n_sim × FUTURE_LEN ε from
    flow conditioned on C_t, then convert to sp_return.

    r(t, p, w) = r̄_train + σ̂(t) × ε_{p, w}     (Gemini formula)

    Args:
      sigma_log  : (n_origin,) log σ̂(t)
      cond_norm  : (n_origin, K) z-scored cond
      flow       : Conditional NSF
      r_mean_train, n_sim, device

    Returns:
      sp_returns : (n_origin, n_sim, FUTURE_LEN)
    """
    n_origin = len(sigma_log)
    sigma = np.exp(sigma_log)                          # (n_origin,) per-week std
    n_samples_per_origin = n_sim * FUTURE_LEN
    out = np.zeros((n_origin, n_sim, FUTURE_LEN), dtype=np.float32)

    # Process origins in chunks to bound memory
    for s in range(0, n_origin, chunk):
        e = min(s + chunk, n_origin)
        b = e - s
        ctx = torch.from_numpy(cond_norm[s:e].astype(np.float32)).to(device)  # (b, K)
        # flow.sample(n, context=ctx) returns (b, n, 1)
        eps_samples = flow.sample(n_samples_per_origin, context=ctx)
        eps_np = eps_samples.cpu().numpy().reshape(b, n_sim, FUTURE_LEN)
        sig_b = sigma[s:e][:, None, None]              # (b, 1, 1)
        # r = r̄_train + σ̂(t) × ε
        out[s:e] = (r_mean_train + sig_b * eps_np).astype(np.float32)
    return out


# =====================================================================
# Metrics
# =====================================================================

def compute_cvar(returns_flat, alpha=0.05):
    sorted_r = np.sort(returns_flat)
    n_tail = max(1, int(alpha * len(sorted_r)))
    return float(sorted_r[:n_tail].mean())


def compute_var(returns_flat, alpha=0.05):
    return float(np.quantile(returns_flat, alpha))


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


def coverage(actual_flat, sim_flat, alpha_low, alpha_high):
    lo_q = float(np.quantile(sim_flat, alpha_low))
    hi_q = float(np.quantile(sim_flat, alpha_high))
    cov = float(((actual_flat >= lo_q) & (actual_flat <= hi_q)).mean())
    return cov, lo_q, hi_q


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
    actual_med = np.median(actual_cum, axis=0)
    actual_p10 = np.percentile(actual_cum, 10, axis=0)
    actual_p90 = np.percentile(actual_cum, 90, axis=0)
    ax.plot(x, actual_med, color="red", lw=2, ls="--", label="Actual median (test)")
    ax.plot(x, actual_p10, color="red", lw=1, ls=":", label="Actual 10/90 pctile")
    ax.plot(x, actual_p90, color="red", lw=1, ls=":")
    ax.set_xlabel("week (future)")
    ax.set_ylabel("cumulative sp_return")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    saved fanchart: {save_path}")


def plot_histogram(actual_flat, sim_flat, title, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    lo = float(min(actual_flat.min(), np.quantile(sim_flat, 0.001)))
    hi = float(max(actual_flat.max(), np.quantile(sim_flat, 0.999)))
    bins = np.linspace(lo, hi, 100)
    ax.hist(sim_flat, bins=bins, density=True, alpha=0.4, color="C0",
            label=f"Flow scenarios (n={len(sim_flat):,})")
    ax.hist(actual_flat, bins=bins, density=True, alpha=0.5, color="red",
            label=f"Actual sp_return (n={len(actual_flat):,})")
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
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="HAR-RV σ̂ × Cond Flow ε | C_t scenarios (regime-conditional tail-risk)")
    ap.add_argument("--fold", default="F1")
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--n-sim", type=int, default=1000, help="paths per origin")
    ap.add_argument("--chunk", type=int, default=64, help="origins processed per GPU batch")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ckpt-name", default=None,
                    help="Cond Flow ckpt filename (default: scenario_3m_flow_1d_cond_{fold}_skewt_df5.pt)")
    ap.add_argument("--v14-test-pred-npz", default=None,
                    help="v14 MTL test_preds.npz path (required only if Flow ckpt cond_features "
                         "contains 'v14_oof_log_std'). y_pred_log_std[t] aligns with HAR origin t in order.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f" HAR-RV σ̂ × Cond Flow ε|C_t | --fold {args.fold} | "
          f"n_sim={args.n_sim} per origin | device={device}")
    print("=" * 78)

    # 1) Load HAR-RV test predictions
    print(f"\n[1] Load HAR-RV test predictions")
    har_csv = os.path.join(RESULT, f"har_rv_{args.fold}_test_predictions.csv")
    if not os.path.exists(har_csv):
        sys.exit(f"[FATAL] HAR-RV test preds 없음: {har_csv}\n"
                 f"  → 먼저: python train_har_rv.py --fold {args.fold}")
    har_df = pd.read_csv(har_csv, parse_dates=["date"])
    sigma_log_all = har_df["pred_log_std"].values
    actual_log_all = har_df["actual_log_std"].values
    print(f"    HAR origins n={len(sigma_log_all)}, "
          f"σ̂ range [{np.exp(sigma_log_all).min():.4f}, {np.exp(sigma_log_all).max():.4f}], "
          f"mean σ̂={np.exp(sigma_log_all).mean():.4f}")

    # 2) Load Cond Flow + meta
    print(f"\n[2] Load Cond Flow ckpt")
    flow, meta = load_cond_flow(args.fold, device, ckpt_name=args.ckpt_name)
    cond_features = meta["cond_features"]
    cond_mean = meta["cond_mean"]
    cond_std  = meta["cond_std"]
    r_mean_train = meta["r_mean_train"]
    print(f"    cond_features (K={len(cond_features)}): {cond_features}")

    # Sanity: macro 7 prefix must match
    macro_in_ckpt = [c for c in cond_features if c in COND_FEATURES_MACRO]
    if macro_in_ckpt != COND_FEATURES_MACRO:
        print(f"  ⚠ WARN: macro cond order mismatch — ckpt {macro_in_ckpt} vs "
              f"COND_FEATURES_MACRO {COND_FEATURES_MACRO}")

    # Build OOF lookup dicts for test-side cond columns
    oof_lookups = {}
    # har_oof_log_std at test origin = HAR test prediction (already OOS)
    if "har_oof_log_std" in cond_features:
        har_csv2 = os.path.join(RESULT, f"har_rv_{args.fold}_test_predictions.csv")
        h = pd.read_csv(har_csv2, parse_dates=["date"])
        oof_lookups["har_oof_log_std"] = dict(zip(pd.to_datetime(h["date"]), h["pred_log_std"]))
        print(f"    OOF cond 'har_oof_log_std' ← {har_csv2}  (n={len(h)})")
    # v14_oof_log_std at test origin = v14 MTL test prediction (npz), aligned by order
    if "v14_oof_log_std" in cond_features:
        if args.v14_test_pred_npz is None:
            sys.exit(f"[FATAL] ckpt cond_features 에 'v14_oof_log_std' 가 있는데 "
                     f"--v14-test-pred-npz 가 지정되지 않았습니다.")
        v14_path = args.v14_test_pred_npz
        if not os.path.isabs(v14_path):
            v14_path = os.path.join(RESULT, v14_path)
        if not os.path.exists(v14_path):
            sys.exit(f"[FATAL] v14 test_preds.npz 없음: {v14_path}")
        v14_npz = np.load(v14_path, allow_pickle=False)
        v14_preds = np.asarray(v14_npz["y_pred_log_std"], dtype=np.float64)
        # Align by HAR test origin order — both follow csv origin order (PAST_LEN-1 offset)
        har_dates_test = pd.to_datetime(har_df["date"].values)
        if len(v14_preds) != len(har_dates_test):
            print(f"  ⚠ WARN: v14 preds n={len(v14_preds)} != HAR origin n={len(har_dates_test)}; "
                  f"잘릴 수 있음.")
        n_match = min(len(v14_preds), len(har_dates_test))
        oof_lookups["v14_oof_log_std"] = dict(zip(har_dates_test[:n_match], v14_preds[:n_match]))
        print(f"    OOF cond 'v14_oof_log_std' ← {v14_path}  (n={n_match})")

    # 3) Build origin-aligned cond vectors + actual future paths
    test_csv = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    if not os.path.exists(test_csv):
        sys.exit(f"[FATAL] test CSV 없음: {test_csv}")
    print(f"\n[3] Align origins → cond + actual future path  ({test_csv})")
    cond_norm, actual_paths, valid_mask = build_origin_table(
        test_csv, har_df, cond_features, cond_mean, cond_std, oof_lookups=oof_lookups,
    )
    n_valid = int(valid_mask.sum())
    print(f"    valid origins: {n_valid}/{len(har_df)}")
    sigma_log   = sigma_log_all[valid_mask]
    actual_log  = actual_log_all[valid_mask]
    cond_norm   = cond_norm[valid_mask]
    actual_paths = actual_paths[valid_mask]
    har_dates_valid = har_df["date"].values[valid_mask]
    if n_valid < 5:
        sys.exit(f"[FATAL] valid origins {n_valid} < 5 — cannot generate scenarios.")

    # 4) Generate conditional scenarios
    print(f"\n[4] Sample Cond Flow: {args.n_sim} paths × {FUTURE_LEN}w per origin")
    sim_paths = generate_cond_scenarios(
        sigma_log, cond_norm, flow, r_mean_train,
        n_sim=args.n_sim, device=device, chunk=args.chunk,
    )
    print(f"    sim_paths shape: {sim_paths.shape}  "
          f"(n_origin × n_sim × {FUTURE_LEN})  "
          f"total samples = {sim_paths.size:,}")

    # 5) Metrics
    actual_flat = actual_paths.ravel()
    sim_flat = sim_paths.ravel()
    var5_actual  = compute_var(actual_flat, 0.05)
    var5_sim     = compute_var(sim_flat,    0.05)
    var1_actual  = compute_var(actual_flat, 0.01)
    var1_sim     = compute_var(sim_flat,    0.01)
    cvar5_actual = compute_cvar(actual_flat, 0.05)
    cvar5_sim    = compute_cvar(sim_flat,    0.05)
    cvar1_actual = compute_cvar(actual_flat, 0.01)
    cvar1_sim    = compute_cvar(sim_flat,    0.01)
    emd_pooled = compute_emd_1d(actual_flat, sim_flat, n_bins=200)

    cov50, lo50, hi50 = coverage(actual_flat, sim_flat, 0.25, 0.75)
    cov80, lo80, hi80 = coverage(actual_flat, sim_flat, 0.10, 0.90)
    cov95, lo95, hi95 = coverage(actual_flat, sim_flat, 0.025, 0.975)

    def moments(x):
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        sk = float(pd.Series(x).skew())
        kt = float(pd.Series(x).kurt())
        return m, s, sk, kt

    a_m, a_s, a_sk, a_kt = moments(actual_flat)
    s_m, s_s, s_sk, s_kt = moments(sim_flat)

    print(f"\n[5] Metrics — fold {args.fold}")
    print(f"    moments              {'actual':>12s}  {'sim':>12s}  {'diff':>12s}")
    print(f"      mean               {a_m:+12.5f}  {s_m:+12.5f}  {s_m - a_m:+12.5f}")
    print(f"      std                {a_s:12.5f}  {s_s:12.5f}  {s_s - a_s:+12.5f}")
    print(f"      skew               {a_sk:+12.3f}  {s_sk:+12.3f}  {s_sk - a_sk:+12.3f}")
    print(f"      ex_kurt            {a_kt:+12.3f}  {s_kt:+12.3f}  {s_kt - a_kt:+12.3f}")
    print(f"    VaR 5%               {var5_actual:+12.5f}  {var5_sim:+12.5f}  "
          f"{var5_sim - var5_actual:+12.5f}")
    print(f"    VaR 1%               {var1_actual:+12.5f}  {var1_sim:+12.5f}  "
          f"{var1_sim - var1_actual:+12.5f}")
    print(f"    CVaR 5%              {cvar5_actual:+12.5f}  {cvar5_sim:+12.5f}  "
          f"{cvar5_sim - cvar5_actual:+12.5f}")
    print(f"    CVaR 1%              {cvar1_actual:+12.5f}  {cvar1_sim:+12.5f}  "
          f"{cvar1_sim - cvar1_actual:+12.5f}")
    print(f"    EMD (pooled hist)    = {emd_pooled:.6f}")
    print(f"    Coverage             50%={cov50:.3f}  80%={cov80:.3f}  95%={cov95:.3f}")

    # 6) Plots — output prefix derived from ckpt name (so cond vs cond_oof 분리)
    ckpt_used = args.ckpt_name or f"scenario_3m_flow_1d_cond_{args.fold}_skewt_df5.pt"
    out_prefix = "sensitivity_har_rv_" + (
        ckpt_used.replace("scenario_3m_flow_1d_", "").replace(".pt", "")
    )   # e.g. "sensitivity_har_rv_cond_F1_skewt_df5" or "..._cond_oof_F1_skewt_df5"
    print(f"\n[6] Plots  (output prefix: {out_prefix})")
    title_base = (f"HAR-RV σ̂ × Cond Flow ε|C — fold {args.fold} K={len(cond_features)} "
                  f"(n_origin={n_valid}, n_sim={args.n_sim})")
    fan_path = os.path.join(RESULT, f"{out_prefix}_fanchart.png")
    hist_path = os.path.join(RESULT, f"{out_prefix}_histogram.png")
    plot_fanchart(actual_paths, sim_paths, title_base + "\nfan chart (cumulative)", fan_path)
    plot_histogram(actual_flat, sim_flat, title_base + "\nweekly sp_return histogram", hist_path)

    # 7) Save summary
    summary = dict(
        fold=args.fold,
        model=("HAR-RV (sp_log_std_13w + sp_std_13w, OLS) σ̂ × "
               "Cond NSF (Skew-t df=5, α learnable, K=7 macro) ε|C_t"),
        cond_features=cond_features,
        n_origin=int(n_valid),
        n_sim_per_origin=int(args.n_sim),
        moments_actual=dict(mean=a_m, std=a_s, skew=a_sk, ex_kurt=a_kt),
        moments_sim   =dict(mean=s_m, std=s_s, skew=s_sk, ex_kurt=s_kt),
        var_5pct_actual=var5_actual, var_5pct_sim=var5_sim,
        var_5pct_diff=var5_sim - var5_actual,
        var_1pct_actual=var1_actual, var_1pct_sim=var1_sim,
        var_1pct_diff=var1_sim - var1_actual,
        cvar_5pct_actual=cvar5_actual, cvar_5pct_sim=cvar5_sim,
        cvar_5pct_diff=cvar5_sim - cvar5_actual,
        cvar_1pct_actual=cvar1_actual, cvar_1pct_sim=cvar1_sim,
        cvar_1pct_diff=cvar1_sim - cvar1_actual,
        emd_pooled=emd_pooled,
        coverage_50=cov50, coverage_80=cov80, coverage_95=cov95,
        r_mean_train=r_mean_train,
        r_std_train=meta["r_std_train"],
        ckpt=os.path.basename(meta.get("train_csv", "")),
        flow_alpha_final=meta.get("alpha_final"),
    )
    summary_path = os.path.join(RESULT, f"{out_prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  saved summary: {summary_path}")


if __name__ == "__main__":
    main()
