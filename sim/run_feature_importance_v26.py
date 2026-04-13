"""Feature importance screening on v26 data with new fold structure + 3M holding + no-lev 21-grid.

Setup:
  - Folds (gap 6M, train 240M, test 54M):
      W1: 1991-01~2010-12 → 2011-07~2015-12
      W2: 1996-01~2015-12 → 2016-07~2020-12
      W3: 2001-01~2020-12 → 2021-07~2025-12
  - Action: 21-point grid on w_s + w_b = 1 line (no leverage)
  - Holding: quarterly rebalance (agent decides every 3 months)
  - Borrow cost: none (no leverage)
  - PP dynamics: pp *= (1 + r), no deflation

Policy (per candidate feature f):
  α_t = clip(α₀ + α_f · f_z, 0.5, 5.0)
  β_t = clip(β₀ + β_f · f_z, 0.5, 5.0)

Evolution: CMA-ES 4D on (α₀, α_f, β₀, β_f).
Fitness:   mean excess Sharpe across 20 bootstrap starts (horizon 120M).
Baseline:  2D (constant α, β) with same setup.

All features standardized using train-only stats.

29 candidate features (17 from v25 + 12 new 6M/3M stats).
"""
import sys, numpy as np, pandas as pd, warnings, time
from pathlib import Path
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from scipy.stats import norm

# ---------- Data (v26) ----------
train_df = pd.read_csv("data/monthly_noleak_v26_train.csv")
test_df = pd.read_csv("data/monthly_noleak_v26_test.csv")
train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["tbill_fwd"] = full["tbill"].shift(-1)

SIGMA = (full["vix"].values / 100.0) / np.sqrt(12.0)
TB = full["tbill_fwd"].fillna(full["tbill"]).values
SP_NEXT = full["sp_next_return"].fillna(0).values
DATES = full["date"].values

MU_FIXED = 0.006
HORIZON = 120
N_STARTS = 20
REBALANCE = 3
SEED = 42
POPSIZE = 60
N_GEN = 15
CLIP_AB = (0.5, 5.0)

N_GRID = 21
GRID = np.array([(w, 1.0 - w) for w in np.linspace(0.0, 1.0, N_GRID)])

FOLDS = [
    ("W1", "1991-01-01", "2010-12-31", "2011-07-01", "2015-12-31"),
    ("W2", "1996-01-01", "2015-12-31", "2016-07-01", "2020-12-31"),
    ("W3", "2001-01-01", "2020-12-31", "2021-07-01", "2025-12-31"),
]

CANDIDATES = [
    "metabolism",
    "m2_growth", "m2_3m",
    "sp_return", "ndx_return",
    "sp_1m", "ndx_1m", "ndx_3m",
    "sentiment", "yield_curve", "credit_spread",
    "sp_52wh_ratio", "sp_52wl_ratio", "sp_in_range",
    "ndx_52wh_ratio", "ndx_52wl_ratio", "ndx_in_range",
    "sp_3m_mean", "sp_6m_mean", "sp_3m_std", "sp_6m_std",
    "ndx_3m_mean", "ndx_6m_mean", "ndx_3m_std", "ndx_6m_std",
    "vix_3m_mean", "vix_6m_mean", "vix_3m_std", "vix_6m_std",
]


def date_range_idx(start, end):
    mask = (full["date"] >= pd.Timestamp(start)) & (full["date"] <= pd.Timestamp(end))
    idxs = full.index[mask].values
    return int(idxs[0]), int(idxs[-1] + 1)


def precompute_policy_f(sigma, tb, f_z, a0, a_f, b0, b_f, mu=MU_FIXED):
    """No-leverage action space, 21-grid."""
    T = len(sigma)
    alpha_t = np.clip(a0 + a_f * f_z, *CLIP_AB)
    beta_t  = np.clip(b0 + b_f * f_z, *CLIP_AB)
    er_matrix = np.zeros((len(GRID), T))
    for k, (w_s, w_b) in enumerate(GRID):
        mu_port = w_s * mu + w_b * tb
        sigma_port = w_s * sigma
        if w_s == 0.0:
            er = np.where(mu_port >= 0, alpha_t * mu_port, beta_t * mu_port)
        else:
            z = mu_port / np.maximum(sigma_port, 1e-12)
            E_plus = sigma_port * norm.pdf(z) + mu_port * norm.cdf(z)
            E_minus = mu_port - E_plus
            er = alpha_t * E_plus + beta_t * E_minus
        er_matrix[k] = er
    best = np.argmax(er_matrix, axis=0)
    return GRID[best, 0], GRID[best, 1]


def simulate_rets_q(start_idx, horizon, w_s_arr, w_b_arr, sp, tb, n_max, rebalance=REBALANCE):
    """Quarterly rebalancing: action changes every `rebalance` steps; in between hold."""
    rets = []
    current_ws = w_s_arr[start_idx]
    current_wb = w_b_arr[start_idx]
    for step in range(horizon):
        i = start_idx + step
        if i >= n_max:
            break
        if step % rebalance == 0:
            current_ws = w_s_arr[i]
            current_wb = w_b_arr[i]
        r = current_ws * sp[i] + current_wb * tb[i]
        rets.append(r)
    return np.array(rets)


def excess_sharpe(rets, tb_slice, ann=12):
    if len(rets) < 6:
        return 0.0
    excess = rets - tb_slice[:len(rets)]
    sd = excess.std()
    if sd < 1e-4:
        return 0.0
    return excess.mean() / sd * np.sqrt(ann)


def fitness(a0, a_f, b0, b_f, slc, f_arr, f_mean, f_std, n_starts=N_STARTS):
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; sp = SP_NEXT[ws_:we_]
    f_w = f_arr[ws_:we_]
    T = we_ - ws_
    horizon = min(HORIZON, T - 1)
    if horizon < 12:
        return -1e10
    f_z = (f_w - f_mean) / max(f_std, 1e-8)
    w_s_arr, w_b_arr = precompute_policy_f(sig, tb, f_z, a0, a_f, b0, b_f)
    max_start = max(0, T - horizon - 1)
    starts = np.linspace(0, max_start, n_starts, dtype=int) if max_start > 0 else np.array([0])
    sharpes = []
    for s in starts:
        s = int(s)
        rets = simulate_rets_q(s, horizon, w_s_arr, w_b_arr, sp, tb, T)
        sharpes.append(excess_sharpe(rets, tb[s:s + len(rets)]))
    return float(np.mean(sharpes))


def cma_es_nd(fn, mu0, sigma0, bounds, popsize=POPSIZE, n_gen=N_GEN, seed=SEED):
    rng = np.random.default_rng(seed)
    mu = np.array(mu0, dtype=float)
    sigma = np.array([sigma0] * len(mu))
    bounds = np.array(bounds)
    for _ in range(n_gen):
        pop = rng.normal(mu, sigma, size=(popsize, len(mu)))
        pop = np.clip(pop, bounds[:, 0], bounds[:, 1])
        fits = np.array([fn(*p) for p in pop])
        elite_n = max(8, popsize // 4)
        idx = np.argsort(fits)[-elite_n:]
        elite = pop[idx]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
    final_best = float(np.max(fits))
    return mu, final_best


def evaluate_test(slc, params, f_arr, f_mean, f_std):
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; sp = SP_NEXT[ws_:we_]
    f_w = f_arr[ws_:we_]
    f_z = (f_w - f_mean) / max(f_std, 1e-8)
    a0, a_f, b0, b_f = params
    w_s_arr, w_b_arr = precompute_policy_f(sig, tb, f_z, a0, a_f, b0, b_f)
    T = we_ - ws_
    rets = []
    current_ws = w_s_arr[0]
    current_wb = w_b_arr[0]
    for step in range(T):
        if step % REBALANCE == 0:
            current_ws = w_s_arr[step]
            current_wb = w_b_arr[step]
        r = current_ws * sp[step] + current_wb * tb[step]
        rets.append(r)
    rets = np.array(rets)
    sh = excess_sharpe(rets, tb[:len(rets)])
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    mean_ws = float(np.mean(w_s_arr))
    return sh, ann, mean_ws


if __name__ == "__main__":
    print("=" * 140, flush=True)
    print("Feature Importance Screening — v26 data, new folds, 3M rebalance, no-lev 21-grid", flush=True)
    print(f"  Candidates: {len(CANDIDATES)} features (17 existing + 12 new 3M/6M stats)", flush=True)
    print(f"  CMA-ES (screening): popsize {POPSIZE} × {N_GEN} gen, seed {SEED}", flush=True)
    print(f"  Rebalance every {REBALANCE} months, Horizon {HORIZON}M", flush=True)
    print("=" * 140, flush=True)

    fold_info = []
    for fold_name, tr_s_d, tr_e_d, te_s_d, te_e_d in FOLDS:
        tr_s, tr_e = date_range_idx(tr_s_d, tr_e_d)
        te_s, te_e = date_range_idx(te_s_d, te_e_d)
        fold_info.append({
            "name": fold_name, "tr": (tr_s, tr_e), "te": (te_s, te_e),
            "tr_range": f"{str(DATES[tr_s])[:7]}~{str(DATES[tr_e-1])[:7]}",
            "te_range": f"{str(DATES[te_s])[:7]}~{str(DATES[te_e-1])[:7]}",
        })
        print(f"  {fold_name}: Train {fold_info[-1]['tr_range']} ({tr_e-tr_s}M)  "
              f"Test {fold_info[-1]['te_range']} ({te_e-te_s}M)", flush=True)

    # 2D Baseline per fold
    print("\n[2D Baseline (constant α, β, no feature)]", flush=True)
    print(f"  {'Fold':<4}  {'α*':>6}  {'β*':>6}  {'Train':>7}  {'Test':>7}  {'AnnRet':>7}  {'μw_s':>5}", flush=True)
    baseline = {}
    for fi in fold_info:
        dummy = np.zeros(len(SIGMA))
        def fn(a0, b0, _slc=fi["tr"], _f=dummy):
            return fitness(a0, 0.0, b0, 0.0, _slc, _f, 0.0, 1.0)
        mu2d, tr_fit = cma_es_nd(fn, mu0=(2.0, 2.0), sigma0=0.6,
                                  bounds=((1.0, 3.0), (1.0, 3.0)))
        a2d, b2d = mu2d
        test_sh, test_ann, mean_ws = evaluate_test(fi["te"], (a2d, 0.0, b2d, 0.0), dummy, 0.0, 1.0)
        baseline[fi["name"]] = {"a": a2d, "b": b2d,
                                "train_sh": tr_fit, "test_sh": test_sh,
                                "test_ann": test_ann, "mean_ws": mean_ws}
        print(f"  {fi['name']:<4}  {a2d:>6.3f}  {b2d:>6.3f}  {tr_fit:>7.4f}  {test_sh:>+7.4f}  "
              f"{test_ann:>+6.2f}%  {mean_ws:>5.2f}", flush=True)

    # Per-feature 4D runs
    print(f"\n{'='*160}", flush=True)
    print("[Per-Feature 4D Add-in]", flush=True)
    print(f"{'='*160}", flush=True)
    hdr = (f"  {'Feature':<18}  {'Fold':<4}  {'α₀':>6} {'α_f':>7} {'β₀':>6} {'β_f':>7}  "
           f"{'TrainSh':>8}  {'TestSh':>8}  {'ΔTest':>7}  {'AnnRet':>8}  {'μw_s':>5}")
    print("\n" + hdr, flush=True)
    print("-" * 160, flush=True)

    rows = []
    t_start = time.time()
    for feature in CANDIDATES:
        if feature not in full.columns:
            print(f"  [SKIP] {feature} not in v26 data", flush=True)
            continue
        f_arr = full[feature].values.astype(float)
        if np.isnan(f_arr).any():
            f_arr = np.nan_to_num(f_arr, nan=0.0)

        for fi in fold_info:
            f_tr = f_arr[fi["tr"][0]:fi["tr"][1]]
            f_mean = float(np.mean(f_tr))
            f_std = float(np.std(f_tr))

            def fn(a0, a_f, b0, b_f, _slc=fi["tr"], _fa=f_arr, _fm=f_mean, _fs=f_std):
                return fitness(a0, a_f, b0, b_f, _slc, _fa, _fm, _fs)

            mu4d, train_fit = cma_es_nd(
                fn, mu0=(2.0, 0.0, 2.0, 0.0), sigma0=0.6,
                bounds=((1.0, 3.0), (-1.5, 1.5), (1.0, 3.0), (-1.5, 1.5)),
            )
            a0, a_f, b0, b_f = mu4d
            test_sh, test_ann, mean_ws = evaluate_test(fi["te"], mu4d, f_arr, f_mean, f_std)
            delta = test_sh - baseline[fi["name"]]["test_sh"]

            print(f"  {feature:<18}  {fi['name']:<4}  "
                  f"{a0:>6.3f} {a_f:>+7.3f} {b0:>6.3f} {b_f:>+7.3f}  "
                  f"{train_fit:>8.3f}  {test_sh:>+8.3f}  {delta:>+7.3f}  "
                  f"{test_ann:>+7.2f}%  {mean_ws:>5.2f}", flush=True)

            rows.append({
                "feature": feature, "fold": fi["name"],
                "a0": a0, "a_f": a_f, "b0": b0, "b_f": b_f,
                "f_mean_train": f_mean, "f_std_train": f_std,
                "train_sh": train_fit, "test_sh": test_sh,
                "baseline_test_sh": baseline[fi["name"]]["test_sh"],
                "delta_test_sh": delta,
                "test_ann_ret": test_ann, "mean_ws": mean_ws,
            })

    elapsed = time.time() - t_start
    print(f"\n  Elapsed: {elapsed:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby("feature").agg({
        "train_sh": "mean",
        "test_sh": "mean",
        "delta_test_sh": "mean",
        "test_ann_ret": "mean",
        "mean_ws": "mean",
        "a_f": lambda x: list(np.round(x, 3)),
        "b_f": lambda x: list(np.round(x, 3)),
    }).rename(columns={
        "train_sh": "train_sh_mean",
        "test_sh": "test_sh_mean",
        "delta_test_sh": "delta_mean",
        "test_ann_ret": "ann_ret_mean",
        "mean_ws": "mean_ws_avg",
        "a_f": "a_f_per_fold",
        "b_f": "b_f_per_fold",
    })
    agg = agg.sort_values("delta_mean", ascending=False)

    base_mean = np.mean([baseline[fn]["test_sh"] for fn in ["W1", "W2", "W3"]])
    base_train = np.mean([baseline[fn]["train_sh"] for fn in ["W1", "W2", "W3"]])
    print(f"\n{'='*160}", flush=True)
    print(f"Feature Importance Ranking (sorted by mean test ΔSharpe vs 2D baseline)", flush=True)
    print(f"  2D baseline mean: Train_ShEx = {base_train:.3f}, Test_ShEx = {base_mean:.3f}", flush=True)
    print(f"{'='*160}", flush=True)
    print(f"  {'Feature':<18}  {'Train':>7}  {'Test':>7}  {'ΔTest':>7}  {'AnnRet':>8}  {'μw_s':>5}  "
          f"{'α_f per fold':<30}  {'β_f per fold':<30}", flush=True)
    print("-" * 160, flush=True)
    for feat, r in agg.iterrows():
        af_str = str(r["a_f_per_fold"])
        bf_str = str(r["b_f_per_fold"])
        print(f"  {feat:<18}  {r['train_sh_mean']:>7.3f}  {r['test_sh_mean']:>+7.3f}  "
              f"{r['delta_mean']:>+7.3f}  {r['ann_ret_mean']:>+7.2f}%  {r['mean_ws_avg']:>5.2f}  "
              f"{af_str:<30}  {bf_str:<30}", flush=True)

    out_csv = Path("result/feature_importance_v26.csv")
    out_csv.parent.mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False)
    agg.to_csv(Path("result/feature_importance_v26_agg.csv"))
    print(f"\n  Per-fold: {out_csv}", flush=True)
    print(f"  Ranking:  result/feature_importance_v26_agg.csv", flush=True)
    print("\nDone.", flush=True)
