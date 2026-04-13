"""3-fold Sharpe evolution with multiple state-conditional features.

Features (from Phase 9 importance screening):
    metabolism                    — macro cost signal (prior +0.18 uplift in main run)
    ndx_52wh_ratio                — #1 in screening (+0.289 mean ΔSharpe)
    ndx_in_range                  — #2 (+0.156)
    ndx_return                    — #3 (+0.119)

Policy:
    α_t = clip(α₀ + Σᵢ α_i · f_i_z, 0.5, 5.0)
    β_t = clip(β₀ + Σᵢ β_i · f_i_z, 0.5, 5.0)
  where f_i_z = (f_i - mean_train_i) / std_train_i (strict train-only standardization)

Evolution: CMA-ES 10D on (α₀, α_met, α_52wh, α_inrange, α_ret, β₀, β_met, β_52wh, β_inrange, β_ret)
Fitness:   mean excess Sharpe over 20 bootstrap starts, horizon 120M.
Baselines: 2D constant (α, β) within same script for apples-to-apples comparison.

Borrow cost: tbill. No metabolism deflation. Features observed at shift 0.

Folds:
    W1: train 1990-05 ~ 2010-06, test 2010-08 ~ 2015-06
    W2: train 1996-01 ~ 2015-06, test 2015-08 ~ 2020-06
    W3: train 2001-01 ~ 2020-06, test 2020-08 ~ 2025-06
"""
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from scipy.stats import norm

# ---------- Data ----------
train_df = pd.read_csv("data/monthly_noleak_v25_train.csv")
test_df = pd.read_csv("data/monthly_noleak_v25_test.csv")
train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["tbill_fwd"] = full["tbill"].shift(-1)

SIGMA = (full["vix"].values / 100.0) / np.sqrt(12.0)
TB = full["tbill_fwd"].fillna(full["tbill"]).values
SP_NEXT = full["sp_next_return"].fillna(0).values
DATES = full["date"].values

FEATURES = ["metabolism", "ndx_52wh_ratio", "ndx_in_range", "ndx_return"]
F_ARRS = {name: full[name].values.astype(float) for name in FEATURES}

MU_FIXED = 0.006
HORIZON = 120
N_STARTS = 20
SEED = 42
POPSIZE = 100
N_GEN = 20
CLIP_AB = (0.5, 5.0)

GRID = []
for wb in np.linspace(0, 1, 5):
    for lev in np.linspace(0, 1, 5):
        ws = (1 - wb) + lev
        GRID.append((ws, wb))
GRID = np.array(GRID)

FOLDS = [
    ("W1", "1990-05-01", "2010-06-30", "2010-08-01", "2015-06-30"),
    ("W2", "1996-01-01", "2015-06-30", "2015-08-01", "2020-06-30"),
    ("W3", "2001-01-01", "2020-06-30", "2020-08-01", "2025-06-30"),
]


def date_range_idx(start, end):
    mask = (full["date"] >= pd.Timestamp(start)) & (full["date"] <= pd.Timestamp(end))
    idxs = full.index[mask].values
    return int(idxs[0]), int(idxs[-1] + 1)


def compute_z_matrix(slc, f_means, f_stds):
    """Return (T, F) z-score matrix for features in window slc."""
    ws_, we_ = slc
    cols = []
    for name in FEATURES:
        f_w = F_ARRS[name][ws_:we_]
        z = (f_w - f_means[name]) / max(f_stds[name], 1e-8)
        cols.append(z)
    return np.stack(cols, axis=1)  # (T, F)


def precompute_policy_multi(sigma, tb, z_mat, a0, a_vec, b0, b_vec, mu=MU_FIXED):
    """z_mat: (T, F). a_vec, b_vec: (F,)."""
    T = len(sigma)
    alpha_t = np.clip(a0 + z_mat @ a_vec, *CLIP_AB)
    beta_t  = np.clip(b0 + z_mat @ b_vec, *CLIP_AB)
    er_matrix = np.zeros((len(GRID), T))
    for k, (w_s, w_b) in enumerate(GRID):
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        mu_port = w_s * mu + w_b * tb - borrow * tb
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


def simulate_rets(start_idx, horizon, w_s_arr, w_b_arr, sp, tb, n_max):
    rets = []
    for step in range(horizon):
        i = start_idx + step
        if i >= n_max:
            break
        w_s = w_s_arr[i]; w_b = w_b_arr[i]
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        r = w_s * sp[i] + w_b * tb[i] - borrow * tb[i]
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


def fitness_multi(params, slc, f_means, f_stds, n_starts=N_STARTS):
    """params = [a0, a_f1, ..., a_fF, b0, b_f1, ..., b_fF]  (length = 2 + 2F)."""
    F = len(FEATURES)
    a0 = params[0]; a_vec = np.array(params[1:1+F])
    b0 = params[1+F]; b_vec = np.array(params[2+F:2+2*F])
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; sp = SP_NEXT[ws_:we_]
    T = we_ - ws_
    horizon = min(HORIZON, T - 1)
    if horizon < 12:
        return -1e10
    z_mat = compute_z_matrix(slc, f_means, f_stds)
    w_s_arr, w_b_arr = precompute_policy_multi(sig, tb, z_mat, a0, a_vec, b0, b_vec)
    max_start = max(0, T - horizon - 1)
    starts = np.linspace(0, max_start, n_starts, dtype=int) if max_start > 0 else np.array([0])
    sharpes = []
    for s in starts:
        s = int(s)
        rets = simulate_rets(s, horizon, w_s_arr, w_b_arr, sp, tb, T)
        sharpes.append(excess_sharpe(rets, tb[s:s + len(rets)]))
    return float(np.mean(sharpes))


def fitness_2d(a, b, slc, n_starts=N_STARTS):
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; sp = SP_NEXT[ws_:we_]
    T = we_ - ws_
    horizon = min(HORIZON, T - 1)
    if horizon < 12:
        return -1e10
    z_zero = np.zeros((T, len(FEATURES)))
    a_vec = np.zeros(len(FEATURES)); b_vec = np.zeros(len(FEATURES))
    w_s_arr, w_b_arr = precompute_policy_multi(sig, tb, z_zero, a, a_vec, b, b_vec)
    max_start = max(0, T - horizon - 1)
    starts = np.linspace(0, max_start, n_starts, dtype=int) if max_start > 0 else np.array([0])
    sharpes = []
    for s in starts:
        s = int(s)
        rets = simulate_rets(s, horizon, w_s_arr, w_b_arr, sp, tb, T)
        sharpes.append(excess_sharpe(rets, tb[s:s + len(rets)]))
    return float(np.mean(sharpes))


def cma_es_nd(fn, mu0, sigma0, bounds, popsize=POPSIZE, n_gen=N_GEN, seed=SEED):
    """fn receives params as a single array/list argument."""
    rng = np.random.default_rng(seed)
    mu = np.array(mu0, dtype=float)
    sigma = np.array([sigma0] * len(mu))
    bounds = np.array(bounds)
    history = []
    for gen in range(n_gen):
        pop = rng.normal(mu, sigma, size=(popsize, len(mu)))
        pop = np.clip(pop, bounds[:, 0], bounds[:, 1])
        fits = np.array([fn(list(p)) for p in pop])
        elite_n = max(10, popsize // 4)
        idx = np.argsort(fits)[-elite_n:]
        elite = pop[idx]; elite_fits = fits[idx]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
        history.append({
            "gen": gen + 1, "mu": mu.copy(),
            "best_fit": float(np.max(fits)), "elite_mean": float(elite_fits.mean()),
        })
    return mu, history


def test_with_multi(slc, params, f_means, f_stds):
    F = len(FEATURES)
    a0 = params[0]; a_vec = np.array(params[1:1+F])
    b0 = params[1+F]; b_vec = np.array(params[2+F:2+2*F])
    ws_, we_ = slc
    sig = SIGMA[ws_:we_]; tb = TB[ws_:we_]; sp = SP_NEXT[ws_:we_]
    T = we_ - ws_
    z_mat = compute_z_matrix(slc, f_means, f_stds)
    w_s_arr, w_b_arr = precompute_policy_multi(sig, tb, z_mat, a0, a_vec, b0, b_vec)
    rets = []
    for step in range(T):
        w_s = w_s_arr[step]; w_b = w_b_arr[step]
        total = w_s + w_b; borrow = max(0.0, total - 1.0)
        r = w_s * sp[step] + w_b * tb[step] - borrow * tb[step]
        rets.append(r)
    rets = np.array(rets)
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    vol = rets.std() * np.sqrt(12) * 100
    sharpe_ex = excess_sharpe(rets, tb[:len(rets)])
    ds = rets[rets < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    calmar = abs(ann / mdd) if mdd != 0 else 0
    mean_total = float(np.mean(w_s_arr + w_b_arr))
    # Action distribution
    uniq = {}
    for ws, wb in zip(w_s_arr, w_b_arr):
        key = (round(ws, 2), round(wb, 2))
        uniq[key] = uniq.get(key, 0) + 1
    return {
        "ann_ret": ann, "vol": vol, "sharpe_ex": sharpe_ex, "sortino": sortino,
        "calmar": calmar, "mdd": mdd, "final_pp": float(cum[-1]),
        "mean_total": mean_total, "action_uniq": uniq,
        "alpha_t_mean": float(np.clip(a0 + z_mat @ a_vec, *CLIP_AB).mean()),
        "beta_t_mean": float(np.clip(b0 + z_mat @ b_vec, *CLIP_AB).mean()),
    }


def metrics_bh(rets, tb_slice):
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    vol = rets.std() * np.sqrt(12) * 100
    sharpe_ex = excess_sharpe(rets, tb_slice)
    ds = rets[rets < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    calmar = abs(ann / mdd) if mdd != 0 else 0
    return dict(ann_ret=ann, vol=vol, sharpe_ex=sharpe_ex, sortino=sortino,
                calmar=calmar, mdd=mdd, final_pp=float(cum[-1]))


if __name__ == "__main__":
    F = len(FEATURES)
    D = 2 + 2 * F  # 10
    print("=" * 140, flush=True)
    print(f"3-fold Multi-Feature Sharpe Evolution ({D}D)", flush=True)
    print(f"  Features ({F}): {FEATURES}", flush=True)
    print(f"  Policy: α_t = clip(α₀ + Σ α_i·f_i_z, {CLIP_AB}), same for β", flush=True)
    print(f"  Borrow cost: tbill. No PP deflation.", flush=True)
    print(f"  CMA-ES: popsize {POPSIZE} × {N_GEN} gen, seed {SEED}", flush=True)
    print(f"    α₀, β₀ ∈ [1, 3],  α_i, β_i ∈ [-1.5, 1.5]", flush=True)
    print("=" * 140, flush=True)

    # Bounds: (α₀, α_f × F, β₀, β_f × F) — 2 + 2F dims
    bounds_base = [(1.0, 3.0)]
    bounds_slope = [(-1.5, 1.5)] * F
    bounds_full = bounds_base + bounds_slope + bounds_base + bounds_slope
    mu0_full = [2.0] + [0.0] * F + [2.0] + [0.0] * F

    all_rows = []
    fold_summary = []

    for fold_name, tr_s_d, tr_e_d, te_s_d, te_e_d in FOLDS:
        tr_s, tr_e = date_range_idx(tr_s_d, tr_e_d)
        te_s, te_e = date_range_idx(te_s_d, te_e_d)
        train_start = str(DATES[tr_s])[:7]
        train_end = str(DATES[tr_e - 1])[:7]
        test_start = str(DATES[te_s])[:7]
        test_end = str(DATES[te_e - 1])[:7]

        # Per-feature train stats
        f_means = {}; f_stds = {}
        train_slices = {}
        for name in FEATURES:
            tr = F_ARRS[name][tr_s:tr_e]
            f_means[name] = float(np.mean(tr))
            f_stds[name] = float(np.std(tr))
            train_slices[name] = tr

        # Correlation matrix (on train)
        corr_mat = np.corrcoef(np.stack([train_slices[n] for n in FEATURES]))

        print(f"\n{'='*140}", flush=True)
        print(f"[{fold_name}]  Train: {train_start}~{train_end} ({tr_e-tr_s}M)  "
              f"Test: {test_start}~{test_end} ({te_e-te_s}M)", flush=True)
        print(f"{'='*140}", flush=True)
        print(f"  Train feature stats:", flush=True)
        for name in FEATURES:
            print(f"    {name:<18} mean={f_means[name]:>+8.5f} std={f_stds[name]:>.5f}", flush=True)

        print(f"\n  Feature correlation matrix (train):", flush=True)
        print(f"    {'':18}  " + "  ".join(f"{n[:12]:>12}" for n in FEATURES), flush=True)
        for i, n in enumerate(FEATURES):
            vals = "  ".join(f"{corr_mat[i, j]:>+12.3f}" for j in range(F))
            print(f"    {n:<18}  {vals}", flush=True)

        # ---------- 2D baseline ----------
        def fn2d(p, _slc=(tr_s, tr_e)):
            return fitness_2d(p[0], p[1], _slc)
        print(f"\n[Evolution 2D baseline]", flush=True)
        print(f"  {'Gen':>4}  {'μ(α)':>7} {'μ(β)':>7}  {'Best':>8}  {'EliteMean':>10}", flush=True)
        mu2d, hist2d = cma_es_nd(
            fn2d, mu0=(2.0, 2.0), sigma0=0.6,
            bounds=((1.0, 3.0), (1.0, 3.0)),
        )
        for h in hist2d[::4] + [hist2d[-1]]:
            m = h['mu']
            print(f"  {h['gen']:>4}  {m[0]:>7.3f} {m[1]:>7.3f}  "
                  f"{h['best_fit']:>8.3f}  {h['elite_mean']:>10.3f}", flush=True)
        a2d, b2d = mu2d
        # Test 2D
        params2d_full = [a2d] + [0.0]*F + [b2d] + [0.0]*F
        r2d = test_with_multi((te_s, te_e), params2d_full, f_means, f_stds)
        print(f"\n[Evolved 2D]  α={a2d:.3f}, β={b2d:.3f}, λ={b2d/a2d:.3f}", flush=True)
        print(f"  Test: AnnRet={r2d['ann_ret']:+.2f}%  ShEx={r2d['sharpe_ex']:+.3f}  MDD={r2d['mdd']:+.1f}%  μTot={r2d['mean_total']:.3f}", flush=True)

        # ---------- 10D multi-feature ----------
        def fnMD(params, _slc=(tr_s, tr_e), _fm=f_means, _fs=f_stds):
            return fitness_multi(params, _slc, _fm, _fs)
        print(f"\n[Evolution {D}D: metabolism + 3 NDX features]", flush=True)
        hdr = "Gen   " + " ".join(f"{'α_'+FEATURES[i][:5]:>7}" for i in range(F))
        hdr += "  " + " ".join(f"{'β_'+FEATURES[i][:5]:>7}" for i in range(F))
        hdr += "   α₀    β₀    Best  EliteMean"
        muMD, histMD = cma_es_nd(
            fnMD, mu0=mu0_full, sigma0=0.6,
            bounds=bounds_full,
        )
        print(f"  Gen   {'α₀':>6} {'β₀':>6} | " + " | ".join(f"{n[:8]:>8}" for n in FEATURES) + f"  {'Best':>7} {'EliteMn':>8}", flush=True)
        for h in histMD[::4] + [histMD[-1]]:
            m = h['mu']
            a_str = " ".join(f"{m[1+i]:>+.3f}" for i in range(F))
            b_str = " ".join(f"{m[2+F+i]:>+.3f}" for i in range(F))
            print(f"  {h['gen']:>3}   {m[0]:>6.3f} {m[1+F]:>6.3f} | α: {a_str} | β: {b_str}  "
                  f"{h['best_fit']:>7.3f} {h['elite_mean']:>8.3f}", flush=True)

        a0 = muMD[0]; a_vec = muMD[1:1+F]
        b0 = muMD[1+F]; b_vec = muMD[2+F:2+2*F]
        lam_base = b0 / a0 if a0 > 0 else float('inf')

        print(f"\n[Evolved {D}D]  α₀={a0:.3f} β₀={b0:.3f} λ_base={lam_base:.3f}", flush=True)
        for i, name in enumerate(FEATURES):
            print(f"    {name:<18}  α_f={a_vec[i]:>+.3f}  β_f={b_vec[i]:>+.3f}", flush=True)

        rMD = test_with_multi((te_s, te_e), list(muMD), f_means, f_stds)

        # ---------- Baselines (Kahneman, Symmetric, B&H, 50/50) ----------
        strategies = [
            (f"Evolved-{D}D", list(muMD), True),
            ("Evolved-2D", params2d_full, False),
            ("Kahneman", [1.0] + [0.0]*F + [2.25] + [0.0]*F, False),
            ("Symmetric", [1.0] + [0.0]*F + [1.0] + [0.0]*F, False),
        ]
        print(f"\n[OOS Test: {test_start}~{test_end}]", flush=True)
        print(f"  {'Strategy':<14}  {'α₀':>5} {'β₀':>5}  "
              f"{'AnnRet':>8}  {'Vol':>6}  {'ShEx':>6}  {'Sortino':>7}  {'Calmar':>6}  "
              f"{'MDD':>8}  {'FinPP':>6}  {'μTotal':>6}  {'α_mean':>6} {'β_mean':>6}", flush=True)
        print("-" * 140, flush=True)
        for name, params, is_main in strategies:
            r = test_with_multi((te_s, te_e), params, f_means, f_stds)
            a_ = params[0]; b_ = params[1+F]
            print(f"  {name:<14}  {a_:>5.2f} {b_:>5.2f}  "
                  f"{r['ann_ret']:>+7.2f}%  {r['vol']:>5.1f}%  {r['sharpe_ex']:>+6.2f}  "
                  f"{r['sortino']:>+7.2f}  {r['calmar']:>+6.2f}  {r['mdd']:>+7.1f}%  "
                  f"{r['final_pp']:>6.3f}  {r['mean_total']:>6.3f}  "
                  f"{r['alpha_t_mean']:>6.3f} {r['beta_t_mean']:>6.3f}", flush=True)
            save = {"fold": fold_name, "strategy": name,
                    "a0": a_, "b0": b_}
            for i, fn in enumerate(FEATURES):
                save[f"a_{fn}"] = params[1+i]
                save[f"b_{fn}"] = params[2+F+i]
            save.update({k: v for k, v in r.items() if k != "action_uniq"})
            all_rows.append(save)

        sp_test = SP_NEXT[te_s:te_e]; tb_test = TB[te_s:te_e]
        for name, rets in [("B&H", sp_test), ("50/50", 0.5*sp_test + 0.5*tb_test)]:
            m = metrics_bh(rets, tb_test)
            print(f"  {name:<14}  {'':>5} {'':>5}  "
                  f"{m['ann_ret']:>+7.2f}%  {m['vol']:>5.1f}%  {m['sharpe_ex']:>+6.2f}  "
                  f"{m['sortino']:>+7.2f}  {m['calmar']:>+6.2f}  {m['mdd']:>+7.1f}%  "
                  f"{m['final_pp']:>6.3f}", flush=True)
            all_rows.append({"fold": fold_name, "strategy": name, **m})

        # Action dist
        print(f"\n[Evolved-{D}D action distribution ({fold_name} test, top)]", flush=True)
        print(f"  {'(w_s, w_b)':<14}  {'total':>6}  {'count':>5}  {'pct':>5}", flush=True)
        for (ws, wb), cnt in sorted(rMD["action_uniq"].items(), key=lambda x: -x[1])[:8]:
            print(f"  ({ws:>4.2f}, {wb:>4.2f})    {ws+wb:>6.2f}  {cnt:>5}  "
                  f"{cnt/sum(rMD['action_uniq'].values())*100:>4.0f}%", flush=True)

        fold_summary.append({
            "fold": fold_name, "a0": a0, "b0": b0, "lambda_base": lam_base,
            **{f"a_{n}": a_vec[i] for i, n in enumerate(FEATURES)},
            **{f"b_{n}": b_vec[i] for i, n in enumerate(FEATURES)},
            "a2d": a2d, "b2d": b2d,
            "test_sh_multi": rMD["sharpe_ex"], "test_sh_2d": r2d["sharpe_ex"],
            "test_ann_multi": rMD["ann_ret"], "test_ann_2d": r2d["ann_ret"],
            "test_mdd_multi": rMD["mdd"], "test_mdd_2d": r2d["mdd"],
        })

    out_csv = Path("result/evolved_sharpe_multifeat.csv")
    out_csv.parent.mkdir(exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    print(f"\n\n{'='*140}", flush=True)
    print("Summary: 2D baseline vs 10D multi-feature", flush=True)
    print(f"{'='*140}", flush=True)
    print(f"  {'Fold':<4}  {'2D_ShEx':>8}  {'{}D_ShEx':>9}  {'ΔShEx':>7}  "
          f"{'2D_AnnRet':>10}  {'{}D_AnnRet':>11}  "
          f"{'2D_MDD':>8}  {'{}D_MDD':>9}".format(D, D, D), flush=True)
    for fs in fold_summary:
        delta = fs['test_sh_multi'] - fs['test_sh_2d']
        print(f"  {fs['fold']:<4}  {fs['test_sh_2d']:>+8.3f}  {fs['test_sh_multi']:>+9.3f}  {delta:>+7.3f}  "
              f"{fs['test_ann_2d']:>+9.2f}%  {fs['test_ann_multi']:>+10.2f}%  "
              f"{fs['test_mdd_2d']:>+7.1f}%  {fs['test_mdd_multi']:>+8.1f}%", flush=True)

    mean_sh_2d = np.mean([fs['test_sh_2d'] for fs in fold_summary])
    mean_sh_md = np.mean([fs['test_sh_multi'] for fs in fold_summary])
    mean_ann_2d = np.mean([fs['test_ann_2d'] for fs in fold_summary])
    mean_ann_md = np.mean([fs['test_ann_multi'] for fs in fold_summary])
    print(f"\n  Mean across folds:", flush=True)
    print(f"    2D baseline:     ShEx={mean_sh_2d:+.3f}  AnnRet={mean_ann_2d:+.2f}%", flush=True)
    print(f"    {D}D multi-feat:  ShEx={mean_sh_md:+.3f}  AnnRet={mean_ann_md:+.2f}%", flush=True)
    print(f"    ΔShEx: {mean_sh_md - mean_sh_2d:+.3f}   ΔAnnRet: {mean_ann_md - mean_ann_2d:+.2f}%", flush=True)

    print(f"\nResults saved: {out_csv}", flush=True)
    print("\nDone.", flush=True)
