"""Phase 13: Neuroevolution — CMA-ES로 초소형 MLP 가중치를 직접 진화.

구조:
  - LGBM → p_up (feature extractor, 1회 학습)
  - Tiny MLP: [p_up, vix_z, metab_z, pp] → 4 hidden → sigmoid → w ∈ [0, 1]
  - CMA-ES가 MLP 가중치(~25개)를 직접 탐색
  - Fitness: PP-Calmar on train

기초대사 (metabolism):
  max(m2_growth, tbill, mich/100/12)
  세 가지 구매력 침식 채널의 max — 임의 상수 없음.
    - m2_growth: 유동성 팽창에 의한 구매력 희석
    - tbill: 무위험 수익 기회비용
    - mich: 경제 주체의 기대인플레이션 (University of Michigan Survey)

핵심:
  - PP가 observation에 들어감 → 항상성 feedback loop
  - 예측 없음. "내 상태 + 환경"에 반응하는 policy를 진화
  - 파라미터 25개 → 360개월 데이터에 적합

한계:
  - 단일 역사 path
  - Calmar fitness는 path-dependent
"""
import sys, time, numpy as np, pandas as pd, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import lightgbm as lgb

# ── Data ──
train_df = pd.read_csv("data/monthly_noleak_v28_train.csv")
test_df  = pd.read_csv("data/monthly_noleak_v28_test.csv")
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full = full.sort_values("date").reset_index(drop=True)
N = len(full)

r = full["sp_next_return"].fillna(0).values
y3 = np.full(N, np.nan)
for t in range(N - 3):
    y3[t] = (1 + r[t]) * (1 + r[t+1]) * (1 + r[t+2]) - 1.0
full["y3"] = y3
full["label"] = (full["y3"] >= 0.10).astype(float)

EXCLUDE = {"date", "sp_next_return", "label", "y3"}
LGBM_FEATURES = [c for c in full.columns if c not in EXCLUDE]

LGBM_PARAMS = dict(objective="binary", metric="binary_logloss",
                   learning_rate=0.03, num_leaves=15, max_depth=5,
                   min_data_in_leaf=20, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=5,
                   lambda_l2=1.0, verbose=-1, seed=42)

WINDOWS = {
    "W1": dict(tr=("1991-01-01", "2010-12-31"), te=("2011-07-01", "2015-12-31")),
    "W2": dict(tr=("1996-01-01", "2015-12-31"), te=("2016-07-01", "2020-12-31")),
    "W3": dict(tr=("2001-01-01", "2020-12-31"), te=("2021-07-01", "2025-12-31")),
}


# ── LGBM ──
def fit_lgbm(X, y, seed=42):
    n = len(y); sp = int(n * 0.85)
    dt = lgb.Dataset(X[:sp], label=y[:sp])
    dv = lgb.Dataset(X[sp:], label=y[sp:], reference=dt)
    params = {**LGBM_PARAMS, "seed": seed}
    return lgb.train(params, dt, 2000, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])


# ── Tiny MLP ──
# Input: [p_up, vix_z, metab_z, pp, gamma_z] = 5
# Hidden: 4 nodes (ReLU)
# Output: 1 (sigmoid → w ∈ [0, 1])
# Params: W1(5×4)=20 + b1(4) + W2(4×1)=4 + b2(1) = 29

INPUT_DIM = 5
HIDDEN_DIM = 4
PARAM_DIM = INPUT_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * 1 + 1  # 29


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def unpack_params(params):
    """Unpack flat parameter vector into MLP weights."""
    idx = 0
    W1 = params[idx:idx + INPUT_DIM * HIDDEN_DIM].reshape(INPUT_DIM, HIDDEN_DIM)
    idx += INPUT_DIM * HIDDEN_DIM
    b1 = params[idx:idx + HIDDEN_DIM]
    idx += HIDDEN_DIM
    W2 = params[idx:idx + HIDDEN_DIM].reshape(HIDDEN_DIM, 1)
    idx += HIDDEN_DIM
    b2 = params[idx]
    return W1, b1, W2, b2


def mlp_forward(z, W1, b1, W2, b2):
    """z: [4] → w: scalar ∈ [0, 1]"""
    h = np.maximum(z @ W1 + b1, 0)  # ReLU
    return sigmoid((h @ W2 + b2).item())


def simulate(params, p_up, vix_z, metab_z, gamma_z, sp_next, tbill, metab, rebal_months=3):
    """Run policy through market, rebalance every rebal_months."""
    W1, b1, W2, b2 = unpack_params(params)
    T = len(sp_next)
    pp = 1.0
    rets = np.empty(T)
    ws = np.empty(T)
    w = 1.0  # initial

    for t in range(T):
        if t % rebal_months == 0:
            z = np.array([p_up[t], vix_z[t], metab_z[t], pp, gamma_z[t]])
            w = mlp_forward(z, W1, b1, W2, b2)
        ws[t] = w
        port_ret = w * sp_next[t] + (1 - w) * tbill[t]
        rets[t] = port_ret
        pp = pp * (1 + port_ret) / (1 + metab[t])

    return rets, ws, pp


def calmar(rets):
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    if mdd >= 0 or eq[-1] <= 0:
        return -10.0
    n_yr = len(rets) / 12
    ann = eq[-1] ** (1 / n_yr) - 1
    return ann / abs(mdd)


def pp_calmar(params, p_up, vix_z, metab_z, gamma_z, sp_next, tbill, metab, rebal_months=3):
    """PP 기반 Calmar: PP trajectory의 성장률/drawdown.
    rebal_months마다 w를 결정하고, 그 사이는 고정.
    현금 도피하면 PP가 metabolism에 먹혀서 Calmar가 나빠진다."""
    W1, b1, W2, b2 = unpack_params(params)
    T = len(sp_next)
    pp = 1.0
    pp_history = np.empty(T + 1)
    pp_history[0] = pp
    w = 1.0  # initial

    for t in range(T):
        if t % rebal_months == 0:
            z = np.array([p_up[t], vix_z[t], metab_z[t], pp, gamma_z[t]])
            w = mlp_forward(z, W1, b1, W2, b2)
        port_ret = w * sp_next[t] + (1 - w) * tbill[t]
        pp = pp * (1 + port_ret) / (1 + metab[t])
        pp_history[t + 1] = pp

    # Calmar on PP trajectory
    peak = np.maximum.accumulate(pp_history)
    mdd = float((pp_history / peak - 1).min())
    if mdd >= 0 or pp_history[-1] <= 0:
        return -10.0
    n_yr = T / 12
    ann = pp_history[-1] ** (1 / n_yr) - 1
    return ann / abs(mdd)


def sharpe(rets, tb):
    if rets.std() < 1e-10:
        return 0.0
    return (rets.mean() - tb.mean()) / rets.std() * np.sqrt(12)


# ── CMA-ES ──
def cmaes(fit_fn, dim, popsize=100, n_gen=50, sigma0=0.5, seed=42):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 0.3, dim)  # small random init
    sigma = np.full(dim, sigma0)
    best_sol = mu.copy()
    best_fit = -np.inf

    for gen in range(n_gen):
        pop = rng.normal(mu, sigma, size=(popsize, dim))
        fits = np.array([fit_fn(p) for p in pop])
        elite_n = max(10, popsize // 4)
        idx = np.argsort(fits)[-elite_n:]
        elite = pop[idx]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.01)
        if fits.max() > best_fit:
            best_fit = fits.max()
            best_sol = pop[fits.argmax()].copy()

    return best_sol, best_fit


# ── Main ──
if __name__ == "__main__":
    t0 = time.time()

    print(f"Neuroevolution: {INPUT_DIM} input → {HIDDEN_DIM} hidden → 1 output")
    print(f"Parameters: {PARAM_DIM}")
    print()

    for fold_name, fold_cfg in WINDOWS.items():
        print(f"{'='*80}")
        print(f"  FOLD {fold_name}")
        print(f"{'='*80}")

        tr_start, tr_end = fold_cfg["tr"]
        te_start, te_end = fold_cfg["te"]

        tr_mask = (full["date"] >= tr_start) & (full["date"] <= tr_end)
        te_mask = (full["date"] >= te_start) & (full["date"] <= te_end)
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]

        print(f"  Train: {tr_start} ~ {tr_end} ({len(tr_idx)} months)")
        print(f"  Test:  {te_start} ~ {te_end} ({len(te_idx)} months)")

        # ── LGBM p_up ──
        lgbm_mask = tr_mask & full["label"].notna()
        lgbm_idx = np.where(lgbm_mask)[0]
        X_all = full[LGBM_FEATURES].values
        y_all = full["label"].values
        Xtr = X_all[lgbm_idx]
        ytr = y_all[lgbm_idx].astype(int)
        mdl = fit_lgbm(Xtr, ytr)

        p_up_tr = mdl.predict(X_all[tr_idx], num_iteration=mdl.best_iteration)
        p_up_te = mdl.predict(X_all[te_idx], num_iteration=mdl.best_iteration)

        # ── Feature prep ──
        # VIX z-score (standardized on train)
        vix_tr = full.loc[tr_idx, "vix"].values
        vix_te = full.loc[te_idx, "vix"].values
        vix_mu, vix_sd = vix_tr.mean(), vix_tr.std() + 1e-8
        vix_z_tr = (vix_tr - vix_mu) / vix_sd
        vix_z_te = (vix_te - vix_mu) / vix_sd

        # Metabolism: max(m2_growth, tbill, mich) — 세 채널 max, 임의 상수 없음
        m2_tr = full.loc[tr_idx, "m2_growth"].values
        tb_tr = full.loc[tr_idx, "tbill"].values / 12.0
        mich_tr = full.loc[tr_idx, "mich"].values / 100.0 / 12.0
        metab_tr = np.maximum(np.maximum(m2_tr, tb_tr), mich_tr)
        m2_te = full.loc[te_idx, "m2_growth"].values
        tb_te = full.loc[te_idx, "tbill"].values / 12.0
        mich_te = full.loc[te_idx, "mich"].values / 100.0 / 12.0
        metab_te = np.maximum(np.maximum(m2_te, tb_te), mich_te)
        met_mu, met_sd = metab_tr.mean(), metab_tr.std() + 1e-8
        metab_z_tr = (metab_tr - met_mu) / met_sd
        metab_z_te = (metab_te - met_mu) / met_sd

        # Gamma z-score (market risk aversion)
        gamma_tr = full.loc[tr_idx, "gamma"].values
        gamma_te = full.loc[te_idx, "gamma"].values
        gam_mu, gam_sd = gamma_tr.mean(), gamma_tr.std() + 1e-8
        gamma_z_tr = (gamma_tr - gam_mu) / gam_sd
        gamma_z_te = (gamma_te - gam_mu) / gam_sd

        # S&P next return
        sp_next_tr = full.loc[tr_idx, "sp_next_return"].values
        sp_next_te = full.loc[te_idx, "sp_next_return"].values

        # ── CMA-ES on train ──
        print(f"\n  CMA-ES: {PARAM_DIM} params, popsize=100, gen=50...")

        def fitness(params):
            return pp_calmar(params, p_up_tr, vix_z_tr, metab_z_tr, gamma_z_tr,
                             sp_next_tr, tb_tr, metab_tr)

        t_evo = time.time()
        best_params, best_fit = cmaes(fitness, PARAM_DIM, popsize=100, n_gen=50, seed=42)
        evo_time = time.time() - t_evo
        print(f"  CMA-ES done in {evo_time:.1f}s, best Calmar={best_fit:.3f}")

        # ── Train evaluation ──
        tr_rets, tr_ws, tr_pp = simulate(best_params, p_up_tr, vix_z_tr, metab_z_tr, gamma_z_tr,
                                          sp_next_tr, tb_tr, metab_tr)
        print(f"  Train: Calmar={calmar(tr_rets):.3f}, PP={tr_pp:.3f}, "
              f"mean_w={tr_ws.mean():.2f}, min_w={tr_ws.min():.2f}, max_w={tr_ws.max():.2f}")

        # ── Test evaluation ──
        te_rets, te_ws, te_pp = simulate(best_params, p_up_te, vix_z_te, metab_z_te, gamma_z_te,
                                          sp_next_te, tb_te, metab_te)

        te_cal = calmar(te_rets)
        te_sh = sharpe(te_rets, tb_te)
        te_eq = np.cumprod(1 + te_rets)
        te_mdd = float((te_eq / np.maximum.accumulate(te_eq) - 1).min())
        te_ann = te_eq[-1] ** (12 / len(te_idx)) - 1
        te_mean_w = te_ws.mean()

        # B&H
        bh_eq = np.cumprod(1 + sp_next_te)
        bh_mdd = float((bh_eq / np.maximum.accumulate(bh_eq) - 1).min())
        bh_ann = bh_eq[-1] ** (12 / len(te_idx)) - 1
        bh_cal = bh_ann / abs(bh_mdd) if bh_mdd < 0 else -999
        bh_sh = sharpe(sp_next_te, tb_te)

        # Static same W
        r_static = te_mean_w * sp_next_te + (1 - te_mean_w) * tb_te
        s_eq = np.cumprod(1 + r_static)
        s_mdd = float((s_eq / np.maximum.accumulate(s_eq) - 1).min())
        s_ann = s_eq[-1] ** (12 / len(te_idx)) - 1
        s_cal = s_ann / abs(s_mdd) if s_mdd < 0 else -999
        s_sh = sharpe(r_static, tb_te)

        # 50/50
        r5050 = 0.5 * sp_next_te + 0.5 * tb_te
        eq5050 = np.cumprod(1 + r5050)
        mdd5050 = float((eq5050 / np.maximum.accumulate(eq5050) - 1).min())
        ann5050 = eq5050[-1] ** (12 / len(te_idx)) - 1
        cal5050 = ann5050 / abs(mdd5050) if mdd5050 < 0 else -999
        sh5050 = sharpe(r5050, tb_te)

        # Selective leverage (Phase 11)
        w_sel = np.where(p_up_te >= 0.2, 2.0, 1.0)
        r_sel = w_sel * sp_next_te + (1 - w_sel) * tb_te
        eq_sel = np.cumprod(1 + r_sel)
        mdd_sel = float((eq_sel / np.maximum.accumulate(eq_sel) - 1).min())
        ann_sel = eq_sel[-1] ** (12 / len(te_idx)) - 1
        cal_sel = ann_sel / abs(mdd_sel) if mdd_sel < 0 else -999
        sh_sel = sharpe(r_sel, tb_te)
        w_sel_mean = w_sel.mean()

        print(f"\n  {'':>25} {'Return':>8} {'Sharpe':>8} {'Calmar':>8} {'MDD':>8} {'MeanW':>7}")
        print(f"  {'Neuroevolution':<25} {te_ann*100:>7.2f}% {te_sh:>8.3f} {te_cal:>8.3f} {te_mdd*100:>7.1f}% {te_mean_w:>7.2f}")
        print(f"  {'Static same W':<25} {s_ann*100:>7.2f}% {s_sh:>8.3f} {s_cal:>8.3f} {s_mdd*100:>7.1f}% {te_mean_w:>7.2f}")
        print(f"  {'Selective lev (P11)':<25} {ann_sel*100:>7.2f}% {sh_sel:>8.3f} {cal_sel:>8.3f} {mdd_sel*100:>7.1f}% {w_sel_mean:>7.2f}")
        print(f"  {'B&H':<25} {bh_ann*100:>7.2f}% {bh_sh:>8.3f} {bh_cal:>8.3f} {bh_mdd*100:>7.1f}% {'1.00':>7}")
        print(f"  {'50/50':<25} {ann5050*100:>7.2f}% {sh5050:>8.3f} {cal5050:>8.3f} {mdd5050*100:>7.1f}% {'0.50':>7}")

        # w distribution
        print(f"\n  w distribution: min={te_ws.min():.3f} p25={np.percentile(te_ws,25):.3f} "
              f"med={np.median(te_ws):.3f} p75={np.percentile(te_ws,75):.3f} max={te_ws.max():.3f}")

        # corr with p_up
        corr_pup = np.corrcoef(p_up_te, te_ws)[0, 1] if te_ws.std() > 1e-6 else 0
        print(f"  corr(p_up, w) = {corr_pup:+.3f}")
        print(f"  Final PP = {te_pp:.3f}")
        print()

    print(f"Total elapsed: {time.time()-t0:.1f}s")
