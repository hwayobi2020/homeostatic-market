"""Phase 12: Population-based Temporal Evolution.

진짜 진화 구현:
- Layer 1 (CMA-ES): 각 agent가 자기 (α, β) reward function에 맞는 policy 학습
- Layer 2 (Evolution): train 구간에서 매 월 PP tracking, death, 연간 번식

Genotype (진화 대상): (α, β) — reward = α·r⁺ + β·r⁻
Phenotype (CMA-ES 학습): w = clip(sigmoid(θᵀz + b), 0, 2)

Fold 구조:
  - Warmup (120M): LGBM + 초기 population CMA-ES 학습
  - Evolution (120M): population이 시장 겪으면서 사망/번식
  - Gap (6M)
  - Test (~54M): 생존자 고정 투입, OOS 평가

한계:
  - Evolution 구간 120개월에 major crash 1~2회 (selection event 희소)
  - 200 agents가 같은 market path 경험 (agent 간 상호작용 없음)
  - 단일 역사 path이므로 운 vs 전략 구분 본질적으로 어려움
"""
import sys, time, numpy as np, pandas as pd, warnings
from pathlib import Path
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import lightgbm as lgb

# ── Data ──
train_df = pd.read_csv("data/monthly_noleak_v26_train.csv")
test_df  = pd.read_csv("data/monthly_noleak_v26_test.csv")
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full = full.sort_values("date").reset_index(drop=True)
N = len(full)

# 3M compound forward return for LGBM target
r = full["sp_next_return"].fillna(0).values
y3 = np.full(N, np.nan)
for t in range(N - 3):
    y3[t] = (1 + r[t]) * (1 + r[t+1]) * (1 + r[t+2]) - 1.0
full["y3"] = y3
full["label"] = (full["y3"] >= 0.10).astype(float)

EXCLUDE = {"date", "sp_next_return", "label", "y3"}
LGBM_FEATURES = [c for c in full.columns if c not in EXCLUDE]

# Policy features (Phase 7/10 feature importance 기반 선별)
POLICY_FEATS = ["vix", "sp_1m", "ndx_in_range", "ndx_52wh_ratio",
                "metabolism", "tbill", "sp_6m_mean"]
FEAT_DIM = len(POLICY_FEATS) + 1  # +1 for p_up
PARAM_DIM = FEAT_DIM + 1  # +1 for bias

# ── Fold 정의 ──
# Train 240M을 warmup 120M + evolution 120M으로 분할
WINDOWS = {
    "W1": dict(
        warmup=("1991-01-01", "2000-12-31"),
        evo=("2001-01-01", "2010-12-31"),
        test=("2011-07-01", "2015-12-31"),
    ),
    "W2": dict(
        warmup=("1996-01-01", "2005-12-31"),
        evo=("2006-01-01", "2015-12-31"),
        test=("2016-07-01", "2020-12-31"),
    ),
    "W3": dict(
        warmup=("2001-01-01", "2010-12-31"),
        evo=("2011-01-01", "2020-12-31"),
        test=("2021-07-01", "2025-12-31"),
    ),
}

# ── LGBM ──
LGBM_PARAMS = dict(objective="binary", metric="binary_logloss",
                   learning_rate=0.03, num_leaves=15, max_depth=5,
                   min_data_in_leaf=20, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=5,
                   lambda_l2=1.0, verbose=-1, seed=42)


def fit_lgbm(X, y, seed=42):
    n = len(y); sp = int(n * 0.85)
    dt = lgb.Dataset(X[:sp], label=y[:sp])
    dv = lgb.Dataset(X[sp:], label=y[sp:], reference=dt)
    params = {**LGBM_PARAMS, "seed": seed}
    return lgb.train(params, dt, 2000, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])


def get_lgbm_pup(train_start, train_end, predict_idx):
    """Train LGBM on [train_start, train_end], predict on predict_idx."""
    tr_mask = ((full["date"] >= train_start) & (full["date"] <= train_end)
               & (full["label"].notna()))
    tr_idx = np.where(tr_mask)[0]
    X_all = full[LGBM_FEATURES].values
    y_all = full["label"].values
    Xtr = X_all[tr_idx]; ytr = y_all[tr_idx].astype(int)
    model = fit_lgbm(Xtr, ytr)
    p = model.predict(X_all[predict_idx], num_iteration=model.best_iteration)
    return p


# ── Policy ──
def policy_w(Z, theta, b):
    """w ∈ [0, 2]: sigmoid * 2"""
    s = 1.0 / (1.0 + np.exp(-np.clip(Z @ theta + b, -10, 10)))
    return s * 2.0  # [0, 2]


def build_features(idx, p_up_arr, train_idx_for_stats):
    """Build standardized feature matrix.
    train_idx_for_stats: indices used to compute mean/std for standardization.
    """
    raw = full.loc[idx, POLICY_FEATS].values.astype(float)
    ref = full.loc[train_idx_for_stats, POLICY_FEATS].values.astype(float)
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0) + 1e-8
    Z_feat = np.nan_to_num((raw - mu) / sd, nan=0.0)
    # prepend p_up column
    Z = np.column_stack([p_up_arr.reshape(-1, 1), Z_feat])
    return Z


# ── CMA-ES (policy optimizer) ──
def es_search(fit_fn, dim, sigma0=0.8, popsize=60, n_gen=25, bounds=(-5, 5), seed=42):
    rng = np.random.default_rng(seed)
    mu = np.zeros(dim); mu[-1] = -1.0  # bias init: slightly negative → w near 1.0
    sigma = np.full(dim, sigma0)
    best_sol = mu.copy(); best_fit = -np.inf
    for gen in range(n_gen):
        pop = np.clip(rng.normal(mu, sigma, size=(popsize, dim)), bounds[0], bounds[1])
        fits = np.array([fit_fn(p) for p in pop])
        elite_n = max(8, popsize // 4)
        idx = np.argsort(fits)[-elite_n:]
        elite = pop[idx]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
        if fits.max() > best_fit:
            best_fit = fits.max()
            best_sol = pop[fits.argmax()].copy()
    return best_sol, best_fit


def train_policy(alpha, beta, Z_train, r_next_train, tb_train, seed=42):
    """Train policy for agent with given (alpha, beta) reward params.
    Fitness = sum of asymmetric rewards (LinExc-style, path-independent).
    """
    def fit_fn(params):
        theta, b = params[:-1], params[-1]
        w = policy_w(Z_train, theta, b)
        port_r = w * r_next_train + (1 - w) * tb_train
        # metabolism: max(m2, tbill) — already in data, but for simplicity
        # use raw port return as r for reward calc
        r_net = port_r - tb_train  # excess over risk-free
        reward_pos = alpha * np.maximum(r_net, 0)
        reward_neg = beta * np.minimum(r_net, 0)  # beta > 0, r_net < 0 → negative
        return float(np.sum(reward_pos + reward_neg))

    sol, fit = es_search(fit_fn, PARAM_DIM, seed=seed)
    return sol[:-1], sol[-1]  # theta, b


# ── Agent ──
class Agent:
    __slots__ = ["alpha", "beta", "theta", "b", "pp", "birth_month", "agent_id"]
    _counter = 0

    def __init__(self, alpha, beta, theta, b, birth_month=0):
        Agent._counter += 1
        self.agent_id = Agent._counter
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.b = b
        self.pp = 1.0
        self.birth_month = birth_month

    @property
    def lam(self):
        return self.beta / self.alpha if self.alpha > 0 else float("inf")

    def decide_w(self, z_row):
        """Single month decision."""
        s = 1.0 / (1.0 + np.exp(-np.clip(z_row @ self.theta + self.b, -10, 10)))
        return s * 2.0  # [0, 2]


# ── Evolution parameters ──
POP_INIT = 200
DEATH_THRESHOLD = 0.70
MUTATION_SIGMA = 0.30
ALPHA_BETA_SUM = 3.25        # α + β = 3.25 (Kahneman: α=1, β=2.25)
ALPHA_RANGE = (0.25, 3.00)   # α ∈ [0.25, 3.00], β = 3.25 - α ∈ [0.25, 3.00]
REPRODUCTION_INTERVAL = 12   # months
IMMIGRANT_PER_YEAR = 10
POP_CAP = 300
RNG_SEED = 42


def run_fold(fold_name, fold_cfg):
    print(f"\n{'='*80}")
    print(f"  FOLD {fold_name}")
    print(f"{'='*80}")

    rng = np.random.default_rng(RNG_SEED)
    Agent._counter = 0

    warmup_start, warmup_end = fold_cfg["warmup"]
    evo_start, evo_end = fold_cfg["evo"]
    test_start, test_end = fold_cfg["test"]

    # Index masks
    warmup_mask = (full["date"] >= warmup_start) & (full["date"] <= warmup_end)
    evo_mask = (full["date"] >= evo_start) & (full["date"] <= evo_end)
    test_mask = (full["date"] >= test_start) & (full["date"] <= test_end)
    warmup_idx = np.where(warmup_mask)[0]
    evo_idx = np.where(evo_mask)[0]
    test_idx = np.where(test_mask)[0]

    print(f"  Warmup: {warmup_start} ~ {warmup_end} ({len(warmup_idx)} months)")
    print(f"  Evolution: {evo_start} ~ {evo_end} ({len(evo_idx)} months)")
    print(f"  Test: {test_start} ~ {test_end} ({len(test_idx)} months)")

    # ── Step 1: LGBM on warmup → p_up for warmup (in-sample) and evo (OOS) ──
    print("\n  [1] LGBM training on warmup data...")
    warmup_evo_idx = np.concatenate([warmup_idx, evo_idx])
    p_warmup_evo = get_lgbm_pup(warmup_start, warmup_end, warmup_evo_idx)
    p_warmup = p_warmup_evo[:len(warmup_idx)]
    p_evo = p_warmup_evo[len(warmup_idx):]

    # ── Step 2: Build feature matrices ──
    Z_warmup = build_features(warmup_idx, p_warmup, warmup_idx)
    r_next_warmup = full.loc[warmup_idx, "sp_next_return"].values
    tb_warmup = full.loc[warmup_idx, "tbill"].values / 12.0

    # ── Step 3: Initialize population ──
    print(f"\n  [2] Initializing {POP_INIT} agents...")
    population = []
    for i in range(POP_INIT):
        alpha = rng.uniform(*ALPHA_RANGE)
        beta = ALPHA_BETA_SUM - alpha
        seed_i = RNG_SEED + i
        theta, b = train_policy(alpha, beta, Z_warmup, r_next_warmup, tb_warmup, seed=seed_i)
        agent = Agent(alpha, beta, theta, b, birth_month=0)
        population.append(agent)

    init_lambdas = [a.lam for a in population]
    print(f"  Initial λ distribution: mean={np.mean(init_lambdas):.2f}, "
          f"median={np.median(init_lambdas):.2f}, "
          f"min={np.min(init_lambdas):.2f}, max={np.max(init_lambdas):.2f}")

    # ── Step 4: Evolution on evo period ──
    print(f"\n  [3] Evolution running ({len(evo_idx)} months)...")

    # Precompute per-month data for evolution period
    evo_dates = full.loc[evo_idx, "date"].values
    evo_r_next = full.loc[evo_idx, "sp_next_return"].values
    evo_tbill = full.loc[evo_idx, "tbill"].values / 12.0
    evo_m2 = full.loc[evo_idx, "m2_growth"].values  # monthly m2 growth
    evo_metab = np.maximum(evo_m2, evo_tbill)  # metabolism = max(m2, tbill)

    # Build Z for each evo month using warmup stats for standardization
    Z_evo = build_features(evo_idx, p_evo, warmup_idx)

    # Track history
    death_log = []  # (month, agent_id, alpha, beta, lambda, pp_at_death)
    pop_log = []    # (month, n_alive, mean_lambda, median_lambda, mean_pp)
    birth_log = []  # (month, agent_id, parent_id, alpha, beta)

    months_since_repro = 0
    lgbm_retrain_year = None

    for m_idx, m in enumerate(range(len(evo_idx))):
        month_date = evo_dates[m]
        z_row = Z_evo[m]
        r_next_m = evo_r_next[m]
        tb_m = evo_tbill[m]
        metab_m = evo_metab[m]

        # Each agent decides and updates PP
        for agent in population:
            w = agent.decide_w(z_row)
            port_ret = w * r_next_m + (1 - w) * tb_m
            agent.pp = agent.pp * (1 + port_ret) / (1 + metab_m)

        # Death check
        dead = [a for a in population if a.pp < DEATH_THRESHOLD]
        for a in dead:
            death_log.append(dict(
                month=m, date=str(month_date)[:10],
                agent_id=a.agent_id, alpha=a.alpha, beta=a.beta,
                lam=a.lam, pp=a.pp, birth=a.birth_month
            ))
        population = [a for a in population if a.pp >= DEATH_THRESHOLD]

        # Log population state
        if population:
            lambdas = [a.lam for a in population]
            pps = [a.pp for a in population]
            pop_log.append(dict(
                month=m, date=str(month_date)[:10],
                n_alive=len(population),
                mean_lam=np.mean(lambdas), median_lam=np.median(lambdas),
                std_lam=np.std(lambdas),
                mean_pp=np.mean(pps), min_pp=np.min(pps), max_pp=np.max(pps),
                mean_alpha=np.mean([a.alpha for a in population]),
                mean_beta=np.mean([a.beta for a in population]),
            ))
        else:
            pop_log.append(dict(
                month=m, date=str(month_date)[:10],
                n_alive=0, mean_lam=np.nan, median_lam=np.nan,
                std_lam=np.nan, mean_pp=np.nan, min_pp=np.nan, max_pp=np.nan,
                mean_alpha=np.nan, mean_beta=np.nan,
            ))

        months_since_repro += 1

        # Reproduction every 12 months
        if months_since_repro >= REPRODUCTION_INTERVAL and len(population) > 0:
            months_since_repro = 0

            # Current year for LGBM retrain
            current_date_str = str(month_date)[:10]

            # Retrain LGBM with expanding window [warmup_start, current]
            lgbm_train_end = current_date_str
            # Retrain p_up for remaining evo months
            remaining_evo_idx = evo_idx[m+1:]
            if len(remaining_evo_idx) > 0:
                p_remaining = get_lgbm_pup(warmup_start, lgbm_train_end, remaining_evo_idx)
                # Update Z_evo for remaining months
                Z_remaining = build_features(remaining_evo_idx, p_remaining,
                                             np.concatenate([warmup_idx, evo_idx[:m+1]]))
                Z_evo[m+1:] = Z_remaining

            # Sort by PP, top 50% reproduce
            population.sort(key=lambda a: a.pp, reverse=True)
            n_parents = max(1, len(population) // 2)
            parents = population[:n_parents]

            # Expanding training data for CMA-ES: [warmup_start, current]
            expanding_idx = np.concatenate([warmup_idx, evo_idx[:m+1]])
            expanding_p_up = get_lgbm_pup(warmup_start, lgbm_train_end, expanding_idx)
            Z_expanding = build_features(expanding_idx, expanding_p_up, expanding_idx)
            r_next_expanding = full.loc[expanding_idx, "sp_next_return"].values
            tb_expanding = full.loc[expanding_idx, "tbill"].values / 12.0

            new_agents = []
            for parent in parents:
                if len(population) + len(new_agents) >= POP_CAP:
                    break
                child_alpha = np.clip(parent.alpha + rng.normal(0, MUTATION_SIGMA),
                                      *ALPHA_RANGE)
                child_beta = ALPHA_BETA_SUM - child_alpha
                seed_c = RNG_SEED + Agent._counter + 1000
                theta_c, b_c = train_policy(child_alpha, child_beta,
                                            Z_expanding, r_next_expanding, tb_expanding,
                                            seed=seed_c)
                child = Agent(child_alpha, child_beta, theta_c, b_c, birth_month=m)
                new_agents.append(child)
                birth_log.append(dict(
                    month=m, date=current_date_str,
                    agent_id=child.agent_id, parent_id=parent.agent_id,
                    alpha=child_alpha, beta=child_beta, lam=child.lam
                ))

            # Random immigrants
            for _ in range(IMMIGRANT_PER_YEAR):
                if len(population) + len(new_agents) >= POP_CAP:
                    break
                imm_alpha = rng.uniform(*ALPHA_RANGE)
                imm_beta = ALPHA_BETA_SUM - imm_alpha
                seed_imm = RNG_SEED + Agent._counter + 2000
                theta_imm, b_imm = train_policy(imm_alpha, imm_beta,
                                                Z_expanding, r_next_expanding, tb_expanding,
                                                seed=seed_imm)
                imm = Agent(imm_alpha, imm_beta, theta_imm, b_imm, birth_month=m)
                new_agents.append(imm)
                birth_log.append(dict(
                    month=m, date=current_date_str,
                    agent_id=imm.agent_id, parent_id=-1,
                    alpha=imm_alpha, beta=imm_beta, lam=imm.lam
                ))

            population.extend(new_agents)

            n_dead_total = len(death_log)
            n_born_total = len(birth_log)
            print(f"    Month {m:>3} ({current_date_str}): alive={len(population)}, "
                  f"dead_total={n_dead_total}, born_this_round={len(new_agents)}, "
                  f"λ mean={np.mean([a.lam for a in population]):.2f} "
                  f"median={np.median([a.lam for a in population]):.2f}")

    # ── Evolution summary ──
    print(f"\n  [4] Evolution complete.")
    print(f"  Deaths: {len(death_log)}, Births: {len(birth_log)}")
    if population:
        final_lambdas = [a.lam for a in population]
        final_pps = [a.pp for a in population]
        print(f"  Survivors: {len(population)}")
        print(f"  Survivor λ: mean={np.mean(final_lambdas):.3f}, "
              f"median={np.median(final_lambdas):.3f}, "
              f"std={np.std(final_lambdas):.3f}")
        print(f"  Survivor α: mean={np.mean([a.alpha for a in population]):.3f}")
        print(f"  Survivor β: mean={np.mean([a.beta for a in population]):.3f}")
        print(f"  Survivor PP: mean={np.mean(final_pps):.3f}, "
              f"min={np.min(final_pps):.3f}, max={np.max(final_pps):.3f}")
    else:
        print("  ALL DEAD. No survivors.")

    # ── Step 5: Test (OOS evaluation) ──
    if population and len(test_idx) > 0:
        print(f"\n  [5] Test evaluation ({len(test_idx)} months, no evolution)...")

        # LGBM trained on full [warmup_start, evo_end]
        full_train_end = evo_end
        p_test = get_lgbm_pup(warmup_start, full_train_end, test_idx)

        # Stats from full train period for standardization
        full_train_idx = np.concatenate([warmup_idx, evo_idx])
        Z_test = build_features(test_idx, p_test, full_train_idx)
        test_r_next = full.loc[test_idx, "sp_next_return"].values
        test_tb = full.loc[test_idx, "tbill"].values / 12.0
        test_m2 = full.loc[test_idx, "m2_growth"].values
        test_metab = np.maximum(test_m2, test_tb)

        # Run each survivor through test (PP reset to 1.0 for fair comparison)
        agent_test_results = []
        for agent in population:
            pp = 1.0
            ws = []
            monthly_rets = []
            for t in range(len(test_idx)):
                w = agent.decide_w(Z_test[t])
                ws.append(w)
                port_ret = w * test_r_next[t] + (1 - w) * test_tb[t]
                monthly_rets.append(port_ret)
                pp = pp * (1 + port_ret) / (1 + test_metab[t])

            ws = np.array(ws)
            monthly_rets = np.array(monthly_rets)
            eq = np.cumprod(1 + monthly_rets)
            peak = np.maximum.accumulate(eq)
            mdd = float((eq / peak - 1).min())
            n_yr = len(test_idx) / 12
            ann_ret = eq[-1] ** (1 / n_yr) - 1 if eq[-1] > 0 else -1
            vol = monthly_rets.std() * np.sqrt(12)
            sharpe = ((monthly_rets.mean() - test_tb.mean()) / monthly_rets.std()
                      * np.sqrt(12)) if monthly_rets.std() > 1e-10 else 0

            agent_test_results.append(dict(
                agent_id=agent.agent_id, alpha=agent.alpha, beta=agent.beta,
                lam=agent.lam, final_pp=pp, ann_ret=ann_ret, sharpe=sharpe,
                mdd=mdd, mean_w=ws.mean(), vol=vol
            ))

        # B&H benchmark
        bh_eq = np.cumprod(1 + test_r_next)
        bh_peak = np.maximum.accumulate(bh_eq)
        bh_mdd = float((bh_eq / bh_peak - 1).min())
        n_yr = len(test_idx) / 12
        bh_ann = bh_eq[-1] ** (1 / n_yr) - 1
        bh_vol = test_r_next.std() * np.sqrt(12)
        bh_sharpe = ((test_r_next.mean() - test_tb.mean()) / test_r_next.std()
                     * np.sqrt(12)) if test_r_next.std() > 1e-10 else 0

        # Population average
        df_test = pd.DataFrame(agent_test_results)
        print(f"\n  Survivor Test Results (n={len(df_test)}):")
        print(f"    Population avg: Ann Ret={df_test['ann_ret'].mean()*100:.2f}%, "
              f"Sharpe={df_test['sharpe'].mean():.3f}, "
              f"MDD={df_test['mdd'].mean()*100:.1f}%, "
              f"Mean W={df_test['mean_w'].mean():.2f}")
        print(f"    B&H:            Ann Ret={bh_ann*100:.2f}%, "
              f"Sharpe={bh_sharpe:.3f}, "
              f"MDD={bh_mdd*100:.1f}%")

        # Top 10 survivors by PP (from evolution period)
        population.sort(key=lambda a: a.pp, reverse=True)
        print(f"\n  Top 10 survivors (by evolution PP):")
        print(f"    {'ID':>5} {'α':>6} {'β':>6} {'λ':>6} {'EvoPP':>7} "
              f"{'TestRet':>8} {'TestSh':>7} {'TestMDD':>8} {'MeanW':>6}")
        for agent in population[:10]:
            tr = [r for r in agent_test_results if r["agent_id"] == agent.agent_id][0]
            print(f"    {agent.agent_id:>5} {agent.alpha:>6.2f} {agent.beta:>6.2f} "
                  f"{agent.lam:>6.2f} {agent.pp:>7.3f} "
                  f"{tr['ann_ret']*100:>7.2f}% {tr['sharpe']:>7.3f} "
                  f"{tr['mdd']*100:>7.1f}% {tr['mean_w']:>6.2f}")
    else:
        df_test = pd.DataFrame()

    # ── Save results ──
    df_pop = pd.DataFrame(pop_log)
    df_death = pd.DataFrame(death_log)
    df_birth = pd.DataFrame(birth_log)

    df_pop.to_csv(f"result/pop_evo_{fold_name}_population.csv", index=False)
    df_death.to_csv(f"result/pop_evo_{fold_name}_deaths.csv", index=False)
    df_birth.to_csv(f"result/pop_evo_{fold_name}_births.csv", index=False)
    if len(df_test):
        df_test.to_csv(f"result/pop_evo_{fold_name}_test.csv", index=False)

    return dict(
        fold=fold_name,
        n_survivors=len(population),
        n_deaths=len(death_log),
        n_births=len(birth_log),
        survivor_lambdas=[a.lam for a in population] if population else [],
        survivor_alphas=[a.alpha for a in population] if population else [],
        survivor_betas=[a.beta for a in population] if population else [],
        pop_log=df_pop,
        death_log=df_death,
        test_results=df_test,
    )


# ── Main ──
if __name__ == "__main__":
    t0 = time.time()
    all_results = {}

    for fold_name, fold_cfg in WINDOWS.items():
        result = run_fold(fold_name, fold_cfg)
        all_results[fold_name] = result

    # ── Cross-fold comparison ──
    print(f"\n{'='*80}")
    print("  CROSS-FOLD SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Fold':<6} {'Surv':>5} {'Dead':>5} {'Born':>5} "
          f"{'λ mean':>7} {'λ med':>7} {'λ std':>7} "
          f"{'α mean':>7} {'β mean':>7}")
    for fold_name, r in all_results.items():
        if r["survivor_lambdas"]:
            print(f"  {fold_name:<6} {r['n_survivors']:>5} {r['n_deaths']:>5} {r['n_births']:>5} "
                  f"{np.mean(r['survivor_lambdas']):>7.3f} "
                  f"{np.median(r['survivor_lambdas']):>7.3f} "
                  f"{np.std(r['survivor_lambdas']):>7.3f} "
                  f"{np.mean(r['survivor_alphas']):>7.3f} "
                  f"{np.mean(r['survivor_betas']):>7.3f}")
        else:
            print(f"  {fold_name:<6} {'ALL DEAD':>30}")

    # Phase 9 comparison
    print(f"\n  Phase 9 CMA-ES 'optimized' λ: ~1.04 (single train, Calmar fitness)")
    print(f"  Kahneman-Tversky experimental λ: 2.25")
    print(f"  위 결과와 비교: 진화로 선택된 λ는 최적화로 나온 λ와 같은가?")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed/60:.1f} min")
