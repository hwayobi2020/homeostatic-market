"""Phase 12v2: Population Evolution + MLP Policy.

CMA-ES(선형) → MLP + gradient descent(비선형)로 교체.
나머지 진화 구조(death, reproduction, α+β=3.25)는 동일.

Policy: w = sigmoid(MLP(z)) * 2, w ∈ [0, 2]
MLP: 8 → 16 → 1 (ReLU hidden)
학습: Adam, 직접 미분 가능한 reward objective

한계:
  - 동일 market path 위 병렬 평가 (agent 상호작용 없음)
  - Evolution 구간 120개월, major crash 1~2회
  - MLP가 train data에 overfit 가능 (특히 warmup이 짧을 때)
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

r = full["sp_next_return"].fillna(0).values
y3 = np.full(N, np.nan)
for t in range(N - 3):
    y3[t] = (1 + r[t]) * (1 + r[t+1]) * (1 + r[t+2]) - 1.0
full["y3"] = y3
full["label"] = (full["y3"] >= 0.10).astype(float)

EXCLUDE = {"date", "sp_next_return", "label", "y3"}
LGBM_FEATURES = [c for c in full.columns if c not in EXCLUDE]
POLICY_FEATS = ["vix", "sp_1m", "ndx_in_range", "ndx_52wh_ratio",
                "metabolism", "tbill", "sp_6m_mean"]
FEAT_DIM = len(POLICY_FEATS) + 1  # +1 for p_up

WINDOWS = {
    "W1": dict(warmup=("1991-01-01", "2000-12-31"),
               evo=("2001-01-01", "2010-12-31"),
               test=("2011-07-01", "2015-12-31")),
    "W2": dict(warmup=("1996-01-01", "2005-12-31"),
               evo=("2006-01-01", "2015-12-31"),
               test=("2016-07-01", "2020-12-31")),
    "W3": dict(warmup=("2001-01-01", "2010-12-31"),
               evo=("2011-01-01", "2020-12-31"),
               test=("2021-07-01", "2025-12-31")),
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
    tr_mask = ((full["date"] >= train_start) & (full["date"] <= train_end)
               & (full["label"].notna()))
    tr_idx = np.where(tr_mask)[0]
    X_all = full[LGBM_FEATURES].values
    y_all = full["label"].values
    Xtr = X_all[tr_idx]; ytr = y_all[tr_idx].astype(int)
    model = fit_lgbm(Xtr, ytr)
    p = model.predict(X_all[predict_idx], num_iteration=model.best_iteration)
    return p


def build_features(idx, p_up_arr, train_idx_for_stats):
    raw = full.loc[idx, POLICY_FEATS].values.astype(float)
    ref = full.loc[train_idx_for_stats, POLICY_FEATS].values.astype(float)
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0) + 1e-8
    Z_feat = np.nan_to_num((raw - mu) / sd, nan=0.0)
    Z = np.column_stack([p_up_arr.reshape(-1, 1), Z_feat])
    return Z


# ── MLP Policy (pure numpy, no PyTorch) ──
def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


class PolicyMLP:
    """2-layer MLP: input → hidden (ReLU) → output (sigmoid * 2).
    Forward:  h = relu(Z @ W1 + b1),  w = sigmoid(h @ W2 + b2) * 2
    """
    def __init__(self, input_dim, hidden_dim=16, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        scale1 = np.sqrt(2.0 / input_dim)  # He init
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.normal(0, scale1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale2, (hidden_dim, 1))
        self.b2 = np.array([-1.0])  # bias init: w starts near 1.0

    def forward(self, Z):
        """Z: [T, input_dim] → w: [T]"""
        self.Z = Z
        self.pre_relu = Z @ self.W1 + self.b1           # [T, H]
        self.H = np.maximum(self.pre_relu, 0)            # ReLU
        self.pre_sig = self.H @ self.W2 + self.b2        # [T, 1]
        self.S = sigmoid(self.pre_sig.ravel())            # [T]
        return 1.0 + self.S                                     # w ∈ [1, 2]

    def predict_one(self, z_row):
        """Single observation → w scalar. Caches internals for online update."""
        self._z = z_row
        self._pre_relu = z_row @ self.W1 + self.b1
        self._h = np.maximum(self._pre_relu, 0)
        self._pre_sig = (self._h @ self.W2 + self.b2).item()
        self._s = sigmoid(self._pre_sig)
        return 1.0 + self._s                                     # w ∈ [1, 2]

    def online_update(self, dL_dw, lr=0.001):
        """Single-sample gradient update using cached forward pass."""
        # w = 1 + s, dL_ds = dL_dw
        dL_ds = dL_dw
        # s = sigmoid(pre_sig)
        dL_dpresig = dL_ds * self._s * (1.0 - self._s)
        # pre_sig = h @ W2 + b2
        dL_dW2 = self._h.reshape(-1, 1) * dL_dpresig         # [H, 1]
        dL_db2 = np.array([dL_dpresig])
        dL_dh = self.W2.ravel() * dL_dpresig                  # [H]
        # h = relu(pre_relu)
        dL_dprerelu = dL_dh * (self._pre_relu > 0)           # [H]
        # pre_relu = z @ W1 + b1
        dL_dW1 = self._z.reshape(-1, 1) * dL_dprerelu        # [D, H]
        dL_db1 = dL_dprerelu                                  # [H]
        # SGD update
        self.W1 -= lr * dL_dW1
        self.b1 -= lr * dL_db1
        self.W2 -= lr * dL_dW2
        self.b2 -= lr * dL_db2

    def backward(self, dL_dw):
        """dL_dw: [T] gradient of loss w.r.t. w output."""
        T = len(dL_dw)
        # w = 1 + S → dL_dS = dL_dw
        dL_dS = dL_dw
        # S = sigmoid(pre_sig) → dL_dpresig = dL_dS * S * (1-S)
        dL_dpresig = (dL_dS * self.S * (1.0 - self.S)).reshape(T, 1)  # [T,1]
        # pre_sig = H @ W2 + b2
        dL_dW2 = self.H.T @ dL_dpresig                   # [H, 1]
        dL_db2 = dL_dpresig.sum(axis=0)                   # [1]
        dL_dH = dL_dpresig @ self.W2.T                    # [T, H]
        # H = relu(pre_relu) → dL_dprerelu = dL_dH * (pre_relu > 0)
        dL_dprerelu = dL_dH * (self.pre_relu > 0)        # [T, H]
        # pre_relu = Z @ W1 + b1
        dL_dW1 = self.Z.T @ dL_dprerelu                  # [D, H]
        dL_db1 = dL_dprerelu.sum(axis=0)                  # [H]
        return dL_dW1, dL_db1, dL_dW2, dL_db2


def train_policy_mlp(alpha, beta, Z_train, r_next_train, tb_train, metab_train=None,
                     epochs=300, lr=0.005, seed=42):
    rng = np.random.default_rng(seed)
    model = PolicyMLP(input_dim=Z_train.shape[1], rng=rng)
    if metab_train is None:
        metab_train = tb_train
    excess = r_next_train - metab_train  # vs metabolism

    # Adam state
    params = [model.W1, model.b1, model.W2, model.b2]
    m_state = [np.zeros_like(p) for p in params]
    v_state = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        w = model.forward(Z_train)       # [T]
        r_net = w * excess               # w * (sp - tb)

        # reward = alpha * relu(r_net) - beta * relu(-r_net)
        # loss = -sum(reward)
        # dL/dw = -(alpha * (r_net>0) * excess - beta * (r_net<0) * (-excess))
        #       = -(alpha * (r_net>0) * excess + beta * (r_net<0) * excess)
        # Wait: d/dw [relu(w*e)] = e if w*e > 0 else 0
        # d/dw [-relu(-w*e)] = e if w*e < 0 else 0
        # So d_reward/dw = alpha*e*(r_net>=0) + beta*e*(r_net<0)
        #                  (beta term: -beta * d/dw[relu(-w*e)] = -beta*(-e)*(r_net<0) = beta*e*(r_net<0))
        # Actually: reward = alpha*relu(w*e) - beta*relu(-w*e)
        # d/dw relu(w*e) = e * (w*e > 0)
        # d/dw relu(-w*e) = -e * (-w*e > 0) = -e * (w*e < 0)
        # So d_reward/dw = alpha*e*(we>0) - beta*(-e)*(we<0)
        #                = alpha*e*(we>0) + beta*e*(we<0)
        # dL/dw = -d_reward/dw

        pos_mask = (r_net >= 0).astype(float)
        neg_mask = (r_net < 0).astype(float)
        dL_dw = -(alpha * excess * pos_mask + beta * excess * neg_mask)

        grads_list = model.backward(dL_dw)
        # Adam update
        t_step = epoch + 1
        for i, (param, grad) in enumerate(zip(params, grads_list)):
            m_state[i] = beta1 * m_state[i] + (1 - beta1) * grad
            v_state[i] = beta2 * v_state[i] + (1 - beta2) * grad ** 2
            m_hat = m_state[i] / (1 - beta1 ** t_step)
            v_hat = v_state[i] / (1 - beta2 ** t_step)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            param -= update
            # Write back (numpy arrays are mutable, but be explicit)
            if i == 0: model.W1 = param
            elif i == 1: model.b1 = param
            elif i == 2: model.W2 = param
            elif i == 3: model.b2 = param

    return model


# ── Agent ──
class Agent:
    _counter = 0

    def __init__(self, alpha, beta, model, birth_month=0):
        Agent._counter += 1
        self.agent_id = Agent._counter
        self.alpha = alpha
        self.beta = beta
        self.model = model
        self.pp = 1.0
        self.birth_month = birth_month
        self.monthly_rets = []  # track for Sharpe computation

    @property
    def lam(self):
        return self.beta / self.alpha if self.alpha > 0 else float("inf")

    @property
    def sharpe(self):
        if len(self.monthly_rets) < 3:
            return -999.0
        r = np.array(self.monthly_rets)
        if r.std() < 1e-10:
            return 0.0
        return r.mean() / r.std() * np.sqrt(12)

    @property
    def calmar(self):
        if len(self.monthly_rets) < 3:
            return -999.0
        r = np.array(self.monthly_rets)
        eq = np.cumprod(1 + r)
        peak = np.maximum.accumulate(eq)
        mdd = float((eq / peak - 1).min())
        if mdd >= 0:
            return -999.0
        n_yr = len(r) / 12
        ann = eq[-1] ** (1 / n_yr) - 1 if eq[-1] > 0 else -1
        return ann / abs(mdd)

    def decide_w(self, z_row):
        return self.model.predict_one(z_row)


# ── Evolution parameters ──
POP_INIT = 200
DEATH_THRESHOLD = 0.50
MUTATION_SIGMA = 0.30
ALPHA_BETA_SUM = 3.25
ALPHA_RANGE = (0.25, 3.00)
REPRODUCTION_INTERVAL = 12
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

    warmup_mask = (full["date"] >= warmup_start) & (full["date"] <= warmup_end)
    evo_mask = (full["date"] >= evo_start) & (full["date"] <= evo_end)
    test_mask = (full["date"] >= test_start) & (full["date"] <= test_end)
    warmup_idx = np.where(warmup_mask)[0]
    evo_idx = np.where(evo_mask)[0]
    test_idx = np.where(test_mask)[0]

    print(f"  Warmup: {warmup_start} ~ {warmup_end} ({len(warmup_idx)} months)")
    print(f"  Evolution: {evo_start} ~ {evo_end} ({len(evo_idx)} months)")
    print(f"  Test: {test_start} ~ {test_end} ({len(test_idx)} months)")

    # ── LGBM ──
    print("\n  [1] LGBM training...")
    warmup_evo_idx = np.concatenate([warmup_idx, evo_idx])
    p_warmup_evo = get_lgbm_pup(warmup_start, warmup_end, warmup_evo_idx)
    p_warmup = p_warmup_evo[:len(warmup_idx)]
    p_evo = p_warmup_evo[len(warmup_idx):]

    Z_warmup = build_features(warmup_idx, p_warmup, warmup_idx)
    r_next_warmup = full.loc[warmup_idx, "sp_next_return"].values
    tb_warmup = full.loc[warmup_idx, "tbill"].values / 12.0
    m2_warmup = full.loc[warmup_idx, "m2_growth"].values
    metab_warmup = np.maximum(m2_warmup, tb_warmup)

    # ── Initialize population ──
    print(f"\n  [2] Initializing {POP_INIT} agents (MLP training)...")
    t_init = time.time()
    population = []
    for i in range(POP_INIT):
        alpha = rng.uniform(*ALPHA_RANGE)
        beta = ALPHA_BETA_SUM - alpha
        seed_i = RNG_SEED + i
        model = train_policy_mlp(alpha, beta, Z_warmup, r_next_warmup, tb_warmup,
                                 metab_train=metab_warmup, seed=seed_i)
        agent = Agent(alpha, beta, model, birth_month=0)
        population.append(agent)
    print(f"  Init time: {time.time()-t_init:.1f}s")

    init_lambdas = [a.lam for a in population]
    print(f"  Initial lambda: mean={np.mean(init_lambdas):.2f}, "
          f"median={np.median(init_lambdas):.2f}")

    # ── Evolution ──
    print(f"\n  [3] Evolution ({len(evo_idx)} months)...")

    evo_dates = full.loc[evo_idx, "date"].values
    evo_r_next = full.loc[evo_idx, "sp_next_return"].values
    evo_tbill = full.loc[evo_idx, "tbill"].values / 12.0
    evo_m2 = full.loc[evo_idx, "m2_growth"].values
    evo_metab = np.maximum(evo_m2, evo_tbill)
    Z_evo = build_features(evo_idx, p_evo, warmup_idx)

    death_log = []
    pop_log = []
    months_since_repro = 0

    for m in range(len(evo_idx)):
        month_date = evo_dates[m]
        z_row = Z_evo[m]
        r_next_m = evo_r_next[m]
        tb_m = evo_tbill[m]
        metab_m = evo_metab[m]

        for agent in population:
            w = agent.decide_w(z_row)
            port_ret = w * r_next_m + (1 - w) * tb_m
            agent.monthly_rets.append(port_ret)
            agent.pp = agent.pp * (1 + port_ret) / (1 + metab_m)

            # Online gradient update: agent observes result, updates MLP
            excess_m = r_next_m - metab_m
            r_net = w * excess_m
            if r_net >= 0:
                dL_dw = -agent.alpha * excess_m
            else:
                dL_dw = -agent.beta * excess_m
            agent.model.online_update(dL_dw, lr=0.001)

        dead = [a for a in population if a.pp < DEATH_THRESHOLD]
        for a in dead:
            death_log.append(dict(month=m, date=str(month_date)[:10],
                                  agent_id=a.agent_id, alpha=a.alpha, beta=a.beta,
                                  lam=a.lam, pp=a.pp))
        population = [a for a in population if a.pp >= DEATH_THRESHOLD]

        if population:
            lambdas = [a.lam for a in population]
            pps = [a.pp for a in population]
            pop_log.append(dict(month=m, date=str(month_date)[:10],
                                n_alive=len(population),
                                mean_lam=np.mean(lambdas), median_lam=np.median(lambdas),
                                mean_pp=np.mean(pps)))

        months_since_repro += 1

        if months_since_repro >= REPRODUCTION_INTERVAL and len(population) > 0:
            months_since_repro = 0
            current_date_str = str(month_date)[:10]

            # Retrain LGBM expanding
            remaining_evo_idx = evo_idx[m+1:]
            if len(remaining_evo_idx) > 0:
                p_remaining = get_lgbm_pup(warmup_start, current_date_str, remaining_evo_idx)
                Z_remaining = build_features(remaining_evo_idx, p_remaining,
                                             np.concatenate([warmup_idx, evo_idx[:m+1]]))
                Z_evo[m+1:] = Z_remaining

            # Expanding training data
            expanding_idx = np.concatenate([warmup_idx, evo_idx[:m+1]])
            expanding_p_up = get_lgbm_pup(warmup_start, current_date_str, expanding_idx)
            Z_expanding = build_features(expanding_idx, expanding_p_up, expanding_idx)
            r_next_expanding = full.loc[expanding_idx, "sp_next_return"].values
            tb_expanding = full.loc[expanding_idx, "tbill"].values / 12.0
            m2_expanding = full.loc[expanding_idx, "m2_growth"].values
            metab_expanding = np.maximum(m2_expanding, tb_expanding)

            population.sort(key=lambda a: a.calmar, reverse=True)
            n_parents = max(1, len(population) // 2)
            parents = population[:n_parents]

            new_agents = []
            for parent in parents:
                if len(population) + len(new_agents) >= POP_CAP:
                    break
                child_alpha = np.clip(parent.alpha + rng.normal(0, MUTATION_SIGMA),
                                      *ALPHA_RANGE)
                child_beta = ALPHA_BETA_SUM - child_alpha
                seed_c = RNG_SEED + Agent._counter + 1000
                model_c = train_policy_mlp(child_alpha, child_beta,
                                           Z_expanding, r_next_expanding, tb_expanding,
                                           metab_train=metab_expanding, seed=seed_c)
                child = Agent(child_alpha, child_beta, model_c, birth_month=m)
                new_agents.append(child)

            for _ in range(IMMIGRANT_PER_YEAR):
                if len(population) + len(new_agents) >= POP_CAP:
                    break
                imm_alpha = rng.uniform(*ALPHA_RANGE)
                imm_beta = ALPHA_BETA_SUM - imm_alpha
                seed_imm = RNG_SEED + Agent._counter + 2000
                model_imm = train_policy_mlp(imm_alpha, imm_beta,
                                             Z_expanding, r_next_expanding, tb_expanding,
                                             metab_train=metab_expanding,
                                             seed=seed_imm)
                imm = Agent(imm_alpha, imm_beta, model_imm, birth_month=m)
                new_agents.append(imm)

            population.extend(new_agents)
            calmars = [a.calmar for a in population if a.calmar > -999]
            print(f"    Month {m:>3} ({current_date_str}): alive={len(population)}, "
                  f"dead={len(death_log)}, born={len(new_agents)}, "
                  f"lam med={np.median([a.lam for a in population]):.2f}, "
                  f"calmar med={np.median(calmars):.2f}")

    # ── Evolution summary ──
    print(f"\n  [4] Evolution complete. Deaths={len(death_log)}")
    if population:
        final_lambdas = [a.lam for a in population]
        final_pps = [a.pp for a in population]
        print(f"  Survivors: {len(population)}")
        print(f"  lambda: mean={np.mean(final_lambdas):.3f}, "
              f"median={np.median(final_lambdas):.3f}, std={np.std(final_lambdas):.3f}")
        print(f"  alpha: mean={np.mean([a.alpha for a in population]):.3f}")
        print(f"  beta: mean={np.mean([a.beta for a in population]):.3f}")
        print(f"  PP: mean={np.mean(final_pps):.3f}, "
              f"min={np.min(final_pps):.3f}, max={np.max(final_pps):.3f}")
    else:
        print("  ALL DEAD.")

    # ── Test ──
    if population and len(test_idx) > 0:
        print(f"\n  [5] Test ({len(test_idx)} months)...")
        p_test = get_lgbm_pup(warmup_start, evo_end, test_idx)
        full_train_idx = np.concatenate([warmup_idx, evo_idx])
        Z_test = build_features(test_idx, p_test, full_train_idx)
        test_r_next = full.loc[test_idx, "sp_next_return"].values
        test_tb = full.loc[test_idx, "tbill"].values / 12.0
        test_m2 = full.loc[test_idx, "m2_growth"].values
        test_metab = np.maximum(test_m2, test_tb)

        agent_results = []
        for agent in population:
            pp = 1.0; ws = []; rets = []
            for t in range(len(test_idx)):
                w = agent.decide_w(Z_test[t])
                ws.append(w)
                port_ret = w * test_r_next[t] + (1 - w) * test_tb[t]
                rets.append(port_ret)
                pp = pp * (1 + port_ret) / (1 + test_metab[t])
                # Online update in test too
                excess_t = test_r_next[t] - test_metab[t]
                r_net_t = w * excess_t
                if r_net_t >= 0:
                    dL_dw = -agent.alpha * excess_t
                else:
                    dL_dw = -agent.beta * excess_t
                agent.model.online_update(dL_dw, lr=0.001)

            ws = np.array(ws); rets = np.array(rets)
            eq = np.cumprod(1 + rets)
            peak = np.maximum.accumulate(eq)
            mdd = float((eq / peak - 1).min())
            n_yr = len(test_idx) / 12
            ann = eq[-1] ** (1/n_yr) - 1 if eq[-1] > 0 else -1
            sh = ((rets.mean() - test_tb.mean()) / rets.std()
                  * np.sqrt(12)) if rets.std() > 1e-10 else 0

            # corr with p_up
            corr_pup = np.corrcoef(p_test, ws)[0, 1] if ws.std() > 1e-6 else 0

            agent_results.append(dict(
                agent_id=agent.agent_id, alpha=agent.alpha, beta=agent.beta,
                lam=agent.lam, evo_pp=agent.pp, final_pp=pp,
                ann_ret=ann, sharpe=sh, mdd=mdd, mean_w=ws.mean(),
                corr_pup=corr_pup
            ))

        # B&H + static leverage benchmarks
        bh_eq = np.cumprod(1 + test_r_next)
        bh_mdd = float((bh_eq / np.maximum.accumulate(bh_eq) - 1).min())
        n_yr = len(test_idx) / 12
        bh_ann = bh_eq[-1] ** (1/n_yr) - 1
        bh_sh = ((test_r_next.mean() - test_tb.mean()) / test_r_next.std()
                 * np.sqrt(12)) if test_r_next.std() > 1e-10 else 0

        # 50/50
        r5050 = 0.5 * test_r_next + 0.5 * test_tb
        eq5050 = np.cumprod(1 + r5050)
        mdd5050 = float((eq5050 / np.maximum.accumulate(eq5050) - 1).min())
        ann5050 = eq5050[-1] ** (1/n_yr) - 1
        sh5050 = ((r5050.mean() - test_tb.mean()) / r5050.std()
                  * np.sqrt(12)) if r5050.std() > 1e-10 else 0

        df_r = pd.DataFrame(agent_results)

        # Add evo_calmar from agent objects
        evo_calmar_map = {a.agent_id: a.calmar for a in population}
        df_r["evo_calmar"] = df_r["agent_id"].map(evo_calmar_map)

        # === Top by Evo Calmar (10%, 20%) ===
        df_cal = df_r.sort_values("evo_calmar", ascending=False)
        n_10 = max(1, len(df_r) // 10)
        n_20 = max(1, len(df_r) // 5)
        top10 = df_cal.head(n_10)
        top20 = df_cal.head(n_20)
        top10_w = top10["mean_w"].mean()
        top20_w = top20["mean_w"].mean()

        def static_metrics(w_fixed):
            rs = w_fixed * test_r_next + (1 - w_fixed) * test_tb
            eq = np.cumprod(1 + rs)
            mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
            ann = eq[-1] ** (1/n_yr) - 1
            sh = (rs.mean() - test_tb.mean()) / rs.std() * np.sqrt(12) if rs.std() > 1e-10 else 0
            return ann, sh, mdd

        s10_ann, s10_sh, s10_mdd = static_metrics(top10_w)
        s20_ann, s20_sh, s20_mdd = static_metrics(top20_w)

        # === Top 5 by Evo PP ===
        df_pp = df_r.sort_values("evo_pp", ascending=False)
        top5_pp = df_pp.head(5)
        top5_pp_w = top5_pp["mean_w"].mean()

        r_static_pp = top5_pp_w * test_r_next + (1 - top5_pp_w) * test_tb
        eq_static_pp = np.cumprod(1 + r_static_pp)
        mdd_static_pp = float((eq_static_pp / np.maximum.accumulate(eq_static_pp) - 1).min())
        ann_static_pp = eq_static_pp[-1] ** (1/n_yr) - 1
        sh_static_pp = ((r_static_pp.mean() - test_tb.mean()) / r_static_pp.std()
                        * np.sqrt(12)) if r_static_pp.std() > 1e-10 else 0

        print(f"\n  {'':>25} {'Return':>8} {'Sharpe':>8} {'MDD':>8} {'MeanW':>7} {'lam':>6}")
        print(f"  {'Top10% by EvoCalmar':<25} {top10['ann_ret'].mean()*100:>7.2f}% {top10['sharpe'].mean():>8.3f} {top10['mdd'].mean()*100:>7.1f}% {top10_w:>7.2f} {top10['lam'].median():>6.2f}")
        print(f"  {'  Static same W':<25} {s10_ann*100:>7.2f}% {s10_sh:>8.3f} {s10_mdd*100:>7.1f}% {top10_w:>7.2f}")
        print(f"  {'Top20% by EvoCalmar':<25} {top20['ann_ret'].mean()*100:>7.2f}% {top20['sharpe'].mean():>8.3f} {top20['mdd'].mean()*100:>7.1f}% {top20_w:>7.2f} {top20['lam'].median():>6.2f}")
        print(f"  {'  Static same W':<25} {s20_ann*100:>7.2f}% {s20_sh:>8.3f} {s20_mdd*100:>7.1f}% {top20_w:>7.2f}")
        print(f"  {'Top5 by EvoPP':<25} {top5_pp['ann_ret'].mean()*100:>7.2f}% {top5_pp['sharpe'].mean():>8.3f} {top5_pp['mdd'].mean()*100:>7.1f}% {top5_pp_w:>7.2f} {top5_pp['lam'].mean():>6.2f}")
        print(f"  {'  Static same W':<25} {ann_static_pp*100:>7.2f}% {sh_static_pp:>8.3f} {mdd_static_pp*100:>7.1f}% {top5_pp_w:>7.2f}")
        print(f"  {'B&H (w=1)':<25} {bh_ann*100:>7.2f}% {bh_sh:>8.3f} {bh_mdd*100:>7.1f}% {'1.00':>7}")
        print(f"  {'50/50':<25} {ann5050*100:>7.2f}% {sh5050:>8.3f} {mdd5050*100:>7.1f}% {'0.50':>7}")

        print(f"\n  Top 10 by Evo Calmar:")
        print(f"    {'ID':>5} {'a':>5} {'b':>5} {'lam':>5} {'EvoCal':>7} {'EvoPP':>7} "
              f"{'Ret':>7} {'TeSh':>6} {'MDD':>7} {'W':>5} {'corr':>6}")
        for _, row in df_cal.head(10).iterrows():
            print(f"    {int(row['agent_id']):>5} {row['alpha']:>5.2f} {row['beta']:>5.2f} "
                  f"{row['lam']:>5.2f} {row['evo_calmar']:>7.3f} {row['evo_pp']:>7.3f} "
                  f"{row['ann_ret']*100:>6.1f}% {row['sharpe']:>6.3f} "
                  f"{row['mdd']*100:>6.1f}% {row['mean_w']:>5.2f} {row['corr_pup']:>6.3f}")

        df_r.to_csv(f"result/pop_evo_mlp_{fold_name}_test.csv", index=False)

    pd.DataFrame(pop_log).to_csv(f"result/pop_evo_mlp_{fold_name}_population.csv", index=False)
    pd.DataFrame(death_log).to_csv(f"result/pop_evo_mlp_{fold_name}_deaths.csv", index=False)

    return dict(fold=fold_name, n_survivors=len(population), n_deaths=len(death_log),
                survivor_lambdas=[a.lam for a in population] if population else [])


# ── Main ──
if __name__ == "__main__":
    t0 = time.time()
    results = {}
    for fold_name, fold_cfg in WINDOWS.items():
        results[fold_name] = run_fold(fold_name, fold_cfg)

    print(f"\n{'='*80}")
    print("  CROSS-FOLD SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Fold':<6} {'Surv':>5} {'Dead':>5} {'lam mean':>9} {'lam med':>9}")
    for fn, r in results.items():
        if r["survivor_lambdas"]:
            print(f"  {fn:<6} {r['n_survivors']:>5} {r['n_deaths']:>5} "
                  f"{np.mean(r['survivor_lambdas']):>9.3f} "
                  f"{np.median(r['survivor_lambdas']):>9.3f}")
        else:
            print(f"  {fn:<6} {'ALL DEAD':>20}")

    print(f"\n  Phase 12 linear (CMA-ES): lambda median W1=0.60, W2=0.41, W3=0.41")
    print(f"  Phase 9 CMA-ES optimized: lambda ~1.04")
    print(f"  Kahneman: lambda 2.25")
    print(f"\n  Total: {(time.time()-t0)/60:.1f} min")
