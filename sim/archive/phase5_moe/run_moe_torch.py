"""
MoE Pre-trained (Pure PyTorch)
Phase 1: Expert 1 (HWM) 단독 100K → Freeze
Phase 2: Expert 2 (Return) 단독 100K → Freeze
Phase 3: Router (HWM on actual portfolio) 단독 100K
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
device = torch.device("cpu")


# === Actor-Critic (Continuous: Expert용) ===
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_low=0.0, act_high=1.0):
        super().__init__()
        self.act_low = act_low
        self.act_high = act_high
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(64, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        x = self.shared(obs)
        mean = torch.sigmoid(self.actor_mean(x))
        mean = mean * (self.act_high - self.act_low) + self.act_low
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(x)
        return mean, std, value

    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()
        action = torch.clamp(action, self.act_low, self.act_high)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate(self, obs, action):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value.squeeze(-1), entropy


# === Router Actor-Critic (Discrete: Expert 0 or 1) ===
class RouterActorCritic(nn.Module):
    def __init__(self, obs_dim, n_experts=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor_logits = nn.Linear(64, n_experts)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        x = self.shared(obs)
        logits = self.actor_logits(x)
        value = self.critic(x)
        return logits, value

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate(self, obs, action):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action.long())
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy


# === Rollout Buffer ===
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, gamma=0.99, lam=0.95):
        returns = []
        advantages = []
        gae = 0
        values = self.values + [0.0]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return returns, advantages

    def get_tensors(self, gamma=0.99, lam=0.95):
        returns, advantages = self.compute_returns(gamma, lam)
        obs = torch.stack(self.obs)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return obs, actions, old_log_probs, returns, advantages

    def clear(self):
        self.__init__()


# === PPO Update ===
def ppo_update(model, optimizer, buffer, epochs=10, clip_range=0.2, ent_coef=0.1,
               gamma=0.99, batch_size=64):
    obs, actions, old_log_probs, returns, advantages = buffer.get_tensors(gamma)
    n = len(obs)
    for _ in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            log_probs, values, entropy = model.evaluate(obs[idx], actions[idx])
            ratio = (log_probs - old_log_probs[idx]).exp()
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns[idx] - values).pow(2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + 0.5 * value_loss + ent_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    buffer.clear()


# === Observation ===
def make_obs(data, idx, pp, hwm, last_action):
    row = data.iloc[min(idx, len(data) - 1)]
    return torch.tensor([
        pp - hwm, pp - 1.0,
        float(row["ndx_1m"]), float(row["sp_1m"]),
        float(row["ndx_3m"]), float(row["vix"]) / 100,
        float(row["m2_3m"]), float(row["sentiment"]),
        last_action, pp / max(hwm, 1e-8),
    ], dtype=torch.float32)


def make_router_obs(data, idx, pp, hwm, e1_act, e2_act, last_choice):
    row = data.iloc[min(idx, len(data) - 1)]
    return torch.tensor([
        pp - hwm, pp - 1.0,
        e1_act, e2_act,
        float(row["vix"]) / 100, float(row["m2_3m"]),
        float(row["sentiment"]), float(row["sp_1m"]),
        float(row["ndx_1m"]), last_choice,
    ], dtype=torch.float32)


def compute_metrics(rets, name):
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100
    vol = rets.std() * np.sqrt(12) * 100
    sharpe = rets.mean() / max(rets.std(), 1e-8) * np.sqrt(12)
    ds = rets[rets < 0]
    ds_std = np.sqrt(np.mean(ds ** 2)) if len(ds) > 0 else 1e-8
    sortino = rets.mean() / ds_std * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    win = (rets > 0).mean() * 100
    print(f"{name:>20}: ret={ann:+6.2f}% vol={vol:5.1f}% sharpe={sharpe:+5.2f} "
          f"sortino={sortino:+5.2f} mdd={mdd:5.1f}% win={win:.0f}%", flush=True)


# ===================================================================
# Phase 1 & 2: Expert 단독 학습
# ===================================================================
def train_expert(train_data, reward_type, n_episodes, ep_len, premium, ent_coef, seed):
    """Expert를 단독으로 학습시킨다. reward_type: 'hwm' or 'return'"""
    n_train = len(train_data)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ActorCritic(10, 0.0, 1.0)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    buf = RolloutBuffer()
    rng = np.random.RandomState(seed)

    for ep in range(n_episodes):
        max_start = n_train - ep_len
        start_idx = rng.randint(0, max(1, max_start))

        pp = 1.0
        hwm = 1.0
        last_act = 0.5

        for t in range(ep_len):
            idx = start_idx + t
            if idx >= n_train:
                break

            row = train_data.iloc[idx]
            next_ret = float(row["sp_next_return"])
            tb_r = float(row["tbill"])
            met = float(row["metabolism"]) + premium

            obs = make_obs(train_data, idx, pp, hwm, last_act)
            with torch.no_grad():
                act, lp, val = model.get_action(obs.unsqueeze(0))
            a = float(act.squeeze().clamp(0, 1))

            port_ret = a * next_ret + (1 - a) * tb_r
            pp = pp * (1 + port_ret) / (1 + met)

            if reward_type == "hwm":
                reward = pp - hwm  # 이전 HWM 기준 (새 고점이면 양수!)
            else:  # return
                reward = port_ret

            hwm = max(hwm, pp)  # reward 계산 후에 HWM 업데이트

            done = (t == ep_len - 1)
            buf.add(obs, act.squeeze(), lp.squeeze(), reward, val.squeeze().item(), float(done))
            last_act = a

        if len(buf.obs) >= 64:
            ppo_update(model, optimizer, buf, ent_coef=ent_coef)

        if (ep + 1) % 200 == 0:
            print(f"    Ep {ep+1}: pp={pp:.4f} hwm={hwm:.4f} act={last_act:.2f}", flush=True)

    return model


# ===================================================================
# Phase 3: Router 학습 (Expert Frozen)
# ===================================================================
def train_router(train_data, expert1, expert2, n_episodes, ep_len, premium, ent_coef, seed):
    """Frozen Expert 위에 Router만 학습. Reward = actual PP - actual HWM."""
    n_train = len(train_data)
    router = RouterActorCritic(10, n_experts=2)
    optimizer = optim.Adam(router.parameters(), lr=3e-4)
    buf = RolloutBuffer()
    rng = np.random.RandomState(seed)

    expert1.eval()
    expert2.eval()

    for ep in range(n_episodes):
        max_start = n_train - ep_len
        start_idx = rng.randint(0, max(1, max_start))

        pp = 1.0
        hwm = 1.0
        last_e1 = 0.5
        last_e2 = 0.5
        last_choice = 0.0

        for t in range(ep_len):
            idx = start_idx + t
            if idx >= n_train:
                break

            row = train_data.iloc[idx]
            next_ret = float(row["sp_next_return"])
            tb_r = float(row["tbill"])
            met = float(row["metabolism"]) + premium

            # Frozen Expert actions (각자 실제 PP/HWM 기반 관찰)
            e1_obs = make_obs(train_data, idx, pp, hwm, last_e1)
            e2_obs = make_obs(train_data, idx, pp, hwm, last_e2)
            with torch.no_grad():
                e1_act, _, _ = expert1.get_action(e1_obs.unsqueeze(0))
                e2_act, _, _ = expert2.get_action(e2_obs.unsqueeze(0))
            e1_a = float(e1_act.squeeze().clamp(0, 1))
            e2_a = float(e2_act.squeeze().clamp(0, 1))

            # Router 선택
            r_obs = make_router_obs(train_data, idx, pp, hwm, e1_a, e2_a, last_choice)
            with torch.no_grad():
                r_act, r_lp, r_val = router.get_action(r_obs.unsqueeze(0))
            choice = int(r_act.squeeze())

            # 선택된 Expert의 action 실행
            final_act = e1_a if choice == 0 else e2_a
            port_ret = final_act * next_ret + (1 - final_act) * tb_r
            new_pp = pp * (1 + port_ret) / (1 + met)

            # Router Reward: 이전 HWM 기준 (새 고점이면 양수!)
            r_reward = new_pp - hwm
            new_hwm = max(hwm, new_pp)

            done = (t == ep_len - 1)
            buf.add(r_obs, r_act.squeeze(), r_lp.squeeze(), r_reward, r_val.squeeze().item(), float(done))

            pp = new_pp
            hwm = new_hwm
            last_e1 = e1_a
            last_e2 = e2_a
            last_choice = float(choice)

        if len(buf.obs) >= 64:
            ppo_update(router, optimizer, buf, ent_coef=ent_coef)

        if (ep + 1) % 200 == 0:
            print(f"    Ep {ep+1}: pp={pp:.4f} hwm={hwm:.4f} "
                  f"choice={'E1' if last_choice == 0 else 'E2'}", flush=True)

    return router


# ===================================================================
# Main
# ===================================================================
def main():
    TRAIN_PATH = "data/monthly_noleak_train.csv"
    TEST_PATH = "data/monthly_noleak_test.csv"
    PREMIUM = 0.02 / 12
    EP_LEN = 120
    N_EP_EXPERT = 834   # ~100K steps (834 × 120 = 100,080)
    N_EP_ROUTER = 834   # ~100K steps
    ENT_COEF = 0.1

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    # === Phase 1: Expert 1 (HWM) 단독 학습 ===
    print("=" * 60, flush=True)
    print("Phase 1: Expert 1 (HWM Reward) 단독 학습...", flush=True)
    print("=" * 60, flush=True)
    expert1 = train_expert(train_data, "hwm", N_EP_EXPERT, EP_LEN, PREMIUM, ENT_COEF, seed=42)

    # === Phase 2: Expert 2 (Return) 단독 학습 ===
    print("=" * 60, flush=True)
    print("Phase 2: Expert 2 (Return Reward) 단독 학습...", flush=True)
    print("=" * 60, flush=True)
    expert2 = train_expert(train_data, "return", N_EP_EXPERT, EP_LEN, PREMIUM, ENT_COEF, seed=123)

    # === Freeze ===
    for p in expert1.parameters():
        p.requires_grad = False
    for p in expert2.parameters():
        p.requires_grad = False
    print("\nExpert 1, 2 Frozen.\n", flush=True)

    # === Phase 3: Router 학습 ===
    print("=" * 60, flush=True)
    print("Phase 3: Router (HWM on actual portfolio) 학습...", flush=True)
    print("=" * 60, flush=True)
    router = train_router(train_data, expert1, expert2, N_EP_ROUTER, EP_LEN, PREMIUM, ENT_COEF, seed=777)

    # === 테스트 ===
    print("\n" + "=" * 60, flush=True)
    print("Testing (2021~2025)...", flush=True)
    print("=" * 60, flush=True)
    expert1.eval()
    expert2.eval()
    router.eval()

    torch.manual_seed(999)  # 테스트 stochastic 재현성
    pp = 1.0
    hwm = 1.0
    last_e1 = 0.5
    last_e2 = 0.5
    last_choice = 0.0

    history = {"pp": [], "hwm": [], "choice": [], "e1_act": [], "e2_act": [],
               "final_act": [], "results": [], "e1_prob": []}

    for t in range(len(test_data)):
        row = test_data.iloc[t]
        next_ret = float(row["sp_next_return"])
        tb_r = float(row["tbill"])
        met = float(row["metabolism"]) + PREMIUM

        e1_obs = make_obs(test_data, t, pp, hwm, last_e1)
        e2_obs = make_obs(test_data, t, pp, hwm, last_e2)
        with torch.no_grad():
            e1_act, _, _ = expert1.get_action(e1_obs.unsqueeze(0), deterministic=True)
            e2_act, _, _ = expert2.get_action(e2_obs.unsqueeze(0), deterministic=True)
        e1_a = float(e1_act.squeeze().clamp(0, 1))
        e2_a = float(e2_act.squeeze().clamp(0, 1))

        r_obs = make_router_obs(test_data, t, pp, hwm, e1_a, e2_a, last_choice)
        with torch.no_grad():
            # Router는 stochastic 유지 (학습된 확률 분포 그대로 사용)
            r_act, _, _ = router.get_action(r_obs.unsqueeze(0), deterministic=False)
            # 실제 확률 추출
            logits, _ = router.forward(r_obs.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            e1_prob = float(probs[0, 0])
        choice = int(r_act.squeeze())

        final_act = e1_a if choice == 0 else e2_a
        port_ret = final_act * next_ret + (1 - final_act) * tb_r
        pp = pp * (1 + port_ret) / (1 + met)
        hwm = max(hwm, pp)

        history["pp"].append(pp)
        history["hwm"].append(hwm)
        history["choice"].append(float(choice))
        history["e1_act"].append(e1_a)
        history["e2_act"].append(e2_a)
        history["final_act"].append(final_act)
        history["results"].append(next_ret)
        history["e1_prob"].append(e1_prob)

        last_e1 = e1_a
        last_e2 = e2_a
        last_choice = float(choice)

    final_acts = np.array(history["final_act"])
    e1_acts = np.array(history["e1_act"])
    e2_acts = np.array(history["e2_act"])
    choices = np.array(history["choice"])
    results = np.array(history["results"])
    tb = test_data["tbill"].values[:len(final_acts)]

    moe_rets = final_acts * results + (1 - final_acts) * tb
    e1_rets = e1_acts * results + (1 - e1_acts) * tb
    e2_rets = e2_acts * results + (1 - e2_acts) * tb

    print(f"\n=== MoE Pre-trained (PyTorch) ===\n", flush=True)
    compute_metrics(moe_rets, "MoE (Router)")
    compute_metrics(e1_rets, "Expert1 (HWM)")
    compute_metrics(e2_rets, "Expert2 (Return)")
    compute_metrics(results, "S&P B&H")
    compute_metrics(0.5 * results + 0.5 * tb, "50/50")

    e1_probs = np.array(history["e1_prob"])
    e1_pct = (1 - choices).mean() * 100
    e2_pct = choices.mean() * 100
    print(f"\nRouter 실제 선택: Expert1={e1_pct:.0f}% Expert2={e2_pct:.0f}%", flush=True)
    print(f"Router 학습 확률: E1={e1_probs.mean():.3f} E2={1-e1_probs.mean():.3f} "
          f"(min={e1_probs.min():.3f} max={e1_probs.max():.3f})", flush=True)
    print(f"Expert1 action: mean={e1_acts.mean():.2f}  Expert2 action: mean={e2_acts.mean():.2f}  "
          f"Final action: mean={final_acts.mean():.2f}", flush=True)

    dates = pd.to_datetime(test_data["date"].values[:len(final_acts)])
    print(flush=True)
    for year in range(2021, 2027):
        mask = [d.year == year for d in dates]
        if not any(mask):
            continue
        idx = [i for i, m in enumerate(mask) if m]
        yr_ret = (np.prod(1 + moe_rets[idx]) - 1) * 100
        yr_e1_pct = (1 - np.mean(choices[idx])) * 100
        yr_e1_prob = np.mean(e1_probs[idx])
        yr_e1 = np.mean(e1_acts[idx])
        yr_e2 = np.mean(e2_acts[idx])
        yr_final = np.mean(final_acts[idx])
        print(f"{year}: E1={yr_e1_pct:.0f}% prob={yr_e1_prob:.2f} e1={yr_e1:.2f} e2={yr_e2:.2f} "
              f"final={yr_final:.2f} ret={yr_ret:+.1f}%", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
