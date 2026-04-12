"""Walk-forward 결과에 B&H 비교 추가."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Load data
train_df = pd.read_csv("data/monthly_noleak_v25_train.csv")
test_df = pd.read_csv("data/monthly_noleak_v25_test.csv")
train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])
full = pd.concat([train_df, test_df]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
SP_NEXT = full["sp_next_return"].fillna(0).values
DATES = full["date"].values

# Load evolved results
results = pd.read_csv("result/evolved_alpha_beta_walkforward.csv")

TRAIN_MONTHS = 60
TEST_MONTHS = 12
SLIDE = 12


def bh_metrics(rets):
    """B&H on returns."""
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 and cum[-1] > 0 else -100
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100 if len(cum) > 0 else 0
    return ann, mdd


print("=" * 115, flush=True)
print("Walk-forward Agent vs Buy & Hold", flush=True)
print("=" * 115, flush=True)

print(f"\n{'W':>3}  {'TrainEnd':>10}  {'λ':>5}  "
      f"{'Agent_Ret':>9}  {'B&H_Ret':>9}  {'Diff':>8}  "
      f"{'Agent_MDD':>9}  {'B&H_MDD':>9}  "
      f"{'Agent_Died':>10}  {'Verdict':>10}", flush=True)
print("-" * 115, flush=True)

agent_rets = []
bh_rets = []
wins = 0
ties = 0
losses = 0

# Window start index calculation
start = 0
for w_idx in range(len(results)):
    row = results.iloc[w_idx]
    # Compute B&H for this test window
    test_start = start + TRAIN_MONTHS
    test_end = test_start + TEST_MONTHS

    test_rets = SP_NEXT[test_start:test_end]
    bh_ann, bh_mdd = bh_metrics(test_rets)

    agent_ann = row["ann_ret"]
    diff = agent_ann - bh_ann

    if diff > 1.0:
        verdict = "WIN"
        wins += 1
    elif abs(diff) <= 1.0:
        verdict = "TIE"
        ties += 1
    else:
        verdict = "LOSS"
        losses += 1

    died_str = f"step {int(row['died_step'])}" if row["died_step"] > 0 else "—"

    print(f"{int(row['window']):>3}  {row['train_end']:>10}  {row['lambda']:>5.2f}  "
          f"{agent_ann:>+8.2f}%  {bh_ann:>+8.2f}%  {diff:>+7.2f}%  "
          f"{row['mdd']:>+8.1f}%  {bh_mdd:>+8.1f}%  "
          f"{died_str:>10}  {verdict:>10}", flush=True)

    agent_rets.append(agent_ann)
    bh_rets.append(bh_ann)
    start += SLIDE

# Summary
print("\n" + "=" * 115, flush=True)
print("Aggregate Summary", flush=True)
print("=" * 115, flush=True)

agent_rets = np.array(agent_rets)
bh_rets = np.array(bh_rets)

print(f"  Agent mean return:  {agent_rets.mean():+.2f}%/yr", flush=True)
print(f"  B&H mean return:    {bh_rets.mean():+.2f}%/yr", flush=True)
print(f"  Difference (Agent - B&H): {(agent_rets - bh_rets).mean():+.2f}%/yr", flush=True)
print(flush=True)
print(f"  Agent median return:  {np.median(agent_rets):+.2f}%/yr", flush=True)
print(f"  B&H median return:    {np.median(bh_rets):+.2f}%/yr", flush=True)
print(flush=True)

agent_volatility = agent_rets.std()
bh_volatility = bh_rets.std()
print(f"  Agent window return std: {agent_volatility:.2f}%", flush=True)
print(f"  B&H   window return std: {bh_volatility:.2f}%", flush=True)
print(flush=True)

print(f"  Win (Agent > B&H by >1%):  {wins}/{len(results)} ({wins/len(results)*100:.0f}%)", flush=True)
print(f"  Tie (within ±1%):           {ties}/{len(results)} ({ties/len(results)*100:.0f}%)", flush=True)
print(f"  Loss (Agent < B&H by >1%): {losses}/{len(results)} ({losses/len(results)*100:.0f}%)", flush=True)

# Cumulative comparison
agent_cum = np.prod(1 + agent_rets / 100)
bh_cum = np.prod(1 + bh_rets / 100)
total_years = len(results)
print(flush=True)
print(f"  Cumulative over {total_years} years:", flush=True)
print(f"    Agent: {(agent_cum-1)*100:+.1f}% total ({(agent_cum**(1/total_years)-1)*100:+.2f}%/yr compounded)", flush=True)
print(f"    B&H:   {(bh_cum-1)*100:+.1f}% total ({(bh_cum**(1/total_years)-1)*100:+.2f}%/yr compounded)", flush=True)

# Death analysis
died = results["died_step"] > 0
print(flush=True)
print(f"  Death events: {died.sum()}/{len(results)} windows", flush=True)
print(f"    Agent died (PP<0.95) in {died.sum()} windows", flush=True)

# λ regime correlation
# Compute train B&H for each window (need train period)
train_bh_rets = []
start = 0
for w_idx in range(len(results)):
    train_start = start
    train_end = train_start + TRAIN_MONTHS
    tr_rets = SP_NEXT[train_start:train_end]
    tr_cum = np.cumprod(1 + tr_rets)
    tr_ann = (tr_cum[-1] ** (12 / len(tr_rets)) - 1) * 100 if len(tr_rets) > 0 else 0
    train_bh_rets.append(tr_ann)
    start += SLIDE

train_bh_rets = np.array(train_bh_rets)
lambdas = results["lambda"].values

# Correlation: λ vs train B&H
from scipy.stats import pearsonr, spearmanr
pr, pv = pearsonr(lambdas, train_bh_rets)
sr, sv = spearmanr(lambdas, train_bh_rets)
print(flush=True)
print(f"  corr(evolved λ, train B&H return):", flush=True)
print(f"    Pearson = {pr:+.3f} (p={pv:.3f})", flush=True)
print(f"    Spearman = {sr:+.3f} (p={sv:.3f})", flush=True)
print(f"    → 음의 상관이면 'bull train → 낮은 λ' (procyclical)", flush=True)

print("\nDone.", flush=True)
