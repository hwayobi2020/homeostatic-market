# Homeostatic Financial Agent

**Investment Behavior Emerges from Social Homeostasis Without Behavioral Injection**

## Abstract

We propose a reinforcement learning framework where an agent's sole objective is maintaining its relative socioeconomic position (social homeostasis), with no explicit investment rules injected. Unlike traditional Agent-Based Models that hardcode behavioral biases (e.g., loss aversion, herding), our model places a fully symmetric reward function on the agent and lets environmental pressure (wealth growth of peer groups) drive investment behavior to emerge naturally.

Using Fed Distributional Financial Accounts (DFA) data as peer-group benchmarks and S&P 500 as the investment vehicle, we show that:
- Realistic investment behavior emerges from homeostasis alone
- The agent achieves Sharpe 0.66 with MDD -16.7% out-of-sample (2019-2025)
- Wealth inequality dynamics are structurally reproduced without injection

## Paper Outline

### 1. Introduction
- Behavioral finance injects biases (loss aversion, FOMO) into utility functions
- We ask: can these behaviors **emerge** from a simpler principle?
- Homeostasis as the minimal cognitive architecture (Maturana & Varela, autopoiesis)
- Key distinction: **bias injection into math vs. environmental pressure**

### 2. Related Work
- Homeostatic RL: Yoshida et al. (2024) - integrated behaviors emerge from homeostasis
- Keeping Up with the Joneses: Gali (1994) - relative consumption in utility (injection approach)
- Aspiration Adaptation Theory: Selten (1998) - dynamic reference points
- Agent-Based Models in finance: behavioral rules as parameters

### 3. Model

#### 3.1 Social Homeostasis Framework
- Agent maintains relative position among peer wealth group
- Reward: `-|purchasing_power - setpoint|` (fully symmetric)
- No loss aversion, no asymmetric penalties

#### 3.2 Percentile-Specific Metabolism
- Each wealth percentile faces different "metabolic pressure"
- **Top 1%**: Fed DFA Total Net Worth growth (~6.3%/yr)
- **Middle 50-90th**: Fed DFA Total Net Worth growth (~5.2%/yr)
- **Bottom 50%**: Average Hourly Earnings growth (~3.2%/yr)
- Key insight: metabolism is not M2 (monetary inflation) but **peer-group wealth growth**

#### 3.3 Environment Design
- Step = 1 quarter (matches DFA publication cycle)
- S&P 500 quarterly returns as investment vehicle
- T-bill returns for uninvested portion
- DFA data lagged 1 quarter (publication delay, no data leakage)
- Rolling features shifted 1 period (no look-ahead bias)

#### 3.4 RL Algorithm
- PPO (Proximal Policy Optimization)
- State: [purchasing_power, sp_1q_lag, sp_2q_lag, metabolism_1q_lag, tbill, last_action]
- Action: investment ratio [0, 1]
- Train: 2006-2018, Test: 2019-2025

### 4. Results

#### 4.1 Top 1% Agent (Out-of-Sample 2019-2025)

| Metric | Homeostatic RL | Buy & Hold | LightGBM | Profit-Max RL |
|--------|---------------|------------|----------|---------------|
| Return | +8.4%/yr | +15.7%/yr | +7.6%/yr | +11.5%/yr |
| Volatility | 9.1% | 17.2% | 11.8% | 18.6% |
| Sharpe | **0.66** | 0.80 | 0.45 | 0.53 |
| MDD | **-16.7%** | -24.8% | -20.3% | -31.7% |
| Avg Position | 50% | 100% | 56% | 85% |
| Final PP | **1.04** | - | - | - |

- Agent successfully maintains Top 1% position (PP > 1.0)
- Lower return than B&H but significantly better risk management
- Outperforms LightGBM and Profit-Max RL on risk-adjusted basis

#### 4.2 Emergent Behaviors (No Injection)
- **Market timing**: Agent reduces allocation before downturns (2020 Q1: 28%)
- **Regime adaptation**: 45% when metabolism < 0, 52% when > 0
- **Risk management**: MDD reduced by 1/3 vs Buy & Hold
- **Inequality reproduction**: Top invests heavily, Bottom cannot invest at all

#### 4.3 Key Finding: Bias is in the Environment, Not the Math
- Symmetric reward function produces asymmetric behavior
- "Loss aversion" emerges from environmental pressure, not utility function
- Aligns with ecological/evolutionary economics rather than behavioral economics

### 5. Discussion
- Homeostatic framework as alternative to behavioral bias injection
- Implications for understanding wealth inequality as emergent phenomenon
- Limitations: single asset, quarterly granularity, US market only

### 6. Conclusion
- Investment behavior can emerge from social homeostasis alone
- Environmental pressure (peer-group wealth growth) is sufficient to drive realistic behavior
- The framework provides a biologically-grounded alternative to behavioral finance

## Project Structure

```
homeostatic-market/
├── env/
│   ├── single_agent_env.py           # Original 2-layer environment
│   ├── real_data_env.py              # Real S&P 500 + M2 environment
│   ├── weekly_env.py                 # Weekly step environment
│   ├── weekly_env_adaptive.py        # Adaptive setpoint experiment
│   ├── weekly_env_percentile.py      # Weekly percentile environment
│   └── quarterly_env_percentile.py   # Quarterly percentile (current)
├── sim/
│   ├── run_experiment.py             # Phase 1 experiments
│   ├── run_lag_comparison.py         # Observation lag experiments
│   ├── run_dual_homeostasis.py       # 1-layer vs 2-layer comparison
│   ├── run_contrarian.py             # Contrarian strategy test
│   ├── run_real_data.py              # Real market data evaluation
│   └── run_lgbm.py                   # LightGBM comparison
├── analysis/
│   └── stylized_facts.py            # Statistical analysis tools
├── data/                            # Market data (S&P, DFA, CPI, M2, T-bill)
├── models/                          # Trained PPO models
├── plots/                           # Visualizations
└── docs/                           # Documentation
```

## Data Sources

| Data | Source | Frequency | FRED Code |
|------|--------|-----------|-----------|
| S&P 500 | Yahoo Finance | Daily | ^GSPC |
| Top 1% Net Worth | Fed DFA | Quarterly | WFRBLT01026 |
| 50-90th Net Worth | Fed DFA | Quarterly | WFRBLN40080 |
| Average Hourly Earnings | BLS via FRED | Monthly | CES0500000003 |
| 3-Month T-Bill | FRED | Daily | DTB3 |
| M2 Money Supply | FRED | Weekly | WM2NS |
| VIX | CBOE via Yahoo | Daily | ^VIX |
| CPI | BLS via FRED | Monthly | CPIAUCSL |

## Theoretical Background

- **Maturana & Varela (1984)** - Autopoiesis: "to live is to know"
- **Yoshida et al. (2024)** - Homeostatic RL: integrated behaviors emerge from homeostasis
- **Damasio** - Somatic Marker Hypothesis
- **Selten (1998)** - Aspiration Adaptation Theory
- **Gali (1994)** - Keeping Up with the Joneses

## Requirements

```
gymnasium
stable-baselines3
torch
numpy
pandas
pandas-datareader
yfinance
lightgbm
scikit-learn
matplotlib
python-docx
```
