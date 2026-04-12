"""Analytic policy + λ search on W1.

Policy: Myopic closed-form, assuming:
  μ = 0.006 (constant monthly equity premium)
  σ = VIX / 100 / √12 (observable)

Reward family: r⁺ - λ·|r⁻|  (α=1 fixed, λ = β is loss aversion coefficient)

W1 train 1991~2011 → search λ on train
W1 test 2011~2016 → OOS

Fitness variants:
  - Sharpe
  - Sortino (downside)
  - Final wealth
  - Calmar
"""
import sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from scipy.stats import norm
from scipy.optimize import minimize

# Data
train = pd.read_csv("data/monthly_noleak_v25_train.csv")
test = pd.read_csv("data/monthly_noleak_v25_test.csv")
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
full = pd.concat([train, test]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full["tbill_fwd"] = full["tbill"].shift(-1)
full["metab_fwd"] = full["metabolism"].shift(-1)

MU_FIXED = 0.006   # monthly equity premium (Gemini 제안)


def sigma_from_vix(vix_value):
    """Monthly vol from VIX (annualized %)."""
    return (vix_value / 100.0) / np.sqrt(12.0)


def expected_reward(w_s, w_b, mu, sigma, tb, met, alpha, beta):
    """E[α·r⁺ + β·r⁻] under Normal assumption."""
    total = w_s + w_b
    borrow = max(0.0, total - 1.0)
    mu_port = w_s * mu + w_b * tb - borrow * met
    sigma_port = w_s * sigma  # stock only stochastic
    if sigma_port < 1e-10:
        if mu_port >= 0:
            return alpha * mu_port
        else:
            return beta * mu_port
    z = mu_port / sigma_port
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    E_plus = sigma_port * phi_z + mu_port * Phi_z
    E_minus = mu_port - E_plus
    return alpha * E_plus + beta * E_minus


def analytic_policy(vix, tb, met, alpha, beta, mu=MU_FIXED):
    """Given state and (α, β), find optimal (w_s, w_b).
    Parameterize via (w_b, lev) ∈ [0,1]² → w_s = (1-w_b) + lev, total ∈ [1,2]."""
    sigma = sigma_from_vix(vix)

    def neg_er(params):
        w_b, lev = params
        w_s = (1 - w_b) + lev
        return -expected_reward(w_s, w_b, mu, sigma, tb, met, alpha, beta)

    # Multi-start to avoid local optima
    best_val = np.inf
    best_params = (0.5, 0.5)
    for x0 in [(0.1, 0.9), (0.5, 0.5), (0.9, 0.1), (0.0, 1.0), (1.0, 0.0)]:
        res = minimize(neg_er, x0=x0,
                       bounds=[(0, 1), (0, 1)],
                       method='L-BFGS-B')
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x

    w_b_opt = float(np.clip(best_params[0], 0, 1))
    lev_opt = float(np.clip(best_params[1], 0, 1))
    w_s_opt = (1 - w_b_opt) + lev_opt
    return w_s_opt, w_b_opt


def simulate(data, alpha, beta):
    """Run analytic policy through data, return (ws, wb, rets, pps)."""
    ws_arr, wb_arr, rets = [], [], []
    pp = 1.0
    pps = [1.0]
    for i in range(len(data)):
        row = data.iloc[i]
        vix = float(row["vix"])
        tb = float(row["tbill_fwd"]) if not pd.isna(row["tbill_fwd"]) else float(row["tbill"])
        met = float(row["metab_fwd"]) if not pd.isna(row["metab_fwd"]) else float(row["metabolism"])
        sp = float(row["sp_next_return"]) if not pd.isna(row["sp_next_return"]) else 0.0

        w_s, w_b = analytic_policy(vix, tb, met, alpha, beta)
        total = w_s + w_b
        borrow = max(0.0, total - 1.0)
        r = w_s * sp + w_b * tb - borrow * met

        pp *= (1 + r)
        pp /= (1 + met)
        pps.append(pp)
        ws_arr.append(w_s)
        wb_arr.append(w_b)
        rets.append(r)
    return np.array(ws_arr), np.array(wb_arr), np.array(rets), np.array(pps)


def fitness_sharpe(rets):
    if rets.std() < 1e-10:
        return 0
    return rets.mean() / rets.std() * np.sqrt(12)


def fitness_sortino(rets):
    ds = rets[rets < 0]
    if len(ds) == 0:
        return 100
    ds_std = np.sqrt(np.mean(ds ** 2))
    if ds_std < 1e-10:
        return 0
    return rets.mean() / ds_std * np.sqrt(12)


def fitness_growth(rets):
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    return (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 else 0


def fitness_calmar(rets):
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 else 0
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    return abs(ann / mdd) if mdd != 0 else 0


def full_metrics(rets, name):
    cum = np.cumprod(1 + rets)
    n_yr = len(rets) / 12
    ann = (cum[-1] ** (1 / n_yr) - 1) * 100 if n_yr > 0 else 0
    vol = rets.std() * np.sqrt(12) * 100
    sh = fitness_sharpe(rets)
    so = fitness_sortino(rets)
    ca = fitness_calmar(rets)
    peak = np.maximum.accumulate(cum)
    mdd = ((cum - peak) / peak).min() * 100
    print(f"    {name:<30}: ret={ann:+6.2f}% vol={vol:5.1f}% sharpe={sh:+5.2f} sortino={so:+5.2f} calmar={ca:+5.2f} mdd={mdd:5.1f}%", flush=True)


if __name__ == "__main__":
    # W1
    train_start, test_start, test_end = "1991-01-01", "2011-01-01", "2016-01-01"
    train_data = full[(full["date"] >= train_start) & (full["date"] < test_start)].reset_index(drop=True)
    test_data = full[(full["date"] >= test_start) & (full["date"] < test_end)].reset_index(drop=True)

    print("=" * 100, flush=True)
    print(f"W1 Analytic Policy + λ Grid Search", flush=True)
    print(f"  Train {train_start[:4]}~{test_start[:4]} ({len(train_data)}M)", flush=True)
    print(f"  Test  {test_start[:4]}~{test_end[:4]} ({len(test_data)}M)", flush=True)
    print(f"  Policy: myopic closed-form, μ={MU_FIXED}, σ=VIX/100/√12", flush=True)
    print(f"  Reward: r⁺ - λ·|r⁻|   (α=1, λ = β)", flush=True)
    print("=" * 100, flush=True)

    # λ grid (log-uniform)
    LAMBDAS = np.round(np.concatenate([
        np.linspace(0.5, 1.0, 6),
        np.linspace(1.0, 3.0, 11),
        np.linspace(3.0, 10.0, 8),
    ]), 3)
    LAMBDAS = np.unique(LAMBDAS)

    print(f"\n[1] Grid search on Train (λ ∈ [{LAMBDAS.min()}, {LAMBDAS.max()}], {len(LAMBDAS)} points)", flush=True)
    print(f"  {'λ':>6}  {'Sharpe':>8}  {'Sortino':>8}  {'Calmar':>8}  {'Growth%':>8}  {'FinalPP':>8}  {'MeanWs':>7}  {'MeanWb':>7}", flush=True)

    train_results = []
    for lam in LAMBDAS:
        ws, wb, rets, pps = simulate(train_data, alpha=1.0, beta=lam)
        sh = fitness_sharpe(rets)
        so = fitness_sortino(rets)
        ca = fitness_calmar(rets)
        gr = fitness_growth(rets)
        train_results.append({"lambda": lam, "sharpe": sh, "sortino": so,
                               "calmar": ca, "growth": gr,
                               "final_pp": pps[-1], "mean_ws": ws.mean(), "mean_wb": wb.mean()})
        print(f"  {lam:>6.3f}  {sh:+8.3f}  {so:+8.3f}  {ca:+8.3f}  {gr:+8.2f}  {pps[-1]:>8.3f}  {ws.mean():>7.3f}  {wb.mean():>7.3f}", flush=True)

    # Find optimal λ per fitness
    print(f"\n[2] Best λ by fitness metric", flush=True)
    lam_sharpe = max(train_results, key=lambda x: x["sharpe"])["lambda"]
    lam_sortino = max(train_results, key=lambda x: x["sortino"])["lambda"]
    lam_calmar = max(train_results, key=lambda x: x["calmar"])["lambda"]
    lam_growth = max(train_results, key=lambda x: x["growth"])["lambda"]

    print(f"  λ* (Sharpe)  = {lam_sharpe}", flush=True)
    print(f"  λ* (Sortino) = {lam_sortino}", flush=True)
    print(f"  λ* (Calmar)  = {lam_calmar}", flush=True)
    print(f"  λ* (Growth)  = {lam_growth}", flush=True)
    print(f"  Kahneman-Tversky empirical: 2.25", flush=True)

    # OOS evaluation at key λ values
    print(f"\n[3] OOS Test Evaluation (W1 test 2011~2016)", flush=True)

    key_lambdas = [1.0, 2.25, 5.0, lam_sharpe, lam_sortino, lam_calmar, lam_growth]
    key_lambdas = sorted(set([round(x, 3) for x in key_lambdas]))

    for lam in key_lambdas:
        ws, wb, rets, pps = simulate(test_data, alpha=1.0, beta=lam)
        note = ""
        if abs(lam - lam_sharpe) < 0.01: note += " [λ*Sharpe]"
        if abs(lam - lam_sortino) < 0.01: note += " [λ*Sortino]"
        if abs(lam - lam_calmar) < 0.01: note += " [λ*Calmar]"
        if abs(lam - lam_growth) < 0.01: note += " [λ*Growth]"
        if abs(lam - 2.25) < 0.01: note += " [Kahneman]"
        full_metrics(rets, f"λ={lam}{note}")

    # Baselines
    print(f"\n  Baselines:", flush=True)
    sp_next = test_data["sp_next_return"].values
    tb_fwd = test_data["tbill_fwd"].fillna(test_data["tbill"]).values
    full_metrics(sp_next, "B&H (100% SP)")
    full_metrics(0.5 * sp_next + 0.5 * tb_fwd, "50/50")

    # Detailed action diagnostics at λ*Sharpe
    print(f"\n[4] Action Distribution at λ*(Sharpe) = {lam_sharpe} on test", flush=True)
    ws, wb, rets, pps = simulate(test_data, alpha=1.0, beta=lam_sharpe)
    print(f"  w_s: mean={ws.mean():+.3f} std={ws.std():.3f} range [{ws.min():.2f}, {ws.max():.2f}]", flush=True)
    print(f"  w_b: mean={wb.mean():+.3f} std={wb.std():.3f} range [{wb.min():.2f}, {wb.max():.2f}]", flush=True)
    print(f"  lev: mean={(ws+wb-1).mean():+.3f}", flush=True)
    print(f"  Time aggressive (w_s>1.5): {(ws>1.5).mean()*100:.1f}%", flush=True)
    print(f"  Time defensive  (w_s<0.3): {(ws<0.3).mean()*100:.1f}%", flush=True)
    print(f"  Time unleveraged (total≈1): {(ws+wb < 1.05).mean()*100:.1f}%", flush=True)
    print(f"  PP: final={pps[-1]:.3f} min={pps.min():.3f} max={pps.max():.3f}", flush=True)

    print(f"\nDone.", flush=True)
