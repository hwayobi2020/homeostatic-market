"""GARCH-X(past)-skewt — 과거 거시만 받는 GARCH + Hansen(1994) skew-t 혁신.

목적 (공정 비교)
---------------
MAC-Flow(maskall)와 *받는 정보를 일치*시킨 baseline.  둘 다:
  · 과거 거시(origin 시점까지 알려진 tbill·metab)는 사용
  · 미래 거시 *경로* 는 사용하지 않음 (maskall = 미래 마스킹 / 여기 = 미래 거시 미주입)
모형 형태만 다르다(GARCH 분산 vs flow 분포).

혁신 분포 = **Hansen(1994) 표준화 skew-t** (대칭 t 가 아니라 비대칭).  → GARCH 에게 비대칭을
낼 *가장 공정한* 분포를 부여한다.  단, skew-t 의 왜도 모수 λ 는 *상수* 이므로 GARCH 의 왜도는
거시 상태와 무관한 *고정* 값이다.  거시 *경로에 반응하는* 조건부 왜도는 MAC-Flow 만 내며,
그 대조는 §4.2(반사실: 거시 따라 skew 변화)·§4.3.1(full vs maskall)에서 본다.

spec
----
  mean : y_t = μ + ε_t,  ε_t = σ_t·η_t
  var  : σ²_t = (ω + α·ε²_{t-1} + β·σ²_{t-1}) · exp(γ1·x1_t + γ2·x2_t)
         x1=z(tbill_wr), x2=z(metab_13w) (train 통계 표준화).  exp(·) → σ²>0.
  innov: η_t ~ Hansen(1994) 표준화 skew-t (자유도 ν, 왜도 λ; mean 0, var 1)
  fit  : train MLE (L-BFGS-B).  test: origin 별 σ² 필터 → 13주 forward 시뮬,
         거시는 origin 값 고정(미래 미주입).
  eval : train_garch_x 와 동일 정의(crps/cov/std/cvar/skew/exkurt) 재사용 → 같은 4 fold.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/train_garch_xpast.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
RESULT_DIR = os.path.join(HERE, "result")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
PAST_LEN = 52
FUT = 13
N_SIM = 1000
SEED = 2026
INDPRO_LAG = 2

sys.path.insert(0, HERE)
from train_garch_x import crps_pooled, cvar, var_q, emd1d   # noqa: E402  (동일 지표)


# ---------- Hansen(1994) 표준화 skew-t ----------
def _hansen_abc(nu, lam):
    """표준화 상수 a, b, c (mean 0, var 1 보장)."""
    log_c = gammaln((nu + 1) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * (nu - 2.0))
    c = np.exp(log_c)
    a = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    b = np.sqrt(1.0 + 3.0 * lam ** 2 - a ** 2)
    return a, b, c


def hansen_logpdf(eta, nu, lam):
    """log g(eta) — 표준화 Hansen skew-t 밀도 (eta = 표준화 잔차)."""
    a, b, c = _hansen_abc(nu, lam)
    bz_a = b * eta + a
    denom = np.where(bz_a < 0.0, 1.0 - lam, 1.0 + lam)   # eta < -a/b → (1-λ) 가지
    inner = 1.0 + (1.0 / (nu - 2.0)) * (bz_a / denom) ** 2
    return np.log(b) + np.log(c) - ((nu + 1.0) / 2.0) * np.log(inner)


def hansen_sample(nu, lam, size, rng):
    """표준화 Hansen skew-t 샘플 (역CDF).  표준 t ppf 를 (1∓λ) 두 가지로 나눠 생성."""
    a, b, c = _hansen_abc(nu, lam)
    scale = np.sqrt((nu - 2.0) / nu)
    u = rng.uniform(size=size)
    z = np.empty(size)
    cut = (1.0 - lam) / 2.0
    left = u < cut
    # 왼쪽 가지: F(z)=(1-λ)·T_ν(·) → z = [(1-λ)·scale·T_ppf(u/(1-λ)) - a]/b
    ul = u[left]
    z[left] = ((1.0 - lam) * scale * student_t.ppf(ul / (1.0 - lam), nu) - a) / b
    # 오른쪽 가지: T_ppf(0.5 + (u-cut)/(1+λ))
    ur = u[~left]
    z[~left] = ((1.0 + lam) * scale
                * student_t.ppf(0.5 + (ur - cut) / (1.0 + lam), nu) - a) / b
    return z


# ---------- data ----------
def get_metab13(df):
    if "metab_13w" in df.columns and pd.to_numeric(df["metab_13w"], errors="coerce").notna().any():
        return pd.to_numeric(df["metab_13w"], errors="coerce").values
    m2 = pd.to_numeric(df["m2_growth_lag"], errors="coerce").rolling(13).sum()
    cpi = pd.to_numeric(df["cpi_wr_lag"], errors="coerce").rolling(13).sum()
    ind = pd.to_numeric(df["log_indpro"], errors="coerce").diff(13).shift(INDPRO_LAG)
    return (m2 - ind - cpi).values


def build(df):
    y = pd.to_numeric(df["sp_return"], errors="coerce").values
    tb = pd.to_numeric(df["tbill_wr"], errors="coerce").values
    mt = get_metab13(df)
    return y, tb, mt


# ---------- GARCH-X(past)-skewt ----------
def _safe_exp_arg(g1, g2, x1, x2):
    return np.clip(g1 * x1 + g2 * x2, -10.0, 10.0)


def neg_ll(p, y, x1, x2):
    mu, om, al, be, g1, g2, nu, lam = p
    n = len(y)
    eps = y - mu
    s2 = np.empty(n)
    s2[0] = np.var(eps)
    for t in range(1, n):
        base = om + al * eps[t - 1] ** 2 + be * s2[t - 1]
        s2[t] = base * np.exp(_safe_exp_arg(g1, g2, x1[t], x2[t]))
        if s2[t] <= 1e-12:
            s2[t] = 1e-12
    eta = eps / np.sqrt(s2)
    ll = hansen_logpdf(eta, nu, lam) - 0.5 * np.log(s2)   # Jacobian: dη/dε = 1/σ
    return -np.sum(ll)


def fit_garch_xpast(y, x1, x2):
    mu0 = float(np.mean(y)); v0 = float(np.var(y))
    p0 = [mu0, 0.05 * v0, 0.08, 0.88, 0.0, 0.0, 7.0, 0.0]
    bnds = [(-1, 1), (1e-10, None), (1e-6, 0.5), (1e-6, 0.999),
            (-1.5, 1.5), (-1.5, 1.5), (2.1, 60.0), (-0.95, 0.95)]
    res = minimize(neg_ll, p0, args=(y, x1, x2), method="L-BFGS-B", bounds=bnds)
    return res.x


def filter_var(p, y, x1, x2):
    mu, om, al, be, g1, g2, nu, lam = p
    eps = y - mu; n = len(y); s2 = np.empty(n)
    s2[0] = np.nanvar(eps[~np.isnan(eps)])
    for t in range(1, n):
        prev = s2[t - 1] if np.isfinite(s2[t - 1]) else np.nanvar(eps)
        e2 = eps[t - 1] ** 2 if np.isfinite(eps[t - 1]) else prev
        a1 = x1[t] if np.isfinite(x1[t]) else 0.0
        a2 = x2[t] if np.isfinite(x2[t]) else 0.0
        base = om + al * e2 + be * prev
        s2[t] = max(base * np.exp(_safe_exp_arg(g1, g2, a1, a2)), 1e-12)
    return s2, eps


def simulate(p, eps_orig, s2_orig, x1_o, x2_o, n_sim, rng):
    """origin 에서 13주 forward.  거시는 origin 값 고정(미래 미주입).  혁신 = Hansen skew-t."""
    mu, om, al, be, g1, g2, nu, lam = p
    macro_factor = float(np.exp(_safe_exp_arg(g1, g2,
                                              x1_o if np.isfinite(x1_o) else 0.0,
                                              x2_o if np.isfinite(x2_o) else 0.0)))
    s2 = np.full(n_sim, s2_orig)
    e2prev = np.full(n_sim, eps_orig ** 2)
    out = np.empty((n_sim, FUT))
    for h in range(FUT):
        base = om + al * e2prev + be * s2
        s2 = np.maximum(base * macro_factor, 1e-12)
        eta = hansen_sample(nu, lam, n_sim, rng)     # 표준화 skew-t (var 1)
        eps = np.sqrt(s2) * eta
        out[:, h] = mu + eps
        e2prev = eps ** 2
    return out


def run_fold(fold):
    sp = os.path.join(RESULT_DIR, f"garch_xpast_{fold}_summary.json")
    if os.path.exists(sp):
        print(f"[skip] {os.path.basename(sp)}"); return
    tr = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_train.csv"))
    te = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_test.csv"))
    ytr, tbtr, mttr = build(tr)
    yte, tbte, mtte = build(te)

    m1 = np.isfinite(tbtr); m2v = np.isfinite(mttr)
    mu1, sd1 = float(np.nanmean(tbtr[m1])), float(np.nanstd(tbtr[m1]) + 1e-12)
    mu2, sd2 = float(np.nanmean(mttr[m2v])), float(np.nanstd(mttr[m2v]) + 1e-12)
    x1tr = (tbtr - mu1) / sd1; x2tr = (mttr - mu2) / sd2
    x1te = (tbte - mu1) / sd1; x2te = (mtte - mu2) / sd2

    m = np.isfinite(ytr) & np.isfinite(x1tr) & np.isfinite(x2tr)
    yf, x1f, x2f = ytr[m], x1tr[m], x2tr[m]
    print(f"\n{'='*60}\n fold={fold}  train n={len(yf)}  test n={len(yte)}")
    p = fit_garch_xpast(yf, x1f, x2f)
    mu, om, al, be, g1, g2, nu, lam = p
    print(f"  params: mu={mu:+.5f} om={om:.2e} al={al:.3f} be={be:.3f} "
          f"g_tbill={g1:+.3f} g_metab={g2:+.3f} nu={nu:.1f} lam(skew)={lam:+.3f}")

    s2_te, eps_te = filter_var(p, yte, x1te, x2te)
    rng = np.random.default_rng(SEED)
    origins = [t for t in range(PAST_LEN, len(yte) - FUT)
               if np.isfinite(yte[t]) and np.isfinite(s2_te[t])
               and np.all(np.isfinite(yte[t + 1:t + 1 + FUT]))]
    sim = np.empty((len(origins), N_SIM, FUT), dtype=np.float32)
    act = np.empty((len(origins), FUT), dtype=np.float32)
    for i, t in enumerate(origins):
        sim[i] = simulate(p, eps_te[t], s2_te[t], x1te[t], x2te[t], N_SIM, rng)
        act[i] = yte[t + 1:t + 1 + FUT]
    print(f"  origins={len(origins)}  n_sim={N_SIM}  (skew-t 혁신, 미래 거시 미주입)")

    crps_m, crps_s = crps_pooled(sim, act)
    af = act.ravel(); sf = sim.ravel()
    std_a = float(af.std(ddof=1)); std_s = float(sf.std(ddof=1))

    def _sk(a):
        a = np.asarray(a, float); return float(np.mean(((a - a.mean()) / (a.std() + 1e-12)) ** 3))

    def _ek(a):
        a = np.asarray(a, float); return float(np.mean(((a - a.mean()) / (a.std() + 1e-12)) ** 4) - 3.0)

    skew_a, skew_s = _sk(af), _sk(sf)
    kurt_a, kurt_s = _ek(af), _ek(sf)
    cv5a, cv5s = cvar(af, .05), cvar(sf, .05)
    cv1a, cv1s = cvar(af, .01), cvar(sf, .01)
    # 커버리지: MAC-Flow(train_garch_flow.evaluate_test) 와 동일하게 *전역 풀링* 구간.
    #   전 origin × 전 sim × 전 시점을 합친 분포에서 백분위 한 쌍을 뽑아 전부를 판정한다.
    #   (이전에는 axis=1 원점별이라 MAC-Flow 와 정의가 달랐다.)
    cov = {}
    for lvl, lo, hi in [(50, 25, 75), (80, 10, 90), (95, 2.5, 97.5)]:
        L = np.percentile(sf, lo); H = np.percentile(sf, hi)
        cov[lvl] = float(((af >= L) & (af <= H)).mean())
    print(f"  CRPS={crps_m:.5f}  std a/s/ratio={std_a:.5f}/{std_s:.5f}/{std_s/std_a:.3f}  "
          f"cov 50/80/95={cov[50]:.3f}/{cov[80]:.3f}/{cov[95]:.3f}")
    print(f"  skew a/s={skew_a:+.4f}/{skew_s:+.4f}  exkurt a/s={kurt_a:+.4f}/{kurt_s:+.4f}  "
          f"(skew-t λ={lam:+.3f} → sim skew 는 *고정* 비대칭)")

    summ = dict(fold=fold, model="GARCH-X(past)-skewt (과거 거시, 미래경로 배제, Hansen skew-t)",
                params=dict(mu=mu, omega=om, alpha=al, beta=be,
                            gamma_tbill=g1, gamma_metab=g2, nu=nu, lambda_skew=lam),
                n_origins=len(origins), seed=SEED,
                test_eval=dict(crps_pooled=crps_m, crps_std=crps_s,
                               emd=emd1d(sf, af), std_actual=std_a, std_sim=std_s,
                               std_ratio=std_s / std_a, coverage_50=cov[50],
                               coverage_80=cov[80], coverage_95=cov[95],
                               cvar_5pct_diff=cv5s - cv5a, cvar_1pct_diff=cv1s - cv1a,
                               var_1pct_diff=var_q(sf, .01) - var_q(af, .01),
                               skew_actual=skew_a, skew_sim=skew_s,
                               exkurt_actual=kurt_a, exkurt_sim=kurt_s))
    os.makedirs(RESULT_DIR, exist_ok=True)
    json.dump(summ, open(sp, "w"), indent=2, default=str)
    print(f"  saved {os.path.basename(sp)}")


def main():
    print(f"[garch-xpast-skewt] 과거 거시 GARCH-X + Hansen skew-t (미래경로 배제)  {len(FOLDS)} fold")
    for fold in FOLDS:
        if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                   for s in ("train", "test")):
            print(f"[skip {fold}] csv 없음"); continue
        try:
            run_fold(fold)
        except Exception as e:
            print(f"[FAIL] {fold}: {e!r}")
    print("\n[done] garch_xpast_*_summary.json → §4.1.1 에서 MAC-Flow(maskall) 와 비교 "
          "(둘 다 과거 거시 사용·미래경로 배제, GARCH 엔 skew-t 부여).")


if __name__ == "__main__":
    main()
