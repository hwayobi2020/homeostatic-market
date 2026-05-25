"""GARCH(1,1)-X — 분산식에 거시 외생항(macro turbulence) 넣은 거시-조건부 baseline.

배경(2026-05-23): 우리 flow 가 "거시-조건부 시나리오 생성"을 주장하려면, 거시를 못 받는
  GARCH-N 이 아니라 **거시를 받는 GARCH-X** 와 비교해야 공정(리뷰어 지적 대비).
  표준 arch 패키지는 exogenous 를 평균식에만 넣어 → 분산식 GARCH-X 는 custom MLE 로 직접.

스펙:
  mean : y_t = μ + ε_t
  var  : σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ1·Z1_t + γ2·Z2_t
         Z1 = |Δ13 tbill_wr| / scale,  Z2 = |Δ13 metab_26w| / scale  (비음수 → σ²>0 보장)
         (우리 model-free 발견: |Δmacro|→vol 동시상관 강함. 그 신호를 GARCH 에 직접 모수화.)
  innov: standardized Student-t(ν)  (꼬리 — flow 와 공정 비교)
  fit  : train MLE (scipy L-BFGS-B).  test: origin 별 미래 macro 경로로 13주 시뮬 → 분포.
  eval : CRPS/cov/std_ratio/CVaR (flow 와 동일 정의, inline).  같은 gap29 4 fold.

flow 차별점 검증: GARCH-X 가 거시-조건부 변동성을 잡으면 → flow 우위는 "분포 유연성(비가우시안
  꼬리·결합경로)" 으로 좁혀짐.  GARCH-X 가 못 잡으면(γ≈0) → per-week GARCH 가 window-scale
  거시를 못 써서 flow 의 path-conditioning 이 진짜 edge.

Usage (Colab): !python colab/dual_3ch/train_garch_x.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
RESULT_DIR = os.path.join(HERE, "result")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
PAST_LEN = 52
FUT = 13
DW = 13                # 거시 난기류 = |X_t − X_{t-DW}|
N_SIM = 1000
SEED = 2026
INDPRO_LAG = 2


# ---------- metrics (inline, flow 와 동일 정의) ----------
def crps_ensemble(sim, y):
    n = len(sim); s = np.sort(sim)
    t1 = np.mean(np.abs(sim - y))
    i = np.arange(n)
    t2 = (2.0 / (n * n)) * np.sum((2 * i + 1 - n) * s)
    return float(t1 - 0.5 * t2)


def crps_pooled(sim_paths, act):
    no, ns, T = sim_paths.shape
    vals = [crps_ensemble(sim_paths[t, :, w], act[t, w])
            for t in range(no) for w in range(T)]
    return float(np.mean(vals)), float(np.std(vals))


def cvar(x, a):
    s = np.sort(x); k = max(1, int(a * len(s)))
    return float(s[:k].mean())


def var_q(x, a):
    return float(np.quantile(x, a))


def emd1d(a, b, nb=200):
    lo = float(min(a.min(), b.min())); hi = float(max(a.max(), b.max()))
    if hi - lo < 1e-12:
        return 0.0
    bins = np.linspace(lo, hi, nb + 1)
    ah = np.histogram(a, bins=bins)[0]; bh = np.histogram(b, bins=bins)[0]
    return float(np.mean(np.abs(np.cumsum(ah) / max(ah.sum(), 1)
                                - np.cumsum(bh) / max(bh.sum(), 1))) * (hi - lo))


# ---------- data ----------
def ensure_metab(df, Wm=26):
    col = f"metab_{Wm}w"
    if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
        return pd.to_numeric(df[col], errors="coerce").values
    m2 = pd.to_numeric(df["m2_growth_lag"], errors="coerce").rolling(Wm).sum()
    cpi = pd.to_numeric(df["cpi_wr_lag"], errors="coerce").rolling(Wm).sum()
    ind = pd.to_numeric(df["log_indpro"], errors="coerce").diff(Wm).shift(INDPRO_LAG)
    return (m2 - ind - cpi).values


def build_series(df):
    y = pd.to_numeric(df["sp_return"], errors="coerce").values
    tb = pd.to_numeric(df["tbill_wr"], errors="coerce").values
    mt = ensure_metab(df, 26)
    # 거시 난기류 |Δ13|
    z1 = np.abs(tb - np.concatenate([np.full(DW, np.nan), tb[:-DW]]))
    z2 = np.abs(mt - np.concatenate([np.full(DW, np.nan), mt[:-DW]]))
    return y, z1, z2


# ---------- GARCH-X(1,1)-t ----------
def _unpack(p):
    mu, om, al, be, g1, g2, nu = p
    return mu, om, al, be, g1, g2, nu


def neg_ll(p, y, Z1, Z2):
    mu, om, al, be, g1, g2, nu = _unpack(p)
    n = len(y)
    eps = y - mu
    s2 = np.empty(n)
    s2[0] = np.var(eps)
    for t in range(1, n):
        s2[t] = om + al * eps[t - 1] ** 2 + be * s2[t - 1] + g1 * Z1[t] + g2 * Z2[t]
        if s2[t] <= 1e-12:
            s2[t] = 1e-12
    # standardized Student-t (unit-variance) log density
    c = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)))
    ll = (c - 0.5 * np.log(s2)
          - (nu + 1) / 2 * np.log(1 + eps ** 2 / ((nu - 2) * s2)))
    return -np.sum(ll)


def fit_garch_x(y, Z1, Z2):
    mu0 = float(np.mean(y)); v0 = float(np.var(y))
    p0 = [mu0, 0.05 * v0, 0.08, 0.88, 0.0, 0.0, 7.0]
    bnds = [(-1, 1), (1e-10, None), (1e-6, 0.5), (1e-6, 0.999),
            (0.0, None), (0.0, None), (2.1, 60.0)]
    res = minimize(neg_ll, p0, args=(y, Z1, Z2), method="L-BFGS-B", bounds=bnds)
    return res.x


def filter_var(p, y, Z1, Z2):
    mu, om, al, be, g1, g2, nu = _unpack(p)
    eps = y - mu; n = len(y); s2 = np.empty(n); s2[0] = np.var(eps[~np.isnan(eps)])
    for t in range(1, n):
        prev = s2[t - 1] if np.isfinite(s2[t - 1]) else np.nanvar(eps)
        e2 = eps[t - 1] ** 2 if np.isfinite(eps[t - 1]) else prev
        z1 = Z1[t] if np.isfinite(Z1[t]) else 0.0
        z2 = Z2[t] if np.isfinite(Z2[t]) else 0.0
        s2[t] = max(om + al * e2 + be * prev + g1 * z1 + g2 * z2, 1e-12)
    return s2, eps


def simulate(p, eps_orig, s2_orig, Z1f, Z2f, n_sim, rng):
    """origin 에서 13주 forward 시뮬.  Z*f: (FUT,) 미래 macro 난기류 경로(real=시나리오)."""
    mu, om, al, be, g1, g2, nu = _unpack(p)
    scale = np.sqrt((nu - 2) / nu)
    s2 = np.full(n_sim, s2_orig)          # σ²_origin
    e2prev = np.full(n_sim, eps_orig ** 2)
    out = np.empty((n_sim, FUT))
    for h in range(FUT):
        z1 = Z1f[h] if np.isfinite(Z1f[h]) else 0.0
        z2 = Z2f[h] if np.isfinite(Z2f[h]) else 0.0
        s2 = np.maximum(om + al * e2prev + be * s2 + g1 * z1 + g2 * z2, 1e-12)
        z = rng.standard_t(nu, size=n_sim) * scale
        eps = np.sqrt(s2) * z
        out[:, h] = mu + eps
        e2prev = eps ** 2
    return out


def run_fold(fold):
    sp = os.path.join(RESULT_DIR, f"garch_x_{fold}_summary.json")
    if os.path.exists(sp):
        print(f"[skip] {os.path.basename(sp)}"); return
    tr = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_train.csv"))
    te = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_test.csv"))
    ytr, z1tr, z2tr = build_series(tr)
    yte, z1te, z2te = build_series(te)
    # train: NaN 구간(DW warmup) 제거 후 fit
    m = np.isfinite(ytr) & np.isfinite(z1tr) & np.isfinite(z2tr)
    # 난기류 scale (train 평균) 로 정규화 → O(1)
    sc1 = np.nanmean(z1tr[m]) + 1e-12; sc2 = np.nanmean(z2tr[m]) + 1e-12
    z1tr_s = z1tr / sc1; z2tr_s = z2tr / sc2
    z1te_s = z1te / sc1; z2te_s = z2te / sc2
    yf = ytr[m]; z1f = z1tr_s[m]; z2f = z2tr_s[m]
    print(f"\n{'='*60}\n fold={fold}  train n={len(yf)}  test n={len(yte)}")
    p = fit_garch_x(yf, z1f, z2f)
    mu, om, al, be, g1, g2, nu = _unpack(p)
    print(f"  params: mu={mu:+.5f} om={om:.2e} al={al:.3f} be={be:.3f} "
          f"g_tbill={g1:.3f} g_metab={g2:.3f} nu={nu:.1f}")

    # test 실데이터 recursion → origin 별 σ²
    s2_te, eps_te = filter_var(p, yte, z1te_s, z2te_s)
    rng = np.random.default_rng(SEED)
    origins = [t for t in range(PAST_LEN, len(yte) - FUT)
               if np.isfinite(yte[t]) and np.isfinite(s2_te[t])
               and np.all(np.isfinite(yte[t + 1:t + 1 + FUT]))]
    sim = np.empty((len(origins), N_SIM, FUT), dtype=np.float32)
    act = np.empty((len(origins), FUT), dtype=np.float32)
    for i, t in enumerate(origins):
        Z1fut = z1te_s[t + 1:t + 1 + FUT]; Z2fut = z2te_s[t + 1:t + 1 + FUT]
        sim[i] = simulate(p, eps_te[t], s2_te[t], Z1fut, Z2fut, N_SIM, rng)
        act[i] = yte[t + 1:t + 1 + FUT]
    print(f"  origins={len(origins)}  n_sim={N_SIM}")

    crps_m, crps_s = crps_pooled(sim, act)
    af = act.ravel(); sf = sim.ravel()
    std_a = float(af.std(ddof=1)); std_s = float(sf.std(ddof=1))
    cv5a, cv5s = cvar(af, .05), cvar(sf, .05)
    cv1a, cv1s = cvar(af, .01), cvar(sf, .01)
    cov = {}
    for lvl, lo, hi in [(50, 25, 75), (80, 10, 90), (95, 2.5, 97.5)]:
        L = np.percentile(sim, lo, axis=1); H = np.percentile(sim, hi, axis=1)
        cov[lvl] = float(((act >= L) & (act <= H)).mean())
    print(f"  CRPS={crps_m:.5f}  std a/s/ratio={std_a:.5f}/{std_s:.5f}/{std_s/std_a:.3f}  "
          f"cov 50/80/95={cov[50]:.3f}/{cov[80]:.3f}/{cov[95]:.3f}")

    summ = dict(fold=fold, model="GARCH(1,1)-X-t (macro turbulence in variance)",
                params=dict(mu=mu, omega=om, alpha=al, beta=be, gamma_tbill=g1,
                            gamma_metab=g2, nu=nu),
                n_origins=len(origins), seed=SEED,
                test_eval=dict(crps_pooled=crps_m, crps_std=crps_s,
                               emd=emd1d(sf, af), std_actual=std_a, std_sim=std_s,
                               std_ratio=std_s / std_a, coverage_50=cov[50],
                               coverage_80=cov[80], coverage_95=cov[95],
                               cvar_5pct_diff=cv5s - cv5a, cvar_1pct_diff=cv1s - cv1a,
                               var_1pct_diff=var_q(sf, .01) - var_q(af, .01)))
    os.makedirs(RESULT_DIR, exist_ok=True)
    json.dump(summ, open(sp, "w"), indent=2, default=str)
    print(f"  saved {os.path.basename(sp)}")


def main():
    print(f"[garch-x] macro turbulence in variance  {len(FOLDS)} fold")
    for fold in FOLDS:
        if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                   for s in ("train", "test")):
            print(f"[skip {fold}] csv 없음"); continue
        try:
            run_fold(fold)
        except Exception as e:
            print(f"[FAIL] {fold}: {e!r}")

    print("\n" + "=" * 96)
    print("GARCH-X vs GARCH-N vs flow(cond/joint) vs diffusion  (같은 gap29 fold)")
    print("=" * 96)

    def row(path, sub=None):
        if not os.path.exists(path):
            return None
        d = json.load(open(path)); return d.get(sub, d) if sub else d
    for label, get in [
        ("GARCH-X",     lambda f: row(os.path.join(RESULT_DIR, f"garch_x_{f}_summary.json"), "test_eval")),
        ("GARCH-N",     lambda f: row(os.path.join(RESULT_DIR, f"garch_pure_{f}_summary.json"))),
        ("flow cond",   lambda f: row(os.path.join(RESULT_DIR, f"mamba_flow_ar_selfstat_cond_m26_mlp_s2026_{f}_summary.json"), "test_eval")),
        ("flow-joint",  lambda f: row(os.path.join(RESULT_DIR, f"flow_joint_{f}_summary.json"), "test_eval")),
    ]:
        print(f"\n[{label}]   {'fold':<16}{'CRPS':>9}{'cov95':>8}{'std_ratio':>11}{'CVaR5Δ':>10}")
        for fold in FOLDS:
            d = get(fold)
            if not d:
                print(f"{'':<6}{fold:<16}  (없음)"); continue

            def g(k):
                return d.get(k) or 0
            print(f"{'':<6}{fold:<16}{g('crps_pooled'):>9.5f}{g('coverage_95'):>8.3f}"
                  f"{g('std_ratio'):>11.3f}{g('cvar_5pct_diff'):>10.5f}")
    print("\n해석: GARCH-X 가 GARCH-N 보다 나으면 → 거시(난기류)가 GARCH 분산에 기여. "
          "GARCH-X 가 flow 와 동률이면 → flow 우위는 분포유연성으로 좁혀짐. "
          "g_metab≈0 이면 → per-week GARCH 가 window-scale metab 못 씀(=flow path-conditioning 이 edge).")


if __name__ == "__main__":
    main()
