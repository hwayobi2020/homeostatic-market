"""GARCH(1,1)-X-FHS — Filtered Historical Simulation baseline (empirical innovations).

배경(2026-05-25): 대칭 GARCH-t 는 skew=0 이라 실제 좌측비대칭(폭락)을 못 낸다.  하지만
  GARCH 가 skew 를 *못* 한다는 건 아니다 — innovation 분포만 비대칭으로 바꾸면 된다.
  parametric Hansen skew-t 는 공식 오류 위험이 있어, 가정 없는 표준 방법인 FHS 로 검증한다:

스펙:
  - GARCH(1,1)-X 분산 재귀·MLE 는 train_garch_x 와 동일(거시 난기류 외생항 포함).
  - innovation: train 의 **표준화 잔차 ẑ_t = ε_t/σ_t 를 부트스트랩**(empirical).
      → 실제 잔차의 skew·kurtosis 가 시뮬 수익률에 구조적으로 그대로 실린다.
  - eval: train_garch_x 의 crps_pooled/cvar/emd 정의 그대로 재사용(사과-사과).  같은 gap29 4 fold.

검증 질문: flow 는 F_gfc 에서 sim skew=+0.29 로 actual(−0.83) 의 좌측비대칭을 *놓쳤다*.
  GARCH-FHS 가 그 좌측 skew 를 잡으면(=skew_sim 음수) → "GARCH 도 skew 되고 flow 가 졌다".
  못 잡으면(skew_sim≈0) → train 잔차에 좌측 skew 가 없던 것(=잡을 게 없었음).

Usage (Colab): !python colab/dual_3ch/train_garch_fhs.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# train_garch_x 의 fit·필터·지표·데이터로더를 그대로 재사용 (동일 정의 보장).
from train_garch_x import (  # noqa: E402
    FOLDS_DIR, RESULT_DIR, FOLDS, PAST_LEN, FUT, N_SIM, SEED,
    crps_pooled, cvar, var_q, emd1d, build_series, _unpack,
    fit_garch_x, filter_var,
)


def _skew(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def _exkurt(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 4) - 3.0)


def simulate_fhs(p, eps_orig, s2_orig, Z1f, Z2f, n_sim, rng, pool):
    """origin 에서 13주 forward 시뮬.  innovation = empirical 표준화잔차 부트스트랩(pool)."""
    mu, om, al, be, g1, g2, nu = _unpack(p)
    s2 = np.full(n_sim, s2_orig)
    e2prev = np.full(n_sim, eps_orig ** 2)
    out = np.empty((n_sim, FUT))
    for h in range(FUT):
        z1 = Z1f[h] if np.isfinite(Z1f[h]) else 0.0
        z2 = Z2f[h] if np.isfinite(Z2f[h]) else 0.0
        s2 = np.maximum(om + al * e2prev + be * s2 + g1 * z1 + g2 * z2, 1e-12)
        z = rng.choice(pool, size=n_sim)          # FHS: 표준화 잔차 부트스트랩
        eps = np.sqrt(s2) * z
        out[:, h] = mu + eps
        e2prev = eps ** 2
    return out


def run_fold(fold):
    sp = os.path.join(RESULT_DIR, f"garch_fhs_{fold}_summary.json")
    if os.path.exists(sp):
        print(f"[skip] {os.path.basename(sp)}"); return
    tr = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_train.csv"))
    te = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_test.csv"))
    ytr, z1tr, z2tr = build_series(tr)
    yte, z1te, z2te = build_series(te)
    m = np.isfinite(ytr) & np.isfinite(z1tr) & np.isfinite(z2tr)
    sc1 = np.nanmean(z1tr[m]) + 1e-12; sc2 = np.nanmean(z2tr[m]) + 1e-12
    z1tr_s = z1tr / sc1; z2tr_s = z2tr / sc2
    z1te_s = z1te / sc1; z2te_s = z2te / sc2
    yf = ytr[m]; z1f = z1tr_s[m]; z2f = z2tr_s[m]
    print(f"\n{'='*60}\n fold={fold}  train n={len(yf)}  test n={len(yte)}")

    p = fit_garch_x(yf, z1f, z2f)
    mu, om, al, be, g1, g2, nu = _unpack(p)
    print(f"  params: mu={mu:+.5f} al={al:.3f} be={be:.3f} "
          f"g_tbill={g1:.3f} g_metab={g2:.3f} nu={nu:.1f}")

    # FHS innovation pool = train 표준화 잔차 (skew·kurt 보유), unit-var 로 정규화
    s2_tr, eps_tr = filter_var(p, yf, z1f, z2f)
    pool = eps_tr / np.sqrt(s2_tr)
    pool = pool[np.isfinite(pool)]
    pool = pool / (pool.std() + 1e-12)
    print(f"  resid pool: n={len(pool)}  skew={_skew(pool):+.4f}  "
          f"exkurt={_exkurt(pool):+.4f}  (이게 sim 에 그대로 실림)")

    s2_te, eps_te = filter_var(p, yte, z1te_s, z2te_s)
    rng = np.random.default_rng(SEED)
    origins = [t for t in range(PAST_LEN, len(yte) - FUT)
               if np.isfinite(yte[t]) and np.isfinite(s2_te[t])
               and np.all(np.isfinite(yte[t + 1:t + 1 + FUT]))]
    sim = np.empty((len(origins), N_SIM, FUT), dtype=np.float32)
    act = np.empty((len(origins), FUT), dtype=np.float32)
    for i, t in enumerate(origins):
        sim[i] = simulate_fhs(p, eps_te[t], s2_te[t],
                              z1te_s[t + 1:t + 1 + FUT], z2te_s[t + 1:t + 1 + FUT],
                              N_SIM, rng, pool)
        act[i] = yte[t + 1:t + 1 + FUT]
    print(f"  origins={len(origins)}  n_sim={N_SIM}")

    crps_m, crps_s = crps_pooled(sim, act)
    af = act.ravel(); sf = sim.ravel()
    std_a = float(af.std(ddof=1)); std_s = float(sf.std(ddof=1))
    skew_a, skew_s = _skew(af), _skew(sf)
    kurt_a, kurt_s = _exkurt(af), _exkurt(sf)
    cov = {}
    for lvl, lo, hi in [(50, 25, 75), (80, 10, 90), (95, 2.5, 97.5)]:
        L = np.percentile(sim, lo, axis=1); H = np.percentile(sim, hi, axis=1)
        cov[lvl] = float(((act >= L) & (act <= H)).mean())
    print(f"  CRPS={crps_m:.5f}  std a/s/ratio={std_a:.5f}/{std_s:.5f}/{std_s/std_a:.3f}  "
          f"cov 50/80/95={cov[50]:.3f}/{cov[80]:.3f}/{cov[95]:.3f}")
    print(f"  skew a/s={skew_a:+.4f}/{skew_s:+.4f}  exkurt a/s={kurt_a:+.4f}/{kurt_s:+.4f}  "
          f"(FHS = empirical skew; flow F_gfc 는 +0.29 로 부호 틀렸음)")

    summ = dict(fold=fold, model="GARCH(1,1)-X-FHS (empirical innovations)",
                params=dict(mu=mu, alpha=al, beta=be, gamma_tbill=g1, gamma_metab=g2),
                n_origins=len(origins), seed=SEED,
                pool_skew=_skew(pool), pool_exkurt=_exkurt(pool),
                test_eval=dict(crps_pooled=crps_m, crps_std=crps_s, emd=emd1d(sf, af),
                               std_actual=std_a, std_sim=std_s, std_ratio=std_s / std_a,
                               coverage_50=cov[50], coverage_80=cov[80], coverage_95=cov[95],
                               cvar_5pct_diff=cvar(sf, .05) - cvar(af, .05),
                               cvar_1pct_diff=cvar(sf, .01) - cvar(af, .01),
                               var_1pct_diff=var_q(sf, .01) - var_q(af, .01),
                               skew_actual=skew_a, skew_sim=skew_s,
                               exkurt_actual=kurt_a, exkurt_sim=kurt_s))
    os.makedirs(RESULT_DIR, exist_ok=True)
    per_oc = np.array([[crps_ensemble(sim[i, :, t], act[i, t]) for t in range(FUT)]
                       for i in range(len(origins))])
    np.save(os.path.join(RESULT_DIR, f"garch_fhs_{fold}_crps_per_origin.npy"), per_oc)
    json.dump(summ, open(sp, "w"), indent=2, default=str)
    print(f"  saved {os.path.basename(sp)}")


def main():
    print(f"[garch-fhs] filtered historical simulation  {len(FOLDS)} fold")
    for fold in FOLDS:
        if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                   for s in ("train", "test")):
            print(f"[skip {fold}] csv 없음"); continue
        try:
            run_fold(fold)
        except Exception as e:
            print(f"[FAIL] {fold}: {e!r}")

    print("\n" + "=" * 70)
    print("GARCH-FHS skew 결과 해석:")
    print("  skew_sim 이 actual(음수) 따라가면 → GARCH-FHS 가 좌측비대칭 잡음 = flow(+0.29) 패배.")
    print("  skew_sim ≈ 0 이면 → train 잔차에 좌측 skew 가 없던 것(잡을 게 없었음).")


if __name__ == "__main__":
    main()
