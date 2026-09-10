# -*- coding: utf-8 -*-
"""재적합 파라미터로 GARCH-ST 시험 지표 재생성 — 리뷰어 1 #4 후속.

`refit_garch_xpast.py` 가 낸 파라미터로 시뮬레이션·평가만 다시 한다.
`train_garch_xpast.py` 는 그대로 두고 filter_var / simulate / 지표 정의만 가져온다.
결과는 `garch_xpast_refit_{fold}_summary.json` 에 따로 쓴다 (기존 파일 미변경).

배경
----
기존 적합은 L-BFGS-B 가 5 회 반복 만에 조기 종료해 8 개 파라미터가 초기값
근처에 머물렀다.  재적합 결과 네 폴드 모두 λ 가 -0.137 ~ -0.153 으로 이동했고
우도비 λ=0 검정이 p = 4.3e-4 ~ 1.1e-6 로 유의했다.  ν 도 7.00 → 9.20~11.80 로
올라 꼬리가 얇아진다.  그래서 GARCH-ST 의 시험 지표(CRPS·EMD·커버리지·CVaR·
왜도·첨도)가 전부 달라진다.

논문 §4.1(:1280-1286)은 "두 모형이 CRPS·EMD 에서 comparable 하고 MAC-Flow 가
cov80/95 와 skew 에서 더 안정적" 이라고 적는다.  재적합이 그 서술을 어느
방향으로 움직이는지 확인하는 것이 이 스크립트의 목적이다.  방향은 미리 알 수
없고, GARCH-ST 쪽이 좋아질 수도 있다.

사용
----
    !python colab/dual_3ch/refit_garch_xpast.py      # 먼저 이걸 돌려 파라미터 생성
    !python colab/dual_3ch/eval_garch_xpast_refit.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from train_garch_xpast import (  # noqa: E402
    build, filter_var, simulate, crps_pooled, cvar, var_q, emd1d,
    FOLDS, FOLDS_DIR, RESULT_DIR, PAST_LEN, FUT, N_SIM, SEED,
)
from train_garch_x import crps_ensemble                            # noqa: E402

REFIT = os.path.join(RESULT_DIR, "refit_garch_xpast.json")
PKEYS = ("mu", "omega", "alpha", "beta", "gamma_tbill", "gamma_metab",
         "nu", "lambda_skew")


def _sk(a):
    a = np.asarray(a, float)
    return float(np.mean(((a - a.mean()) / (a.std() + 1e-12)) ** 3))


def _ek(a):
    a = np.asarray(a, float)
    return float(np.mean(((a - a.mean()) / (a.std() + 1e-12)) ** 4) - 3.0)


def eval_fold(fold, params):
    """train_garch_xpast.run_fold 의 평가부와 같은 정의.  파라미터만 재적합값."""
    tr = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_train.csv"))
    te = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_test.csv"))
    ytr, tbtr, mttr = build(tr)
    yte, tbte, mtte = build(te)

    m1 = np.isfinite(tbtr); m2v = np.isfinite(mttr)
    mu1, sd1 = float(np.nanmean(tbtr[m1])), float(np.nanstd(tbtr[m1]) + 1e-12)
    mu2, sd2 = float(np.nanmean(mttr[m2v])), float(np.nanstd(mttr[m2v]) + 1e-12)
    x1te = (tbte - mu1) / sd1
    x2te = (mtte - mu2) / sd2

    p = np.array([float(params[k]) for k in PKEYS], dtype=float)
    lam = p[7]

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

    crps_m, crps_s = crps_pooled(sim, act)
    af = act.ravel(); sf = sim.ravel()
    std_a = float(af.std(ddof=1)); std_s = float(sf.std(ddof=1))
    cov = {}
    for lvl, lo, hi in [(50, 25, 75), (80, 10, 90), (95, 2.5, 97.5)]:
        L = np.percentile(sf, lo); H = np.percentile(sf, hi)
        cov[lvl] = float(((af >= L) & (af <= H)).mean())
    cv5a, cv5s = cvar(af, .05), cvar(sf, .05)
    cv1a, cv1s = cvar(af, .01), cvar(sf, .01)

    te_eval = dict(crps_pooled=crps_m, crps_std=crps_s, emd=emd1d(sf, af),
                   std_actual=std_a, std_sim=std_s, std_ratio=std_s / std_a,
                   coverage_50=cov[50], coverage_80=cov[80], coverage_95=cov[95],
                   cvar_5pct_diff=cv5s - cv5a, cvar_1pct_diff=cv1s - cv1a,
                   var_1pct_diff=var_q(sf, .01) - var_q(af, .01),
                   skew_actual=_sk(af), skew_sim=_sk(sf),
                   exkurt_actual=_ek(af), exkurt_sim=_ek(sf))

    # DM 검정용: 원점×주 CRPS 와 조건 시점 날짜.  MAC-Flow 의
    # garch_flow_ar_*_crps_per_origin.npy / *_origin_dates.npy 와 같은 규약이다
    # (원점 t 의 조건 시점 = te 의 t 행, 예측 구간 = t+1 .. t+FUT).
    per_oc = np.array([[crps_ensemble(sim[i, :, w], act[i, w])
                        for w in range(FUT)]
                       for i in range(len(origins))])
    date_col = te["date"].values if "date" in te.columns else None
    dates = (np.array([str(date_col[t]) for t in origins]) if date_col is not None
             else np.array([str(t) for t in origins]))
    pref = os.path.join(RESULT_DIR, f"garch_xpast_refit_{fold}")
    np.save(f"{pref}_crps_per_origin.npy", per_oc)
    np.save(f"{pref}_origin_dates.npy", dates)

    print(f"  origins={len(origins)}  n_sim={N_SIM}  lam={lam:+.5f}")
    print(f"  saved per-origin CRPS: {os.path.basename(pref)}_crps_per_origin.npy "
          f"{per_oc.shape}")
    print(f"  CRPS={crps_m:.5f}  cov 50/80/95="
          f"{cov[50]:.3f}/{cov[80]:.3f}/{cov[95]:.3f}  "
          f"skew a/s={te_eval['skew_actual']:+.4f}/{te_eval['skew_sim']:+.4f}")

    summ = dict(fold=fold,
                model="GARCH-X(past)-skewt [재적합: 퍼센트 스케일 + 조인 종료조건]",
                params={k: float(params[k]) for k in PKEYS},
                n_origins=len(origins), seed=SEED, test_eval=te_eval)
    sp = os.path.join(RESULT_DIR, f"garch_xpast_refit_{fold}_summary.json")
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=2, default=str)
    print(f"  saved {os.path.basename(sp)}")
    return te_eval


def main():
    if not os.path.exists(REFIT):
        print(f"[FATAL] {REFIT} 없음.  먼저 refit_garch_xpast.py 를 돌려라.")
        return
    with open(REFIT, encoding="utf-8") as fh:
        refit = {r["fold"]: r for r in json.load(fh)}

    rows = []
    for fold in FOLDS:
        if fold not in refit:
            print(f"[skip {fold}] 재적합 결과 없음")
            continue
        print(f"\n{'=' * 70}\n fold={fold}")
        try:
            new = eval_fold(fold, refit[fold]["params"])
        except Exception as e:                                    # noqa: BLE001
            print(f"  [FAIL] {e!r}")
            continue

        old = {}
        op = os.path.join(RESULT_DIR, f"garch_xpast_{fold}_summary.json")
        if os.path.exists(op):
            with open(op, encoding="utf-8") as fh:
                old = (json.load(fh).get("test_eval") or {})
        rows.append((fold, old, new))

    if not rows:
        print("\n결과 없음")
        return

    # 지표별 전/후.  actual 열은 실측이라 재적합과 무관하게 같다.
    METRICS = [
        ("crps_pooled", ".5f", None, "낮을수록 좋다"),
        ("emd", ".6f", None, "낮을수록 좋다"),
        ("coverage_80", ".3f", None, "명목 0.80"),
        ("coverage_95", ".3f", None, "명목 0.95"),
        ("coverage_50", ".3f", None, "명목 0.50"),
        ("std_ratio", ".3f", None, "1.0 이 정확"),
        ("skew_sim", "+.4f", "skew_actual", "실측에 가까울수록 좋다"),
        ("exkurt_sim", "+.3f", "exkurt_actual", "실측에 가까울수록 좋다"),
        ("cvar_5pct_diff", "+.5f", None, "0 이 정확"),
        ("cvar_1pct_diff", "+.5f", None, "0 이 정확"),
        ("var_1pct_diff", "+.5f", None, "0 이 정확"),
    ]

    print("\n" + "=" * 110)
    print("[GARCH-ST 재적합 전/후]  기존 → 재적합, 괄호 안은 변화량")
    print("=" * 110)
    for key, spec, act_key, note in METRICS:
        act = ""
        if act_key:
            vals = [r[2].get(act_key) for r in rows
                    if isinstance(r[2].get(act_key), (int, float))]
            if vals:
                act = "  실측 " + " / ".join(format(v, spec) for v in vals)
        print(f"\n■ {key}  ({note}){act}")
        for fold, old, new in rows:
            o, n = old.get(key), new.get(key)
            if not isinstance(n, (int, float)):
                print(f"    {fold:20} -")
                continue
            a = (format(new[act_key], spec)
                 if act_key and isinstance(new.get(act_key), (int, float)) else None)
            if isinstance(o, (int, float)):
                d = n - o
                rel = (f"  {d / abs(o) * 100:+.1f}%" if abs(o) > 1e-12 else "")
                line = (f"    {fold:20} {format(o, spec):>12} → "
                        f"{format(n, spec):>12}   ({format(d, '+' + spec.lstrip('+'))}{rel})")
            else:
                line = f"    {fold:20} {'-':>12} → {format(n, spec):>12}"
            if a:
                line += f"   [실측 {a}]"
            print(line)

    print("\n[읽는 법]  논문 §4.1(:1280-1286) 은 두 모형이 CRPS·EMD 에서")
    print("  comparable 하고 MAC-Flow 가 cov80/95·skew 에서 더 안정적이라고 쓴다.")
    print("  · CRPS 가 크게 내려가면 'comparable' 서술이 GARCH-ST 우위로 바뀔 수 있다.")
    print("  · skew_sim 이 actual 에 가까워지면 'MAC-Flow 가 skew 에서 더 안정적'")
    print("    이라는 절이 약해진다.  MAC-Flow 값과 폴드별로 대조해야 한다.")
    print("  · λ 는 여전히 상수라 폴드 안에서 왜도가 시점에 따라 변하지는 못한다.")
    print("    §4.2·§4.3.1 의 조건부 왜도 대조는 이 재적합과 무관하다.")


if __name__ == "__main__":
    main()
