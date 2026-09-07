# -*- coding: utf-8 -*-
"""GARCH-ST 재적합 진단 — 리뷰어 1 #4.

기존 `train_garch_xpast.py` 는 건드리지 않는다.  거기서 neg_ll / build 만 가져와
적합만 다시 하고, 결과를 별도 파일(refit_garch_xpast.json)에 쓴다.

무엇이 문제였나
---------------
저장된 적합값이 네 폴드 모두 초기값 근처였다 (α 0.080, β 0.879, ν 7.00,
λ -0.0007~-0.0009; 초기값은 0.08, 0.88, 7.0, 0.0).  데이터가 다른데 결과가 같다.

합성 데이터로 재현해 원인을 확인했다.  유한차분 잡음이 아니라 **기울기 크기
불균형에 의한 조기 종료** 다.  y 가 원 수익률(주간 sd 0.02~0.035)이라
∂nll/∂ω 가 1/σ² 스케일로 커져 기울기 노름을 독점하고, 반복당 f 의 상대 감소가
L-BFGS-B 기본 종료 조건(factr·epsmch = 2.22e-9) 아래로 떨어져 5 회 만에
success=True 로 끝난다.  그래서 `res.success` 검사로는 이 실패를 못 잡는다.

무엇을 바꿨나
-------------
  1. y 를 퍼센트로 올려 적합 (FIT_SCALE=100).  GARCH 는 스케일 등변이라
     μ→cμ, ω→c²ω 이고 α,β,γ,ν,λ 는 불변이다.  적합 후 μ, ω 만 되돌린다.
     등변성은 수치로 확인했다: nll(p0, ×100) − nll(p0, ×1) = n·log(100) 상수.
  2. ftol/gtol 을 조이고 반복 상한을 올린다.  리스케일만으로는 부족하다.
  3. λ 다중 출발 (-0.4, -0.2, 0.0, +0.2).  리뷰어 질문이 λ 에 관한 것이라
     초기값 의존을 배제해야 한다.
  4. λ=0 제약 적합을 따로 해 우도비 검정을 낸다.  "λ 가 정말 추정되며 0 과
     유의하게 다른가" 에 직접 답한다.

이 스크립트는 기존 결과를 덮어쓰지 않는다.  시뮬레이션·평가도 하지 않는다.
적합이 제대로 되는지부터 확인하고, 그 다음에 본 실행을 다시 돌릴지 정한다.

사용
----
    !python colab/dual_3ch/refit_garch_xpast.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from train_garch_xpast import (  # noqa: E402
    neg_ll, build, hansen_sample, FOLDS, FOLDS_DIR, RESULT_DIR, SEED,
)

FIT_SCALE = 100.0
LAM_STARTS = [-0.4, -0.2, 0.0, 0.2]
OPTS = dict(maxiter=5000, maxfun=100000, ftol=1e-14, gtol=1e-12)
OUT = os.path.join(RESULT_DIR, "refit_garch_xpast.json")

BNDS = [(-1 * FIT_SCALE, 1 * FIT_SCALE), (1e-10, None),
        (1e-6, 0.5), (1e-6, 0.999),
        (-1.5, 1.5), (-1.5, 1.5), (2.1, 60.0), (-0.95, 0.95)]


def _p0(ys, lam0):
    return [float(np.mean(ys)), 0.05 * float(np.var(ys)),
            0.08, 0.88, 0.0, 0.0, 7.0, lam0]


def fit_free(ys, x1, x2):
    """λ 다중 출발 자유 적합.  가장 낮은 neg_ll 을 고른다."""
    best = None
    for lam0 in LAM_STARTS:
        r = minimize(neg_ll, _p0(ys, lam0), args=(ys, x1, x2),
                     method="L-BFGS-B", bounds=BNDS, options=OPTS)
        print(f"      lam0={lam0:+.2f} → neg_ll={r.fun:.4f}  "
              f"lam={r.x[7]:+.5f} nu={r.x[6]:.2f}  nit={r.nit}")
        if best is None or r.fun < best.fun:
            best = r
    return best


def fit_lam0(ys, x1, x2):
    """λ=0 제약 적합.  나머지 7 개만 움직인다 (우도비 검정의 귀무모형)."""
    def nll7(q, y_, a_, b_):
        return neg_ll(np.concatenate([q, [0.0]]), y_, a_, b_)

    return minimize(nll7, _p0(ys, 0.0)[:7], args=(ys, x1, x2),
                    method="L-BFGS-B", bounds=BNDS[:7], options=OPTS)


def sample_skew(nu, lam, n=400000, seed=SEED):
    """적합된 (ν, λ) 가 만드는 혁신 분포의 왜도.  시뮬 왜도의 상한이 된다."""
    z = hansen_sample(nu, lam, n, np.random.default_rng(seed))
    return float(np.mean(((z - z.mean()) / (z.std() + 1e-12)) ** 3))


def one_fold(fold):
    tr = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_train.csv"))
    ytr, tbtr, mttr = build(tr)
    m1 = np.isfinite(tbtr); m2v = np.isfinite(mttr)
    x1 = (tbtr - np.nanmean(tbtr[m1])) / (np.nanstd(tbtr[m1]) + 1e-12)
    x2 = (mttr - np.nanmean(mttr[m2v])) / (np.nanstd(mttr[m2v]) + 1e-12)
    m = np.isfinite(ytr) & np.isfinite(x1) & np.isfinite(x2)
    y, x1, x2 = ytr[m], x1[m], x2[m]
    ys = y * FIT_SCALE

    print(f"\n{'=' * 78}\n fold={fold}  n={len(ys)}")
    print("  [자유 적합 — λ 다중 출발]")
    rf = fit_free(ys, x1, x2)
    print("  [제약 적합 — λ=0]")
    r0 = fit_lam0(ys, x1, x2)
    print(f"      neg_ll={r0.fun:.4f}  nit={r0.nit}")

    # 우도비 검정: 2(제약 neg_ll − 자유 neg_ll) ~ χ²(1)
    lr = 2.0 * (float(r0.fun) - float(rf.fun))
    pval = float(chi2.sf(lr, df=1)) if lr > 0 else 1.0

    p = rf.x.copy()
    p[0] /= FIT_SCALE            # μ 되돌리기
    p[1] /= FIT_SCALE ** 2       # ω 되돌리기 (분산이라 c²)
    mu, om, al, be, g1, g2, nu, lam = p

    # 기존 저장값(=조기 종료 결과)과 대조
    old = {}
    sp = os.path.join(RESULT_DIR, f"garch_xpast_{fold}_summary.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as fh:
            d = json.load(fh)
        old = d.get("params") or {}

    # 학습기간 실측 왜도.  λ 는 상수라 학습기간의 *평균* 비대칭 하나만 잡는다.
    # 학습기간에 위기가 없으면 λ̂ 이 작게 나오고, 시험기간 위기의 큰 음의 왜도는
    # 애초에 낼 수 없다.  적합 실패와 구조적 한계를 구분하려면 이 값이 필요하다.
    _z = (y - y.mean()) / (y.std() + 1e-12)
    skew_train = float(np.mean(_z ** 3))
    sk = sample_skew(nu, lam)
    print(f"  재적합: mu={mu:+.6f} om={om:.3e} al={al:.4f} be={be:.4f} "
          f"g_tbill={g1:+.4f} g_metab={g2:+.4f} nu={nu:.2f} lam={lam:+.5f}")
    if old:
        print(f"  기존값: al={old.get('alpha'):.4f} be={old.get('beta'):.4f} "
              f"nu={old.get('nu'):.2f} lam={old.get('lambda_skew'):+.5f}")
    print(f"  학습기간 실측 왜도 = {skew_train:+.4f}   "
          f"혁신 왜도(λ,ν 로부터) = {sk:+.4f}")
    print(f"  우도비 λ=0 검정: LR={lr:.3f}  p={pval:.4g}"
          f"   → {'λ 는 0 과 유의하게 다르다' if pval < 0.05 else 'λ=0 을 기각 못 한다'}")

    return dict(fold=fold, n=int(len(ys)),
                neg_ll_free=float(rf.fun), neg_ll_lam0=float(r0.fun),
                lr_stat=lr, lr_pvalue=pval,
                nit=int(rf.nit), success=bool(rf.success), message=str(rf.message),
                params=dict(mu=float(mu), omega=float(om), alpha=float(al),
                            beta=float(be), gamma_tbill=float(g1),
                            gamma_metab=float(g2), nu=float(nu),
                            lambda_skew=float(lam)),
                innov_skew=sk, skew_train=skew_train, old_params=old)


def main():
    print("#" * 78)
    print("# GARCH-ST 재적합 — 퍼센트 스케일 + 조인 종료조건 + λ 다중 출발")
    print(f"#  FIT_SCALE={FIT_SCALE}  lam0={LAM_STARTS}  opts={OPTS}")
    print("#  기존 결과 파일은 덮어쓰지 않는다.")
    print("#" * 78)

    rows = []
    for fold in FOLDS:
        try:
            rows.append(one_fold(fold))
        except Exception as e:                                    # noqa: BLE001
            print(f"  [FAIL {fold}] {e!r}")
    if not rows:
        print("\n결과 없음")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)

    print("\n" + "=" * 100)
    print("[재적합 요약]  괄호 안은 기존(조기 종료) 값")
    print("=" * 100)
    hdr = ("{:18} {:>18} {:>14} {:>11} {:>11} {:>9} {:>9}"
           .format("fold", "lambda", "nu", "skew_train", "innov_skew",
                   "LR", "p"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        p = r["params"]; o = r["old_params"] or {}

        def pair(new, key, spec):
            s = format(new, spec)
            return (f"{s} ({format(o[key], spec)})"
                    if isinstance(o.get(key), (int, float)) else s)

        print("{:18} {:>18} {:>14} {:>11} {:>11} {:>9} {:>9}".format(
            r["fold"],
            pair(p["lambda_skew"], "lambda_skew", "+.4f"),
            pair(p["nu"], "nu", ".2f"),
            format(r["skew_train"], "+.4f"),
            format(r["innov_skew"], "+.4f"),
            format(r["lr_stat"], ".2f"),
            format(r["lr_pvalue"], ".3g")))

    print(f"\n저장: {os.path.basename(OUT)}")
    print("\n[읽는 법]")
    print("  · lambda 가 기존값(0 근처)에서 크게 벗어나면 예전 적합이 실패였다는 뜻이다.")
    print("  · LR 검정이 유의하면 λ 는 실제로 추정되며 0 과 다르다 — 리뷰어 1 #4 에")
    print("    'λ 는 추정되지만 값이 작다' 가 아니라 근거 있는 답을 할 수 있다.")
    print("  · 재적합값이 기존과 크게 다르면 GARCH-ST 의 모든 지표(CRPS·커버리지·")
    print("    CVaR·왜도)를 다시 만들어야 한다.  §4.1.1 비교 전체가 영향을 받는다.")
    print("  · skew_train 과 lambda 를 같이 봐라.  λ 는 상수라 학습기간의 평균")
    print("    비대칭 하나만 잡는다.  skew_train 이 0 근처인데 λ̂ 도 0 근처면")
    print("    적합 실패가 아니라 학습기간에 잡을 비대칭이 없었던 것이고, 시험기간")
    print("    위기의 큰 음의 왜도는 재적합해도 못 낸다 — 구조적 한계다.")


if __name__ == "__main__":
    main()
