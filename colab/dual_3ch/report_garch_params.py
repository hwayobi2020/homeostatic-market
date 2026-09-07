# -*- coding: utf-8 -*-
"""GARCH-ST 폴드별 적합 파라미터 표 — 리뷰어 1 #4.

리뷰어 지적
-----------
  "Table 11 은 네 폴드 모두에서 왜도가 0.01 로 나온다.  의미 있는 변동이
   없다는 말인가?  그 파라미터가 추정되는 것임을 고려하면 말이다. ...
   각 폴드에서 GARCH-ST 의 적합된 왜도 파라미터를 보고하라.  나아가 그것이
   고정된 것이 아니라 추정된 것임을 확인하라."

λ 는 고정이 아니다.  train_garch_xpast.fit_garch_xpast 가 μ, ω, α, β,
γ_tbill, γ_metab, ν, λ 8 개를 L-BFGS-B 로 동시에 최대우도 추정한다
(λ 초기값 0.0, 경계 (-0.95, 0.95)).  이 스크립트는 그 추정값을 폴드별로 찍는다.

다만 Hansen(1994) skew-t 의 λ 는 *상수* 라, 추정되더라도 시뮬레이션 내내
왜도가 고정이다 (train_garch_xpast.py:11 주석).  구조적 한계와 추정 결과는
구분해서 답해야 한다.

사용
----
    !python colab/dual_3ch/report_garch_params.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    pat = os.path.join(HERE, "result", "garch_xpast_*_summary.json")
    files = sorted(glob.glob(pat))
    if not files:
        print(f"결과 없음: {pat}")
        return

    hdr = ("{:20} {:>10} {:>8} {:>8} {:>8} {:>10} {:>10} {:>11} {:>10}"
           .format("fold", "lambda", "nu", "alpha", "beta", "g_tbill",
                   "g_metab", "skew_actual", "skew_sim"))
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        p = d.get("params") or {}
        te = d.get("test_eval") or {}

        def g(src, *keys):
            for k in keys:
                if isinstance(src.get(k), (int, float)):
                    return src[k]
            return None

        def fmt(v, spec, w):
            return format(v, spec) if isinstance(v, (int, float)) else "-".rjust(w)

        row = dict(fold=d.get("fold", os.path.basename(f)),
                   lam=g(p, "lambda_skew"), nu=g(p, "nu"),
                   alpha=g(p, "alpha"), beta=g(p, "beta"),
                   g1=g(p, "gamma_tbill"), g2=g(p, "gamma_metab"),
                   sk_a=g(te, "skew_actual"), sk_s=g(te, "skew_sim"))
        rows.append(row)
        print("{:20} {:>10} {:>8} {:>8} {:>8} {:>10} {:>10} {:>11} {:>10}".format(
            str(row["fold"]),
            fmt(row["lam"], "+.5f", 10), fmt(row["nu"], ".2f", 8),
            fmt(row["alpha"], ".3f", 8), fmt(row["beta"], ".3f", 8),
            fmt(row["g1"], "+.3f", 10), fmt(row["g2"], "+.3f", 10),
            fmt(row["sk_a"], "+.4f", 11), fmt(row["sk_s"], "+.4f", 10)))

    lams = [r["lam"] for r in rows if isinstance(r["lam"], (int, float))]
    if lams:
        print(f"\nlambda 범위 {min(lams):+.5f} ~ {max(lams):+.5f}  "
              f"(경계 -0.95 ~ +0.95, 초기값 0.0 에서 자유 추정)")
        if max(abs(x) for x in lams) < 0.01:
            print("→ 네 폴드 모두 0 근처로 추정됐다.  고정이 아니라 추정 결과다.")
    print("\n[읽는 법] lambda 가 경계(-0.95/+0.95)에 붙어 있으면 최적화가 막힌 것이고,")
    print("  0 근처면 우도가 대칭 혁신을 선호한 것이다.  전자면 재적합이 필요하다.")


if __name__ == "__main__":
    main()
