# -*- coding: utf-8 -*-
"""§4.1 리포팅 — 네 모형을 한 표로.

읽는 것 (전부 `result/` 의 summary json, 재실행 없음)
--------------------------------------------------
  MAC-Flow  : garch_flow_ar_{FLOW_TAG}_s{seed}_{fold}_summary.json   (5 시드)
  CondVAE   : vae_baseline_{VG_TAG}_s{seed}_{fold}_summary.json      (5 시드)
  CondGAN   : gan_baseline_{GAN_TAG}_s{seed}_{fold}_summary.json     (5 시드)
  GARCH-ST  : {GARCH_PREFIX}_{fold}_summary.json                     (MLE, 1 판)

내는 것
-------
  1. 폴드별 지표 표 (5 시드는 평균 ± 표준오차)
  2. MAC-Flow vs 각 베이스라인 시드 t검정
     - 베이스라인이 5 시드면 Welch 2 표본, GARCH 면 1 표본(상수 대비)
  3. 4 폴드 통합 — 목표가 있는 지표는 |오차| 로 바꿔 평균 후 검정
     (왜도·첨도·IHL 은 실측, cov 는 명목, std_ratio 는 1, CVaR/VaR diff 는 0)

사용
----
    !python colab/dual_3ch/report_section41.py
    # 태그 바꾸기: FLOW_TAG=... GARCH_PREFIX=garch_xpast_matched ...
"""
import json
import os
import sys

import numpy as np

try:
    from scipy import stats
except Exception:                                                  # noqa: BLE001
    sys.exit("[FATAL] scipy 필요")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "result")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FOLDS = [f for f in os.environ.get(
    "R41_FOLDS", "F_gfc,F_long_A,F_long_B_origin,F_long").split(",") if f.strip()]
SEEDS = [int(s) for s in os.environ.get(
    "R41_SEEDS", "2026,2027,2028,2029,2030").split(",") if s.strip()]

FLOW_TAG = os.environ.get("FLOW_TAG", "rvAbl_full_fpath_novol_d2")
VG_TAG = os.environ.get("VG_TAG", "t3skfu_c128h192_lr0d0001")
GAN_TAG = os.environ.get("GAN_TAG", "t3skfu_c192h128_lr0d0003")
GARCH_PREFIX = os.environ.get("GARCH_PREFIX", "garch_xpast_matched_fut")

# (지표, 목표).  목표가 None 이면 값 자체를 쓴다(낮을수록 좋음).
#   'ACT' 는 같은 json 의 실측 짝(skew_actual 등)을 목표로 삼는다.
METRICS = [
    ("crps_pooled",      None,  "CRPS"),
    ("emd",              None,  "EMD"),
    ("coverage_80",      0.80,  "cov80"),
    ("coverage_95",      0.95,  "cov95"),
    ("std_ratio",        1.00,  "std비"),
    ("skew_sim",         "ACT", "왜도"),
    ("exkurt_sim",       "ACT", "초과첨도"),
    ("cvar_1pct_diff",   0.0,   "CVaR1%오차"),
    ("cvar_5pct_diff",   0.0,   "CVaR5%오차"),
    ("var_1pct_diff",    0.0,   "VaR1%오차"),
    ("ihl_mean_diff",    0.0,   "IHL평균오차"),
    ("ihl_cvar10_diff",  0.0,   "IHL CVaR10오차"),
    ("ihl_cvar5_diff",   0.0,   "IHL CVaR5오차"),
]
ACT_OF = {"skew_sim": "skew_actual", "exkurt_sim": "exkurt_actual"}


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("test_eval") or {}


def seeded(prefix_fmt, fold):
    """5 시드 test_eval 목록.  하나라도 없으면 그 시드는 건너뛴다."""
    out = []
    for s in SEEDS:
        te = _load(os.path.join(RESULT_DIR, prefix_fmt.format(s=s, f=fold)))
        if te:
            out.append(te)
    return out


def target_of(key, te):
    tgt = dict((k, v) for k, v, _ in METRICS)[key]
    if tgt == "ACT":
        return te.get(ACT_OF[key])
    return tgt


def err(val, tgt):
    return abs(val - tgt) if tgt is not None else val


def collect(fold):
    """모형 → test_eval 목록 (GARCH 는 길이 1)."""
    d = {}
    d["MAC-Flow"] = seeded(f"garch_flow_ar_{FLOW_TAG}_s{{s}}_{{f}}_summary.json", fold)
    d["CondVAE"] = seeded(f"vae_baseline_{VG_TAG}_s{{s}}_{{f}}_summary.json", fold)
    d["CondGAN"] = seeded(f"gan_baseline_{GAN_TAG}_s{{s}}_{{f}}_summary.json", fold)
    g = _load(os.path.join(RESULT_DIR, f"{GARCH_PREFIX}_{fold}_summary.json"))
    d["GARCH-ST"] = [g] if g else []
    return d


def fmt(v, n=5):
    return "—" if v is None else f"{v:+.{n}f}"


def main():
    print("#" * 108)
    print("# §4.1 통합 리포트 — 네 모형, 같은 지표축 (꼬리위험 포함)")
    print(f"#  MAC-Flow={FLOW_TAG}  VAE={VG_TAG}  GAN={GAN_TAG}  GARCH={GARCH_PREFIX}")
    print(f"#  folds={FOLDS}  seeds={SEEDS}")
    print("#" * 108)

    data = {f: collect(f) for f in FOLDS}
    have = {m for f in FOLDS for m, v in data[f].items() if v}
    missing = {m for f in FOLDS for m, v in data[f].items() if not v}
    if missing:
        print(f"\n[없음] {sorted(missing)} — 해당 모형은 표에서 빠진다")

    # ---------------- 1. 폴드별 표 ----------------
    for f in FOLDS:
        print(f"\n{'=' * 108}\n[{f}]")
        hdr = f"{'지표':<16}" + "".join(f"{m:>22}" for m in
                                       ["MAC-Flow", "CondVAE", "CondGAN", "GARCH-ST"]
                                       if m in have)
        print(hdr)
        for key, _t, label in METRICS:
            row = f"{label:<16}"
            for m in ["MAC-Flow", "CondVAE", "CondGAN", "GARCH-ST"]:
                if m not in have:
                    continue
                tes = data[f].get(m) or []
                vals = [te[key] for te in tes if key in te]
                if not vals:
                    row += f"{'—':>22}"
                elif len(vals) == 1:
                    row += f"{vals[0]:>22.5f}"
                else:
                    a = np.array(vals, float)
                    row += f"{a.mean():>14.5f}±{a.std(ddof=1) / np.sqrt(len(a)):.5f}"
            print(row)
        # 실측 참조값
        ref = next((te for m in ["MAC-Flow", "GARCH-ST"] for te in (data[f].get(m) or [])), {})
        print(f"{'(실측)':<16}왜도 {fmt(ref.get('skew_actual'), 4)}  "
              f"초과첨도 {fmt(ref.get('exkurt_actual'), 3)}  "
              f"IHL평균 {fmt(ref.get('ihl_mean_actual'))}  "
              f"IHL CVaR10 {fmt(ref.get('ihl_cvar10_actual'))}")

    # ---------------- 2. 4 폴드 통합 검정 ----------------
    print(f"\n{'=' * 108}")
    print("[4 폴드 통합 · 목표 대비 |오차| 평균]  낮을수록 좋다")
    print("  MAC-Flow 는 시드별 4 폴드 평균 5 개.  베이스라인이 5 시드면 Welch,")
    print("  GARCH(MLE 1 판)면 그 상수에 대한 1 표본 t검정이다.")
    print("=" * 108)
    print(f"{'지표':<16}{'MAC-Flow':>20}{'상대':<12}{'베이스라인':>14}{'t':>8}{'p':>9}")

    for key, _t, label in METRICS:
        # MAC-Flow 시드별 4폴드 평균 오차
        mac = []
        for s_i in range(len(SEEDS)):
            per_fold = []
            for f in FOLDS:
                tes = data[f].get("MAC-Flow") or []
                if s_i >= len(tes) or key not in tes[s_i]:
                    per_fold = []
                    break
                tgt = target_of(key, tes[s_i])
                per_fold.append(err(tes[s_i][key], tgt))
            if per_fold:
                mac.append(float(np.mean(per_fold)))
        if not mac:
            continue
        X = np.array(mac, float)

        for base in ["CondVAE", "CondGAN", "GARCH-ST"]:
            if base not in have:
                continue
            rows = []
            n_seed_b = max(len(data[f].get(base) or []) for f in FOLDS)
            for s_i in range(max(1, n_seed_b)):
                per_fold = []
                for f in FOLDS:
                    tes = data[f].get(base) or []
                    if s_i >= len(tes) or key not in tes[s_i]:
                        per_fold = []
                        break
                    tgt = target_of(key, tes[s_i])
                    per_fold.append(err(tes[s_i][key], tgt))
                if per_fold:
                    rows.append(float(np.mean(per_fold)))
            if not rows:
                continue
            Y = np.array(rows, float)
            if Y.size == 1:
                t, p = stats.ttest_1samp(X, Y[0])
                btxt = f"{Y[0]:>14.5f}"
            else:
                t, p = stats.ttest_ind(X, Y, equal_var=False)
                btxt = f"{Y.mean():>8.5f}±{Y.std(ddof=1) / np.sqrt(Y.size):.5f}"
            mark = "" if not np.isfinite(p) else ("***" if p < .01 else
                                                  "**" if p < .05 else
                                                  "*" if p < .10 else "")
            print(f"{label:<16}{X.mean():>12.5f}±{X.std(ddof=1) / np.sqrt(X.size):.5f}"
                  f"{'vs ' + base:<12}{btxt}{t:>8.2f}{p:>9.4f} {mark}")
        print()

    print("  *** p<.01  ** p<.05  * p<.10")
    print("  주의: 이 검정은 학습(시드) 변동만 반영한다.  원점 표본 불확실성은")
    print("        DM 검정(dm_garch_compare.py / run_thin_compare.py)이 담당한다.")


if __name__ == "__main__":
    main()
