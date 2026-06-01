"""raw-vol 튜닝 결과 집계 — Phase 2 summary json → encoder별/본모형 지표 표.

목적
----
master(run_master_rawvol_tuning.py)가 본모형 압축기를 val NLL 로 선택했으나,
본 논문은 tail-risk 논문이므로 NLL 만으로 채택을 확정하면 안 된다.  Phase 2 가
4 encoder 를 5-seed × 3-fold 로 모두 돌렸으므로, 그 결과를 꼬리 지표(skew_sim,
CVaR_1%, coverage)까지 포함해 집계하여 *올바른 기준*으로 본모형을 판단한다.

읽는 파일 (학습 0, summary json 만):
  - ablation : garch_flow_ar_rvP2abl_{enc}_*_{fold}_summary.json   (Table 4.7)
  - main     : garch_flow_ar_rvP2main_*_{fold}_summary.json        (Table 4.1)

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !python colab/dual_3ch/agg_rawvol_tuning.py
출력 표 전체를 그대로 paste 하면 Table 4.7/4.1 작성 + 본모형 채택 기준 검토.
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "result")

ENCODERS = ["mlp", "lstm", "transformer", "mamba"]
FOLDS_TUNE = ["F_long_A", "F_long_B_origin", "F_long"]
FOLDS_MAIN = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]

# (key, 라벨, 좋은 방향)  — skew_sim 은 좌측(−) 이 위기 재현
METRICS = [
    ("per_week_nll_z", "NLL/wk", "낮을수록"),
    ("crps_pooled",    "CRPS",   "낮을수록"),
    ("emd",            "EMD",    "낮을수록"),
    ("cvar_1pct_sim",  "CVaR1%", "깊을수록(−)"),
    ("skew_sim",       "skew",   "좌측(−)"),
    ("coverage_80",    "cov80",  "0.80 근접"),
    ("coverage_95",    "cov95",  "0.95 근접"),
    ("std_ratio",      "std비",  "1.0 근접"),
]


def load_te(path):
    try:
        return json.load(open(path)).get("test_eval", {})
    except (json.JSONDecodeError, OSError):
        return {}


def fold_of(fname):
    for fold in FOLDS_MAIN:
        if fname.endswith(f"_{fold}_summary.json"):
            return fold
    return None


def ms(vals):
    a = np.asarray([v for v in vals if isinstance(v, (int, float))], dtype=float)
    if len(a) == 0:
        return None, None, 0
    return float(a.mean()), float(a.std(ddof=0)), len(a)


def fmt(mn, sd, n):
    return f"{mn:+.4f}±{sd:.4f}(n{n})" if mn is not None else "    NA    "


# ════════════════════════════════════════════════════════════════════
# Table 4.7 — encoder ablation : 5 seed × 3 fold pooled (per encoder)
# ════════════════════════════════════════════════════════════════════
print("=" * 120)
print("[Table 4.7] Past Encoder ablation — 5 seed × 3 fold pooled (rvP2abl)")
print("  방향: NLL/CRPS/EMD 낮을수록 | CVaR1% 깊을수록(−) | skew 좌측(−) | cov80→0.80 cov95→0.95 | std비→1.0")
print("=" * 120)
hdr = f"{'encoder':<12}" + "".join(f"{lbl:>20}" for _, lbl, _ in METRICS)
print(hdr)
print("-" * len(hdr))

abl_means = {}   # enc -> {key: mean}
for enc in ENCODERS:
    files = glob.glob(os.path.join(RESULT_DIR, f"garch_flow_ar_rvP2abl_{enc}_*_summary.json"))
    files = [f for f in files if fold_of(os.path.basename(f)) in FOLDS_TUNE]
    coll = defaultdict(list)
    for f in files:
        te = load_te(f)
        for key, _, _ in METRICS:
            if isinstance(te.get(key), (int, float)):
                coll[key].append(te[key])
    row = f"{enc:<12}"
    abl_means[enc] = {}
    for key, _, _ in METRICS:
        mn, sd, n = ms(coll[key])
        abl_means[enc][key] = mn
        row += f"{fmt(mn, sd, n):>20}"
    print(row)
print("-" * len(hdr))

# 채택 기준 비교 안내
print("\n[채택 기준 검토] val NLL 최저 ≠ tail 최고일 수 있음. 아래로 교차 확인:")
for crit_key, crit_lbl, good in [("per_week_nll_z", "NLL/wk(낮을수록)", "min"),
                                 ("skew_sim", "skew(좌측 −)", "min"),
                                 ("cvar_1pct_sim", "CVaR1%(깊을수록 −)", "min"),
                                 ("coverage_80", "cov80(0.80 근접)", "near80")]:
    vals = {e: abl_means[e].get(crit_key) for e in ENCODERS if abl_means[e].get(crit_key) is not None}
    if not vals:
        print(f"  {crit_lbl:<22}: (데이터 없음)")
        continue
    if good == "min":
        win = min(vals, key=vals.get)
    else:  # cov80 → |x−0.80| 최소
        win = min(vals, key=lambda e: abs(vals[e] - 0.80))
    detail = "  ".join(f"{e}={vals[e]:+.4f}" for e in ENCODERS if e in vals)
    print(f"  {crit_lbl:<22} best={win:<12} | {detail}")


# ════════════════════════════════════════════════════════════════════
# Table 4.1 — 본모형 (rvP2main) : fold별 5-seed mean±std
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 120)
print("[Table 4.1] 본모형 (rvP2main) — fold별 5-seed mean±std")
print("=" * 120)
main_files = glob.glob(os.path.join(RESULT_DIR, "garch_flow_ar_rvP2main_*_summary.json"))
if not main_files:
    print("  (rvP2main summary 없음)")
else:
    # main tag 추출 (압축기/flow 확인)
    tags = set(re.sub(r"_(F_gfc|F_long_A|F_long_B_origin|F_long)_summary\.json$", "",
                      os.path.basename(f)) for f in main_files)
    print(f"  main tag(s): {sorted(tags)}")
    hdr2 = f"{'fold':<18}" + "".join(f"{lbl:>20}" for _, lbl, _ in METRICS)
    print(hdr2)
    print("-" * len(hdr2))
    for fold in FOLDS_MAIN:
        ff = [f for f in main_files if fold_of(os.path.basename(f)) == fold]
        coll = defaultdict(list)
        for f in ff:
            te = load_te(f)
            for key, _, _ in METRICS:
                if isinstance(te.get(key), (int, float)):
                    coll[key].append(te[key])
        row = f"{fold:<18}"
        for key, _, _ in METRICS:
            mn, sd, n = ms(coll[key])
            row += f"{fmt(mn, sd, n):>20}"
        print(row)

print("\n[안내] 이 출력을 paste 하면 Table 4.7/4.1 작성 + 본모형 채택 기준(NLL vs tail) 검토.")
