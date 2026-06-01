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
# Phase 1a/1b — 선택 근거 (단일 seed 2026 VAL NLL) + Table 3.7 재료
#   주의: 마스터 select_1a/1b 는 이 VAL NLL 로 본모형을 골랐다.
#   아래 Table 4.7 (5-seed TEST) 과 순위가 다를 수 있다 (val≠test, 1seed≠5seed).
# ════════════════════════════════════════════════════════════════════
P1A_PAST_DIM = [32, 64, 128]
P1A_DROPOUT = [0.1, 0.2]
P1B_FLOW_LAYERS = [4, 6, 8]
P1B_FLOW_HIDDEN = [32, 64, 128]


def read_val(tag, fold):
    fp = os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{fold}_summary.json")
    try:
        return json.load(open(fp)).get("best_val_nll_per_week")
    except (json.JSONDecodeError, OSError):
        return None


print("=" * 120)
print("[Phase 1a] 압축기 선택 근거 — 단일 seed(2026) 3-fold avg VAL NLL (= 마스터 선택 기준, Table 3.7 재료)")
print("=" * 120)
p1a_best = {}
for enc in ENCODERS:
    cands = []
    for pd in P1A_PAST_DIM:
        for dr in P1A_DROPOUT:
            tag = f"rvP1a_{enc}_pd{pd}_dr{dr}"
            vals = [read_val(tag, fold) for fold in FOLDS_TUNE]
            if all(v is not None for v in vals):
                cands.append((pd, dr, sum(vals) / len(vals)))
    if cands:
        pd, dr, avg = min(cands, key=lambda x: x[2])
        p1a_best[enc] = (pd, dr, avg)
        allc = "  ".join(f"pd{c[0]}dr{c[1]}={c[2]:.4f}"
                         for c in sorted(cands, key=lambda x: x[2]))
        print(f"  {enc:12s} best: pd={pd} dr={dr}  avg_val_NLL={avg:.4f}")
        print(f"  {'':12s}   전체: {allc}")
    else:
        print(f"  {enc:12s} (3-fold 미완)")
if p1a_best:
    win = min(p1a_best, key=lambda e: p1a_best[e][2])
    rank = "  >  ".join(f"{e}({p1a_best[e][2]:.4f})"
                        for e in sorted(p1a_best, key=lambda e: p1a_best[e][2]))
    print(f"  → 마스터 선택(val NLL 최저) = {win}")
    print(f"    val NLL 순위: {rank}")

print("\n" + "=" * 120)
print("[Phase 1b] flow 격자 — 단일 seed(2026) 3-fold avg VAL NLL (압축기=마스터 선택 고정)")
print("=" * 120)
fb = []
for nl in P1B_FLOW_LAYERS:
    for nh in P1B_FLOW_HIDDEN:
        tag = f"rvP1b_fl{nl}_fh{nh}"
        vals = [read_val(tag, fold) for fold in FOLDS_TUNE]
        if all(v is not None for v in vals):
            fb.append((nl, nh, sum(vals) / len(vals)))
for nl, nh, avg in sorted(fb, key=lambda x: x[2]):
    print(f"  layers={nl} hidden={nh:>3}  avg_val_NLL={avg:.4f}")
if fb:
    nl, nh, avg = min(fb, key=lambda x: x[2])
    print(f"  → best flow = layers={nl} hidden={nh} (avg_val_NLL={avg:.4f})")

# [1b-on-MLP] MLP 본모형용 flow 재튜닝 (run_mlp_main_rawvol.py)
print("\n[Phase 1b-on-MLP] MLP 압축기 위 flow 재튜닝 — 3-fold avg VAL NLL (Table 3.7 MLP flow)")
fbm = []
for nl in P1B_FLOW_LAYERS:
    for nh in P1B_FLOW_HIDDEN:
        tag = f"rvP1bMlp_fl{nl}_fh{nh}"
        vals = [read_val(tag, fold) for fold in FOLDS_TUNE]
        if all(v is not None for v in vals):
            fbm.append((nl, nh, sum(vals) / len(vals)))
if not fbm:
    print("  (rvP1bMlp 결과 없음 — run_mlp_main_rawvol.py 먼저)")
else:
    for nl, nh, avg in sorted(fbm, key=lambda x: x[2]):
        print(f"  layers={nl} hidden={nh:>3}  avg_val_NLL={avg:.4f}")
    nl, nh, avg = min(fbm, key=lambda x: x[2])
    print(f"  → MLP best flow = layers={nl} hidden={nh} (avg_val_NLL={avg:.4f})")


# ════════════════════════════════════════════════════════════════════
# Table 4.7 — encoder ablation : 5 seed × 3 fold pooled (per encoder)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 120)
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
print("[Table 4.1] 본모형 — fold별 5-seed mean±std (본모형 tag별 분리)")
print("=" * 120)
# rvP2main_ (구 LSTM) + rvP2mainMlp_ (신 MLP) 모두 매칭, tag-stem 별 분리 집계
main_files = glob.glob(os.path.join(RESULT_DIR, "garch_flow_ar_rvP2main*_summary.json"))
if not main_files:
    print("  (rvP2main* summary 없음)")
else:
    def stem(fn):
        return re.sub(r"_(F_gfc|F_long_A|F_long_B_origin|F_long)_summary\.json$", "",
                      os.path.basename(fn))
    stem_to_files = defaultdict(list)
    for f in main_files:
        stem_to_files[stem(f)].append(f)
    # seed 접미사 제거한 모형 그룹으로 다시 묶기 (rvP2main..._s2026.. → 모형 단위)
    model_to_files = defaultdict(list)
    for st, fs in stem_to_files.items():
        model = re.sub(r"_s\d+$", "", st)
        model_to_files[model].extend(fs)
    for model in sorted(model_to_files):
        files = model_to_files[model]
        kind = "MLP(신, 채택)" if "Mlp" in model else "LSTM(구, val선택)"
        print(f"\n  ── 본모형: {model}  [{kind}] ──")
        hdr2 = f"{'fold':<18}" + "".join(f"{lbl:>20}" for _, lbl, _ in METRICS)
        print(hdr2)
        print("-" * len(hdr2))
        for fold in FOLDS_MAIN:
            ff = [f for f in files if fold_of(os.path.basename(f)) == fold]
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
