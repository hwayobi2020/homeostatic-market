"""§4 결과 전체를 목차(TOC) 순서로 한 번에 출력하는 집계 스크립트.

배경
----
§4 실행 스크립트들이 제각각 (run-order / 일부는 metric 미출력으로) 흩어져 찍혀서
결과가 목차 순서로 정리되지 않는다.  특히 재학습 계열(4.1.1 GARCH / 4.1.2 GAN·VAE /
4.3.1·4.3.3 ablation)은 *summary JSON 에만 저장*하고 표를 안 찍는다.

이 스크립트는 **학습/torch import 0** — 이미 저장된 summary JSON + 시나리오 캐시만 읽어
§4.1.1 → 4.3.5 순서로 표를 출력한다.  (nflows/mamba 등 deps 불필요, 즉시 실행.)

읽는 파일 (전부 result/ 아래):
  4.1.1  garch_x_{fold}_summary.json                              (GARCH(1,1)-X-t)
  4.1.2  {vae,gan}_baseline_{fold}_summary.json                   (Cond VAE/GAN)
  4.1.x  garch_flow_ar_rvP2mainMlp_pd64_fl4_fh128_s{seed}_{fold}_summary.json  (본모형, 비교기준)
  4.2.1  pathshape_full_cache/{fold}_s{seed}.json   (LEVEL 9-grid)
  4.2.2  pathshape_full_cache/{fold}_s{seed}.json   (SHAPE + JOINT)
  4.2.3  debasement_cache_v4/{fold}_s{seed}.json    (저금리 metab 레벨/경로/진폭)
  4.3.1  garch_flow_ar_rvAbl_{metab_drop,maskall,fedrate}_s{seed}_{fold}_summary.json
  4.3.3  garch_flow_ar_rvAbl_{win26,win52}_s{seed}_{fold}_summary.json
  4.3.4  garch_flow_ar_rvP2abl_{enc}_*_{fold}_summary.json        (encoder ablation)
  4.3.2  feature importance / 4.3.5 per-τ 첨도 → 각 진단 스크립트가 자체 출력 (포인터만)

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !python colab/dual_3ch/agg_section4.py
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "result")

FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
FOLDS_TUNE = ["F_long_A", "F_long_B_origin", "F_long"]      # encoder ablation 은 3 fold

MAIN_TAG = "rvP2mainMlp_pd64_fl4_fh128"                     # 본모형 (LOCKED)
ENCODERS = ["mlp", "lstm", "transformer", "mamba"]

PATHSHAPE_CACHE = os.path.join(RESULT_DIR, "pathshape_full_cache")
DEBASE_CACHE = os.path.join(RESULT_DIR, "debasement_cache_v4")

PCTL_LABEL = {10: "lo", 50: "mid", 90: "hi"}
PCTLS = [10, 50, 90]
SHAPE_ORDER = ["flat", "ramp_up", "ramp_down", "hump", "trough", "step_up", "step_down"]
JOINT_ORDER = ["easing", "tightening", "crisis"]
METAB_LEVELS = [10, 30, 50, 70, 90]
METAB_SHAPES = ["flat", "ramp_up", "ramp_down", "step_up", "step_down", "hump", "trough"]
AMP_MULTS = [1.0, 3.0, 5.0]
AMP_PAIRS = [("ramp_up", "ramp_down"), ("step_up", "step_down"), ("hump", "trough")]

# 생성형 모델 비교(4.1) — baseline(garch_x / vae / gan)과 본모형이 *공통으로 보유*한 key.
#   주의: GARCH/VAE/GAN summary 에는 per_week_nll_z 없음(우도 정의 다름), CVaR 는 diff 로만 저장 →
#         4.1 표에서는 NLL·절대 CVaR 제외, 공통 비교 가능한 지표만.
GEN_METRICS = [
    ("crps_pooled", "CRPS",   "낮을수록"),
    ("emd",         "EMD",    "낮을수록"),
    ("skew_sim",    "skew_s", "좌측(−)"),
    ("exkurt_sim",  "exk_s",  "act근접"),
    ("coverage_80", "cov80",  "0.80"),
    ("coverage_95", "cov95",  "0.95"),
    ("std_ratio",   "std비",  "1.0"),
]
# 본모형/ablation 풀 지표(4.3.1/4.3.3/4.3.4) — 모두 garch_flow(main_worker) 산출이라 NLL·CVaR 절대 보유.
FULL_METRICS = [
    ("per_week_nll_z", "NLL/wk", "낮을수록"),
    ("crps_pooled",    "CRPS",   "낮을수록"),
    ("emd",            "EMD",    "낮을수록"),
    ("cvar_1pct_sim",  "CVaR1%", "깊을수록(−)"),
    ("skew_sim",       "skew",   "좌측(−)"),
    ("exkurt_sim",     "exk_s",  "act근접"),
    ("coverage_80",    "cov80",  "0.80"),
    ("coverage_95",    "cov95",  "0.95"),
    ("std_ratio",      "std비",  "1.0"),
]


# ───────────────────────── 공통 유틸 ─────────────────────────
def load_te(path):
    try:
        return json.load(open(path)).get("test_eval", {})
    except (json.JSONDecodeError, OSError):
        return {}


def fold_of(fname):
    for fold in FOLDS:
        if fname.endswith(f"_{fold}_summary.json"):
            return fold
    return None


def ms(vals):
    a = np.asarray([v for v in vals if isinstance(v, (int, float))], dtype=float)
    if len(a) == 0:
        return None, None, 0
    return float(a.mean()), float(a.std(ddof=0)), len(a)


def fmt(mn, sd, n):
    return f"{mn:+.4f}±{sd:.4f}(n{n})" if mn is not None else "      NA      "


def collect_summaries(pattern, folds=FOLDS):
    """result/{pattern} glob → {fold: [test_eval dict, ...]}."""
    out = defaultdict(list)
    for f in glob.glob(os.path.join(RESULT_DIR, pattern)):
        fold = fold_of(os.path.basename(f))
        if fold in folds:
            out[fold].append(load_te(f))
    return out


def print_metric_table(model_rows, metrics, folds=FOLDS):
    """model_rows = [(label, {fold:[te,...]}), ...] → fold별 모델 행 표."""
    hdr = f"{'fold / model':<26}" + "".join(f"{lbl:>16}" for _, lbl, _ in metrics)
    print(hdr)
    print("-" * len(hdr))
    for fold in folds:
        for i, (label, fmap) in enumerate(model_rows):
            tes = fmap.get(fold, [])
            tag = f"{fold:<16}{label:<10}" if i == 0 else f"{'':<16}{label:<10}"
            row = f"{tag:<26}"
            for key, _, _ in metrics:
                mn, sd, n = ms([te.get(key) for te in tes])
                row += f"{fmt(mn, sd, n):>16}"
            print(row)
        print()


def load_scen_caches(cache_dir):
    """{fold: [res dict, ...]}  (시나리오 캐시 — seed 별)."""
    out = defaultdict(list)
    if not os.path.isdir(cache_dir):
        return out
    for fold in FOLDS:
        for f in sorted(glob.glob(os.path.join(cache_dir, f"{fold}_s*.json"))):
            try:
                out[fold].append(json.load(open(f)))
            except (json.JSONDecodeError, OSError):
                pass
    return out


def agg(vals):
    a = np.asarray([v for v in vals if v is not None], float)
    return (float(a.mean()), float(a.std(ddof=0))) if len(a) else (float("nan"), float("nan"))


def _build_shape_std():
    """pathshape build_shapes 와 동일 — [B] 경로 σ 표시용 (range-norm, zero-mean)."""
    FL = 13
    t = np.arange(FL, dtype=float)
    mid = (FL - 1) / 2.0

    def _norm(d):
        d = np.asarray(d, float) - np.mean(d)
        rng = d.max() - d.min()
        return d / rng if rng > 1e-12 else d

    ramp = _norm(np.linspace(0.0, 1.0, FL))
    hump = _norm(1.0 - np.abs(t - mid) / mid)
    step = _norm(np.where(t < mid, 0.0, 1.0))
    sh = {"flat": np.zeros(FL), "ramp_up": ramp, "ramp_down": -ramp,
          "hump": hump, "trough": -hump, "step_up": step, "step_down": -step}
    return {k: float(np.std(v)) for k, v in sh.items()}


SHAPE_STD = _build_shape_std()


# ════════════════════════════════════════════════════════════════════
def sec_4_1_1():
    print("\n" + "#" * 110)
    print("# §4.1.1  정보 일치 비교 — MAC-Flow(maskall)  vs  GARCH-X(past)-skewt")
    print("#  둘 다 과거 거시 사용 + 미래 거시 경로 배제.  GARCH 엔 skew-t(공정).")
    print("#  → 동일 정보·미래경로 미사용 조건에서 생성 head(flow vs GARCH) 차이.  미래경로 기여는 §4.3.1.")
    print("#" * 110)
    maskall = collect_summaries("garch_flow_ar_rvAbl_maskall_*_summary.json")
    garch = collect_summaries("garch_xpast_*_summary.json")
    if not garch:
        print("  [GARCH-X(past) 결과 없음] → train_garch_xpast.py 선행 필요.")
    if not any(maskall.values()):
        print("  [MAC-Flow(maskall) 결과 없음] → run_ablations_rawvol.py(maskall) 필요.")
    print_metric_table([("MAC-Flow(maskall)", maskall), ("GARCH-ST(past)", garch)], GEN_METRICS)


def sec_4_1_2():
    print("\n" + "#" * 110)
    print("# §4.1.2  path 없는 level-기반 생성모델과의 비교 — Cond VAE / Cond GAN  vs  MAC-Flow")
    print("#" * 110)
    main = collect_summaries(f"garch_flow_ar_{MAIN_TAG}_*_summary.json")
    vae = collect_summaries("vae_baseline_*_summary.json")
    gan = collect_summaries("gan_baseline_*_summary.json")
    if not vae and not gan:
        print("  [VAE/GAN 결과 없음] → run_vae_gan_all.py 선행 필요.")
    print_metric_table([("MAC-Flow", main), ("VAE", vae), ("GAN", gan)], GEN_METRICS)


def sec_4_2_1():
    print("\n" + "#" * 110)
    print("# §4.2.1  반사실 시나리오 맵 — LEVEL 9-grid (tbill × metab, p10/50/90 flat)")
    print("#  각 칸 = skew / UWcvar1(intra-horizon 1% CVaR, 깊을수록 −)   (seed 평균)")
    print("#" * 110)
    caches = load_scen_caches(PATHSHAPE_CACHE)
    if not any(caches.values()):
        print("  [pathshape 캐시 없음] → analyze_pathshape_rawvol.py 선행 필요."); return
    for fold in FOLDS:
        loaded = caches.get(fold, [])
        if not loaded:
            print(f"\n=== {fold}: (결과 없음)"); continue
        print(f"\n=== {fold}  ({len(loaded)} seed) " + "=" * 50)
        for pt in PCTLS:
            cells = []
            for pm in PCTLS:
                key = f"{PCTL_LABEL[pt]}_{PCTL_LABEL[pm]}"
                sk = agg([d["level"][key]["skew"] for d in loaded])[0]
                uw = agg([d["level"][key]["uw_cvar1"] for d in loaded])[0]
                cells.append(f"metab={PCTL_LABEL[pm]:>3}: {sk:+.2f}/{uw:+.4f}")
            print(f"    tbill={PCTL_LABEL[pt]:>3}   " + "   ".join(cells))


def sec_4_2_2():
    print("\n" + "#" * 110)
    print("# §4.2.2  path 시나리오 검정 — SHAPE(평균 고정, 모양만) + JOINT(동시 경로)")
    print("#  순서효과 = 같은 pathσ 의 up vs down 의 UWcvar1 차이.  skew 는 순서 무감각(분산효과만).")
    print("#" * 110)
    caches = load_scen_caches(PATHSHAPE_CACHE)
    if not any(caches.values()):
        print("  [pathshape 캐시 없음] → analyze_pathshape_rawvol.py 선행 필요."); return
    for fold in FOLDS:
        loaded = caches.get(fold, [])
        if not loaded:
            print(f"\n=== {fold}: (결과 없음)"); continue
        print(f"\n=== {fold}  ({len(loaded)} seed) " + "=" * 50)
        for var, label in [("shape_tbill", "tbill-shape (metab flat)"),
                           ("shape_metab", "metab-shape (tbill flat)")]:
            print(f"  [SHAPE: {label}]  {'shape':>10}{'pathσ':>7}{'skew':>9}{'UWcvar1':>12}{'term':>11}")
            for name in SHAPE_ORDER:
                sk = agg([d[var][name]["skew"] for d in loaded])[0]
                uc = agg([d[var][name]["uw_cvar1"] for d in loaded])[0]
                tm = agg([d[var][name]["term_mean"] for d in loaded])[0]
                print(f"    {name:>20}{SHAPE_STD[name]:>7.3f}{sk:>+9.3f}{uc:>+12.4f}{tm:>+11.4f}")
        print(f"  [JOINT: 동시 경로]  {'scenario':>12}{'skew':>9}{'UWcvar1':>12}{'term':>11}")
        for name in JOINT_ORDER:
            sk = agg([d["joint"][name]["skew"] for d in loaded])[0]
            uc = agg([d["joint"][name]["uw_cvar1"] for d in loaded])[0]
            tm = agg([d["joint"][name]["term_mean"] for d in loaded])[0]
            print(f"    {name:>22}{sk:>+9.3f}{uc:>+12.4f}{tm:>+11.4f}")


def sec_4_2_3():
    print("\n" + "#" * 110)
    print("# §4.2.3  debasement 시나리오 — 저금리(tbill=p10) 고정 하 유동성(metab) 레벨/경로/진폭")
    print("#" * 110)
    caches = load_scen_caches(DEBASE_CACHE)
    if not any(caches.values()):
        print("  [debasement_cache_v4 없음] → analyze_debasement_rawvol.py 선행 필요."); return
    for fold in FOLDS:
        loaded = caches.get(fold, [])
        if not loaded:
            print(f"\n=== {fold}: (결과 없음)"); continue
        print(f"\n=== {fold}  ({len(loaded)} seed) " + "=" * 50)

        print("  [A] metab 레벨 sweep (flat)   metab   skew / cvar1 / UWmean / UWcvar1")
        for pm in METAB_LEVELS:
            k = f"metab_p{pm}"
            sk = agg([d["level_lowrate"][k]["skew"] for d in loaded])[0]
            cv = agg([d["level_lowrate"][k]["cvar1"] for d in loaded])[0]
            uw = agg([d["level_lowrate"][k]["uw_mean"] for d in loaded])[0]
            uc = agg([d["level_lowrate"][k]["uw_cvar1"] for d in loaded])[0]
            print(f"      metab p{pm:<2}   skew {sk:+.3f}   cvar1 {cv:+.4f}   "
                  f"UWmean {uw:+.4f}   UWcvar1 {uc:+.4f}")

        print("  [B] metab 경로 (평균=p50, 진폭=p90−p10)   shape   pathσ / skew / UWcvar1 / term")
        for name in METAB_SHAPES:
            sk = agg([d["dynamic_lowrate"][name]["skew"] for d in loaded])[0]
            uc = agg([d["dynamic_lowrate"][name]["uw_cvar1"] for d in loaded])[0]
            tm = agg([d["dynamic_lowrate"][name]["term_mean"] for d in loaded])[0]
            print(f"      {name:<10} σ{SHAPE_STD[name]:.2f}   skew {sk:+.3f}   "
                  f"UWcvar1 {uc:+.4f}   term {tm:+.4f}")

        # [C] amp_sweep 존재 시 (v4)
        if "amp_sweep" in loaded[0]:
            print("  [C] SHAPE × 진폭 — 진폭별 up vs down UWcvar1(gap)·skew·exkurt")
            for mult in AMP_MULTS:
                psz = agg([d["amp_sweep"][f"step_up_x{mult:.0f}"]["mb_path_std_z"]
                           for d in loaded])[0]
                tag = "≈리만" if mult == 1 else ("≈COVID" if mult == 5 else "")
                print(f"    --- {mult:.0f}× {tag:<6} (step 경로std~{psz:.1f}) ---")
                for a, b in AMP_PAIRS:
                    ua = agg([d["amp_sweep"][f"{a}_x{mult:.0f}"]["uw_cvar1"] for d in loaded])[0]
                    ub = agg([d["amp_sweep"][f"{b}_x{mult:.0f}"]["uw_cvar1"] for d in loaded])[0]
                    ska = agg([d["amp_sweep"][f"{a}_x{mult:.0f}"]["skew"] for d in loaded])[0]
                    ek = agg([d["amp_sweep"][f"{a}_x{mult:.0f}"]["exkurt"] for d in loaded])[0]
                    print(f"      {a:>9}/{b:<9} UWcvar1 {ua:+.4f}/{ub:+.4f} (gap{ub-ua:+.4f})  "
                          f"skew {ska:+.2f}  exk {ek:+.1f}")
        else:
            print("  [C] (amp_sweep 없음 — 구 캐시.  v4 재실행 시 표시)")


def sec_4_3_1():
    print("\n" + "#" * 110)
    print("# §4.3.1  feature ablation — metab_drop / maskall(path-mask) / fedrate  vs  본모형(full)")
    print("#  본모형 대비 NLL·skew 변화로 기여 판단.  maskall = 미래 거시 경로 조건화 제거(=path 효과).")
    print("#" * 110)
    rows = [("full(본모형)", collect_summaries(f"garch_flow_ar_{MAIN_TAG}_*_summary.json"))]
    for name in ["metab_drop", "maskall", "fedrate"]:
        m = collect_summaries(f"garch_flow_ar_rvAbl_{name}_*_summary.json")
        rows.append((name, m))
        if not any(m.values()):
            print(f"  [{name} 결과 없음]")
    print_metric_table(rows, FULL_METRICS)


def sec_4_3_2():
    print("\n" + "#" * 110)
    print("# §4.3.2  feature importance (permutation ΔNLL/week)")
    print("#  → analyze_feature_importance_rawvol.py 가 자체 출력 (이 집계기는 JSON-only 라 별도).")
    print("#  요약: sp_std_13w 가 NLL 지배 / tbill·metab 은 ≈0(모양 축이라 NLL 과소대표).")
    print("#" * 110)


def sec_4_3_3():
    print("\n" + "#" * 110)
    print("# §4.3.3  time-window ablation — target horizon 13w(본모형) vs 26w(win26) vs 52w(win52)")
    print("#  주의: horizon 다르면 NLL/wk(주당)는 비교 가능, CRPS·CVaR 는 스케일 달라 *동일 horizon 끼리만*.")
    print("#" * 110)
    rows = [("13w(본모형)", collect_summaries(f"garch_flow_ar_{MAIN_TAG}_*_summary.json"))]
    for name in ["win26", "win52"]:
        m = collect_summaries(f"garch_flow_ar_rvAbl_{name}_*_summary.json")
        rows.append((name, m))
        if not any(m.values()):
            print(f"  [{name} 결과 없음]")
    print_metric_table(rows, FULL_METRICS)


def sec_4_3_4():
    print("\n" + "#" * 110)
    print("# §4.3.4  Past Encoder ablation — MLP/LSTM/Transformer/Mamba (5 seed × 3 fold pooled)")
    print("#" * 110)
    hdr = f"{'encoder':<12}" + "".join(f"{lbl:>16}" for _, lbl, _ in FULL_METRICS)
    print(hdr)
    print("-" * len(hdr))
    any_data = False
    for enc in ENCODERS:
        files = glob.glob(os.path.join(RESULT_DIR, f"garch_flow_ar_rvP2abl_{enc}_*_summary.json"))
        files = [f for f in files if fold_of(os.path.basename(f)) in FOLDS_TUNE]
        coll = defaultdict(list)
        for f in files:
            te = load_te(f)
            for key, _, _ in FULL_METRICS:
                if isinstance(te.get(key), (int, float)):
                    coll[key].append(te[key])
        row = f"{enc:<12}"
        for key, _, _ in FULL_METRICS:
            mn, sd, n = ms(coll[key])
            if n:
                any_data = True
            row += f"{fmt(mn, sd, n):>16}"
        print(row)
    if not any_data:
        print("  (rvP2abl encoder summary 없음)")


def sec_4_3_5():
    print("\n" + "#" * 110)
    print("# §4.3.5  teacher-forcing gate — per-τ 초과첨도 (sim free-running vs actual)")
    print("#  → diag_perstep_kurtosis_rawvol.py 가 자체 출력 (모델 inference 필요라 이 집계기 밖).")
    print("#  요약: 4 fold 전부 τ1≈τ13 (평평) → exposure bias 아님 → scheduled sampling 무효(스킵).")
    print("#" * 110)


def main():
    print("=" * 110)
    print("§4 RESULTS — 목차(TOC) 순서 집계  (JSON/캐시 only, 학습·torch 0)")
    print("  4.1.1 GARCH | 4.1.2 VAE/GAN | 4.2.1 LEVEL | 4.2.2 SHAPE/JOINT | 4.2.3 debasement")
    print("  4.3.1 feature abl | 4.3.2 importance | 4.3.3 window | 4.3.4 encoder | 4.3.5 per-τ gate")
    print("=" * 110)
    sec_4_1_1()
    sec_4_1_2()
    sec_4_2_1()
    sec_4_2_2()
    sec_4_2_3()
    sec_4_3_1()
    sec_4_3_2()
    sec_4_3_3()
    sec_4_3_4()
    sec_4_3_5()
    print("\n" + "=" * 110)
    print("[안내] 결과 없는 섹션 = 해당 실행 스크립트 미완 (메시지의 '선행 필요' 참조).")
    print("       4.3.2/4.3.5 는 각 진단 스크립트 출력을 그대로 쓰면 됨 (여기선 요약 포인터).")
    print("=" * 110)


if __name__ == "__main__":
    main()
