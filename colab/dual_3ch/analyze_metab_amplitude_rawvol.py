"""§4.2.3 — metab(유동성) 경로 *진폭 증폭* 민감도 (리만급 k=1 → k=2,3).

질문: metab 경로의 순서효과(↑ vs ↓)·꼬리효과가 진폭을 키우면(k배) 살아나는가?
  · k=1 (리만급, step std(z)≈1.16) = pathshape 의 metab-shape 표가 곧 그 값 → 여기선 생략(중복 회피).
  · k=2,3,5 : 학습 분포(실제 13주 경로 std p95≈0.82) 밖 → **OOD 외삽** (해석 주의).
    k=5 는 step std(z)≈5.8 ≈ 실제 COVID 진폭(diag_metab_crisis_dispersion 의 ~5배).

배경(왜 이 실험):
  · 금리 경로 순서효과는 중심(UWmean)·꼬리(UWcvar1) 모두 견고(하강이 깊음, 4/4).
  · metab 순서효과는 리만급(k=1)에선 중심만 약하게, 꼬리엔 도달 못 함.
  → 진폭을 키우면 metab 효과가 꼬리까지 가는지 확인 (diag: 실제 COVID = 우리 step 의 ~5배).

설계: 금리 = **저금리(p10) flat 고정** (metab 가 무는 영역), metab 만 shape×k 주입.  재학습 0, inference 만.
  shape = flat / ramp_up / ramp_down / step_up / step_down.
  지표: UWmean(중심) + UWcvar1(꼬리) — 순서효과(↑ vs ↓)가 꼬리까지 가는지.
  paired: (fold,seed) 안에서 같은 torch.manual_seed → 칸/모양 간 차이 = 순수 효과.
  재진입: (fold,seed) 결과 json 캐시 → 중단돼도 이어받기.  4 fold × 3 seed.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/analyze_metab_amplitude_rawvol.py
"""
import json
import os
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_pathshape_rawvol as PS   # noqa: E402  (import 시 patch_rawvol() 적용됨)

FUTURE_LEN = PS.FUTURE_LEN
FOLDS = PS.FOLDS
SEEDS = PS.SEEDS
SHAPES = PS.SHAPES                       # zero-mean, range(max-min)=1 정규화 편차

K_LIST = [2, 3, 5]                       # 진폭 배수 (리만급 k=1 은 pathshape 참조; k=5 ≈ COVID)
TEST_SHAPES = ["ramp_up", "ramp_down", "step_up", "step_down"]
CACHE_DIR = os.path.join(PS.RESULT_DIR, "metab_amplitude_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def run_fold_seed(fold, seed, device):
    """한 (fold,seed): flat baseline + (k × shape) metab 단독 주입 metric dict."""
    ctx = PS.load_fold_seed(fold, seed, device)
    if ctx is None:
        return None
    tb_lo = ctx["tb_z"][10]                       # ★ 금리 = 저금리(p10) flat — metab 가 무는 영역
    mb_c = ctx["mb_z"][50]                        # metab 중심 (p50), 여기서 ±k 진폭 swing
    mb_rng = ctx["mb_z"][90] - ctx["mb_z"][10]    # 진폭 기준 (p90-p10, z 단위)
    tb_flat = np.full(FUTURE_LEN, tb_lo)

    def _run(metab_path):
        torch.manual_seed(seed)                   # paired
        sim = PS.rollout_path(ctx["model"], ctx["Xte"], ctx["extra"], ctx["last_sp"],
                              tb_flat, metab_path, device)
        return PS.sim_metrics(sim, ctx["rescale"])

    res = {"mb_rng": float(mb_rng)}
    res["flat"] = _run(np.full(FUTURE_LEN, mb_c))           # k 무관 (편차 0)
    for k in K_LIST:
        kk = {}
        for name in TEST_SHAPES:
            kk[name] = _run(mb_c + SHAPES[name] * mb_rng * k)
        res[f"k{k}"] = kk
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print(f"# §4.2.3 metab 진폭 증폭 민감도 — k={K_LIST} (k=5≈COVID, 리만급 k=1 은 pathshape), device={device}")
    print(f"# 금리 = 저금리(p10) 고정, metab 만 shape×k 주입, 재학습 0.  지표: UWmean(중심) + UWcvar1(꼬리).")
    print("#" * 100)

    for fold in FOLDS:
        for seed in SEEDS:
            cache = os.path.join(CACHE_DIR, f"{fold}_s{seed}.json")
            if os.path.exists(cache):
                print(f"[skip] {fold} s{seed}")
                continue
            r = run_fold_seed(fold, seed, device)
            if r is None:
                continue
            json.dump(r, open(cache, "w"), indent=2)
            print(f"[done] {fold} s{seed}")

    print("\n" + "=" * 100)
    print("[집계] 4 fold × seed 평균±std")
    _summarize()


def _agg(vals):
    a = np.asarray([v for v in vals if v is not None], float)
    return (float(a.mean()), float(a.std(ddof=0))) if len(a) else (float("nan"), float("nan"))


def _summarize():
    for fold in FOLDS:
        caches = [os.path.join(CACHE_DIR, f"{fold}_s{s}.json") for s in SEEDS]
        loaded = [json.load(open(c)) for c in caches if os.path.exists(c)]
        if not loaded:
            print(f"\n=== {fold}: (결과 없음)"); continue
        mb_rng = float(np.mean([d["mb_rng"] for d in loaded]))
        print(f"\n=== {fold}  ({len(loaded)} seed, metab p90-p10={mb_rng:.2f} z) " + "=" * 36)

        fm = _agg([d["flat"]["uw_mean"] for d in loaded])
        fc = _agg([d["flat"]["uw_cvar1"] for d in loaded])
        print(f"  flat 기준선: UWmean={fm[0]:+.5f}±{fm[1]:.5f}   UWcvar1={fc[0]:+.5f}±{fc[1]:.5f}")
        print(f"  {'k':>3} {'stepσz':>7}  {'shape':>10} {'UWmean(m±sd)':>18} {'UWcvar1(m±sd)':>18}")
        for k in K_LIST:
            step_sigma_z = PS.SHAPE_STD["step_up"] * mb_rng * k     # 주입 metab 경로 std(z)
            for name in TEST_SHAPES:
                um = _agg([d[f"k{k}"][name]["uw_mean"] for d in loaded])
                uc = _agg([d[f"k{k}"][name]["uw_cvar1"] for d in loaded])
                tag = f"{step_sigma_z:.2f}" if name == "step_up" else ""
                print(f"  {k:>3} {tag:>7}  {name:>10} {um[0]:+.5f}±{um[1]:.5f} {uc[0]:+.5f}±{uc[1]:.5f}")
            # 순서효과 Δ = (상승 − 하강).  음수 = 상승이 더 깊음.
            for fam in ("ramp", "step"):
                um_u = _agg([d[f"k{k}"][f"{fam}_up"]["uw_mean"] for d in loaded])[0]
                um_d = _agg([d[f"k{k}"][f"{fam}_down"]["uw_mean"] for d in loaded])[0]
                uc_u = _agg([d[f"k{k}"][f"{fam}_up"]["uw_cvar1"] for d in loaded])[0]
                uc_d = _agg([d[f"k{k}"][f"{fam}_down"]["uw_cvar1"] for d in loaded])[0]
                print(f"      └ 순서Δ({fam}↑−↓):  UWmean={um_u - um_d:+.5f}   UWcvar1={uc_u - uc_d:+.5f}")

    print("\n[읽는 법] (1) flat→step UWmean·UWcvar1 깊어짐 = 진폭 반응(분산).")
    print("          (2) 순서Δ(↑−↓) 가 k 커질수록 |커지고| 특히 UWcvar1 에서 일관 부호 → metab 순서효과가 꼬리까지 산다.")
    print("          (3) 순서Δ 가 k 무관하게 0/노이즈면 → metab 은 진폭 키워도 순서효과 없음(분산만).")
    print("  ※ k=1=리만급(step σz≈1.16, pathshape metab-shape 표 참조, in-support 경계).")
    print("     k=2·3·5 는 실제 13주 경로 std p95(0.82) 밖 OOD 외삽(k=5 step σz≈5.8 ≈ COVID).  in-support(k=1) 와 분리 해석.")


if __name__ == "__main__":
    main()
