"""§4 조합(JOINT) — 금리=이어지는 step_up(zero-start, in-dist) × 유동성=평균분산 일치(zero-mean), k=1.

각 축에 맞는 스킴(stock/flow 분리):
  · 금리(tbill, floored level): zero-START step_up — anchor(실현 마지막값)에서 위로만 상승.
    저금리(0%)서도 0→+ 라 음수 없음 = in-distribution, OOD 없음.
  · 유동성(metab, flow): zero-MEAN ramp/step up·down — 평균(0)·분산 일치(moment-matched).

조합: 현실적 금리 인상(step_up) 배경에서, 평균분산 통제된 유동성 경로(팽창 vs 수축)가
      보유기간 손실(IHL)을 어떻게 바꾸나.  k=1(현실 진폭).

지표: skew / uw_mean / uw_cvar10(IHL 10%).  seed 평균(paired).
분해용: flat(양축 지속) + rate_only(금리만 step↑) + joint(금리step↑ × 유동성 4종).
기존 zero-mean 모듈(SHAPES·rollout_paths) 재사용.  캐시: result/pathshape_joint_ratestep_metabmm_k1_cache/.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/pathshape_joint_ratestep_metabmm_k1.py
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

import analyze_pathshape_rawvol as PS                                   # noqa: E402
import pathshape_zeromean_anchored_rawvol as ZM                         # noqa: E402

CACHE_DIR = os.path.join(PS.RESULT_DIR, f"pathshape_joint_ratestep_metabmm_k1_cache{PS.CACHE_SUFFIX}")
os.makedirs(CACHE_DIR, exist_ok=True)

FOLDS = PS.FOLDS
LABELS = {"F_gfc": "금융위기", "F_long_A": "회복기",
          "F_long_B_origin": "코로나", "F_long": "긴축기"}
SEEDS = PS.SEEDS
FUTURE_LEN = PS.FUTURE_LEN
K = 1                                              # 현실 진폭(in-support)만

# 금리: zero-START step_up (anchor→anchor+진폭, 상승만 → in-distribution)
_idx = np.arange(FUTURE_LEN)
_half = FUTURE_LEN // 2
RATE_STEPUP = (_idx >= _half).astype(float)        # [0..0, 1..1]  (상승 계단, 음수 없음)

# 유동성: zero-MEAN 평균분산 일치 (ZM.SHAPES 재사용)
METAB_SHAPES = {
    "유동성 ramp↑": ZM.SHAPES["ramp_up"],
    "유동성 ramp↓": ZM.SHAPES["ramp_down"],
    "유동성 step↑": ZM.SHAPES["step_up"],
    "유동성 step↓": ZM.SHAPES["step_down"],
}


def compute_fold_seed(fold, seed, device):
    ctx = PS.load_fold_seed(fold, seed, device)
    if ctx is None:
        return None
    rescale = ctx["rescale"]
    range_tb = ctx["tb_z"][90] - ctx["tb_z"][10]
    range_mb = ctx["mb_z"][90] - ctx["mb_z"][10]
    anc_tb = ctx["real_tb_fut"][:, 0:1]
    anc_mb = ctx["real_mb_fut"][:, 0:1]
    tb_flat = anc_tb + np.zeros((1, FUTURE_LEN))
    mb_flat = anc_mb + np.zeros((1, FUTURE_LEN))
    tb_stepup = anc_tb + RATE_STEPUP[None, :] * (K * range_tb)          # 금리 상승(고정 배경)

    def metr(sim):
        m = PS.sim_metrics(sim, rescale)
        return {"skew": m["skew"], "uw_mean": m["uw_mean"], "uw_cvar10": m["uw_cvar10"]}

    res = {}
    res["flat"] = metr(ZM.rollout_paths(ctx, tb_flat, mb_flat, seed, device))         # 양축 지속
    res["rate_only"] = metr(ZM.rollout_paths(ctx, tb_stepup, mb_flat, seed, device))  # 금리만 step↑
    res["joint"] = {}
    for name, mb_shape in METAB_SHAPES.items():
        mb_path = anc_mb + mb_shape[None, :] * (K * range_mb)
        res["joint"][name] = metr(ZM.rollout_paths(ctx, tb_stepup, mb_path, seed, device))
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 114)
    print(f"# §4 JOINT (금리 zero-start step↑ × 유동성 zero-mean 평균분산일치) — LOCKED {PS.TAG_PREFIX}, device={device}")
    print(f"#   금리=상승 step(음수 없음, in-dist) 고정 배경, 유동성=평균분산 일치 4종.  k=1.  지표 skew/uw_mean/uw_cvar10")
    print("#" * 114)

    for fold in FOLDS:
        for seed in SEEDS:
            cache = os.path.join(CACHE_DIR, f"{fold}_s{seed}.json")
            if os.path.exists(cache):
                print(f"[skip] {fold} s{seed}")
                continue
            r = compute_fold_seed(fold, seed, device)
            if r is None:
                continue
            json.dump(r, open(cache, "w"), indent=2)
            print(f"[done] {fold} s{seed}")

    _summarize()


def _agg(vals):
    a = np.asarray([v for v in vals if v is not None], float)
    return float(a.mean()) if len(a) else float("nan")


def _summarize():
    print("\n" + "=" * 114)
    print("[집계] fold × seed 평균  (값 = skew / uw_mean / uw_cvar10)")
    for fold in FOLDS:
        label = LABELS.get(fold, fold)
        paths = [os.path.join(CACHE_DIR, f"{fold}_s{s}.json") for s in SEEDS]
        loaded = [json.load(open(p)) for p in paths if os.path.exists(p)]
        if not loaded:
            print(f"\n=== {label} [{fold}]: (결과 없음)"); continue
        print(f"\n=== {label} [{fold}]  ({len(loaded)} seed) " + "=" * 50)

        for tag, key in (("flat (양축 지속)", "flat"), ("rate_only (금리 step↑ 단독)", "rate_only")):
            sk = _agg([d[key]["skew"] for d in loaded])
            um = _agg([d[key]["uw_mean"] for d in loaded])
            uc = _agg([d[key]["uw_cvar10"] for d in loaded])
            print(f"  {tag:<26} skew={sk:+.3f}  uw_mean={um:+.4f}  uw_cvar10={uc:+.4f}")
        print(f"  [금리 step↑ + 유동성 (평균분산 일치)]")
        print(f"    {'시나리오':<16} {'skew':>9} {'uw_mean':>10} {'uw_cvar10':>10}")
        for name in METAB_SHAPES:
            sk = _agg([d["joint"][name]["skew"] for d in loaded])
            um = _agg([d["joint"][name]["uw_mean"] for d in loaded])
            uc = _agg([d["joint"][name]["uw_cvar10"] for d in loaded])
            print(f"    {name:<16} {sk:>+9.3f} {um:>+10.4f} {uc:>+10.4f}")
        # 유동성 방향 효과 (금리 step↑ 배경): up vs down
        for a, b, tag in (("유동성 ramp↑", "유동성 ramp↓", "ramp ↑vs↓"),
                          ("유동성 step↑", "유동성 step↓", "step ↑vs↓")):
            ua = _agg([d["joint"][a]["uw_cvar10"] for d in loaded])
            ub = _agg([d["joint"][b]["uw_cvar10"] for d in loaded])
            deeper = a if ua < ub else b
            print(f"    [{tag}, uw_cvar10] {a}={ua:+.4f} vs {b}={ub:+.4f} → {deeper} 더 깊음")
    print("\n[읽는 법] 금리 step↑(상승, in-dist)는 고정 배경. 유동성은 평균분산 일치(zero-mean).")
    print("  rate_only 대비 joint = 금리인상 배경에 유동성 경로가 더하는 효과. up vs down = 유동성 방향 효과.")


if __name__ == "__main__":
    main()
