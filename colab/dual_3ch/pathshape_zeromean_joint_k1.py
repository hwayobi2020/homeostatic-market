"""§4 경로(PATH) — *조인트(JOINT)* 시나리오, zero-MEAN, k=1 only.

두 축(금리·유동성)을 동시에 움직이는 정책조합 시나리오.
  · 금리(tbill) = step (FOMC 이산 결정 성격) · 유동성(metab) = ramp (점진 흐름).
  · 두 shape 모두 zero-mean → 각 축 평균=앵커, 분산 고정(tbill σ≈0.50 / metab σ≈0.31).
  · 4 시나리오는 방향 조합만 다르고 축별 평균·분산은 전부 동일 → 순수 "정책방향 조합" 효과.

  k=1 only (현실 진폭: 금리 ~4.75%p, 유동성 ~4.5%p; k=3 OOD 제외).

4 시나리오 (tbill_shape, metab_shape):
  · 완화/위기대응 = step_down(금리↓) + ramp_up(유동성↑)
  · 긴축        = step_up(금리↑)   + ramp_down(유동성↓)
  · 발산A       = step_down(금리↓) + ramp_down(유동성↓)
  · 발산B       = step_up(금리↑)   + ramp_up(유동성↑)

지표: skew / uw_mean / uw_cvar1(IHL 1%).  seed 평균(paired).
기존 zero-mean 모듈(pathshape_zeromean_anchored_rawvol)의 SHAPES·rollout_paths 재사용.
캐시: result/pathshape_zeromean_joint_k1_cache/.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/pathshape_zeromean_joint_k1.py
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

CACHE_DIR = os.path.join(PS.RESULT_DIR, "pathshape_zeromean_joint_k1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

FOLDS = PS.FOLDS
LABELS = {"F_gfc": "금융위기", "F_long_A": "회복기",
          "F_long_B_origin": "코로나", "F_long": "긴축기"}
SEEDS = PS.SEEDS
FUTURE_LEN = PS.FUTURE_LEN
K = 1                                              # 현실 진폭(in-support)만

# (tbill_shape, metab_shape) — 금리=step, 유동성=ramp
SCEN = {
    "완화/위기대응(금리↓+유동성↑)": ("step_down", "ramp_up"),
    "긴축(금리↑+유동성↓)":          ("step_up",   "ramp_down"),
    "발산A(금리↓+유동성↓)":          ("step_down", "ramp_down"),
    "발산B(금리↑+유동성↑)":          ("step_up",   "ramp_up"),
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

    def metr(sim):
        m = PS.sim_metrics(sim, rescale)
        return {"skew": m["skew"], "uw_mean": m["uw_mean"], "uw_cvar1": m["uw_cvar1"]}

    res = {"flat": metr(ZM.rollout_paths(ctx, tb_flat, mb_flat, seed, device)), "joint": {}}
    for name, (tb_sh, mb_sh) in SCEN.items():
        tb_path = anc_tb + ZM.SHAPES[tb_sh][None, :] * (K * range_tb)
        mb_path = anc_mb + ZM.SHAPES[mb_sh][None, :] * (K * range_mb)
        res["joint"][name] = metr(ZM.rollout_paths(ctx, tb_path, mb_path, seed, device))
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 114)
    print(f"# §4 PATH JOINT (zero-mean, k=1 현실진폭) — LOCKED {PS.TAG_PREFIX}, device={device}, n_sim={ZM.N_SIM}")
    print(f"#   금리=step / 유동성=ramp, 두 축 동시 주입.  4 시나리오 축별 평균·분산 동일(방향조합만 차이)")
    print(f"#   shape σ: step={ZM.SHAPE_STD['step_up']:.2f} / ramp={ZM.SHAPE_STD['ramp_up']:.2f}")
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
    print("[집계] fold × seed 평균  (값 = skew / uw_mean / uw_cvar1).  flat=양축 마지막값 지속 baseline")
    for fold in FOLDS:
        label = LABELS.get(fold, fold)
        paths = [os.path.join(CACHE_DIR, f"{fold}_s{s}.json") for s in SEEDS]
        loaded = [json.load(open(p)) for p in paths if os.path.exists(p)]
        if not loaded:
            print(f"\n=== {label} [{fold}]: (결과 없음)"); continue
        print(f"\n=== {label} [{fold}]  ({len(loaded)} seed) " + "=" * 50)
        fsk = _agg([d["flat"]["skew"] for d in loaded])
        fum = _agg([d["flat"]["uw_mean"] for d in loaded])
        fuc = _agg([d["flat"]["uw_cvar1"] for d in loaded])
        print(f"  flat baseline:  skew={fsk:+.3f}  uw_mean={fum:+.4f}  uw_cvar1={fuc:+.4f}")
        print(f"    {'시나리오':<26} {'skew':>9} {'uw_mean':>10} {'uw_cvar1':>10}")
        for name in SCEN:
            sk = _agg([d["joint"][name]["skew"] for d in loaded])
            um = _agg([d["joint"][name]["uw_mean"] for d in loaded])
            uc = _agg([d["joint"][name]["uw_cvar1"] for d in loaded])
            print(f"    {name:<26} {sk:>+9.3f} {um:>+10.4f} {uc:>+10.4f}")
        ea = _agg([d["joint"]["완화/위기대응(금리↓+유동성↑)"]["uw_cvar1"] for d in loaded])
        ti = _agg([d["joint"]["긴축(금리↑+유동성↓)"]["uw_cvar1"] for d in loaded])
        deeper = "완화/위기대응" if ea < ti else "긴축"
        print(f"    [완화 vs 긴축] 완화={ea:+.4f} vs 긴축={ti:+.4f} → {deeper} 더 깊음")
    print("\n[읽는 법] 4 시나리오 축별 평균·분산 동일 → 차이 = 순수 정책방향 조합 효과.")
    print("  완화/위기대응(금리↓+유동성↑) vs 긴축(금리↑+유동성↓)이 핵심 대비.  k=1=현실진폭.")


if __name__ == "__main__":
    main()
