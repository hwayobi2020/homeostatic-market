"""§4.2.3 화폐 절하(debasement) 시나리오 — 저금리 고정 하 유동성(metab) 상세·동적 분석.

4.2.1(9-grid)이 거시 *수준* interaction을 거칠게(p10/50/90, flat) 보였다면, 여기서는
thesis 핵심 국면인 **저금리(tbill=p10 고정)** 를 확대경으로:

  [A] metab 미세 레벨 sweep (p10·p30·p50·p70·p90, flat) — 저금리에서 유동성 *수준*이
      오를수록 좌꼬리(skew)·손실(CVaR·IHL)이 어떻게 깊어지는가 (4.2.1 저금리열 zoom-in).
  [B] metab 시변 *경로* (flat / ramp_up=유동성 확대 / ramp_down=축소) — 저금리에서
      유동성이 horizon 동안 *팽창*(debasement)하면 꼬리가 어떻게 전개되는가.
      (4.2.1=정적 레벨 vs 여기=동적 경로 → 비중복.  mid-rate에서 약했던 metab-path를
       저금리에서 재검증하는 미결 질문도 해결.)

LOCKED MAC-Flow 본모형, inference only (재학습 0).  pathshape 인프라 재사용.
*상대* 프레이밍: 절대 꼬리수치는 위기 OOD라 §5 한계 — 여기선 "유동성 수준/경로 *간* 차이".

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/analyze_debasement_rawvol.py
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

# pathshape 모듈 import = patch_rawvol() + set_cond_cols(ENC_COLS) + 인프라 함수 일괄 셋업.
# (import 시 main()은 안 돈다 — __main__ 가드.)
import analyze_pathshape_rawvol as PS                                # noqa: E402

# 저금리 sweep을 위해 percentile 격자를 촘촘히 (load_fold_seed 가 모듈 PCTLS 참조).
PS.PCTLS = [10, 30, 50, 70, 90]

FOLDS = PS.FOLDS                       # F_gfc / F_long_A / F_long_B_origin / F_long
SEEDS = PS.SEEDS                       # 2026/27/28
FUTURE_LEN = PS.FUTURE_LEN
CACHE_DIR = os.path.join(PS.RESULT_DIR, "debasement_cache_v2")   # shape 확장 → 새 캐시
os.makedirs(CACHE_DIR, exist_ok=True)

LOW_PCTL = 10                          # tbill 저금리 고정 수준
METAB_LEVELS = [10, 30, 50, 70, 90]    # [A] 미세 레벨 sweep
# [B] 동적: 점진(ramp)·급격(step)·충격왔다감(hump/trough) × 확대(up)/축소(down)
METAB_SHAPES = ["flat", "ramp_up", "ramp_down", "step_up", "step_down", "hump", "trough"]


@torch.no_grad()
def run_fold_seed(fold, seed, device):
    ctx = PS.load_fold_seed(fold, seed, device)
    if ctx is None:
        return None

    tb_lo = ctx["tb_z"][LOW_PCTL]
    tb_flat_lo = np.full(FUTURE_LEN, tb_lo, dtype=float)   # 저금리 13주 고정

    def _run(tbill_path, metab_path):
        torch.manual_seed(seed)                            # paired (시나리오 간 동일 noise)
        sim = PS.rollout_path(ctx["model"], ctx["Xte"], ctx["extra"], ctx["last_sp"],
                              tbill_path, metab_path, device)
        return PS.sim_metrics(sim, ctx["rescale"])

    res = {"level_lowrate": {}, "dynamic_lowrate": {}}

    # [A] 저금리 고정 + metab 미세 레벨(flat)
    for pm in METAB_LEVELS:
        mb_flat = np.full(FUTURE_LEN, ctx["mb_z"][pm], dtype=float)
        res["level_lowrate"][f"metab_p{pm}"] = _run(tb_flat_lo, mb_flat)

    # [B] 저금리 고정 + metab 시변 경로 (평균=p50, 진폭=p90-p10, 모양만)
    mb_c = ctx["mb_z"][50]
    mb_rng = ctx["mb_z"][90] - ctx["mb_z"][10]
    for name in METAB_SHAPES:
        dev = PS.SHAPES[name]                              # flat / ramp_up(↑확대) / ramp_down(↓축소)
        mb_path = mb_c + dev * mb_rng
        res["dynamic_lowrate"][name] = _run(tb_flat_lo, mb_path)

    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print(f"# §4.2.3 Debasement (저금리 고정 + metab 상세/동적) — LOCKED {PS.TAG_PREFIX}, "
          f"device={device}, n_sim={PS.N_SIM}")
    print(f"# tbill=p{LOW_PCTL} 고정 | [A] metab flat sweep {METAB_LEVELS} | [B] metab 경로 {METAB_SHAPES}")
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

    # ── 집계: seed 평균±std ──
    print("\n" + "=" * 100)
    print("[집계] Debasement — 4 fold × 3 seed 평균±std  (저금리 고정)")

    def _agg(vals):
        a = np.asarray([v for v in vals if v is not None], float)
        return (float(a.mean()), float(a.std(ddof=0))) if len(a) else (float("nan"), float("nan"))

    for fold in FOLDS:
        loaded = []
        for s in SEEDS:
            c = os.path.join(CACHE_DIR, f"{fold}_s{s}.json")
            if os.path.exists(c):
                loaded.append(json.load(open(c)))
        if not loaded:
            print(f"\n=== {fold}: (결과 없음)")
            continue
        print(f"\n=== {fold}  ({len(loaded)} seed) " + "=" * 56)

        # [A] 저금리에서 metab 레벨 sweep
        print(f"  [A] tbill=p{LOW_PCTL} 고정, metab 레벨 sweep (flat)  — skew / cvar1 / UWmean / UWcvar1")
        for pm in METAB_LEVELS:
            key = f"metab_p{pm}"
            sk = _agg([d["level_lowrate"][key]["skew"] for d in loaded])
            cv = _agg([d["level_lowrate"][key]["cvar1"] for d in loaded])
            uw = _agg([d["level_lowrate"][key]["uw_mean"] for d in loaded])
            uc = _agg([d["level_lowrate"][key]["uw_cvar1"] for d in loaded])
            print(f"    metab p{pm:<2}  skew {sk[0]:+.3f}  cvar1 {cv[0]:+.4f}  "
                  f"UWmean {uw[0]:+.4f}  UWcvar1 {uc[0]:+.4f}")

        # [B] 저금리에서 metab 동적 경로 (UWmean 제외 — 순서민감 UWcvar1·skew 로 판단)
        print(f"  [B] tbill=p{LOW_PCTL} 고정, metab 경로  "
              f"— pathσ / skew(분산효과) / UWcvar1(순서민감 꼬리) / term")
        for name in METAB_SHAPES:
            sg = PS.SHAPE_STD[name]
            sk = _agg([d["dynamic_lowrate"][name]["skew"] for d in loaded])
            uc = _agg([d["dynamic_lowrate"][name]["uw_cvar1"] for d in loaded])
            tm = _agg([d["dynamic_lowrate"][name]["term_mean"] for d in loaded])
            print(f"    {name:<10} σ{sg:.2f}  skew {sk[0]:+.3f}±{sk[1]:.3f}  "
                  f"UWcvar1 {uc[0]:+.4f}±{uc[1]:.4f}  term {tm[0]:+.4f}")

    print("\n[판정] [A] 저금리에서 metab 레벨↑ → skew 더 좌(−)·UWcvar1 더 깊으면 = 저금리 유동성 꼬리증폭 (4.2.1 zoom).")
    print("       [B] *분산효과*: pathσ↑(flat<ramp/hump<step) 따라 skew 더 좌 → 유동성 *변동 자체*가 모양 reshape.")
    print("           *순서효과*: 같은 σ 내 up vs down (ramp↑/↓, step↑/↓, hump/trough) 의 UWcvar1 차이 +fold 일관 → 순서위험.")
    print("           (skew 는 순서무감각이라 up≈down 당연 — 순서는 UWcvar1 로만 본다.)")
    print("       term 비슷한데 UWcvar1 만 다르면 → 순수 경로위험.  *상대* 비교만 (절대수치 OOD 한계 §5).")


if __name__ == "__main__":
    main()
