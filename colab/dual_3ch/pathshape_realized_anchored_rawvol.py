"""§4 경로(PATH) 시나리오 — *realized-anchored* 연속경로 (entry-jump 제거).

이전 flat-injection(p50 기반) PATH 의 문제:
  과거(실현·지속적) 끝에 p50 flat/shape 를 미래로 붙이면 진입점에서 점프 → 학습에서 본 적 없는
  OOD 조합. 주입값이 실현 최근값에서 멀수록 점프가 커지고, 순서효과(ramp↑/↓)는 진입점 자체가
  달라 path-dependence 와 entry-jump 가 섞인다.

이 버전 (사용자 설계):
  · 각 origin 의 *실현 마지막값*(real_*_fut[:,0])에서 flat 으로 연속(τ=0 deviation=0) → 진입 점프 0.
    = "실데이터를 끊김 없이 이어놓고" 거기서 shape 분기.
  · shape 는 zero-START (진입=0): flat / ramp↑↓ / step↑↓ / hump∩ / trough∪.
  · 한 축만 shape, 다른 축은 *마지막값 flat* 유지 (격리·연속).
  · 진폭 2단:  k=1 = 리만(GFC)급 = (tb/mb)_z[90]-[10] (step σz≈1.16, 학습분포 in-support 경계)
              k=3 = 리만 ×3 (OOD 외삽; COVID 는 ×5).  → 1×=in-support, 3×=증폭 스트레스로 분리 해석.

해석:
  · hump∩ vs trough∪ : 시작·끝 모두 마지막값(복귀) → 순수 순서효과(경로의존), 점프·끝점 통제 = 코어.
  · ramp↑/↓, step↑/↓ : 시작 같으나 끝 변위 → 상승 vs 하강 *방향* 효과(보조).
  · flat : 양 축 마지막값 flat = "현 상태 지속" baseline (양 축 공통, 1회 계산).

지표: skew / uw_mean / uw_cvar10(intra-horizon loss 10%).  seed 평균(paired: shape간 동일 난수).
모델 추론 필요 (Colab GPU).  캐시: result/pathshape_anchored_k1k3_cache/.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/pathshape_realized_anchored_rawvol.py
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

import analyze_pathshape_rawvol as PS                                # noqa: E402

CACHE_DIR = os.path.join(PS.RESULT_DIR, f"pathshape_anchored_k1k3_cache{PS.CACHE_SUFFIX}")
os.makedirs(CACHE_DIR, exist_ok=True)

FOLDS = PS.FOLDS
LABELS = {"F_gfc": "금융위기", "F_long_A": "회복기",
          "F_long_B_origin": "코로나", "F_long": "긴축기"}
SEEDS = PS.SEEDS
PAST_LEN = PS.PAST_LEN
FUTURE_LEN = PS.FUTURE_LEN
N_SIM = PS.N_SIM
CHUNK = PS.CHUNK

K_LIST = [1, 3]                                  # 1=리만급(in-support), 3=리만×3(OOD); COVID=5
NONFLAT = ["ramp_up", "ramp_down", "step_up", "step_down", "hump", "trough"]


def _tbill_wr(annual_pct):
    return (1.0 + annual_pct / 100.0) ** (1.0 / 52.0) - 1.0


def build_shapes(T):
    """모두 τ=0 에서 deviation=0 (진입 연속). peak=1 정규화 (× 진폭으로 스케일)."""
    idx = np.arange(T)
    half = T // 2
    step_up = (idx >= half).astype(float)                        # 0..0,1..1
    hump = np.interp(idx, [0, half, T - 1], [0.0, 1.0, 0.0])      # ∩ 0→+1→0
    return {
        "flat":      np.zeros(T),
        "ramp_up":   np.linspace(0.0, 1.0, T),
        "ramp_down": np.linspace(0.0, -1.0, T),
        "step_up":   step_up,
        "step_down": -step_up,
        "hump":      hump,                                        # ∩  (시작·끝=0)
        "trough":    -hump,                                       # ∪  (시작·끝=0)
    }


SHAPES = build_shapes(FUTURE_LEN)
SHAPE_STD = {k: float(np.std(v)) for k, v in SHAPES.items()}     # 정규화 형상 std (×K×range = 실제 주입 σz)


@torch.no_grad()
def rollout_paths(ctx, tb_path, mb_path, seed, device):
    """per-origin tb/mb z 경로 (n,FUT) 주입 → ar_sample. (n,n_sim,T) z. paired seed."""
    model = ctx["model"]; Xte = ctx["Xte"]; extra = ctx["extra"]; last_sp = ctx["last_sp"]
    n = Xte.shape[0]
    out = []
    torch.manual_seed(seed)
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK); k = e - s
        ft = torch.tensor(tb_path[s:e], device=device, dtype=Xte.dtype)              # (k,FUT)
        fm = torch.tensor(mb_path[s:e], device=device, dtype=Xte.dtype).reshape(k, FUTURE_LEN, 1)
        ex = extra[s:e] if extra is not None else None
        sim = model.ar_sample(Xte[s:e, :PAST_LEN, :], ft, last_sp[s:e], N_SIM,
                              extra_context=ex, future_macro_z=fm)
        out.append(sim.cpu())
    return torch.cat(out, dim=0).numpy()


def compute_fold_seed(fold, seed, device):
    ctx = PS.load_fold_seed(fold, seed, device)
    if ctx is None:
        return None
    rescale = ctx["rescale"]
    range_tb = ctx["tb_z"][90] - ctx["tb_z"][10]        # 리만급(1×) 진폭, z
    range_mb = ctx["mb_z"][90] - ctx["mb_z"][10]
    anc_tb = ctx["real_tb_fut"][:, 0:1]                 # (n,1) 실현 마지막값(연속 진입)
    anc_mb = ctx["real_mb_fut"][:, 0:1]
    tb_flat = anc_tb + np.zeros((1, FUTURE_LEN))        # (n,FUT) 마지막값 flat
    mb_flat = anc_mb + np.zeros((1, FUTURE_LEN))

    def metr(sim):
        m = PS.sim_metrics(sim, rescale)
        return {"skew": m["skew"], "uw_mean": m["uw_mean"], "uw_cvar10": m["uw_cvar10"]}

    res = {"flat": {}, "shape_tbill": {}, "shape_metab": {}}
    # 공통 baseline: 양 축 마지막값 flat ("현 상태 지속")
    res["flat"] = metr(rollout_paths(ctx, tb_flat, mb_flat, seed, device))
    for K in K_LIST:
        res["shape_tbill"][f"k{K}"] = {}
        res["shape_metab"][f"k{K}"] = {}
        for name in NONFLAT:
            sh = SHAPES[name][None, :]                              # (1,FUT)
            tb_path = anc_tb + sh * (K * range_tb)                  # 금리 shape, metab flat
            res["shape_tbill"][f"k{K}"][name] = metr(rollout_paths(ctx, tb_path, mb_flat, seed, device))
            mb_path = anc_mb + sh * (K * range_mb)                  # 유동성 shape, tbill flat
            res["shape_metab"][f"k{K}"][name] = metr(rollout_paths(ctx, tb_flat, mb_path, seed, device))
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 114)
    print(f"# §4 PATH (realized-anchored, jump-free) — LOCKED {PS.TAG_PREFIX}, device={device}, n_sim={N_SIM}")
    print(f"#   진입=실현 마지막값 flat 연속, shape=zero-start.  진폭 k={K_LIST} (1=리만/in-support, 3=리만×3/OOD)")
    print(f"#   한 축 shape, 다른 축 마지막값 flat.  지표 = skew / uw_mean / uw_cvar10(IHL 10%)")
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
    print("[집계] fold × seed 평균  (값 = skew / uw_mean / uw_cvar10).  flat=양축 마지막값 지속 baseline")
    for fold in FOLDS:
        label = LABELS.get(fold, fold)
        paths = [os.path.join(CACHE_DIR, f"{fold}_s{s}.json") for s in SEEDS]
        loaded = [json.load(open(p)) for p in paths if os.path.exists(p)]
        if not loaded:
            print(f"\n=== {label} [{fold}]: (결과 없음)"); continue
        print(f"\n=== {label} [{fold}]  ({len(loaded)} seed) " + "=" * 50)
        fsk = _agg([d["flat"]["skew"] for d in loaded])
        fum = _agg([d["flat"]["uw_mean"] for d in loaded])
        fuc = _agg([d["flat"]["uw_cvar10"] for d in loaded])
        print(f"  flat baseline (양축 마지막값 지속):  skew={fsk:+.3f}  uw_mean={fum:+.4f}  uw_cvar10={fuc:+.4f}")

        for var, head in (("shape_tbill", "금리(tbill) 경로 (metab flat)"),
                          ("shape_metab", "유동성(metab) 경로 (tbill flat)")):
            print(f"  [{head}]")
            for K in K_LIST:
                tag = "리만급/in-support" if K == 1 else f"리만×{K}/OOD"
                kk = f"k{K}"
                print(f"    k={K} ({tag})    {'shape':>10} {'fσ':>5} {'skew':>9} {'uw_mean':>10} {'uw_cvar10':>10}")
                for name in NONFLAT:
                    sk = _agg([d[var][kk][name]["skew"] for d in loaded])
                    um = _agg([d[var][kk][name]["uw_mean"] for d in loaded])
                    uc = _agg([d[var][kk][name]["uw_cvar10"] for d in loaded])
                    print(f"    {'':>17}{name:>10} {SHAPE_STD[name]:>5.2f} {sk:>+9.3f} {um:>+10.4f} {uc:>+10.4f}")
                # 순서/방향 효과 (uw_cvar10)
                pairs = [("hump", "trough", "순서∩∪"), ("step_up", "step_down", "방향step"),
                         ("ramp_up", "ramp_down", "방향ramp")]
                for a, b, tagp in pairs:
                    ua = _agg([d[var][kk][a]["uw_cvar10"] for d in loaded])
                    ub = _agg([d[var][kk][b]["uw_cvar10"] for d in loaded])
                    deeper = a if ua < ub else b
                    print(f"    {'':>17}[{tagp}] {a}={ua:+.4f} vs {b}={ub:+.4f} → {deeper} 더 깊음")
    print("\n[읽는 법] 모든 shape 진입=실현 마지막값(연속) → entry-jump 없음.")
    print("  hump∩ vs trough∪: 시작·끝 동일·중간 순서만 → IHL 차 = 순수 경로의존(코어).")
    print("  step·ramp ↑/↓: 끝 변위 → 상승/하강 *방향* 효과(보조).")
    print("  k=1=리만(in-support) / k=3=리만×3(OOD 외삽) 분리 해석.  순서효과가 k=3서도 부호 일관이면 진폭 키워도 경로의존 유지.")


if __name__ == "__main__":
    main()
