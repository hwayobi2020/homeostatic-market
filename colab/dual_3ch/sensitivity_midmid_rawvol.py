"""§4.1 sensitivity test — (중간금리 ∩ 중간유동성) 베이스라인에서 한 축 ±Δ 평행이동.

설계 근거:
  ±Δ 교란은 *mid* 베이스라인에서만 대칭적으로 성립한다. 저금리에서 금리를 더 내리거나
  저유동성에서 유동성을 더 빼는 것은 한쪽으로 막힌 교란이라 무의미.
  → 실현상태가 (tbill=mid, metab=mid)인 origin만 베이스라인으로 삼고 위·아래로 흔든다.
  그 (mid,mid) 칸이 양쪽 다 채워진 fold = 금융위기(F_gfc)·코로나(F_long_B_origin) 둘뿐.

방법:
  · 베이스라인 = 선택 origin의 *실현* 미래 13주 거시 z 경로(real_tb_fut / real_mb_fut).
  · 한 축만 전 구간 평행이동(+Δz), 다른 축은 실현값 유지.
      금리: Δ%p(연율) → mid(2.5%) 기준 주간 tbill_wr 증분 → /csd 로 z.
      유동성: Δ(13주 raw) → /csd 로 z.
  · 교란경로로 ar_sample → sim_metrics 로 skew / intra-horizon-loss(1% CVaR).
  · Δ지표 = (Δ값 결과) − (Δ=0 = 실현 베이스라인).  seed 평균.

주의: 이건 *모델 추론*이 필요하다(교란마다 다른 sim). 재-binning 같은 공짜 아님. Colab GPU.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/sensitivity_midmid_rawvol.py
"""
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

FOLDS = ["F_gfc", "F_long_B_origin"]
LABELS = {"F_gfc": "금융위기", "F_long_B_origin": "코로나"}
SEEDS = PS.SEEDS
PAST_LEN = PS.PAST_LEN
FUTURE_LEN = PS.FUTURE_LEN
N_SIM = PS.N_SIM
CHUNK = PS.CHUNK

TBILL_DELTAS = [-1.0, -0.5, 0.0, 0.5, 1.0]      # %p (연율)
METAB_DELTAS = [-0.02, -0.01, 0.0, 0.01, 0.02]  # 13주 raw (= -2%/-1%/0/+1%/+2%)


def _tbill_wr(annual_pct):
    return (1.0 + annual_pct / 100.0) ** (1.0 / 52.0) - 1.0


# 절대 레벨 (model-on-realized 와 동일) — 실현상태 binning 기준
TBILL_LV = (_tbill_wr(0.25), _tbill_wr(2.5), _tbill_wr(5.0))
METAB_LV = (-0.02, 0.0, 0.025)
TBILL_EDGES = ((TBILL_LV[0] + TBILL_LV[1]) / 2.0, (TBILL_LV[1] + TBILL_LV[2]) / 2.0)
METAB_EDGES = ((METAB_LV[0] + METAB_LV[1]) / 2.0, (METAB_LV[1] + METAB_LV[2]) / 2.0)


def abin(v, edges):
    lo_mid, mid_hi = edges
    if not np.isfinite(v):
        return None
    if v < lo_mid:
        return "lo"
    if v > mid_hi:
        return "hi"
    return "mid"


@torch.no_grad()
def run_perturbed(ctx, sel, axis, dz, device):
    """선택 origin(sel)의 실현 경로에서 axis 한 축만 +dz(z) 평행이동 → ar_sample. (m, n_sim, T) z."""
    model = ctx["model"]; Xte = ctx["Xte"]; extra = ctx["extra"]; last_sp = ctx["last_sp"]
    tb = ctx["real_tb_fut"][sel].copy()                  # (m, FUT) z
    mb = ctx["real_mb_fut"][sel].copy()                  # (m, FUT) z
    if axis == "tbill":
        tb = tb + dz
    else:
        mb = mb + dz
    Xs = Xte[sel]; lsps = last_sp[sel]
    exs = extra[sel] if extra is not None else None
    m = len(sel)
    out = []
    for s in range(0, m, CHUNK):
        e = min(m, s + CHUNK); k = e - s
        ft = torch.tensor(tb[s:e], device=device, dtype=Xte.dtype)                 # (k, FUT)
        fm = torch.tensor(mb[s:e], device=device, dtype=Xte.dtype).reshape(k, FUTURE_LEN, 1)
        ex = exs[s:e] if exs is not None else None
        sim = model.ar_sample(Xs[s:e, :PAST_LEN, :], ft, lsps[s:e], N_SIM,
                              extra_context=ex, future_macro_z=fm)
        out.append(sim.cpu())
    return torch.cat(out, dim=0).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 110)
    print("# §4.1 sensitivity — (mid tbill ∩ mid metab) 베이스라인 ±Δ 평행이동 (금융위기·코로나)")
    print(f"#   LOCKED {PS.TAG_PREFIX}, device={device}, n_sim={N_SIM}")
    print(f"#   금리 Δ(%p): {TBILL_DELTAS}   유동성 Δ: {METAB_DELTAS}")
    print(f"#   값 = skew / IHL(1%);  Δ = (해당 Δ) − (Δ=0 실현 베이스라인);  seed {SEEDS} 평균")
    print("#" * 110)

    for fold in FOLDS:
        label = LABELS.get(fold, fold)
        # seed별 결과 누적: acc[axis][delta] = list of (skew, ihl)
        acc = {"tbill": {d: [] for d in TBILL_DELTAS},
               "metab": {d: [] for d in METAB_DELTAS}}
        n_sel = None
        for seed in SEEDS:
            ctx = PS.load_fold_seed(fold, seed, device)
            if ctx is None:
                print(f"[skip] {fold} s{seed}  (ctx None)")
                continue
            tbm = ctx["real_tb_mean"]; mbm = ctx["real_mb_mean"]
            sel = [i for i in range(len(tbm))
                   if abin(float(tbm[i]), TBILL_EDGES) == "mid"
                   and abin(float(mbm[i]), METAB_EDGES) == "mid"]
            if len(sel) == 0:
                print(f"[warn] {fold} s{seed}  (mid,mid) origin 0개 — skip")
                continue
            n_sel = len(sel)
            rsel = dict(ctx["rescale"])
            rsel["s2"] = ctx["rescale"]["s2"][sel]
            rsel["e2"] = ctx["rescale"]["e2"][sel]
            csd = ctx["csd"]; ti = ctx["ti"]; mi = ctx["mi"]

            for d in TBILL_DELTAS:
                dz = (_tbill_wr(2.5 + d) - _tbill_wr(2.5)) / csd[ti]
                m = PS.sim_metrics(run_perturbed(ctx, sel, "tbill", dz, device), rsel)
                acc["tbill"][d].append((m["skew"], m["uw_cvar1"]))
            for d in METAB_DELTAS:
                dz = d / csd[mi]
                m = PS.sim_metrics(run_perturbed(ctx, sel, "metab", dz, device), rsel)
                acc["metab"][d].append((m["skew"], m["uw_cvar1"]))
            print(f"[done] {fold} s{seed}  (mid,mid) n={len(sel)}")

        # 집계·출력
        print("\n" + "=" * 110)
        print(f"=== {label}  [{fold}]   (mid,mid) 베이스라인 origin = {n_sel}개/seed ===")
        if n_sel is None:
            print("  (결과 없음)"); continue
        for axis, deltas, unit in (("tbill", TBILL_DELTAS, "%p"),
                                   ("metab", METAB_DELTAS, "raw13w")):
            sk = {d: float(np.mean([x[0] for x in acc[axis][d]])) for d in deltas}
            il = {d: float(np.mean([x[1] for x in acc[axis][d]])) for d in deltas}
            sk0 = sk[0.0]; il0 = il[0.0]
            head = "금리(tbill)" if axis == "tbill" else "유동성(metab)"
            print(f"\n  [{head} 축 sweep]   Δ단위={unit}")
            print(f"    {'Δ':>8}" + "".join(f"{d:>+10.2f}" for d in deltas))
            print(f"    {'skew':>8}" + "".join(f"{sk[d]:>+10.3f}" for d in deltas))
            print(f"    {'Δskew':>8}" + "".join(f"{sk[d]-sk0:>+10.3f}" for d in deltas))
            print(f"    {'IHL1%':>8}" + "".join(f"{il[d]:>+10.3f}" for d in deltas))
            print(f"    {'ΔIHL':>8}" + "".join(f"{il[d]-il0:>+10.3f}" for d in deltas))

    print("\n[읽는 법] Δ=0 열이 실현 베이스라인. Δskew/ΔIHL 이 단조(+Δ금리→? / +Δ유동성→skew↓?)면")
    print("          그 축이 꼬리를 *인과적으로* 움직인다는 모델 내부 증거 (mid 출발이라 양방향 대칭).")


if __name__ == "__main__":
    main()
