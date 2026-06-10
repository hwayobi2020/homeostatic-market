"""§4.1 model-on-realized — 각 origin에 *실현* 미래 거시경로를 주입한 모델 출력을 실현상태로 binning.

표 4.2(실측)와 동일 격자(실현 tbill × 실현 metab, 절대레벨 bin)에 모델 결과를 얹어:
  · 모델이 *실제 시나리오*를 받았을 때 현실을 재현하는지 (모델 vs 실측 직접 대조)
  · 이 격자의 각 칸 = 이후 ±Δ 민감도 테스트의 *베이스라인(0,0)* — 시작점.

per-origin: 그 origin의 실현 미래경로 주입 → 1000 sim → 그 origin의 skew/UWcvar1/uw_mean
           (UWcvar1 은 origin당 1000 sim이라 잘 추정됨; 실측 표의 n=1 문제 없음).
그 다음 origin을 실현상태(미래13주 평균 tbill·metab)로 절대 bin → 칸 평균.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/model_on_realized_rawvol.py
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

CACHE_DIR = os.path.join(PS.RESULT_DIR, "model_realized_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

FOLDS = PS.FOLDS
SEEDS = PS.SEEDS
FUTURE_LEN = PS.FUTURE_LEN
CHUNK = PS.CHUNK
N_SIM = PS.N_SIM
PAST_LEN = PS.PAST_LEN
BINS = ["lo", "mid", "hi"]


def _tbill_wr(annual_pct):
    return (1.0 + annual_pct / 100.0) ** (1.0 / 52.0) - 1.0


# 절대 레벨 (compare/coverage 와 동일) — 실현상태 binning 기준
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
def rollout_realized(ctx, device):
    """각 origin에 그 origin의 실현 미래 거시경로(z)를 주입해 ar_sample. (n, n_sim, T) z 반환."""
    model = ctx["model"]; Xte = ctx["Xte"]; extra = ctx["extra"]; last_sp = ctx["last_sp"]
    tb = ctx["real_tb_fut"]; mb = ctx["real_mb_fut"]            # (n, FUTURE_LEN) z
    n = Xte.shape[0]
    out = []
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK); k = e - s
        ft = torch.tensor(tb[s:e], device=device, dtype=Xte.dtype)                 # (k, FUT)
        fm = torch.tensor(mb[s:e], device=device, dtype=Xte.dtype).reshape(k, FUTURE_LEN, 1)
        ex = extra[s:e] if extra is not None else None
        sim = model.ar_sample(Xte[s:e, :PAST_LEN, :], ft, last_sp[s:e], N_SIM,
                              extra_context=ex, future_macro_z=fm)
        out.append(sim.cpu())
    return torch.cat(out, dim=0).numpy()


def per_origin_metrics(sim_z, rescale):
    """origin별 skew / uw_mean / uw_cvar1 (각 origin의 1000 sim 기반)."""
    r = rescale
    sim_raw = PS.forward_garch_rescale(sim_z * r["tsd"] + r["tmu"], r["s2"], r["e2"],
                                       r["om"], r["al"], r["be"], r["mu"])           # (n, nsim, T)
    res = []
    for i in range(sim_raw.shape[0]):
        a = sim_raw[i]                                                              # (nsim, T)
        f = a.ravel()
        cum = np.concatenate([np.zeros((a.shape[0], 1), dtype=a.dtype),
                              np.cumsum(a, axis=1)], axis=1)
        uw = cum.min(axis=1)                                                        # (nsim,) intra-horizon loss
        res.append((PS._skew(f), float(uw.mean()), float(PS.compute_cvar(uw, 0.01))))
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 110)
    print(f"# §4.1 model-on-realized — 실현경로 주입 모델 출력, 실현상태(tbill×metab 절대bin) 격자")
    print(f"#   LOCKED {PS.TAG_PREFIX}, device={device}, n_sim={N_SIM} | 칸=skew / UWcvar1 / uw_mean (n)")
    print("#" * 110)

    for fold, label in FOLDS:
        rows = []   # (tb_bin, mb_bin, skew, uw_mean, uw_cvar1)
        for seed in SEEDS:
            cache = os.path.join(CACHE_DIR, f"{fold}_s{seed}.json")
            if os.path.exists(cache):
                rows.extend(json.load(open(cache)))
                continue
            ctx = PS.load_fold_seed(fold, seed, device)
            if ctx is None:
                continue
            sim = rollout_realized(ctx, device)
            m = per_origin_metrics(sim, ctx["rescale"])
            tbm = ctx["real_tb_mean"]; mbm = ctx["real_mb_mean"]
            rs = [[abin(float(tbm[i]), TBILL_EDGES), abin(float(mbm[i]), METAB_EDGES),
                   m[i][0], m[i][1], m[i][2]] for i in range(len(m))]
            json.dump(rs, open(cache, "w"))
            rows.extend(rs)
            print(f"[done] {fold} s{seed}  (origins={len(rs)})")

        print("\n" + "=" * 110)
        print(f"=== {label}  [{fold}] ===")
        if not rows:
            print("  (결과 없음)"); continue
        filled = 0
        print(f"  [모델 on 실현경로]  skew / UWcvar1 / uw_mean (n)")
        print(f"    {'tbill＼metab':<12}" + "".join(f"{('metab ' + b):>26}" for b in BINS))
        for a in BINS:
            row = f"    {('tbill ' + a):<12}"
            for b in BINS:
                cell = [r for r in rows if r[0] == a and r[1] == b]
                if not cell:
                    row += f"{'—':>26}"
                    continue
                filled += 1
                sk = np.mean([c[2] for c in cell]); um = np.mean([c[3] for c in cell])
                uc = np.mean([c[4] for c in cell]); nn = len(cell)
                row += f"{f'{sk:+.2f}/{uc:+.3f}/{um:+.3f}(n{nn})':>26}"
            print(row)
        print(f"  채워진 칸 = {filled}/9  (n = origin×seed pooled)")

    print("\n[읽는 법] 이 격자 = 표 4.2(실측) 옆에 놓고 칸별 모델 vs 실측 대조.")
    print("          각 칸 = ±Δ 민감도 테스트의 베이스라인(0,0). 다음 단계가 여기서 ±Δ.")


if __name__ == "__main__":
    main()
