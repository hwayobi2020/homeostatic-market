"""표 4.2/4.3 재작성 — model-on-realized, *일관 집계*(origin pooled, seed 평균).

설계 (2026-06 확정):
  지표 = skew / uw_mean(평균 IHL) / IHL CVaR10%(worst-decile).  (1% 폐기: 실측 49→1점)
  집계 = Model·Factual 둘 다 'bin 안 origin 합쳐(pool)' 계산.
         Model 은 seed별 pool → 3 seed 평균(seed 합치기 금지 = 혼합 왜곡 방지). skew 는 seed-std 병기.
  n = origin 개수(model·factual 동일).  n<30 = 별표(제외 권장).
  Model: model-on-realized 재샘플(추론) 필요(raw sim 캐시 없음).  Factual: fold CSV 로 pool.
  ⚠ rolling origin 중첩으로 skew 실효표본 < n×13 → skew 는 방향+seed std 로 읽기.

축: 4.2 = metab 레벨, 4.3 = tbill 레벨 (marginal, 다른 축 무시).
캐시: result/table4243_pooled_cache/{fold}_s{seed}.json (모델 bin metric, 재진입).

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/build_table_42_43_pooled.py
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

import analyze_pathshape_rawvol as PS                                 # noqa: E402
import model_on_realized_rawvol as MR                                 # noqa: E402

RESULT_DIR = PS.RESULT_DIR
FOLDS_DIR = PS.FOLDS_DIR
CACHE_DIR = os.path.join(RESULT_DIR, f"table4243_pooled_cache{PS.CACHE_SUFFIX}")
os.makedirs(CACHE_DIR, exist_ok=True)

FOLDS = PS.FOLDS
SEEDS = PS.SEEDS
PAST_LEN = PS.PAST_LEN
FUTURE_LEN = PS.FUTURE_LEN
LABELS = MR.LABELS
BINS = ["lo", "mid", "hi"]
BINLAB = {"lo": "Low", "mid": "Mid", "hi": "Hi"}
N_STAR = 30                                  # n<30 = 별표
TI = PS.ENC_COLS.index("tbill_wr")
MI = PS.ENC_COLS.index("metab_13w")
AXES = [("metab", "4.2", MI, MR.METAB_EDGES), ("tbill", "4.3", TI, MR.TBILL_EDGES)]


def _skew(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def cvar(a, q):
    a = np.asarray(a, float)
    k = max(1, int(len(a) * q))
    return float(np.sort(a)[:k].mean())


def underwater(arr2d):
    """(M, T) → (M,) 진입대비 최악 누적낙폭(<=0)."""
    cum = np.cumsum(arr2d, axis=1)
    z0 = np.zeros((arr2d.shape[0], 1), dtype=cum.dtype)
    return np.concatenate([z0, cum], axis=1).min(axis=1)


def metrics_pool(paths2d):
    """paths2d: (M, T) pooled 경로 → skew(주간) / uw_mean / IHL10%."""
    f = paths2d.ravel()
    uw = underwater(paths2d)
    return dict(skew=_skew(f), uw_mean=float(uw.mean()), ihl10=cvar(uw, 0.10), n=int(paths2d.shape[0]))


# ── 모델: (fold,seed) → 축·bin 별 pooled metric ──
@torch.no_grad()
def model_fold_seed(fold, seed, device):
    cache = os.path.join(CACHE_DIR, f"{fold}_s{seed}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    ctx = PS.load_fold_seed(fold, seed, device)
    if ctx is None:
        return None
    sim_z = MR.rollout_realized(ctx, device)                          # (n, n_sim, T) z
    r = ctx["rescale"]
    sim_raw = PS.forward_garch_rescale(sim_z * r["tsd"] + r["tmu"], r["s2"], r["e2"],
                                       r["om"], r["al"], r["be"], r["mu"])   # (n, n_sim, T)
    macro = {"metab": ctx["real_mb_mean"], "tbill": ctx["real_tb_mean"]}      # (n,) raw
    out = {}
    for axis, _tbl, _ci, edges in AXES:
        bvals = [MR.abin(float(v), edges) for v in macro[axis]]
        cells = {}
        for b in BINS:
            idx = [i for i, bb in enumerate(bvals) if bb == b]
            if not idx:
                continue
            pooled = sim_raw[idx].reshape(len(idx) * sim_raw.shape[1], FUTURE_LEN)
            m = metrics_pool(pooled)
            m["n_origin"] = len(idx)
            cells[b] = m
        out[axis] = cells
    json.dump(out, open(cache, "w"))
    return out


# ── 실측: fold → 축·bin 별 pooled metric (seed 무관) ──
def factual_fold(fold, device):
    bp = os.path.join(RESULT_DIR, f"garch_flow_ar_{PS.TAG_PREFIX}_s{SEEDS[0]}_{fold}_best.pt")
    if not os.path.exists(bp):
        return None
    cond_stats = torch.load(bp, map_location="cpu")["meta"]["cond_stats"]
    cmu = np.asarray(cond_stats["mean"], float); csd = np.asarray(cond_stats["std"], float)
    test_csv = PS.garch_preprocess_fold(FOLDS_DIR, fold, RESULT_DIR)["test"]
    valid_mask, z_te, df_te = PS.compute_valid_mask(test_csv, cond_stats)
    sp = (df_te["sp_return"].to_numpy(float) * df_te["garch_sigma"].to_numpy(float)
          + df_te["garch_mu"].to_numpy(float))                        # raw 주간수익률
    n_w = z_te.shape[0] - PAST_LEN - FUTURE_LEN + 1
    win = lambda arr: np.stack([arr[w + PAST_LEN: w + PAST_LEN + FUTURE_LEN]
                                for w in range(n_w)])[valid_mask]
    real = win(sp)                                                    # (n, T)
    tb_mean = np.stack([z_te[w + PAST_LEN: w + PAST_LEN + FUTURE_LEN, TI]
                        for w in range(n_w)])[valid_mask].mean(1) * csd[TI] + cmu[TI]
    mb_mean = np.stack([z_te[w + PAST_LEN: w + PAST_LEN + FUTURE_LEN, MI]
                        for w in range(n_w)])[valid_mask].mean(1) * csd[MI] + cmu[MI]
    macro = {"metab": mb_mean, "tbill": tb_mean}
    out = {}
    for axis, _tbl, _ci, edges in AXES:
        bvals = [MR.abin(float(v), edges) for v in macro[axis]]
        cells = {}
        for b in BINS:
            idx = [i for i, bb in enumerate(bvals) if bb == b]
            if not idx:
                continue
            m = metrics_pool(real[idx])
            m["n_origin"] = len(idx)
            cells[b] = m
        out[axis] = cells
    return out


def _agg_seed(per_seed, axis, b, key):
    vals = [d[axis][b][key] for d in per_seed if axis in d and b in d[axis]]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 110)
    print(f"# 표 4.2/4.3 재작성 — origin pooled, seed 평균.  지표 skew / uw_mean / IHL CVaR10%  (device={device})")
    print(f"#  Model: seed별 pool→{len(SEEDS)}seed 평균(skew±std) | Factual: pool | n=origin 수 | n<{N_STAR}=*")
    print("#" * 110)

    for axis, tbl, _ci, _e in AXES:
        print("\n" + "=" * 110)
        print(f"### 표 {tbl}  model-on-realized — {axis} 축  (칸 = skew(±std) / uw_mean / IHL10%, n)")
        for fold in FOLDS:
            per_seed = [d for d in (model_fold_seed(fold, s, device) for s in SEEDS) if d is not None]
            fac = factual_fold(fold, device)
            if not per_seed or fac is None:
                print(f"  {LABELS.get(fold, fold)}: (결과 없음)"); continue
            print(f"\n  [{LABELS.get(fold, fold)}]")
            print(f"    {'bin':<5}{'MODEL skew(±sd)/uwm/ihl10 (n)':<42}{'HISTORICAL skew/uwm/ihl10 (n)':<38}")
            for b in BINS:
                mc = per_seed[0].get(axis, {}).get(b)
                fc = fac.get(axis, {}).get(b)
                if mc is None and fc is None:
                    continue
                if mc is not None:
                    sk, sd = _agg_seed(per_seed, axis, b, "skew")
                    um, _ = _agg_seed(per_seed, axis, b, "uw_mean")
                    ih, _ = _agg_seed(per_seed, axis, b, "ihl10")
                    nm = mc["n_origin"]
                    star = "*" if nm < N_STAR else ""
                    mstr = f"{sk:+.2f}(±{sd:.2f})/{um:+.3f}/{ih:+.3f} (n{nm}{star})"
                else:
                    mstr = "—"
                if fc is not None:
                    nf = fc["n_origin"]; star = "*" if nf < N_STAR else ""
                    fstr = f"{fc['skew']:+.2f}/{fc['uw_mean']:+.3f}/{fc['ihl10']:+.3f} (n{nf}{star})"
                else:
                    fstr = "—"
                print(f"    {BINLAB[b]:<5}{mstr:<42}{fstr:<38}")
    print("\n[주] skew 는 rolling origin 중첩으로 노이즈 큼 → 방향+seed std 로 해석. "
          "IHL10% 도 작은 bin(n작음)은 worst-decile 점수 적어 참고.")


if __name__ == "__main__":
    main()
