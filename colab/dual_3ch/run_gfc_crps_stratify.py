"""F_gfc — 금리 변동/평탄 구간별로 self-stat(MLP-Flow) vs GARCH 의 CRPS 비교.

사용자 목표 (2026-05-23):
  "13주 동안 금리가 25bp 이상 변동한 origin 에서, 우리 MLP-Flow 가 GARCH 보다
   CRPS 가 더 좋다(낮다)는 걸 보고 싶다."
  마스크/논마스크 무관 — 이미 돌린 모델 그대로. 전체 평균에선 GARCH 가 이기지만
  (평탄 구간이 다수), 금리 격변 구간만 떼면 Flow 의 풍부한 조건화가 이길 수 있나.

단위 = 13주 (origin 하나 = 13주 예측 하나). per-origin CRPS = 그 origin 13주 평균.
CRPS 는 Flow·GARCH 동일하게 샘플로 계산 → model-class 무관 공정 비교 (NLL 불가).

전제(이미 있어야 함):
  - self-stat: mamba_flow_ar_selfstat_mask_mlp_s{seed}_F_gfc_best.pt  (run_gfc_selfstat)
  - GARCH   : garch_pure_F_gfc_crps_per_origin.npy  (train_garch_ar 재실행으로 생성)
self-stat 의 per-origin CRPS npy 가 없으면 ckpt 로 evaluate_test 재호출(sampling, 학습X).

Usage (Colab):
    !python colab/dual_3ch/train_garch_ar.py --fold F_gfc --use-arx 0 --dist normal
    !python colab/dual_3ch/run_gfc_crps_stratify.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import train_mamba_flow_ar as T            # noqa: E402
from train_mamba_flow_ar import (          # noqa: E402
    MambaFlowAR, evaluate_test)
from best_specs import BEST_SPECS          # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLD = "F_gfc"
SEEDS = [2026, 2027, 2028, 2029, 2030]
PAST, FUT = 52, 13
BP = 0.25   # 25bp = 0.25 연율 %p


def set_cond_cols(cols):
    cols = list(cols)
    T.COND_COLS = cols
    T.N_CHANNELS = len(cols)
    T.SP_CH = cols.index("sp_return")
    T.TBILL_CH = cols.index("tbill_wr")
    T.MACRO_CH = [c for c in range(len(cols)) if c not in (T.SP_CH, T.TBILL_CH)]
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass


def _nd(arr):
    """날짜 배열을 'YYYY-MM-DD' 로 정규화 (Timestamp str '..00:00:00' vs '..' 차이 흡수)."""
    return (pd.to_datetime(pd.Series(np.asarray(arr).astype(str)))
            .dt.strftime("%Y-%m-%d").values)


def selfstat_per_origin(seed, dev):
    """self-stat per-origin per-week CRPS npy 로드 (없으면 ckpt 로 evaluate_test 재계산)."""
    tag = f"selfstat_mask_mlp_s{seed}"
    prefix = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{FOLD}")
    crps_npy, dates_npy = f"{prefix}_crps_per_origin.npy", f"{prefix}_origin_dates.npy"
    if os.path.exists(crps_npy) and os.path.exists(dates_npy):
        return np.load(crps_npy), np.load(dates_npy, allow_pickle=True)
    ckpt = f"{prefix}_best.pt"
    if not os.path.exists(ckpt):
        return None, None
    ck = torch.load(ckpt, map_location=dev)
    meta = ck["meta"]
    set_cond_cols(meta["cond_cols"])
    T.MASK_FUTURE_TBILL = True               # self-stat 은 마스크 학습
    model = MambaFlowAR(
        d_input=len(meta["cond_cols"]), d_model=meta["d_model"],
        n_flow_layers=meta["n_flow_layers"], n_flow_hidden=meta["n_flow_hidden"],
        n_flow_blocks=meta["n_flow_blocks"], n_flow_bins=meta["n_flow_bins"],
        flow_tail_bound=meta["flow_tail_bound"], dropout=meta["dropout"],
        extra_context_dim=meta["extra_context_dim"],
        encoder_type="mlp", mlp_num_layers=BEST_SPECS["mlp"]["mlp_num_layers"],
        direct_prev_return=True).to(dev)
    test_csv = os.path.join(FOLDS_DIR, f"{FOLD}_test.csv")
    print(f"  [resample] {tag} (evaluate_test, no training)")
    evaluate_test(model, ck["model_state"], test_csv, meta["cond_stats"],
                  meta["target_stats"], prefix, 1000, seed, dev, chunk_origins=8,
                  extra_cond_cols=meta["extra_cond_cols"], extra_stats=meta["extra_stats"])
    return np.load(crps_npy), np.load(dates_npy, allow_pickle=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1) self-stat per-origin CRPS (seed 평균) ──
    crps_list, base_dates = [], None
    for seed in SEEDS:
        c, d = selfstat_per_origin(seed, dev)
        if c is None:
            print(f"[skip seed {seed}] ckpt/npy 없음")
            continue
        d = _nd(d)
        if base_dates is None:
            base_dates = d
        if not np.array_equal(d, base_dates):
            sys.exit(f"[FATAL] seed {seed} origin date 순서 불일치")
        crps_list.append(c)
    if not crps_list:
        sys.exit("[FATAL] self-stat per-origin CRPS 없음 (ckpt 누락?)")
    self_crps = np.mean(crps_list, axis=0).mean(axis=1)   # (n_origin,) 13주 평균
    self_dates = base_dates
    print(f"[self-stat] {len(crps_list)} seed 평균, n_origin={len(self_crps)}")

    # ── 2) GARCH per-origin CRPS ──
    gpref = os.path.join(RESULT_DIR, f"garch_pure_{FOLD}")
    if not os.path.exists(f"{gpref}_crps_per_origin.npy"):
        sys.exit(f"[FATAL] {gpref}_crps_per_origin.npy 없음 — 먼저:\n"
                 f"  !python colab/dual_3ch/train_garch_ar.py --fold {FOLD} --use-arx 0 --dist normal")
    g_crps = np.load(f"{gpref}_crps_per_origin.npy").mean(axis=1)   # (n_origin,)
    g_dates = _nd(np.load(f"{gpref}_origin_dates.npy", allow_pickle=True))
    g_map = {d: i for i, d in enumerate(g_dates)}

    # ── 3) origin 별 horizon 13주 금리 변화 (부호, 연율 %p) ──
    test_csv = os.path.join(FOLDS_DIR, f"{FOLD}_test.csv")
    dft = pd.read_csv(test_csv)
    dft_dates = _nd(dft["date"])
    didx = {d: i for i, d in enumerate(dft_dates)}
    rate_pp = (((1.0 + pd.to_numeric(dft["tbill_wr"], errors="coerce")) ** 52 - 1.0)
               * 100.0).values

    rows = []
    for i, cd in enumerate(self_dates):
        cd = str(cd)
        if cd not in g_map or cd not in didx:
            continue
        r = didx[cd]                       # conditioning row (마지막 과거)
        if r + FUT >= len(rate_pp):
            continue
        chg = rate_pp[r + FUT] - rate_pp[r]   # horizon 13주 금리 변화
        rows.append(dict(date=cd, chg=float(chg),
                         self=float(self_crps[i]), garch=float(g_crps[g_map[cd]])))

    R = pd.DataFrame(rows)
    if len(R) == 0:
        print("[FATAL] 매칭된 origin 0개 — date 정규화 후에도 불일치")
        print("  self_dates[:3] :", list(self_dates[:3]))
        print("  g_dates[:3]    :", list(g_dates[:3]))
        print("  csv dates[:3]  :", list(dft_dates[:3]))
        return
    R["regime"] = np.where(R.chg <= -BP, "인하",
                  np.where(R.chg >= BP, "인상", "평탄"))

    print("\n" + "=" * 92)
    print(f"F_gfc — 금리 변화 구간별 CRPS (per-origin=13주 단위)  self-stat(MLP-Flow) vs GARCH")
    print("  CRPS 낮을수록 좋음 | diff = self − garch  ( <0 = Flow 우위 )")
    print("=" * 92)
    print(f"{'구간':<14}{'n':>5}{'self CRPS':>14}{'GARCH CRPS':>14}{'diff(self−garch)':>20}")
    order = [("전체", R), ("평탄", R[R.regime == "평탄"]),
             ("변동(인하+인상)", R[R.regime != "평탄"]),
             ("  └ 인하", R[R.regime == "인하"]),
             ("  └ 인상", R[R.regime == "인상"])]
    for lab, sub in order:
        if len(sub) == 0:
            print(f"{lab:<14}{0:>5}{'n/a':>14}")
            continue
        s, g = sub["self"].mean(), sub["garch"].mean()
        win = "  ← Flow 우위" if s < g else ""
        print(f"{lab:<14}{len(sub):>5}{s:>14.5f}{g:>14.5f}{s - g:>+20.5f}{win}")
    out = os.path.join(RESULT_DIR, "gfc_crps_stratify.csv")
    R.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
