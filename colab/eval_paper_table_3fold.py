"""Paper 표 17 행 평가 — sp_return 정규화 + 3-fold walk-forward 통합.

각 변종 × 3 fold × 5 seed = 15 measurement → sp NLL full + tail (raw 단위, Jacobian 보정).

Jacobian 보정: ckpt 가 sp_return 을 정규화한 경우 (stats_targets_all 에 SP idx 항이 있으면)
    NLL_raw = NLL_z + log(σ_sp_train)
    σ_sp_train ≈ 0.025 → log(σ) ≈ -3.69
    raw 단위 NLL = z 단위 NLL - 3.69 정도 (값 자체는 σ 마다 다름)

paper 표 17 행 정합성:
    1-3 : Base (K2_pure_ppbond / without liq / add bondpp)        — k2_104 폴더
    4   : Stage 1 (MacroExpander)                                 — sp target 없음 (n/a)
    5   : Stage 1 (vix정규)                                       — sp target 없음 (n/a)
    6   : Stage 2 (PriceGenerator)
    7   : Stage 2 (대조군 / SingleStage)
    8   : MTL(2 채널, liq)                                        — joint
    9-13: MTL(2 채널, vix정규/bondpp정규/stockpp정규/...) [no-liq]
    14-17: MTL(3 채널, liq, ...) / MTL(3 채널, ..., [no-liq])
"""
import torch  # MUST be first

import argparse
import json
import math
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "dual_3ch"))
from favar_flow import MultiStepFAVARFlow  # noqa: E402

LOG2PI = math.log(2 * math.pi)
SEEDS = [42, 123, 777, 0, 99]
FOLDS = ["F1", "F2", "F3"]

# paper 표 17 행 정의 — (paper_row, label, folder, prefix_template)
# prefix_template 의 {fold} 자리는 _F1/_F2/_F3 또는 빈 문자열 (fold 없는 경우)
VARIANTS = [
    # 1-3: Base — k2_104 폴더 (별도 작업 필요)
    ( 1, "Base (K2_pure_ppbond)",                                   "k2_104",   "K2_104{fold}"),
    ( 2, "Base (without liquidity)",                                "k2_104",   "K2_104_2ch{fold}"),
    ( 3, "Base (add bondpp_3m정규)",                                 "k2_104",   "K2_104_addbp_normbp_normsr{fold}"),
    # 4-5: Stage 1 (sp target 없음 → eval 시 n/a)
    ( 4, "Stage 1 (MacroExpander)",                                 "dual_3ch", "stage1{fold}"),
    ( 5, "Stage 1 (vix정규)",                                       "dual_3ch", "stage1vix_normvix{fold}"),
    # 6-7: Stage 2 / SingleStage
    ( 6, "Stage 2 (PriceGenerator)",                                "dual_3ch", "stage2_normsr{fold}"),
    ( 7, "Stage 2 (대조군)",                                         "dual_3ch", "singlestage_normsr{fold}"),
    # 8: MTL(2 채널, liq) = joint
    ( 8, "MTL(2 채널, liq)",                                        "dual_3ch", "joint_normsr{fold}"),
    # 9-13: MTL 2 채널, no-liq cond
    ( 9, "MTL(2 채널, vix정규) [no-liq]",                            "dual_3ch", "mtl_2ch_vix_noliq_normvix_normsr{fold}"),
    (10, "MTL(2 채널, bondpp정규) [no-liq]",                         "dual_3ch", "mtl_bp2_noliq_normbp_normsr{fold}"),
    (11, "MTL(2 채널, stockpp정규) [no-liq]",                        "dual_3ch", "mtl_pps2_noliq_normsp_normsr{fold}"),
    (12, "MTL(3 채널, bondpp+vix정규) [no-liq]",                     "dual_3ch", "mtl_bp_vix_noliq_normbp_normvix_normsr{fold}"),
    (13, "MTL(3 채널, bondpp+stockpp정규) [no-liq]",                 "dual_3ch", "mtl_bp_stockpp_noliq_normbp_normsp_normsr{fold}"),
    # 14-17: MTL 3 채널
    # 행 14 는 cond=1ch 통일 (target 에 excess_liq 있는 redundancy 제거) → 행 16 과 동일 ckpt 매핑.
    (14, "MTL(3 채널, bondpp정규) [no-liq, = 행16]",                 "dual_3ch", "mtl_bp3_noliq_normbp_normsr{fold}"),
    (15, "MTL(3 채널, liq, vix정규)",                                "dual_3ch", "mtl3_normvix_normsr{fold}"),
    (16, "MTL(3 채널, bondpp정규) [no-liq]",                         "dual_3ch", "mtl_bp3_noliq_normbp_normsr{fold}"),
    (17, "MTL(3 채널, bondpp+vix정규) [no-liq]",                     "dual_3ch", "mtl_bp_vix_noliq_normbp_normvix_normsr{fold}"),
]


def get_cond_stats(ckpt):
    s = ckpt.get("stats_train")
    if s is None:
        s = ckpt.get("stats_cond")
    if s is None:
        raise KeyError(f"ckpt 에 stats_train / stats_cond 둘 다 없음. keys={list(ckpt.keys())}")
    return s


def load_test_windows(test_csv, ckpt):
    cond_cols = ckpt["cond_cols"]
    target_cols = ckpt["target_cols"]
    config = ckpt["config"]
    L = config["total_len"]
    past_len = config["past_len"]

    df = pd.read_csv(test_csv)
    n = len(df)
    n_w = n - L + 1

    X = np.zeros((n_w, L, len(target_cols)), dtype=np.float32)
    C = np.zeros((n_w, L, len(cond_cols)),   dtype=np.float32)
    for i in range(n_w):
        X[i] = df[target_cols].iloc[i : i + L].values
        C[i] = df[cond_cols].iloc[i : i + L].values

    stats_cond = get_cond_stats(ckpt)
    cmu = np.asarray(stats_cond["mean"], dtype=np.float32)
    csd = np.asarray(stats_cond["std"],  dtype=np.float32)
    C = (C - cmu) / csd

    # 모든 정규화 채널 적용 (stats_targets_all 우선, 없으면 stats_target / stats_target_extra / stats_target_list 호환)
    stats_targets_all = ckpt.get("stats_targets_all")
    if stats_targets_all:
        for ch_key, s in stats_targets_all.items():
            ch = int(ch_key)
            X[..., ch] = (X[..., ch] - float(s["mean"])) / float(s["std"])
    else:
        # legacy 단일/이중 정규화 호환
        st = ckpt.get("stats_target")
        if st is not None:
            ch = int(st["channel"])
            X[..., ch] = (X[..., ch] - float(st["mean"])) / float(st["std"])
        st_extra = ckpt.get("stats_target_extra")
        if st_extra is not None:
            ch = int(st_extra["channel"])
            X[..., ch] = (X[..., ch] - float(st_extra["mean"])) / float(st_extra["std"])
        st_list = ckpt.get("stats_target_list")
        if st_list:
            for s in st_list:
                ch = int(s["channel"])
                X[..., ch] = (X[..., ch] - float(s["mean"])) / float(s["std"])

    mask_future_ch = ckpt.get("mask_future_ch", []) or []
    for ch in mask_future_ch:
        C[:, past_len:, ch] = 0.0

    return torch.from_numpy(X), torch.from_numpy(C), past_len, target_cols, config


def define_tail_windows(test_csv, L, past_len, percentile=10.0):
    df = pd.read_csv(test_csv)
    sp = df["sp_return"].values
    n = len(df)
    n_w = n - L + 1
    future_means = np.array([sp[i + past_len : i + L].mean() for i in range(n_w)])
    threshold = float(np.percentile(future_means, percentile))
    return future_means <= threshold, future_means, threshold


def evaluate_ckpt(ckpt_path, test_csv, tail_mask, device):
    """Returns (full_nll_sp_raw, tail_nll_sp_raw, sp_log_sigma_correction).

    Jacobian 보정: sp 가 정규화되어 있으면 raw 단위로 변환 (NLL_raw = NLL_z + log σ).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    X, C, past_len, target_cols, config = load_test_windows(test_csv, ckpt)

    n_w = X.shape[0]
    if n_w != len(tail_mask):
        raise RuntimeError(f"window 수 불일치: ckpt {n_w} vs tail_mask {len(tail_mask)}")

    model = MultiStepFAVARFlow(
        K=config["K"], d_cond=config["d_cond"], d_target=config["d_target"],
        d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"],
        time_reverse=False, use_wavelet=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        z, log_det_J, log_scale = model(X.to(device), C.to(device))
        nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale
        nll_future = nll_td[:, past_len:, :]

    # sp_return 의 Jacobian 보정 (정규화된 경우만)
    sp_log_sigma = 0.0
    stats_targets_all = ckpt.get("stats_targets_all", None)
    if stats_targets_all:
        for ch_key, s in stats_targets_all.items():
            if int(ch_key) < len(target_cols) and target_cols[int(ch_key)] == "sp_return":
                sp_log_sigma = float(np.log(float(s["std"])))
                break
    elif ckpt.get("stats_target") and target_cols[int(ckpt["stats_target"]["channel"])] == "sp_return":
        sp_log_sigma = float(np.log(float(ckpt["stats_target"]["std"])))

    if "sp_return" in target_cols:
        sp_idx = target_cols.index("sp_return")
        sp_nll_per_win = nll_future[:, :, sp_idx].mean(dim=1).cpu().numpy()
        sp_nll_per_win_raw = sp_nll_per_win + sp_log_sigma   # Jacobian 보정
        full_nll_sp_raw = float(sp_nll_per_win_raw.mean())
        tail_nll_sp_raw = float(sp_nll_per_win_raw[tail_mask].mean())
    else:
        full_nll_sp_raw = float("nan")
        tail_nll_sp_raw = float("nan")

    return full_nll_sp_raw, tail_nll_sp_raw, sp_log_sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",     default=ROOT, help="project root (homeostatic-market)")
    ap.add_argument("--out-subdir", default="result_paper_final",
                    help="ckpt 가 저장된 폴더명 (colab/<folder>/<out_subdir>/)")
    ap.add_argument("--data-folds", default=os.path.join(ROOT, "data", "folds"),
                    help="3-fold CSV 폴더 (F1/F2/F3_test.csv)")
    ap.add_argument("--percentile", type=float, default=10.0,
                    help="폭락 윈도우 분위수 (기본 10 = 하위 10%)")
    ap.add_argument("--L", type=int, default=104)
    ap.add_argument("--past-len", type=int, default=52)
    ap.add_argument("--save-csv", default=None,
                    help="raw measurements CSV path (default: <root>/result_paper_final.csv)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_csv = args.save_csv or os.path.join(args.root, "result_paper_final.csv")

    # fold 별 tail_mask 미리 계산
    fold_tail_masks = {}
    for fold in FOLDS:
        test_csv = os.path.join(args.data_folds, f"{fold}_test.csv")
        if not os.path.exists(test_csv):
            print(f"[WARN] {fold} test CSV missing: {test_csv}")
            continue
        tail_mask, _, threshold = define_tail_windows(test_csv, args.L, args.past_len, args.percentile)
        fold_tail_masks[fold] = (tail_mask, threshold, test_csv)
        n_tail = int(tail_mask.sum()); n_total = len(tail_mask)
        print(f"  [{fold}] n_tail/total = {n_tail}/{n_total}, threshold = {threshold:+.6f}")

    print(f"\n# device = {device}, percentile = {args.percentile}\n")

    # paper 표 17 행 × 3 fold × 5 seed 결과 수집
    # incremental save: 한 ckpt 평가 끝날 때마다 CSV 갱신
    rows = []
    if os.path.exists(save_csv):
        existing = pd.read_csv(save_csv).to_dict("records")
        rows.extend(existing)
        print(f"  [resume] existing rows: {len(rows)}")
    done_keys = set((r["paper_row"], r["fold"], r["seed"]) for r in rows)

    for paper_row, label, folder, prefix_tmpl in VARIANTS:
        for fold in FOLDS:
            if fold not in fold_tail_masks:
                continue
            tail_mask, threshold, test_csv = fold_tail_masks[fold]
            for seed in SEEDS:
                if (paper_row, fold, seed) in done_keys:
                    continue
                fold_suffix = f"_{fold}"
                prefix = prefix_tmpl.format(fold=fold_suffix)
                ckpt_path = os.path.join(args.root, "colab", folder, args.out_subdir,
                                         f"{prefix}_seed{seed}_best.pt")
                if not os.path.exists(ckpt_path):
                    rows.append(dict(paper_row=paper_row, label=label, fold=fold, seed=seed,
                                     full_nll_sp_raw=float("nan"), tail_nll_sp_raw=float("nan"),
                                     sp_log_sigma=float("nan"), ckpt_exists=False,
                                     ckpt_path=ckpt_path))
                else:
                    try:
                        full_sp, tail_sp, sp_log_sigma = evaluate_ckpt(
                            ckpt_path, test_csv, tail_mask, device)
                        rows.append(dict(paper_row=paper_row, label=label, fold=fold, seed=seed,
                                         full_nll_sp_raw=full_sp, tail_nll_sp_raw=tail_sp,
                                         sp_log_sigma=sp_log_sigma, ckpt_exists=True,
                                         ckpt_path=ckpt_path))
                    except Exception as e:
                        print(f"  ERROR  [{label}] fold={fold} seed={seed}: {e}")
                        rows.append(dict(paper_row=paper_row, label=label, fold=fold, seed=seed,
                                         full_nll_sp_raw=float("nan"), tail_nll_sp_raw=float("nan"),
                                         sp_log_sigma=float("nan"), ckpt_exists=True,
                                         ckpt_path=ckpt_path, error=str(e)))
                pd.DataFrame(rows).to_csv(save_csv, index=False)

    df = pd.DataFrame(rows)
    print(f"\n  saved raw measurements: {save_csv}  (rows={len(df)})")

    # 변종별 집계 (paper 표 순서, 3-fold 통합 = pooled 15 measurement)
    agg_rows = []
    for paper_row, label, _, _ in VARIANTS:
        sub = df[df["paper_row"] == paper_row]
        full_vals = sub["full_nll_sp_raw"].dropna().values
        tail_vals = sub["tail_nll_sp_raw"].dropna().values
        agg_rows.append(dict(
            paper_row=paper_row,
            label=label,
            full_median=float(np.median(full_vals)) if len(full_vals) > 0 else None,
            full_mean=  float(full_vals.mean())     if len(full_vals) > 0 else None,
            full_std=   float(full_vals.std(ddof=1)) if len(full_vals) > 1 else 0.0,
            tail_median=float(np.median(tail_vals)) if len(tail_vals) > 0 else None,
            tail_mean=  float(tail_vals.mean())     if len(tail_vals) > 0 else None,
            tail_std=   float(tail_vals.std(ddof=1)) if len(tail_vals) > 1 else 0.0,
            n=int(len(full_vals)),
        ))

    print()
    print("=" * 140)
    print(f"sp_return Test NLL  (full + tail, 3-fold pooled, raw 단위 — Jacobian 보정 포함, "
          f"폭락 = future {args.L - args.past_len}w 평균 하위 {args.percentile}%)")
    print("=" * 140)
    print(f"{'#':>3}  {'variant':45s}  "
          f"{'full median':>11s}  {'full mean ± std':>20s}  "
          f"{'tail median':>11s}  {'tail mean ± std':>20s}  {'n':>3s}")
    print("-" * 140)
    for r in agg_rows:
        if r["n"] > 0:
            print(f"{r['paper_row']:>3d}  {r['label']:45s}  "
                  f"{r['full_median']:>+11.4f}  "
                  f"{r['full_mean']:>+8.4f} ± {r['full_std']:>6.3f}     "
                  f"{r['tail_median']:>+11.4f}  "
                  f"{r['tail_mean']:>+8.4f} ± {r['tail_std']:>6.3f}     "
                  f"{r['n']:>3d}")
        else:
            print(f"{r['paper_row']:>3d}  {r['label']:45s}  "
                  f"{'—':>11s}  {'— (no ckpt or n/a)':>20s}  "
                  f"{'—':>11s}  {'— (no ckpt or n/a)':>20s}  {0:>3d}")
    print("=" * 140)

    # 변종별 집계 CSV
    agg_csv = os.path.join(args.root, "result_paper_final_agg.csv")
    pd.DataFrame(agg_rows).to_csv(agg_csv, index=False)
    print(f"\n  saved aggregated table: {agg_csv}")


if __name__ == "__main__":
    main()
