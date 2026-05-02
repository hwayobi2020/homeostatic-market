"""Tail-NLL 평가 — 폭락 윈도우 (future 52주 평균 sp_return 하위 10%) 의 sp_return 채널 NLL.

15 변종 × 5 seed ckpt 자동 스캔 → 시험 데이터 reload → 폭락 윈도우 NLL 측정.
콜라브 셀에 표만 출력 (파일 저장 X).

폭락 정의: future 52주 평균 sp_return 분포의 하위 10% 윈도우.

ckpt 키 호환:
  - 구 패턴: stats_train (cond stats), mask_future_ch optional
  - 신 패턴: stats_cond + stats_target (target normalize 변종)

Stage 1 / Stage 1 (vix) 변종은 sp_return 이 target 에 없어 tail_nll_sp = NaN.

사용법:
  !python /content/drive/MyDrive/Colab\\ Notebooks/homeostatic-market/colab/eval_tail_nll.py
"""
import torch  # MUST be first

import argparse
import math
import os
import sys
import warnings

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

# 표 순서 (사용자 paper 보고용 — 사진의 순서 그대로)
VARIANTS = [
    # (label,                              folder,     prefix)
    ("Base (K2_pure_ppbond)",              "k2_104",   "K2_104"),
    ("Base (without liquidity)",           "k2_104",   "K2_104_2ch"),
    ("Base (add bondpp)",                  "k2_104",   "K2_104_addbp"),
    ("Stage 1 (MacroExpander)",            "dual_3ch", "stage1"),
    ("Stage 1 (vix)",                      "dual_3ch", "stage1vix"),
    ("Stage 2 (PriceGenerator)",           "dual_3ch", "stage2"),
    ("Stage 2 (대조군)",                    "dual_3ch", "singlestage"),
    ("MTL(2 채널, liq)",                    "dual_3ch", "joint"),
    ("MTL(2 채널, vix)",                    "dual_3ch", "mtl_2ch_vix"),
    ("MTL(2 채널, bondpp_3m)",              "dual_3ch", "mtl_bp2"),
    ("MTL(3 채널, bondpp_3m, vix)",         "dual_3ch", "mtl_bp_vix"),
    ("MTL(2 채널, bondpp_3m정규)",          "dual_3ch", "mtl_bp2_normbp"),
    ("MTL(3 채널, liq, bondpp_3m)",         "dual_3ch", "mtl_bp3"),
    ("MTL(3 채널, liq, bondpp_3m정규)",     "dual_3ch", "mtl_bp3_normbp"),
    ("MTL(3 채널, liq, vix)",               "dual_3ch", "mtl3"),
    ("MTL(2 채널, stockpp_3m)",             "dual_3ch", "mtl_pps2"),
    # liq를 cond에서 제거한 mtl_bp3 변종 (cond=[tbill_wr] 1ch only)
    ("MTL(3 채널, bondpp_3m) [no-liq cond]",        "dual_3ch", "mtl_bp3_noliq"),
    ("MTL(3 채널, bondpp_3m정규) [no-liq cond]",    "dual_3ch", "mtl_bp3_noliq_normbp"),
]


def get_cond_stats(ckpt):
    """ckpt 의 cond stats — stats_train 또는 stats_cond 호환."""
    s = ckpt.get("stats_train")
    if s is None:
        s = ckpt.get("stats_cond")
    if s is None:
        raise KeyError(f"ckpt 에 stats_train / stats_cond 둘 다 없음. keys={list(ckpt.keys())}")
    return s


def load_test_windows(test_csv, ckpt):
    """ckpt 안의 cond_cols / target_cols / stats / mask 그대로 적용해 시험 윈도우 X, C 생성."""
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

    # cond z-score (train stats)
    stats_cond = get_cond_stats(ckpt)
    cmu = np.asarray(stats_cond["mean"], dtype=np.float32)
    csd = np.asarray(stats_cond["std"],  dtype=np.float32)
    C = (C - cmu) / csd

    # target normalize (선택 — bondpp / vix 변종)
    stats_target = ckpt.get("stats_target", None)
    if stats_target is not None:
        ch = int(stats_target["channel"])
        tmu = float(stats_target["mean"])
        tsd = float(stats_target["std"])
        X[..., ch] = (X[..., ch] - tmu) / tsd

    # mask future channels (변종에 따라 없을 수도)
    mask_future_ch = ckpt.get("mask_future_ch", []) or []
    for ch in mask_future_ch:
        C[:, past_len:, ch] = 0.0

    return torch.from_numpy(X), torch.from_numpy(C), past_len, target_cols, config


def define_tail_windows(test_csv, L, past_len, percentile=10.0):
    """future (L-past_len)주 평균 sp_return 하위 percentile% 윈도우 인덱스 (bool array)."""
    df = pd.read_csv(test_csv)
    sp = df["sp_return"].values
    n = len(df)
    n_w = n - L + 1
    future_means = np.array([sp[i + past_len : i + L].mean() for i in range(n_w)])
    threshold = float(np.percentile(future_means, percentile))
    return future_means <= threshold, future_means, threshold


def evaluate_ckpt(ckpt_path, test_csv, tail_mask, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    X, C, past_len, target_cols, config = load_test_windows(test_csv, ckpt)

    n_w = X.shape[0]
    if n_w != len(tail_mask):
        raise RuntimeError(f"window 수 불일치: ckpt {n_w} vs tail_mask {len(tail_mask)} "
                           f"(L={config['total_len']}, past_len={past_len})")

    model = MultiStepFAVARFlow(
        K=config["K"],
        d_cond=config["d_cond"],
        d_target=config["d_target"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        time_reverse=False,
        use_wavelet=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        z, log_det_J, log_scale = model(X.to(device), C.to(device))
        nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale  # [B, L, D]
        nll_future = nll_td[:, past_len:, :]                 # [B, F, D]

    # sp_return 채널 — target 에 있을 때만
    if "sp_return" in target_cols:
        sp_idx = target_cols.index("sp_return")
        sp_nll_per_win = nll_future[:, :, sp_idx].mean(dim=1).cpu().numpy()  # [B]
        full_nll_sp = float(sp_nll_per_win.mean())
        tail_nll_sp = float(sp_nll_per_win[tail_mask].mean())
    else:
        full_nll_sp = float("nan")
        tail_nll_sp = float("nan")

    return full_nll_sp, tail_nll_sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",     default=ROOT, help="project root (homeostatic-market)")
    ap.add_argument("--test-csv", default=os.path.join(ROOT, "data", "weekly_ppbond_test.csv"))
    ap.add_argument("--percentile", type=float, default=10.0,
                    help="폭락 윈도우 분위수 (기본 10 = 하위 10%)")
    ap.add_argument("--L", type=int, default=104, help="window length (모든 변종 동일)")
    ap.add_argument("--past-len", type=int, default=52)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 폭락 윈도우 정의 (한 번만)
    tail_mask, future_means, threshold = define_tail_windows(
        args.test_csv, args.L, args.past_len, args.percentile)
    n_tail  = int(tail_mask.sum())
    n_total = len(tail_mask)
    print(f"\n# 폭락 정의: future {args.L - args.past_len}주 평균 sp_return 하위 "
          f"{args.percentile}%  |  threshold = {threshold:+.6f}  |  "
          f"n_tail/total = {n_tail}/{n_total}  |  device = {device}\n")

    rows = []
    for label, folder, prefix in VARIANTS:
        for seed in SEEDS:
            ckpt_path = os.path.join(args.root, "colab", folder, "result",
                                     f"{prefix}_seed{seed}_best.pt")
            if not os.path.exists(ckpt_path):
                rows.append(dict(variant=label, seed=seed, tail_nll_sp=float("nan")))
                continue
            try:
                _, tail_sp = evaluate_ckpt(ckpt_path, args.test_csv, tail_mask, device)
                rows.append(dict(variant=label, seed=seed, tail_nll_sp=tail_sp))
            except Exception as e:
                rows.append(dict(variant=label, seed=seed, tail_nll_sp=float("nan")))
                print(f"  ERROR  [{label}] seed={seed}: {e}")

    df = pd.DataFrame(rows)

    # 변종별 집계 (표 순서 유지)
    agg_rows = []
    for label, _, _ in VARIANTS:
        sp_vals = df[df["variant"] == label]["tail_nll_sp"].dropna().values
        if len(sp_vals) > 0:
            agg_rows.append(dict(
                variant=label,
                median=float(np.median(sp_vals)),
                mean=float(sp_vals.mean()),
                std=float(sp_vals.std(ddof=1)) if len(sp_vals) > 1 else 0.0,
                n=int(len(sp_vals)),
            ))
        else:
            agg_rows.append(dict(variant=label, median=None, mean=None, std=None, n=0))

    # paper-style 표 — 콜라브 셀에 깔끔히 출력
    title = (f"Tail-NLL  (sp_return 채널, 폭락 윈도우 = future "
             f"{args.L - args.past_len}w 평균 하위 {args.percentile}%, n_tail={n_tail})")
    print("=" * 90)
    print(title)
    print("=" * 90)
    head = f"{'#':>3}  {'variant':40s}  {'median':>10s}  {'mean ± std':>20s}  {'n':>3s}"
    print(head)
    print("-" * 90)
    for i, r in enumerate(agg_rows, 1):
        if r["n"] > 0:
            print(f"{i:>3d}  {r['variant']:40s}  "
                  f"{r['median']:>+10.4f}  "
                  f"{r['mean']:>+8.4f} ± {r['std']:>6.3f}     "
                  f"{r['n']:>3d}")
        else:
            print(f"{i:>3d}  {r['variant']:40s}  {'—':>10s}  {'— (no ckpt)':>20s}  {0:>3d}")
    print("=" * 90)


if __name__ == "__main__":
    main()
