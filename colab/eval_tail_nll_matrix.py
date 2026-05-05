"""Tail-NLL 평가 — 매트릭스 ckpt (matrix_v*_best.pt) 자동 스캔.

평가 대상:
  - colab/dual_3ch/result/matrix_v{1..7}_*_F{1..3}_seed{42..46}_best.pt (105개)
  - data/folds_v33/{F1,F2,F3}_test.csv 의 future 52주 평균 sp_return 하위 10% 폭락 윈도우

평가 metric:
  - sp_return 채널 raw NLL (sp 정규화 안 함, 매트릭스 학습 시 sp raw 유지)
  - full NLL (전 test 윈도우 평균) + tail NLL (폭락 윈도우만 평균)

매트릭스 ckpt 키 (train_matrix.py 정의):
  - stats_cond:        cond z-score stats {mean, std}
  - stats_target_map:  {idx: stats}  bondpp/stockpp target 정규화 (sp_return 제외)
  - mask_future_ch:    future cond mask 채널 인덱스 list
  - variant:           {id: 1..7, name: ...}
  - config:            K, d_model, n_heads, n_layers, d_cond, d_target, past_len, total_len

사용법 (Colab):
  cd /content/drive/MyDrive/Colab\\ Notebooks/homeostatic-market
  !python colab/eval_tail_nll_matrix.py

한계:
  - n_tail per fold ≈ 7~8 (n_test 67~78 의 10%) → pooled n_tail × 5 seed 통계 약함.
  - 정규 가정 marginal 만 baseline. fat-tail empirical marginal 미측정.
  - tail 분포 형태 자체 (EMD / CVaR) 미측정 — paper main metric 별개 작업.
"""
import torch  # MUST be first (Windows DLL load order, also Linux import safety)

import argparse
import glob
import math
import os
import re
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

VARIANT_LABELS = {
    1: "1.base",
    2: "2.base_bp",
    3: "3.base_sp",
    4: "4.base_bp_sp",
    5: "5.mtl_bp",
    6: "6.mtl_sp",
    7: "7.mtl_bp_sp",
}

CKPT_NAME_RE = re.compile(r"matrix_v(\d+)_.*_(F[123])_seed(\d+)_best\.pt$")


def load_test_windows(test_csv, ckpt):
    """ckpt 의 cond_cols/target_cols/stats/mask 그대로 적용해 시험 윈도우 X, C 생성."""
    cond_cols   = ckpt["cond_cols"]
    target_cols = ckpt["target_cols"]
    config      = ckpt["config"]
    L        = config["total_len"]
    past_len = config["past_len"]

    df = pd.read_csv(test_csv)
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        raise RuntimeError(f"not enough test windows: n={n}, L={L}")

    X = np.zeros((n_w, L, len(target_cols)), dtype=np.float32)
    C = np.zeros((n_w, L, len(cond_cols)),   dtype=np.float32)
    for i in range(n_w):
        X[i] = df[target_cols].iloc[i : i + L].values
        C[i] = df[cond_cols].iloc[i : i + L].values

    # cond z-score (train stats, 항상 적용)
    sc  = ckpt["stats_cond"]
    cmu = np.asarray(sc["mean"], dtype=np.float32)
    csd = np.asarray(sc["std"],  dtype=np.float32)
    C = (C - cmu) / csd

    # target normalize (multi-channel — stats_target_map dict, sp_return 채널은 raw 유지)
    stm = ckpt.get("stats_target_map", {}) or {}
    for idx_str, st in stm.items():
        idx = int(idx_str)
        tmu = float(st["mean"])
        tsd = float(st["std"])
        X[..., idx] = (X[..., idx] - tmu) / tsd

    # mask future cond channels (paper main thesis: 금리 시나리오만 활성)
    mask_future_ch = ckpt.get("mask_future_ch", []) or []
    for ch in mask_future_ch:
        C[:, past_len:, ch] = 0.0

    return torch.from_numpy(X), torch.from_numpy(C), past_len, target_cols, config


def define_tail_windows(test_csv, L, past_len, percentile=10.0):
    """future (L-past_len)주 평균 sp_return 하위 percentile% 윈도우 인덱스."""
    df = pd.read_csv(test_csv)
    sp = df["sp_return"].values
    n = len(df)
    n_w = n - L + 1
    future_means = np.array([sp[i + past_len : i + L].mean() for i in range(n_w)])
    threshold = float(np.percentile(future_means, percentile))
    return future_means <= threshold, future_means, threshold


def evaluate_ckpt(ckpt_path, ckpt, test_csv, tail_mask, device):
    X, C, past_len, target_cols, config = load_test_windows(test_csv, ckpt)
    n_w = X.shape[0]
    if n_w != len(tail_mask):
        raise RuntimeError(f"window 수 불일치: ckpt {n_w} vs tail_mask {len(tail_mask)} "
                           f"(L={config['total_len']}, past_len={past_len})")

    model = MultiStepFAVARFlow(
        K=config["K"], d_cond=config["d_cond"], d_target=config["d_target"],
        d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"],
        time_reverse=False, use_wavelet=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        z, log_det_J, log_scale = model(X.to(device), C.to(device))
        nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale  # [B, L, D]
        nll_future = nll_td[:, past_len:, :]                # [B, F, D]

    if "sp_return" not in target_cols:
        return float("nan"), float("nan")
    sp_idx = target_cols.index("sp_return")
    sp_nll_per_win = nll_future[:, :, sp_idx].mean(dim=1).cpu().numpy()  # [B]
    full_nll_sp = float(sp_nll_per_win.mean())
    tail_nll_sp = float(sp_nll_per_win[tail_mask].mean()) if tail_mask.any() else float("nan")
    return full_nll_sp, tail_nll_sp


def parse_meta_from_filename(ckpt_path):
    """파일명에서 (variant_id, fold, seed) 추출."""
    fname = os.path.basename(ckpt_path)
    m = CKPT_NAME_RE.search(fname)
    if not m:
        return None, None, None
    return int(m.group(1)), m.group(2), int(m.group(3))


def main():
    ap = argparse.ArgumentParser(description="Tail-NLL evaluator for paper matrix ckpts (v33)")
    ap.add_argument("--root",        default=ROOT, help="project root (homeostatic-market)")
    ap.add_argument("--result-dir",  default=None,
                    help="ckpt 폴더 (기본 root/colab/dual_3ch/result)")
    ap.add_argument("--folds-dir",   default=None,
                    help="fold csv 폴더 (기본 root/data/folds_v33)")
    ap.add_argument("--out-csv",     default=None,
                    help="결과 raw csv (기본 result-dir/tail_nll_matrix.csv)")
    ap.add_argument("--percentile",  type=float, default=10.0,
                    help="폭락 윈도우 분위수 (기본 10 = 하위 10%)")
    ap.add_argument("--ckpt-glob",   default="matrix_v*_best.pt",
                    help="ckpt 파일 글롭 패턴")
    ap.add_argument("--L",           type=int, default=104, help="window length")
    ap.add_argument("--past-len",    type=int, default=52)
    args = ap.parse_args()

    result_dir = args.result_dir or os.path.join(args.root, "colab", "dual_3ch", "result")
    folds_dir  = args.folds_dir  or os.path.join(args.root, "data", "folds_v33")
    out_csv    = args.out_csv    or os.path.join(result_dir, "tail_nll_matrix.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths = sorted(glob.glob(os.path.join(result_dir, args.ckpt_glob)))
    print(f"\nFound {len(ckpt_paths)} ckpt(s) (expected 105 = 7×3×5)")
    print(f"  result_dir = {result_dir}")
    print(f"  folds_dir  = {folds_dir}")
    print(f"  device     = {device}")

    if not ckpt_paths:
        print(f"  ERROR: no ckpts at glob '{args.ckpt_glob}' in {result_dir}")
        sys.exit(1)

    # === 1. fold 별 폭락 윈도우 정의 (한 번만) ===
    print(f"\nDefining tail windows (future {args.L - args.past_len}w "
          f"평균 sp_return ≤ {args.percentile}% 분위수):")
    fold_meta = {}
    for fold in ["F1", "F2", "F3"]:
        test_csv = os.path.join(folds_dir, f"{fold}_test.csv")
        if not os.path.exists(test_csv):
            print(f"  WARN: missing {test_csv}")
            continue
        tail_mask, future_means, threshold = define_tail_windows(
            test_csv, L=args.L, past_len=args.past_len, percentile=args.percentile)
        n_tail  = int(tail_mask.sum())
        n_total = len(tail_mask)
        fold_meta[fold] = dict(
            test_csv=test_csv, tail_mask=tail_mask,
            threshold=threshold, n_tail=n_tail, n_total=n_total,
        )
        print(f"  {fold}: n_total={n_total}, n_tail={n_tail} "
              f"(threshold ≤ {threshold:+.6f})")

    # === 2. ckpt 평가 (load → forward → tail-mask average) ===
    print(f"\nEvaluating {len(ckpt_paths)} ckpts...")
    rows = []
    n_done = 0
    n_err  = 0
    for ckpt_path in ckpt_paths:
        variant_id, fold, seed = parse_meta_from_filename(ckpt_path)
        if variant_id is None:
            print(f"  SKIP unparseable filename: {os.path.basename(ckpt_path)}")
            n_err += 1
            continue
        if fold not in fold_meta:
            print(f"  SKIP missing fold meta: {fold}")
            n_err += 1
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  ERROR load {os.path.basename(ckpt_path)}: {e}")
            n_err += 1
            continue
        try:
            full_sp, tail_sp = evaluate_ckpt(
                ckpt_path, ckpt,
                fold_meta[fold]["test_csv"],
                fold_meta[fold]["tail_mask"],
                device)
        except Exception as e:
            print(f"  ERROR eval {os.path.basename(ckpt_path)}: {e}")
            n_err += 1
            continue

        rows.append(dict(
            variant_id  =variant_id,
            variant_name=VARIANT_LABELS.get(variant_id, str(variant_id)),
            fold        =fold,
            seed        =seed,
            full_nll_sp =full_sp,
            tail_nll_sp =tail_sp,
            n_tail_fold =fold_meta[fold]["n_tail"],
            ckpt        =os.path.basename(ckpt_path),
        ))
        n_done += 1
        if n_done % 20 == 0:
            print(f"  ... {n_done}/{len(ckpt_paths)} done")

    print(f"\nFinished: {n_done} ok, {n_err} err / {len(ckpt_paths)} total")

    if not rows:
        print("  ERROR: no successful evaluations")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values(["variant_id", "fold", "seed"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw run table → {out_csv}  ({len(df)} rows × {len(df.columns)} cols)")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    # === 3. Table A: variant × fold × tail_nll_sp (5 seed median ± std) ===
    print("\n" + "=" * 100)
    print("Table A — Variant × Fold × tail_nll_sp (raw, 5 seed median ± std)")
    print("=" * 100)
    def fmt_med_std(s):
        return f"{s.median():+.3f} ± {s.std():.3f}"
    table_a = df.pivot_table(index="variant_name", columns="fold",
                             values="tail_nll_sp", aggfunc=fmt_med_std)
    print(table_a.to_string())

    # === 4. Table B: pooled (n=15) per variant — full + tail ===
    print("\n" + "=" * 100)
    print("Table B — Pooled (n=15) per variant — full + tail")
    print("=" * 100)
    agg = df.groupby("variant_name").agg(
        n        =("tail_nll_sp", "count"),
        full_med =("full_nll_sp", "median"),
        full_std =("full_nll_sp", "std"),
        tail_med =("tail_nll_sp", "median"),
        tail_mean=("tail_nll_sp", "mean"),
        tail_std =("tail_nll_sp", "std"),
        tail_min =("tail_nll_sp", "min"),
        tail_max =("tail_nll_sp", "max"),
    ).round(4)
    print(agg.to_string())

    # === 5. Ranking by tail_med (낮을수록 폭락 시기 학습 효과 큼) ===
    print("\n" + "=" * 100)
    print("Ranking by tail_med (낮을수록 폭락 시기 학습 효과 큼)")
    print("=" * 100)
    rank = agg.sort_values("tail_med")[["tail_med", "tail_mean", "tail_std", "full_med", "n"]]
    print(rank.to_string())

    # === 6. Limitations (결론보다 먼저 명시) ===
    print("\n" + "=" * 100)
    print("LIMITATIONS")
    print("=" * 100)
    n_tail_total = sum(fm["n_tail"] for fm in fold_meta.values())
    print(f"  - n_tail per fold ≈ 7~8 → 5 seed × {len(fold_meta)} fold "
          f"= pooled tail 윈도우 ({n_tail_total} per seed × 5 seed) NLL 평균")
    print(f"  - 정규 가정 marginal 만 baseline. fat-tail empirical marginal 미측정.")
    print(f"  - tail 분포 자체 형태 (Earth Mover Distance / CVaR) 미측정 "
          f"— paper main metric 별개 작업.")


if __name__ == "__main__":
    main()
