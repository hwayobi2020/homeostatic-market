"""PIT (Probability Integral Transform) KS test — latent z 분포가 N(0,1)을 따르는지 평가.

Normalizing Flow 의 표준 calibration 평가법:
  - Flow 학습 가정: z ~ N(0,1) (Gaussian base distribution)
  - 학습이 잘 되면 test data forward → z 가 진짜 N(0,1) 분포 따름
  - z 분포 vs N(0,1) KS test → 모델 calibration 점수
  - KS 작음(p 큼) = 잘 calibrated, KS 큼(p 작음) = mis-calibrated

모델 비교 = Base vs MTL 의 calibration 우위 확인.

[한계 — paper 에 명시 필요]
  - z 의 unit-of-analysis 가 [n_w × L] 점인데 시계열 의존성 있음 → KS p-value 보수적 해석
  - 표준 NF 평가법이지만 시계열 자기상관 고려 시 block bootstrap 보강 권장
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "dual_3ch"))
from favar_flow import MultiStepFAVARFlow  # noqa: E402

SEEDS = [42, 123, 777, 0, 99]
FOLDS = ["F1", "F2", "F3"]

# 핵심 3 변종 (eval_scenario_metrics.py 와 동일 명세)
VARIANTS = [
    ( 2, "Base (cond=tbill 1ch)",                      "k2_104",   "K2_104_2ch_normsr"),
    (11, "MTL stockpp정규 [no-liq] (best)",             "dual_3ch", "mtl_pps2_noliq_normsp_normsr"),
    (15, "MTL liq+vix정규 (vix-based)",                 "dual_3ch", "mtl3_normvix_normsr"),
]


def get_cond_stats(ckpt):
    s = ckpt.get("stats_train")
    if s is None:
        s = ckpt.get("stats_cond")
    if s is None:
        raise KeyError("ckpt 에 stats_train / stats_cond 둘 다 없음")
    return s


def load_test_data(test_csv, ckpt):
    """test_csv → sliding window X (target), C (cond) — ckpt 의 stats 로 정규화 적용."""
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
    s_cond = get_cond_stats(ckpt)
    cmu = np.asarray(s_cond["mean"], dtype=np.float32)
    csd = np.asarray(s_cond["std"],  dtype=np.float32)
    C = (C - cmu) / csd

    # target normalize (정규화된 채널만)
    stats_targets_all = ckpt.get("stats_targets_all") or {}
    if not stats_targets_all:
        st = ckpt.get("stats_target")
        if st is not None:
            stats_targets_all = {int(st["channel"]): st}
    for ch_key, s in stats_targets_all.items():
        ch = int(ch_key)
        X[..., ch] = (X[..., ch] - float(s["mean"])) / float(s["std"])

    # mask future cond channels (학습 시와 동일)
    mask_future_ch = ckpt.get("mask_future_ch", []) or []
    for ch in mask_future_ch:
        C[:, past_len:, ch] = 0.0

    return torch.from_numpy(X), torch.from_numpy(C), past_len, target_cols, config


@torch.no_grad()
def compute_z(model, X, C, device):
    """Forward → z [n_w, L, D_target]."""
    z, _, _ = model(X.to(device), C.to(device))
    return z.cpu().numpy()


def pit_ks_sp_channel(z, target_cols):
    """sp_return 채널 z 분포 vs N(0,1) KS test (paper main 관심).
    Returns dict.
    """
    if "sp_return" not in target_cols:
        return None
    ch = target_cols.index("sp_return")
    z_flat = z[..., ch].reshape(-1)   # [n_w * L]
    # vs theoretical N(0,1) (kstest one-sample)
    ks_stat, ks_p = stats.kstest(z_flat, 'norm')
    return dict(
        n=int(len(z_flat)),
        z_mean=float(z_flat.mean()),
        z_std=float(z_flat.std(ddof=1)),
        ks_stat=float(ks_stat),
        ks_p=float(ks_p),
    )


def process_one_ckpt(ckpt_path, test_csv, device):
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    X, C, past_len, target_cols, config = load_test_data(test_csv, ckpt)
    if "sp_return" not in target_cols:
        return None

    model = MultiStepFAVARFlow(
        K=config["K"], d_cond=config["d_cond"], d_target=config["d_target"],
        d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"],
        time_reverse=False, use_wavelet=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    z = compute_z(model, X, C, device)
    return pit_ks_sp_channel(z, target_cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out-subdir", default="result_paper_final")
    ap.add_argument("--data-folds", default=os.path.join(ROOT, "data", "folds"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu 권장 (P1 시나리오 생성 GPU 와 충돌 회피)")
    ap.add_argument("--save-csv", default=None,
                    help="raw CSV path (default: <root>/result_pit_ks.csv)")
    args = ap.parse_args()

    save_csv = args.save_csv or os.path.join(args.root, "result_pit_ks.csv")
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"device = {device}\n")

    rows = []
    for paper_row, label, folder, prefix in VARIANTS:
        print(f"=== 행 {paper_row}: {label} ({prefix}) ===")
        for fold in FOLDS:
            test_csv = os.path.join(args.data_folds, f"{fold}_test.csv")
            for seed in SEEDS:
                ckpt_path = os.path.join(args.root, "colab", folder, args.out_subdir,
                                         f"{prefix}_{fold}_seed{seed}_best.pt")
                m = process_one_ckpt(ckpt_path, test_csv, device)
                if m is None:
                    print(f"  [missing] {os.path.basename(ckpt_path)}")
                    rows.append(dict(paper_row=paper_row, label=label, fold=fold, seed=seed,
                                     n=0, z_mean=float('nan'), z_std=float('nan'),
                                     ks_stat=float('nan'), ks_p=float('nan')))
                    continue
                rows.append(dict(paper_row=paper_row, label=label, fold=fold, seed=seed, **m))
                print(f"  {fold}_seed{seed}: KS={m['ks_stat']:.4f} p={m['ks_p']:.3e} "
                      f"z_mean={m['z_mean']:+.3f} z_std={m['z_std']:.3f}")
        print()

    df = pd.DataFrame(rows)
    df.to_csv(save_csv, index=False)
    print(f"saved raw: {save_csv}  (rows={len(df)})")

    # 변종별 집계
    print("\n" + "=" * 120)
    print(f"{'#':>3}  {'variant':<55s}  {'KS median':>10s}  {'KS mean ± std':>22s}  "
          f"{'KS_p med':>11s}  {'n':>3s}")
    print("=" * 120)
    for paper_row, label, _, _ in VARIANTS:
        sub = df[df["paper_row"] == paper_row].dropna(subset=["ks_stat"])
        if len(sub) == 0:
            print(f"{paper_row:>3d}  {label:<55s}  (no data)")
            continue
        ks_med  = float(sub["ks_stat"].median())
        ks_mean = float(sub["ks_stat"].mean())
        ks_std  = float(sub["ks_stat"].std(ddof=1)) if len(sub) > 1 else 0.0
        p_med   = float(sub["ks_p"].median())
        print(f"{paper_row:>3d}  {label:<55s}  {ks_med:>10.4f}  "
              f"{ks_mean:>+10.4f} ± {ks_std:>8.4f}  {p_med:>11.3e}  {len(sub):>3d}")
    print("=" * 120)


if __name__ == "__main__":
    main()
