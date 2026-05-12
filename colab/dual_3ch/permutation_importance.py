"""Permutation feature importance for v14 MTL — single fold F1.

Breiman (2001) permutation importance:
  baseline_loss = loss(model, test_data)
  for each feature i:
    test_data_shuffled = shuffle cond[:, :, i] along window axis (시간 구조 보존,
                         window 간 매칭만 셔플 — Breiman 표준)
    shuffled_loss = loss(model, test_data_shuffled)
    Δ_i = shuffled_loss − baseline_loss

  Δ > 0: feature i 가 도움 (shuffle 시 성능 하락)
  Δ ≈ 0: feature i 의 marginal contribution 작음
  Δ < 0: feature i 가 noise/해로움 (shuffle 시 성능 향상)

Repeats:
  5 ckpts (seeds) × 10 shuffles = 50 trials per feature
  → 평균 Δ ± 표준편차

Output:
  result/permutation_importance_v14_F1.csv
  result/permutation_importance_v14_F1.json

Loss functions (DM test 와 동일):
  MSE   = mean ((y_log − y_pred_log)²)
  QLIKE = mean ((σ/σ̂)² − 2 log(σ/σ̂) − 1)   on σ-scale

Citation:
  Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from train_vol_pilot_3m_mtl import (   # noqa: E402
    CausalTransformerVolMTL, load_windows_mtl, mask_future_channels,
    PAST_LEN, FUTURE_LEN, L, D_MODEL, N_HEADS, N_LAYERS,
)


def mse_loss(y_log, y_pred_log):
    return float(np.mean((y_log - y_pred_log) ** 2))


def qlike_loss(y_log, y_pred_log):
    sigma_true = np.exp(y_log)
    sigma_pred = np.exp(y_pred_log)
    ratio = sigma_true / np.clip(sigma_pred, 1e-12, None)
    return float(np.mean(ratio ** 2 - 2.0 * np.log(np.clip(ratio, 1e-12, None)) - 1.0))


def load_v14_ckpts(fold: str):
    pattern = os.path.join(HERE, "result",
                           f"vol_pilot_3m_mtl_msel_v14_*_{fold}_seed*_best.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"[FATAL] no v14 ckpts: {pattern}")
    ckpts = []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        ckpts.append(d)
    return ckpts


def build_model_from_ckpt(ckpt, device):
    cfg = ckpt["config"]
    model = CausalTransformerVolMTL(
        d_cond=cfg["d_cond"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        past_len=cfg["past_len"], future_len=cfg["future_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_vol(model, X_past, C, device):
    """Return numpy y_pred_log (n_windows,)."""
    Xp = X_past.to(device)
    Cd = C.to(device)
    vol, _aux = model(Cd, Xp)
    return vol.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Permutation importance for v14 MTL (Breiman 2001)")
    ap.add_argument("--fold", default="F1")
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--n-shuffles", type=int, default=10,
                    help="permutation 반복 횟수 (per seed). default 10.")
    ap.add_argument("--seed", type=int, default=2026, help="random seed for shuffles")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*84}")
    print(f" Permutation Feature Importance — v14 MTL (fold={args.fold}, n_shuffles={args.n_shuffles})")
    print(f"   device: {device}")
    print(f"{'='*84}")

    # Load 5 ckpts
    ckpts = load_v14_ckpts(args.fold)
    n_seeds = len(ckpts)
    cols_cond = ckpts[0]["cond_cols"]
    cols_target = ckpts[0]["target_cols"]
    aux_col = ckpts[0]["aux_col"]
    stats_c = ckpts[0]["stats_cond"]
    stats_a = ckpts[0]["stats_aux"]
    mask_future = ckpts[0]["mask_future_ch"]
    print(f"\n[1] Loaded {n_seeds} v14 ckpts")
    print(f"    cond_cols  : {cols_cond}")
    print(f"    target_main: {cols_target}")
    print(f"    target_aux : {aux_col}")

    # Load test data
    test_csv = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    if not os.path.exists(test_csv):
        sys.exit(f"[FATAL] missing {test_csv}")
    Xte_past, Cte, yte, ate, _, _ = load_windows_mtl(
        test_csv, cols_cond, cols_target, aux_col, L=L,
        cond_stats=stats_c, aux_stats=stats_a)
    Cte_m = mask_future_channels(Cte, PAST_LEN, mask_future)
    y_log = yte.numpy()
    print(f"\n[2] Test data: n_windows={Xte_past.shape[0]}")

    # Baseline: 5 seed mean prediction
    print(f"\n[3] Baseline prediction (5 seed mean)")
    preds_base = []
    for ckpt in ckpts:
        model = build_model_from_ckpt(ckpt, device)
        preds_base.append(predict_vol(model, Xte_past, Cte_m, device))
    preds_base = np.stack(preds_base)
    y_pred_base = preds_base.mean(axis=0)
    base_mse = mse_loss(y_log, y_pred_base)
    base_qlike = qlike_loss(y_log, y_pred_base)
    print(f"    baseline MSE   = {base_mse:.5f}")
    print(f"    baseline QLIKE = {base_qlike:.5f}")

    # Permutation: per seed × per feature × per shuffle
    print(f"\n[4] Permutation importance: {len(cols_cond)} features × {n_seeds} seeds × {args.n_shuffles} shuffles")
    rng = np.random.default_rng(args.seed)
    n_w = Cte_m.shape[0]

    rows = []
    for i, fname in enumerate(cols_cond):
        d_mse_list = []
        d_qlike_list = []
        for shuf in range(args.n_shuffles):
            perm = rng.permutation(n_w)  # window 축 shuffle
            Cte_shuf = Cte_m.clone()
            Cte_shuf[:, :, i] = Cte_m[perm, :, i]
            # 5 seed prediction (per shuffle)
            preds_s = []
            for ckpt in ckpts:
                model = build_model_from_ckpt(ckpt, device)
                preds_s.append(predict_vol(model, Xte_past, Cte_shuf, device))
            preds_s = np.stack(preds_s)
            y_pred_s = preds_s.mean(axis=0)
            d_mse_list.append(mse_loss(y_log, y_pred_s) - base_mse)
            d_qlike_list.append(qlike_loss(y_log, y_pred_s) - base_qlike)
        d_mse = np.array(d_mse_list)
        d_qlike = np.array(d_qlike_list)
        row = dict(
            feature=fname,
            d_mse_mean=float(d_mse.mean()),
            d_mse_std=float(d_mse.std(ddof=1)),
            d_qlike_mean=float(d_qlike.mean()),
            d_qlike_std=float(d_qlike.std(ddof=1)),
            n_shuffles=int(args.n_shuffles),
        )
        rows.append(row)
        print(f"    [{i}] {fname:22s}  ΔMSE = {d_mse.mean():+.5f} ± {d_mse.std(ddof=1):.5f}   "
              f"ΔQLIKE = {d_qlike.mean():+.5f} ± {d_qlike.std(ddof=1):.5f}")

    # Sort by ΔMSE descending (큰 양수 = 가장 중요)
    rows_sorted = sorted(rows, key=lambda r: -r["d_mse_mean"])
    print(f"\n{'='*84}")
    print(" RANKING (sorted by ΔMSE, descending — 큰 양수일수록 중요, 음수는 해로움)")
    print(f"{'='*84}")
    print(f"  {'rank':4s}  {'feature':24s}  {'ΔMSE':>16s}     {'ΔQLIKE':>16s}     verdict")
    print("  " + "-" * 82)
    for k, r in enumerate(rows_sorted, start=1):
        verdict = "  IMPORTANT" if r["d_mse_mean"] > 0.005 else (
                  "  ≈ ZERO"    if abs(r["d_mse_mean"]) <= 0.005 else
                  "  HARMFUL ⚠ ")
        print(f"  {k:4d}  {r['feature']:24s}  "
              f"{r['d_mse_mean']:+8.5f} ± {r['d_mse_std']:.5f}   "
              f"{r['d_qlike_mean']:+8.5f} ± {r['d_qlike_std']:.5f}    {verdict}")

    # Save
    csv_path = os.path.join(HERE, "result", f"permutation_importance_v14_{args.fold}.csv")
    json_path = os.path.join(HERE, "result", f"permutation_importance_v14_{args.fold}.json")
    pd.DataFrame(rows_sorted).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(dict(fold=args.fold, n_seeds=n_seeds,
                       n_shuffles=args.n_shuffles,
                       baseline_mse=base_mse, baseline_qlike=base_qlike,
                       rows=rows_sorted), f, indent=2)
    print(f"\n  saved CSV  : {csv_path}")
    print(f"  saved JSON : {json_path}")
    print(f"{'='*84}")


if __name__ == "__main__":
    main()
