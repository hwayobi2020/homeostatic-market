"""Refit-on-Full-Train (Gu-Kelly-Xiu 2020 패턴) for v14 MTL.

Step 1 (이미 끝남): standard train/val 학습 → best_epoch per seed (summary.json 에서 읽음).
Step 2 (이 스크립트): train + val 합쳐 Full_Train → 모델 초기화 → best_epoch 까지만 blind 학습
                     (Val 셋 없음, early stopping 없음).
Step 3 (이 스크립트): test 평가 + npz 저장.

cond_stats 정책 (a 옵션): Step 1 의 train_stats 그대로 재사용 (학습 dynamics 보존).
Full_Train 의 새 stats 안 만듦 — best_epoch 의 optimality 보장 위해.

Output prefix: vol_pilot_3m_mtl_refit_msel_v{id}_{name}_{fold}_seed{N}_*

Citation:
  Gu, Kelly, Xiu (2020). "Empirical Asset Pricing via Machine Learning."
  Review of Financial Studies, 33(5), 2223-2273.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_vol_pilot_3m_mtl import (   # noqa: E402
    CausalTransformerVolMTL, load_windows_mtl, mask_future_channels,
    pearson_corr, spearman_corr,
    VARIANTS, build_spec,
    PAST_LEN, FUTURE_LEN, L, D_MODEL, N_HEADS, N_LAYERS, LR, BATCH, GRAD_CLIP,
)

ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))


def load_step1_meta(variant_id, variant_name, fold, seed, result_dir):
    """Load Step 1 ckpt + summary to get best_epoch + stats."""
    pat = f"vol_pilot_3m_mtl_msel_v{variant_id}_{variant_name}_{fold}_seed{seed}"
    ckpt_path = os.path.join(result_dir, f"{pat}_best.pt")
    summary_path = os.path.join(result_dir, f"{pat}_summary.json")
    if not os.path.exists(ckpt_path) or not os.path.exists(summary_path):
        sys.exit(f"[FATAL] Step 1 ckpt/summary 없음:\n  {ckpt_path}\n  {summary_path}\n"
                 "  → 먼저 train_vol_pilot_3m_mtl.py 로 Step 1 학습 필요.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    summary = json.load(open(summary_path))
    return ckpt, summary


def make_full_train_csv(train_csv, val_csv, fold, out_dir):
    """Concat train+val CSV (시간 순서 유지)."""
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    full_csv = os.path.join(out_dir, f"_refit_full_train_{fold}.csv")
    full_df.to_csv(full_csv, index=False)
    print(f"    Full_Train CSV: {full_csv}")
    print(f"    train n={len(train_df)} + val n={len(val_df)} = full n={len(full_df)}")
    return full_csv


def run_refit(spec, train_csv, val_csv, test_csv, save_dir, seed, fold_tag, lambda_aux=1.0):
    cols_cond = spec["cols_cond"]
    cols_target = spec["cols_target"]
    aux_col = spec["aux_col"]
    mask_future = spec["mask_future_ch"]

    # 1) Load Step 1 meta (best_epoch + stats)
    ckpt_step1, summary_step1 = load_step1_meta(spec["variant_id"], spec["name"], fold_tag, seed, save_dir)
    best_epoch_step1 = int(summary_step1["best_epoch"])
    stats_c_step1 = ckpt_step1["stats_cond"]
    stats_a_step1 = ckpt_step1["stats_aux"]
    print(f"\n{'='*72}")
    print(f"[Variant {spec['variant_id']} = {spec['name']}_{fold_tag}] seed={seed}  REFIT")
    print(f"  Step 1 best_epoch = {best_epoch_step1}  (val_mse={summary_step1['val_mse']:.5f})")
    print(f"  cond stats reused from Step 1 train_stats (Gu-Kelly-Xiu 2020 패턴)")
    print(f"{'='*72}")

    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = f"vol_pilot_3m_mtl_refit_msel_v{spec['variant_id']}_{spec['name']}_{fold_tag}_seed{seed}"
    ckpt_path = os.path.join(save_dir, f"{tag}_best.pt")
    pred_path = os.path.join(save_dir, f"{tag}_test_preds.npz")
    summary_path = os.path.join(save_dir, f"{tag}_summary.json")

    if os.path.exists(ckpt_path) and os.path.exists(pred_path) and os.path.exists(summary_path):
        print(f"[SKIP] {tag} — refit 결과 이미 존재")
        return

    # 2) Make Full_Train CSV (train + val)
    full_csv = make_full_train_csv(train_csv, val_csv, fold_tag, save_dir)

    # 3) Load windows — Full_Train 위에서, Step 1 stats 재사용
    Xtr_past, Ctr, ytr, atr, _, _ = load_windows_mtl(
        full_csv, cols_cond, cols_target, aux_col, L=L,
        cond_stats=stats_c_step1, aux_stats=stats_a_step1)
    Xte_past, Cte, yte, ate, _, _ = load_windows_mtl(
        test_csv, cols_cond, cols_target, aux_col, L=L,
        cond_stats=stats_c_step1, aux_stats=stats_a_step1)
    print(f"  Full_Train windows: {Xtr_past.shape[0]}")
    print(f"  Test windows: {Xte_past.shape[0]}")

    Ctr_m = mask_future_channels(Ctr, PAST_LEN, mask_future)
    Cte_m = mask_future_channels(Cte, PAST_LEN, mask_future)

    # 4) Init model from scratch, train exactly best_epoch
    model = CausalTransformerVolMTL(
        d_cond=len(cols_cond), d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        past_len=PAST_LEN, future_len=FUTURE_LEN
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    train_ds = TensorDataset(Xtr_past, Ctr_m, ytr, atr)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)

    y_train_mean = float(ytr.mean())
    baseline_test_mse = float(((yte - y_train_mean) ** 2).mean())

    print(f"\n  Refit — blind train to epoch {best_epoch_step1} (no early stopping):")
    for epoch in range(1, best_epoch_step1 + 1):
        model.train()
        for xb, cb, yb, ab in train_dl:
            xb = xb.to(device); cb = cb.to(device); yb = yb.to(device); ab = ab.to(device)
            v_pred, a_pred = model(cb, xb)
            loss_v = ((v_pred - yb) ** 2).mean()
            loss_a = ((a_pred - ab) ** 2).mean()
            loss = loss_v + lambda_aux * loss_a
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == best_epoch_step1:
            # Quick test eval (for log only — selection 안 함)
            model.eval()
            with torch.no_grad():
                v_t, _ = model(Cte_m.to(device), Xte_past.to(device))
                v_t = v_t.cpu().numpy()
            yte_np = yte.cpu().numpy()
            test_mse = float(np.mean((yte_np - v_t) ** 2))
            test_p = pearson_corr(yte_np, v_t)
            print(f"    ep {epoch:3d}: test_mse={test_mse:.5f}  test_p={test_p:+.3f}")

    # 5) Final test prediction (use final ckpt, no selection)
    model.eval()
    with torch.no_grad():
        v_t, _ = model(Cte_m.to(device), Xte_past.to(device))
        v_t = v_t.cpu().numpy()
    yte_np = yte.cpu().numpy()
    test_mse = float(np.mean((yte_np - v_t) ** 2))
    ss_res = float(np.sum((yte_np - v_t) ** 2))
    ss_tot = float(np.sum((yte_np - yte_np.mean()) ** 2))
    test_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    test_p = pearson_corr(yte_np, v_t)
    test_s = spearman_corr(yte_np, v_t)

    print(f"\n  Refit final (epoch {best_epoch_step1}, blind):")
    print(f"    test MSE  = {test_mse:.5f}")
    print(f"    test R²   = {test_r2:+.4f}")
    print(f"    test Pearson  = {test_p:+.4f}")
    print(f"    test Spearman = {test_s:+.4f}")
    print(f"    baseline test MSE = {baseline_test_mse:.5f}")

    # 6) Save
    torch.save({
        "model_state": model.state_dict(),
        "model_type": "causal_transformer_vol_mtl_3m_refit",
        "cond_cols": cols_cond,
        "target_cols": cols_target,
        "aux_col": aux_col,
        "lambda_aux": lambda_aux,
        "stats_cond": stats_c_step1,
        "stats_aux": stats_a_step1,
        "mask_future_ch": mask_future,
        "config": dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                       d_cond=len(cols_cond), past_len=PAST_LEN,
                       future_len=FUTURE_LEN, total_len=L),
        "variant": dict(id=spec["variant_id"], name=spec["name"]),
        "refit_epoch": best_epoch_step1,
        "step1_summary_ref": os.path.basename(
            os.path.join(save_dir, f"vol_pilot_3m_mtl_msel_v{spec['variant_id']}_{spec['name']}_{fold_tag}_seed{seed}_summary.json")
        ),
    }, ckpt_path)
    np.savez(pred_path,
             y_pred_log_std=v_t,
             y_actual_log_std=yte_np,
             y_pred_std=np.exp(v_t),
             y_actual_std=np.exp(yte_np),
             baseline_test_mse=baseline_test_mse)
    print(f"  saved refit ckpt: {ckpt_path}")
    print(f"  saved refit preds: {pred_path}")

    summary = dict(
        model_type="causal_transformer_vol_mtl_3m_refit",
        variant_id=spec["variant_id"],
        variant_name=spec["name"],
        fold=fold_tag,
        seed=seed,
        cond_cols=cols_cond,
        target_cols=cols_target,
        aux_col=aux_col,
        lambda_aux=lambda_aux,
        mask_future_ch=mask_future,
        future_len=FUTURE_LEN,
        refit_epoch=best_epoch_step1,
        step1_best_epoch=best_epoch_step1,
        step1_val_mse=summary_step1["val_mse"],
        test_mse=test_mse,
        test_r2=test_r2,
        test_pearson_std=test_p,
        test_spearman_std=test_s,
        baseline_test_mse=baseline_test_mse,
        full_train_windows=int(Xtr_past.shape[0]),
        test_windows=int(Xte_past.shape[0]),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Refit-on-Full-Train for v14 MTL (Gu-Kelly-Xiu 2020)")
    ap.add_argument("--variant", type=int, default=14, choices=list(VARIANTS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--fold", default="F1")
    ap.add_argument("--folds-dir", default=os.path.join(ROOT, "data", "folds_v33_vix_expanding"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--lambda-aux", type=float, default=1.0)
    args = ap.parse_args()

    spec = build_spec(args.variant)
    train_csv = os.path.join(args.folds_dir, f"{args.fold}_train.csv")
    val_csv   = os.path.join(args.folds_dir, f"{args.fold}_val.csv")
    test_csv  = os.path.join(args.folds_dir, f"{args.fold}_test.csv")
    for p in [train_csv, val_csv, test_csv]:
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing {p}")

    print(f"\n[Refit-on-Full-Train | --fold {args.fold} --variant {args.variant}]")
    for seed in args.seeds:
        run_refit(spec, train_csv, val_csv, test_csv, args.out_dir, seed,
                  fold_tag=args.fold, lambda_aux=args.lambda_aux)


if __name__ == "__main__":
    main()
