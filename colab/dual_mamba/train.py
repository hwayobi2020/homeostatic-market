"""Dual-Mamba 학습 스크립트.

Loss = NLL_1 + λ · NLL_2  (default λ=1)
모드 : teacher_forcing (Bridge 입력에 관측 future_excess_liq 사용)

CLI:
    python train.py --tag dualmamba_K2 \
        --K 2 --d-model 64 --n-layers 2 \
        --lr 5e-4 --batch 32 --max-epochs 60 --patience 15 \
        --lambda-nll2 1.0 --val-fraction 0.15 \
        --train-csv data/weekly_ppbond_train.csv \
        --test-csv  data/weekly_ppbond_test.csv \
        --out-dir   result

산출:
    result/<tag>_best.pt           : best val_loss 체크포인트
    result/<tag>_trainlog.csv      : per-epoch log
    result/<tag>_summary.json      : 최종 요약
    result/<tag>_stats.json        : z-score 통계
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Windows MKL DLL 충돌 방지: torch 를 numpy/pandas 보다 먼저 import
import torch

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    compute_stats, save_stats, build_windows, build_dataloader,
    time_split_train_val, P_DEFAULT, F_DEFAULT, W_DEFAULT,
)
from dual_mamba import DualMambaPipeline


# ══════════════════════════════════════════════════════════════════
# 학습 / 평가 루프
# ══════════════════════════════════════════════════════════════════

def epoch_loop(model: DualMambaPipeline,
               loader,
               device: torch.device,
               optimizer=None,
               grad_clip: float = 5.0,
               lambda_nll2: float = 1.0):
    """train (optimizer 있음) 또는 eval (없음) 모드.

    Return:
        dict {nll1, nll2, total_loss}  (epoch 평균, batch-weighted)
    """
    is_train = optimizer is not None
    model.train(is_train)
    sums = {"nll1": 0.0, "nll2": 0.0, "total": 0.0, "n": 0}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            out = model(batch)
            loss = out["nll1"] + lambda_nll2 * out["nll2"]
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bsz = batch["past_cond"].shape[0]
        sums["nll1"]  += out["nll1"].item()  * bsz
        sums["nll2"]  += out["nll2"].item()  * bsz
        sums["total"] += loss.item()         * bsz
        sums["n"]     += bsz
    n = max(sums["n"], 1)
    return {"nll1": sums["nll1"] / n,
            "nll2": sums["nll2"] / n,
            "total": sums["total"] / n}


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",         type=str, default="dualmamba_K2")
    ap.add_argument("--train-csv",   type=str, default="data/weekly_ppbond_train.csv")
    ap.add_argument("--test-csv",    type=str, default="data/weekly_ppbond_test.csv")
    ap.add_argument("--out-dir",     type=str, default="result")

    # 모델
    ap.add_argument("--K",           type=int, default=2)
    ap.add_argument("--d-model",     type=int, default=64)
    ap.add_argument("--n-layers",    type=int, default=2)
    ap.add_argument("--log-scale-clamp", type=float, default=4.0)

    # 윈도우
    ap.add_argument("--P",           type=int, default=P_DEFAULT)
    ap.add_argument("--F",           type=int, default=F_DEFAULT)
    ap.add_argument("--W",           type=int, default=W_DEFAULT)

    # 학습
    ap.add_argument("--lr",          type=float, default=5e-4)
    ap.add_argument("--wd",          type=float, default=1e-4)
    ap.add_argument("--batch",       type=int, default=32)
    ap.add_argument("--max-epochs",  type=int, default=60)
    ap.add_argument("--patience",    type=int, default=15)
    ap.add_argument("--grad-clip",   type=float, default=5.0)
    ap.add_argument("--lambda-nll2", type=float, default=1.0)
    ap.add_argument("--val-fraction",type=float, default=0.15)
    ap.add_argument("--seed",        type=int, default=42)

    # 시스템
    ap.add_argument("--device",      type=str, default="auto")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    # ── 디바이스 ─────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[setup] device={device}")
    if device.type == "cuda":
        print(f"[setup] cuda name={torch.cuda.get_device_name(0)}")

    # 시드
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 데이터 ───────────────────────────────────────────────────
    here = Path(__file__).resolve().parent
    train_csv = (here / args.train_csv) if not Path(args.train_csv).is_absolute() else Path(args.train_csv)
    test_csv  = (here / args.test_csv)  if not Path(args.test_csv).is_absolute()  else Path(args.test_csv)
    out_dir   = (here / args.out_dir)   if not Path(args.out_dir).is_absolute()   else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] train_csv = {train_csv}")
    print(f"[data] test_csv  = {test_csv}")
    df_tr_full = pd.read_csv(train_csv)
    df_te      = pd.read_csv(test_csv)
    print(f"[data] train rows={len(df_tr_full)}, test rows={len(df_te)}")

    stats = compute_stats(df_tr_full)
    stats_path = out_dir / f"{args.tag}_stats.json"
    save_stats(stats, stats_path)
    print(f"[data] z-score stats saved → {stats_path}")

    win_tr_full = build_windows(df_tr_full, stats, P=args.P, F=args.F, W=args.W)
    win_te      = build_windows(df_te,      stats, P=args.P, F=args.F, W=args.W)
    print(f"[data] train windows = {win_tr_full['past_cond'].shape[0]}, "
          f"test = {win_te['past_cond'].shape[0]}")

    win_tr, win_va = time_split_train_val(win_tr_full, val_fraction=args.val_fraction)
    print(f"[data] train/val split = {win_tr['past_cond'].shape[0]} / {win_va['past_cond'].shape[0]}")

    loader_tr = build_dataloader(win_tr, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers)
    loader_va = build_dataloader(win_va, batch_size=args.batch, shuffle=False,
                                  num_workers=args.num_workers)
    loader_te = build_dataloader(win_te, batch_size=args.batch, shuffle=False,
                                  num_workers=args.num_workers)

    # ── 모델 ─────────────────────────────────────────────────────
    model = DualMambaPipeline(
        K=args.K, d_model=args.d_model, n_layers=args.n_layers,
        window=args.W, log_scale_clamp=args.log_scale_clamp,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params = {n_params:,}, K={args.K}, d_model={args.d_model}, "
          f"n_layers={args.n_layers}, window={args.W}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # ── 학습 루프 ────────────────────────────────────────────────
    log_rows = []
    best_val = float("inf")
    best_path = out_dir / f"{args.tag}_best.pt"
    no_improve = 0
    t_start = time.time()
    print(f"\n[train] start  max_epochs={args.max_epochs}  patience={args.patience}  "
          f"lambda_nll2={args.lambda_nll2}")

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        tr  = epoch_loop(model, loader_tr, device, optimizer=optimizer,
                         grad_clip=args.grad_clip, lambda_nll2=args.lambda_nll2)
        va  = epoch_loop(model, loader_va, device, optimizer=None,
                         lambda_nll2=args.lambda_nll2)
        elapsed = time.time() - t0

        improved = va["total"] < best_val - 1e-6
        if improved:
            best_val = va["total"]
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "stats": stats,
                "epoch": epoch,
                "val_loss": va["total"],
            }, best_path)
        else:
            no_improve += 1

        msg = (f"[ep {epoch:3d}/{args.max_epochs}] "
               f"tr nll1={tr['nll1']:+.4f} nll2={tr['nll2']:+.4f} tot={tr['total']:+.4f}  "
               f"va nll1={va['nll1']:+.4f} nll2={va['nll2']:+.4f} tot={va['total']:+.4f}  "
               f"({elapsed:.1f}s){'  ★' if improved else ''}")
        print(msg, flush=True)

        log_rows.append({
            "epoch": epoch,
            "tr_nll1": tr["nll1"], "tr_nll2": tr["nll2"], "tr_total": tr["total"],
            "va_nll1": va["nll1"], "va_nll2": va["nll2"], "va_total": va["total"],
            "elapsed_s": elapsed, "best": improved,
        })

        if no_improve >= args.patience:
            print(f"[train] early stop @ epoch {epoch} (patience {args.patience})")
            break

    train_time = time.time() - t_start
    print(f"\n[train] done in {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"[train] best val_loss = {best_val:.4f}  saved → {best_path}")

    # ── 학습 로그 저장 ───────────────────────────────────────────
    log_path = out_dir / f"{args.tag}_trainlog.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"[log] saved → {log_path}")

    # ── 최종 평가 (best 체크포인트 → test set) ────────────────────
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te = epoch_loop(model, loader_te, device, optimizer=None,
                    lambda_nll2=args.lambda_nll2)
    print(f"\n[test] nll1={te['nll1']:+.4f}  nll2={te['nll2']:+.4f}  total={te['total']:+.4f}")

    summary = {
        "tag": args.tag,
        "args": vars(args),
        "n_params": n_params,
        "n_windows_train": int(win_tr["past_cond"].shape[0]),
        "n_windows_val":   int(win_va["past_cond"].shape[0]),
        "n_windows_test":  int(win_te["past_cond"].shape[0]),
        "best_val_loss": best_val,
        "best_epoch": int(ckpt["epoch"]),
        "test_metrics": te,
        "train_time_s": train_time,
        "device": str(device),
    }
    summary_path = out_dir / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] saved → {summary_path}")


if __name__ == "__main__":
    main()
