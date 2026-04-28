"""Dual-Mamba 학습 스크립트.

학습 모드:
    joint       : loss = NLL_1 + λ·NLL_2, 한 옵티마이저, 단일 early stop
                  (teacher_forcing 에선 stage 간 gradient 단절이라 사실상 단일 시점 정지의 두 단일모델)
    sequential  : Phase A (Stage 1 단독, NLL_1 만, 자기 early stop)
                  → Phase B (Stage 1 freeze, Stage 2 단독, NLL_2 만, 자기 early stop)
                  각 stage 가 자기 val 최적 epoch 까지 학습 가능

CLI 예:
    python train.py --tag dualmamba_seq \
        --training-mode sequential \
        --K 2 --d-model 64 --n-layers 2 \
        --lr 5e-4 --wd 1e-3 --batch 64 \
        --max-epochs 60 --patience 15 \
        --val-fraction 0.15 \
        --train-csv data/weekly_ppbond_train.csv \
        --test-csv  data/weekly_ppbond_test.csv \
        --out-dir   result

산출:
    result/<tag>_best.pt           : 최종 체크포인트 (joint best 또는 Stage1 best + Stage2 best 합)
    result/<tag>_trainlog.csv      : per-epoch log (sequential 은 phase 컬럼 추가)
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
               lambda_nll2: float = 1.0,
               loss_mode: str = "joint"):
    """train (optimizer 있음) 또는 eval (없음) 모드.

    loss_mode:
        "joint"   : loss = nll1 + λ·nll2  (양쪽 stage 함께 업데이트)
        "stage1"  : loss = nll1          (Stage 1 단독 학습용)
        "stage2"  : loss = nll2          (Stage 2 단독 학습용, Stage 1 freeze 가정)

    Return:
        dict {nll1, nll2, total}  (epoch 평균, batch-weighted)
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
            if loss_mode == "joint":
                loss = out["nll1"] + lambda_nll2 * out["nll2"]
            elif loss_mode == "stage1":
                loss = out["nll1"]
            elif loss_mode == "stage2":
                loss = out["nll2"]
            else:
                raise ValueError(f"Unknown loss_mode: {loss_mode}")
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


def train_phase(model: DualMambaPipeline,
                loader_tr, loader_va,
                device: torch.device,
                params,                       # which params optimizer updates
                loss_mode: str,               # "stage1" | "stage2" | "joint"
                track_metric: str,            # "nll1" | "nll2" | "total"
                lr: float, wd: float, grad_clip: float,
                max_epochs: int, patience: int,
                lambda_nll2: float = 1.0,
                phase_name: str = "phase",
                ckpt_path: Path | None = None):
    """단일 학습 phase. 최선 val 의 metric 달성 시 ckpt_path (state_dict 만) 저장.

    Return: (log_rows, best_metric, best_epoch)
    """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    log_rows = []
    best_metric = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr = epoch_loop(model, loader_tr, device, optimizer=optimizer,
                        grad_clip=grad_clip, lambda_nll2=lambda_nll2,
                        loss_mode=loss_mode)
        va = epoch_loop(model, loader_va, device, optimizer=None,
                        lambda_nll2=lambda_nll2, loss_mode=loss_mode)
        elapsed = time.time() - t0

        improved = va[track_metric] < best_metric - 1e-6
        if improved:
            best_metric = va[track_metric]
            best_epoch = epoch
            no_improve = 0
            if ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        msg = (f"[{phase_name} ep {epoch:3d}/{max_epochs}] "
               f"tr nll1={tr['nll1']:+.4f} nll2={tr['nll2']:+.4f}  "
               f"va nll1={va['nll1']:+.4f} nll2={va['nll2']:+.4f}  "
               f"track={va[track_metric]:+.4f} "
               f"({elapsed:.1f}s){'  ★' if improved else ''}")
        print(msg, flush=True)

        log_rows.append({
            "phase": phase_name, "epoch": epoch,
            "tr_nll1": tr["nll1"], "tr_nll2": tr["nll2"], "tr_total": tr["total"],
            "va_nll1": va["nll1"], "va_nll2": va["nll2"], "va_total": va["total"],
            "track_metric": va[track_metric], "elapsed_s": elapsed, "best": improved,
        })

        if no_improve >= patience:
            print(f"[{phase_name}] early stop @ epoch {epoch} (patience {patience})")
            break

    return log_rows, best_metric, best_epoch


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
    ap.add_argument("--training-mode", type=str, default="joint",
                    choices=["joint", "sequential"],
                    help="joint: NLL_1 + λ·NLL_2 한 번에 / sequential: Stage1 단독 → freeze → Stage2 단독")
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

    # ── 학습 ─────────────────────────────────────────────────────
    best_path = out_dir / f"{args.tag}_best.pt"
    t_start = time.time()
    all_log_rows = []
    best_summary: dict = {}

    if args.training_mode == "joint":
        print(f"\n[train] mode=joint  max_epochs={args.max_epochs}  "
              f"patience={args.patience}  lambda_nll2={args.lambda_nll2}")

        # state_dict 저장용 임시 path (전체 model state)
        tmp_state_path = out_dir / f"{args.tag}_joint_state.pt"
        log_rows, best_metric, best_epoch = train_phase(
            model, loader_tr, loader_va, device,
            params=model.parameters(),
            loss_mode="joint", track_metric="total",
            lr=args.lr, wd=args.wd, grad_clip=args.grad_clip,
            max_epochs=args.max_epochs, patience=args.patience,
            lambda_nll2=args.lambda_nll2,
            phase_name="joint",
            ckpt_path=tmp_state_path,
        )
        all_log_rows.extend(log_rows)

        # state 복원 후 full ckpt 저장
        model.load_state_dict(torch.load(tmp_state_path, map_location=device))
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args), "stats": stats,
            "epoch": best_epoch, "val_loss": best_metric,
        }, best_path)
        tmp_state_path.unlink()

        best_summary = {
            "training_mode": "joint",
            "best_val_loss": best_metric, "best_epoch": best_epoch,
        }
        print(f"\n[train] joint done. best val_total = {best_metric:.4f} @ ep {best_epoch}")

    elif args.training_mode == "sequential":
        print(f"\n[train] mode=sequential  max_epochs={args.max_epochs}  "
              f"patience={args.patience}")

        # ── Phase A: Stage 1 단독 학습 ────────────────────────────
        stage1_state_path = out_dir / f"{args.tag}_stage1_state.pt"
        print(f"\n[Phase A] Stage 1 (MacroExpander) 단독 학습 시작")
        print(f"          loss = NLL_1, track = va_nll1, params = {sum(p.numel() for p in model.stage1.parameters()):,}")

        log_rows_a, best_nll1, best_ep_a = train_phase(
            model, loader_tr, loader_va, device,
            params=model.stage1.parameters(),
            loss_mode="stage1", track_metric="nll1",
            lr=args.lr, wd=args.wd, grad_clip=args.grad_clip,
            max_epochs=args.max_epochs, patience=args.patience,
            phase_name="A",
            ckpt_path=stage1_state_path,
        )
        all_log_rows.extend(log_rows_a)
        print(f"\n[Phase A] done. best va_nll1 = {best_nll1:+.4f} @ ep {best_ep_a}")

        # Stage 1 best 복원 + freeze
        model.load_state_dict(torch.load(stage1_state_path, map_location=device))
        for p in model.stage1.parameters():
            p.requires_grad = False
        print(f"\n[Stage 1 freeze] requires_grad=False applied to {sum(p.numel() for p in model.stage1.parameters()):,} params")

        # ── Phase B: Stage 2 단독 학습 ────────────────────────────
        stage2_state_path = out_dir / f"{args.tag}_stage2_state.pt"
        print(f"\n[Phase B] Stage 2 (PriceGenerator) 단독 학습 시작 (Stage 1 freeze)")
        print(f"          loss = NLL_2, track = va_nll2, params = {sum(p.numel() for p in model.stage2.parameters()):,}")

        log_rows_b, best_nll2, best_ep_b = train_phase(
            model, loader_tr, loader_va, device,
            params=model.stage2.parameters(),
            loss_mode="stage2", track_metric="nll2",
            lr=args.lr, wd=args.wd, grad_clip=args.grad_clip,
            max_epochs=args.max_epochs, patience=args.patience,
            phase_name="B",
            ckpt_path=stage2_state_path,
        )
        all_log_rows.extend(log_rows_b)
        print(f"\n[Phase B] done. best va_nll2 = {best_nll2:+.4f} @ ep {best_ep_b}")

        # 최종: Stage 1 best (이미 frozen 적용됨) + Stage 2 best 합쳐서 저장
        # stage2_state_path 는 Phase B 의 best 시점 model state 전체.
        # Stage 1 부분은 frozen 동안 고정이라 변화 없음 → 이대로 사용.
        model.load_state_dict(torch.load(stage2_state_path, map_location=device))
        # requires_grad 복원 (다음 사용 시 영향 없도록)
        for p in model.stage1.parameters():
            p.requires_grad = True
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args), "stats": stats,
            "epoch_phase_a": best_ep_a, "epoch_phase_b": best_ep_b,
            "val_nll1": best_nll1, "val_nll2": best_nll2,
        }, best_path)
        stage1_state_path.unlink()
        stage2_state_path.unlink()

        best_summary = {
            "training_mode": "sequential",
            "best_val_nll1": best_nll1, "best_epoch_phase_a": best_ep_a,
            "best_val_nll2": best_nll2, "best_epoch_phase_b": best_ep_b,
        }
    else:
        raise ValueError(f"Unknown training-mode: {args.training_mode}")

    train_time = time.time() - t_start
    print(f"\n[train] total time {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"[train] saved → {best_path}")

    # ── 학습 로그 저장 ───────────────────────────────────────────
    log_path = out_dir / f"{args.tag}_trainlog.csv"
    pd.DataFrame(all_log_rows).to_csv(log_path, index=False)
    print(f"[log] saved → {log_path}")

    # ── 최종 평가 (best 체크포인트 → test set) ────────────────────
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te = epoch_loop(model, loader_te, device, optimizer=None,
                    lambda_nll2=args.lambda_nll2, loss_mode="joint")
    print(f"\n[test] nll1={te['nll1']:+.4f}  nll2={te['nll2']:+.4f}  total={te['total']:+.4f}")

    summary = {
        "tag": args.tag,
        "args": vars(args),
        "n_params": n_params,
        "n_windows_train": int(win_tr["past_cond"].shape[0]),
        "n_windows_val":   int(win_va["past_cond"].shape[0]),
        "n_windows_test":  int(win_te["past_cond"].shape[0]),
        **best_summary,
        "test_metrics": te,
        "train_time_s": train_time,
        "device": str(device),
    }
    summary_path = out_dir / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] saved → {summary_path}")


if __name__ == "__main__":
    main()
