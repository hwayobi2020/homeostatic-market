"""학습된 Dual-Mamba 체크포인트로 시나리오 생성 + 진단.

기능:
    1. Past 윈도우 + 사용자 미래 tbill 시나리오 → 미래 excess_liq + sp_return 샘플 생성
    2. Test set marginal 진단:
        - sp_return generated vs observed: mean/std/skew/kurt
        - excess_liq generated vs observed
    3. PIT (Probability Integral Transform) calibration:
        z 의 표준 정규성 가정 → P(Z<z_i) 의 rank histogram 이 Uniform(0,1) 인지

CLI:
    python eval.py --ckpt result/<tag>_best.pt \
        --test-csv data/weekly_ppbond_test.csv \
        --n-samples 100 --out-dir result
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    compute_stats, build_windows, build_dataloader, P_DEFAULT, F_DEFAULT, W_DEFAULT,
)
from dual_mamba import DualMambaPipeline


# ══════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════

def marginal_stats(arr: np.ndarray) -> dict:
    """[N, F] flatten 한 marginal 통계 (mean/std/skew/kurt)."""
    x = arr.flatten()
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return {"mean": float("nan"), "std": float("nan"),
                "skew": float("nan"), "kurt": float("nan"), "n": n}
    m  = x.mean()
    sd = x.std()
    skew = ((x - m) ** 3).mean() / (sd ** 3 + 1e-12)
    kurt = ((x - m) ** 4).mean() / (sd ** 4 + 1e-12) - 3.0   # excess kurt
    return {"mean": float(m), "std": float(sd),
            "skew": float(skew), "kurt": float(kurt), "n": n}


def pit_calibration(z: np.ndarray) -> dict:
    """z ~ N(0,1) 가정 → CDF rank 가 Uniform(0,1) 분포여야 정상."""
    from scipy.stats import norm  # mambapy 환경에 scipy 있다고 가정
    x = z.flatten()
    x = x[np.isfinite(x)]
    ranks = norm.cdf(x)
    return {
        "rank_mean": float(ranks.mean()),
        "rank_std":  float(ranks.std()),
        "n":         int(len(ranks)),
    }


# ══════════════════════════════════════════════════════════════════
# Main eval
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       type=str, required=True)
    ap.add_argument("--test-csv",   type=str, default="data/weekly_ppbond_test.csv")
    ap.add_argument("--out-dir",    type=str, default="result")
    ap.add_argument("--n-samples",  type=int, default=100,
                    help="윈도우당 미래 시나리오 샘플 개수")
    ap.add_argument("--P",          type=int, default=P_DEFAULT)
    ap.add_argument("--F",          type=int, default=F_DEFAULT)
    ap.add_argument("--W",          type=int, default=W_DEFAULT)
    ap.add_argument("--batch",      type=int, default=16)
    ap.add_argument("--device",     type=str, default="auto")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--max-windows",type=int, default=50,
                    help="시간 절약 위해 처음 N 윈도우만 평가 (None=전부)")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[setup] device={device}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    here     = Path(__file__).resolve().parent
    ckpt_path = (here / args.ckpt) if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    test_csv  = (here / args.test_csv) if not Path(args.test_csv).is_absolute() else Path(args.test_csv)
    out_dir   = (here / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 체크포인트 로드 ─────────────────────────────────────────
    print(f"[load] ckpt = {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    train_args = ckpt["args"]
    stats      = ckpt["stats"]
    print(f"[load] best_epoch = {ckpt['epoch']}, val_loss = {ckpt['val_loss']:.4f}")
    print(f"[load] train_args: K={train_args['K']}, d_model={train_args['d_model']}, "
          f"n_layers={train_args['n_layers']}")

    model = DualMambaPipeline(
        K=train_args["K"], d_model=train_args["d_model"],
        n_layers=train_args["n_layers"], window=args.W,
        log_scale_clamp=train_args["log_scale_clamp"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── 테스트 데이터 ────────────────────────────────────────────
    df_te = pd.read_csv(test_csv)
    win_te = build_windows(df_te, stats, P=args.P, F=args.F, W=args.W, return_dates=True)
    n_w_full = win_te["past_cond"].shape[0]
    n_w = n_w_full if args.max_windows is None else min(args.max_windows, n_w_full)
    print(f"[data] test windows = {n_w_full} (eval first {n_w})")

    # ── NLL on test (forward only) ──────────────────────────────
    print(f"\n[eval] computing NLL on test set ...")
    loader = build_dataloader(win_te, batch_size=args.batch, shuffle=False)
    sums = {"nll1": 0.0, "nll2": 0.0, "n": 0}
    z1_all, z2_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            bsz = batch["past_cond"].shape[0]
            sums["nll1"] += out["nll1"].item() * bsz
            sums["nll2"] += out["nll2"].item() * bsz
            sums["n"]    += bsz
            P = batch["past_cond"].shape[1]
            z1_all.append(out["z1"][:, P:, :].cpu().numpy())
            z2_all.append(out["z2"][:, P:, :].cpu().numpy())
    nll1_test = sums["nll1"] / sums["n"]
    nll2_test = sums["nll2"] / sums["n"]
    z1_arr = np.concatenate(z1_all, axis=0)   # [N, F, 1]
    z2_arr = np.concatenate(z2_all, axis=0)
    print(f"[eval] test NLL1={nll1_test:+.4f}  NLL2={nll2_test:+.4f}  total={nll1_test+nll2_test:+.4f}")

    pit_z1 = pit_calibration(z1_arr)
    pit_z2 = pit_calibration(z2_arr)
    print(f"[eval] PIT z1: rank_mean={pit_z1['rank_mean']:.4f} (이상 0.500)")
    print(f"[eval] PIT z2: rank_mean={pit_z2['rank_mean']:.4f}")

    # ── Sample generation ───────────────────────────────────────
    print(f"\n[gen] generating {args.n_samples} samples per window ({n_w} windows)...")
    keys_pull = ["past_cond", "future_tbill", "past_buffer_raw",
                 "past_excess_liq_obs", "past_sp_obs", "past_cum_observed"]
    sub = {k: win_te[k][:n_w].to(device) for k in keys_pull}
    obs_future_excess_liq = win_te["future_excess_liq_obs"][:n_w].cpu().numpy()
    obs_future_sp         = win_te["future_sp_obs"][:n_w].cpu().numpy()

    gen_excess_liq = np.zeros((args.n_samples, n_w, args.F, 1), dtype=np.float32)
    gen_sp         = np.zeros((args.n_samples, n_w, args.F, 1), dtype=np.float32)
    with torch.no_grad():
        for s in range(args.n_samples):
            torch.manual_seed(args.seed + s + 1)
            out = model.generate(
                past_cond=sub["past_cond"],
                future_tbill=sub["future_tbill"],
                past_buffer_raw=sub["past_buffer_raw"],
                past_excess_liq=sub["past_excess_liq_obs"],
                past_sp=sub["past_sp_obs"],
                past_cum_observed=sub["past_cum_observed"],
            )
            gen_excess_liq[s] = out["future_excess_liq"].cpu().numpy()
            gen_sp[s]         = out["future_sp"].cpu().numpy()

    # ── Marginal 통계 ───────────────────────────────────────────
    obs_eq_stats = marginal_stats(obs_future_excess_liq)
    gen_eq_stats = marginal_stats(gen_excess_liq)
    obs_sp_stats = marginal_stats(obs_future_sp)
    gen_sp_stats = marginal_stats(gen_sp)
    print(f"\n[gen] excess_liq marginals (z-score 단위):")
    print(f"      obs : mean={obs_eq_stats['mean']:+.3f} std={obs_eq_stats['std']:.3f} "
          f"skew={obs_eq_stats['skew']:+.2f} kurt={obs_eq_stats['kurt']:+.2f}  n={obs_eq_stats['n']}")
    print(f"      gen : mean={gen_eq_stats['mean']:+.3f} std={gen_eq_stats['std']:.3f} "
          f"skew={gen_eq_stats['skew']:+.2f} kurt={gen_eq_stats['kurt']:+.2f}  n={gen_eq_stats['n']}")
    print(f"\n[gen] sp_return marginals (z-score 단위):")
    print(f"      obs : mean={obs_sp_stats['mean']:+.3f} std={obs_sp_stats['std']:.3f} "
          f"skew={obs_sp_stats['skew']:+.2f} kurt={obs_sp_stats['kurt']:+.2f}  n={obs_sp_stats['n']}")
    print(f"      gen : mean={gen_sp_stats['mean']:+.3f} std={gen_sp_stats['std']:.3f} "
          f"skew={gen_sp_stats['skew']:+.2f} kurt={gen_sp_stats['kurt']:+.2f}  n={gen_sp_stats['n']}")

    # ── 저장 ────────────────────────────────────────────────────
    tag = ckpt["args"]["tag"]
    np.savez_compressed(
        out_dir / f"{tag}_eval_samples.npz",
        gen_excess_liq=gen_excess_liq, gen_sp=gen_sp,
        obs_excess_liq=obs_future_excess_liq, obs_sp=obs_future_sp,
        z1=z1_arr, z2=z2_arr,
    )
    summary = {
        "ckpt": str(ckpt_path), "best_epoch": int(ckpt["epoch"]),
        "val_loss_best": float(ckpt["val_loss"]),
        "n_windows_evaluated": int(n_w),
        "n_samples_per_window": int(args.n_samples),
        "test_nll1": nll1_test, "test_nll2": nll2_test,
        "test_nll_total": nll1_test + nll2_test,
        "pit_z1": pit_z1, "pit_z2": pit_z2,
        "marginal_excess_liq_obs": obs_eq_stats,
        "marginal_excess_liq_gen": gen_eq_stats,
        "marginal_sp_return_obs": obs_sp_stats,
        "marginal_sp_return_gen": gen_sp_stats,
    }
    summary_path = out_dir / f"{tag}_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[save] {out_dir / f'{tag}_eval_samples.npz'}")
    print(f"[save] {summary_path}")


if __name__ == "__main__":
    main()
