"""Marginal CVaR_5% — sp_return 채널의 future 52주 분포 lower tail 평가.

각 ckpt 마다:
  - past 52주 z 추출 (forward 1회, future 자리는 0-padding)
  - future 52주 z ~ N(0, I), 윈도우당 N=200 samples
  - inverse(z, c) → 생성 sp_return future 시퀀스
  - 모든 (window × sample × time) flatten → empirical CVaR_alpha
        = mean of values <= alpha-quantile

비교 대상: 시험 기간 future 52주 sp_return 실측 분포의 동일 정의 CVaR.

ckpt 키 호환 (eval_tail_nll.py 와 동일):
  - 구 패턴: stats_train (cond stats), mask_future_ch optional
  - 신 패턴: stats_cond + stats_target (target normalize 변종)

Stage 1 / Stage 1 (vix) 변종은 sp_return 이 target 에 없어 gen_cvar = NaN.

표 순서·라벨은 eval_tail_nll.py 와 paper 표에 맞춰 16 변종 × 5 seed.

사용법 (Colab):
  !python /content/drive/MyDrive/Colab\\ Notebooks/homeostatic-market/colab/eval_cvar.py
  !python ... /eval_cvar.py --alpha 0.10 --n-samples 100   # 옵션 변경
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

SEEDS = [42, 123, 777, 0, 99]

# 표 순서 — paper 표 / eval_tail_nll.py 와 동일
VARIANTS = [
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
]


def get_cond_stats(ckpt):
    s = ckpt.get("stats_train")
    if s is None:
        s = ckpt.get("stats_cond")
    if s is None:
        raise KeyError(f"ckpt 에 stats_train / stats_cond 둘 다 없음. keys={list(ckpt.keys())}")
    return s


def load_test_windows(test_csv, ckpt):
    """eval_tail_nll.py 와 동일 — ckpt 안의 cond_cols / target_cols / stats / mask 그대로 적용."""
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

    stats_target = ckpt.get("stats_target", None)
    if stats_target is not None:
        ch = int(stats_target["channel"])
        tmu = float(stats_target["mean"])
        tsd = float(stats_target["std"])
        X[..., ch] = (X[..., ch] - tmu) / tsd

    mask_future_ch = ckpt.get("mask_future_ch", []) or []
    for ch in mask_future_ch:
        C[:, past_len:, ch] = 0.0

    return torch.from_numpy(X), torch.from_numpy(C), past_len, target_cols, config


def compute_real_cvar(test_csv, L, past_len, alpha):
    """시험 기간 future (L - past_len)주 sp_return raw 분포의 empirical CVaR_alpha."""
    df = pd.read_csv(test_csv)
    sp = df["sp_return"].values
    n = len(df)
    n_w = n - L + 1
    pool = np.concatenate([sp[i + past_len : i + L] for i in range(n_w)])
    threshold = np.quantile(pool, alpha)
    cvar = pool[pool <= threshold].mean()
    return float(cvar), float(threshold), int(len(pool))


def evaluate_ckpt(ckpt_path, test_csv, alpha, n_samples, batch_per_window, device):
    """ckpt 의 sp_return 채널 marginal CVaR_alpha (생성 분포).

    past-z swap (favar_flow.conditional_generate_favar 와 동일 로직):
      x_seed = [x_past, 0_pad]  → forward → z_full → z_past_seed = z_full[:, :P, :]
      z_future ~ N(0, I) (윈도우당 n_samples 개)
      z_new = [z_past_seed.expand(N), z_future]
      x_gen = inverse(z_new, c.expand(N))
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    X, C, past_len, target_cols, config = load_test_windows(test_csv, ckpt)

    if "sp_return" not in target_cols:
        return float("nan")
    sp_idx = target_cols.index("sp_return")
    L = config["total_len"]
    F = L - past_len

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

    # sp_return 정규화 여부 (거의 없지만 안전 처리)
    stats_target = ckpt.get("stats_target", None)
    sp_is_normed = (stats_target is not None and int(stats_target["channel"]) == sp_idx)
    if sp_is_normed:
        sp_mu = float(stats_target["mean"])
        sp_sd = float(stats_target["std"])

    n_w, _, D = X.shape
    chunks = []

    with torch.no_grad():
        for w in range(n_w):
            x_w = X[w:w+1].to(device)        # [1, L, D]
            c_w = C[w:w+1].to(device)        # [1, L, d_cond]
            # past z 추출 (zero-pad future)
            x_seed = torch.cat(
                [x_w[:, :past_len, :], x_w.new_zeros(1, F, D)], dim=1)
            z_full_seed, _, _ = model(x_seed, c_w)
            z_past_seed = z_full_seed[:, :past_len, :]   # [1, P, D]

            n_done = 0
            while n_done < n_samples:
                bs = min(batch_per_window, n_samples - n_done)
                z_past_b   = z_past_seed.expand(bs, -1, -1)
                z_future_b = torch.randn(bs, F, D, device=device)
                z_new      = torch.cat([z_past_b, z_future_b], dim=1)
                c_b        = c_w.expand(bs, -1, -1)
                x_gen      = model.inverse(z_new, c_b)              # [bs, L, D]
                sp_future  = x_gen[:, past_len:, sp_idx].detach().cpu().numpy()
                if sp_is_normed:
                    sp_future = sp_future * sp_sd + sp_mu
                chunks.append(sp_future.reshape(-1))
                n_done += bs

    pool = np.concatenate(chunks)
    threshold = np.quantile(pool, alpha)
    cvar = pool[pool <= threshold].mean()
    return float(cvar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",     default=ROOT, help="project root (homeostatic-market)")
    ap.add_argument("--test-csv", default=os.path.join(ROOT, "data", "weekly_ppbond_test.csv"))
    ap.add_argument("--alpha",    type=float, default=0.05,
                    help="CVaR 분위수 (기본 0.05 = 하위 5%)")
    ap.add_argument("--n-samples", type=int,  default=200,
                    help="윈도우당 conditional generation 샘플 수 (기본 200)")
    ap.add_argument("--batch-per-window", type=int, default=64,
                    help="한 번 inverse 호출에 처리할 sample 수 (OOM 시 축소)")
    ap.add_argument("--L",        type=int, default=104)
    ap.add_argument("--past-len", type=int, default=52)
    ap.add_argument("--start-index", type=int, default=1,
                    help="VARIANTS 시작 index (1-based). 이전 변종은 스캔에서 skip — 도중 멈춤 후 재개용")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_cvar, real_threshold, n_pool = compute_real_cvar(
        args.test_csv, args.L, args.past_len, args.alpha)

    print(f"\n# Marginal CVaR_{args.alpha*100:.0f}%  "
          f"(sp_return future {args.L - args.past_len}w 분포 하위 tail 평균)")
    print(f"# real CVaR (test 실측)        = {real_cvar:+.6f}")
    print(f"# {args.alpha*100:.0f}%-quantile (test 실측)   = {real_threshold:+.6f}")
    print(f"# n_pool (실측 future sample 수) = {n_pool}")
    print(f"# device = {device}  |  N samples/window = {args.n_samples}\n")

    rows = []
    if args.start_index > 1:
        print(f"# start-index = {args.start_index} → 1~{args.start_index - 1} 번 변종은 스캔 skip\n")
    for i, (label, folder, prefix) in enumerate(VARIANTS, start=1):
        if i < args.start_index:
            for seed in SEEDS:
                rows.append(dict(variant=label, seed=seed, gen_cvar=float("nan")))
            continue
        for seed in SEEDS:
            ckpt_path = os.path.join(args.root, "colab", folder, "result",
                                     f"{prefix}_seed{seed}_best.pt")
            if not os.path.exists(ckpt_path):
                rows.append(dict(variant=label, seed=seed, gen_cvar=float("nan")))
                continue
            try:
                gen_cvar = evaluate_ckpt(
                    ckpt_path, args.test_csv, args.alpha,
                    args.n_samples, args.batch_per_window, device)
                rows.append(dict(variant=label, seed=seed, gen_cvar=gen_cvar))
                tag = "—" if math.isnan(gen_cvar) else f"{gen_cvar:+.6f}"
                print(f"  done   [{label:38s}] seed={seed}: gen_cvar = {tag}")
            except Exception as e:
                rows.append(dict(variant=label, seed=seed, gen_cvar=float("nan")))
                print(f"  ERROR  [{label}] seed={seed}: {e}")

    df = pd.DataFrame(rows)

    # 변종별 집계 (표 순서 유지)
    agg_rows = []
    for label, _, _ in VARIANTS:
        vals = df[df["variant"] == label]["gen_cvar"].dropna().values
        if len(vals) > 0:
            med = float(np.median(vals))
            agg_rows.append(dict(
                variant=label,
                median=med,
                mean=float(vals.mean()),
                std=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                gap=med - real_cvar,
                n=int(len(vals)),
            ))
        else:
            agg_rows.append(dict(variant=label, median=None, mean=None,
                                 std=None, gap=None, n=0))

    title = (f"Marginal CVaR_{args.alpha*100:.0f}%  (sp_return 채널, "
             f"future {args.L - args.past_len}w, N={args.n_samples}/win, "
             f"real = {real_cvar:+.4f})")
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)
    head = (f"{'#':>3}  {'variant':40s}  {'gen median':>11s}  "
            f"{'mean ± std':>20s}  {'gap (gen−real)':>15s}  {'n':>3s}")
    print(head)
    print("-" * 110)
    for i, r in enumerate(agg_rows, 1):
        if r["n"] > 0:
            print(f"{i:>3d}  {r['variant']:40s}  "
                  f"{r['median']:>+11.4f}  "
                  f"{r['mean']:>+8.4f} ± {r['std']:>6.3f}     "
                  f"{r['gap']:>+15.4f}  "
                  f"{r['n']:>3d}")
        else:
            print(f"{i:>3d}  {r['variant']:40s}  {'—':>11s}  "
                  f"{'— (no ckpt)':>20s}  {'—':>15s}  {0:>3d}")
    print("=" * 110)


if __name__ == "__main__":
    main()
