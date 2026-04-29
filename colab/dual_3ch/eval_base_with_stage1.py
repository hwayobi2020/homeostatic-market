"""Base + inference swap evaluation.

Load Base K2_104 ckpt (trained with observed excess_liq_wr in future cond).
At inference, replace future cond ch=2 with Stage 1's deterministic mean prediction.
Compute test NLL. No retraining.

Comparison target: Stage 2 (which was retrained with Stage 1 generation).
Expected: Base+swap ≈ Stage 2 if Base is robust to cond swap (no train/inference mismatch penalty).
"""
import torch  # MUST be first

import argparse
import json
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
sys.path.insert(0, HERE)
from favar_flow import MultiStepFAVARFlow  # noqa: E402

L = 104
PAST_LEN = 52

COLS_TARGET = ["sp_return"]
COLS_COND   = ["tbill_wr", "tbill_26w_lag", "excess_liq_wr"]
EXCESS_LIQ_CH_IDX = 2
STAGE1_MASK_FUTURE_CH = [1, 2]

LOG2PI = math.log(2 * math.pi)


def load_window_arrays(csv_path, cols_cond, cols_target, L=104):
    df = pd.read_csv(csv_path)
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        return None, None
    target   = np.zeros((n_w, L, len(cols_target)), dtype=np.float32)
    cond_raw = np.zeros((n_w, L, len(cols_cond)),   dtype=np.float32)
    for i in range(n_w):
        target[i]   = df[cols_target].iloc[i : i + L].values
        cond_raw[i] = df[cols_cond].iloc[i : i + L].values
    return torch.from_numpy(target), torch.from_numpy(cond_raw)


def zscore(C_raw, stats):
    mu = np.asarray(stats["mean"], dtype=np.float32)
    sd = np.asarray(stats["std"],  dtype=np.float32)
    return (C_raw - torch.from_numpy(mu)) / torch.from_numpy(sd)


def mask_future_channels(C, past_len, mask_channels):
    C = C.clone()
    for ch in mask_channels:
        C[:, past_len:, ch] = 0.0
    return C


def nll_per_step_channel(X, C, model, past_len, device):
    B, L_, D = X.shape
    F_ = L_ - past_len
    z, log_det_J, log_scale = model(X.to(device), C.to(device))
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale
    nll_future = nll_td[:, past_len:, :].sum(dim=(1, 2))
    return nll_future / (F_ * D)


def load_flow_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MultiStepFAVARFlow(
        K=cfg["K"], d_cond=cfg["d_cond"], d_target=cfg["d_target"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        time_reverse=False, use_wavelet=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("stats_train", None)


@torch.no_grad()
def stage1_generate_mean(stage1_model, past_target_raw, cond_z_full, past_len, device):
    B, P, D = past_target_raw.shape
    F_len = cond_z_full.shape[1] - past_len
    past_target_raw = past_target_raw.to(device)
    cond_z_full     = cond_z_full.to(device)
    x_init = torch.cat(
        [past_target_raw, past_target_raw.new_zeros(B, F_len, D)], dim=1
    )
    z_full, _, _ = stage1_model(x_init, cond_z_full)
    z_new = z_full.clone()
    z_new[:, past_len:, :] = 0.0
    x_gen = stage1_model.inverse(z_new, cond_z_full)
    return x_gen[:, past_len:, :].cpu()


def replace_future_with_stage1(cond_raw_full, stage1_model, stage1_stats, past_len, device, batch_size=64):
    N = cond_raw_full.shape[0]
    cond_modified = cond_raw_full.clone()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        cond_batch_raw = cond_raw_full[start:end]
        s1_cond_z = zscore(cond_batch_raw, stage1_stats)
        s1_cond_z = mask_future_channels(s1_cond_z, past_len, STAGE1_MASK_FUTURE_CH)
        past_target_raw = cond_batch_raw[:, :past_len, EXCESS_LIQ_CH_IDX:EXCESS_LIQ_CH_IDX + 1]
        future_target_gen = stage1_generate_mean(
            stage1_model, past_target_raw, s1_cond_z, past_len, device
        )
        cond_modified[start:end, past_len:, EXCESS_LIQ_CH_IDX] = future_target_gen[:, :, 0]
    return cond_modified


def run(seed, base_ckpt_dir, stage1_ckpt_dir, test_csv, out_dir, device):
    base_ckpt    = os.path.join(base_ckpt_dir,   f"K2_104_seed{seed}_best.pt")
    stage1_ckpt  = os.path.join(stage1_ckpt_dir, f"stage1_seed{seed}_best.pt")
    if not os.path.exists(base_ckpt):
        print(f"[FAIL] Base ckpt missing: {base_ckpt}")
        return None
    if not os.path.exists(stage1_ckpt):
        print(f"[FAIL] Stage 1 ckpt missing: {stage1_ckpt}")
        return None

    print(f"\n[seed {seed}]")
    base_model,   base_stats   = load_flow_ckpt(base_ckpt,   device)
    stage1_model, stage1_stats = load_flow_ckpt(stage1_ckpt, device)
    print(f"  Base   ckpt: {base_ckpt}")
    print(f"  Stage1 ckpt: {stage1_ckpt}")

    Xte, Cte_raw = load_window_arrays(test_csv, COLS_COND, COLS_TARGET, L=L)
    if Xte is None:
        print("[FAIL] not enough test windows")
        return None

    # 1) Base oracle eval (sanity check)
    Cte_oracle_z = zscore(Cte_raw, base_stats)
    test_nll_oracle = nll_per_step_channel(Xte, Cte_oracle_z, base_model, PAST_LEN, device).mean().item()

    # 2) Base + inference swap (Stage 1 generation in future ch=2)
    Cte_raw_swap = replace_future_with_stage1(
        Cte_raw, stage1_model, stage1_stats, PAST_LEN, device
    )
    diff = (Cte_raw_swap - Cte_raw)[:, PAST_LEN:, EXCESS_LIQ_CH_IDX]
    print(f"  cond ch=2 future diff (gen − obs): mean={float(diff.mean()):+.5f}  std={float(diff.std()):.5f}")

    Cte_swap_z = zscore(Cte_raw_swap, base_stats)
    test_nll_swap = nll_per_step_channel(Xte, Cte_swap_z, base_model, PAST_LEN, device).mean().item()

    print(f"  Base oracle test NLL : {test_nll_oracle:+.4f}")
    print(f"  Base + swap test NLL : {test_nll_swap:+.4f}")
    print(f"  Δ (swap − oracle)    : {test_nll_swap - test_nll_oracle:+.4f}")

    return dict(
        seed=seed,
        test_nll_base_oracle=test_nll_oracle,
        test_nll_base_swap=test_nll_swap,
        delta=test_nll_swap - test_nll_oracle,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777, 0, 99])
    ap.add_argument("--base-ckpt-dir",   default=os.path.join(HERE, "..", "k2_104", "result"))
    ap.add_argument("--stage1-ckpt-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--test-csv",        default=os.path.join(HERE, "data", "weekly_ppbond_test.csv"))
    ap.add_argument("--out-dir",         default=os.path.join(HERE, "result"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for seed in args.seeds:
        r = run(seed, args.base_ckpt_dir, args.stage1_ckpt_dir, args.test_csv, args.out_dir, device)
        if r is not None:
            results.append(r)

    if len(results) == 0:
        print("[FAIL] no results")
        return

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "base_with_stage1_swap_results.csv"), index=False)

    print(f"\n{'=' * 70}")
    print(f"Summary (n={len(df)})")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))
    print(f"\nBase oracle    test NLL: mean={df.test_nll_base_oracle.mean():+.4f} ± {df.test_nll_base_oracle.std():.4f}  median={df.test_nll_base_oracle.median():+.4f}")
    print(f"Base + swap    test NLL: mean={df.test_nll_base_swap.mean():+.4f} ± {df.test_nll_base_swap.std():.4f}  median={df.test_nll_base_swap.median():+.4f}")
    print(f"Δ (swap − oracle)      : mean={df.delta.mean():+.4f} ± {df.delta.std():.4f}  median={df.delta.median():+.4f}")


if __name__ == "__main__":
    main()
