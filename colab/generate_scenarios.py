"""시나리오 생성 (batched sampling) — paper 표 변종별 future 52w sp_return 시나리오.

Past-z swap 기반 conditional generation (Phase 15 패턴):
1. x_full_init = [x_past_obs, 0_future] → forward → z_full_det
2. z_past = z_full_det[:past_len]  (관측값에 대응되는 결정 z)
3. z_future_random = torch.randn(N, n_w, future_len, D_target)  (공통 seed)
4. z_new = [z_past_fixed.expand_to_N, z_future_random]
5. inverse(z_new, c) → x_gen
6. raw 단위 변환: x_gen[ch] = x_gen[ch] · std + mean (정규화 채널 역변환)

저장 형식 (per ckpt):
  scenarios/{prefix}_{fold}_seed{seed}_scenarios.npy  shape = [N, n_w, future_len]
  → sp_return 채널만 추출 + raw 단위
"""
import torch  # MUST be first

import argparse
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
FOLDS = ["F1", "F2", "F3"]
COMMON_RANDOM_SEED = 12345   # 모든 변종 공통 random z (paired comparison)

# 파일럿 3 변종 (사용자 선택: base + best 2)
VARIANTS = [
    # (paper_row, label, folder, prefix)
    ( 2, "Base (cond=tbill 1ch)",                  "k2_104",   "K2_104_2ch_normsr"),
    (11, "MTL stockpp정규 [no-liq] (best, structural)", "dual_3ch", "mtl_pps2_noliq_normsp_normsr"),
    (15, "MTL liq+vix정규 (vix-based, emotional)",     "dual_3ch", "mtl3_normvix_normsr"),
]


def get_cond_stats(ckpt):
    s = ckpt.get("stats_train")
    if s is None:
        s = ckpt.get("stats_cond")
    if s is None:
        raise KeyError(f"ckpt 에 stats_train / stats_cond 둘 다 없음")
    return s


def load_test_data(test_csv, ckpt):
    """test_csv 에서 sliding window 추출 + ckpt 의 stats 로 정규화 적용."""
    cond_cols = ckpt["cond_cols"]
    target_cols = ckpt["target_cols"]
    config = ckpt["config"]
    L = config["total_len"]
    past_len = config["past_len"]
    future_len = L - past_len

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

    # target normalize (정규화된 채널만)
    stats_targets_all = ckpt.get("stats_targets_all") or {}
    if not stats_targets_all:
        st = ckpt.get("stats_target")
        if st is not None:
            stats_targets_all = {int(st["channel"]): st}

    for ch_key, s in stats_targets_all.items():
        ch = int(ch_key)
        X[..., ch] = (X[..., ch] - float(s["mean"])) / float(s["std"])

    # mask future cond channels
    mask_future_ch = ckpt.get("mask_future_ch", []) or []
    for ch in mask_future_ch:
        C[:, past_len:, ch] = 0.0

    return torch.from_numpy(X), torch.from_numpy(C), past_len, future_len, target_cols, config, stats_targets_all


@torch.no_grad()
def generate_scenarios_batched(model, X, C, past_len, future_len, N, device, batch_size=256):
    """Past-z swap conditional generation. Batched on (N × n_window) dim.

    Returns: x_gen tensor of shape [N, n_w, L, D_target] (정규화된 단위)
    """
    n_w, L, D = X.shape
    Dc = C.shape[-1]

    # Step 1: forward 1 회 — z_past_fixed 추출
    x_full_init = X.clone()
    x_full_init[:, past_len:, :] = 0.0
    z_full_det, _, _ = model(x_full_init.to(device), C.to(device))
    z_past_fixed = z_full_det[:, :past_len, :].cpu()                # [n_w, P, D]

    # Step 2: random z_future (공통 seed)
    torch.manual_seed(COMMON_RANDOM_SEED)
    z_future_all = torch.randn(N, n_w, future_len, D)               # [N, n_w, F, D]

    # Step 3: inverse — batched on (N) 차원, n_w 차원 그대로
    x_gen_all = torch.zeros(N, n_w, L, D)
    for i in range(0, N, batch_size):
        N_chunk = min(batch_size, N - i)
        z_past_chunk  = z_past_fixed.unsqueeze(0).expand(N_chunk, n_w, past_len, D)
        z_full_chunk  = torch.cat([z_past_chunk, z_future_all[i:i+N_chunk]], dim=2)   # [Nc, n_w, L, D]
        c_chunk       = C.unsqueeze(0).expand(N_chunk, n_w, L, Dc)

        z_flat = z_full_chunk.reshape(N_chunk * n_w, L, D).to(device)
        c_flat = c_chunk.reshape(N_chunk * n_w, L, Dc).to(device)
        x_gen_flat = model.inverse(z_flat, c_flat).cpu()                              # [Nc*n_w, L, D]
        x_gen_all[i:i+N_chunk] = x_gen_flat.reshape(N_chunk, n_w, L, D)
    return x_gen_all


def denormalize(x_gen, target_cols, stats_targets_all):
    """정규화된 x_gen → raw 단위로 역변환."""
    x_raw = x_gen.clone()
    for ch_key, s in stats_targets_all.items():
        ch = int(ch_key)
        if ch < x_raw.shape[-1]:
            x_raw[..., ch] = x_raw[..., ch] * float(s["std"]) + float(s["mean"])
    return x_raw


def process_one_ckpt(ckpt_path, test_csv, out_path, N, device, batch_size):
    if os.path.exists(out_path):
        print(f"  [SKIP] {os.path.basename(out_path)} 이미 존재")
        return
    print(f"  generating: {os.path.basename(ckpt_path)} → {os.path.basename(out_path)}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    X, C, past_len, future_len, target_cols, config, stats_targets = load_test_data(test_csv, ckpt)
    if "sp_return" not in target_cols:
        print(f"    [SKIP] target 에 sp_return 없음")
        return
    sp_idx = target_cols.index("sp_return")

    model = MultiStepFAVARFlow(
        K=config["K"], d_cond=config["d_cond"], d_target=config["d_target"],
        d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"],
        time_reverse=False, use_wavelet=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_gen = generate_scenarios_batched(model, X, C, past_len, future_len, N, device, batch_size)
    x_gen_raw = denormalize(x_gen, target_cols, stats_targets)

    # future 52w sp_return 채널만 추출
    future_sp = x_gen_raw[:, :, past_len:, sp_idx].numpy().astype(np.float32)   # [N, n_w, F]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, future_sp)
    print(f"    saved shape={future_sp.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out-subdir", default="result_paper_final",
                    help="ckpt 폴더 (colab/<folder>/<out_subdir>/)")
    ap.add_argument("--data-folds", default=os.path.join(ROOT, "data", "folds"))
    ap.add_argument("--N", type=int, default=1000, help="시나리오 수 per window")
    ap.add_argument("--batch-size", type=int, default=256, help="N 차원 batch size")
    ap.add_argument("--save-folder", default="scenarios",
                    help="저장 sub-folder (colab/<folder>/<out_subdir>/<save_folder>/)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}, N = {args.N}, common_seed = {COMMON_RANDOM_SEED}\n")

    for paper_row, label, folder, prefix in VARIANTS:
        print(f"\n=== 행 {paper_row}: {label} ({prefix}) ===")
        for fold in FOLDS:
            test_csv = os.path.join(args.data_folds, f"{fold}_test.csv")
            for seed in SEEDS:
                ckpt_path = os.path.join(args.root, "colab", folder, args.out_subdir,
                                         f"{prefix}_{fold}_seed{seed}_best.pt")
                out_path = os.path.join(args.root, "colab", folder, args.out_subdir, args.save_folder,
                                        f"{prefix}_{fold}_seed{seed}_scenarios.npy")
                if not os.path.exists(ckpt_path):
                    print(f"  [missing ckpt] {ckpt_path}")
                    continue
                process_one_ckpt(ckpt_path, test_csv, out_path, args.N, device, args.batch_size)


if __name__ == "__main__":
    main()
