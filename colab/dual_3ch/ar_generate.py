"""AR scenario generation — weekly 1-step Causal Transformer ckpt 위에서.

Best variant (default: variant 8 best_base_mtl) 의 ckpt 들 (3 fold × 5 seed = 15 ckpt) 위에서:
  - Past 52w 관측 (test set 의 sliding origins)
  - Model 1-step 예측 (μ, log σ²) → Gaussian sample
  - Sample 을 target_past 에 append (sliding)
  - 52 step 반복 → 1 시나리오 path
  - N=100 번 반복 → 100 시나리오

Output: colab/dual_3ch/result/ar_scenarios_v{N}_F{F}_seed{S}.npz
  - scenarios: (n_origins, n_sim=100, 52, D_target)
  - actual: (n_origins, 52, D_target)
  - meta: {variant_id, fold, seed, target_cols}

paper main thesis: "금리 시나리오만으로 주가 시나리오 생성" 직접 구현.
한계: error compounding (step 별 오차 누적), 단 N 시나리오 평균 분포 평가가 본질.
"""
import torch  # MUST be first

import argparse
import glob
import json
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
sys.path.insert(0, HERE)
from train_matrix_1step import CausalTransformer1Step  # noqa: E402

LOG2PI = math.log(2 * math.pi)
PAST_LEN = 52
FUTURE_LEN_AR = 52   # AR scenario 길이 (paper main 1년)


def load_test_data(test_csv, cols_cond, cols_target, total_len, cond_stats, target_stats_map):
    """Test csv 에서 sliding origins — past 52w + future 52w (실측) 추출."""
    df = pd.read_csv(test_csv)
    n = len(df)
    n_w = n - total_len + 1
    if n_w <= 0:
        return None, None
    X = np.zeros((n_w, total_len, len(cols_target)), dtype=np.float32)
    C = np.zeros((n_w, total_len, len(cols_cond)),   dtype=np.float32)
    for i in range(n_w):
        X[i] = df[cols_target].iloc[i : i + total_len].values
        C[i] = df[cols_cond].iloc[i : i + total_len].values

    # cond z-score (train stats 사용)
    cmu = np.asarray(cond_stats["mean"], dtype=np.float32)
    csd = np.asarray(cond_stats["std"],  dtype=np.float32)
    C = (C - cmu) / csd

    # target normalize (aux 만, sp_return raw 유지)
    for idx_str, st in (target_stats_map or {}).items():
        idx = int(idx_str)
        X[..., idx] = (X[..., idx] - st["mean"]) / st["std"]

    return torch.from_numpy(X), torch.from_numpy(C)


@torch.no_grad()
def ar_generate_one_origin(model, target_past_init, cond_full, past_len, future_len_ar,
                            mask_future_ch, n_sim, device):
    """1 origin 에서 N=n_sim 시나리오 AR generation.

    Args:
      target_past_init: (1, past_len, D_target)  — 초기 past target
      cond_full:        (1, past_len + future_len_ar, D_cond)  — 전체 window cond

    Returns:
      paths: (n_sim, future_len_ar, D_target)  — N 시나리오
    """
    D_target = target_past_init.shape[-1]
    paths = np.zeros((n_sim, future_len_ar, D_target), dtype=np.float32)
    target_cols = model.target_cols

    for sim in range(n_sim):
        current_target = target_past_init.clone().to(device)  # (1, past_len, D_target)
        for t in range(future_len_ar):
            # cond window: past 52w + 1 step ahead
            # cond_full 에서 [t : t + past_len + 1] 슬라이스 (sliding)
            cond_window = cond_full[:, t : t + past_len + 1, :].to(device)  # (1, past_len+1, D_cond)

            # mask future cond — past_len 이후 (= 1 step) 의 idx 1.. 채널들 0
            cond_window_masked = cond_window.clone()
            for ch in mask_future_ch:
                cond_window_masked[:, past_len:, ch] = 0.0  # 1 step future = idx past_len

            outputs = model(cond_window_masked, current_target)

            # sample 각 채널
            sample = np.zeros(D_target, dtype=np.float32)
            for i, col in enumerate(target_cols):
                mu, log_var = outputs[col]
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(mu)
                sample_t = (mu + std * eps).item()
                sample[i] = sample_t

            paths[sim, t, :] = sample

            # roll target_past: shift left, append sample
            new_target = current_target.clone()
            new_target[:, :-1, :] = current_target[:, 1:, :].clone()
            new_target[:, -1, :] = torch.tensor(sample, device=device).unsqueeze(0)
            current_target = new_target

    return paths


def parse_meta_from_filename(ckpt_path):
    fname = os.path.basename(ckpt_path)
    m = re.search(r"m1step_v(\d+)_.*_(F[123])_seed(\d+)_best\.pt$", fname)
    if not m:
        return None, None, None
    return int(m.group(1)), m.group(2), int(m.group(3))


def main():
    ap = argparse.ArgumentParser(description="AR scenario generation from m1step_v* ckpts")
    ap.add_argument("--root",       default=os.path.dirname(HERE))
    ap.add_argument("--result-dir", default=None,
                    help="ckpt + output 폴더 (기본 root/colab/dual_3ch/result)")
    ap.add_argument("--folds-dir",  default=None,
                    help="fold csv 폴더 (기본 root/data/folds_v33)")
    ap.add_argument("--variants",   nargs="+", type=int, default=[8],
                    help="AR generation 할 variant id (기본 [8] = best_base_mtl)")
    ap.add_argument("--n-sim",      type=int, default=100,
                    help="origin 당 시나리오 수")
    ap.add_argument("--ckpt-glob",  default="m1step_v{v}_*_F{f}_seed*_best.pt",
                    help="ckpt 파일 패턴 ({v}={variant_id}, {f}={fold})")
    args = ap.parse_args()

    repo_root  = os.path.normpath(os.path.dirname(args.root)) if os.path.basename(args.root) == "homeostatic-market" else args.root
    # safer:
    project_root = HERE
    while os.path.basename(project_root) != "homeostatic-market" and project_root != os.path.dirname(project_root):
        project_root = os.path.dirname(project_root)

    result_dir = args.result_dir or os.path.join(project_root, "colab", "dual_3ch", "result")
    folds_dir  = args.folds_dir  or os.path.join(project_root, "data", "folds_v33")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, n_sim={args.n_sim}, variants={args.variants}")

    for variant_id in args.variants:
        for fold in ["F1", "F2", "F3"]:
            # ckpt 패턴 (variant 별 normalize tag 다양 — glob 으로 매칭)
            pattern = os.path.join(result_dir, f"m1step_v{variant_id}_*_{fold}_seed*_best.pt")
            ckpts = sorted(glob.glob(pattern))
            if not ckpts:
                print(f"\n[SKIP] no ckpt at {pattern}")
                continue

            test_csv = os.path.join(folds_dir, f"{fold}_test.csv")
            if not os.path.exists(test_csv):
                print(f"\n[SKIP] no test_csv: {test_csv}")
                continue

            print(f"\n{'=' * 72}")
            print(f"Variant {variant_id}  Fold {fold}  ({len(ckpts)} ckpts)")
            print(f"{'=' * 72}")

            for ckpt_path in ckpts:
                vid, f, seed = parse_meta_from_filename(ckpt_path)
                if vid is None:
                    print(f"  SKIP unparseable: {os.path.basename(ckpt_path)}")
                    continue
                out_npz = os.path.join(result_dir, f"ar_scenarios_v{variant_id}_{fold}_seed{seed}.npz")
                if os.path.exists(out_npz):
                    print(f"  [SKIP] {os.path.basename(out_npz)} exists")
                    continue

                # ckpt 로드
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                cfg  = ckpt["config"]
                cond_cols   = ckpt["cond_cols"]
                target_cols = ckpt["target_cols"]
                stats_cond  = ckpt["stats_cond"]
                stats_target_map = ckpt.get("stats_target_map", {})
                mask_future_ch   = ckpt["mask_future_ch"]

                # model 재생성 + state load
                model = CausalTransformer1Step(
                    d_cond=cfg["d_cond"], target_cols=target_cols,
                    d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                    past_len=cfg["past_len"], future_len=cfg.get("future_len", 1),
                ).to(device)
                model.load_state_dict(ckpt["model_state"])
                model.eval()

                # test data load
                total_len = cfg["past_len"] + FUTURE_LEN_AR
                X_full, C_full = load_test_data(
                    test_csv, cond_cols, target_cols, total_len,
                    stats_cond, stats_target_map)
                if X_full is None:
                    print(f"  [SKIP] not enough test windows for L={total_len}")
                    continue
                n_origins = X_full.shape[0]
                print(f"  ckpt={os.path.basename(ckpt_path)}  n_origins={n_origins}")

                # AR generation per origin
                scenarios_all = np.zeros((n_origins, args.n_sim, FUTURE_LEN_AR, len(target_cols)), dtype=np.float32)
                actual_all    = X_full[:, cfg["past_len"]:, :].numpy().astype(np.float32)

                for i in range(n_origins):
                    target_past_i = X_full[i:i+1, :cfg["past_len"], :]   # (1, 52, D)
                    cond_full_i   = C_full[i:i+1, :, :]                   # (1, total_len, D_cond)

                    paths = ar_generate_one_origin(
                        model, target_past_i, cond_full_i,
                        past_len=cfg["past_len"], future_len_ar=FUTURE_LEN_AR,
                        mask_future_ch=mask_future_ch, n_sim=args.n_sim,
                        device=device)
                    scenarios_all[i] = paths
                    if (i + 1) % 5 == 0:
                        print(f"    origin {i+1}/{n_origins} done")

                # 저장
                np.savez_compressed(
                    out_npz,
                    scenarios=scenarios_all,
                    actual=actual_all,
                    target_cols=np.array(target_cols),
                    variant_id=variant_id,
                    fold=fold,
                    seed=seed,
                    n_sim=args.n_sim,
                )
                print(f"  saved → {out_npz}  shape={scenarios_all.shape}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
