"""Vuong Closeness Test — 두 conditional flow 모델의 sp_return per-window NLL 비교.

Vuong (1989, Econometrica): non-nested 두 모델이 둘 다 misspecified 일 때
  ΔNLL_t = NLL_B(t) - NLL_A(t)  per window
  Z = √n · mean(ΔNLL) / SE(ΔNLL)  →  표준정규
  |Z| < 1.96 (5%, 양측) → fail to reject equivalence (통계적 동률)
  Z > 0  → A 우세 (NLL_A < NLL_B = A 가 진실에 더 가까움)
  Z < 0  → B 우세

Per-window NLL 은 eval_tail_nll.py 와 동일 정의 (sp_return 채널, future 52w 평균).
5 seed per-window NLL 평균 → seed noise reduce → ΔNLL 시계열 길이 n_w.

자기상관 보정: 52w sliding overlap → Bartlett kernel HAC (Newey-West 1987).
HAC lag 기본 52 (1년 weekly), --hac-lag 0 으로 끄기 가능 (i.i.d. SE 만).

사용법 (Colab, repo 동기 후):
  # default: best (mtl_bp2) vs vix baseline (mtl3)
  !python eval_vuong.py
  # 다른 비교: target vs cond
  !python eval_vuong.py --a-prefix mtl_bp2 --b-prefix K2_104_addbp --b-folder k2_104
"""
import torch  # MUST be first

import argparse
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
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

LOG2PI = math.log(2 * math.pi)
SEEDS = [42, 123, 777, 0, 99]


# ──────────────────────────────────────────────────────────────────
# eval_tail_nll.py 와 동일한 ckpt 로딩 헬퍼 (호환 유지)
# ──────────────────────────────────────────────────────────────────
def get_cond_stats(ckpt):
    s = ckpt.get("stats_train")
    if s is None:
        s = ckpt.get("stats_cond")
    if s is None:
        raise KeyError(f"ckpt 에 stats_train / stats_cond 둘 다 없음. keys={list(ckpt.keys())}")
    return s


def load_test_windows(test_csv, ckpt):
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


def sp_is_normalized(ckpt) -> bool:
    """sp_return 자체가 stats_target 으로 정규화되었는가 — 두 모델이 같은 상태여야 NLL 비교 가능."""
    target_cols = ckpt["target_cols"]
    if "sp_return" not in target_cols:
        return False
    sp_idx = target_cols.index("sp_return")
    stats_target = ckpt.get("stats_target", None)
    return stats_target is not None and int(stats_target["channel"]) == sp_idx


# ──────────────────────────────────────────────────────────────────
# Per-window NLL 추출 + seed 평균
# ──────────────────────────────────────────────────────────────────
def per_window_sp_nll(ckpt_path, test_csv, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    X, C, past_len, target_cols, config = load_test_windows(test_csv, ckpt)
    if "sp_return" not in target_cols:
        return None, None
    sp_idx = target_cols.index("sp_return")

    model = MultiStepFAVARFlow(
        K=config["K"], d_cond=config["d_cond"], d_target=config["d_target"],
        d_model=config["d_model"], n_heads=config["n_heads"],
        n_layers=config["n_layers"], time_reverse=False, use_wavelet=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        z, _, log_scale = model(X.to(device), C.to(device))
        nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale          # [B, L, D]
        nll_future = nll_td[:, past_len:, sp_idx]                    # [B, F]
        per_win = nll_future.mean(dim=1).cpu().numpy()               # [B]
    return per_win, sp_is_normalized(ckpt)


def avg_per_window_sp_nll(folder, prefix, root, test_csv, device):
    seed_arrs = []
    missing = []
    sp_norm_states = []
    for seed in SEEDS:
        ckpt_path = os.path.join(root, "colab", folder, "result",
                                 f"{prefix}_seed{seed}_best.pt")
        if not os.path.exists(ckpt_path):
            missing.append(seed)
            continue
        per_win, sp_norm = per_window_sp_nll(ckpt_path, test_csv, device)
        if per_win is None:
            raise RuntimeError(f"[{prefix} seed{seed}] target 에 sp_return 없음 (Vuong 불가 변종)")
        seed_arrs.append(per_win)
        sp_norm_states.append(sp_norm)
    if not seed_arrs:
        raise RuntimeError(f"[{prefix}] ckpt 하나도 못 찾음")
    if len(set(sp_norm_states)) > 1:
        raise RuntimeError(f"[{prefix}] seed 간 sp_return 정규화 상태 불일치: {sp_norm_states}")
    arr = np.stack(seed_arrs, axis=0)         # [n_seed, n_w]
    return arr.mean(axis=0), len(seed_arrs), missing, sp_norm_states[0]


# ──────────────────────────────────────────────────────────────────
# Bartlett kernel HAC variance estimator (Newey-West 1987)
# ──────────────────────────────────────────────────────────────────
def hac_variance_of_mean(x, lag):
    """평균 ΔNLL 의 분산 추정용 Bartlett kernel. 반환값은 (n × Var(mean)) = LongRunVar."""
    n = len(x)
    xc = x - x.mean()
    s0 = float((xc * xc).sum() / n)
    var = s0
    for k in range(1, lag + 1):
        if k >= n:
            break
        autocov = float((xc[k:] * xc[:-k]).sum() / n)
        weight = 1.0 - k / (lag + 1.0)
        var += 2.0 * weight * autocov
    return max(var, 1e-18)


def vuong_stat(delta, hac_lag):
    n = len(delta)
    mean = float(delta.mean())

    iid_var = float(delta.var(ddof=1))
    iid_se = math.sqrt(iid_var / n)
    z_iid = mean / iid_se
    p_iid = 2.0 * (1.0 - stats.norm.cdf(abs(z_iid)))

    if hac_lag is None or hac_lag <= 0:
        return dict(mean=mean, iid_se=iid_se, z_iid=z_iid, p_iid=p_iid,
                    hac_se=None, z_hac=None, p_hac=None)
    lrv = hac_variance_of_mean(delta, hac_lag)
    hac_se = math.sqrt(lrv / n)
    z_hac = mean / hac_se
    p_hac = 2.0 * (1.0 - stats.norm.cdf(abs(z_hac)))
    return dict(mean=mean, iid_se=iid_se, z_iid=z_iid, p_iid=p_iid,
                hac_se=hac_se, z_hac=z_hac, p_hac=p_hac)


# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",     default=ROOT)
    ap.add_argument("--test-csv", default=os.path.join(ROOT, "data", "weekly_ppbond_test.csv"))
    ap.add_argument("--a-folder", default="dual_3ch", help="모델 A folder under colab/")
    ap.add_argument("--a-prefix", default="mtl_bp2",
                    help="모델 A prefix (기본 best = mtl_bp2 = MTL 2ch sp+bondpp_3m)")
    ap.add_argument("--b-folder", default="dual_3ch")
    ap.add_argument("--b-prefix", default="mtl3",
                    help="모델 B prefix (기본 vix baseline = mtl3 = MTL 3ch liq+vix)")
    ap.add_argument("--hac-lag",  type=int, default=52,
                    help="HAC Bartlett lag (기본 52w=1y, 0 으로 끄기)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n# Vuong Closeness Test  (sp_return 채널, future 52w per-window NLL, seed-mean)")
    print(f"# A : folder={args.a_folder:8s}  prefix={args.a_prefix}")
    print(f"# B : folder={args.b_folder:8s}  prefix={args.b_prefix}")
    print(f"# device = {device}  |  HAC lag = {args.hac_lag} (Bartlett kernel)\n")

    nll_a, n_seed_a, miss_a, sp_norm_a = avg_per_window_sp_nll(
        args.a_folder, args.a_prefix, args.root, args.test_csv, device)
    nll_b, n_seed_b, miss_b, sp_norm_b = avg_per_window_sp_nll(
        args.b_folder, args.b_prefix, args.root, args.test_csv, device)

    if sp_norm_a != sp_norm_b:
        raise RuntimeError(
            f"sp_return 정규화 상태 불일치 (A={sp_norm_a}, B={sp_norm_b}) — "
            "NLL 단위 다름, Vuong 비교 부적합")
    if len(nll_a) != len(nll_b):
        raise RuntimeError(
            f"window count 불일치: A {len(nll_a)} vs B {len(nll_b)} — L/past_len 다름")

    delta = nll_b - nll_a    # B - A : 양수 → A 우세 (NLL_A 가 더 작음)
    n = len(delta)
    mean_a = float(nll_a.mean())
    mean_b = float(nll_b.mean())
    out = vuong_stat(delta, args.hac_lag)

    print(f"# A seeds used = {n_seed_a}/5  (missing: {miss_a})")
    print(f"# B seeds used = {n_seed_b}/5  (missing: {miss_b})")
    print(f"# n_w (test windows) = {n}")
    print(f"# sp_return 정규화 상태 일치: {sp_norm_a}\n")

    print("=" * 80)
    print("Vuong Closeness Test  ΔNLL = NLL_B − NLL_A")
    print("=" * 80)
    print(f"  mean NLL  A                       = {mean_a:+.6f}")
    print(f"  mean NLL  B                       = {mean_b:+.6f}")
    print(f"  mean ΔNLL (B − A)                 = {out['mean']:+.6f}")
    print(f"  n_w                               = {n}")
    print()
    print(f"  i.i.d. SE                         = {out['iid_se']:.6f}")
    print(f"  Vuong Z (i.i.d.)                  = {out['z_iid']:+.4f}")
    print(f"  p-value (two-sided, i.i.d.)       = {out['p_iid']:.4f}")
    print()
    if out["hac_se"] is not None:
        print(f"  HAC SE  (Bartlett, lag={args.hac_lag:>3d})        = {out['hac_se']:.6f}")
        print(f"  Vuong Z (HAC)                     = {out['z_hac']:+.4f}")
        print(f"  p-value (two-sided, HAC)          = {out['p_hac']:.4f}")
    print()

    # Decision: HAC 우선, 없으면 i.i.d.
    Z = out["z_hac"] if out["hac_se"] is not None else out["z_iid"]
    label = "HAC" if out["hac_se"] is not None else "i.i.d."
    if abs(Z) < 1.96:
        verdict = "fail to reject equivalence  (통계적 동률, 5%)"
    elif Z > 0:
        verdict = "A 우세 (NLL_A < NLL_B = A 가 진실에 더 가까움)"
    else:
        verdict = "B 우세 (NLL_B < NLL_A = B 가 진실에 더 가까움)"
    print(f"  Decision (5% 양측, {label}):  {verdict}")
    print("=" * 80)


if __name__ == "__main__":
    main()
