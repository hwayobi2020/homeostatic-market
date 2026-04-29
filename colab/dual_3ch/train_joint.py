"""Joint single-stage 대조군 — Stage 1/2 cascade 없이 (excess_liq_wr, sp_return) 동시 생성.

cond (2ch × 104w): tbill_wr, tbill_26w_lag (past + future 모두 활성, mask 없음)
target (2ch × past+future 52+52w): excess_liq_wr, sp_return (joint AR generation)

Architecture: MultiStepFAVARFlow K=2, d_target=2 (Base K2_104 의 d_target=1 과 다른 점만 차이).
z-score fix 적용 (test cond uses train mu/sd).

Comparison logic:
    Stage 1 + Stage 2 cascade  : sp_return NLL ≈ -1.88 (Stage 2)
    Joint single-stage (이거)   : sp_return 채널만 분리한 NLL = ?

    if Joint sp_return < Stage 2 → cascade 가 정보 손실 → "homeostatic bottleneck" 주장 약화
    if Joint sp_return ≈ Stage 2 → cascade 의 손실 없음
    if Joint sp_return > Stage 2 → cascade 가 도움 (forced mediator 효과)

Per-channel NLL 도 분리 보고 (excess_liq_wr / sp_return).
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
K_STEPS = 2
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
LR = 5e-4
BATCH = 32
MAX_EPOCHS = 60
PATIENCE = 15

COLS_TARGET = ["excess_liq_wr", "sp_return"]   # 2ch joint
COLS_COND_NO_26W   = ["tbill_wr"]                          # default 1ch
COLS_COND_WITH_26W = ["tbill_wr", "tbill_26w_lag"]         # legacy 2ch reproduce
COLS_COND = COLS_COND_NO_26W

LOG2PI = math.log(2 * math.pi)


def load_windows(csv_path, cols_cond, cols_target, L=104, stats=None):
    df = pd.read_csv(csv_path)
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        return None, None, None
    X = np.zeros((n_w, L, len(cols_target)), dtype=np.float32)
    C = np.zeros((n_w, L, len(cols_cond)), dtype=np.float32)
    for i in range(n_w):
        X[i] = df[cols_target].iloc[i : i + L].values
        C[i] = df[cols_cond].iloc[i : i + L].values
    if stats is None:
        mu = C.reshape(-1, len(cols_cond)).mean(axis=0)
        sd = C.reshape(-1, len(cols_cond)).std(axis=0) + 1e-8
    else:
        mu = np.asarray(stats["mean"], dtype=np.float32)
        sd = np.asarray(stats["std"],  dtype=np.float32)
    C = (C - mu) / sd
    return torch.from_numpy(X), torch.from_numpy(C), {"mean": mu.tolist(), "std": sd.tolist()}


def nll_per_step_per_channel(X, C, model, past_len, device):
    """Returns total mean NLL/step/channel AND per-channel breakdown.

    Returns:
        nll_mean   : float — 평균 NLL/step/channel (모든 채널 평균)
        nll_per_ch : np.ndarray [D] — 채널별 평균 NLL/step
    """
    B, L_, D = X.shape
    F_ = L_ - past_len
    z, log_det_J, log_scale = model(X.to(device), C.to(device))
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale  # [B, L, D]
    nll_future = nll_td[:, past_len:, :]                 # [B, F, D]
    nll_per_ch = nll_future.mean(dim=(0, 1)).detach().cpu().numpy()  # [D]
    nll_mean = float(nll_per_ch.mean())
    return nll_mean, nll_per_ch


def loss_fn(X, C, model, past_len, device):
    B, L_, D = X.shape
    F_ = L_ - past_len
    z, log_det_J, log_scale = model(X.to(device), C.to(device))
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale
    nll_future = nll_td[:, past_len:, :].sum(dim=(1, 2))
    return (nll_future / (F_ * D)).mean()


def run(train_csv, test_csv, save_dir, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n{'=' * 70}")
    print(f"[Joint] seed={seed}")
    print(f"  cond   = {COLS_COND}    (D_COND={len(COLS_COND)})")
    print(f"  target = {COLS_TARGET}  (D_TARGET={len(COLS_TARGET)})  joint generation")
    print(f"{'=' * 70}")

    Xtr, Ctr, stats_tr = load_windows(train_csv, COLS_COND, COLS_TARGET, L=L)
    Xte, Cte, _        = load_windows(test_csv,  COLS_COND, COLS_TARGET, L=L, stats=stats_tr)

    print(f"  z-score stats (train): mean={[round(m,5) for m in stats_tr['mean']]}")
    print(f"                            std={[round(s,5) for s in stats_tr['std']]}")
    Cte_np = Cte.numpy().reshape(-1, Cte.shape[-1])
    print(f"  test z mean per channel: {[round(float(m),4) for m in Cte_np.mean(axis=0)]}")
    print(f"  test z std  per channel: {[round(float(s),4) for s in Cte_np.std(axis=0)]}")

    n_w_tr = Xtr.shape[0]
    n_val = max(int(n_w_tr * 0.15), 1)
    Xtr_, Ctr_ = Xtr[:-n_val], Ctr[:-n_val]
    Xv,  Cv  = Xtr[-n_val:], Ctr[-n_val:]
    print(f"  train_windows={Xtr_.shape[0]}, val_windows={Xv.shape[0]}, test_windows={Xte.shape[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiStepFAVARFlow(
        K=K_STEPS, d_cond=len(COLS_COND), d_target=len(COLS_TARGET),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        time_reverse=False, use_wavelet=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    best_test = float("nan")
    best_test_per_ch = None
    pat = 0
    log = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(Xtr_.shape[0])
        losses = []
        for i in range(0, len(perm), BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = loss_fn(Xtr_[idx], Ctr_[idx], model, PAST_LEN, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_nll,  val_per_ch  = nll_per_step_per_channel(Xv,  Cv,  model, PAST_LEN, device)
            test_nll, test_per_ch = nll_per_step_per_channel(Xte, Cte, model, PAST_LEN, device)
        train_nll = float(np.mean(losses))
        print(f"  ep{epoch:>3d}  train={train_nll:+.4f}  val={val_nll:+.4f}  test={test_nll:+.4f}  "
              f"| test_per_ch: {COLS_TARGET[0]}={test_per_ch[0]:+.4f} {COLS_TARGET[1]}={test_per_ch[1]:+.4f}")
        log.append(dict(
            epoch=epoch, train=train_nll, val=val_nll, test=test_nll,
            **{f"test_{COLS_TARGET[0]}": float(test_per_ch[0]),
               f"test_{COLS_TARGET[1]}": float(test_per_ch[1])},
        ))

        if val_nll < best_val - 1e-4:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test = test_nll
            best_test_per_ch = test_per_ch.copy()
            best_epoch = epoch
            pat = 0
        else:
            pat += 1
        if pat >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val={best_val:+.4f}, test={best_test:+.4f}")
    print(f"    test per channel: {COLS_TARGET[0]}={best_test_per_ch[0]:+.4f}  {COLS_TARGET[1]}={best_test_per_ch[1]:+.4f}")

    os.makedirs(save_dir, exist_ok=True)
    tag = f"joint_seed{seed}"
    ckpt_path    = os.path.join(save_dir, f"{tag}_best.pt")
    log_path     = os.path.join(save_dir, f"{tag}_trainlog.csv")
    summary_path = os.path.join(save_dir, f"{tag}_summary.json")

    if best_state is not None:
        torch.save({
            "model_state":   best_state,
            "cond_cols":     COLS_COND,
            "target_cols":   COLS_TARGET,
            "stats_train":   stats_tr,
            "config": dict(K=K_STEPS, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                           d_cond=len(COLS_COND), d_target=len(COLS_TARGET),
                           past_len=PAST_LEN, total_len=L),
        }, ckpt_path)
    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        stage="joint",
        seed=seed,
        cond_cols=COLS_COND,
        target_cols=COLS_TARGET,
        best_epoch=best_epoch,
        val=best_val,
        test=best_test,
        test_per_channel={
            COLS_TARGET[0]: float(best_test_per_ch[0]),
            COLS_TARGET[1]: float(best_test_per_ch[1]),
        },
        n_params=n_params,
        device=str(device),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {ckpt_path}, {log_path}, {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--train-csv", default=os.path.join(HERE, "data", "weekly_ppbond_train.csv"))
    ap.add_argument("--test-csv",  default=os.path.join(HERE, "data", "weekly_ppbond_test.csv"))
    ap.add_argument("--out-dir",   default=os.path.join(HERE, "result"))
    ap.add_argument("--with-26w", action="store_true", help="Include tbill_26w_lag")
    args = ap.parse_args()
    if args.with_26w:
        global COLS_COND
        COLS_COND = COLS_COND_WITH_26W
        print(f"[--with-26w] cond = {COLS_COND}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for seed in args.seeds:
        r = run(args.train_csv, args.test_csv, args.out_dir, seed=seed)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        df_rows = []
        for r in results:
            df_rows.append({
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "val": r["val"],
                "test_mean": r["test"],
                f"test_{COLS_TARGET[0]}": r["test_per_channel"][COLS_TARGET[0]],
                f"test_{COLS_TARGET[1]}": r["test_per_channel"][COLS_TARGET[1]],
            })
        df = pd.DataFrame(df_rows)
        df.to_csv(os.path.join(args.out_dir, "joint_multiseed_results.csv"), index=False)
        print(f"\n[Joint] multi-seed (n={len(df)})")
        print(f"  val:                  mean={df.val.mean():+.4f} ± {df.val.std():.4f}  median={df.val.median():+.4f}")
        print(f"  test mean (2ch avg):  mean={df.test_mean.mean():+.4f} ± {df.test_mean.std():.4f}  median={df.test_mean.median():+.4f}")
        col_a = f"test_{COLS_TARGET[0]}"
        col_b = f"test_{COLS_TARGET[1]}"
        print(f"  test {COLS_TARGET[0]:<14s}: mean={df[col_a].mean():+.4f} ± {df[col_a].std():.4f}  median={df[col_a].median():+.4f}")
        print(f"  test {COLS_TARGET[1]:<14s}: mean={df[col_b].mean():+.4f} ± {df[col_b].std():.4f}  median={df[col_b].median():+.4f}")
        print(f"\n  ★ comparison vs Stage 2 sp_return: Stage 2 mean=-1.88, this joint sp_return mean={df[col_b].mean():+.4f}")


if __name__ == "__main__":
    main()
