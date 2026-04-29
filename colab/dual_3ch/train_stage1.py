"""Stage 1 (MacroExpander) — Single FAVAR Flow.

cond past   (3ch × past 52w)  : tbill_wr, tbill_26w_lag, excess_liq_26w_lag
cond future (3ch × future 52w): tbill_wr, tbill_26w_lag from user scenario;
                                excess_liq_26w_lag MASKED to 0 (= train mean in z-space)
target (1ch × past+future 52+52w): excess_liq_yoy

Architecture: MultiStepFAVARFlow K=2, d_model=64, n_heads=4, n_layers=2 (NLL only).
z-score fix applied (test cond uses train mu/sd).
"""
import torch  # MUST be first — Windows DLL load-order workaround for numpy/pandas conflict

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

# ── 고정 하이퍼파라미터 ────────
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

COLS_TARGET = ["excess_liq_wr"]   # weekly raw — Stage 2 의 cond ch=2 와 직접 일치
COLS_COND   = ["tbill_wr", "tbill_26w_lag", "excess_liq_wr"]
# future cond: only tbill_wr 시나리오. tbill_26w_lag (deterministic) 와 excess_liq_wr (target) 는 마스킹.
MASK_FUTURE_CH = [1, 2]

LOG2PI = math.log(2 * math.pi)


def load_windows(csv_path, cols_cond, cols_target, L=104, stats=None):
    """stats=None: train mode (compute mu/sd). stats=<dict>: test mode (apply train stats)."""
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


def mask_future_channels(C, past_len, mask_channels):
    """Set future portion of specified channels to 0 (= train mean in z-score space)."""
    C = C.clone()
    for ch in mask_channels:
        C[:, past_len:, ch] = 0.0
    return C


def nll_per_step_channel(X, C, model, past_len, device):
    """Mean NLL per step per channel over future portion."""
    B, L_, D = X.shape
    F_ = L_ - past_len
    z, log_det_J, log_scale = model(X.to(device), C.to(device))
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale  # [B, L, D]
    nll_future = nll_td[:, past_len:, :].sum(dim=(1, 2))  # [B]
    return nll_future / (F_ * D)


def run(train_csv, test_csv, save_dir, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n{'=' * 70}")
    print(f"[Stage1] seed={seed}")
    print(f"  cond   = {COLS_COND}")
    print(f"  target = {COLS_TARGET}")
    print(f"  mask_future_ch = {MASK_FUTURE_CH} (= excess_liq_26w_lag future masked to 0)")
    print(f"{'=' * 70}")

    Xtr, Ctr, stats_tr = load_windows(train_csv, COLS_COND, COLS_TARGET, L=L)
    Xte, Cte, _        = load_windows(test_csv,  COLS_COND, COLS_TARGET, L=L, stats=stats_tr)

    print(f"  z-score stats (train): mean={[round(m,4) for m in stats_tr['mean']]}")
    print(f"                            std={[round(s,4) for s in stats_tr['std']]}")
    Cte_np = Cte.numpy().reshape(-1, Cte.shape[-1])
    print(f"  test z mean per channel: {[round(float(m),4) for m in Cte_np.mean(axis=0)]}")
    print(f"  test z std  per channel: {[round(float(s),4) for s in Cte_np.std(axis=0)]}")

    Ctr = mask_future_channels(Ctr, PAST_LEN, MASK_FUTURE_CH)
    Cte = mask_future_channels(Cte, PAST_LEN, MASK_FUTURE_CH)

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
    pat = 0
    log = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(Xtr_.shape[0])
        losses = []
        for i in range(0, len(perm), BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            loss = nll_per_step_channel(Xtr_[idx], Ctr_[idx], model, PAST_LEN, device).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_nll  = nll_per_step_channel(Xv,  Cv,  model, PAST_LEN, device).mean().item()
            test_nll = nll_per_step_channel(Xte, Cte, model, PAST_LEN, device).mean().item()
        train_nll = float(np.mean(losses))
        print(f"  ep{epoch:>3d}  train={train_nll:+.4f}  val={val_nll:+.4f}  test={test_nll:+.4f}")
        log.append(dict(epoch=epoch, train=train_nll, val=val_nll, test=test_nll))

        if val_nll < best_val - 1e-4:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test = test_nll
            best_epoch = epoch
            pat = 0
        else:
            pat += 1
        if pat >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val={best_val:+.4f}, test={best_test:+.4f}")

    os.makedirs(save_dir, exist_ok=True)
    tag = f"stage1_seed{seed}"
    ckpt_path    = os.path.join(save_dir, f"{tag}_best.pt")
    log_path     = os.path.join(save_dir, f"{tag}_trainlog.csv")
    summary_path = os.path.join(save_dir, f"{tag}_summary.json")

    if best_state is not None:
        torch.save({
            "model_state":     best_state,
            "cond_cols":       COLS_COND,
            "target_cols":     COLS_TARGET,
            "mask_future_ch":  MASK_FUTURE_CH,
            "stats_train":     stats_tr,
            "config": dict(K=K_STEPS, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                           d_cond=len(COLS_COND), d_target=len(COLS_TARGET),
                           past_len=PAST_LEN, total_len=L),
        }, ckpt_path)
    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        stage="stage1",
        seed=seed,
        cond_cols=COLS_COND,
        target_cols=COLS_TARGET,
        mask_future_ch=MASK_FUTURE_CH,
        best_epoch=best_epoch,
        val=best_val,
        test=best_test,
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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for seed in args.seeds:
        r = run(args.train_csv, args.test_csv, args.out_dir, seed=seed)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.out_dir, "stage1_multiseed_results.csv"), index=False)
        v_mean, v_std = df.val.mean(),  df.val.std()
        t_mean, t_std = df.test.mean(), df.test.std()
        v_med, t_med = df.val.median(), df.test.median()
        print(f"\n[Stage1] multi-seed (n={len(df)})")
        print(f"  val:  mean={v_mean:+.4f} ± {v_std:.4f}  median={v_med:+.4f}")
        print(f"  test: mean={t_mean:+.4f} ± {t_std:.4f}  median={t_med:+.4f}")


if __name__ == "__main__":
    main()
