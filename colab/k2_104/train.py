"""K2_104 Single-Stage FAVAR Flow — Colab GPU 학습.

Base 모델 (K2_pure_ppbond) condition 변형:
    cond (4ch × 104w): excess_liq_yoy, tbill_wr, tbill_26w_lag, excess_liq_26w_lag
    target (1ch × 52w future): sp_return  (past 52w 도 자기관측 입력)

Architecture: MultiStepFAVARFlow K=2, d_model=64, n_heads=4, n_layers=2
Loss: NLL only (no wavelet, no PINN)
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

# ── 고정 하이퍼파라미터 (sim/train_favar_ppbond.py 와 동일) ────────
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

COLS_TARGET = ["sp_return"]
COND_K2_104_NO_26W   = ["tbill_wr", "excess_liq_wr"]                       # default
COND_K2_104_WITH_26W = ["tbill_wr", "tbill_26w_lag", "excess_liq_wr"]      # legacy reproduce
COND_K2_104 = COND_K2_104_NO_26W

LOG2PI = math.log(2 * math.pi)


def load_windows(csv_path, cols_cond, cols_target, L=104, stats=None):
    """Load windowed (X, C) tensors from CSV.

    stats : optional dict with keys 'mean', 'std' — train z-score parameters.
            If None, compute from current data (train mode).
            If provided, apply those train stats to current data (test mode).
            Avoids the train/test independent-normalization bug that breaks NLL calibration.
    """
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


def nll_per_step_channel(X, C, model, past_len, device):
    """Mean NLL per step per channel over future portion."""
    B, L_, D = X.shape
    F = L_ - past_len
    z, log_det_J, log_scale = model(X.to(device), C.to(device))
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale  # [B, L, D]
    nll_future = nll_td[:, past_len:, :].sum(dim=(1, 2))  # [B]
    return nll_future / (F * D)


def run(condition_name, cols_cond, train_csv, test_csv, save_dir, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n{'=' * 70}\n[{condition_name}] seed={seed} cond={cols_cond} (D_COND={len(cols_cond)})\n{'=' * 70}")

    Xtr, Ctr, stats_tr = load_windows(train_csv, cols_cond, COLS_TARGET, L=L)
    Xte, Cte, _ = load_windows(test_csv, cols_cond, COLS_TARGET, L=L, stats=stats_tr)
    print(f"  z-score stats: mean={[round(m,4) for m in stats_tr['mean']]}")
    print(f"                   std={[round(s,4) for s in stats_tr['std']]}")
    Cte_np = Cte.numpy().reshape(-1, Cte.shape[-1])
    print(f"  test z mean per channel: {[round(float(m),4) for m in Cte_np.mean(axis=0)]}")
    print(f"  test z std  per channel: {[round(float(s),4) for s in Cte_np.std(axis=0)]}")
    if Xtr is None or Xte is None:
        print("[FAIL] not enough windows")
        return None

    n_w_tr = Xtr.shape[0]
    n_val = max(int(n_w_tr * 0.15), 1)
    Xtr_, Ctr_ = Xtr[:-n_val], Ctr[:-n_val]
    Xv, Cv = Xtr[-n_val:], Ctr[-n_val:]
    print(f"  train_windows={Xtr_.shape[0]}, val_windows={Xv.shape[0]}, test_windows={Xte.shape[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiStepFAVARFlow(
        K=K_STEPS,
        d_cond=len(cols_cond),
        d_target=len(COLS_TARGET),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        time_reverse=False,
        use_wavelet=False,
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
            val_nll = nll_per_step_channel(Xv, Cv, model, PAST_LEN, device).mean().item()
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
    tag = f"{condition_name}_seed{seed}"
    ckpt_path = os.path.join(save_dir, f"{tag}_best.pt")
    log_path = os.path.join(save_dir, f"{tag}_trainlog.csv")
    summary_path = os.path.join(save_dir, f"{tag}_summary.json")

    if best_state is not None:
        torch.save({
            "model_state": best_state,
            "cond_cols": cols_cond,
            "target_cols": COLS_TARGET,
            "stats_train": stats_tr,
            "config": dict(K=K_STEPS, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                            d_cond=len(cols_cond), d_target=len(COLS_TARGET),
                            past_len=PAST_LEN, total_len=L),
        }, ckpt_path)
    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        condition=condition_name,
        seed=seed,
        n_cond=len(cols_cond),
        cond_cols=cols_cond,
        target_cols=COLS_TARGET,
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
    ap.add_argument("--test-csv", default=os.path.join(HERE, "data", "weekly_ppbond_test.csv"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--with-26w", action="store_true", help="Include tbill_26w_lag in cond (legacy reproduce)")
    args = ap.parse_args()
    if args.with_26w:
        global COND_K2_104
        COND_K2_104 = COND_K2_104_WITH_26W
        print(f"[--with-26w] cond = {COND_K2_104}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for seed in args.seeds:
        r = run("K2_104", COND_K2_104, args.train_csv, args.test_csv, args.out_dir, seed=seed)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.out_dir, "K2_104_multiseed_results.csv"), index=False)
        v_mean, v_std = df.val.mean(), df.val.std()
        t_mean, t_std = df.test.mean(), df.test.std()
        print(f"\n[K2_104] multi-seed (n={len(df)}): val={v_mean:+.4f}±{v_std:.4f}  test={t_mean:+.4f}±{t_std:.4f}")


if __name__ == "__main__":
    main()
