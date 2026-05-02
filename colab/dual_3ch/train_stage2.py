"""Stage 2 (PriceGenerator) — Paired test vs Base K2_104.

Architecture / cond / target / training: identical to colab/k2_104/train.py (Base).
ONLY DIFFERENCE: future portion of cond ch=2 (excess_liq_wr) is REPLACED by
Stage 1's deterministic mean prediction (z_future=0 inverse). No Bridge needed —
Stage 1 target == Stage 2 cond ch=2 (둘 다 excess_liq_wr weekly raw).

Paired with same-seed Stage 1 ckpt (stage1_seed{S}_best.pt → train_stage2 seed S).
"""
import torch  # MUST be first — Windows DLL load-order workaround

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

# ── 고정 하이퍼파라미터 (Base K2_104 와 정확히 동일) ────────
L = 104
PAST_LEN = 52
K_STEPS = 2
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
LR = 5e-4
BATCH = 32
MAX_EPOCHS = 60
PATIENCE = 30

COLS_TARGET = ["sp_return"]
COLS_COND_NO_26W   = ["tbill_wr", "excess_liq_wr"]                       # default
COLS_COND_WITH_26W = ["tbill_wr", "tbill_26w_lag", "excess_liq_wr"]      # legacy reproduce
COLS_COND = COLS_COND_NO_26W

STAGE1_COLS_COND_NO_26W      = ["tbill_wr", "excess_liq_wr"]
STAGE1_COLS_COND_WITH_26W    = ["tbill_wr", "tbill_26w_lag", "excess_liq_wr"]
STAGE1_MASK_FUTURE_CH_NO_26W   = [1]
STAGE1_MASK_FUTURE_CH_WITH_26W = [1, 2]
STAGE1_COLS_COND      = STAGE1_COLS_COND_NO_26W
STAGE1_MASK_FUTURE_CH = STAGE1_MASK_FUTURE_CH_NO_26W
EXCESS_LIQ_CH_IDX     = 1   # default 2ch layout

LOG2PI = math.log(2 * math.pi)


def load_window_arrays(csv_path, cols_cond, cols_target, L=104):
    """Returns raw target and raw cond tensors (NOT z-scored)."""
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


def compute_train_stats(C_raw):
    flat = C_raw.reshape(-1, C_raw.shape[-1]).numpy()
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0) + 1e-8
    return {"mean": mu.tolist(), "std": sd.tolist()}


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


def load_stage1(stage1_ckpt_path, device):
    ckpt = torch.load(stage1_ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MultiStepFAVARFlow(
        K=cfg["K"], d_cond=cfg["d_cond"], d_target=cfg["d_target"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        time_reverse=False, use_wavelet=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["stats_train"]


@torch.no_grad()
def stage1_generate_mean(stage1_model, past_target_raw, cond_z_full, past_len, device):
    """Stage 1 deterministic mean: forward + future z=0 + inverse → mean future target.

    past_target_raw : [B, P, 1]   raw past excess_liq_wr (Stage 1 target)
    cond_z_full     : [B, L, 3]   z-scored cond (Stage 1 layout, future ch=1,2 masked)

    Returns: [B, F, 1] raw future excess_liq_wr (model's mean prediction)
    """
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
    """For each window, run Stage 1 → replace future ch=2 of cond_raw with Stage 1 mean.

    cond_raw_full : [N, L, 3]   raw cond
    Returns       : [N, L, 3]   raw cond with future ch=2 replaced by Stage 1 mean prediction
    """
    N = cond_raw_full.shape[0]
    cond_modified = cond_raw_full.clone()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        cond_batch_raw = cond_raw_full[start:end]   # [b, L, 3]

        # Stage 1 cond: same raw cond, z-scored with Stage 1 stats, future channels [1,2] masked to 0
        s1_cond_z = zscore(cond_batch_raw, stage1_stats)
        s1_cond_z = mask_future_channels(s1_cond_z, past_len, STAGE1_MASK_FUTURE_CH)

        # Stage 1 past target = past portion of cond ch=EXCESS_LIQ_CH_IDX (raw, observed)
        past_target_raw = cond_batch_raw[:, :past_len, EXCESS_LIQ_CH_IDX:EXCESS_LIQ_CH_IDX + 1]
        future_target_gen = stage1_generate_mean(
            stage1_model, past_target_raw, s1_cond_z, past_len, device
        )                                                        # [b, F, 1]

        cond_modified[start:end, past_len:, EXCESS_LIQ_CH_IDX] = future_target_gen[:, :, 0]
    return cond_modified


def run(seed, train_csv, test_csv, save_dir, stage1_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n{'=' * 70}")
    print(f"[Stage2] seed={seed}")
    print(f"  cond   = {COLS_COND}")
    print(f"  target = {COLS_TARGET}")
    print(f"  Stage 1 ckpt = stage1_seed{seed}_best.pt (paired)")
    print(f"{'=' * 70}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, Ctr_raw = load_window_arrays(train_csv, COLS_COND, COLS_TARGET, L=L)
    Xte, Cte_raw = load_window_arrays(test_csv,  COLS_COND, COLS_TARGET, L=L)
    if Xtr is None or Xte is None:
        print("[FAIL] not enough windows")
        return None

    stats_tr = compute_train_stats(Ctr_raw)   # Base z-score stats (3ch)
    print(f"  Base z-score stats: mean={[round(m,5) for m in stats_tr['mean']]}")
    print(f"                        std={[round(s,5) for s in stats_tr['std']]}")

    # Load Stage 1 (paired seed)
    stage1_ckpt = os.path.join(stage1_dir, f"stage1_seed{seed}_best.pt")
    if not os.path.exists(stage1_ckpt):
        print(f"[FAIL] Stage 1 ckpt missing: {stage1_ckpt}")
        return None
    stage1_model, stage1_stats = load_stage1(stage1_ckpt, device)
    print(f"  loaded Stage 1: {stage1_ckpt}")

    # Replace future ch=2 with Stage 1 mean prediction (no Bridge)
    print(f"  Stage 1 generation → train windows ...")
    Ctr_raw_mod = replace_future_with_stage1(
        Ctr_raw, stage1_model, stage1_stats, PAST_LEN, device
    )
    print(f"  Stage 1 generation → test windows ...")
    Cte_raw_mod = replace_future_with_stage1(
        Cte_raw, stage1_model, stage1_stats, PAST_LEN, device
    )

    # Diagnostic: how different is generated future ch=2 from observed?
    diff_train = (Ctr_raw_mod - Ctr_raw)[:, PAST_LEN:, EXCESS_LIQ_CH_IDX]
    diff_test  = (Cte_raw_mod - Cte_raw)[:, PAST_LEN:, EXCESS_LIQ_CH_IDX]
    print(f"  cond ch={EXCESS_LIQ_CH_IDX} future diff (gen − obs): "
          f"train mean={float(diff_train.mean()):+.5f} std={float(diff_train.std()):.5f}, "
          f"test  mean={float(diff_test.mean()):+.5f} std={float(diff_test.std()):.5f}")

    # z-score with Base train stats
    Ctr = zscore(Ctr_raw_mod, stats_tr)
    Cte = zscore(Cte_raw_mod, stats_tr)
    Cte_np = Cte.numpy().reshape(-1, Cte.shape[-1])
    print(f"  test z mean per channel: {[round(float(m),4) for m in Cte_np.mean(axis=0)]}")
    print(f"  test z std  per channel: {[round(float(s),4) for s in Cte_np.std(axis=0)]}")

    # Train/val split (last 15% of train as val)
    n_w_tr = Xtr.shape[0]
    n_val = max(int(n_w_tr * 0.15), 1)
    Xtr_, Ctr_ = Xtr[:-n_val], Ctr[:-n_val]
    Xv,  Cv  = Xtr[-n_val:], Ctr[-n_val:]
    print(f"  train_windows={Xtr_.shape[0]}, val_windows={Xv.shape[0]}, test_windows={Xte.shape[0]}")

    # Stage 2 model (same architecture as Base K2_104)
    model = MultiStepFAVARFlow(
        K=K_STEPS, d_cond=len(COLS_COND), d_target=len(COLS_TARGET),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        time_reverse=False, use_wavelet=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Stage 2 params={n_params:,}, device={device}")

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
    tag = f"stage2_seed{seed}"
    ckpt_path    = os.path.join(save_dir, f"{tag}_best.pt")
    log_path     = os.path.join(save_dir, f"{tag}_trainlog.csv")
    summary_path = os.path.join(save_dir, f"{tag}_summary.json")

    if best_state is not None:
        torch.save({
            "model_state":   best_state,
            "cond_cols":     COLS_COND,
            "target_cols":   COLS_TARGET,
            "stats_train":   stats_tr,
            "stage1_ckpt":   stage1_ckpt,
            "config": dict(K=K_STEPS, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                           d_cond=len(COLS_COND), d_target=len(COLS_TARGET),
                           past_len=PAST_LEN, total_len=L),
        }, ckpt_path)
    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        stage="stage2",
        seed=seed,
        cond_cols=COLS_COND,
        target_cols=COLS_TARGET,
        stage1_ckpt=stage1_ckpt,
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
    ap.add_argument("--train-csv",  default=os.path.join(HERE, "data", "weekly_ppbond_train.csv"))
    ap.add_argument("--test-csv",   default=os.path.join(HERE, "data", "weekly_ppbond_test.csv"))
    ap.add_argument("--out-dir",    default=os.path.join(HERE, "result"))
    ap.add_argument("--stage1-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--with-26w", action="store_true", help="Include tbill_26w_lag in cond")
    args = ap.parse_args()
    if args.with_26w:
        global COLS_COND, STAGE1_COLS_COND, STAGE1_MASK_FUTURE_CH, EXCESS_LIQ_CH_IDX
        COLS_COND = COLS_COND_WITH_26W
        STAGE1_COLS_COND = STAGE1_COLS_COND_WITH_26W
        STAGE1_MASK_FUTURE_CH = STAGE1_MASK_FUTURE_CH_WITH_26W
        EXCESS_LIQ_CH_IDX = 2   # 3ch layout
        print(f"[--with-26w] cond = {COLS_COND}, EXCESS_LIQ_CH_IDX = {EXCESS_LIQ_CH_IDX}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for seed in args.seeds:
        r = run(seed, args.train_csv, args.test_csv, args.out_dir, args.stage1_dir)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.out_dir, "stage2_multiseed_results.csv"), index=False)
        v_mean, v_std = df.val.mean(),  df.val.std()
        t_mean, t_std = df.test.mean(), df.test.std()
        v_med, t_med  = df.val.median(), df.test.median()
        print(f"\n[Stage2] multi-seed (n={len(df)})")
        print(f"  val:  mean={v_mean:+.4f} ± {v_std:.4f}  median={v_med:+.4f}")
        print(f"  test: mean={t_mean:+.4f} ± {t_std:.4f}  median={t_med:+.4f}")


if __name__ == "__main__":
    main()
