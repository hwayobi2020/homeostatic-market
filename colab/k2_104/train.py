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
PATIENCE = 30

COLS_TARGET = ["sp_return"]
SP_TARGET_IDX = 0
COND_K2_104_NO_26W   = ["tbill_wr", "excess_liq_wr"]                       # default 2ch (행 1)
COND_K2_104_WITH_26W = ["tbill_wr", "tbill_26w_lag", "excess_liq_wr"]      # legacy reproduce
COND_K2_104_NO_LIQ   = ["tbill_wr"]                                         # 1ch (행 2 without liquidity)
COND_K2_104 = COND_K2_104_NO_26W
# future cond mask: tbill_wr 외 channels mask (paper main thesis: 금리 시나리오만 활성)
MASK_FUTURE_CH_NO_26W   = [1]      # mask excess_liq_wr in future
MASK_FUTURE_CH_WITH_26W = [1, 2]   # mask tbill_26w_lag + excess_liq_wr (legacy)
MASK_FUTURE_CH_NO_LIQ   = []        # cond 1ch tbill only — mask 무관
MASK_FUTURE_CH = MASK_FUTURE_CH_NO_26W

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


def mask_future_channels(C, past_len, mask_channels):
    """Set future portion of specified cond channels to 0 (= train mean in z-space)."""
    C = C.clone()
    for ch in mask_channels:
        C[:, past_len:, ch] = 0.0
    return C


def nll_per_step_channel(X, C, model, past_len, device):
    """Mean NLL per step per channel over future portion."""
    B, L_, D = X.shape
    F = L_ - past_len
    z, log_det_J, log_scale = model(X.to(device), C.to(device))
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale  # [B, L, D]
    nll_future = nll_td[:, past_len:, :].sum(dim=(1, 2))  # [B]
    return nll_future / (F * D)


def run(condition_name, cols_cond, train_csv, test_csv, save_dir, seed=42,
        normalize_sp=False, val_csv=None, fold_tag=None,
        warmup_epochs=0, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tag_norm_sr = "_normsr" if normalize_sp else ""
    fold_str    = f"_{fold_tag}" if fold_tag else ""
    tag_extra   = (f"_warmup{warmup_epochs}" if warmup_epochs > 0 else "") + \
                  (f"_pat{patience}"          if patience != PATIENCE else "")
    tag_full    = f"{condition_name}{tag_norm_sr}{tag_extra}{fold_str}_seed{seed}"
    ckpt_path   = os.path.join(save_dir, f"{tag_full}_best.pt")
    summary_path_pre = os.path.join(save_dir, f"{tag_full}_summary.json")
    if os.path.exists(ckpt_path) and os.path.exists(summary_path_pre):
        print(f"[SKIP] {tag_full} — ckpt + summary 이미 존재")
        with open(summary_path_pre) as f:
            return json.load(f)
    print(f"\n{'=' * 70}\n[{condition_name}{tag_norm_sr}{fold_str}] seed={seed} cond={cols_cond} (D_COND={len(cols_cond)}) normalize_sp={normalize_sp}\n{'=' * 70}")

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

    # future cond mask 적용 (paper main thesis: 금리 시나리오만 활성)
    masked_names = [cols_cond[i] for i in MASK_FUTURE_CH]
    print(f"  mask_future_ch = {MASK_FUTURE_CH} -> {masked_names} masked in future")
    Ctr = mask_future_channels(Ctr, PAST_LEN, MASK_FUTURE_CH)
    Cte = mask_future_channels(Cte, PAST_LEN, MASK_FUTURE_CH)

    stats_targets_all = {}
    if normalize_sp:
        smu = float(Xtr[..., SP_TARGET_IDX].mean())
        ssd = float(Xtr[..., SP_TARGET_IDX].std()) + 1e-8
        Xtr[..., SP_TARGET_IDX] = (Xtr[..., SP_TARGET_IDX] - smu) / ssd
        Xte[..., SP_TARGET_IDX] = (Xte[..., SP_TARGET_IDX] - smu) / ssd
        stats_targets_all[SP_TARGET_IDX] = {"mean": smu, "std": ssd,
                                            "channel": int(SP_TARGET_IDX),
                                            "channel_name": COLS_TARGET[SP_TARGET_IDX]}
        print(f"  ★ target normalize sp_return (train): mean={smu:+.5f}, std={ssd:.5f}")

    if val_csv is not None:
        Xv, Cv, _ = load_windows(val_csv, cols_cond, COLS_TARGET, L=L, stats=stats_tr)
        Cv = mask_future_channels(Cv, PAST_LEN, MASK_FUTURE_CH)
        if normalize_sp:
            s = stats_targets_all[SP_TARGET_IDX]
            Xv[..., SP_TARGET_IDX] = (Xv[..., SP_TARGET_IDX] - s["mean"]) / s["std"]
        Xtr_, Ctr_ = Xtr, Ctr
        print(f"  train_windows={Xtr_.shape[0]} (full), val_windows={Xv.shape[0]} (val_csv), test_windows={Xte.shape[0]}")
    else:
        n_w_tr = Xtr.shape[0]
        n_val = max(int(n_w_tr * 0.15), 1)
        Xtr_, Ctr_ = Xtr[:-n_val], Ctr[:-n_val]
        Xv, Cv = Xtr[-n_val:], Ctr[-n_val:]
        print(f"  train_windows={Xtr_.shape[0]}, val_windows={Xv.shape[0]} (auto 15%), test_windows={Xte.shape[0]}")

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

        # Warmup: epoch < warmup_epochs 에선 best 갱신 무시 (early saturation 차단)
        if epoch >= warmup_epochs and val_nll < best_val - 1e-4:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test = test_nll
            best_epoch = epoch
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val={best_val:+.4f}, test={best_test:+.4f}")

    os.makedirs(save_dir, exist_ok=True)
    tag = tag_full
    log_path = os.path.join(save_dir, f"{tag}_trainlog.csv")
    summary_path = os.path.join(save_dir, f"{tag}_summary.json")

    if best_state is not None:
        torch.save({
            "model_state": best_state,
            "cond_cols": cols_cond,
            "target_cols": COLS_TARGET,
            "stats_train": stats_tr,
            "stats_targets_all": stats_targets_all,
            "mask_future_ch": MASK_FUTURE_CH,
            "normalize_sp": normalize_sp,
            "config": dict(K=K_STEPS, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                            d_cond=len(cols_cond), d_target=len(COLS_TARGET),
                            past_len=PAST_LEN, total_len=L),
        }, ckpt_path)
    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        condition=condition_name,
        fold=fold_tag,
        seed=seed,
        warmup_epochs=warmup_epochs,
        patience=patience,
        normalize_sp=normalize_sp,
        n_cond=len(cols_cond),
        cond_cols=cols_cond,
        target_cols=COLS_TARGET,
        mask_future_ch=MASK_FUTURE_CH,
        stats_targets_all={int(k): v for k, v in stats_targets_all.items()},
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
    ap.add_argument("--val-csv",   default=None)
    ap.add_argument("--test-csv", default=os.path.join(HERE, "data", "weekly_ppbond_test.csv"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--fold", default=None, choices=["F1", "F2", "F3"],
                    help="walk-forward fold")
    ap.add_argument("--normalize-sp", action="store_true",
                    help="train mean/std 로 sp_return target 정규화")
    ap.add_argument("--no-liq", action="store_true",
                    help="cond 에서 excess_liq_wr 제거 (cond=[tbill_wr] 1ch only) — 행 2 'without liquidity'")
    ap.add_argument("--with-26w", action="store_true", help="Include tbill_26w_lag in cond (legacy reproduce)")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="best ckpt 갱신 차단 구간 (early saturation 방지). default 0=끔")
    ap.add_argument("--patience", type=int, default=PATIENCE,
                    help=f"early stop patience. default {PATIENCE}")
    args = ap.parse_args()
    if args.with_26w and args.no_liq:
        raise ValueError("--with-26w 와 --no-liq 동시 사용 불가")
    global COND_K2_104, MASK_FUTURE_CH
    if args.with_26w:
        COND_K2_104 = COND_K2_104_WITH_26W
        MASK_FUTURE_CH = MASK_FUTURE_CH_WITH_26W
        condition_name = "K2_104"
        print(f"[--with-26w] cond = {COND_K2_104}, mask = {MASK_FUTURE_CH}")
    elif args.no_liq:
        COND_K2_104 = COND_K2_104_NO_LIQ
        MASK_FUTURE_CH = MASK_FUTURE_CH_NO_LIQ
        condition_name = "K2_104_2ch"   # paper 표 행 2 라벨 ("without liquidity")
        print(f"[--no-liq] cond = {COND_K2_104}, mask = {MASK_FUTURE_CH}")
    else:
        condition_name = "K2_104"        # 행 1 default

    if args.fold is not None:
        repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
        folds_dir = os.path.join(repo_root, "data", "folds")
        args.train_csv = os.path.join(folds_dir, f"{args.fold}_train.csv")
        args.val_csv   = os.path.join(folds_dir, f"{args.fold}_val.csv")
        args.test_csv  = os.path.join(folds_dir, f"{args.fold}_test.csv")
        print(f"[--fold {args.fold}] train={args.train_csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for seed in args.seeds:
        r = run(condition_name, COND_K2_104, args.train_csv, args.test_csv, args.out_dir,
                seed=seed, normalize_sp=args.normalize_sp,
                val_csv=args.val_csv, fold_tag=args.fold,
                warmup_epochs=args.warmup_epochs, patience=args.patience)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        tag_norm_sr = "_normsr" if args.normalize_sp else ""
        fold_str    = f"_{args.fold}" if args.fold else ""
        tag_extra   = (f"_warmup{args.warmup_epochs}" if args.warmup_epochs > 0 else "") + \
                      (f"_pat{args.patience}"          if args.patience != PATIENCE else "")
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.out_dir, f"{condition_name}{tag_norm_sr}{tag_extra}{fold_str}_multiseed_results.csv"), index=False)
        v_mean, v_std = df.val.mean(), df.val.std()
        t_mean, t_std = df.test.mean(), df.test.std()
        print(f"\n[{condition_name}{tag_norm_sr}{tag_extra}{fold_str}] multi-seed (n={len(df)}): val={v_mean:+.4f}±{v_std:.4f}  test={t_mean:+.4f}±{t_std:.4f}")


if __name__ == "__main__":
    main()
