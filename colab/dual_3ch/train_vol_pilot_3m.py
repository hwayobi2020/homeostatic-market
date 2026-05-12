"""Vol pilot 3M — 3개월 (13주) vol prediction.

vol_pilot_v2 (52주 vol) 학습 결과 R²(calibrated) ≈ −0.05 (본질적 학습 X) 였음.
1년 vol 자체가 macro 변수만으로 학습 어려움 가설 → horizon 단축 (3M).

Target:
  y = log( std(sp_return future 13w) ) per origin (단일 scalar)

Window:
  past 52w + future 13w = L=65 (1년 macro context 유지, 3개월 ahead vol)

Architecture / Loss / Eval: vol_pilot_v2 동일 (변경: future_len, L 만)
ckpt prefix: vol_pilot_3m_*
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
import torch.nn as nn

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# === 3M-specific ===
PAST_LEN = 104    # 2y
FUTURE_LEN = 13   # 3 months
L = PAST_LEN + FUTURE_LEN  # 117
# ===================

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
LR = 1e-4
BATCH = 32
MAX_EPOCHS = 60
PATIENCE = 30
GRAD_CLIP = 1.0
MIN_EPOCH = 20   # selection 시 best_epoch 가 이 값 이상이어야 함 — ep < 20 의 partial-learning 함정 차단 (F1 의 ep 12 케이스 등)

# 3M (13w) horizon 일치 macro 변수 — gdp_13w_proxy_lag (yoy 변환의 56w lookback) 제거,
# ADS Business Conditions Index (daily, lag 1w) 로 GDP 대체. fold gap 15w 와 정합.
# sp_std_13w (raw past 13w std, log 안 한 것) 도 cond 에 추가 — magnitude direct read-out.
COLS_COND_BASE        = ["tbill_wr",        "m2_13w_cum_lag", "ads_lag", "cpi_13w_cum_lag", "sp_std_13w"]
COLS_COND_BASE_VIX    = ["tbill_wr", "vix", "m2_13w_cum_lag", "ads_lag", "cpi_13w_cum_lag", "sp_std_13w"]

VARIANTS = {
    1:  dict(name="base",                cond_extra=[]),
    2:  dict(name="base_bp",             cond_extra=["bondpp_13w_lag"]),
    3:  dict(name="base_sp",             cond_extra=["stockpp_13w_lag"]),
    4:  dict(name="base_bp_sp",          cond_extra=["bondpp_13w_lag", "stockpp_13w_lag"]),
    10: dict(name="base_bondsum",          cond_extra=["tbill_13w_cum"]),
    11: dict(name="base_stocksum",         cond_extra=["sp_13w_cum"]),
    12: dict(name="base_bondsum_stocksum", cond_extra=["tbill_13w_cum", "sp_13w_cum"]),
    13: dict(name="base_metab_ads_wti_har1_13",
             cond_extra=["metab_13w", "wti_wr", "sp_log_std_13w"]),
        # sp_log_std_13w 단일 horizon (HAR-RV 와 동일, 공정 비교).
        # 4 horizon (har4) 은 fold gap 53w 필요 → MTL 도입 후 별도 실험으로.
}


def build_spec(variant_id, with_vix=False):
    if variant_id not in VARIANTS:
        raise ValueError(f"variant must be in {list(VARIANTS)}, got {variant_id}")
    v = VARIANTS[variant_id]
    base = COLS_COND_BASE_VIX if with_vix else COLS_COND_BASE
    cols_cond = base + v["cond_extra"]
    cols_target = ["sp_return"]
    mask_future_ch = list(range(1, len(cols_cond)))
    return dict(
        variant_id=variant_id,
        name=v["name"],
        cols_cond=cols_cond,
        cols_target=cols_target,
        mask_future_ch=mask_future_ch,
        with_vix=with_vix,
    )


# =====================================================================
# Model — vol_pilot_v2 동일 (future_len 만 다름)
# =====================================================================

class CausalTransformerVolScalar(nn.Module):
    def __init__(self, d_cond, d_model, n_heads, n_layers,
                 past_len, future_len, dropout=0.1):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.total_len = past_len + future_len

        self.input_dim = d_cond + 1
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.pos_emb = nn.Embedding(self.total_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        causal_mask = torch.triu(
            torch.ones(self.total_len, self.total_len), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.scalar_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, cond, target_past):
        B = cond.shape[0]
        device = cond.device

        target_full = torch.zeros(B, self.total_len, 1, dtype=cond.dtype, device=device)
        target_full[:, :self.past_len, :] = target_past

        x = torch.cat([cond, target_full], dim=-1)
        x = self.input_proj(x)

        pos = torch.arange(self.total_len, device=device)
        x = x + self.pos_emb(pos).unsqueeze(0)

        h = self.encoder(x, mask=self.causal_mask, is_causal=True)
        h_future = h[:, self.past_len:, :].mean(dim=1)
        return self.scalar_head(h_future).squeeze(-1)


# =====================================================================
# Data
# =====================================================================

def load_windows(csv_path, cols_cond, cols_target, L=65, cond_stats=None):
    df = pd.read_csv(csv_path)
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        return None, None, None, None
    X = np.zeros((n_w, L, len(cols_target)), dtype=np.float32)
    C = np.zeros((n_w, L, len(cols_cond)),   dtype=np.float32)
    for i in range(n_w):
        X[i] = df[cols_target].iloc[i : i + L].values
        C[i] = df[cols_cond].iloc[i : i + L].values

    if cond_stats is None:
        cmu = C.reshape(-1, len(cols_cond)).mean(axis=0)
        csd = C.reshape(-1, len(cols_cond)).std(axis=0) + 1e-8
        cond_stats_out = {"mean": cmu.tolist(), "std": csd.tolist()}
    else:
        cmu = np.asarray(cond_stats["mean"], dtype=np.float32)
        csd = np.asarray(cond_stats["std"],  dtype=np.float32)
        cond_stats_out = cond_stats
    C = (C - cmu) / csd

    sp_idx = cols_target.index("sp_return")
    future_sp = X[:, PAST_LEN:, sp_idx]
    actual_std = future_sp.std(axis=1, ddof=1)
    actual_std = np.maximum(actual_std, 1e-8)
    y_log_std = np.log(actual_std).astype(np.float32)

    X_past = X[:, :PAST_LEN, :]

    return (torch.from_numpy(X_past),
            torch.from_numpy(C),
            torch.from_numpy(y_log_std),
            cond_stats_out)


def mask_future_channels(C, past_len, mask_channels):
    C = C.clone()
    for ch in mask_channels:
        C[:, past_len:, ch] = 0.0
    return C


def spearman_corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    ra = pd.Series(a).rank().values - pd.Series(a).rank().mean()
    rb = pd.Series(b).rank().values - pd.Series(b).rank().mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def pearson_corr(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


def ic_loss(pred, target, eps=1e-8):
    """IC loss = 1 − Pearson(pred, target) — torch differentiable.

    Batch-level Pearson correlation. 최소화하면 Pearson 최대화.
    Scale-invariant: magnitude 학습 안 함, rank 만 최적화.
    Quant finance literature 의 IC (Information Coefficient) loss 표준 형태.

    참고: Gu, Kelly, Xiu 2020 *RFS* "Empirical Asset Pricing via ML"
    """
    pred_c   = pred - pred.mean()
    target_c = target - target.mean()
    cov      = (pred_c * target_c).mean()
    p_std    = pred_c.pow(2).mean().clamp_min(eps).sqrt()
    t_std    = target_c.pow(2).mean().clamp_min(eps).sqrt()
    corr     = cov / (p_std * t_std)
    return 1.0 - corr


# =====================================================================
# Train
# =====================================================================

def run(spec, train_csv, val_csv, test_csv, save_dir, seed,
        max_epochs, patience, batch, lr, fold_tag, loss_mode="mse"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cols_cond   = spec["cols_cond"]
    cols_target = spec["cols_target"]
    mask_future = spec["mask_future_ch"]

    fold_str = f"_{fold_tag}" if fold_tag else ""
    vix_tag  = "_vix" if spec.get("with_vix") else ""
    # sel_tag determined by loss_mode (mse or ic)
    sel_tag = "_isel" if loss_mode == "ic" else "_msel"
    tag_full = f"vol_pilot_3m{sel_tag}{vix_tag}_v{spec['variant_id']}_{spec['name']}{fold_str}_seed{seed}"
    ckpt_path    = os.path.join(save_dir, f"{tag_full}_best.pt")
    summary_path = os.path.join(save_dir, f"{tag_full}_summary.json")
    log_path     = os.path.join(save_dir, f"{tag_full}_trainlog.csv")
    pred_path    = os.path.join(save_dir, f"{tag_full}_test_preds.npz")

    if os.path.exists(ckpt_path) and os.path.exists(summary_path):
        print(f"[SKIP] {tag_full} — ckpt + summary 이미 존재")
        with open(summary_path) as f:
            return json.load(f)

    print(f"\n{'='*72}")
    print(f"[Variant {spec['variant_id']} = {spec['name']}{fold_str}] seed={seed}  "
          f"(Vol scalar regression — 3M target, past={PAST_LEN}+future={FUTURE_LEN}, L={L})")
    print(f"  cond   = {cols_cond}  (D_COND={len(cols_cond)})")
    print(f"  target = log( std(future 13w sp_return) )  — 단일 scalar per origin")
    print(f"  mask_future_ch = {mask_future}")
    print(f"{'='*72}")

    Xtr_past, Ctr, ytr, stats_c = load_windows(train_csv, cols_cond, cols_target, L=L)
    if Xtr_past is None:
        print(f"[FAIL] not enough train windows: {train_csv}")
        return None
    Xte_past, Cte, yte, _ = load_windows(test_csv, cols_cond, cols_target, L=L, cond_stats=stats_c)
    if Xte_past is None:
        print(f"[FAIL] not enough test windows: {test_csv}")
        return None

    Ctr = mask_future_channels(Ctr, PAST_LEN, mask_future)
    Cte = mask_future_channels(Cte, PAST_LEN, mask_future)

    if val_csv is not None and os.path.exists(val_csv):
        Xv_past, Cv, yv, _ = load_windows(val_csv, cols_cond, cols_target, L=L, cond_stats=stats_c)
        if Xv_past is None:
            print(f"[FAIL] not enough val windows: {val_csv}")
            return None
        Cv = mask_future_channels(Cv, PAST_LEN, mask_future)
        Xtr_past_, Ctr_, ytr_ = Xtr_past, Ctr, ytr
        print(f"  train_w={Xtr_past_.shape[0]} (full), val_w={Xv_past.shape[0]} (val_csv), "
              f"test_w={Xte_past.shape[0]}")
    else:
        n_w_tr = Xtr_past.shape[0]
        n_val = max(int(n_w_tr * 0.15), 1)
        Xtr_past_, Ctr_, ytr_ = Xtr_past[:-n_val], Ctr[:-n_val], ytr[:-n_val]
        Xv_past,   Cv,   yv   = Xtr_past[-n_val:], Ctr[-n_val:], ytr[-n_val:]
        print(f"  train_w={Xtr_past_.shape[0]}, val_w={Xv_past.shape[0]} (auto 15%), "
              f"test_w={Xte_past.shape[0]}")

    y_train_mean = float(ytr_.mean())
    baseline_test_mse = float(((yte.numpy() - y_train_mean) ** 2).mean())
    print(f"  log_std baseline (train mean) = {y_train_mean:+.4f}")
    print(f"  baseline test MSE (constant pred) = {baseline_test_mse:.6f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformerVolScalar(
        d_cond=len(cols_cond),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        past_len=PAST_LEN, future_len=FUTURE_LEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # loss_mode 에 따라 selection 기준 다름:
    #   "mse": val_mse_min (lower better). loss = MSE.
    #   "ic" : val_pearson_max (higher better). loss = 1 - Pearson.
    best_val_mse = float("inf")
    best_val_pearson = -float("inf")
    best_val_mse_at_best = float("nan")
    best_val_pearson_at_best = float("nan")
    print(f"  loss_mode: {loss_mode}  (selection: {'val_pearson_max' if loss_mode=='ic' else 'val_mse_min'})")
    best_state = None
    best_epoch = -1
    best_test_mse = None
    best_test_r2 = None
    best_test_pearson = None
    best_test_spearman = None
    best_test_pred = None
    best_test_actual = None
    pat = 0
    log = []

    def eval_split(X_past, C, y):
        model.eval()
        with torch.no_grad():
            pred = model(C.to(device), X_past.to(device)).cpu().numpy()
            actual = y.numpy()
            mse = float(((pred - actual) ** 2).mean())
            pred_std = np.exp(pred)
            actual_std = np.exp(actual)
            pearson = pearson_corr(pred_std, actual_std)
            spearman = spearman_corr(pred_std, actual_std)
        return mse, pearson, spearman, pred, actual

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(Xtr_past_.shape[0])
        losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            xp = Xtr_past_[idx].to(device)
            cb = Ctr_[idx].to(device)
            yb = ytr_[idx].to(device)
            pred = model(cb, xp)
            if loss_mode == "ic":
                loss = ic_loss(pred, yb)
            else:
                loss = ((pred - yb) ** 2).mean()   # MSE
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_mse,  val_p,  val_s,  _, _ = eval_split(Xv_past,  Cv,  yv)
        test_mse, test_p, test_s, test_pred, test_actual = eval_split(Xte_past, Cte, yte)
        val_r2  = 1.0 - val_mse  / float(((yv.numpy()  - y_train_mean) ** 2).mean() + 1e-12)
        test_r2 = 1.0 - test_mse / baseline_test_mse if baseline_test_mse > 0 else float("nan")

        print(f"  ep{epoch:>3d}  train_mse={train_loss:.5f}  "
              f"val_mse={val_mse:.5f} (R²={val_r2:+.3f})  "
              f"test_mse={test_mse:.5f} (R²={test_r2:+.3f})  "
              f"test_pearson={test_p:+.3f}  test_spearman={test_s:+.3f}")

        log.append(dict(epoch=epoch, train_mse=train_loss,
                        val_mse=val_mse, val_r2=val_r2,
                        test_mse=test_mse, test_r2=test_r2,
                        test_pearson=test_p, test_spearman=test_s))

        # Selection: loss_mode 에 따라
        # min_epoch 제약: epoch < MIN_EPOCH 동안은 selection 안 함
        improved = False
        if epoch < MIN_EPOCH:
            pat = 0  # reset patience to prevent early stop
        elif loss_mode == "ic":
            if not np.isfinite(val_p):
                pat += 1
            elif val_p > best_val_pearson + 1e-4:
                best_val_pearson = val_p
                best_val_mse_at_best = val_mse
                improved = True
            else:
                pat += 1
        else:  # mse
            if not np.isfinite(val_mse):
                pat += 1
            elif val_mse < best_val_mse - 1e-6:
                best_val_mse = val_mse
                best_val_pearson_at_best = val_p
                improved = True
            else:
                pat += 1
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test_mse = test_mse
            best_test_r2 = test_r2
            best_test_pearson = test_p
            best_test_spearman = test_s
            best_test_pred = test_pred
            best_test_actual = test_actual
            best_epoch = epoch
            pat = 0

        if (not np.isfinite(test_mse)) or test_mse > 100.0:
            print(f"  ⚠ divergence (test_mse={test_mse:.2f}) — early termination")
            break
        if pat >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val_mse={best_val_mse:.5f}, val_pearson={best_val_pearson_at_best:+.4f}")
    print(f"    test MSE   = {best_test_mse:.5f}")
    print(f"    test R²    = {best_test_r2:+.4f}")
    print(f"    test Pearson  (std)  = {best_test_pearson:+.4f}")
    print(f"    test Spearman (std)  = {best_test_spearman:+.4f}")
    print(f"    baseline test MSE   = {baseline_test_mse:.5f}")

    os.makedirs(save_dir, exist_ok=True)
    if best_state is not None:
        torch.save({
            "model_state":     best_state,
            "model_type":      "causal_transformer_vol_scalar_3m",
            "cond_cols":       cols_cond,
            "target_cols":     cols_target,
            "stats_cond":      stats_c,
            "mask_future_ch":  mask_future,
            "config": dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                           d_cond=len(cols_cond), past_len=PAST_LEN,
                           future_len=FUTURE_LEN, total_len=L),
            "variant": dict(id=spec["variant_id"], name=spec["name"]),
        }, ckpt_path)
        np.savez(pred_path,
                 y_pred_log_std=best_test_pred,
                 y_actual_log_std=best_test_actual,
                 y_pred_std=np.exp(best_test_pred),
                 y_actual_std=np.exp(best_test_actual),
                 y_train_log_std_mean=y_train_mean,
                 baseline_test_mse=baseline_test_mse)
        print(f"  saved test preds: {pred_path}")

    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        model_type="causal_transformer_vol_scalar_3m",
        variant_id=spec["variant_id"],
        variant_name=spec["name"],
        fold=fold_tag,
        seed=seed,
        cond_cols=cols_cond,
        target_cols=cols_target,
        mask_future_ch=mask_future,
        future_len=FUTURE_LEN,
        loss_mode=loss_mode,
        selection_criterion=("val_pearson_max" if loss_mode == "ic" else "val_mse_min"),
        best_epoch=best_epoch,
        val_mse=best_val_mse if loss_mode != "ic" else best_val_mse_at_best,
        val_pearson_at_best=best_val_pearson_at_best if loss_mode != "ic" else best_val_pearson,
        test_mse=best_test_mse,
        test_r2=best_test_r2,
        test_pearson_std=best_test_pearson,
        test_spearman_std=best_test_spearman,
        y_train_log_std_mean=y_train_mean,
        baseline_test_mse=baseline_test_mse,
        n_params=n_params,
        device=str(device),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {ckpt_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Vol pilot 3M — 13w future vol")
    ap.add_argument("--variant", type=int, default=1, choices=list(VARIANTS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--fold", default=None)
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--val-csv",   default=None)
    ap.add_argument("--test-csv",  default=None)
    ap.add_argument("--out-dir",   default=os.path.join(HERE, "result"))
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience",   type=int, default=PATIENCE)
    ap.add_argument("--batch",      type=int, default=BATCH)
    ap.add_argument("--lr",         type=float, default=LR)
    ap.add_argument("--vix", action="store_true")
    ap.add_argument("--pilot-split", action="store_true")
    ap.add_argument("--loss-mode", choices=["mse", "ic"], default="mse",
                    help="Loss / selection mode. mse: MSE loss + val_mse_min. ic: 1-Pearson loss + val_pearson_max.")
    args = ap.parse_args()

    spec = build_spec(args.variant, with_vix=args.vix)

    train_csv = val_csv = test_csv = None
    if args.pilot_split:
        repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
        split_dir = os.path.join(repo_root, "data", "pilot_split")
        train_csv = os.path.join(split_dir, "train.csv")
        val_csv   = os.path.join(split_dir, "val.csv")
        test_csv  = os.path.join(split_dir, "test.csv")
        print(f"[--pilot-split]")
    elif args.fold is not None:
        repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
        # 항상 folds_v33_vix_expanding 디렉토리 사용 (vix 컬럼 포함하지만 --vix 안 쓰면 cond 에 안 들어감)
        folds_dir = os.path.join(repo_root, "data", "folds_v33_vix_expanding")
        train_csv = os.path.join(folds_dir, f"{args.fold}_train.csv")
        val_csv   = os.path.join(folds_dir, f"{args.fold}_val.csv")
        test_csv  = os.path.join(folds_dir, f"{args.fold}_test.csv")
        print(f"[--fold {args.fold}]  ({folds_dir})")

    if args.train_csv: train_csv = args.train_csv
    if args.val_csv:   val_csv   = args.val_csv
    if args.test_csv:  test_csv  = args.test_csv

    if train_csv is None or test_csv is None:
        ap.error("Provide --pilot-split or --fold or explicit --train-csv/--test-csv")

    os.makedirs(args.out_dir, exist_ok=True)

    fold_tag = "pilot" if args.pilot_split else args.fold

    for seed in args.seeds:
        run(spec, train_csv, val_csv, test_csv, args.out_dir, seed,
            max_epochs=args.max_epochs, patience=args.patience,
            batch=args.batch, lr=args.lr, fold_tag=fold_tag,
            loss_mode=args.loss_mode)


if __name__ == "__main__":
    main()
