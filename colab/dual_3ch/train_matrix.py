"""Paper matrix 7-variant trainer (v33 data).

paper_plan.txt 매트릭스에 정확 매핑되는 통합 학습 스크립트.

  | # | model                | cond past+future                                      | target                          |
  |---|----------------------|-------------------------------------------------------|---------------------------------|
  | 1 | base                 | tbill, m2_yoy, gdp_yoy, cpi_yoy                       | sp                              |
  | 2 | base bondpp          | + bondpp_13w_lag                                      | sp                              |
  | 3 | base stockpp         | + stockpp_13w_lag                                     | sp                              |
  | 4 | base bondpp+stockpp  | + bondpp_13w_lag + stockpp_13w_lag                    | sp                              |
  | 5 | mtl bondpp           | tbill, m2_yoy, gdp_yoy, cpi_yoy                       | sp + bondpp_13w_lag             |
  | 6 | mtl stockpp          | tbill, m2_yoy, gdp_yoy, cpi_yoy                       | sp + stockpp_13w_lag            |
  | 7 | mtl bondpp+stockpp   | tbill, m2_yoy, gdp_yoy, cpi_yoy                       | sp + bondpp_13w_lag + stockpp_13w_lag |

future cond mask: tbill_wr (idx=0) 만 활성, 나머지 모두 mask=0 (paper main thesis: 금리 시나리오만).

target 정규화 (이전 결정 — 05-02 normalization unified table):
  - sp_return: raw 유지 (paper main metric, fold/variant 간 비교 일관)
  - bondpp_13w_lag (target 일 때): --normalize-bondpp 로 train mean/std z-score
  - stockpp_13w_lag (target 일 때): --normalize-stockpp 로 train mean/std z-score
  - cond 채널은 모두 load_windows 내부에서 자동 z-score (cond_stats)

best ckpt 기준: sp_return 단독 val NLL (raw, mtl 변종도 sp 채널만 — paper 비교 일관)
"""
import torch  # MUST be first (Windows DLL load order)

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
PATIENCE = 30

LOG2PI = math.log(2 * math.pi)

# 매트릭스 base cond = 4채널, future tbill (idx 0) 만 활성
COLS_COND_BASE = ["tbill_wr", "m2_yoy_lag", "gdp_yoy_lag", "cpi_yoy_lag"]

# 7 변종 정의 — paper_plan.txt 매트릭스 1:1 매핑
VARIANTS = {
    1: dict(name="base",          cond_extra=[],                                       target_extra=[]),
    2: dict(name="base_bp",       cond_extra=["bondpp_13w_lag"],                       target_extra=[]),
    3: dict(name="base_sp",       cond_extra=["stockpp_13w_lag"],                      target_extra=[]),
    4: dict(name="base_bp_sp",    cond_extra=["bondpp_13w_lag", "stockpp_13w_lag"],    target_extra=[]),
    5: dict(name="mtl_bp",        cond_extra=[],                                       target_extra=["bondpp_13w_lag"]),
    6: dict(name="mtl_sp",        cond_extra=[],                                       target_extra=["stockpp_13w_lag"]),
    7: dict(name="mtl_bp_sp",     cond_extra=[],                                       target_extra=["bondpp_13w_lag", "stockpp_13w_lag"]),
}


def build_spec(variant_id):
    if variant_id not in VARIANTS:
        raise ValueError(f"variant must be in 1..7, got {variant_id}")
    v = VARIANTS[variant_id]
    cols_cond = COLS_COND_BASE + v["cond_extra"]
    cols_target = ["sp_return"] + v["target_extra"]
    # future cond mask: idx 0 (tbill) 만 살림, 나머지 전부 mask=0
    mask_future_ch = list(range(1, len(cols_cond)))
    return dict(
        variant_id=variant_id,
        name=v["name"],
        cols_cond=cols_cond,
        cols_target=cols_target,
        mask_future_ch=mask_future_ch,
    )


def load_windows(csv_path, cols_cond, cols_target, L=104,
                 cond_stats=None, target_stats_map=None, normalize_target_idxs=()):
    """Sliding window loader.

    cond:   train mean/std z-score 항상 적용 (cond_stats 가 None 이면 학습용으로 산출)
    target: normalize_target_idxs 에 명시된 채널만 train mean/std z-score
            (sp_return idx 0 은 기본 미포함 = raw 유지)
    """
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

    # cond z-score (always)
    if cond_stats is None:
        cmu = C.reshape(-1, len(cols_cond)).mean(axis=0)
        csd = C.reshape(-1, len(cols_cond)).std(axis=0) + 1e-8
        cond_stats_out = {"mean": cmu.tolist(), "std": csd.tolist()}
    else:
        cmu = np.asarray(cond_stats["mean"], dtype=np.float32)
        csd = np.asarray(cond_stats["std"],  dtype=np.float32)
        cond_stats_out = cond_stats
    C = (C - cmu) / csd

    # target normalize (only specified idxs)
    target_stats_map_out = {}
    for idx in normalize_target_idxs:
        if target_stats_map is not None and idx in target_stats_map:
            tmu = target_stats_map[idx]["mean"]
            tsd = target_stats_map[idx]["std"]
        else:
            tmu = float(X[..., idx].mean())
            tsd = float(X[..., idx].std()) + 1e-8
        X[..., idx] = (X[..., idx] - tmu) / tsd
        target_stats_map_out[idx] = {"mean": tmu, "std": tsd,
                                     "channel": int(idx),
                                     "channel_name": cols_target[idx]}

    return torch.from_numpy(X), torch.from_numpy(C), cond_stats_out, target_stats_map_out


def mask_future_channels(C, past_len, mask_channels):
    C = C.clone()
    for ch in mask_channels:
        C[:, past_len:, ch] = 0.0
    return C


def nll_per_step_per_channel(X, C, model, past_len, device):
    z, log_det_J, log_scale = model(X.to(device), C.to(device))
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale  # [B, L, D]
    nll_future = nll_td[:, past_len:, :]                 # [B, F, D]
    nll_per_ch = nll_future.mean(dim=(0, 1)).detach().cpu().numpy()  # [D]
    nll_mean = float(nll_per_ch.mean())
    return nll_mean, nll_per_ch


def loss_fn(X, C, model, past_len, device):
    z, log_det_J, log_scale = model(X.to(device), C.to(device))
    F_ = X.shape[1] - past_len
    D  = X.shape[2]
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale
    nll_future = nll_td[:, past_len:, :].sum(dim=(1, 2))
    return (nll_future / (F_ * D)).mean()


def run(spec, train_csv, val_csv, test_csv, save_dir, seed,
        max_epochs, patience, batch, lr, fold_tag,
        normalize_bondpp, normalize_stockpp):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cols_cond   = spec["cols_cond"]
    cols_target = spec["cols_target"]
    mask_future = spec["mask_future_ch"]

    # target idxs to normalize — only when the variable is in target
    norm_idxs = []
    apply_normbp = bool(normalize_bondpp and "bondpp_13w_lag" in cols_target)
    apply_normsp = bool(normalize_stockpp and "stockpp_13w_lag" in cols_target)
    if apply_normbp:
        norm_idxs.append(cols_target.index("bondpp_13w_lag"))
    if apply_normsp:
        norm_idxs.append(cols_target.index("stockpp_13w_lag"))

    sp_idx = cols_target.index("sp_return")  # best-ckpt 기준 채널

    # tag 구성
    fold_str = f"_{fold_tag}" if fold_tag else ""
    norm_tag = ("_normbp" if apply_normbp else "") + ("_normsp" if apply_normsp else "")
    tag_full = f"matrix_v{spec['variant_id']}_{spec['name']}{norm_tag}{fold_str}_seed{seed}"
    ckpt_path    = os.path.join(save_dir, f"{tag_full}_best.pt")
    summary_path = os.path.join(save_dir, f"{tag_full}_summary.json")
    log_path     = os.path.join(save_dir, f"{tag_full}_trainlog.csv")

    if os.path.exists(ckpt_path) and os.path.exists(summary_path):
        print(f"[SKIP] {tag_full} — ckpt + summary 이미 존재")
        with open(summary_path) as f:
            return json.load(f)

    print(f"\n{'='*72}")
    print(f"[Variant {spec['variant_id']} = {spec['name']}{norm_tag}{fold_str}] seed={seed}")
    print(f"  cond   = {cols_cond}    (D_COND={len(cols_cond)})")
    print(f"  target = {cols_target}  (D_TARGET={len(cols_target)})")
    print(f"  mask_future_ch = {mask_future} → "
          f"{[cols_cond[i] for i in mask_future]} masked in future")
    print(f"  normalize target idxs = {norm_idxs}  (sp_return idx 0 raw)")
    print(f"  normbp={apply_normbp}, normsp={apply_normsp}")
    print(f"{'='*72}")

    # train/test 윈도우
    Xtr, Ctr, stats_c, stats_t_map = load_windows(
        train_csv, cols_cond, cols_target, L=L,
        normalize_target_idxs=norm_idxs)
    if Xtr is None:
        print(f"[FAIL] not enough train windows: {train_csv}")
        return None
    Xte, Cte, _, _ = load_windows(
        test_csv, cols_cond, cols_target, L=L,
        cond_stats=stats_c, normalize_target_idxs=norm_idxs,
        target_stats_map=stats_t_map)
    if Xte is None:
        print(f"[FAIL] not enough test windows: {test_csv}")
        return None

    print(f"  cond z-score (train): mean={[round(m,5) for m in stats_c['mean']]}")
    print(f"                          std={[round(s,5) for s in stats_c['std']]}")
    for idx, st in stats_t_map.items():
        print(f"  target normalize ({st['channel_name']}): "
              f"mean={st['mean']:+.5f}, std={st['std']:.5f}")

    Ctr = mask_future_channels(Ctr, PAST_LEN, mask_future)
    Cte = mask_future_channels(Cte, PAST_LEN, mask_future)

    if val_csv is not None and os.path.exists(val_csv):
        Xv, Cv, _, _ = load_windows(
            val_csv, cols_cond, cols_target, L=L,
            cond_stats=stats_c, normalize_target_idxs=norm_idxs,
            target_stats_map=stats_t_map)
        if Xv is None:
            print(f"[FAIL] not enough val windows: {val_csv}")
            return None
        Cv = mask_future_channels(Cv, PAST_LEN, mask_future)
        Xtr_, Ctr_ = Xtr, Ctr
        print(f"  train_w={Xtr_.shape[0]} (full), "
              f"val_w={Xv.shape[0]} (val_csv), test_w={Xte.shape[0]}")
    else:
        n_w_tr = Xtr.shape[0]
        n_val = max(int(n_w_tr * 0.15), 1)
        Xtr_, Ctr_ = Xtr[:-n_val], Ctr[:-n_val]
        Xv,  Cv    = Xtr[-n_val:], Ctr[-n_val:]
        print(f"  train_w={Xtr_.shape[0]}, val_w={Xv.shape[0]} (auto 15%), "
              f"test_w={Xte.shape[0]}  (no val_csv)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiStepFAVARFlow(
        K=K_STEPS, d_cond=len(cols_cond), d_target=len(cols_target),
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        time_reverse=False, use_wavelet=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_sp = float("inf")
    best_state = None
    best_epoch = -1
    best_test = float("nan")
    best_test_per_ch = None
    best_val_per_ch  = None
    pat = 0
    log = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(Xtr_.shape[0])
        losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
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
        per_ch_str = " ".join([f"{c}={v:+.3f}" for c, v in zip(cols_target, test_per_ch)])
        print(f"  ep{epoch:>3d}  train={train_nll:+.4f}  "
              f"val_mean={val_nll:+.4f}  test_mean={test_nll:+.4f}  | {per_ch_str}")
        row = dict(epoch=epoch, train=train_nll, val_mean=val_nll, test_mean=test_nll)
        for c, v in zip(cols_target, val_per_ch):
            row[f"val_{c}"]  = float(v)
        for c, v in zip(cols_target, test_per_ch):
            row[f"test_{c}"] = float(v)
        log.append(row)

        # best ckpt 기준: sp_return 단독 val NLL (raw 단위)
        val_sp = float(val_per_ch[sp_idx])
        if val_sp < best_val_sp - 1e-4:
            best_val_sp = val_sp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test = test_nll
            best_test_per_ch = test_per_ch.copy()
            best_val_per_ch  = val_per_ch.copy()
            best_epoch = epoch
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val_sp={best_val_sp:+.4f}, test_mean={best_test:+.4f}")
    if best_test_per_ch is not None:
        print(f"    test per channel: " +
              ", ".join([f"{c}={v:+.4f}" for c, v in zip(cols_target, best_test_per_ch)]))

    os.makedirs(save_dir, exist_ok=True)
    if best_state is not None:
        torch.save({
            "model_state":       best_state,
            "cond_cols":         cols_cond,
            "target_cols":       cols_target,
            "stats_cond":        stats_c,
            "stats_target_map":  {int(k): v for k, v in stats_t_map.items()},
            "mask_future_ch":    mask_future,
            "normalize_bondpp":  apply_normbp,
            "normalize_stockpp": apply_normsp,
            "config": dict(K=K_STEPS, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                           d_cond=len(cols_cond), d_target=len(cols_target),
                           past_len=PAST_LEN, total_len=L),
            "variant": dict(id=spec["variant_id"], name=spec["name"]),
        }, ckpt_path)
    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        variant_id=spec["variant_id"],
        variant_name=spec["name"],
        fold=fold_tag,
        seed=seed,
        cond_cols=cols_cond,
        target_cols=cols_target,
        mask_future_ch=mask_future,
        normalize_bondpp=apply_normbp,
        normalize_stockpp=apply_normsp,
        target_stats_map={int(k): v for k, v in stats_t_map.items()},
        best_epoch=best_epoch,
        val_sp_return=best_val_sp,
        test_mean=best_test,
        val_per_channel ={c: float(v) for c, v in zip(cols_target, best_val_per_ch)}  if best_val_per_ch  is not None else {},
        test_per_channel={c: float(v) for c, v in zip(cols_target, best_test_per_ch)} if best_test_per_ch is not None else {},
        n_params=n_params,
        device=str(device),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {ckpt_path}")
    print(f"         {log_path}")
    print(f"         {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Paper matrix 7-variant trainer (v33 data)")
    ap.add_argument("--variant", type=int, required=True, choices=list(VARIANTS.keys()),
                    help="1..7, paper_plan.txt 매트릭스 매핑")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--fold", default=None, choices=["F1", "F2", "F3"],
                    help="walk-forward fold. 주어지면 data/folds_v33/{fold}_{train,val,test}.csv 자동 매핑")
    ap.add_argument("--train-csv", default=None, help="override train csv")
    ap.add_argument("--val-csv",   default=None, help="override val csv")
    ap.add_argument("--test-csv",  default=None, help="override test csv")
    ap.add_argument("--out-dir",   default=os.path.join(HERE, "result"))
    ap.add_argument("--normalize-bondpp", action="store_true",
                    help="apply z-score to bondpp_13w_lag target channel (only if it's in target)")
    ap.add_argument("--normalize-stockpp", action="store_true",
                    help="apply z-score to stockpp_13w_lag target channel (only if it's in target)")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience",   type=int, default=PATIENCE)
    ap.add_argument("--batch",      type=int, default=BATCH)
    ap.add_argument("--lr",         type=float, default=LR)
    args = ap.parse_args()

    spec = build_spec(args.variant)

    # fold-based data path resolution
    if args.fold is not None:
        repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
        folds_dir = os.path.join(repo_root, "data", "folds_v33")
        train_csv = os.path.join(folds_dir, f"{args.fold}_train.csv")
        val_csv   = os.path.join(folds_dir, f"{args.fold}_val.csv")
        test_csv  = os.path.join(folds_dir, f"{args.fold}_test.csv")
        print(f"[--fold {args.fold}]")
        print(f"  train = {train_csv}")
        print(f"  val   = {val_csv}")
        print(f"  test  = {test_csv}")
    else:
        train_csv = os.path.join(HERE, "data", "weekly_v33_train.csv")
        val_csv   = None
        test_csv  = os.path.join(HERE, "data", "weekly_v33_test.csv")

    # explicit CLI override
    if args.train_csv: train_csv = args.train_csv
    if args.val_csv:   val_csv   = args.val_csv
    if args.test_csv:  test_csv  = args.test_csv

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for seed in args.seeds:
        r = run(spec, train_csv, val_csv, test_csv, args.out_dir, seed,
                max_epochs=args.max_epochs, patience=args.patience,
                batch=args.batch, lr=args.lr, fold_tag=args.fold,
                normalize_bondpp=args.normalize_bondpp,
                normalize_stockpp=args.normalize_stockpp)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        apply_normbp = bool(args.normalize_bondpp  and "bondpp_13w_lag"  in spec["cols_target"])
        apply_normsp = bool(args.normalize_stockpp and "stockpp_13w_lag" in spec["cols_target"])
        norm_tag = ("_normbp" if apply_normbp else "") + ("_normsp" if apply_normsp else "")
        fold_str = f"_{args.fold}" if args.fold else ""
        df_rows = []
        for r in results:
            row = dict(seed=r["seed"], best_epoch=r["best_epoch"],
                       val_sp_return=r["val_sp_return"], test_mean=r["test_mean"])
            for c, v in r["test_per_channel"].items():
                row[f"test_{c}"] = v
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        out_csv = os.path.join(
            args.out_dir,
            f"matrix_v{spec['variant_id']}_{spec['name']}{norm_tag}{fold_str}_multiseed.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[Variant {spec['variant_id']} {spec['name']}{norm_tag}{fold_str}] multi-seed (n={len(df)})")
        print(f"  val_sp_return: mean={df.val_sp_return.mean():+.4f} ± {df.val_sp_return.std():.4f}  "
              f"median={df.val_sp_return.median():+.4f}")
        for c in spec["cols_target"]:
            col = f"test_{c}"
            print(f"  test {c:<18s}: mean={df[col].mean():+.4f} ± {df[col].std():.4f}  "
                  f"median={df[col].median():+.4f}")


if __name__ == "__main__":
    main()
