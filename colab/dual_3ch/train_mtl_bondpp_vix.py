"""MTL 3채널 target — (sp_return, pp_bond_13w_lag, vix_wr) 동시 학습.

cond past   (default 2ch × 52w): tbill_wr, excess_liq_wr (관측)
cond future (1ch × 52w): tbill_wr 시나리오 활성, excess_liq_wr mask (=0)
target       (3ch × past+future): sp_return, pp_bond_13w_lag, vix_wr (joint AR generation)

--normalize-bondpp: pp_bond_13w_lag 채널만 train mean/std 로 standardize.
                    sp_return, vix_wr 채널은 raw 유지.
--with-26w        : cond 에 tbill_26w_lag 포함 (legacy 3ch reproduce).

비교 대상:
  MTL 2ch (sp + bondpp_3m):           sp_return median = -2.20 ★ (현 best, 외생 변수 X)
  MTL 3ch (sp + liq + vix):           sp_return median = -2.21    (외생 vix, OOS std 0.10)
  이 모델 (sp + bondpp + vix):         sp_return median = ?
  → 항상성 산식(bondpp) 위에 외생 변수(vix) 추가가 도움 되는지 확인
  → vix 채널 자체의 OOS NLL 폭발 (+628~921 nat) 패턴이 sp_return 학습을 흔드는지 확인
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
PATIENCE = 30

COLS_TARGET = ["sp_return", "pp_bond_13w_lag", "vix_wr"]
COLS_COND_NO_26W   = ["tbill_wr", "excess_liq_wr"]                          # default 2ch
COLS_COND_WITH_26W = ["tbill_wr", "tbill_26w_lag", "excess_liq_wr"]         # legacy 3ch reproduce
COLS_COND = COLS_COND_NO_26W
MASK_FUTURE_CH_NO_26W   = [1]      # mask excess_liq_wr in future (no 26w)
MASK_FUTURE_CH_WITH_26W = [1, 2]   # mask tbill_26w_lag + excess_liq_wr (legacy)
MASK_FUTURE_CH = MASK_FUTURE_CH_NO_26W
SP_TARGET_IDX     = 0      # sp_return 위치 in COLS_TARGET
BONDPP_TARGET_IDX = 1      # pp_bond_13w_lag 위치 in COLS_TARGET
VIX_TARGET_IDX    = 2      # vix_wr 위치 in COLS_TARGET

LOG2PI = math.log(2 * math.pi)


def load_windows(csv_path, cols_cond, cols_target, L=104,
                 cond_stats=None, target_stats=None, normalize_target_idx=None):
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
    else:
        cmu = np.asarray(cond_stats["mean"], dtype=np.float32)
        csd = np.asarray(cond_stats["std"],  dtype=np.float32)
    C = (C - cmu) / csd

    tstats_out = None
    if normalize_target_idx is not None:
        if target_stats is None:
            tmu = float(X[..., normalize_target_idx].mean())
            tsd = float(X[..., normalize_target_idx].std()) + 1e-8
        else:
            tmu = float(target_stats["mean"])
            tsd = float(target_stats["std"])
        X[..., normalize_target_idx] = (X[..., normalize_target_idx] - tmu) / tsd
        tstats_out = {"mean": tmu, "std": tsd, "channel": int(normalize_target_idx),
                      "channel_name": cols_target[normalize_target_idx]}

    return torch.from_numpy(X), torch.from_numpy(C), \
           {"mean": cmu.tolist(), "std": csd.tolist()}, tstats_out


def mask_future_channels(C, past_len, mask_channels):
    C = C.clone()
    for ch in mask_channels:
        C[:, past_len:, ch] = 0.0
    return C


def nll_per_step_per_channel(X, C, model, past_len, device):
    z, log_det_J, log_scale = model(X.to(device), C.to(device))
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale
    nll_future = nll_td[:, past_len:, :]
    nll_per_ch = nll_future.mean(dim=(0, 1)).detach().cpu().numpy()
    nll_mean = float(nll_per_ch.mean())
    return nll_mean, nll_per_ch


def loss_fn(X, C, model, past_len, device):
    z, log_det_J, log_scale = model(X.to(device), C.to(device))
    F_ = X.shape[1] - past_len
    D  = X.shape[2]
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale
    nll_future = nll_td[:, past_len:, :].sum(dim=(1, 2))
    return (nll_future / (F_ * D)).mean()


def run(train_csv, test_csv, save_dir, seed=42, normalize_bondpp=False, no_liq=False,
        normalize_vix=False, normalize_sp=False, val_csv=None, fold_tag=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    masked_names = [COLS_COND[i] for i in MASK_FUTURE_CH]
    tag_norm_b  = "_normbp"  if normalize_bondpp else ""
    tag_norm_v  = "_normvix" if normalize_vix    else ""
    tag_norm_sr = "_normsr"  if normalize_sp     else ""
    tag_norm    = tag_norm_b + tag_norm_v + tag_norm_sr
    tag_liq     = "_noliq" if no_liq else ""
    fold_str    = f"_{fold_tag}" if fold_tag else ""
    tag_full    = f"mtl_bp_vix{tag_liq}{tag_norm}{fold_str}_seed{seed}"
    ckpt_path   = os.path.join(save_dir, f"{tag_full}_best.pt")
    summary_path_pre = os.path.join(save_dir, f"{tag_full}_summary.json")
    if os.path.exists(ckpt_path) and os.path.exists(summary_path_pre):
        print(f"[SKIP] {tag_full} — ckpt + summary 이미 존재")
        with open(summary_path_pre) as f:
            return json.load(f)
    print(f"\n{'=' * 70}")
    print(f"[MTL_bp_vix{tag_liq}{tag_norm}{fold_str}] seed={seed}")
    if val_csv is not None:
        print(f"  val_csv = {val_csv}  (별도 val sliding window)")
    print(f"  cond   = {COLS_COND}    (D_COND={len(COLS_COND)})")
    print(f"  target = {COLS_TARGET}  (D_TARGET={len(COLS_TARGET)})  joint")
    print(f"  mask_future_ch = {MASK_FUTURE_CH} → {masked_names} masked in future")
    print(f"  normalize_bondpp = {normalize_bondpp} (target idx {BONDPP_TARGET_IDX} = {COLS_TARGET[BONDPP_TARGET_IDX]})")
    print(f"  normalize_vix    = {normalize_vix} (target idx {VIX_TARGET_IDX} = {COLS_TARGET[VIX_TARGET_IDX]})")
    print(f"  normalize_sp     = {normalize_sp} (target idx {SP_TARGET_IDX} = {COLS_TARGET[SP_TARGET_IDX]})")
    print(f"{'=' * 70}")

    # 1차: 첫 번째 정규화 채널은 load_windows 에서 처리 (bondpp 우선, 없으면 vix, 없으면 sp)
    if normalize_bondpp:
        norm_idx = BONDPP_TARGET_IDX
    elif normalize_vix:
        norm_idx = VIX_TARGET_IDX
    elif normalize_sp:
        norm_idx = SP_TARGET_IDX
    else:
        norm_idx = None
    Xtr, Ctr, stats_c, stats_t = load_windows(train_csv, COLS_COND, COLS_TARGET,
                                              L=L, normalize_target_idx=norm_idx)
    Xte, Cte, _, _             = load_windows(test_csv,  COLS_COND, COLS_TARGET,
                                              L=L, cond_stats=stats_c,
                                              normalize_target_idx=norm_idx, target_stats=stats_t)

    # 2차/3차: 추가 채널 manual normalize (load_windows 가 단일 채널만 처리)
    stats_targets_all = {}        # idx -> {mean, std, channel_name}
    if stats_t is not None:
        stats_targets_all[int(stats_t["channel"])] = stats_t
    # vix
    if normalize_vix and VIX_TARGET_IDX not in stats_targets_all:
        vmu = float(Xtr[..., VIX_TARGET_IDX].mean())
        vsd = float(Xtr[..., VIX_TARGET_IDX].std()) + 1e-8
        Xtr[..., VIX_TARGET_IDX] = (Xtr[..., VIX_TARGET_IDX] - vmu) / vsd
        Xte[..., VIX_TARGET_IDX] = (Xte[..., VIX_TARGET_IDX] - vmu) / vsd
        stats_targets_all[VIX_TARGET_IDX] = {"mean": vmu, "std": vsd,
                                             "channel": int(VIX_TARGET_IDX),
                                             "channel_name": COLS_TARGET[VIX_TARGET_IDX]}
        print(f"  ★ target normalize EXTRA vix_wr (train): mean={vmu:+.5f}, std={vsd:.5f}")
    # sp
    if normalize_sp and SP_TARGET_IDX not in stats_targets_all:
        smu = float(Xtr[..., SP_TARGET_IDX].mean())
        ssd = float(Xtr[..., SP_TARGET_IDX].std()) + 1e-8
        Xtr[..., SP_TARGET_IDX] = (Xtr[..., SP_TARGET_IDX] - smu) / ssd
        Xte[..., SP_TARGET_IDX] = (Xte[..., SP_TARGET_IDX] - smu) / ssd
        stats_targets_all[SP_TARGET_IDX] = {"mean": smu, "std": ssd,
                                            "channel": int(SP_TARGET_IDX),
                                            "channel_name": COLS_TARGET[SP_TARGET_IDX]}
        print(f"  ★ target normalize EXTRA sp_return (train): mean={smu:+.5f}, std={ssd:.5f}")
    # backward-compat: 기존 stats_t_extra 는 vix 단일 정규화 시점 (bondpp+vix only) 의미. 이번 dict 로 일반화.
    stats_t_extra = stats_targets_all.get(VIX_TARGET_IDX) if normalize_bondpp and normalize_vix else None

    print(f"  cond z-score stats (train): mean={[round(m,5) for m in stats_c['mean']]}")
    print(f"                                std={[round(s,5) for s in stats_c['std']]}")
    if stats_t is not None:
        print(f"  ★ target normalize (train): mean={stats_t['mean']:+.5f}, std={stats_t['std']:.5f}")

    Ctr = mask_future_channels(Ctr, PAST_LEN, MASK_FUTURE_CH)
    Cte = mask_future_channels(Cte, PAST_LEN, MASK_FUTURE_CH)

    if val_csv is not None:
        Xv, Cv, _, _ = load_windows(val_csv, COLS_COND, COLS_TARGET, L=L,
                                     cond_stats=stats_c,
                                     normalize_target_idx=norm_idx, target_stats=stats_t)
        # 추가 채널 정규화 적용
        if normalize_vix and VIX_TARGET_IDX in stats_targets_all and (norm_idx != VIX_TARGET_IDX):
            v = stats_targets_all[VIX_TARGET_IDX]
            Xv[..., VIX_TARGET_IDX] = (Xv[..., VIX_TARGET_IDX] - v["mean"]) / v["std"]
        if normalize_sp and SP_TARGET_IDX in stats_targets_all and (norm_idx != SP_TARGET_IDX):
            s = stats_targets_all[SP_TARGET_IDX]
            Xv[..., SP_TARGET_IDX] = (Xv[..., SP_TARGET_IDX] - s["mean"]) / s["std"]
        Cv = mask_future_channels(Cv, PAST_LEN, MASK_FUTURE_CH)
        Xtr_, Ctr_ = Xtr, Ctr
        print(f"  train_windows={Xtr_.shape[0]} (full train), "
              f"val_windows={Xv.shape[0]} (from val_csv), test_windows={Xte.shape[0]}")
    else:
        n_w_tr = Xtr.shape[0]
        n_val = max(int(n_w_tr * 0.15), 1)
        Xtr_, Ctr_ = Xtr[:-n_val], Ctr[:-n_val]
        Xv,  Cv  = Xtr[-n_val:], Ctr[-n_val:]
        print(f"  train_windows={Xtr_.shape[0]}, val_windows={Xv.shape[0]} (auto split 15%), "
              f"test_windows={Xte.shape[0]}")

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
        per_ch_str = " ".join([f"{c}={v:+.3f}" for c, v in zip(COLS_TARGET, test_per_ch)])
        print(f"  ep{epoch:>3d}  train={train_nll:+.4f}  val={val_nll:+.4f}  test={test_nll:+.4f}  | {per_ch_str}")
        row = dict(epoch=epoch, train=train_nll, val=val_nll, test=test_nll)
        for c, v in zip(COLS_TARGET, test_per_ch):
            row[f"test_{c}"] = float(v)
        log.append(row)

        # best ckpt 기준: sp_return 단독 val NLL (paper 주제와 일관)
        sp_idx_val = COLS_TARGET.index("sp_return")
        val_metric = float(val_per_ch[sp_idx_val])
        if val_metric < best_val - 1e-4:
            best_val = val_metric
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
    print(f"    test per channel: " + ", ".join([f"{c}={v:+.4f}" for c, v in zip(COLS_TARGET, best_test_per_ch)]))

    os.makedirs(save_dir, exist_ok=True)
    tag = tag_full
    log_path     = os.path.join(save_dir, f"{tag}_trainlog.csv")
    summary_path = os.path.join(save_dir, f"{tag}_summary.json")

    if best_state is not None:
        torch.save({
            "model_state":   best_state,
            "cond_cols":     COLS_COND,
            "target_cols":   COLS_TARGET,
            "stats_cond":    stats_c,
            "stats_target":  stats_t,
            "stats_target_extra": stats_t_extra,   # backward-compat (vix 단일 case)
            "stats_targets_all":  stats_targets_all,  # 신: 정규화된 모든 target 채널의 stats dict
            "mask_future_ch": MASK_FUTURE_CH,
            "normalize_bondpp": normalize_bondpp,
            "normalize_vix":   normalize_vix,
            "normalize_sp":    normalize_sp,
            "config": dict(K=K_STEPS, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                           d_cond=len(COLS_COND), d_target=len(COLS_TARGET),
                           past_len=PAST_LEN, total_len=L),
        }, ckpt_path)
    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        stage=f"mtl_bp_vix{tag_liq}{tag_norm}{fold_str}",
        fold=fold_tag,
        no_liq=no_liq,
        normalize_vix=normalize_vix,
        normalize_sp=normalize_sp,
        seed=seed,
        cond_cols=COLS_COND,
        target_cols=COLS_TARGET,
        mask_future_ch=MASK_FUTURE_CH,
        normalize_bondpp=normalize_bondpp,
        target_stats_train=stats_t,
        stats_targets_all={int(k): v for k, v in stats_targets_all.items()},
        best_epoch=best_epoch,
        val=best_val,
        test=best_test,
        test_per_channel={c: float(v) for c, v in zip(COLS_TARGET, best_test_per_ch)},
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
    ap.add_argument("--test-csv",  default=os.path.join(HERE, "data", "weekly_ppbond_test.csv"))
    ap.add_argument("--out-dir",   default=os.path.join(HERE, "result"))
    ap.add_argument("--fold", default=None, choices=["F1", "F2", "F3"],
                    help="walk-forward fold. data/folds/{fold}_{train,val,test}.csv 자동 매핑")
    ap.add_argument("--normalize-bondpp", action="store_true",
                    help="train mean/std 로 pp_bond_13w_lag target 정규화")
    ap.add_argument("--normalize-vix", action="store_true",
                    help="train mean/std 로 vix_wr target 정규화 (vix_wr 채널 NLL 폭발 방지)")
    ap.add_argument("--normalize-sp", action="store_true",
                    help="train mean/std 로 sp_return target 정규화 (multi-channel scale balance)")
    ap.add_argument("--with-26w", action="store_true",
                    help="cond 에 tbill_26w_lag 포함 (legacy 3ch reproduce)")
    ap.add_argument("--no-liq", action="store_true",
                    help="cond 에서 excess_liq_wr 제거 (cond=[tbill_wr] 1ch only)")
    args = ap.parse_args()
    if args.with_26w and args.no_liq:
        raise ValueError("--with-26w 와 --no-liq 동시 사용 불가")
    global COLS_COND, MASK_FUTURE_CH
    if args.with_26w:
        COLS_COND = COLS_COND_WITH_26W
        MASK_FUTURE_CH = MASK_FUTURE_CH_WITH_26W
        print(f"[--with-26w] cond = {COLS_COND}, mask = {MASK_FUTURE_CH}")
    elif args.no_liq:
        COLS_COND = ["tbill_wr"]
        MASK_FUTURE_CH = []
        print(f"[--no-liq] cond = {COLS_COND}, mask = {MASK_FUTURE_CH}")

    if args.fold is not None:
        repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
        folds_dir = os.path.join(repo_root, "data", "folds")
        args.train_csv = os.path.join(folds_dir, f"{args.fold}_train.csv")
        args.val_csv   = os.path.join(folds_dir, f"{args.fold}_val.csv")
        args.test_csv  = os.path.join(folds_dir, f"{args.fold}_test.csv")
        print(f"[--fold {args.fold}] train={args.train_csv}")
        print(f"               val  ={args.val_csv}")
        print(f"               test ={args.test_csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = []
    for seed in args.seeds:
        r = run(args.train_csv, args.test_csv, args.out_dir, seed=seed,
                normalize_bondpp=args.normalize_bondpp, no_liq=args.no_liq,
                normalize_vix=args.normalize_vix, normalize_sp=args.normalize_sp,
                val_csv=args.val_csv, fold_tag=args.fold)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        tag_norm_b  = "_normbp"  if args.normalize_bondpp else ""
        tag_norm_v  = "_normvix" if args.normalize_vix    else ""
        tag_norm_sr = "_normsr"  if args.normalize_sp     else ""
        tag_norm    = tag_norm_b + tag_norm_v + tag_norm_sr
        tag_liq     = "_noliq" if args.no_liq else ""
        fold_str    = f"_{args.fold}" if args.fold else ""
        df_rows = []
        for r in results:
            row = {"seed": r["seed"], "best_epoch": r["best_epoch"], "val": r["val"], "test_mean": r["test"]}
            for c, v in r["test_per_channel"].items():
                row[f"test_{c}"] = v
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        out_csv = os.path.join(args.out_dir, f"mtl_bp_vix{tag_liq}{tag_norm}{fold_str}_multiseed_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[MTL_bp_vix{tag_liq}{tag_norm}{fold_str}] multi-seed (n={len(df)})")
        print(f"  val:                  mean={df.val.mean():+.4f} ± {df.val.std():.4f}  median={df.val.median():+.4f}")
        print(f"  test mean (3ch avg):  mean={df.test_mean.mean():+.4f} ± {df.test_mean.std():.4f}  median={df.test_mean.median():+.4f}")
        for c in COLS_TARGET:
            col = f"test_{c}"
            print(f"  test {c:<18s}: mean={df[col].mean():+.4f} ± {df[col].std():.4f}  median={df[col].median():+.4f}")
        print(f"\n  ★ vs MTL 2ch (sp+bp) sp_return median=-2.20: this sp_return median={df['test_sp_return'].median():+.4f}")


if __name__ == "__main__":
    main()
