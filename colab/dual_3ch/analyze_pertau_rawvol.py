"""§4.2 보강 — 시점별(per-τ) 분석: step↑ vs step↓의 *시간 프로파일*로 순서 의존 검증.

배경: analyze_pathshape_rawvol.py 는 13주를 pool(ravel)한 marginal 로 skew/cvar 을 봤다.
  pooling 은 시간축을 뭉개므로, step↑(거시 후반 극단)·step↓(전반 극단) 처럼 *시간 역순*
  경로가 위험을 후반/전반으로 *재배치*해도 pool 하면 같은 marginal → "순서 불변"으로 오인.

따라서 σ=0.5(step, pooled 에서 효과 최대) 만 골라 **τ=1..13 시점별** skew/cvar/std 를 본다.
  step↑ 위험이 후반 τ, step↓ 위험이 전반 τ 로 *거울상* 이면 → 순서/경로 의존 입증.
  시점별로도 동일하면 → 진짜 순서 불변.

LOCKED 본모형(rvP2mainMlp_pd64_fl4_fh128, extra=std+skew), raw-vol, inference.
시나리오 = {flat, step_up, step_down} × {tbill-shape, metab-shape}.  4 fold × 3 seed, paired.
재진입: (fold,seed) json 캐시.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/analyze_pertau_rawvol.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from rawvol_helpers import patch_rawvol                              # noqa: E402
patch_rawvol()

import train_garch_flow as T                                        # noqa: E402
from train_garch_flow import (                                      # noqa: E402
    MambaFlowAR, cached_load_windows_seq, load_extra_context,
    compute_valid_mask, forward_garch_rescale, garch_preprocess_fold,
    compute_cvar, PAST_LEN, FUTURE_LEN,
)

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
CACHE_DIR = os.path.join(RESULT_DIR, "pertau_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
DC_COLS_LIST = ["sp_std_13w", "sp_skew_13w"]

LK_D_MODEL = 128; LK_MLP_LAYERS = 4
LK_FLOW_LAYERS = 4; LK_FLOW_HIDDEN = 128
LK_PD = 64; LK_DROPOUT = 0.2
TAG_PREFIX = f"rvP2mainMlp_pd{LK_PD}_fl{LK_FLOW_LAYERS}_fh{LK_FLOW_HIDDEN}"

FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEEDS = [2026, 2027, 2028]
PCTLS = [10, 50, 90]
N_ORIGIN_MAX = 200
N_SIM = 1000
CHUNK = 8


def set_cond_cols(cols):
    cols = list(cols)
    T.COND_COLS = cols; T.N_CHANNELS = len(cols)
    T.SP_CH = cols.index("sp_return"); T.TBILL_CH = cols.index("tbill_wr")
    T.MACRO_CH = [c for c in range(len(cols)) if c not in (T.SP_CH, T.TBILL_CH)]
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass


set_cond_cols(ENC_COLS)
T.MASK_FUTURE_TBILL = False
T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
T.ENCODER_MASK_SP = False


# step↑/↓ 편차 (zero-mean, range 1) — σ=0.5 그룹만
def _norm(d):
    d = np.asarray(d, float); d = d - d.mean(); r = d.max() - d.min()
    return d / r if r > 1e-12 else d


_t = np.arange(FUTURE_LEN, dtype=float); _mid = (FUTURE_LEN - 1) / 2.0
_step_up = _norm(np.where(_t < _mid, 0.0, 1.0))    # 전반 low, 후반 high
SHAPES = {"flat": np.zeros(FUTURE_LEN), "step_up": _step_up, "step_down": -_step_up}


def _skew(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def rebuild_model(ckpt, device):
    model = MambaFlowAR(
        d_input=len(ENC_COLS), d_model=LK_D_MODEL,
        n_flow_layers=LK_FLOW_LAYERS, n_flow_hidden=LK_FLOW_HIDDEN,
        dropout=LK_DROPOUT, extra_context_dim=len(DC_COLS_LIST),
        encoder_type="mlp", mlp_num_layers=LK_MLP_LAYERS,
        direct_prev_return=True, use_past_summary=True,
        past_encoder_type="mlp", past_summary_dim=LK_PD,
    ).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return model


@torch.no_grad()
def rollout_path(model, Xte, extra, last_sp, tbill_path, metab_path, device):
    n = Xte.shape[0]
    tb = torch.tensor(np.asarray(tbill_path, float), device=device, dtype=Xte.dtype)
    mb = torch.tensor(np.asarray(metab_path, float), device=device, dtype=Xte.dtype)
    out = []
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK); k = e - s
        fut_tbill = tb.unsqueeze(0).expand(k, FUTURE_LEN).contiguous()
        fmac = mb.reshape(1, FUTURE_LEN, 1).expand(k, FUTURE_LEN, 1).contiguous()
        sim = model.ar_sample(Xte[s:e, :PAST_LEN, :], fut_tbill, last_sp[s:e], N_SIM,
                              extra_context=(extra[s:e] if extra is not None else None),
                              future_macro_z=fmac)
        out.append(sim.cpu())
    return torch.cat(out, dim=0).numpy()         # (n_orig, n_sim, FUTURE_LEN)


def load_fold_seed(fold, seed, device):
    bp = os.path.join(RESULT_DIR, f"garch_flow_ar_{TAG_PREFIX}_s{seed}_{fold}_best.pt")
    if not os.path.exists(bp):
        print(f"[missing] {os.path.basename(bp)}"); return None
    ckpt = torch.load(bp, map_location=device)
    meta = ckpt["meta"]; cond_stats = meta["cond_stats"]; target_stats = meta["target_stats"]
    extra_stats = meta.get("extra_stats")
    model = rebuild_model(ckpt, device)
    gp = garch_preprocess_fold(FOLDS_DIR, fold, RESULT_DIR)
    origin_csv = gp["train"]
    Xte, Yte, _, _, _ = cached_load_windows_seq(origin_csv, cond_stats=cond_stats, target_stats=target_stats)
    Xte_dev = Xte.to(device)
    Xte_extra, _, n_ex, _ = load_extra_context(origin_csv, DC_COLS_LIST, extra_stats=extra_stats)
    if n_ex != Xte.shape[0]:
        sys.exit(f"[FATAL] extra {n_ex} != main {Xte.shape[0]}")
    Xte_extra_dev = Xte_extra.to(device)
    valid_mask, z_te, df_te = compute_valid_mask(origin_csv, cond_stats)
    n_w = z_te.shape[0] - PAST_LEN - FUTURE_LEN + 1
    last_full = z_te[PAST_LEN - 1: PAST_LEN - 1 + n_w, T.SP_CH]
    last_sp_dev = torch.from_numpy(last_full[valid_mask].astype(np.float32)).to(device)
    df_tr = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_train.csv"))
    cmu = np.asarray(cond_stats["mean"], float); csd = np.asarray(cond_stats["std"], float)
    ti = ENC_COLS.index("tbill_wr"); mi = ENC_COLS.index("metab_13w")
    tb_z = {p: (np.percentile(df_tr["tbill_wr"].dropna(), p) - cmu[ti]) / csd[ti] for p in PCTLS}
    mb_z = {p: (np.percentile(df_tr["metab_13w"].dropna(), p) - cmu[mi]) / csd[mi] for p in PCTLS}
    tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
    _gsig = df_te["garch_sigma"].to_numpy(float); _gz = df_te["sp_return"].to_numpy(float)
    _oidx = np.where(valid_mask)[0]; n_orig = len(_oidx)
    _om = float(df_te["garch_omega"].iloc[0]); _al = float(df_te["garch_alpha"].iloc[0])
    _be = float(df_te["garch_beta"].iloc[0]); _mu = float(df_te["garch_mu"].iloc[0])
    _orow = _oidx + (PAST_LEN - 1)
    _s2 = _gsig[_orow] ** 2; _e2 = (_gz[_orow] * _gsig[_orow]) ** 2
    if n_orig > N_ORIGIN_MAX:
        idx = np.linspace(0, n_orig - 1, N_ORIGIN_MAX).astype(int)
        Xte_dev = Xte_dev[idx]; Xte_extra_dev = Xte_extra_dev[idx]
        last_sp_dev = last_sp_dev[idx]; _s2 = _s2[idx]; _e2 = _e2[idx]
    rescale = dict(tmu=tmu, tsd=tsd, s2=_s2, e2=_e2, om=_om, al=_al, be=_be, mu=_mu)
    return dict(model=model, Xte=Xte_dev, extra=Xte_extra_dev, last_sp=last_sp_dev,
                rescale=rescale, tb_z=tb_z, mb_z=mb_z)


def pertau_metrics(sim_z, rescale):
    """(n_orig, n_sim, FUTURE_LEN) → 시점별 skew/cvar1/std 리스트."""
    r = rescale
    sim_raw = forward_garch_rescale(sim_z * r["tsd"] + r["tmu"], r["s2"], r["e2"],
                                    r["om"], r["al"], r["be"], r["mu"])  # (n_orig, n_sim, T)
    out = []
    for tau in range(FUTURE_LEN):
        f = sim_raw[:, :, tau].ravel()
        out.append(dict(skew=_skew(f), cvar1=compute_cvar(f, 0.01),
                        std=float(f.std(ddof=1))))
    return out


def run_fold_seed(fold, seed, device):
    ctx = load_fold_seed(fold, seed, device)
    if ctx is None:
        return None
    res = {"tbill": {}, "metab": {}}
    tb_c = ctx["tb_z"][50]; tb_rng = ctx["tb_z"][90] - ctx["tb_z"][10]
    mb_c = ctx["mb_z"][50]; mb_rng = ctx["mb_z"][90] - ctx["mb_z"][10]
    tb_flat = np.full(FUTURE_LEN, tb_c); mb_flat = np.full(FUTURE_LEN, mb_c)

    def _run(tbill_path, metab_path):
        torch.manual_seed(seed)
        return pertau_metrics(rollout_path(ctx["model"], ctx["Xte"], ctx["extra"],
                                           ctx["last_sp"], tbill_path, metab_path, device),
                              ctx["rescale"])

    for name, dev in SHAPES.items():
        res["tbill"][name] = _run(tb_c + dev * tb_rng, mb_flat)
        res["metab"][name] = _run(tb_flat, mb_c + dev * mb_rng)
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print(f"# §4.2 per-τ — step↑/↓ 시간 프로파일로 순서 의존 검증, LOCKED {TAG_PREFIX}, device={device}")
    print("#" * 100)
    for fold in FOLDS:
        for seed in SEEDS:
            c = os.path.join(CACHE_DIR, f"{fold}_s{seed}.json")
            if os.path.exists(c):
                print(f"[skip] {fold} s{seed}"); continue
            r = run_fold_seed(fold, seed, device)
            if r is None:
                continue
            json.dump(r, open(c, "w"), indent=2); print(f"[done] {fold} s{seed}")
    print("\n" + "=" * 100)
    _summarize()


def _avg_over_seeds(loaded, var, shape, metric):
    """seed 평균 per-τ 배열 (FUTURE_LEN,)."""
    arrs = [[d[var][shape][tau][metric] for tau in range(FUTURE_LEN)] for d in loaded]
    return np.asarray(arrs, float).mean(axis=0)


def _summarize():
    for fold in FOLDS:
        caches = [os.path.join(CACHE_DIR, f"{fold}_s{s}.json") for s in SEEDS]
        loaded = [json.load(open(c)) for c in caches if os.path.exists(c)]
        if not loaded:
            print(f"\n=== {fold}: 없음"); continue
        print(f"\n=== {fold}  ({len(loaded)} seed) " + "=" * 60)
        for var in ["tbill", "metab"]:
            print(f"  [{var}-shape] 시점별 skew  (step↑=전반low·후반high / step↓=전반high·후반low)")
            su = _avg_over_seeds(loaded, var, "step_up", "skew")
            sd = _avg_over_seeds(loaded, var, "step_down", "skew")
            fl = _avg_over_seeds(loaded, var, "flat", "skew")
            print(f"    {'τ':>3} {'flat':>8} {'step_up':>9} {'step_dn':>9} {'(up−dn)':>9}")
            for tau in range(FUTURE_LEN):
                print(f"    {tau+1:>3} {fl[tau]:>+8.3f} {su[tau]:>+9.3f} {sd[tau]:>+9.3f} {su[tau]-sd[tau]:>+9.3f}")
            cu = _avg_over_seeds(loaded, var, "step_up", "cvar1")
            cd = _avg_over_seeds(loaded, var, "step_down", "cvar1")
            print(f"    [cvar1] step↑ 후반↓(깊어짐)·step↓ 전반↓ 이면 → 위험 시점 역전 = 순서 의존")
            print(f"    {'τ':>3} {'cvar↑':>9} {'cvar↓':>9} {'(↑−↓)':>9}")
            for tau in range(FUTURE_LEN):
                print(f"    {tau+1:>3} {cu[tau]:>+9.5f} {cd[tau]:>+9.5f} {cu[tau]-cd[tau]:>+9.5f}")
    print("\n[판정] step↑·step↓ 의 skew/cvar 시점 프로파일이 *거울상*(전반↔후반)이면 →")
    print("       모델이 거시 극단 시점에 위험을 배치 = 순서/경로 의존 (pooling 이 가렸던 것).")


if __name__ == "__main__":
    main()
