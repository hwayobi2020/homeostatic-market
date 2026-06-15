"""§4.2 핵심 — 경로별 *최대낙폭(max drawdown)* 분포로 주가 경로의존성 검증.

직관(사용자): 100→110→100 (낙폭 0) vs 100→90→100 (낙폭 −10%) — *끝점·누적 같아도*
  중간에 겪는 위험(낙폭)은 다르다.  pooled 주간수익률 skew 는 순서에 무감각하지만,
  주가(누적)는 경로의존적이며 그 위험은 *max drawdown* 에 드러난다.

따라서 시뮬 경로마다 13주 누적 로그수익 경로의 최대낙폭을 계산해 분포를 본다.
  shape별 비교 — 특히 ramp↑ vs ramp↓, step↑ vs step↓ (평균·σ 동일, 순서만 반대):
    낙폭 분포가 다르면 → 주가 경로/순서 의존 입증 (수익률 pooled 가 가렸던 것).
  terminal 누적도 함께 출력 → "끝점 비슷, 낙폭 다름" 확인.

LOCKED 본모형(rvP2mainMlp_pd64_fl4_fh128, extra=std+skew), raw-vol, inference.
shape = {flat, ramp_up, ramp_down, step_up, step_down} × {tbill, metab}.  4 fold × 3 seed.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/analyze_drawdown_rawvol.py
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
CACHE_DIR = os.path.join(RESULT_DIR, "drawdown_cache")
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


def _norm(d):
    d = np.asarray(d, float); d = d - d.mean(); r = d.max() - d.min()
    return d / r if r > 1e-12 else d


_t = np.arange(FUTURE_LEN, dtype=float); _mid = (FUTURE_LEN - 1) / 2.0
_ramp = _norm(np.linspace(0.0, 1.0, FUTURE_LEN))
_step = _norm(np.where(_t < _mid, 0.0, 1.0))
SHAPES = {"flat": np.zeros(FUTURE_LEN), "ramp_up": _ramp, "ramp_down": -_ramp,
          "step_up": _step, "step_down": -_step}


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
    return torch.cat(out, dim=0).numpy()        # (n_orig, n_sim, FUTURE_LEN)


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


def path_metrics(sim_z, rescale):
    """경로별 최대낙폭(max drawdown) + terminal 누적 분포.

    sim_z: (n_orig, n_sim, FUTURE_LEN) z-score 주간수익.
    누적 로그수익 cum, running peak 대비 낙폭 dd(≤0), 경로별 최저 dd = max drawdown.
    """
    r = rescale
    sim_raw = forward_garch_rescale(sim_z * r["tsd"] + r["tmu"], r["s2"], r["e2"],
                                    r["om"], r["al"], r["be"], r["mu"])  # (no, ns, T) 주간수익
    cum = np.cumsum(sim_raw, axis=2)                       # 누적 로그수익 경로
    run_max = np.maximum.accumulate(cum, axis=2)           # running peak
    dd = cum - run_max                                     # 낙폭 경로 (≤0)
    mdd = dd.min(axis=2).ravel()                           # 경로별 max drawdown (가장 음수)
    term = cum[:, :, -1].ravel()                           # terminal 누적수익
    return dict(
        mdd_mean=float(mdd.mean()),
        mdd_cvar5=compute_cvar(mdd, 0.05),                 # 최악 5% 낙폭 평균
        mdd_cvar1=compute_cvar(mdd, 0.01),
        term_mean=float(term.mean()),
        term_std=float(term.std(ddof=1)),
        term_cvar1=compute_cvar(term, 0.01),               # 13주 보유 누적 1% 꼬리손실
    )


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
        return path_metrics(rollout_path(ctx["model"], ctx["Xte"], ctx["extra"],
                                         ctx["last_sp"], tbill_path, metab_path, device),
                            ctx["rescale"])

    for name, dev in SHAPES.items():
        res["tbill"][name] = _run(tb_c + dev * tb_rng, mb_flat)
        res["metab"][name] = _run(tb_flat, mb_c + dev * mb_rng)
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print(f"# §4.2 max drawdown — 주가 경로의존성, LOCKED {TAG_PREFIX}, device={device}")
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


def _agg(loaded, var, shape, metric):
    a = np.asarray([d[var][shape][metric] for d in loaded], float)
    return float(a.mean()), float(a.std(ddof=0))


def _summarize():
    for fold in FOLDS:
        caches = [os.path.join(CACHE_DIR, f"{fold}_s{s}.json") for s in SEEDS]
        loaded = [json.load(open(c)) for c in caches if os.path.exists(c)]
        if not loaded:
            print(f"\n=== {fold}: 없음"); continue
        print(f"\n=== {fold}  ({len(loaded)} seed) " + "=" * 55)
        for var in ["tbill", "metab"]:
            print(f"  [{var}-shape]  낙폭 깊을수록(−) 위험 / terminal 은 끝점 누적")
            print(f"    {'shape':>10} {'MDD_mean':>16} {'MDD_cvar1':>16} {'term_mean':>14} {'term_cvar1':>14}")
            for name in SHAPES:
                mdm = _agg(loaded, var, name, "mdd_mean")
                mdc = _agg(loaded, var, name, "mdd_cvar1")
                tm = _agg(loaded, var, name, "term_mean")
                tc = _agg(loaded, var, name, "term_cvar1")
                print(f"    {name:>10} {mdm[0]:+.5f}±{mdm[1]:.5f} {mdc[0]:+.5f}±{mdc[1]:.5f} "
                      f"{tm[0]:+.5f}±{tm[1]:.5f} {tc[0]:+.5f}±{tc[1]:.5f}")
    print("\n[판정] ramp_up vs ramp_down (또는 step_up/down) 의 *MDD* 가 다르고 fold 일관 →")
    print("       주가 경로/순서 의존 입증.  terminal 은 비슷한데 MDD 만 다르면 → 순수 경로위험.")
    print("       MDD 도 같으면 → 낙폭 차원에서도 순서 불변.")


if __name__ == "__main__":
    main()
