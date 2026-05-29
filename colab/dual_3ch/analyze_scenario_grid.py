"""거시 시나리오 9-grid counterfactual — 금리 × 초과유동성(metab) → equity tail 분포.

thesis 핵심: "monetary debasement(+단기금리) 경로가 equity tail 을 조건짓는다"를
MacroVAE 식 시나리오 grid 로 직접 입증한다.  σ 흔들기(sensitivity) 대신 의미있는
거시 수준을 준다:

  금리(tbill_wr)        : 하락(10pctl) / 보합(50pctl) / 상승(90pctl)
  초과유동성(metab_13w) : 감소(10pctl) / 보합(50pctl) / 증가(90pctl)
  → 3 × 3 = 9 케이스.  각 케이스에서 미래 13주 경로를 그 수준으로 flat 고정하고
     학습된 MLP best.pt(재학습 X)로 생성 분포(skew/CVaR/std)를 측정한다.

수준 = fold별 train 분포의 raw percentile → z 변환 (z=0=평균 모호함 회피, 부호·분포
의미 보존, train 관측 범위 안이라 OOD 아님).  과거 경로·extra(volDC)는 actual 유지.

paired: 9 케이스 모두 같은 난수(torch.manual_seed)에서 생성 → 칸 간 차이 = 순수
시나리오 효과 (샘플링 노이즈 제거).

Usage (Colab): !python colab/dual_3ch/analyze_scenario_grid.py
  (메인 인코더=MLP 라 mamba-ssm 불필요 — lazy import.)
"""
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

import train_garch_flow as T                                      # noqa: E402
from train_garch_flow import (                                    # noqa: E402
    MambaFlowAR, cached_load_windows_seq, load_extra_context,
    compute_valid_mask, forward_garch_rescale, garch_preprocess_fold,
    compute_var, compute_cvar, PAST_LEN, FUTURE_LEN,
)
from best_specs import BEST_SPECS                                 # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")

ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
DC_COLS = "sp_std_13w"


def set_cond_cols(cols):
    cols = list(cols)
    T.COND_COLS = cols
    T.N_CHANNELS = len(cols)
    T.SP_CH = cols.index("sp_return")
    T.TBILL_CH = cols.index("tbill_wr")
    T.MACRO_CH = [c for c in range(len(cols)) if c not in (T.SP_CH, T.TBILL_CH)]
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass


set_cond_cols(ENC_COLS)
T.MASK_FUTURE_TBILL = False
T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
T.ENCODER_MASK_SP = False

FOLDS = ["full"]                     # 전 기간 통합 단일 모델 (시나리오 분석 전용)
SEED = 2026
PAST_SUMMARY_DIM = 64                # best.pt tag 의 d{dim}
N_ORIGIN_MAX = 250                   # 전 기간 origin 균등 subsample (속도; paired 라 set 고정)
PCTLS = [10, 50, 90]                 # 하락/보합/상승 (감소/보합/증가)
PCTL_LABEL = {10: "lo", 50: "mid", 90: "hi"}
N_SIM = 1000
CHUNK = 8


def _skew(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def _exkurt(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 4) - 3.0)


def rebuild_model(ckpt, device):
    spec = BEST_SPECS["mlp"]
    model = MambaFlowAR(
        d_input=len(ENC_COLS), d_model=spec["d_model"],
        n_flow_layers=spec["n_flow_layers"], n_flow_hidden=spec["n_flow_hidden"],
        dropout=spec.get("dropout", 0.0), extra_context_dim=1,
        encoder_type="mlp", mlp_num_layers=spec["mlp_num_layers"],
        direct_prev_return=True, use_past_summary=True,
        past_encoder_type="mlp", past_summary_dim=64,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def rollout_scenario(model, Xte_dev, Xte_extra_dev, last_sp_dev,
                     tbill_z, metab_z, metab_idx, device):
    """미래 tbill·metab 경로를 각각 flat z 값으로 고정해 ar_sample.

    tbill_z / metab_z : 미래 13주 전 구간에 채울 z-score scalar (시나리오 수준).
    seed 는 호출 측에서 고정 (paired 비교).
    """
    n = Xte_dev.shape[0]
    out = []
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        k = e - s
        x_past = Xte_dev[s:e, :PAST_LEN, :]
        last_sp = last_sp_dev[s:e]
        extra = Xte_extra_dev[s:e] if Xte_extra_dev is not None else None
        fut_tbill = torch.full((k, FUTURE_LEN), float(tbill_z),
                               device=device, dtype=Xte_dev.dtype)
        fmac = torch.full((k, FUTURE_LEN, 1), float(metab_z),
                          device=device, dtype=Xte_dev.dtype)
        sim = model.ar_sample(x_past, fut_tbill, last_sp, N_SIM,
                              extra_context=extra, future_macro_z=fmac)
        out.append(sim.cpu())
    return torch.cat(out, dim=0).numpy()


def load_fold(fold, device):
    bp = os.path.join(RESULT_DIR,
                      f"garch_flow_ar_macroenc_pastMlp_d{PAST_SUMMARY_DIM}_s{SEED}_{fold}_best.pt")
    if not os.path.exists(bp):
        print(f"[missing best.pt] {os.path.basename(bp)}")
        return None
    ckpt = torch.load(bp, map_location=device)
    meta = ckpt["meta"]
    cond_stats = meta["cond_stats"]; target_stats = meta["target_stats"]
    extra_stats = meta.get("extra_stats")
    model = rebuild_model(ckpt, device)

    # origin = 전 기간 train (시나리오는 OOS 아니므로 test 국한 불필요; full_train 이 GFC·
    # COVID·평시 다 포함).  9-grid 는 paired 라 origin set 고정.
    gp = garch_preprocess_fold(FOLDS_DIR, fold, RESULT_DIR)
    origin_csv = gp["train"]
    Xte, Yte, _, _, _ = cached_load_windows_seq(
        origin_csv, cond_stats=cond_stats, target_stats=target_stats)
    Xte_dev = Xte.to(device)
    Xte_extra, _, n_ex, _ = load_extra_context(
        origin_csv, [DC_COLS], extra_stats=extra_stats)
    if n_ex != Xte.shape[0]:
        sys.exit(f"[FATAL] extra valid {n_ex} != main {Xte.shape[0]} ({fold})")
    Xte_extra_dev = Xte_extra.to(device)

    valid_mask, z_te, df_te = compute_valid_mask(origin_csv, cond_stats)
    n_w = z_te.shape[0] - PAST_LEN - FUTURE_LEN + 1
    last_full = z_te[PAST_LEN - 1: PAST_LEN - 1 + n_w, T.SP_CH]
    last_valid = last_full[valid_mask].astype(np.float32)
    last_sp_dev = torch.from_numpy(last_valid).to(device)

    # 시나리오 수준 = fold별 train raw percentile → z (cond_stats 로 표준화)
    tr_csv = os.path.join(FOLDS_DIR, f"{fold}_train.csv")
    df_tr = pd.read_csv(tr_csv)
    cmu = np.asarray(cond_stats["mean"], float)
    csd = np.asarray(cond_stats["std"], float)
    tbill_i = ENC_COLS.index("tbill_wr")
    metab_i = ENC_COLS.index("metab_13w")
    tbill_z_lvls = {p: (np.percentile(df_tr["tbill_wr"].dropna(), p) - cmu[tbill_i]) / csd[tbill_i]
                    for p in PCTLS}
    metab_z_lvls = {p: (np.percentile(df_tr["metab_13w"].dropna(), p) - cmu[metab_i]) / csd[metab_i]
                    for p in PCTLS}

    # raw rescale 재료 (forward GARCH σ)
    tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
    _gsig = df_te["garch_sigma"].to_numpy(float)
    _gmu = df_te["garch_mu"].to_numpy(float)
    _gz = df_te["sp_return"].to_numpy(float)
    _oidx = np.where(valid_mask)[0]; n_orig = len(_oidx)
    _om = float(df_te["garch_omega"].iloc[0]); _al = float(df_te["garch_alpha"].iloc[0])
    _be = float(df_te["garch_beta"].iloc[0]); _mu = float(_gmu[0])
    _orow = _oidx + (PAST_LEN - 1)
    _s2 = _gsig[_orow] ** 2
    _e2 = (_gz[_orow] * _gsig[_orow]) ** 2
    sig_arr = np.array([[_gsig[int(_oidx[ii]) + PAST_LEN + tau]
                         for tau in range(FUTURE_LEN)] for ii in range(n_orig)])
    mu_arr = np.array([[_gmu[int(_oidx[ii]) + PAST_LEN + tau]
                        for tau in range(FUTURE_LEN)] for ii in range(n_orig)])
    actual_raw = (Yte.numpy() * tsd + tmu) * sig_arr + mu_arr

    # 전 기간 origin 균등 subsample (속도; paired 9-grid 라 origin set 고정 유지)
    if n_orig > N_ORIGIN_MAX:
        idx = np.linspace(0, n_orig - 1, N_ORIGIN_MAX).astype(int)
        Xte_dev = Xte_dev[idx]
        Xte_extra_dev = Xte_extra_dev[idx]
        last_sp_dev = last_sp_dev[idx]
        actual_raw = actual_raw[idx]
        _s2 = _s2[idx]; _e2 = _e2[idx]
        n_orig = N_ORIGIN_MAX

    rescale = dict(tmu=tmu, tsd=tsd, s2=_s2, e2=_e2, om=_om, al=_al, be=_be, mu=_mu)
    return dict(model=model, Xte_dev=Xte_dev, Xte_extra_dev=Xte_extra_dev,
                last_sp_dev=last_sp_dev, metab_idx=metab_i, rescale=rescale,
                actual_raw=actual_raw, n_orig=n_orig,
                tbill_z_lvls=tbill_z_lvls, metab_z_lvls=metab_z_lvls)


def sim_metrics(sim_z, rescale):
    r = rescale
    sim_zt = sim_z * r["tsd"] + r["tmu"]
    sim_raw = forward_garch_rescale(sim_zt, r["s2"], r["e2"],
                                    r["om"], r["al"], r["be"], r["mu"])
    f = sim_raw.ravel()
    return dict(mean=float(f.mean()), std=float(f.std(ddof=1)),
                skew=_skew(f), exkurt=_exkurt(f),
                var5=compute_var(f, 0.05), cvar5=compute_cvar(f, 0.05),
                cvar1=compute_cvar(f, 0.01))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 92)
    print(" 거시 시나리오 9-grid: 금리(tbill) × 초과유동성(metab) → equity tail 분포")
    print(f" 수준 = fold별 train raw percentile {PCTLS} → z (flat 경로), paired seed={SEED}")
    print(f" device={device}, n_sim={N_SIM}")
    print("=" * 92)

    for fold in FOLDS:
        ctx = load_fold(fold, device)
        if ctx is None:
            continue
        af = ctx["actual_raw"].ravel()
        print(f"\n=== {fold}  (actual: std {af.std(ddof=1):.5f}, "
              f"skew {_skew(af):+.3f}, exkurt {_exkurt(af):+.2f}, "
              f"CVaR1 {compute_cvar(af, 0.01):+.5f})")
        print(f"  {'tbill':>6} {'metab':>6} {'mean':>10} {'std':>8} "
              f"{'skew':>8} {'exkurt':>8} {'VaR5':>9} {'CVaR5':>9} {'CVaR1':>9}")
        for pt in PCTLS:                      # 금리: 하락→상승
            for pm in PCTLS:                  # 유동성: 감소→증가
                torch.manual_seed(SEED)        # paired: 9칸 같은 난수
                sim_z = rollout_scenario(
                    ctx["model"], ctx["Xte_dev"], ctx["Xte_extra_dev"],
                    ctx["last_sp_dev"], ctx["tbill_z_lvls"][pt],
                    ctx["metab_z_lvls"][pm], ctx["metab_idx"], device)
                m = sim_metrics(sim_z, ctx["rescale"])
                print(f"  {PCTL_LABEL[pt]:>6} {PCTL_LABEL[pm]:>6} "
                      f"{m['mean']:+.6f} {m['std']:.5f} {m['skew']:+.3f} "
                      f"{m['exkurt']:+.2f} {m['var5']:+.5f} {m['cvar5']:+.5f} "
                      f"{m['cvar1']:+.5f}")

    print("\n[해석] metab(유동성) lo→hi 갈 때 같은 금리 행 안에서 skew/CVaR 가 "
          "악화(좌측↑·tail↑)되고, fold 마다 방향이 일관되면 → debasement→equity tail 입증.")


if __name__ == "__main__":
    main()
