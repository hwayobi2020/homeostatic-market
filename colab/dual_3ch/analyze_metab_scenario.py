"""metab counterfactual + ablation — monetary debasement(metab_13w)가 생성 시나리오를 바꾸는가.

thesis-critical: 논문 기여가 "monetary debasement 를 conditioning 해 equity tail 을
설명한다"이므로, metab 경로를 바꿀 때 생성 분포가 실제로 변하는지(counterfactual)와
metab 정보를 제거하면 분포가 달라지는지(ablation)를 학습된 MLP best.pt 로 재학습 없이
확인한다.  변하지 않으면 metab conditioning 은 장식이고 논문 핵심이 무너진다.

설계:
  - 같은 test origin·같은 모델에서 미래 metab 경로만 perturb (과거 metab·tbill 은 고정).
  - counterfactual: 미래 metab z 를 level shift {-2,-1,0,+1,+2}σ → 생성 분포 지표가
    shift 에 monotonic 반응하는지.
  - ablation: 미래 metab unmask(actual) vs mask(0) → 분포 차이 = metab 기여.
  - raw 단위 복원은 evaluate_test 와 동일 (forward GARCH σ, look-ahead 없음).

best.pt meta 에 past_encoder_type/past_summary_dim 이 저장되지 않으므로, run_garch_macroenc
의 macroenc 셋업(ENC_COLS, encoder=mlp, past d=64)을 복제해 모델을 재구성한다.

Usage (Colab): !python colab/dual_3ch/analyze_metab_scenario.py
"""
import os
import sys

import numpy as np
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

# === macroenc 셋업 복제 (run_garch_macroenc.py 와 동일) ===
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

FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEED = 2026
SHIFTS = [-2.0, -1.0, 0.0, 1.0, 2.0]    # 미래 metab z level shift (σ 단위)
N_SIM = 1000
CHUNK = 8


def _skew(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def _exkurt(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 4) - 3.0)


def rebuild_model(ckpt, device):
    """best.pt model_state 로 macroenc-MLP 모델 재구성 (구조는 학습과 동일해야 load 성공)."""
    spec = BEST_SPECS["mlp"]
    model = MambaFlowAR(
        d_input=len(ENC_COLS),
        d_model=spec["d_model"],
        n_flow_layers=spec["n_flow_layers"],
        n_flow_hidden=spec["n_flow_hidden"],
        dropout=spec.get("dropout", 0.0),
        extra_context_dim=1,                 # sp_std_13w (volDC)
        encoder_type="mlp",
        mlp_num_layers=spec["mlp_num_layers"],
        direct_prev_return=True,
        use_past_summary=True,
        past_encoder_type="mlp",
        past_summary_dim=64,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def rollout(model, Xte_dev, Xte_extra_dev, last_sp_dev, metab_idx,
            metab_shift=0.0, metab_mask=False):
    """AR rollout 으로 z_t 경로 생성.

    metab_shift : 미래 metab z 에 더할 level (σ).  counterfactual 용.
    metab_mask  : True 면 미래 metab 를 통째 0 (정보 제거).  ablation 용.
    과거 구간·tbill·extra(volDC)는 건드리지 않는다.
    """
    n = Xte_dev.shape[0]
    out = []
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        x_past = Xte_dev[s:e, :PAST_LEN, :]
        fut_tbill = Xte_dev[s:e, PAST_LEN:, T.TBILL_CH]
        last_sp = last_sp_dev[s:e]
        extra = Xte_extra_dev[s:e] if Xte_extra_dev is not None else None
        fmac = Xte_dev[s:e, PAST_LEN:, [metab_idx]].clone()    # (k, FUTURE_LEN, 1)
        if metab_mask:
            fmac = torch.zeros_like(fmac)
        else:
            fmac = fmac + metab_shift
        sim = model.ar_sample(x_past, fut_tbill, last_sp, N_SIM,
                              extra_context=extra, future_macro_z=fmac)
        out.append(sim.cpu())
    return torch.cat(out, dim=0).numpy()                        # (n, N_SIM, FUTURE_LEN)


def load_fold(fold, device):
    """best.pt + test 데이터 + raw rescale 재료를 준비.  없으면 None."""
    bp = os.path.join(RESULT_DIR,
                      f"garch_flow_ar_macroenc_pastMlp_s{SEED}_{fold}_best.pt")
    if not os.path.exists(bp):
        print(f"[missing best.pt] {os.path.basename(bp)}")
        return None
    ckpt = torch.load(bp, map_location=device)
    meta = ckpt["meta"]
    cond_stats = meta["cond_stats"]
    target_stats = meta["target_stats"]
    extra_stats = meta.get("extra_stats")
    model = rebuild_model(ckpt, device)

    gp = garch_preprocess_fold(FOLDS_DIR, fold, RESULT_DIR)       # deterministic 재생성
    test_csv = gp["test"]
    Xte, Yte, _, _, n_te = cached_load_windows_seq(
        test_csv, cond_stats=cond_stats, target_stats=target_stats)
    Xte_dev = Xte.to(device)

    Xte_extra, _, n_ex, _ = load_extra_context(
        test_csv, [DC_COLS], extra_stats=extra_stats)
    if n_ex != Xte.shape[0]:
        sys.exit(f"[FATAL] extra valid {n_ex} != main {Xte.shape[0]} ({fold})")
    Xte_extra_dev = Xte_extra.to(device)

    valid_mask, z_te, df_te = compute_valid_mask(test_csv, cond_stats)
    n_w = z_te.shape[0] - PAST_LEN - FUTURE_LEN + 1
    last_full = z_te[PAST_LEN - 1: PAST_LEN - 1 + n_w, T.SP_CH]
    last_valid = last_full[valid_mask].astype(np.float32)
    if len(last_valid) != Xte.shape[0]:
        sys.exit(f"[FATAL] valid mask mismatch {len(last_valid)} vs {Xte.shape[0]}")
    last_sp_dev = torch.from_numpy(last_valid).to(device)

    # raw rescale 재료 (forward GARCH σ; evaluate_test 와 동일)
    tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
    _gsig = df_te["garch_sigma"].to_numpy(float)
    _gmu = df_te["garch_mu"].to_numpy(float)
    _gz = df_te["sp_return"].to_numpy(float)
    _oidx = np.where(valid_mask)[0]
    n_orig = len(_oidx)
    _om = float(df_te["garch_omega"].iloc[0])
    _al = float(df_te["garch_alpha"].iloc[0])
    _be = float(df_te["garch_beta"].iloc[0])
    _mu = float(_gmu[0])
    _orow = _oidx + (PAST_LEN - 1)
    _s2 = _gsig[_orow] ** 2
    _e2 = (_gz[_orow] * _gsig[_orow]) ** 2
    sig_arr = np.array([[_gsig[int(_oidx[ii]) + PAST_LEN + tau]
                         for tau in range(FUTURE_LEN)] for ii in range(n_orig)])
    mu_arr = np.array([[_gmu[int(_oidx[ii]) + PAST_LEN + tau]
                        for tau in range(FUTURE_LEN)] for ii in range(n_orig)])
    actual_raw = (Yte.numpy() * tsd + tmu) * sig_arr + mu_arr

    rescale = dict(tmu=tmu, tsd=tsd, s2=_s2, e2=_e2, om=_om, al=_al, be=_be, mu=_mu)
    return dict(model=model, Xte_dev=Xte_dev, Xte_extra_dev=Xte_extra_dev,
                last_sp_dev=last_sp_dev, rescale=rescale,
                actual_raw=actual_raw, n_orig=n_orig)


def sim_metrics(sim_z, rescale):
    r = rescale
    sim_zt = sim_z * r["tsd"] + r["tmu"]
    sim_raw = forward_garch_rescale(sim_zt, r["s2"], r["e2"],
                                    r["om"], r["al"], r["be"], r["mu"])
    f = sim_raw.ravel()
    return dict(mean=float(f.mean()), std=float(f.std(ddof=1)),
                skew=_skew(f), exkurt=_exkurt(f),
                var5=compute_var(f, 0.05), cvar5=compute_cvar(f, 0.05),
                var1=compute_var(f, 0.01), cvar1=compute_cvar(f, 0.01))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metab_idx = ENC_COLS.index("metab_13w")
    print("=" * 96)
    print(" metab counterfactual + ablation  (MLP 과거요약, d=64)")
    print(" monetary debasement(metab_13w) 경로가 생성 시나리오를 실제로 바꾸는가")
    print(f" device={device}, n_sim={N_SIM}, shifts={SHIFTS} (σ)")
    print("=" * 96)

    for fold in FOLDS:
        ctx = load_fold(fold, device)
        if ctx is None:
            continue
        model = ctx["model"]; rescale = ctx["rescale"]
        af = ctx["actual_raw"].ravel()
        print(f"\n--- {fold}  (n_origin={ctx['n_orig']}; actual: "
              f"std {af.std(ddof=1):.5f}, skew {_skew(af):+.3f}, "
              f"exkurt {_exkurt(af):+.2f})")

        # counterfactual
        print(" [counterfactual] 미래 metab z level shift")
        print(f"   {'shift':>7} {'mean':>10} {'std':>8} {'skew':>8} "
              f"{'exkurt':>8} {'VaR5':>9} {'CVaR5':>9} {'CVaR1':>9}")
        for sh in SHIFTS:
            sim_z = rollout(model, ctx["Xte_dev"], ctx["Xte_extra_dev"],
                            ctx["last_sp_dev"], metab_idx, metab_shift=sh)
            m = sim_metrics(sim_z, rescale)
            tag = "actual" if sh == 0.0 else f"{sh:+.0f}sd"
            print(f"   {tag:>7} {m['mean']:+.6f} {m['std']:.5f} {m['skew']:+.3f} "
                  f"{m['exkurt']:+.2f} {m['var5']:+.5f} {m['cvar5']:+.5f} "
                  f"{m['cvar1']:+.5f}")

        # ablation
        print(" [ablation] 미래 metab unmask(actual) vs mask(0)")
        sim_un = rollout(model, ctx["Xte_dev"], ctx["Xte_extra_dev"],
                         ctx["last_sp_dev"], metab_idx, metab_shift=0.0)
        sim_ma = rollout(model, ctx["Xte_dev"], ctx["Xte_extra_dev"],
                         ctx["last_sp_dev"], metab_idx, metab_mask=True)
        mu_, mm_ = sim_metrics(sim_un, rescale), sim_metrics(sim_ma, rescale)
        print(f"   {'cond':>8} {'mean':>10} {'std':>8} {'skew':>8} "
              f"{'exkurt':>8} {'CVaR5':>9} {'CVaR1':>9}")
        for name, mm in [("unmask", mu_), ("mask0", mm_)]:
            print(f"   {name:>8} {mm['mean']:+.6f} {mm['std']:.5f} {mm['skew']:+.3f} "
                  f"{mm['exkurt']:+.2f} {mm['cvar5']:+.5f} {mm['cvar1']:+.5f}")

    print("\n[해석] counterfactual 에서 shift 에 따라 skew/CVaR 가 monotonic 하게 움직이고, "
          "ablation 에서 unmask vs mask 차이가 뚜렷하면 → metab conditioning 이 실제 작동.")


if __name__ == "__main__":
    main()
