"""§4.1 보강 — intra-horizon loss(IHL) 적합도(calibration): 실제 vs 모델.

§4.2 는 IHL 을 *가상* 시나리오끼리 비교했다.  여기서는 *실제* test fold 에서
모델이 만든 IHL 분포가 실제 실현 IHL 을 얼마나 잘 재현하는지(sim-vs-actual)를 본다.

방법 (검증 origin 마다):
  - 조건 = *실제* 미래 거시 경로(Xte 미래 구간, unmask tbill/metab) — 가상 아님.
  - sim IHL  = 모델 n_sim 경로의 보유기간 내 최저 누적(진입 대비) 분포.
  - actual IHL = 실제 실현 13주 경로의 보유기간 내 최저 누적(진입 대비), 1개 값.
  - Coverage = actual IHL 이 sim IHL 의 80%·95% 중심구간에 드는 origin 비율.
  - 평균 비 = mean(actual IHL) / mean(sim IHL)  (편향 확인).

복원식은 train_garch_flow.py 의 eval(actual: 실현 σ_t / sim: origin-frozen forward σ)
을 *그대로* 따른다 (라인 1293~1328).  LOCKED MLP 본모형, 재학습 0 · inference only.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/analyze_ihl_calibration_rawvol.py
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

# ── raw-vol monkey-patch 먼저 (forward σ origin-frozen) ──
from rawvol_helpers import patch_rawvol                              # noqa: E402
patch_rawvol()

import train_garch_flow as T                                        # noqa: E402
from train_garch_flow import (                                      # noqa: E402
    MambaFlowAR, cached_load_windows_seq, load_extra_context,
    compute_valid_mask, forward_garch_rescale, garch_preprocess_fold,
    PAST_LEN, FUTURE_LEN,
)

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
CACHE_DIR = os.path.join(RESULT_DIR, "ihl_calib_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
DC_COLS_LIST = ["sp_std_13w", "sp_skew_13w"]      # LOCKED extra context (2채널)

# LOCKED 본모형 hyperparameter
LK_D_MODEL = 128
LK_MLP_LAYERS = 4
LK_FLOW_LAYERS = 4
LK_FLOW_HIDDEN = 128
LK_PD = 64
LK_DROPOUT = 0.2
TAG_PREFIX = f"rvP2mainMlp_pd{LK_PD}_fl{LK_FLOW_LAYERS}_fh{LK_FLOW_HIDDEN}"

FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEEDS = [2026, 2027, 2028]
N_SIM = 1000
CHUNK = 8


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


def rebuild_model(ckpt, device):
    model = MambaFlowAR(
        d_input=len(ENC_COLS), d_model=LK_D_MODEL,
        n_flow_layers=LK_FLOW_LAYERS, n_flow_hidden=LK_FLOW_HIDDEN,
        dropout=LK_DROPOUT, extra_context_dim=len(DC_COLS_LIST),
        encoder_type="mlp", mlp_num_layers=LK_MLP_LAYERS,
        direct_prev_return=True, use_past_summary=True,
        past_encoder_type="mlp", past_summary_dim=LK_PD,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def intra_horizon_loss(raw):
    """raw: (..., T) 주별 수익률.  진입(누적=0) prepend 후 보유기간 내 최저 누적 = IHL(≤0)."""
    cum = np.cumsum(raw, axis=-1)
    z0 = np.zeros(cum.shape[:-1] + (1,), dtype=cum.dtype)
    cum0 = np.concatenate([z0, cum], axis=-1)
    return cum0.min(axis=-1)


@torch.no_grad()
def run_fold_seed(fold, seed, device):
    """한 (fold,seed): 실제 미래 거시 조건 하 sim IHL vs actual IHL 적합도."""
    bp = os.path.join(RESULT_DIR, f"garch_flow_ar_{TAG_PREFIX}_s{seed}_{fold}_best.pt")
    if not os.path.exists(bp):
        print(f"[missing best.pt] {os.path.basename(bp)}")
        return None
    ckpt = torch.load(bp, map_location=device)
    meta = ckpt["meta"]
    cond_stats = meta["cond_stats"]; target_stats = meta["target_stats"]
    extra_stats = meta.get("extra_stats")
    model = rebuild_model(ckpt, device)

    # *** test fold (검증 origin) — 가상 아님 ***
    gp = garch_preprocess_fold(FOLDS_DIR, fold, RESULT_DIR)
    test_csv = gp["test"]

    Xte, Yte, _, _, _ = cached_load_windows_seq(
        test_csv, cond_stats=cond_stats, target_stats=target_stats)
    Xte_dev = Xte.to(device)
    Xte_extra, _, n_ex, _ = load_extra_context(
        test_csv, DC_COLS_LIST, extra_stats=extra_stats)
    if n_ex != Xte.shape[0]:
        sys.exit(f"[FATAL] extra valid {n_ex} != main {Xte.shape[0]} ({fold},{seed})")
    Xte_extra_dev = Xte_extra.to(device)

    valid_mask, z_te, df_te = compute_valid_mask(test_csv, cond_stats)
    n = z_te.shape[0]
    n_w = n - PAST_LEN - FUTURE_LEN + 1
    last_full = z_te[PAST_LEN - 1: PAST_LEN - 1 + n_w, T.SP_CH]
    last_sp_valid = last_full[valid_mask].astype(np.float32)
    assert len(last_sp_valid) == Xte.shape[0], \
        f"valid mask mismatch {len(last_sp_valid)} vs {Xte.shape[0]} ({fold},{seed})"
    last_sp_dev = torch.from_numpy(last_sp_valid).to(device)
    n_origins = Xte.shape[0]

    # --- AR rollout: 조건 = 실제 미래 거시(Xte 미래 구간) ---
    torch.manual_seed(seed)
    _um_idx = [T.COND_COLS.index(c) for c in T.FUTURE_UNMASK_MACRO_COLS
               if c in T.COND_COLS]
    sim_chunks = []
    for s in range(0, n_origins, CHUNK):
        e = min(n_origins, s + CHUNK)
        x_past = Xte_dev[s:e, :PAST_LEN, :]
        fut_tbill = Xte_dev[s:e, PAST_LEN:, T.TBILL_CH]              # 실제 미래 tbill
        fmac = Xte_dev[s:e, PAST_LEN:, _um_idx] if _um_idx else None  # 실제 미래 metab
        ex = Xte_extra_dev[s:e]
        sim = model.ar_sample(x_past, fut_tbill, last_sp_dev[s:e], N_SIM,
                              extra_context=ex, future_macro_z=fmac)  # (k, n_sim, T)
        sim_chunks.append(sim.cpu())
    sim_paths_z = torch.cat(sim_chunks, dim=0).numpy()              # (n_v, n_sim, T)

    # --- raw 복원 (train_garch_flow eval 라인 1293~1328 그대로) ---
    tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
    actual_z = Yte.numpy()                                         # (n_v, T)
    actual_zt = actual_z * tsd + tmu
    sim_zt = sim_paths_z * tsd + tmu

    _gsig = df_te["garch_sigma"].to_numpy(float)
    _gmu = df_te["garch_mu"].to_numpy(float)
    _gz = df_te["sp_return"].to_numpy(float)
    _oidx = np.where(valid_mask)[0]
    # actual: 실현 σ_t (filtered) 로 복원 (정답 라벨, 누수 아님)
    sigma_arr = np.array([[_gsig[int(_oidx[ii]) + PAST_LEN + tau]
                           for tau in range(FUTURE_LEN)] for ii in range(n_origins)])
    mu_arr = np.array([[_gmu[int(_oidx[ii]) + PAST_LEN + tau]
                        for tau in range(FUTURE_LEN)] for ii in range(n_origins)])
    actual_raw = actual_zt * sigma_arr + mu_arr                    # (n_v, T)
    # sim: origin-frozen forward σ (raw-vol patch)
    _om = float(df_te["garch_omega"].iloc[0])
    _al = float(df_te["garch_alpha"].iloc[0])
    _be = float(df_te["garch_beta"].iloc[0])
    _mu = float(_gmu[0])
    _orow = _oidx + (PAST_LEN - 1)
    _s2 = _gsig[_orow] ** 2
    _e2 = (_gz[_orow] * _gsig[_orow]) ** 2
    sim_raw = forward_garch_rescale(sim_zt, _s2, _e2, _om, _al, _be, _mu)  # (n_v, n_sim, T)

    # --- intra-horizon loss ---
    actual_ihl = intra_horizon_loss(actual_raw)                    # (n_v,)
    sim_ihl = intra_horizon_loss(sim_raw)                          # (n_v, n_sim)

    lo80 = np.percentile(sim_ihl, 10.0, axis=1); hi80 = np.percentile(sim_ihl, 90.0, axis=1)
    lo95 = np.percentile(sim_ihl, 2.5, axis=1); hi95 = np.percentile(sim_ihl, 97.5, axis=1)
    cov80 = float(((actual_ihl >= lo80) & (actual_ihl <= hi80)).mean())
    cov95 = float(((actual_ihl >= lo95) & (actual_ihl <= hi95)).mean())

    sim_ihl_per_origin_mean = sim_ihl.mean(axis=1)                 # (n_v,)
    return dict(
        n_origin=int(n_origins),
        cov80=cov80, cov95=cov95,
        mean_actual_ihl=float(actual_ihl.mean()),
        mean_sim_ihl=float(sim_ihl.mean()),
        median_actual_ihl=float(np.median(actual_ihl)),
        median_sim_ihl=float(np.median(sim_ihl)),
        # 분포 전체 비교용 보조: per-origin actual & sim-mean
        actual_ihl=actual_ihl.tolist(),
        sim_ihl_origin_mean=sim_ihl_per_origin_mean.tolist(),
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print(f"# §4.1 IHL 적합도 (sim-vs-actual) — LOCKED {TAG_PREFIX}, device={device}, n_sim={N_SIM}")
    print("# 조건 = 실제 미래 거시 경로(Xte unmask).  actual=실현σ복원 / sim=origin-frozen forward σ")
    print("#" * 100)

    for fold in FOLDS:
        for seed in SEEDS:
            cache = os.path.join(CACHE_DIR, f"{fold}_s{seed}.json")
            if os.path.exists(cache):
                print(f"[skip] {fold} s{seed}")
                continue
            r = run_fold_seed(fold, seed, device)
            if r is None:
                continue
            json.dump(r, open(cache, "w"), indent=2)
            print(f"[done] {fold} s{seed}  n={r['n_origin']}  "
                  f"cov80={r['cov80']:.3f} cov95={r['cov95']:.3f}  "
                  f"meanIHL act/sim={r['mean_actual_ihl']:+.4f}/{r['mean_sim_ihl']:+.4f}")

    # ── 집계: seed 평균±std ──
    print("\n" + "=" * 100)
    print("[집계] IHL 적합도 — 4 fold × 3 seed 평균±std")
    print(f"  {'fold':>16} {'n_orig':>7} {'cov80':>14} {'cov95':>14} "
          f"{'mean act IHL':>14} {'mean sim IHL':>14} {'act/sim 비':>10}")

    def _agg(vals):
        a = np.asarray(vals, float)
        return (float(a.mean()), float(a.std(ddof=0)))

    for fold in FOLDS:
        loaded = []
        for s in SEEDS:
            c = os.path.join(CACHE_DIR, f"{fold}_s{s}.json")
            if os.path.exists(c):
                loaded.append(json.load(open(c)))
        if not loaded:
            print(f"  {fold:>16}  (결과 없음)")
            continue
        n_orig = loaded[0]["n_origin"]
        c80_m, c80_s = _agg([d["cov80"] for d in loaded])
        c95_m, c95_s = _agg([d["cov95"] for d in loaded])
        ai_m, ai_s = _agg([d["mean_actual_ihl"] for d in loaded])
        si_m, si_s = _agg([d["mean_sim_ihl"] for d in loaded])
        ratio = ai_m / si_m if abs(si_m) > 1e-9 else float("nan")
        print(f"  {fold:>16} {n_orig:>7d} "
              f"{c80_m:.3f}±{c80_s:.3f} {c95_m:.3f}±{c95_s:.3f} "
              f"{ai_m:+.4f}±{ai_s:.4f} {si_m:+.4f}±{si_s:.4f} {ratio:>9.3f}")

    print("\n[판정] cov80≈0.80 / cov95≈0.95 면 IHL 잘 보정됨.  act/sim 비≈1 이면 편향 없음.")
    print("       (cov 가 낮으면 sim IHL 구간이 좁아 실제 손실을 과소; 높으면 과대.)")
    print("       caveat: 메인 4-fold test origin ~150-200 — 표본 충분하나 fold 1.5y 계열은 적음.")


if __name__ == "__main__":
    main()
