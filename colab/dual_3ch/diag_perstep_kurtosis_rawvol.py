"""4.3.5 gate 진단 — sim 과대첨도가 *exposure bias(teacher forcing)* 때문인가, *flow 자체* 인가.

검증 fold(실제 미래 거시 조건)에서 sim 수익률의 *시점별(per-τ)* 초과첨도를 본다:
  - τ=1→13 로 갈수록 exkurt 가 *증가* → free-running self-conditioning 누적 = exposure bias.
  - τ 따라 *평평* → per-step flow 가 원래 over-fat (teacher forcing 무관, scheduled sampling
    으로 안 고쳐짐).
actual 의 per-τ exkurt 도 같이 (실제는 ~평평할 것 — 비교 기준).

복원식은 train_garch_flow eval(actual=실현σ / sim=origin-frozen forward σ) 그대로.
LOCKED MLP, inference only.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/diag_perstep_kurtosis_rawvol.py
"""
import json
import os
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rawvol_helpers import patch_rawvol                              # noqa: E402
patch_rawvol()
import train_garch_flow as T                                        # noqa: E402
from train_garch_flow import (                                      # noqa: E402
    MambaFlowAR, cached_load_windows_seq, load_extra_context,
    compute_valid_mask, forward_garch_rescale, garch_preprocess_fold,
    PAST_LEN, FUTURE_LEN,
)

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(os.path.normpath(os.path.join(HERE, "..", "..")), "data", "folds_v33_vix_expanding")
CACHE_DIR = os.path.join(RESULT_DIR, "perstep_kurt_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
DC_COLS_LIST = ["sp_std_13w", "sp_skew_13w"]
LK = dict(d_model=128, mlp_layers=4, flow_layers=4, flow_hidden=128, pd=64, dropout=0.2)
TAG_PREFIX = f"rvP2mainMlp_pd{LK['pd']}_fl{LK['flow_layers']}_fh{LK['flow_hidden']}"
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEEDS = [2026, 2027, 2028]
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


def rebuild_model(ckpt, device):
    m = MambaFlowAR(
        d_input=len(ENC_COLS), d_model=LK["d_model"],
        n_flow_layers=LK["flow_layers"], n_flow_hidden=LK["flow_hidden"],
        dropout=LK["dropout"], extra_context_dim=len(DC_COLS_LIST),
        encoder_type="mlp", mlp_num_layers=LK["mlp_layers"],
        direct_prev_return=True, use_past_summary=True,
        past_encoder_type="mlp", past_summary_dim=LK["pd"],
    ).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    return m


def _exkurt(a):
    a = np.asarray(a, float).ravel(); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 4) - 3.0)


@torch.no_grad()
def run_fold_seed(fold, seed, device):
    bp = os.path.join(RESULT_DIR, f"garch_flow_ar_{TAG_PREFIX}_s{seed}_{fold}_best.pt")
    if not os.path.exists(bp):
        print(f"[missing] {os.path.basename(bp)}"); return None
    ckpt = torch.load(bp, map_location=device); meta = ckpt["meta"]
    cond_stats = meta["cond_stats"]; target_stats = meta["target_stats"]
    extra_stats = meta.get("extra_stats")
    model = rebuild_model(ckpt, device)

    gp = garch_preprocess_fold(FOLDS_DIR, fold, RESULT_DIR)
    test_csv = gp["test"]
    Xte, Yte, _, _, _ = cached_load_windows_seq(test_csv, cond_stats=cond_stats, target_stats=target_stats)
    Xte_dev = Xte.to(device)
    Xte_extra, _, n_ex, _ = load_extra_context(test_csv, DC_COLS_LIST, extra_stats=extra_stats)
    if n_ex != Xte.shape[0]:
        sys.exit(f"[FATAL] extra {n_ex} != {Xte.shape[0]} ({fold},{seed})")
    Xte_extra_dev = Xte_extra.to(device)

    valid_mask, z_te, df_te = compute_valid_mask(test_csv, cond_stats)
    n_w = z_te.shape[0] - PAST_LEN - FUTURE_LEN + 1
    last_full = z_te[PAST_LEN - 1: PAST_LEN - 1 + n_w, T.SP_CH]
    last_sp = torch.from_numpy(last_full[valid_mask].astype(np.float32)).to(device)
    n_origins = Xte.shape[0]
    torch.manual_seed(seed)
    _um = [T.COND_COLS.index(c) for c in T.FUTURE_UNMASK_MACRO_COLS if c in T.COND_COLS]

    sim_chunks = []
    for s in range(0, n_origins, CHUNK):
        e = min(n_origins, s + CHUNK)
        x_past = Xte_dev[s:e, :PAST_LEN, :]
        fut_tb = Xte_dev[s:e, PAST_LEN:, T.TBILL_CH]
        fmac = Xte_dev[s:e, PAST_LEN:, _um] if _um else None
        sim = model.ar_sample(x_past, fut_tb, last_sp[s:e], N_SIM,
                              extra_context=Xte_extra_dev[s:e], future_macro_z=fmac)
        sim_chunks.append(sim.cpu())
    sim_z = torch.cat(sim_chunks, dim=0).numpy()                    # (n_v, n_sim, T)

    tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
    actual_zt = Yte.numpy() * tsd + tmu
    sim_zt = sim_z * tsd + tmu
    _gsig = df_te["garch_sigma"].to_numpy(float); _gmu = df_te["garch_mu"].to_numpy(float)
    _gz = df_te["sp_return"].to_numpy(float); _oidx = np.where(valid_mask)[0]
    sigma_arr = np.array([[_gsig[int(_oidx[i]) + PAST_LEN + tau] for tau in range(FUTURE_LEN)]
                          for i in range(n_origins)])
    mu_arr = np.array([[_gmu[int(_oidx[i]) + PAST_LEN + tau] for tau in range(FUTURE_LEN)]
                       for i in range(n_origins)])
    actual_raw = actual_zt * sigma_arr + mu_arr                     # (n_v, T)
    _om = float(df_te["garch_omega"].iloc[0]); _al = float(df_te["garch_alpha"].iloc[0])
    _be = float(df_te["garch_beta"].iloc[0]); _mu = float(_gmu[0])
    _orow = _oidx + (PAST_LEN - 1)
    sim_raw = forward_garch_rescale(sim_zt, _gsig[_orow] ** 2,
                                    (_gz[_orow] * _gsig[_orow]) ** 2, _om, _al, _be, _mu)  # (n_v,n_sim,T)

    sim_exk = [_exkurt(sim_raw[:, :, tau]) for tau in range(FUTURE_LEN)]
    act_exk = [_exkurt(actual_raw[:, tau]) for tau in range(FUTURE_LEN)]
    return dict(n_origin=int(n_origins), sim_exk=sim_exk, act_exk=act_exk)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 92)
    print(f"# 4.3.5 gate: per-τ 초과첨도 (sim free-running vs actual) — LOCKED {TAG_PREFIX}, n_sim={N_SIM}")
    print("#  τ 따라 sim exkurt *증가* → exposure bias / *평평* → flow 자체 over-fat")
    print("#" * 92)

    for fold in FOLDS:
        for seed in SEEDS:
            c = os.path.join(CACHE_DIR, f"{fold}_s{seed}.json")
            if os.path.exists(c):
                print(f"[skip] {fold} s{seed}"); continue
            r = run_fold_seed(fold, seed, device)
            if r is None:
                continue
            json.dump(r, open(c, "w"), indent=2)
            print(f"[done] {fold} s{seed}")

    print("\n" + "=" * 92)
    print("[집계] per-τ 초과첨도 (seed 평균)  — τ=1..13")
    for fold in FOLDS:
        rs = [json.load(open(os.path.join(CACHE_DIR, f"{fold}_s{s}.json")))
              for s in SEEDS if os.path.exists(os.path.join(CACHE_DIR, f"{fold}_s{s}.json"))]
        if not rs:
            print(f"\n=== {fold}: 결과 없음"); continue
        sim = np.mean([r["sim_exk"] for r in rs], axis=0)
        act = np.mean([r["act_exk"] for r in rs], axis=0)
        print(f"\n=== {fold} ===")
        print("  τ      " + " ".join(f"{t+1:>5d}" for t in range(FUTURE_LEN)))
        print("  sim    " + " ".join(f"{v:>5.1f}" for v in sim))
        print("  actual " + " ".join(f"{v:>5.1f}" for v in act))
        print(f"  → sim exkurt τ1={sim[0]:.1f} → τ13={sim[-1]:.1f}  "
              f"(증가폭 {sim[-1]-sim[0]:+.1f})  | slope sign: "
              f"{'증가(exposure bias)' if sim[-1] > sim[0] + 1 else '평평/감소(flow 자체)'}")

    print("\n[판정] sim exkurt 가 τ1→τ13 뚜렷이 증가 = exposure bias(teacher forcing) → scheduled sampling 가치.")
    print("       평평하면 = per-step flow over-fat (teacher forcing 무관) → scheduled sampling 무효.")


if __name__ == "__main__":
    main()
