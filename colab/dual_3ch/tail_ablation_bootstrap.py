"""§4.3.3 유의성 — full vs (maskall / metab_drop) 좌꼬리(skew·CVaR1%·IHL) paired-origin bootstrap.

저장된 ablation 결과는 pooled 스칼라뿐이라, 체크포인트(.pt)로 *추론만* 재샘플하여
per-origin raw sim 을 얻고, origin 을 paired 복원추출해 Δ(full−ablation)의 CI/p 를 구한다.
(학습 0. evaluate_test 의 샘플링 경로 L1276-1328 를 그대로 복제 → 마스킹 정확.)

config (run_ablations_rawvol.py 와 동일 spec):
  full       : cols 5(+metab), MASK_FUTURE_TBILL=False, unmask=[metab_13w]   tag=rvP2mainMlp_pd64_fl4_fh128
  maskall    : cols 5,           MASK_FUTURE_TBILL=True,  unmask=[]            tag=rvAbl_maskall
  metab_drop : cols 4(-metab),   MASK_FUTURE_TBILL=False, unmask=[]            tag=rvAbl_metab_drop

검정: 같은 fold·같은 test origin 에서 full 과 ablation 평가 → origin paired bootstrap.
  공통 seed 2026·2027·2028 사용, origin 별 3-seed sim pool.
  Δskew, ΔCVaR1%, Δuw_cvar1 (full − ablation; 음수 = full 이 더 깊음/꼬리 큼).
  fold별 + 전체 pooled.  p = 2 × min(P(Δ>0), P(Δ<0)).

한계(정직): full·ablation 은 학습이 다른 별도 모델. 3-seed pool 로 줄이되 Δ 에 학습차 일부 포함
  → "config 가 pooled 꼬리통계를 origin 노이즈 이상으로 바꾸는가" 검정.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/tail_ablation_bootstrap.py
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

from rawvol_helpers import patch_rawvol                                # noqa: E402
patch_rawvol()
import train_garch_flow as T                                          # noqa: E402
from train_garch_flow import (                                        # noqa: E402
    MambaFlowAR, cached_load_windows_seq, load_extra_context,
    compute_valid_mask, forward_garch_rescale, garch_preprocess_fold,
    compute_cvar, PAST_LEN, FUTURE_LEN,
)

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
LABELS = {"F_gfc": "금융위기", "F_long_A": "회복기", "F_long_B_origin": "코로나", "F_long": "긴축기"}
SEEDS = [2026, 2027, 2028]              # full·ablation 공통 seed
N_SIM = 1000
CHUNK = 8
N_BOOT = 1000
DC_COLS = ["sp_std_13w", "sp_skew_13w"]
LK = dict(d_model=128, mlp_layers=4, flow_layers=4, flow_hidden=128, pd=64, dropout=0.2)

# config: (cols, mask_future_tbill, unmask, tag_template)
FULL_TAG = "rvP2mainMlp_pd64_fl4_fh128"
CONFIGS = {
    "full":       (["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"], False, ["metab_13w"],
                   "garch_flow_ar_%s_s{seed}_{fold}_best.pt" % FULL_TAG),
    "maskall":    (["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"], True, [],
                   "garch_flow_ar_rvAbl_maskall_s{seed}_{fold}_best.pt"),
    "metab_drop": (["sp_return", "tbill_wr", "ads_lag", "wti_wr"], False, [],
                   "garch_flow_ar_rvAbl_metab_drop_s{seed}_{fold}_best.pt"),
}


def set_cond(cols):
    cols = list(cols)
    T.COND_COLS = cols; T.N_CHANNELS = len(cols)
    T.SP_CH = cols.index("sp_return"); T.TBILL_CH = cols.index("tbill_wr")
    T.MACRO_CH = [c for c in range(len(cols)) if c not in (T.SP_CH, T.TBILL_CH)]
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass


def rebuild(ckpt, d_input, device):
    m = MambaFlowAR(
        d_input=d_input, d_model=LK["d_model"], n_flow_layers=LK["flow_layers"],
        n_flow_hidden=LK["flow_hidden"], dropout=LK["dropout"], extra_context_dim=len(DC_COLS),
        encoder_type="mlp", mlp_num_layers=LK["mlp_layers"], direct_prev_return=True,
        use_past_summary=True, past_encoder_type="mlp", past_summary_dim=LK["pd"],
    ).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    return m


@torch.no_grad()
def sample_config(name, fold, seed, device):
    """returns sim_paths_raw (n_orig, N_SIM, FUTURE_LEN) for given config/fold/seed, or None."""
    cols, mask_tbill, unmask, tagtmpl = CONFIGS[name]
    bp = os.path.join(RESULT_DIR, tagtmpl.format(seed=seed, fold=fold))
    if not os.path.exists(bp):
        print(f"  [missing] {os.path.basename(bp)}"); return None
    set_cond(cols)
    T.MASK_FUTURE_TBILL = mask_tbill
    T.FUTURE_UNMASK_MACRO_COLS = list(unmask)
    T.ENCODER_MASK_SP = False

    ckpt = torch.load(bp, map_location=device); meta = ckpt["meta"]
    cond_stats = meta["cond_stats"]; target_stats = meta["target_stats"]
    extra_stats = meta.get("extra_stats")
    model = rebuild(ckpt, len(cols), device)

    test_csv = garch_preprocess_fold(FOLDS_DIR, fold, RESULT_DIR)["test"]
    Xte, Yte, _, _, _ = cached_load_windows_seq(test_csv, cond_stats=cond_stats, target_stats=target_stats)
    Xte_dev = Xte.to(device)
    Xe, _, n_ex, _ = load_extra_context(test_csv, DC_COLS, extra_stats=extra_stats)
    Xe_dev = Xe.to(device) if Xe is not None else None

    valid_mask, z_te, df_te = compute_valid_mask(test_csv, cond_stats)
    n_w = z_te.shape[0] - PAST_LEN - FUTURE_LEN + 1
    last_full = z_te[PAST_LEN - 1: PAST_LEN - 1 + n_w, T.SP_CH]
    last_sp = torch.from_numpy(last_full[valid_mask].astype(np.float32)).to(device)
    n_orig = Xte_dev.shape[0]

    torch.manual_seed(seed)
    um_idx = [cols.index(c) for c in unmask if c in cols]
    out = []
    for s in range(0, n_orig, CHUNK):
        e = min(n_orig, s + CHUNK)
        xp = Xte_dev[s:e, :PAST_LEN, :]
        ftb = Xte_dev[s:e, PAST_LEN:, T.TBILL_CH]
        lsp = last_sp[s:e]
        ex = Xe_dev[s:e] if Xe_dev is not None else None
        fmac = Xte_dev[s:e, PAST_LEN:, um_idx] if um_idx else None
        sim = model.ar_sample(xp, ftb, lsp, N_SIM, extra_context=ex, future_macro_z=fmac)
        out.append(sim.cpu())
    sim_z = torch.cat(out, dim=0).numpy()                                  # (n_orig, N_SIM, T) z

    # rescale to raw (forward GARCH σ, evaluate_test L1294-1328 와 동일)
    tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
    _gsig = df_te["garch_sigma"].to_numpy(float); _gz = df_te["sp_return"].to_numpy(float)
    _oidx = np.where(valid_mask)[0]
    _om = float(df_te["garch_omega"].iloc[0]); _al = float(df_te["garch_alpha"].iloc[0])
    _be = float(df_te["garch_beta"].iloc[0]); _mu = float(df_te["garch_mu"].iloc[0])
    _orow = _oidx + (PAST_LEN - 1)
    s2 = _gsig[_orow] ** 2
    e2 = (_gz[_orow] * _gsig[_orow]) ** 2
    sim_raw = forward_garch_rescale(sim_z * tsd + tmu, s2, e2, _om, _al, _be, _mu)
    return sim_raw                                                        # (n_orig, N_SIM, T)


def pooled_sims(name, fold, device):
    """3-seed pool → (n_orig, 3*N_SIM, T) raw."""
    chunks = []
    for seed in SEEDS:
        sr = sample_config(name, fold, seed, device)
        if sr is not None:
            chunks.append(sr)
    if not chunks:
        return None
    return np.concatenate(chunks, axis=1)                                 # (n_orig, k*N_SIM, T)


def underwater(sim):
    """(n_orig, M, T) -> per (orig,M) worst cumulative drawdown (<=0)."""
    cum = np.cumsum(sim, axis=2)
    z0 = np.zeros((cum.shape[0], cum.shape[1], 1), dtype=cum.dtype)
    cum0 = np.concatenate([z0, cum], axis=2)
    return cum0.min(axis=2)                                               # (n_orig, M)


def _skew_flat(a):
    m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def stats_from_origins(ret_by_orig, uw_by_orig, idx):
    """resampled origin idx -> pooled skew, cvar1(returns), uw_cvar1."""
    r = np.concatenate([ret_by_orig[i] for i in idx])
    uw = np.concatenate([uw_by_orig[i] for i in idx])
    return _skew_flat(r), compute_cvar(r, 0.01), compute_cvar(uw, 0.01)


def paired_bootstrap(full_raw, abl_raw, rng):
    """returns dict of Δ(full-abl) mean/CI/p for skew, cvar1, uw_cvar1 (paired-origin)."""
    n_orig = full_raw.shape[0]
    full_ret = [full_raw[i].ravel() for i in range(n_orig)]
    abl_ret = [abl_raw[i].ravel() for i in range(n_orig)]
    full_uw = list(underwater(full_raw))
    abl_uw = list(underwater(abl_raw))
    # point estimate (all origins)
    base_full = stats_from_origins(full_ret, full_uw, np.arange(n_orig))
    base_abl = stats_from_origins(abl_ret, abl_uw, np.arange(n_orig))
    d_point = tuple(f - a for f, a in zip(base_full, base_abl))
    d_boot = np.empty((N_BOOT, 3))
    for b in range(N_BOOT):
        idx = rng.integers(0, n_orig, n_orig)
        sf = stats_from_origins(full_ret, full_uw, idx)
        sa = stats_from_origins(abl_ret, abl_uw, idx)
        d_boot[b] = [f - a for f, a in zip(sf, sa)]
    res = {}
    for j, key in enumerate(("skew", "cvar1", "uw_cvar1")):
        col = d_boot[:, j]
        lo, hi = np.percentile(col, [2.5, 97.5])
        p = 2.0 * min((col > 0).mean(), (col < 0).mean())
        res[key] = dict(d=float(d_point[j]), lo=float(lo), hi=float(hi), p=float(p),
                        full=float(base_full[j]), abl=float(base_abl[j]))
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print(f"# §4.3.3 paired-origin bootstrap — full vs maskall/metab_drop (device={device}, B={N_BOOT})")
    print(f"#  Δ = full − ablation (음수 = full 이 더 깊음).  seeds={SEEDS} pool, N_SIM={N_SIM}")
    print("#" * 100)

    rng = np.random.default_rng(2026)
    pool_full_ret, pool_full_uw = [], []
    pool_abl_ret = {"maskall": [], "metab_drop": []}
    pool_abl_uw = {"maskall": [], "metab_drop": []}

    for fold in FOLDS:
        print(f"\n=== {LABELS[fold]} [{fold}] " + "=" * 50)
        full_raw = pooled_sims("full", fold, device)
        if full_raw is None:
            print("  [skip] full 없음"); continue
        for abl in ("maskall", "metab_drop"):
            abl_raw = pooled_sims(abl, fold, device)
            if abl_raw is None:
                print(f"  [skip] {abl} 없음"); continue
            if abl_raw.shape[0] != full_raw.shape[0]:
                print(f"  [WARN] origin 수 불일치 full={full_raw.shape[0]} {abl}={abl_raw.shape[0]} → skip")
                continue
            res = paired_bootstrap(full_raw, abl_raw, rng)
            print(f"  full vs {abl}:")
            for key in ("skew", "cvar1", "uw_cvar1"):
                r = res[key]
                sig = "*" if r["p"] < 0.05 else " "
                print(f"    {key:<9} full={r['full']:+.4f} {abl}={r['abl']:+.4f}  "
                      f"Δ={r['d']:+.4f}  95%CI[{r['lo']:+.4f},{r['hi']:+.4f}]  p={r['p']:.3f} {sig}")
            # accumulate pooled (origin-level), once per abl
            if abl == "maskall":
                for i in range(full_raw.shape[0]):
                    pool_full_ret.append(full_raw[i].ravel())
                pool_full_uw.extend(list(underwater(full_raw)))
            for i in range(abl_raw.shape[0]):
                pool_abl_ret[abl].append(abl_raw[i].ravel())
            pool_abl_uw[abl].extend(list(underwater(abl_raw)))

    # pooled across folds (origin-level paired)
    print("\n" + "=" * 100)
    print("[POOLED across 4 folds] paired-origin bootstrap")
    for abl in ("maskall", "metab_drop"):
        if not pool_abl_ret[abl] or not pool_full_ret:
            print(f"  [skip pooled {abl}]"); continue
        n = min(len(pool_full_ret), len(pool_abl_ret[abl]))
        fr, fu = pool_full_ret[:n], pool_full_uw[:n]
        ar, au = pool_abl_ret[abl][:n], pool_abl_uw[abl][:n]
        base_f = stats_from_origins(fr, fu, np.arange(n))
        base_a = stats_from_origins(ar, au, np.arange(n))
        dpt = tuple(f - a for f, a in zip(base_f, base_a))
        db = np.empty((N_BOOT, 3))
        for b in range(N_BOOT):
            idx = rng.integers(0, n, n)
            sf = stats_from_origins(fr, fu, idx); sa = stats_from_origins(ar, au, idx)
            db[b] = [f - a for f, a in zip(sf, sa)]
        print(f"  full vs {abl} (n_orig={n}):")
        for j, key in enumerate(("skew", "cvar1", "uw_cvar1")):
            col = db[:, j]; lo, hi = np.percentile(col, [2.5, 97.5])
            p = 2.0 * min((col > 0).mean(), (col < 0).mean())
            sig = "*" if p < 0.05 else " "
            print(f"    Δ{key:<9} = {dpt[j]:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  p={p:.3f} {sig}")
    print("\n[판정] Δ<0 & p<0.05 = ablation 제거 시 좌꼬리가 유의하게 얕아짐(=full 이 깊음) → 경로/유동성 기여 유의.")


if __name__ == "__main__":
    main()
