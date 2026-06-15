"""§4.3.3 — full / metab_drop / maskall 이 *실제(actual)* 꼬리를 얼마나 잘 재현하나.

지표: cov80, cov95, skew, IHL mean(uw_mean), IHL CVaR1%(uw_cvar1).
기준: "실제값에 가까운가" = |model − actual|.  Δerror = |full−actual| − |ablation−actual|.
      Δerror < 0 = full 이 실제에 더 가까움(= 경로/유동성이 실제 꼬리 재현에 기여).

IHL(uw_mean/uw_cvar1)은 어느 표에도 저장된 적 없음(표 4.12는 per-step CVaR1%) →
체크포인트로 *추론만(학습 0)* 재샘플하여 per-origin sim 생성 후 계산.

config (run_ablations_rawvol.py 와 동일 spec):
  full       : cols 5(+metab), MASK_FUTURE_TBILL=False, unmask=[metab_13w]  tag=rvP2mainMlp_pd64_fl4_fh128
  maskall    : cols 5,          MASK_FUTURE_TBILL=True,  unmask=[]           tag=rvAbl_maskall
  metab_drop : cols 4(-metab),  MASK_FUTURE_TBILL=False, unmask=[]           tag=rvAbl_metab_drop

비교: fold별 + 전체 pooled.  (pooling 은 fold 스케일 차이로 극단 fold 가 지배 → 둘 다 보고)
유의성: skew·uw_mean·uw_cvar1 에 대해 full vs maskall paired-origin bootstrap (같은 origin).
        metab_drop 은 채널이 달라 origin 이 다르면 점 비교만(페어 p 제외).
cov80/cov95 = 모델 구간이 실제를 덮는 비율 → 값만 보고(actual 행은 —).

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
SEEDS = [2026, 2027, 2028]
N_SIM = 1000
CHUNK = 8
N_BOOT = 1000
DC_COLS = ["sp_std_13w", "sp_skew_13w"]
LK = dict(d_model=128, mlp_layers=4, flow_layers=4, flow_hidden=128, pd=64, dropout=0.2)
ORDER = ["full", "metab_drop", "maskall"]
METRICS_ERR = ["skew", "uw_mean", "uw_cvar1"]      # actual 기준 오차 비교 대상

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
    """returns (sim_raw (n_orig,N_SIM,T), actual_raw (n_orig,T)) or None."""
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
    sim_z = torch.cat(out, dim=0).numpy()

    tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
    _gsig = df_te["garch_sigma"].to_numpy(float); _gz = df_te["sp_return"].to_numpy(float)
    _gmu = df_te["garch_mu"].to_numpy(float)
    _oidx = np.where(valid_mask)[0]
    _om = float(df_te["garch_omega"].iloc[0]); _al = float(df_te["garch_alpha"].iloc[0])
    _be = float(df_te["garch_beta"].iloc[0]); _mu = float(_gmu[0])
    _orow = _oidx + (PAST_LEN - 1)
    s2 = _gsig[_orow] ** 2
    e2 = (_gz[_orow] * _gsig[_orow]) ** 2
    sim_raw = forward_garch_rescale(sim_z * tsd + tmu, s2, e2, _om, _al, _be, _mu)

    # actual raw 미래 13주 (= z × σ + μ), 같은 valid origin
    sp_raw_full = _gz * _gsig + _gmu                                      # 전체 시계열 raw 수익률
    actual = np.stack([sp_raw_full[w + PAST_LEN: w + PAST_LEN + FUTURE_LEN]
                       for w in range(n_w)])[valid_mask]                  # (n_orig, T)
    return sim_raw, actual


def pooled_sims(name, fold, device):
    """3-seed pool → (sim (n_orig, k*N_SIM, T), actual (n_orig, T)) or None."""
    sims = []; actual = None
    for seed in SEEDS:
        r = sample_config(name, fold, seed, device)
        if r is None:
            continue
        sims.append(r[0]); actual = r[1]
    if not sims:
        return None
    return np.concatenate(sims, axis=1), actual


def underwater(sim):
    """(n,M,T) -> (n,M) worst cumulative drawdown(<=0)."""
    cum = np.cumsum(sim, axis=2)
    z0 = np.zeros((cum.shape[0], cum.shape[1], 1), dtype=cum.dtype)
    return np.concatenate([z0, cum], axis=2).min(axis=2)


def _skew(a):
    m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def prep(sim_raw, actual):
    """per-origin 2D 배열 묶음 (bootstrap 입력)."""
    n = sim_raw.shape[0]
    ret2d = sim_raw.reshape(n, -1)                       # (n, M*T)
    uw2d = underwater(sim_raw)                           # (n, M)
    a_ret2d = actual                                     # (n, T)
    a_uw2d = underwater(actual[:, None, :])              # (n, 1)
    return dict(ret=ret2d, uw=uw2d, a_ret=a_ret2d, a_uw=a_uw2d, n=n)


def metrics(P, idx):
    """주어진 origin idx 에서 모델·실제 지표 + cov."""
    mret = P["ret"][idx].ravel(); aret = P["a_ret"][idx].ravel()
    muw = P["uw"][idx].ravel(); auw = P["a_uw"][idx].ravel()
    lo80, hi80 = np.percentile(mret, [10, 90]); lo95, hi95 = np.percentile(mret, [2.5, 97.5])
    return dict(
        cov80=float(((aret >= lo80) & (aret <= hi80)).mean()),
        cov95=float(((aret >= lo95) & (aret <= hi95)).mean()),
        skew=(_skew(mret), _skew(aret)),
        uw_mean=(float(muw.mean()), float(auw.mean())),
        uw_cvar1=(compute_cvar(muw, 0.01), compute_cvar(auw, 0.01)),
    )


def paired_err_bootstrap(Pf, Pa, rng):
    """full(Pf) vs ablation(Pa), 같은 origin 가정.  Δerror=|full-actual|-|abl-actual|."""
    n = Pf["n"]
    out = {k: np.empty(N_BOOT) for k in METRICS_ERR}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        mf = metrics(Pf, idx); ma = metrics(Pa, idx)
        for k in METRICS_ERR:
            ef = abs(mf[k][0] - mf[k][1]); ea = abs(ma[k][0] - ma[k][1])
            out[k][b] = ef - ea
    res = {}
    for k in METRICS_ERR:
        col = out[k]; lo, hi = np.percentile(col, [2.5, 97.5])
        p = 2.0 * min((col > 0).mean(), (col < 0).mean())
        res[k] = dict(d=float(col.mean()), lo=float(lo), hi=float(hi), p=float(p))
    return res


def print_point_table(label, points):
    """points: dict name->metrics(all-origin).  actual 값은 full 기준."""
    print(f"\n  [{label}] cov80 / cov95 / skew / uw_mean / uw_cvar1  (skew·uw 는 model 값)")
    for name in ORDER:
        if name not in points:
            continue
        m = points[name]
        print(f"    {name:<11} {m['cov80']:.3f}  {m['cov95']:.3f}  "
              f"{m['skew'][0]:+.3f}  {m['uw_mean'][0]:+.4f}  {m['uw_cvar1'][0]:+.4f}")
    a = points["full"]
    print(f"    {'actual':<11} {'—':>5}  {'—':>5}  "
          f"{a['skew'][1]:+.3f}  {a['uw_mean'][1]:+.4f}  {a['uw_cvar1'][1]:+.4f}")
    # 실제에 가장 가까운 모델 (skew·uw_mean·uw_cvar1)
    for k in METRICS_ERR:
        errs = {nm: abs(points[nm][k][0] - points[nm][k][1]) for nm in ORDER if nm in points}
        best = min(errs, key=errs.get)
        es = "  ".join(f"{nm}={errs[nm]:.4f}" for nm in ORDER if nm in points)
        print(f"      |{k}−actual|  {es}   → 가장 가까움: {best}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 104)
    print(f"# §4.3.3 actual 재현도 — full/metab_drop/maskall (device={device}, B={N_BOOT}, seeds={SEEDS})")
    print("#  지표 cov80·cov95·skew·uw_mean·uw_cvar1.  유의성=|model−actual| (Δerr<0 & p<0.05 = full 이 실제에 더 가까움)")
    print("#" * 104)
    rng = np.random.default_rng(2026)

    pool = {nm: dict(ret=[], uw=[], a_ret=[], a_uw=[]) for nm in ORDER}

    for fold in FOLDS:
        print(f"\n=== {LABELS[fold]} [{fold}] " + "=" * 50)
        P = {}
        for nm in ORDER:
            r = pooled_sims(nm, fold, device)
            if r is None:
                print(f"  [skip] {nm} 없음"); continue
            P[nm] = prep(r[0], r[1])
        if "full" not in P:
            print("  [skip fold] full 없음"); continue
        points = {nm: metrics(P[nm], np.arange(P[nm]["n"])) for nm in P}
        print_point_table("point", points)

        # 유의성: full vs maskall (같은 origin), full vs metab_drop (origin 같을 때만)
        for abl in ("maskall", "metab_drop"):
            if abl not in P:
                continue
            if P[abl]["n"] != P["full"]["n"]:
                print(f"  [유의성 full vs {abl}] origin 수 불일치({P['full']['n']} vs {P[abl]['n']}) → 점 비교만")
                continue
            res = paired_err_bootstrap(P["full"], P[abl], rng)
            print(f"  [유의성 full vs {abl}] Δerr=|full−act|−|{abl}−act|  (음수=full 더 가까움)")
            for k in METRICS_ERR:
                rr = res[k]; sig = "*" if rr["p"] < 0.05 else " "
                print(f"    Δerr {k:<9} = {rr['d']:+.4f}  95%CI[{rr['lo']:+.4f},{rr['hi']:+.4f}]  p={rr['p']:.3f} {sig}")

        # pooled 누적
        for nm in P:
            for i in range(P[nm]["n"]):
                pool[nm]["ret"].append(P[nm]["ret"][i])
                pool[nm]["uw"].append(P[nm]["uw"][i])
                pool[nm]["a_ret"].append(P[nm]["a_ret"][i])
                pool[nm]["a_uw"].append(P[nm]["a_uw"][i])

    # ── pooled across folds ──
    print("\n" + "=" * 104)
    print("[POOLED across 4 folds]  (주의: fold 스케일 차이로 극단 fold 가 지배)")
    Pp = {}
    for nm in ORDER:
        if not pool[nm]["ret"]:
            continue
        Pp[nm] = dict(ret=np.stack(pool[nm]["ret"]), uw=np.stack(pool[nm]["uw"]),
                      a_ret=np.stack(pool[nm]["a_ret"]), a_uw=np.stack(pool[nm]["a_uw"]),
                      n=len(pool[nm]["ret"]))
    if "full" in Pp:
        points = {nm: metrics(Pp[nm], np.arange(Pp[nm]["n"])) for nm in Pp}
        print_point_table("pooled point", points)
        for abl in ("maskall", "metab_drop"):
            if abl not in Pp:
                continue
            if Pp[abl]["n"] != Pp["full"]["n"]:
                print(f"  [유의성 pooled full vs {abl}] origin 수 불일치 → 점 비교만")
                continue
            res = paired_err_bootstrap(Pp["full"], Pp[abl], rng)
            print(f"  [유의성 pooled full vs {abl}] Δerr (음수=full 더 가까움)")
            for k in METRICS_ERR:
                rr = res[k]; sig = "*" if rr["p"] < 0.05 else " "
                print(f"    Δerr {k:<9} = {rr['d']:+.4f}  95%CI[{rr['lo']:+.4f},{rr['hi']:+.4f}]  p={rr['p']:.3f} {sig}")
    print("\n[판정] Δerr<0 & p<0.05 = 그 ablation 제거분(경로/유동성)이 실제 꼬리 재현에 유의 기여.")


if __name__ == "__main__":
    main()
