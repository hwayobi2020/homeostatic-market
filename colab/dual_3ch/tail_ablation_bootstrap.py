"""§4.3.3 — full / metab_drop / maskall 이 *실제(actual)* 꼬리를 얼마나 잘 재현하나.

지표: cov80, cov95, skew, IHL mean(uw_mean), IHL CVaR1%(uw_cvar1).
기준: "실제값에 가까운가" = |model − actual|.

★ 집계: seed pool(혼합) 금지 — seed = 각각 따로 학습된 다른 모델이라 합치면 혼합분포가
  되어 skew·꼬리(고차모멘트)가 모델간 차이로 왜곡(한 outlier seed가 과대포장).
  → **seed별로 계산 후 평균(+seed std)** = 표 4.9/4.12 와 동일 잣대.

IHL(uw_mean/uw_cvar1)은 어느 표에도 저장된 적 없음 → 체크포인트로 *추론만(학습 0)* 재샘플해 계산.

config (run_ablations_rawvol.py 와 동일 spec):
  full       : cols 5(+metab), MASK_FUTURE_TBILL=False, unmask=[metab_13w]  tag=rvP2mainMlp_pd64_fl4_fh128
  maskall    : cols 5,          MASK_FUTURE_TBILL=True,  unmask=[]           tag=rvAbl_maskall
  metab_drop : cols 4(-metab),  MASK_FUTURE_TBILL=False, unmask=[]           tag=rvAbl_metab_drop

출력: fold별 (a) seed별 skew·uw_cvar1 진단, (b) seed-평균 표 + actual, (c) 실제에 가장 가까운 모델
      + full vs ablation per-seed 승수(몇/3 seed에서 full이 더 가까운가).

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/tail_ablation_bootstrap.py
"""
import os
import sys

import json
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
import fpath_model                                                    # noqa: E402
# full_fpath 로드용 미래요약 차원 — *학습 때와 동일해야* state_dict 일치.
fpath_model.FUTURE_SUMMARY_DIM = int(os.environ.get("FPATH_DIM", "16"))
fpath_model.FUTURE_SUMMARY_HIDDEN = int(os.environ.get("FPATH_HIDDEN", "16"))

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
LABELS = {"F_gfc": "금융위기", "F_long_A": "회복기", "F_long_B_origin": "코로나", "F_long": "긴축기"}
SEEDS = [2026, 2027, 2028, 2029, 2030]
N_SIM = 1000
CHUNK = 8
DC_COLS = ["sp_std_13w", "sp_skew_13w"]
LK = dict(d_model=128, mlp_layers=4, flow_layers=4, flow_hidden=128, pd=64, dropout=0.2)
ORDER = ["full", "full_fpath", "metab_drop", "maskall"]
ERRK = ["uw_mean", "uw_cvar1", "uw_cvar5", "uw_cvar10"]    # actual 기준 오차 비교 (1%는 참고: 실측~1점)
CACHE_DIR = os.path.join(RESULT_DIR, "tail_ablation_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

FULL_TAG = "rvP2mainMlp_pd64_fl4_fh128"
CONFIGS = {
    "full":       (["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"], False, ["metab_13w"],
                   "garch_flow_ar_%s_s{seed}_{fold}_best.pt" % FULL_TAG),
    "full_fpath": (["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"], False, ["metab_13w"],
                   "garch_flow_ar_rvAbl_full_fpath_s{seed}_{fold}_best.pt"),
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


def rebuild(ckpt, d_input, device, cls=MambaFlowAR):
    # cls=MambaFlowARFpath 면 미래경로 요약 포함(state_dict 에 future_encoder/확장 flow 존재).
    m = cls(
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
    _cls = fpath_model.MambaFlowARFpath if name == "full_fpath" else MambaFlowAR
    model = rebuild(ckpt, len(cols), device, cls=_cls)

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

    sp_raw_full = _gz * _gsig + _gmu
    actual = np.stack([sp_raw_full[w + PAST_LEN: w + PAST_LEN + FUTURE_LEN]
                       for w in range(n_w)])[valid_mask]
    return sim_raw, actual


def underwater(sim):
    cum = np.cumsum(sim, axis=2)
    z0 = np.zeros((cum.shape[0], cum.shape[1], 1), dtype=cum.dtype)
    return np.concatenate([z0, cum], axis=2).min(axis=2)


def _skew(a):
    m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def metrics_seed(sim_raw, actual):
    """한 seed: origin 풀 위 model 지표 + 해당 fold actual 지표."""
    ret = sim_raw.reshape(-1)
    uw = underwater(sim_raw).ravel()
    aret = actual.ravel()
    auw = underwater(actual[:, None, :]).ravel()
    lo80, hi80 = np.percentile(ret, [10, 90]); lo95, hi95 = np.percentile(ret, [2.5, 97.5])
    return dict(
        cov80=float(((aret >= lo80) & (aret <= hi80)).mean()),
        cov95=float(((aret >= lo95) & (aret <= hi95)).mean()),
        skew=_skew(ret), uw_mean=float(uw.mean()),
        uw_cvar1=compute_cvar(uw, 0.01), uw_cvar5=compute_cvar(uw, 0.05), uw_cvar10=compute_cvar(uw, 0.10),
        a_skew=_skew(aret), a_uw_mean=float(auw.mean()),
        a_uw_cvar1=compute_cvar(auw, 0.01), a_uw_cvar5=compute_cvar(auw, 0.05), a_uw_cvar10=compute_cvar(auw, 0.10),
    )


def get_metrics(nm, fold, seed, device):
    """캐시된 per-seed metrics (없으면 추론 후 캐시) — 지표 변경 시 재샘플 회피."""
    cp = os.path.join(CACHE_DIR, f"{nm}_{fold}_s{seed}.json")
    if os.path.exists(cp):
        return json.load(open(cp))
    r = sample_config(nm, fold, seed, device)
    if r is None:
        return None
    m = metrics_seed(r[0], r[1])
    json.dump(m, open(cp, "w"))
    return m


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 104)
    print(f"# §4.3.3 actual 재현도 — seed별 계산 후 평균 (pool 금지).  device={device}, seeds={SEEDS}, N_SIM={N_SIM}")
    print("#  지표 uw_mean·IHL1%·IHL5%·IHL10% (+cov,skew).  비교 |model−actual|.  ※IHL1%는 실측~1점이라 참고용, IHL10%(실측~18점)가 robust")
    print("#" * 104)

    for fold in FOLDS:
        print(f"\n=== {LABELS[fold]} [{fold}] " + "=" * 50)
        sr = {nm: [] for nm in ORDER}
        for nm in ORDER:
            for seed in SEEDS:
                m = get_metrics(nm, fold, seed, device)
                if m is not None:
                    sr[nm].append(m)
        if not sr["full"]:
            print("  [skip] full 없음"); continue

        # (a) seed별 진단 (skew, IHL10%)
        print("  [seed별]  IHL1% / IHL10%")
        for nm in ORDER:
            if not sr[nm]:
                continue
            u1 = ", ".join(f"{m['uw_cvar1']:+.3f}" for m in sr[nm])
            uc = ", ".join(f"{m['uw_cvar10']:+.3f}" for m in sr[nm])
            print(f"    {nm:<11} ihl1[{u1}]  ihl10[{uc}]")

        # (b) seed-평균 + actual
        keys = ["cov80", "cov95", "skew", "uw_mean", "uw_cvar1", "uw_cvar5", "uw_cvar10"]
        avg = {nm: {k: float(np.mean([m[k] for m in sr[nm]])) for k in keys} for nm in ORDER if sr[nm]}
        sd = {nm: {k: float(np.std([m[k] for m in sr[nm]])) for k in keys} for nm in ORDER if sr[nm]}
        a = sr["full"][0]
        act = dict(skew=a["a_skew"], uw_mean=a["a_uw_mean"],
                   uw_cvar1=a["a_uw_cvar1"], uw_cvar5=a["a_uw_cvar5"], uw_cvar10=a["a_uw_cvar10"])
        print("\n  [seed-평균]  cov80 cov95  skew   uw_mean   IHL1%    IHL5%    IHL10%")
        for nm in ORDER:
            if nm not in avg:
                continue
            print(f"    {nm:<11} {avg[nm]['cov80']:.3f} {avg[nm]['cov95']:.3f}  "
                  f"{avg[nm]['skew']:+.2f}  {avg[nm]['uw_mean']:+.4f}  "
                  f"{avg[nm]['uw_cvar1']:+.4f}  {avg[nm]['uw_cvar5']:+.4f}  {avg[nm]['uw_cvar10']:+.4f}")
        print(f"    {'actual':<11} {'—':>5} {'—':>5}  {act['skew']:+.2f}  {act['uw_mean']:+.4f}  "
              f"{act['uw_cvar1']:+.4f}  {act['uw_cvar5']:+.4f}  {act['uw_cvar10']:+.4f}")

        # (c) 실제에 가장 가까운 모델 (seed-평균 기준) + per-seed 승수
        print("  [실제 재현도]  |seed평균 − actual|  → 가장 가까움 / full vs ablation per-seed 승수")
        for k in ERRK:
            errs = {nm: abs(avg[nm][k] - act[k]) for nm in ORDER if nm in avg}
            best = min(errs, key=errs.get)
            es = "  ".join(f"{nm}={errs[nm]:.4f}" for nm in ORDER if nm in avg)
            wins = {}
            for abl in ("maskall", "metab_drop"):
                if not sr.get(abl):
                    continue
                n = min(len(sr["full"]), len(sr[abl]))
                ak = {"uw_mean": "a_uw_mean", "uw_cvar1": "a_uw_cvar1",
                      "uw_cvar5": "a_uw_cvar5", "uw_cvar10": "a_uw_cvar10"}[k]
                w = sum(1 for i in range(n)
                        if abs(sr["full"][i][k] - sr["full"][i][ak]) < abs(sr[abl][i][k] - sr[abl][i][ak]))
                wins[abl] = f"{w}/{n}"
            wtxt = "  ".join(f"full>{abl}:{wins[abl]}" for abl in wins)
            print(f"    {k:<9} {es}   → {best}   [{wtxt}]")

    print("\n[판정] full의 |오차|가 가장 작고 per-seed 승수도 높으면 = 경로/유동성이 실제 꼬리 재현에 기여.")
    print("       maskall이 더 작으면 = 그 fold에선 경로 조건화가 실제 재현을 개선 못 함(정직 보고).")


if __name__ == "__main__":
    main()
