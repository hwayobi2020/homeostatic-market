"""§4.2 반사실 시나리오 — LEVEL(평행이동 9-grid) + PATH-SHAPE(같은 평균, 모양만) 분리.

LOCKED MAC-Flow 본모형: 압축기 MLP(pd64,dr0.2) + flow(fl4/fh128), extra=sp_std_13w+sp_skew_13w.
  체크포인트: garch_flow_ar_rvP2mainMlp_pd64_fl4_fh128_s{seed}_{fold}_best.pt  (재학습 X, inference)
  raw-vol: patch_rawvol() 로 forward σ = origin-frozen (GARCH 동학 없음).

두 실험 (직교):
  [A] LEVEL (평행이동 9-grid): tbill·metab 를 각각 p10/p50/p90 수준으로 13주 flat 고정 → 3×3.
      → 레벨/저금리·유동성 interaction (어제 NF-GARCH 로 본 것을 LOCKED 모델로 재실행).
  [B] PATH-SHAPE: *평균은 z(p50) 고정, 진폭은 z(p90)-z(p10)*, 모양만 7종으로 변형
      (flat·ramp↑·ramp↓·hump∩·trough∪·step↑·step↓).  변수별로 (tbill-shape / metab flat),
      (metab-shape / tbill flat) 분리.  → 같은 평균에서 분포가 달라지면 path-dependence 입증.

평가: skew(좌측−)·CVaR1(깊을수록−)·std + exkurt(발산 감시).  4 fold × 3 seed.
  paired: 한 (fold,seed) 안에서 모든 시나리오 같은 torch.manual_seed → 칸/모양 간 차이 = 순수 효과.
  재진입: (fold,seed) 결과를 json 캐시 → 중단돼도 이어받기.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/analyze_pathshape_rawvol.py
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
    compute_var, compute_cvar, PAST_LEN, FUTURE_LEN,
)

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
CACHE_DIR = os.path.join(RESULT_DIR, "pathshape_test_cache")   # ★ test origin(OOS) 재실행 캐시 (옛 train 캐시 pathshape_full_cache 와 분리 — 재사용 방지)
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
SEEDS = [2026, 2027, 2028]        # seed 3 (5에서 축소)
PCTLS = [10, 50, 90]
PCTL_LABEL = {10: "lo", 50: "mid", 90: "hi"}
N_ORIGIN_MAX = 200
LEVEL_WINDOW_WEEKS = 520    # 시나리오 레벨 분위수 = train 최근 ~10년(520주)만 (전체 1971~ 는 70~80년대 고금리로 high 비현실적)
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


# ── shape 편차: zero-mean, range(max-min)=1 로 정규화 (평균·진폭 통제, 모양만 차이) ──
def _norm(d):
    d = np.asarray(d, float)
    d = d - d.mean()
    rng = d.max() - d.min()
    return d / rng if rng > 1e-12 else d


def build_shapes():
    t = np.arange(FUTURE_LEN, dtype=float)
    mid = (FUTURE_LEN - 1) / 2.0
    ramp_up = _norm(np.linspace(0.0, 1.0, FUTURE_LEN))
    tri = 1.0 - np.abs(t - mid) / mid                 # 0→1→0 (peak mid)
    hump = _norm(tri)
    step = np.where(t < mid, 0.0, 1.0)
    step_up = _norm(step)
    return {
        "flat":      np.zeros(FUTURE_LEN),
        "ramp_up":   ramp_up,
        "ramp_down": -ramp_up,
        "hump":      hump,        # ∩
        "trough":    -hump,       # ∪
        "step_up":   step_up,
        "step_down": -step_up,
    }


SHAPES = build_shapes()
SHAPE_STD = {k: float(np.std(v)) for k, v in SHAPES.items()}   # 경로 분산(혼동 투명화; 쌍↑↓ 동일)
JOINT_DEFS = {
    "easing":     ("ramp_down", "ramp_up"),    # 금리↓ + 유동성↑ (완화/QE)
    "tightening": ("ramp_up",   "ramp_down"),  # 금리↑ + 유동성↓ (긴축)
    "crisis":     ("step_down", "step_up"),    # 급금리↓ + 급유동성↑ (위기대응)
}


def _skew(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def _exkurt(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 4) - 3.0)


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


@torch.no_grad()
def rollout_path(model, Xte, extra, last_sp, tbill_path, metab_path, device):
    """미래 tbill·metab 를 13주 *경로 벡터*(z)로 주입해 ar_sample.

    tbill_path / metab_path : (FUTURE_LEN,) z 배열.  seed 는 호출 측 고정(paired).
    """
    n = Xte.shape[0]
    tb = torch.tensor(np.asarray(tbill_path, float), device=device, dtype=Xte.dtype)
    mb = torch.tensor(np.asarray(metab_path, float), device=device, dtype=Xte.dtype)
    out = []
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK); k = e - s
        x_past = Xte[s:e, :PAST_LEN, :]
        lsp = last_sp[s:e]
        ex = extra[s:e] if extra is not None else None
        fut_tbill = tb.unsqueeze(0).expand(k, FUTURE_LEN).contiguous()
        fmac = mb.reshape(1, FUTURE_LEN, 1).expand(k, FUTURE_LEN, 1).contiguous()
        sim = model.ar_sample(x_past, fut_tbill, lsp, N_SIM,
                              extra_context=ex, future_macro_z=fmac)
        out.append(sim.cpu())
    return torch.cat(out, dim=0).numpy()


def load_fold_seed(fold, seed, device):
    bp = os.path.join(RESULT_DIR, f"garch_flow_ar_{TAG_PREFIX}_s{seed}_{fold}_best.pt")
    if not os.path.exists(bp):
        print(f"[missing best.pt] {os.path.basename(bp)}")
        return None
    ckpt = torch.load(bp, map_location=device)
    meta = ckpt["meta"]
    cond_stats = meta["cond_stats"]; target_stats = meta["target_stats"]
    extra_stats = meta.get("extra_stats")
    model = rebuild_model(ckpt, device)

    gp = garch_preprocess_fold(FOLDS_DIR, fold, RESULT_DIR)
    origin_csv = gp["test"]    # ★ 반사실 시나리오 origin = test(OOS). train 은 in-sample(모델 학습 구간) → 무효.
                               #   (시나리오 레벨 percentile 만 train 분포 사용: 아래 df_tr — 그건 origin 아님)
    Xte, Yte, _, _, _ = cached_load_windows_seq(
        origin_csv, cond_stats=cond_stats, target_stats=target_stats)
    Xte_dev = Xte.to(device)
    Xte_extra, _, n_ex, _ = load_extra_context(
        origin_csv, DC_COLS_LIST, extra_stats=extra_stats)
    if n_ex != Xte.shape[0]:
        sys.exit(f"[FATAL] extra valid {n_ex} != main {Xte.shape[0]} ({fold},{seed})")
    Xte_extra_dev = Xte_extra.to(device)

    valid_mask, z_te, df_te = compute_valid_mask(origin_csv, cond_stats)
    n_w = z_te.shape[0] - PAST_LEN - FUTURE_LEN + 1
    last_full = z_te[PAST_LEN - 1: PAST_LEN - 1 + n_w, T.SP_CH]
    last_sp_dev = torch.from_numpy(last_full[valid_mask].astype(np.float32)).to(device)

    # 시나리오 수준 = *절대 정책스탠스* 레벨 (분위수 아님 — 분위는 fold마다 드리프트/Volcker 왜곡).
    #   tbill = 자연이자율(r*) 기준: low 0.25% / mid 2.5%(명목 중립) / high 5%(긴축).  연율 → 주간 tbill_wr → z.
    #   metab = 0(중립=초과유동성 없음) 기준: low −2% / mid 0% / high +2.5% (13주 %).  raw → z.
    #   30/70 분위(debasement sweep)는 앵커 사이 선형보간.  표준화 cmu/csd = 모델 학습 전체 train 통계 유지(불변).
    cmu = np.asarray(cond_stats["mean"], float); csd = np.asarray(cond_stats["std"], float)
    ti = ENC_COLS.index("tbill_wr"); mi = ENC_COLS.index("metab_13w")

    def _tbill_wr(annual_pct):
        return (1.0 + annual_pct / 100.0) ** (1.0 / 52.0) - 1.0

    def _interp(p, lo, mid, hi):     # 앵커 10→lo, 50→mid, 90→hi 선형보간
        return lo + (mid - lo) * (p - 10) / 40.0 if p <= 50 else mid + (hi - mid) * (p - 50) / 40.0

    tb_z = {p: (_tbill_wr(_interp(p, 0.25, 2.5, 5.0)) - cmu[ti]) / csd[ti] for p in PCTLS}
    mb_z = {p: (_interp(p, -0.02, 0.0, 0.025) - cmu[mi]) / csd[mi] for p in PCTLS}

    # origin별 *실현* 미래 13주 거시 z 경로 (model-on-realized / sensitivity 용). Xte 와 동일 정렬.
    _fut_tb = np.stack([z_te[w + PAST_LEN: w + PAST_LEN + FUTURE_LEN, ti]
                        for w in range(n_w)])[valid_mask]
    _fut_mb = np.stack([z_te[w + PAST_LEN: w + PAST_LEN + FUTURE_LEN, mi]
                        for w in range(n_w)])[valid_mask]

    # raw rescale 재료 (raw-vol: omega/alpha/beta=0 → origin-frozen σ)
    tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
    _gsig = df_te["garch_sigma"].to_numpy(float); _gz = df_te["sp_return"].to_numpy(float)
    _oidx = np.where(valid_mask)[0]; n_orig = len(_oidx)
    _om = float(df_te["garch_omega"].iloc[0]); _al = float(df_te["garch_alpha"].iloc[0])
    _be = float(df_te["garch_beta"].iloc[0]); _mu = float(df_te["garch_mu"].iloc[0])
    _orow = _oidx + (PAST_LEN - 1)
    _s2 = _gsig[_orow] ** 2
    _e2 = (_gz[_orow] * _gsig[_orow]) ** 2

    if n_orig > N_ORIGIN_MAX:
        idx = np.linspace(0, n_orig - 1, N_ORIGIN_MAX).astype(int)
        Xte_dev = Xte_dev[idx]; Xte_extra_dev = Xte_extra_dev[idx]
        last_sp_dev = last_sp_dev[idx]; _s2 = _s2[idx]; _e2 = _e2[idx]
        _fut_tb = _fut_tb[idx]; _fut_mb = _fut_mb[idx]

    rescale = dict(tmu=tmu, tsd=tsd, s2=_s2, e2=_e2, om=_om, al=_al, be=_be, mu=_mu)
    real_tb_mean = _fut_tb.mean(axis=1) * csd[ti] + cmu[ti]   # raw 연주간금리 (binning 용, per origin)
    real_mb_mean = _fut_mb.mean(axis=1) * csd[mi] + cmu[mi]   # raw metab
    return dict(model=model, Xte=Xte_dev, extra=Xte_extra_dev, last_sp=last_sp_dev,
                rescale=rescale, tb_z=tb_z, mb_z=mb_z,
                real_tb_fut=_fut_tb, real_mb_fut=_fut_mb,       # (n_orig, FUTURE_LEN) z 경로
                real_tb_mean=real_tb_mean, real_mb_mean=real_mb_mean,
                csd=csd, cmu=cmu, ti=ti, mi=mi)


def sim_metrics(sim_z, rescale):
    r = rescale
    sim_raw = forward_garch_rescale(sim_z * r["tsd"] + r["tmu"], r["s2"], r["e2"],
                                    r["om"], r["al"], r["be"], r["mu"])   # (n_orig, n_sim, T)
    f = sim_raw.ravel()
    # 경로(순서 반영) 지표: 누적 로그수익(진입점=0 기준) → 보유기간 중 최악 underwater
    # (= 최저 누적점, vs 진입).  peak-to-trough 아님 — "이익 토해냄"은 위험으로 안 셈.
    cum = np.cumsum(sim_raw, axis=2)                                       # (n_orig, n_sim, T)
    # 진입점(누적=0)을 시점0으로 prepend → UW=진입대비, MDD=초기peak 기준으로 정확.
    z0 = np.zeros((cum.shape[0], cum.shape[1], 1), dtype=cum.dtype)
    cum0 = np.concatenate([z0, cum], axis=2)                              # (n_orig, n_sim, T+1)
    underw = cum0.min(axis=2).ravel()                                     # intra-horizon loss (진입 대비, ≤0) ★주
    mdd = (cum0 - np.maximum.accumulate(cum0, axis=2)).min(axis=2).ravel()  # peak-to-trough MDD (보조)
    term = cum[:, :, -1].ravel()                                          # terminal 누적수익 (끝점)
    return dict(std=float(f.std(ddof=1)), skew=_skew(f), exkurt=_exkurt(f),
                cvar5=compute_cvar(f, 0.05), cvar1=compute_cvar(f, 0.01),
                uw_mean=float(underw.mean()), uw_cvar1=compute_cvar(underw, 0.01),
                mdd_mean=float(mdd.mean()), mdd_cvar1=compute_cvar(mdd, 0.01),
                term_mean=float(term.mean()), term_cvar1=compute_cvar(term, 0.01))


def run_fold_seed(fold, seed, device):
    """한 (fold,seed): 9-grid(level) + shape(7×2) 전 시나리오 metric dict 반환."""
    ctx = load_fold_seed(fold, seed, device)
    if ctx is None:
        return None
    res = {"level": {}, "shape_tbill": {}, "shape_metab": {}, "joint": {}}

    def _run(tbill_path, metab_path):
        torch.manual_seed(seed)        # paired
        sim = rollout_path(ctx["model"], ctx["Xte"], ctx["extra"], ctx["last_sp"],
                           tbill_path, metab_path, device)
        return sim_metrics(sim, ctx["rescale"])

    # [A] LEVEL 9-grid (flat)
    for pt in PCTLS:
        for pm in PCTLS:
            key = f"{PCTL_LABEL[pt]}_{PCTL_LABEL[pm]}"
            res["level"][key] = _run(np.full(FUTURE_LEN, ctx["tb_z"][pt]),
                                     np.full(FUTURE_LEN, ctx["mb_z"][pm]))

    # [B] SHAPE — 평균=z(p50) 고정, 진폭=z(p90)-z(p10), 모양만
    tb_c = ctx["tb_z"][50]; tb_rng = ctx["tb_z"][90] - ctx["tb_z"][10]
    mb_c = ctx["mb_z"][50]; mb_rng = ctx["mb_z"][90] - ctx["mb_z"][10]
    mb_flat = np.full(FUTURE_LEN, mb_c)
    tb_flat = np.full(FUTURE_LEN, tb_c)
    for name, dev in SHAPES.items():
        res["shape_tbill"][name] = _run(tb_c + dev * tb_rng, mb_flat)   # tbill 모양, metab flat
        res["shape_metab"][name] = _run(tb_flat, mb_c + dev * mb_rng)   # metab 모양, tbill flat

    # [C] JOINT — tbill·metab 동시 경로 (정책 시나리오)
    for name, (ts, ms) in JOINT_DEFS.items():
        res["joint"][name] = _run(tb_c + SHAPES[ts] * tb_rng,
                                  mb_c + SHAPES[ms] * mb_rng)
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print(f"# §4.2 LEVEL(9-grid) + PATH-SHAPE — LOCKED {TAG_PREFIX}, device={device}, n_sim={N_SIM}")
    print(f"# shape 평균=z(p50) 고정 · 진폭=z(p90-p10) · zero-mean 편차 (레벨과 직교)")
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
            print(f"[done] {fold} s{seed}")

    # ── 집계: seed 평균±std ──
    print("\n" + "=" * 100)
    print("[집계] 4 fold × 5 seed 평균±std")
    _summarize()


def _agg(vals):
    a = np.asarray([v for v in vals if v is not None], float)
    return (float(a.mean()), float(a.std(ddof=0)), len(a)) if len(a) else (None, None, 0)


def _summarize():
    metr = ["skew", "cvar1", "std", "exkurt"]
    for fold in FOLDS:
        caches = [os.path.join(CACHE_DIR, f"{fold}_s{s}.json") for s in SEEDS]
        loaded = [json.load(open(c)) for c in caches if os.path.exists(c)]
        if not loaded:
            print(f"\n=== {fold}: (결과 없음)"); continue
        print(f"\n=== {fold}  ({len(loaded)} seed) " + "=" * 60)

        # LEVEL 9-grid
        print("  [LEVEL 9-grid]  tbill×metab  skew / UWcvar1 / MDDcvar1 (seed mean)")
        for pt in PCTLS:
            row = []
            for pm in PCTLS:
                key = f"{PCTL_LABEL[pt]}_{PCTL_LABEL[pm]}"
                sk, _, _ = _agg([d["level"][key]["skew"] for d in loaded])
                uw, _, _ = _agg([d["level"][key]["uw_cvar1"] for d in loaded])
                md, _, _ = _agg([d["level"][key]["mdd_cvar1"] for d in loaded])
                row.append(f"{PCTL_LABEL[pm]}:{sk:+.2f}/{uw:+.4f}/{md:+.4f}")
            print(f"    tbill={PCTL_LABEL[pt]:>3}  " + "   ".join(row))

        # SHAPE (평균 고정) — skew/cvar1/std mean±std, shape 간 차이 vs seed 노이즈
        for var, label in [("shape_tbill", "tbill-shape (metab flat)"),
                           ("shape_metab", "metab-shape (tbill flat)")]:
            print(f"  [SHAPE: {label}]  (평균=p50 / UW=intra-horizon(진입대비,주) · MDD=peak-to-trough(보조) · ↑↓=순서검증)")
            print(f"    {'shape':>10} {'pathσ':>6} {'skew':>8} {'UWmean(m±sd)':>18} {'UWcvar1(m±sd)':>18} {'MDDcvar1(m±sd)':>18} {'termCV1':>9}")
            for name in SHAPES:
                sk = _agg([d[var][name]["skew"] for d in loaded])
                um = _agg([d[var][name]["uw_mean"] for d in loaded])
                uc = _agg([d[var][name]["uw_cvar1"] for d in loaded])
                mc = _agg([d[var][name]["mdd_cvar1"] for d in loaded])
                tc = _agg([d[var][name]["term_cvar1"] for d in loaded])
                print(f"    {name:>10} {SHAPE_STD[name]:>6.3f} {sk[0]:>+8.3f} "
                      f"{um[0]:+.5f}±{um[1]:.5f} {uc[0]:+.5f}±{uc[1]:.5f} {mc[0]:+.5f}±{mc[1]:.5f} {tc[0]:>+9.5f}")

        # JOINT (tbill·metab 동시 경로)
        print(f"  [JOINT: tbill·metab 동시 경로]  (UW=intra-horizon 주 · MDD=peak-to-trough 보조)")
        print(f"    {'scenario':>12} {'skew':>8} {'UWmean(m±sd)':>18} {'UWcvar1(m±sd)':>18} {'MDDcvar1(m±sd)':>18} {'termCV1':>9}")
        for name in JOINT_DEFS:
            sk = _agg([d["joint"][name]["skew"] for d in loaded])
            um = _agg([d["joint"][name]["uw_mean"] for d in loaded])
            uc = _agg([d["joint"][name]["uw_cvar1"] for d in loaded])
            mc = _agg([d["joint"][name]["mdd_cvar1"] for d in loaded])
            tc = _agg([d["joint"][name]["term_cvar1"] for d in loaded])
            print(f"    {name:>12} {sk[0]:>+8.3f} "
                  f"{um[0]:+.5f}±{um[1]:.5f} {uc[0]:+.5f}±{uc[1]:.5f} {mc[0]:+.5f}±{mc[1]:.5f} {tc[0]:>+9.5f}")
    print("\n[판정] *순서/경로 의존*은 순서 반영 지표(underwater=보유 중 최악 누적손실 vs 진입)로 본다.")
    print("       (skew/cvar/term 은 순서 무감각.  peak-to-trough 아님 — 이익 토해냄은 위험 안 셈.)")
    print("       SHAPE 표에서 ramp_up vs ramp_down (또는 step_up/down)의 UWmean·UWcvar1 이")
    print("       다르고 fold 일관 → 주가 경로/순서 의존 입증.")
    print("       term(끝점)은 비슷한데 UW 만 다르면 → 순수 경로위험 (100→110→100 vs 100→90→100).")


if __name__ == "__main__":
    main()
