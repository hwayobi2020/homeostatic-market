"""비중첩 원점 · 미래 정보 없음 조건에서 네 모델 13주 지표 비교 (한 번 실행 + 요약).

리뷰어1 #2
----------
  "Origins appear weekly, horizon is 13 weeks - 13x overlap.  Coverage rates are
   not independent?  Hence, the effective sample is far smaller than nominal.
   Therefore, use thin to non-overlapping windows as a check, ..."

원점이 주 단위라 이웃 원점의 13주 예측 구간이 최대 12주를 공유한다.  원점을
STRIDE(=13) 간격으로 솎으면 예측 구간이 겹치지 않는다.  지평은 13주 그대로
두므로 모델 설정을 바꾸지 않는다.

미래 조건
---------
조건부 생성 모형끼리는 **같은 조건을 주고 구조 차이만 남긴다**.
  · MAC-Flow   : 실현 거시 경로(tbill·metab) 주입 — 전역 요약 + 스텝별 재주입
  · CondVAE/GAN: 원래 세팅 그대로, 미래 tbill 실현 경로를 맥락 벡터로 받음
조건부 모델에서 조건을 빼면 비교가 성립하지 않으므로 flat 으로 막지 않는다.
GARCH-ST 는 구조상 미래 경로를 못 받아 이 비교에서 제외한다
(별도 실행 결과로 이미 확보돼 있고 조건이 바뀌지 않았다).

구간은 네 모델 모두 전역 풀링(train_garch_flow.evaluate_test 와 동일 정의).

한계: 예측 구간 겹침은 사라지지만 과거 52주 입력창은 원점 간 39주가 겹친다.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/run_thin_compare.py
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 논문 본모형(fpath_novol d2) 을 쓰도록 PS import 전에 지정한다.
#   기본값 PS_BASE="full" 이면 TAG_PREFIX 가 rvP2mainMlp_* 가 되어 다른 모델을 읽는다.
os.environ.setdefault("PS_BASE", "fpath_novol")
os.environ.setdefault("FPATH_DIM", "2")

# 베이스라인 조건 입력을 MAC-Flow 와 일치시킨다 (VG import 전에 설정해야 한다).
#   과거 52 주 5 채널(논문 Table 3) + 미래 tbill·metab + 원점 고정 sp_skew_13w.
#   예전에는 train_flow_seq.COND_COLS(6채널, metab 없이 변동성 2채널)를 썼는데
#   그러면 MAC-Flow 만 논문의 핵심 조건 변수(metab)를 받아 비교가 성립하지 않는다.
os.environ.setdefault("VG_EXTRA_COLS", "sp_skew_13w")
os.environ.setdefault("VG_FUTURE_UNMASK", "metab_13w")

from rawvol_helpers import (patch_rawvol, rawstd_preprocess_fold,        # noqa: E402
                            forward_rawvol_rescale, ihl_metrics)
patch_rawvol()                                   # MAC-Flow 와 동일 파이프라인

import analyze_pathshape_rawvol as PS                                    # noqa: E402
import pathshape_zeromean_anchored_rawvol as ZM                          # noqa: E402
import train_garch_flow as T                                             # noqa: E402
import train_garch_xpast as GX                                           # noqa: E402
import train_vae_gan_baseline as VG                                      # noqa: E402

VG.garch_preprocess_fold = rawstd_preprocess_fold   # import-bound 이름 교체
VG.forward_garch_rescale = forward_rawvol_rescale
# VAE/GAN 은 원래 세팅(else 분기) 그대로 — 미래 tbill 실현 경로를 받는다.
# 조건부 모델에서 조건을 빼면 비교 자체가 성립하지 않으므로 flat 으로 막지 않는다.

STRIDE = 13
FOLDS = PS.FOLDS
SEEDS = PS.SEEDS
N_SIM = PS.N_SIM
DM_LAG = 12             # 13주 지평 → 최대 12주 겹침

# 튜닝 결과 (27 조합 x 4 폴드 val CRPS 최소).  run_tuning_all 과 같은 값이어야
# 같은 태그의 결과를 재사용한다.
VG_SPEC = {"vae": dict(lr=1e-4, dctx=128, hid=192),
           "gan": dict(lr=3e-4, dctx=192, hid=128)}
# 세 모델 모두 같은 시드 집합을 쓴다.  한쪽만 여러 시드를 평균하면 그쪽 학습
# 잡음이 √k 배 줄어 차이 d 의 분산이 과소평가되고 유의성이 과장된다.
# MAC-Flow 체크포인트 태그.  기본은 게재판(PS.TAG_PREFIX).  다른 설정을
# 비교해 보려면 FLOW_TAG 로 바꾸고 OUT_SUFFIX 로 결과 파일을 분리한다
# (게재판 결과를 덮지 않는다).
#   예) 8층/hidden 32 를 따로 저장:
#     FLOW_TAG=rvAbl_full_fpath_novol_lr0d0001_fh32_fl8_d2 OUT_SUFFIX=_fl8fh32
FLOW_TAG = os.environ.get("FLOW_TAG", PS.TAG_PREFIX)
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")
# macflow_arrays 는 PS.load_fold_seed 로 모델을 만든다.  거기서 읽는 것은
# PS.TAG_PREFIX 이므로 여기서 같이 덮지 않으면 FLOW_TAG 를 바꿔도 게재판 모델이
# 그대로 쓰인다 (정규화 통계만 바뀌는데 그건 폴드 train 에서 나와 동일하다).
PS.TAG_PREFIX = FLOW_TAG

# 베이스라인 시뮬 배열 캐시.  MAC-Flow 설정만 바꿔 비교할 때 VAE/GAN 40 회를
# 매번 다시 학습하지 않게 한다.  시드가 고정이라 재학습해도 결과가 같다.
# 셀당 sim (원점 x 1000 x 13) float32 ≈ 9MB, 40 셀 ≈ 380MB.
# VG_CACHE=0 으로 끈다.
VG_CACHE = os.environ.get("VG_CACHE", "1") == "1"
VG_CACHE_DIR = os.path.join(PS.RESULT_DIR, "thin_vg_cache")
if VG_CACHE:
    os.makedirs(VG_CACHE_DIR, exist_ok=True)
OUT = os.path.join(PS.RESULT_DIR,
                   f"dm_compare{PS.CACHE_SUFFIX}{OUT_SUFFIX}.json")


def metrics(sim, act):
    """전역 풀링 구간 기준 13주 지표.  sim (n, n_sim, T), act (n, T)."""
    af = np.asarray(act, float).ravel()
    sf = np.asarray(sim, float).ravel()
    cov = {}
    for lvl, lo, hi in [(50, 25.0, 75.0), (80, 10.0, 90.0), (95, 2.5, 97.5)]:
        L, H = np.percentile(sf, lo), np.percentile(sf, hi)
        cov[lvl] = float(((af >= L) & (af <= H)).mean())

    def _sk(x):
        x = np.asarray(x, float)
        return float(np.mean(((x - x.mean()) / (x.std() + 1e-12)) ** 3))

    def _ek(x):
        x = np.asarray(x, float)
        return float(np.mean(((x - x.mean()) / (x.std() + 1e-12)) ** 4) - 3.0)

    from train_garch_x import cvar as _cv, var_q as _vq
    return dict(n_origin=int(np.shape(act)[0]), n_obs=int(af.size),
                crps=float(T.crps_pooled(sim, act)[0]),
                crps_per_origin=crps_per_origin(sim, act),
                cov50=cov[50], cov80=cov[80], cov95=cov[95],
                skew_actual=_sk(af), skew_sim=_sk(sf),
                exkurt_actual=_ek(af), exkurt_sim=_ek(sf),
                std_actual=float(af.std(ddof=1)), std_sim=float(sf.std(ddof=1)),
                # 꼬리위험 축 — 논문 주제이고 §4.2·§4.3 이 쓰는 지표다.
                cvar1_actual=_cv(af, .01), cvar1_sim=_cv(sf, .01),
                cvar5_actual=_cv(af, .05), cvar5_sim=_cv(sf, .05),
                var1_actual=_vq(af, .01), var1_sim=_vq(sf, .01),
                **ihl_metrics(sim, act))


def crps_per_origin(sim, act):
    """원점별 CRPS (13주 평균).  모델 간 짝지은 검정의 표본 단위가 된다.

    crps_pooled 는 원점×시점 전부를 평균해 스칼라 하나만 낸다.  그러면 폴드당
    값이 1 개라 4 폴드 = 표본 4 개가 되어 검정력이 없다.  원점별로 남기면
    폴드당 14 개, 4 폴드 합쳐 56 개 짝이 생긴다.
    """
    sim = np.asarray(sim); act = np.asarray(act)
    out = []
    for i in range(sim.shape[0]):
        out.append(float(np.mean([T.crps_ensemble_sample(sim[i, :, w], act[i, w])
                                  for w in range(sim.shape[2])])))
    return out


def thin_all_offsets(sim, act):
    """STRIDE 간격 부분표본을 오프셋 0..STRIDE-1 전부 만들어 지표를 평균한다.

    오프셋 0 만 쓰면 원점 14 개짜리 부분표본 하나로 판정하게 되어 어느 14 개를
    골랐느냐에 결과가 좌우된다.  13 개 오프셋을 모두 돌려 평균하면 그 임의성이
    사라지고, 각 부분표본 안에서는 예측 구간이 겹치지 않는다는 성질도 유지된다.
    """
    n = np.shape(act)[0]
    per = []
    for off in range(STRIDE):
        idx = np.arange(off, n, STRIDE)
        if idx.size < 2:
            continue
        per.append(metrics(np.asarray(sim)[idx], np.asarray(act)[idx]))
    keys = [k for k in per[0] if k not in ("n_origin", "n_obs", "crps_per_origin")]
    m = {k: float(np.mean([r[k] for r in per])) for k in keys}
    # 원점별 CRPS 는 평균하지 않고 전 오프셋을 이어붙인다 (= 폴드의 전 원점).
    m["crps_per_origin"] = [v for r in per for v in r["crps_per_origin"]]
    m["n_origin"] = float(np.mean([r["n_origin"] for r in per]))
    m["n_obs"] = float(np.mean([r["n_obs"] for r in per]))
    m["n_offset"] = len(per)
    return m


def align(models):
    """모델별 pred_start(예측 첫 주의 test CSV 행)로 교집합을 잡아 행을 맞춘다.

    GARCH-ST 는 train_garch_xpast.py:197-199 규칙, 나머지 셋은
    train_garch_flow.py:496-498 규칙으로 원점을 고른다.  세 모델은 같은 규칙이지만
    안전을 위해 교집합을 잡은 뒤 같은 주에 대해서만 비교한다.
    """
    common = None
    for d in models.values():
        s = set(int(x) for x in d["pred_start"])
        common = s if common is None else (common & s)
    common = np.array(sorted(common), dtype=int)
    out = {}
    for name, d in models.items():
        ps = np.asarray(d["pred_start"], dtype=int)
        pos = {int(v): i for i, v in enumerate(ps)}
        sel = np.array([pos[v] for v in common], dtype=int)
        out[name] = dict(sim=np.asarray(d["sim"])[sel], act=np.asarray(d["act"])[sel])
    return out, common


def _keep_summary(path):
    """run_fold 가 덮어쓰지 않도록 기존 summary 를 잠시 치운다."""
    bak = path + ".thinbak"
    if os.path.exists(path):
        os.replace(path, bak)
        return bak
    return None


def _restore_summary(path, bak):
    if bak and os.path.exists(bak):
        os.replace(bak, path)


def macflow_arrays(fold, seed, device):
    """저장된 ckpt 로 재추론.  미래 조건 = 실현된 거시 경로 (VAE/GAN 과 동일 정보)."""
    ctx = PS.load_fold_seed(fold, seed, device)
    if ctx is None:
        return None

    ckpt = torch.load(os.path.join(
        PS.RESULT_DIR, f"garch_flow_ar_{FLOW_TAG}_s{seed}_{fold}_best.pt"),
        map_location="cpu")
    meta = ckpt["meta"]
    cond_stats, target_stats = meta["cond_stats"], meta["target_stats"]
    gp = T.garch_preprocess_fold(PS.FOLDS_DIR, fold, PS.RESULT_DIR)
    _, Yte, _, _, _ = T.cached_load_windows_seq(
        gp["test"], cond_stats=cond_stats, target_stats=target_stats)
    valid_mask, z_te, df_te = T.compute_valid_mask(gp["test"], cond_stats)

    # 미래 조건 = 실현된 거시 경로 (VAE/GAN 과 동일 정보).
    #   조건부 모델끼리는 같은 조건을 주고 그 조건을 쓰는 *구조* 차이만 남긴다.
    sim_z = ZM.rollout_paths(ctx, ctx["real_tb_fut"], ctx["real_mb_fut"], seed, device)
    r = ctx["rescale"]
    sim_raw = PS.forward_garch_rescale(sim_z * r["tsd"] + r["tmu"], r["s2"], r["e2"],
                                       r["om"], r["al"], r["be"], r["mu"])

    # actual 은 스텝별 실현 σ 로 복원 (evaluate_test 와 동일, 정답 라벨이므로 누수 아님).
    tmu, tsd = float(target_stats["mean"]), float(target_stats["std"])
    gsig = df_te["garch_sigma"].to_numpy(float)
    gmu = df_te["garch_mu"].to_numpy(float)
    oidx = np.where(valid_mask)[0]
    sig = np.array([[gsig[int(o) + T.PAST_LEN + t] for t in range(T.FUTURE_LEN)]
                    for o in oidx])
    mu = np.array([[gmu[int(o) + T.PAST_LEN + t] for t in range(T.FUTURE_LEN)]
                   for o in oidx])
    actual_raw = (Yte.numpy() * tsd + tmu) * sig + mu

    if actual_raw.shape[0] != sim_raw.shape[0]:
        sys.exit(f"[FATAL] origin 수 불일치 {fold} s{seed}: "
                 f"actual {actual_raw.shape[0]} vs sim {sim_raw.shape[0]}")
    return dict(sim=sim_raw, act=actual_raw, pred_start=oidx + T.PAST_LEN)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 112)
    print(f"# 원점 전수(시간순) · 지평 13주 · 미래=실현 거시 경로 · 구간=전역 풀링")
    print(f"# 세 모델 모두 시드 {SEEDS} (시드 비대칭 제거).  DM 검정 HAC lag={DM_LAG}")
    print(f"# device={device}  n_sim={N_SIM}  MAC-Flow={FLOW_TAG}")
    print("#" * 112)

    import dm_test as DM
    out = {f: {} for f in FOLDS}
    dm_out = {f: {} for f in FOLDS}
    thin_out = {f: {} for f in FOLDS}
    pooled_loss = {}

    for fold in FOLDS:
        print(f"\n===== {fold}")
        # 모델별 시드별 배열 수집
        per_model = {}

        for mk in ("vae", "gan"):
            runs = []
            for sd in SEEDS:
                try:
                    sp = VG_SPEC[mk]
                    # run_tuning_all.baseline_cell 과 같은 태그 규약.
                    tag = (f"_t3skfu_c{sp['dctx']}h{sp['hid']}"
                           f"_lr{('%g' % sp['lr']).replace('-', 'm').replace('.', 'd')}"
                           f"_s{sd}")
                    # 베이스라인 시뮬 배열 캐시.  MAC-Flow 설정만 바꿔 비교할 때
                    # 베이스라인 40 회를 매번 다시 학습하는 낭비를 막는다.
                    # 시드가 고정이라 재학습해도 같은 결과가 나온다.
                    cp = os.path.join(VG_CACHE_DIR, f"{mk}{tag}_{fold}.npz")
                    if VG_CACHE and os.path.exists(cp):
                        z = np.load(cp)
                        runs.append(dict(sim=z["sim"], act=z["act"],
                                         pred_start=z["pred_start"]))
                        print(f"  [cache] {os.path.basename(cp)}")
                        continue
                    args = SimpleNamespace(model=mk, fold=fold, folds_dir=PS.FOLDS_DIR,
                                           out_dir=PS.RESULT_DIR, n_sim=N_SIM, seed=sd,
                                           lr=sp["lr"], tag=tag)
                    _saved = list(T.COND_COLS)
                    _saved_cap = (VG.D_CTX, VG.HID)
                    PS.set_cond_cols(PS.ENC_COLS)          # Table 3 의 5 채널
                    VG.COND_COLS = list(T.COND_COLS); VG.N_CH = len(VG.COND_COLS)
                    VG.SP_CH, VG.TBILL_CH = T.SP_CH, T.TBILL_CH
                    VG.D_CTX, VG.HID = sp["dctx"], sp["hid"]
                    try:
                        r = VG.run_fold(mk, fold, args, device)
                        runs.append(r)
                        if VG_CACHE:
                            np.savez(cp, sim=np.asarray(r["sim"], np.float32),
                                     act=np.asarray(r["act"], np.float32),
                                     pred_start=np.asarray(r["pred_start"], int))
                    finally:
                        PS.set_cond_cols(_saved)
                        VG.D_CTX, VG.HID = _saved_cap
                except Exception as e:
                    print(f"  [FAIL {mk} s{sd}] {e!r}")
            if runs:
                per_model[f"Cond{mk.upper()}"] = runs

        runs = []
        for sd in SEEDS:
            try:
                d = macflow_arrays(fold, sd, device)
                if d is not None:
                    runs.append(d)
            except Exception as e:
                print(f"  [FAIL MAC-Flow s{sd}] {e!r}")
        if runs:
            per_model["MAC-Flow"] = runs

        if "MAC-Flow" not in per_model or len(per_model) < 2:
            print("  [skip] 모델 부족"); continue

        # 원점 교집합 (시드 무관, 대표 1개로 잡음)
        rep = {k: v[0] for k, v in per_model.items()}
        _, common = align(rep)
        order = np.argsort(common)
        print(f"  원점 교집합 {len(common)} 개  " +
              " ".join(f"{k}:{len(v)}시드" for k, v in per_model.items()))

        # 시드별 원점 CRPS → 시드 평균 (세 모델 동일 처리)
        loss = {}
        for name, runs in per_model.items():
            per_seed = []
            for d in runs:
                ps = np.asarray(d["pred_start"], int)
                pos = {int(v): i for i, v in enumerate(ps)}
                sel = np.array([pos[v] for v in common], int)
                per_seed.append(crps_per_origin(np.asarray(d["sim"])[sel],
                                                np.asarray(d["act"])[sel]))
            loss[name] = np.mean(per_seed, axis=0)[order]

        # 요약 지표 (시드 평균, 전 원점)
        for name, runs in per_model.items():
            ms = []
            for d in runs:
                ps = np.asarray(d["pred_start"], int)
                pos = {int(v): i for i, v in enumerate(ps)}
                sel = np.array([pos[v] for v in common], int)
                ms.append(metrics(np.asarray(d["sim"])[sel], np.asarray(d["act"])[sel]))
            keys = [k for k in ms[0] if k != "crps_per_origin"]
            m = {k: float(np.mean([r[k] for r in ms])) for k in keys}
            # 시드 성분 표준편차·표준오차 (리뷰어 1 #1: "1,000 draws and five
            # seeds, you have the ingredients for standard errors. Report them.").
            # DM 의 HAC se 는 원점 표본변동이라 성분이 다르다.
            if len(ms) > 1:
                for k in keys:
                    v = np.array([r[k] for r in ms], float)
                    m[k + "_sd"] = float(v.std(ddof=1))
                    m[k + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v)))
            m["n_seed"] = len(ms)
            out[fold][name] = m

        # DM 검정
        base = loss["MAC-Flow"]
        for name in ("CondVAE", "CondGAN"):
            if name in loss:
                dm_out[fold][name] = DM.dm_test(loss[name], base, DM_LAG)

        # 비중첩(STRIDE 간격) 검정 — 리뷰어 1 #2.
        #   원점을 13 주 간격으로 솎으면 예측 구간이 겹치지 않는다.  오프셋 하나만
        #   쓰면 어느 14 개를 골랐느냐에 좌우되므로 13 개 오프셋을 각각 검정하고
        #   DM 통계량·p 를 요약한다 (합치면 겹침이 되살아난다).
        for name in ("CondVAE", "CondGAN"):
            if name not in loss:
                continue
            per_off = []
            for off in range(STRIDE):
                idx = np.arange(off, len(base), STRIDE)
                if idx.size < 8:                    # 표본이 너무 적으면 건너뛴다
                    continue
                # 부분표본 안에서는 겹침이 없으므로 HAC lag 0
                per_off.append(DM.dm_test(loss[name][idx], base[idx], 0))
            if per_off:
                dm_ = [r["dm_hln"] for r in per_off]
                p_ = [r["p_two_sided"] for r in per_off]
                thin_out[fold][name] = dict(
                    n_offset=len(per_off), n_per_offset=int(per_off[0]["n"]),
                    d_mean=float(np.mean([r["d_mean"] for r in per_off])),
                    dm_mean=float(np.mean(dm_)),
                    dm_min=float(np.min(dm_)), dm_max=float(np.max(dm_)),
                    p_median=float(np.median(p_)),
                    n_sig05=int(sum(1 for x in p_ if x < 0.05)),
                    n_pos=int(sum(1 for r in per_off if r["d_mean"] > 0)))

        pooled_loss[fold] = loss

    # ---------- 4 폴드 통합 ----------
    #   폴드를 이어붙여 한 번만 검정한다.  폴드 경계에서 시간이 끊기고 폴드마다
    #   변동성 수준이 다르므로(F_gfc CRPS ~0.019 vs F_long_A ~0.009) 차이 계열이
    #   이질적이다.  폴드별 결과와 같이 봐야 한다.
    pooled_dm = {}
    if pooled_loss:
        names = set.intersection(*[set(v) for v in pooled_loss.values()])
        if "MAC-Flow" in names:
            cat = {k: np.concatenate([pooled_loss[f][k] for f in FOLDS
                                      if f in pooled_loss]) for k in names}
            for name in ("CondVAE", "CondGAN"):
                if name in cat:
                    pooled_dm[name] = DM.dm_test(cat[name], cat["MAC-Flow"],
                                                 DM_LAG)

    json.dump(dict(summary=out, dm=dm_out, thin=thin_out, pooled=pooled_dm),
              open(OUT, "w"), indent=2, default=str)
    print(f"\nsaved {OUT}")

    order_m = ["MAC-Flow", "CondVAE", "CondGAN"]
    cols = [("n_origin", "원점"), ("crps", "CRPS"), ("cov50", "cov50"),
            ("cov80", "cov80"), ("cov95", "cov95"),
            ("skew_actual", "skew실측"), ("skew_sim", "skew모형"),
            # 꼬리위험 축 — 논문 주제이고 §4.2·§4.3 이 쓰는 지표다.
            ("cvar1_actual", "CVaR1실측"), ("cvar1_sim", "CVaR1모형"),
            ("ihl_mean_actual", "IHL평실측"), ("ihl_mean_sim", "IHL평모형"),
            ("ihl_cvar10_actual", "IHLcv실측"), ("ihl_cvar10_sim", "IHLcv모형")]
    print(f"\n{'='*104}\n[요약] 원점 전수 · 시드 {len(SEEDS)}개 평균 · 미래=실현 경로\n{'='*104}")
    print(f"{'fold':<18}{'model':<11}" + "".join(f"{c[1]:>11}" for c in cols))
    for fold in FOLDS:
        for name in order_m:
            r = out[fold].get(name)
            if r is None:
                continue
            cells = [f"{r[k]:>11.1f}" if k == "n_origin"
                     else f"{r[k]:>11.5f}" if k == "crps" else f"{r[k]:>11.4f}"
                     for k, _ in cols]
            print(f"{fold:<18}{name:<11}" + "".join(cells))
        print()

    # 시드 성분 표준오차 — 리뷰어 1 #1 이 요구한 불확실성 표시.
    se_cols = [("crps", "CRPS"), ("cov80", "cov80"), ("cov95", "cov95"),
               ("skew_sim", "skew모형"), ("cvar1_sim", "CVaR1모형"),
               ("ihl_mean_sim", "IHL평균모형"), ("ihl_cvar10_sim", "IHLcv10모형")]
    print(f"{'='*104}\n[시드 표준오차] 시드 {len(SEEDS)}개 · 평균 ± SE "
          f"(DM 의 HAC se 는 원점 성분이라 별개)\n{'='*104}")
    print(f"{'fold':<18}{'model':<11}" +
          "".join(f"{c[1]:>22}" for c in se_cols))
    for fold in FOLDS:
        for name in order_m:
            r = out[fold].get(name)
            if r is None:
                continue
            cells = []
            for k, _ in se_cols:
                se = r.get(k + "_se")
                fmt = "{:.5f}" if k == "crps" else "{:.4f}"
                cells.append("{:>22}".format(
                    (fmt + " ± " + fmt).format(r[k], se) if se is not None
                    else fmt.format(r[k])))
            print(f"{fold:<18}{name:<11}" + "".join(cells))
        print()

    print(f"{'='*104}\n[DM 검정] 원점 시간순 · HAC lag={DM_LAG} · HLN 보정 · 양측\n{'='*104}")
    print(f"{'fold':<18}{'비교':<24}{'n':>5}{'mean diff':>12}{'HAC se':>11}"
          f"{'DM(HLN)':>10}{'p':>9}")
    for fold in FOLDS:
        for name, r in dm_out[fold].items():
            print(f"{fold:<18}{'MAC-Flow vs ' + name:<24}{r['n']:>5}"
                  f"{r['d_mean']:>12.5f}{r['se']:>11.5f}"
                  f"{r['dm_hln']:>10.3f}{r['p_two_sided']:>9.4f}")
    print("  (mean diff > 0 이면 MAC-Flow 의 CRPS 가 낮다 = 우위)")

    print("\n" + "=" * 104)
    print(f"[비중첩 검정] 원점 {STRIDE} 간격 · 오프셋 {STRIDE} 개 각각 검정 "
          f"(부분표본 안에서는 예측구간 겹침 없음, HAC lag=0)")
    print("=" * 104)
    print(f"{'fold':<18}{'비교':<24}{'오프셋':>6}{'n/오프셋':>9}"
          f"{'mean diff':>12}{'DM 평균':>10}{'DM 범위':>18}"
          f"{'p 중앙값':>10}{'p<.05':>7}{'양수':>6}")
    for fold in FOLDS:
        for name, r in thin_out[fold].items():
            rng = "[{:.2f}, {:.2f}]".format(r["dm_min"], r["dm_max"])
            sig = "{}/{}".format(r["n_sig05"], r["n_offset"])
            pos = "{}/{}".format(r["n_pos"], r["n_offset"])
            print("{:<18}{:<24}{:>6}{:>9}{:>12.5f}{:>10.3f}{:>18}"
                  "{:>10.4f}{:>7}{:>6}".format(
                      fold, "MAC-Flow vs " + name, r["n_offset"],
                      r["n_per_offset"], r["d_mean"], r["dm_mean"], rng,
                      r["p_median"], sig, pos))
    print("  각 오프셋은 독립 부분표본이다.  '양수' 는 MAC-Flow 가 이긴 오프셋 수.")

    if pooled_dm:
        print("\n" + "=" * 104)
        print(f"[4 폴드 통합] 폴드를 이어붙여 한 번 검정 · HAC lag={DM_LAG}")
        print("=" * 104)
        print(f"{'비교':<24}{'n':>6}{'mean diff':>12}{'HAC se':>11}"
              f"{'DM(HLN)':>10}{'p':>9}")
        for name, r in pooled_dm.items():
            print(f"{'MAC-Flow vs ' + name:<24}{r['n']:>6}{r['d_mean']:>12.5f}"
                  f"{r['se']:>11.5f}{r['dm_hln']:>10.3f}{r['p_two_sided']:>9.4f}")
        print("  폴드 경계에서 시간이 끊기고 폴드마다 변동성 수준이 다르다.")
        print("  폴드별 표와 같이 봐야 한다.")


if __name__ == "__main__":
    main()
