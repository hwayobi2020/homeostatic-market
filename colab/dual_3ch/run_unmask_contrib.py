# -*- coding: utf-8 -*-
"""미래 경로 마스킹 해제 — 채널별 기여도 (ads_lag / wti_wr).

무엇을
------
본모형(fpath_novol d2)은 미래 13 주에서 tbill_wr 과 metab_13w 만 실제값을 받고, 나머지 인코더 채널
(ads_lag, wti_wr) 의 미래는 0 으로 마스킹된다.  이 스크립트는 그 채널을 풀어 **재학습**한 변형을 만들고
(마스킹은 학습 텐서 단계라 추론만으로는 못 푼다: train_garch_flow.py:407, fpath_model.py:71-76),
기본 모형과 같은 원점·같은 시드·실현 미래 경로에서 적합을 비교한다 = "미래 정보 한 채널의 기여".

  설정(UM_CONFIGS):  ads = +ads_lag,  wti = +wti_wr,  ads_wti = 둘 다     (base = 기존 체크포인트)
  태그:  rvAbl_full_fpath_novol_um-{cfg}_d{DIM}_s{seed}
  단계(UM_STAGES):  train → compare

compare 는 run_macvae_all.stage_compare 와 같은 지표·검정: CRPS 는 원점별 손실의 DM(HAC 12) + 비중첩
오프셋 13 개 DM, 나머지(cov80/95·왜도·CVaR1 오차·IHL CVaR10 오차)는 오프셋 13 개 평균의 Welch.
Δ = 변형 − base.  결과 result/unmask_contrib.json + 화면 표.

비용: 설정 3 × 폴드 4 × 시드 5 = 60 회 학습 (MAC-VAE 학습 단계의 3 배).  UM_SEEDS 로 줄일 수 있다.

사용
----
    !python colab/dual_3ch/run_unmask_contrib.py
    # 예: UM_CONFIGS=ads_wti UM_SEEDS=2026 UM_STAGES=train
"""
import json
import os
import sys
import time

os.environ["FPATH_HEAD"] = "flow"
os.environ["FPATH_NOVOL"] = "1"
os.environ["FPATH_SUMMARY_ONLY"] = "0"
os.environ["FPATH_DIM"] = os.environ.get("FPATH_DIM", "2")
os.environ.setdefault("PS_BASE", "fpath_novol")
os.environ.setdefault("PS_HEAD", "flow")

import torch                                                          # noqa: E402  (numpy 보다 먼저)
import numpy as np                                                    # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_full_fpath as R                                            # noqa: E402  (patch_rawvol + 흐름 헤드 monkeypatch)
import train_garch_flow as T                                          # noqa: E402
from train_garch_flow import main_worker                              # noqa: E402

CONFIGS_ALL = {"ads": ["ads_lag"], "wti": ["wti_wr"], "ads_wti": ["ads_lag", "wti_wr"]}
CONFIGS = [c for c in os.environ.get("UM_CONFIGS", "ads,wti,ads_wti").split(",") if c]
STAGES = [s for s in os.environ.get("UM_STAGES", "train,compare").split(",") if s]
SEEDS = [int(s) for s in os.environ.get("UM_SEEDS", "2026,2027,2028,2029,2030").split(",") if s.strip()]
FOLDS = R.FOLDS
DIM = int(os.environ["FPATH_DIM"])
RESULT_DIR = R.RESULT_DIR
BASE_PREFIX = f"rvAbl_full_fpath_novol_d{DIM}"
OUT_JSON = os.path.join(RESULT_DIR, "unmask_contrib.json")
STRIDE, DM_LAG = 13, 12
LABEL = {"F_gfc": "Financial crisis (2006-2010)", "F_long_A": "Recovery (2011-2015)",
         "F_long_B_origin": "COVID (2016-2020)", "F_long": "Tightening (2021-2025)"}


def prefix_of(cfg):
    return BASE_PREFIX if cfg == "base" else f"rvAbl_full_fpath_novol_um-{cfg}_d{DIM}"


def unmask_of(cfg):
    return ["metab_13w"] + ([] if cfg == "base" else CONFIGS_ALL[cfg])


def _paths(tag, fold):
    base = os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{fold}")
    return {"best": base + "_best.pt", "log": base + "_log.csv", "summary": base + "_summary.json"}


def _set_flags(cfg):
    """run_full_fpath.main 과 같은 데이터 조건 + 이 설정의 unmask 목록.  데이터 캐시는 키에 unmask 가 들어간다."""
    R.set_cond_cols(R.ENC_FULL, "tbill_wr")
    T.MASK_FUTURE_TBILL = False
    T.FUTURE_UNMASK_MACRO_COLS = unmask_of(cfg)
    T.ENCODER_MASK_SP = False
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass


# ────────────────────────────────────────────────────────────── train
def stage_train():
    for cfg in CONFIGS:
        print("#" * 100)
        print(f"# [train] {cfg}: unmask={unmask_of(cfg)}  seeds={SEEDS}  태그 {prefix_of(cfg)}_s*")
        print("#" * 100)
        _set_flags(cfg)
        for seed in SEEDS:
            tag = f"{prefix_of(cfg)}_s{seed}"
            for fold in FOLDS:
                p = _paths(tag, fold)
                if os.path.exists(p["summary"]):
                    print(f"  [skip] {os.path.basename(p['summary'])}"); continue
                spec = dict(R.LOCKED); spec.update(fold=fold, seed=seed, tag=tag)
                print(f"  [run] {tag} fold={fold}")
                t0 = time.time()
                try:
                    main_worker(spec)
                    print(f"    done ({time.time() - t0:.0f}s)")
                except Exception as e:                                # noqa: BLE001
                    print(f"    [FAIL] {tag} {fold}: {e!r}")


# ────────────────────────────────────────────────────────────── compare
def _arrays(PS, RTC, cfg, fold, seed, device):
    """설정별로 unmask 플래그·태그를 세운 뒤 재추론 (같은 원점·같은 시드·실현 거시 경로)."""
    _set_flags(cfg)
    PS.HEAD = "flow"
    PS.TAG_PREFIX = prefix_of(cfg)
    PS.CACHE_SUFFIX = f"_fpath_novol_d{DIM}" if cfg == "base" else f"_fpath_novol_um-{cfg}_d{DIM}"
    RTC.FLOW_TAG = PS.TAG_PREFIX
    return RTC.macflow_arrays(fold, seed, device)


def stage_compare():
    from scipy import stats
    import analyze_pathshape_rawvol as PS                             # noqa: E402
    import run_thin_compare as RTC                                    # noqa: E402
    import report_tail_all as RT                                      # noqa: E402
    import fit_thin_4_1_1 as FT                                       # noqa: E402
    import dm_test as DM                                              # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfgs = ["base"] + CONFIGS
    print("#" * 100)
    print(f"# [compare] 미래 unmask 기여도  설정={cfgs}  seeds={SEEDS}  device={device}")
    print("#   같은 원점(교집합)·같은 시드·미래=실현 거시 경로.  Δ = 변형 − base.")
    print("#   CRPS: DM(HAC 12) — Δ<0 이면 변형 손실이 낮다(정보 기여).  나머지: 비중첩 오프셋 13 개 평균의 Welch.")
    print("#" * 100)
    keys = ["CRPS", "cov80", "cov95", "skew모형", "CVaR1 오차", "IHLcv10 오차"]
    out = {"folds": {}, "pooled": {}}
    pooled = {c: [] for c in cfgs}
    for fold in FOLDS:
        runs = {}
        for cfg in cfgs:
            lst = []
            for sd in SEEDS:
                try:
                    r = _arrays(PS, RTC, cfg, fold, sd, device)
                    if r is not None:
                        lst.append(r)
                except Exception as e:                                # noqa: BLE001
                    print(f"  [{cfg} s{sd}] {e!r}")
            if lst:
                runs[cfg] = lst
        if "base" not in runs or len(runs) < 2:
            print(f"\n===== {fold}: base 와 변형이 모두 있어야 한다 (있는 것: {list(runs)})"); continue
        aligned, order = RT.align(runs)
        if aligned is None:
            print(f"\n===== {fold}: 공통 원점 없음"); continue
        n_orig = len(order)
        print(f"\n===== {LABEL.get(fold, fold)}  원점 {n_orig}  " + "  ".join(f"{c}:{len(aligned[c])}시드" for c in aligned))
        L = {c: np.mean([RT.crps_per_origin(s, a) for s, a in aligned[c]], axis=0) for c in aligned}
        per = {}
        for c in aligned:
            per[c] = {k: [] for k in keys}
            for sim, act in aligned[c]:
                rows = {k: [] for k in keys}
                for o in range(STRIDE):
                    idx = np.arange(o, n_orig, STRIDE)
                    if idx.size < 8:
                        continue
                    m = FT.metrics(sim[idx], act[idx])
                    for k in keys:
                        rows[k].append(m[k])
                for k in keys:
                    per[c][k].append(rows[k])
            pooled[c].append(L[c])
        fold_out = {"n_origin": n_orig, "cfg": {}}
        print(f"  {'cfg':<9}{'CRPS':>9}{'ΔCRPS':>10}{'DM p':>8}{'비중첩 변형우위':>14}"
              + "".join(f"{k:>13}" for k in keys[1:]))
        base_off = {k: np.asarray(per["base"][k], float).mean(axis=0) for k in keys}
        for c in aligned:
            row = {"crps": float(L[c].mean())}
            off = {k: np.asarray(per[c][k], float).mean(axis=0) for k in keys}
            line = f"  {c:<9}{L[c].mean():>9.5f}"
            if c == "base":
                line += f"{'':>10}{'':>8}{'':>14}"
            else:
                dm = DM.dm_test(L[c], L["base"], DM_LAG)
                offs = []
                for o in range(STRIDE):
                    idx = np.arange(o, n_orig, STRIDE)
                    if idx.size >= 8:
                        offs.append(DM.dm_test(L[c][idx], L["base"][idx], 0))
                won = int(sum(r["d_mean"] < 0 for r in offs))
                row.update(dcrps=float(dm["d_mean"]), dm_p=float(dm["p_two_sided"]), off_won=won, off_n=len(offs))
                line += f"{dm['d_mean']:>+10.6f}{dm['p_two_sided']:>8.4f}{won:>10}/{len(offs):<3}"
            for k in keys[1:]:
                v = float(off[k].mean()); cell = f"{v:>+9.4f}"
                if c != "base":
                    tt = stats.ttest_ind(off[k], base_off[k], equal_var=False)
                    row[k] = dict(value=v, delta=float(v - base_off[k].mean()), t=float(tt.statistic), p=float(tt.pvalue))
                    cell += ("*" if tt.pvalue < .05 else " ") + f"({v - base_off[k].mean():+.3f})"
                else:
                    row[k] = dict(value=v)
                    cell += "        "
                line += f"{cell:>13}"[:13].rjust(13)
            print(line)
            fold_out["cfg"][c] = row
        out["folds"][fold] = fold_out

    for c in CONFIGS:
        if pooled.get(c) and pooled.get("base") and len(pooled[c]) == len(pooled["base"]):
            a = np.concatenate(pooled[c]); b = np.concatenate(pooled["base"])
            dm = DM.dm_test(a, b, DM_LAG)
            out["pooled"][c] = dm
            print(f"\n[4 폴드 통합] {c}: CRPS {a.mean():.5f} vs base {b.mean():.5f}  Δ {dm['d_mean']:+.6f}  DM {dm['dm_hln']:+.3f}  p {dm['p_two_sided']:.4f}")
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(f"saved {OUT_JSON}")
    print("  * = Welch p<.05 (오프셋 13개).  괄호 = 변형 − base.  왜도는 실측 부호(음)와 대조해 읽을 것.")


if __name__ == "__main__":
    print(f"[run_unmask_contrib] configs={CONFIGS} stages={STAGES} seeds={SEEDS}")
    if "train" in STAGES:
        stage_train()
    if "compare" in STAGES:
        stage_compare()
    print("\n[done]")
