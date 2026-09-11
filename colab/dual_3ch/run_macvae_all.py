# -*- coding: utf-8 -*-
"""MAC-VAE 원스톱 — 헤드 튜닝 → 5 시드 학습 → MAC-Flow 대 MAC-VAE 짝 비교표 → §4.3 경로형태.

무엇을 답하는가 (R2#3)
----------------------
MAC-Flow 의 인코더·미래경로 요약·스텝별 재주입·AR 롤아웃을 그대로 두고 흐름 헤드만
조건부 VAE 헤드로 바꾼 통제 모형(vae_head.MambaVAEARFpath)을 만들어, 같은 원점·같은
시드에서 MAC-Flow 와 짝지어 비교한다.  차이가 없으면 "성능은 이중 주입 구조가 만들고
헤드는 교체 가능", 왜도 등에서만 갈리면 그 항목이 흐름 헤드의 몫이다.

단계 (env MV_STAGES 로 고른다, 기본 tune,train,compare,pathshape)
-----------------------------------------------------------------
  tune      : 헤드 hidden × latent 격자, 시드 1 개(MV_TUNE_SEED) × 4 폴드.  선택 = 4 폴드 평균
              val NLL(= −ELBO, 주당) 최소.  MAC-Flow 본모형(run_full_fpath)과 같은 val NLL 기준.
              태그 rvAbl_full_fpath_novol_vaehead_h{hid}l{lat}_d2_s{seed}.  격자: MV_HID, MV_LATENT.
  train     : 고른 (hid, lat) 로 시드 MV_SEEDS × 4 폴드.  태그 rvAbl_full_fpath_novol_vaehead_d2_s{seed}
              (PS_HEAD=vae 가 읽는 정식 태그).  튜닝 시드 결과는 재학습 없이 복사한다.
  compare   : 두 헤드를 같은 원점(교집합)·같은 시드로 재추론해
              · CRPS: 원점별 손실의 DM 검정(HAC lag 12, HLN) 폴드별 + 4 폴드 통합,
                      13 간격 비중첩 오프셋 13 개 각각 DM(lag 0)
              · cov80 / cov95 / 왜도 / CVaR1 오차 / IHL CVaR10 오차: 비중첩 오프셋 13 개 평균의 Welch
              결과: result/macvae_compare.json + 화면 표.  Δ 는 (MAC-VAE − MAC-Flow).
  pathshape : §4.3 zero-mean 경로형태 실험을 MAC-VAE 로 (pathshape_zeromean_anchored_rawvol.py,
              PS_HEAD=vae).  캐시 디렉터리는 _vaehead 접미사로 분리된다.

사용 (Colab)
------------
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/run_macvae_all.py
  격자를 줄이려면  MV_HID=192 MV_LATENT=16  (튜닝 4 런),  단계를 고르려면  MV_STAGES=compare
"""
import json
import os
import shutil
import subprocess
import sys
import time

# 학습 러너(run_full_fpath) 가 import 시점에 읽는 설정 — 본모형(fpath_novol d2) 과 동일 + VAE 헤드.
os.environ["FPATH_HEAD"] = "vae"
os.environ["FPATH_NOVOL"] = "1"
os.environ["FPATH_SUMMARY_ONLY"] = "0"
os.environ["FPATH_DIM"] = os.environ.get("FPATH_DIM", "2")
os.environ.setdefault("PS_BASE", "fpath_novol")
os.environ.setdefault("PS_HEAD", "flow")          # compare 단계에서 헤드별로 직접 토글한다

import torch                                                          # noqa: E402  (numpy/pandas 보다 먼저 — Windows 에서 뒤에 오면 c10.dll 초기화 실패)
import numpy as np                                                    # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_full_fpath as R                                            # noqa: E402  (patch_rawvol + 헤드 monkeypatch)
import train_garch_flow as T                                          # noqa: E402
import vae_head                                                       # noqa: E402
from train_garch_flow import main_worker                              # noqa: E402

STAGES = [s for s in os.environ.get("MV_STAGES", "tune,train,compare,pathshape").split(",") if s]
HIDS = [int(v) for v in os.environ.get("MV_HID", "128,192,256").split(",") if v]
LATS = [int(v) for v in os.environ.get("MV_LATENT", "8,16").split(",") if v]
TUNE_SEED = int(os.environ.get("MV_TUNE_SEED", "2026"))
SEEDS = [int(s) for s in os.environ.get("MV_SEEDS", "2026,2027,2028,2029,2030").split(",") if s.strip()]
FOLDS = R.FOLDS
DIM = int(os.environ["FPATH_DIM"])
RESULT_DIR = R.RESULT_DIR
FLOW_PREFIX = f"rvAbl_full_fpath_novol_d{DIM}"
VAE_PREFIX = f"rvAbl_full_fpath_novol_vaehead_d{DIM}"
TUNE_JSON = os.path.join(RESULT_DIR, "macvae_tuning.json")
CMP_JSON = os.path.join(RESULT_DIR, "macvae_compare.json")
STRIDE, DM_LAG = 13, 12
LABEL = {"F_gfc": "Financial crisis (2006-2010)", "F_long_A": "Recovery (2011-2015)",
         "F_long_B_origin": "COVID (2016-2020)", "F_long": "Tightening (2021-2025)"}


def _paths(tag, fold):
    base = os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{fold}")
    return {"best": base + "_best.pt", "log": base + "_log.csv", "summary": base + "_summary.json"}


def _setup_data_flags():
    # run_full_fpath.main 과 동일한 조건 (미래 tbill 보임 + metab unmask, sp 미마스크).
    R.set_cond_cols(R.ENC_FULL, "tbill_wr")
    T.MASK_FUTURE_TBILL = False
    T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
    T.ENCODER_MASK_SP = False


def _train_one(tag, fold, seed, hid, lat):
    p = _paths(tag, fold)
    if os.path.exists(p["summary"]):
        print(f"  [skip] {os.path.basename(p['summary'])}")
        return True
    vae_head.HID, vae_head.LATENT = int(hid), int(lat)          # VAEHead 가 호출 시점 전역을 읽는다
    spec = dict(R.LOCKED); spec.update(fold=fold, seed=seed, tag=tag)
    print(f"  [run] {tag} fold={fold}  (head hid={hid} latent={lat})")
    t0 = time.time()
    try:
        main_worker(spec)
        print(f"    done ({time.time() - t0:.0f}s)")
        return True
    except Exception as e:                                            # noqa: BLE001
        print(f"    [FAIL] {tag} {fold}: {e!r}")
        return False


def _val_nll_per_week(tag, fold):
    p = _paths(tag, fold)["summary"]
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    v = d.get("best_val_nll_per_week")
    if v is None and d.get("best_val_nll") is not None:
        v = float(d["best_val_nll"]) / T.FUTURE_LEN
    return None if v is None else float(v)


# ────────────────────────────────────────────────────────────── tune
def stage_tune():
    print("#" * 100)
    print(f"# [tune] MAC-VAE 헤드 격자 hid={HIDS} × latent={LATS}, seed={TUNE_SEED}, 4 폴드")
    print("#   선택 = 4 폴드 평균 val NLL(−ELBO, 주당) 최소 — run_full_fpath 의 val NLL 체크포인트 기준과 동일")
    print("#" * 100)
    _setup_data_flags()
    grid = {}
    for hid in HIDS:
        for lat in LATS:
            tag = f"rvAbl_full_fpath_novol_vaehead_h{hid}l{lat}_d{DIM}_s{TUNE_SEED}"
            for fold in FOLDS:
                _train_one(tag, fold, TUNE_SEED, hid, lat)
            vals = {f: _val_nll_per_week(tag, f) for f in FOLDS}
            ok = [v for v in vals.values() if v is not None]
            grid[f"h{hid}l{lat}"] = dict(hid=hid, latent=lat, per_fold=vals,
                                         mean=(float(np.mean(ok)) if len(ok) == len(FOLDS) else None))
    print(f"\n{'combo':<10}" + "".join(f"{LABEL[f][:12]:>14}" for f in FOLDS) + f"{'mean':>10}")
    for k, g in grid.items():
        row = "".join(f"{(g['per_fold'][f] if g['per_fold'][f] is not None else float('nan')):>14.4f}" for f in FOLDS)
        print(f"{k:<10}{row}{(g['mean'] if g['mean'] is not None else float('nan')):>10.4f}")
    done = {k: g for k, g in grid.items() if g["mean"] is not None}
    if not done:
        sys.exit("[FATAL] 튜닝 결과 없음")
    best = min(done, key=lambda k: done[k]["mean"])
    sel = dict(best=best, hid=done[best]["hid"], latent=done[best]["latent"],
               criterion="mean val NLL per week (−ELBO)", seed=TUNE_SEED, grid=grid)
    json.dump(sel, open(TUNE_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[tune] 선택 = {best} (hid={sel['hid']}, latent={sel['latent']})  → {TUNE_JSON}")
    return sel


def _selected():
    if os.path.exists(TUNE_JSON):
        d = json.load(open(TUNE_JSON, encoding="utf-8"))
        return int(d["hid"]), int(d["latent"]), d.get("best")
    print(f"[warn] {os.path.basename(TUNE_JSON)} 없음 → 기본값 hid={vae_head.HID} latent={vae_head.LATENT}")
    return vae_head.HID, vae_head.LATENT, None


# ────────────────────────────────────────────────────────────── train
def stage_train():
    hid, lat, best = _selected()
    print("#" * 100)
    print(f"# [train] MAC-VAE 5 시드 학습  hid={hid} latent={lat}  seeds={SEEDS}  태그 {VAE_PREFIX}_s*")
    print("#" * 100)
    _setup_data_flags()
    for seed in SEEDS:
        tag = f"{VAE_PREFIX}_s{seed}"
        for fold in FOLDS:
            # 튜닝 시드는 같은 설정의 결과가 이미 있으므로 복사 (재학습 = 같은 시드·같은 설정이라 동일 결과).
            if seed == TUNE_SEED and best is not None:
                src = _paths(f"rvAbl_full_fpath_novol_vaehead_{best}_d{DIM}_s{seed}", fold)
                dst = _paths(tag, fold)
                if os.path.exists(src["summary"]) and not os.path.exists(dst["summary"]):
                    for k in ("best", "log", "summary"):
                        if os.path.exists(src[k]):
                            shutil.copy2(src[k], dst[k])
                    print(f"  [copy] {os.path.basename(src['summary'])} → {os.path.basename(dst['summary'])}")
            _train_one(tag, fold, seed, hid, lat)


# ────────────────────────────────────────────────────────────── compare
def _head_arrays(PS, RTC, head, fold, seed, device):
    """헤드별 태그로 PS 를 돌려 세운 뒤 재추론 (같은 원점·같은 시드·실현 거시 경로)."""
    PS.HEAD = head
    PS.TAG_PREFIX = VAE_PREFIX if head == "vae" else FLOW_PREFIX
    PS.CACHE_SUFFIX = f"_fpath_novol{'_vaehead' if head == 'vae' else ''}_d{DIM}"
    RTC.FLOW_TAG = PS.TAG_PREFIX
    return RTC.macflow_arrays(fold, seed, device)


def stage_compare():
    import torch
    from scipy import stats
    import analyze_pathshape_rawvol as PS                             # noqa: E402
    import run_thin_compare as RTC                                    # noqa: E402
    import report_tail_all as RT                                      # noqa: E402
    import fit_thin_4_1_1 as FT                                       # noqa: E402
    import dm_test as DM                                              # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print(f"# [compare] MAC-Flow({FLOW_PREFIX}) 대 MAC-VAE({VAE_PREFIX})  seeds={SEEDS}  device={device}")
    print("#   같은 원점(교집합)·같은 시드·미래=실현 거시 경로.  Δ = MAC-VAE − MAC-Flow.")
    print("#   CRPS: DM(HAC lag 12, HLN) — Δ>0 이면 MAC-Flow 손실이 낮다.  나머지: 비중첩 오프셋 13 개 평균의 Welch.")
    print("#" * 100)

    out = {"folds": {}, "pooled": {}}
    pooled = {"flow": [], "vae": []}
    keys = ["CRPS", "cov80", "cov95", "skew모형", "CVaR1 오차", "IHLcv10 오차"]
    for fold in FOLDS:
        runs = {}
        for head in ("flow", "vae"):
            lst = []
            for sd in SEEDS:
                try:
                    r = _head_arrays(PS, RTC, head, fold, sd, device)
                    if r is not None:
                        lst.append(r)
                except Exception as e:                                # noqa: BLE001
                    print(f"  [{head} s{sd}] {e!r}")
            if lst:
                runs["MAC-Flow" if head == "flow" else "MAC-VAE"] = lst
        if len(runs) < 2:
            print(f"\n===== {fold}: 두 헤드 자료가 모두 있어야 한다 (있는 것: {list(runs)})")
            continue
        aligned, order = RT.align(runs)
        if aligned is None:
            print(f"\n===== {fold}: 공통 원점 없음"); continue
        n_orig = len(order)
        print(f"\n===== {LABEL.get(fold, fold)}  원점 {n_orig}  MAC-Flow {len(aligned['MAC-Flow'])}시드  "
              f"MAC-VAE {len(aligned['MAC-VAE'])}시드")

        # CRPS 원점별 손실 (시드 평균) → DM
        L = {}
        for name in ("MAC-Flow", "MAC-VAE"):
            L[name] = np.mean([RT.crps_per_origin(s, a) for s, a in aligned[name]], axis=0)
        dm = DM.dm_test(L["MAC-VAE"], L["MAC-Flow"], DM_LAG)
        pooled["flow"].append(L["MAC-Flow"]); pooled["vae"].append(L["MAC-VAE"])
        offs = []
        for o in range(STRIDE):
            idx = np.arange(o, n_orig, STRIDE)
            if idx.size >= 8:
                offs.append(DM.dm_test(L["MAC-VAE"][idx], L["MAC-Flow"][idx], 0))
        off_p = np.array([r["p_two_sided"] for r in offs], float)
        off_pos = int(sum(r["d_mean"] > 0 for r in offs))
        off_sig = int(np.sum(off_p < 0.05))
        print(f"  CRPS  Flow {L['MAC-Flow'].mean():.5f}  VAE {L['MAC-VAE'].mean():.5f}  "
              f"Δ {dm['d_mean']:+.6f}  DM {dm['dm_hln']:+.3f}  p {dm['p_two_sided']:.4f}   "
              f"| 비중첩 p중앙 {np.median(off_p):.4f}  p<.05 {off_sig}/{len(offs)}  Flow우위 {off_pos}/{len(offs)}")

        # 비중첩 오프셋 지표 → Welch
        per = {name: {k: [] for k in keys} for name in ("MAC-Flow", "MAC-VAE")}
        for name in per:
            for sim, act in aligned[name]:                            # 시드별
                rows = {k: [] for k in keys}
                for o in range(STRIDE):
                    idx = np.arange(o, n_orig, STRIDE)
                    if idx.size < 8:
                        continue
                    m = FT.metrics(sim[idx], act[idx])
                    for k in keys:
                        rows[k].append(m[k])
                for k in keys:
                    per[name][k].append(rows[k])
        fold_out = {"n_origin": n_orig, "crps_dm": dm,
                    "crps_offsets": dict(p_median=float(np.median(off_p)), n=len(offs),
                                         sig=off_sig, flow_better=off_pos), "welch": {}}
        print(f"  {'지표':<14}{'MAC-Flow':>12}{'MAC-VAE':>12}{'Δ(VAE−Flow)':>14}{'t':>8}{'p':>9}")
        for k in keys:
            a = np.asarray(per["MAC-Flow"][k], float).mean(axis=0)     # (13,) 시드 평균 후 오프셋
            b = np.asarray(per["MAC-VAE"][k], float).mean(axis=0)
            tt = stats.ttest_ind(b, a, equal_var=False)
            fold_out["welch"][k] = dict(flow=float(a.mean()), vae=float(b.mean()),
                                        delta=float(b.mean() - a.mean()),
                                        t=float(tt.statistic), p=float(tt.pvalue),
                                        flow_sd=float(a.std(ddof=1)), vae_sd=float(b.std(ddof=1)))
            print(f"  {k:<14}{a.mean():>12.4f}{b.mean():>12.4f}{b.mean() - a.mean():>+14.4f}"
                  f"{tt.statistic:>+8.2f}{tt.pvalue:>9.4f}")
        out["folds"][fold] = fold_out

    if pooled["flow"]:
        f_all, v_all = np.concatenate(pooled["flow"]), np.concatenate(pooled["vae"])
        dm = DM.dm_test(v_all, f_all, DM_LAG)
        out["pooled"]["crps_dm"] = dm
        print(f"\n[4 폴드 통합] CRPS  Flow {f_all.mean():.5f}  VAE {v_all.mean():.5f}  Δ {dm['d_mean']:+.6f}  "
              f"DM {dm['dm_hln']:+.3f}  p {dm['p_two_sided']:.4f}  (n={dm['n']})")
    json.dump(out, open(CMP_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(f"saved {CMP_JSON}")


# ────────────────────────────────────────────────────────────── pathshape
def stage_pathshape():
    print("#" * 100)
    print("# [pathshape] §4.3 zero-mean 경로형태 — MAC-VAE (PS_HEAD=vae).  MAC-Flow 결과는 기존 캐시 그대로.")
    print("#" * 100)
    env = dict(os.environ, PS_HEAD="vae", PS_BASE="fpath_novol", FPATH_DIM=str(DIM))
    for script in ("pathshape_zeromean_anchored_rawvol.py",):
        cmd = [sys.executable, os.path.join(HERE, script)]
        print("  $", " ".join(f"{k}={env[k]}" for k in ("PS_HEAD", "PS_BASE", "FPATH_DIM")), " ".join(cmd))
        rc = subprocess.run(cmd, env=env).returncode
        print(f"  [{script}] exit {rc}")


if __name__ == "__main__":
    print(f"[run_macvae_all] stages={STAGES}")
    if "tune" in STAGES:
        stage_tune()
    if "train" in STAGES:
        stage_train()
    if "compare" in STAGES:
        stage_compare()
    if "pathshape" in STAGES:
        stage_pathshape()
    print("\n[done]")
