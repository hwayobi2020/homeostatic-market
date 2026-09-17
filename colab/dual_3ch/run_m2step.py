# -*- coding: utf-8 -*-
"""R1#7 강건성(2차): 1981-01 이전 M2 를 선형 보간 대신 월간값 계단(전방 채움)으로 바꿔 재학습.

왜
--
리뷰어 1 의 원래 질문은 "1981 이전 M2 의 선형 보간이 학습에 인공물을 남겼는가"다.
run_from1981.py(학습 시작을 1981 로 이동)는 10 년 표본 제거와 표준화 통계 변경을 같이
가져와 보간 효과를 분리하지 못한다.  여기서는 학습 구간·검증·시험·표준화·에폭을 모두
그대로 두고, 1981-01-05 이전 M2 채움 방식만 바꾼다 → 차이 = 보간 효과.

무엇을
------
  1. 폴드 train CSV 의 m2_level 을 1981-01-05 이전 구간만 M2NS 월간값 전방 채움(계단)으로
     교체하고, 같은 식(data/extend_to_1971.py:334-402)으로 m2_growth → m2_growth_lag(2주,
     2021-02 이전) → m2_13w_cum_lag(13주 합) → metab_13w = m2 − indpro − cpi 를 다시 계산.
     모형 입력 중 M2 파생 채널은 metab_13w 뿐(run_full_fpath.ENC_FULL).
     검증: 원래 m2_level 로 같은 식을 돌리면 CSV 의 metab_13w 가 그대로 복원되어야 한다.
     새 폴드 이름 {fold}m2step (val/test 는 복사).  바뀌는 행 = 1981-04 경까지의 학습 행.
  2. 본모형과 같은 spec(run_full_fpath.LOCKED, fpath_novol d2, MAX_EPOCH/PATIENCE 동일)으로
     4 폴드 × 시드 R_M2STEP_SEEDS(기본 5개) 학습.  태그 rvAbl_full_fpath_novol_m2step_d2_s{seed}.
  3. 전 구간 학습본(rvAbl_full_fpath_novol_d2, 5 시드 평균±sd)과 나란히 출력,
     result/robust_m2step.json 저장.

사용 (Colab)
------------
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !R_M2STEP_DRY=1 python colab/dual_3ch/run_m2step.py      # 폴드 생성 + 복원 검증만
    !python colab/dual_3ch/run_m2step.py                     # 학습 + 비교
M2NS 월간 CSV 는 data/fred/M2NS.csv (extend_to_1971.py 가 저장), 없으면 FRED 에서 받는다.
"""
import glob
import json
import os
import shutil
import sys
import time

os.environ["FPATH_NOVOL"] = "1"
os.environ["FPATH_SUMMARY_ONLY"] = "0"
os.environ["FPATH_DIM"] = os.environ.get("FPATH_DIM", "2")
os.environ.pop("FPATH_HEAD", None)                       # 흐름 헤드 (본모형)

import torch                                             # noqa: E402  (pandas 보다 먼저 — Windows c10.dll)
import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

import run_full_fpath as R                               # noqa: E402
import train_garch_flow as T                             # noqa: E402
from train_garch_flow import main_worker                 # noqa: E402

SPLICE = pd.Timestamp("1981-01-05")                      # extend_to_1971.M2_SPLICE_DATE
SPLIT = pd.Timestamp("2021-02-01")                       # extend_to_1971.SPLIT_DATE (m2 lag 2 → 1)
M2_LAG_PRE, M2_LAG_POST, WINDOW = 2, 1, 13
SEEDS = [int(s) for s in os.environ.get("R_M2STEP_SEEDS", "2026,2027,2028,2029,2030").split(",") if s.strip()]
DRY = os.environ.get("R_M2STEP_DRY", "0") == "1"
FOLDS = R.FOLDS
DIM = int(os.environ["FPATH_DIM"])
SUFFIX = "m2step"
BASE_TAG = f"rvAbl_full_fpath_novol_d{DIM}"
NEW_TAG = f"rvAbl_full_fpath_novol_m2step_d{DIM}"
OUT = os.path.join(R.RESULT_DIR, "robust_m2step.json")
KEYS = [("crps_pooled", "CRPS"), ("coverage_80", "cov80"), ("coverage_95", "cov95"), ("skew_sim", "skew")]
LABEL = {"F_gfc": "Financial crisis (2006-2010)", "F_long_A": "Recovery (2011-2015)",
         "F_long_B_origin": "COVID (2016-2020)", "F_long": "Tightening (2021-2025)"}


def load_m2ns():
    p = os.path.join(ROOT, "data", "fred", "M2NS.csv")
    if os.path.exists(p):
        s = pd.read_csv(p, parse_dates=["DATE"]).set_index("DATE").iloc[:, 0].astype(float)
    else:
        import pandas_datareader.data as web
        s = web.DataReader("M2NS", "fred", "1959-01-01", "1990-12-31").iloc[:, 0].astype(float)
        s.index = pd.to_datetime(s.index)
        os.makedirs(os.path.dirname(p), exist_ok=True); s.to_frame("M2NS").to_csv(p, index_label="DATE")
    return s.dropna().sort_index()


def m2_chain(level, dates, df):
    """m2_level → metab_13w (extend_to_1971.py:299,334-335,375,381,401-402 와 같은 식)."""
    g = np.log(level).diff()
    g_lag = g.shift(M2_LAG_PRE).where(dates < SPLIT, g.shift(M2_LAG_POST))
    cum = g_lag.rolling(WINDOW).sum()
    metab = cum - df["indpro_13w_pct_lag"].values - df["cpi_13w_cum_lag"].values
    return g, g_lag, cum, metab


def make_folds(m2ns):
    for fold in FOLDS:
        for split in ("train", "val", "test"):
            src = os.path.join(R.FOLDS_DIR, f"{fold}_{split}.csv")
            dst = os.path.join(R.FOLDS_DIR, f"{fold}{SUFFIX}_{split}.csv")
            if os.path.exists(dst) and not DRY:
                continue
            if split != "train":
                if not os.path.exists(dst): shutil.copy2(src, dst)
                continue
            df = pd.read_csv(src, parse_dates=["date"])
            dates = pd.DatetimeIndex(df["date"])
            # 1) 복원 검증: 원래 m2_level 로 식을 돌리면 CSV metab_13w 가 나와야 한다 (첫 15 행은 1970 자료 필요라 제외)
            _, _, _, recon = m2_chain(pd.Series(df["m2_level"].values, index=dates), dates, df)
            ok = recon.notna().values
            err = np.nanmax(np.abs(recon.values[ok] - df["metab_13w"].values[ok]))
            print(f"  [{fold}] 복원 검증: 비교 행 {int(ok.sum())}/{len(df)}, max|재계산−CSV| = {err:.2e}")
            assert err < 1e-8, "metab_13w 재구성 식이 CSV 와 다르다 — 중단"
            # 2) 계단 채움 m2_level (1970 부터 확장 격자에서 계산해 첫 행의 lag/rolling 도 채운다)
            ext = pd.DatetimeIndex(sorted(set(dates[0] - pd.Timedelta(weeks=k) for k in range(1, WINDOW + M2_LAG_PRE + 4))))
            grid = ext.union(dates)
            step = m2ns.reindex(m2ns.index.union(grid)).sort_index().ffill().reindex(grid)
            orig = pd.Series(df["m2_level"].values, index=dates).reindex(grid)
            level_new = pd.Series(np.where(grid < SPLICE, step.values, orig.values), index=grid)
            # 확장 격자(1970 말)의 orig 는 NaN 이지만 모두 SPLICE 이전이라 step 값이 쓰인다
            assert level_new.loc[dates].notna().all(), "m2_level 계단 채움에 NaN"
            gg = np.log(level_new).diff()
            gg_lag = gg.shift(M2_LAG_PRE).where(grid < SPLIT, gg.shift(M2_LAG_POST))
            ccum = gg_lag.rolling(WINDOW).sum()
            cum_d = ccum.loc[dates].values
            metab_new = cum_d - df["indpro_13w_pct_lag"].values - df["cpi_13w_cum_lag"].values
            changed = np.abs(metab_new - df["metab_13w"].values) > 1e-12
            n_ch = int(np.nansum(changed))
            last = dates[np.where(changed)[0].max()].date() if n_ch else None
            print(f"  [{fold}] 계단 채움: 바뀐 행 {n_ch}/{len(df)}  (~{last}),  "
                  f"max|Δmetab_13w| = {np.nanmax(np.abs(metab_new - df['metab_13w'].values)):.5f},  "
                  f"train 기간 {dates[0].date()} ~ {dates[-1].date()}")
            assert last is None or pd.Timestamp(last) < SPLICE + pd.Timedelta(weeks=WINDOW + M2_LAG_PRE + 1), "1981 이후 행이 바뀜"
            out = df.copy()
            out["m2_level"] = level_new.loc[dates].values
            out["m2_growth"] = gg.loc[dates].values
            out["m2_growth_lag"] = gg_lag.loc[dates].values
            out["m2_13w_cum_lag"] = cum_d
            out["metab_13w"] = metab_new
            if DRY and os.path.exists(dst):
                os.remove(dst)
            out.to_csv(dst, index=False)
            print(f"  [{fold}] saved {os.path.basename(dst)}")


def train_all():
    R.set_cond_cols(R.ENC_FULL, "tbill_wr")
    T.MASK_FUTURE_TBILL = False
    T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
    T.ENCODER_MASK_SP = False
    for seed in SEEDS:
        tag = f"{NEW_TAG}_s{seed}"
        for fold in FOLDS:
            f2 = f"{fold}{SUFFIX}"
            summ = os.path.join(R.RESULT_DIR, f"garch_flow_ar_{tag}_{f2}_summary.json")
            if os.path.exists(summ):
                print(f"  [skip] {os.path.basename(summ)}"); continue
            spec = dict(R.LOCKED); spec.update(fold=f2, seed=seed, tag=tag, folds_dir=R.FOLDS_DIR)
            print(f"  [run] {tag} fold={f2}"); t0 = time.time()
            try:
                main_worker(spec); print(f"    done ({time.time() - t0:.0f}s)")
            except Exception as e:                       # noqa: BLE001
                print(f"    [FAIL] {tag} {f2}: {e!r}")


def _test_eval(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    return d.get("test_eval")


def _msd(v):
    return f"{np.mean(v):.4f}±{np.std(v, ddof=1):.4f}" if len(v) > 1 else (f"{v[0]:.4f}" if v else "—")


def compare():
    print("\n" + "#" * 112)
    print("# 1981 이전 M2: 선형 보간(본모형) vs 월간값 계단 채움 — 학습 구간·검증·시험·spec 동일.  값은 시험셋 전수 원점 기준.")
    print("#" * 112)
    out = {}
    hdr = f"{'Test period':<30}{'metric':<7}{'interp 5-seed':>18}{'step k-seed':>18}{'Δ(step−interp)':>16}{'Δ/sd':>8}"
    print(hdr); print("-" * len(hdr))
    for fold in FOLDS:
        base = [m for m in (_test_eval(p) for p in sorted(glob.glob(
            os.path.join(R.RESULT_DIR, f"garch_flow_ar_{BASE_TAG}_s*_{fold}_summary.json")))) if m]
        new = [m for m in (_test_eval(os.path.join(R.RESULT_DIR, f"garch_flow_ar_{NEW_TAG}_s{s}_{fold}{SUFFIX}_summary.json"))
                           for s in SEEDS) if m]
        if not new:
            print(f"{LABEL[fold]:<30}(m2step 결과 없음)"); continue
        row = {}; first = True
        for k, name in KEYS:
            ba = [m[k] for m in base if k in m]; nv = [m[k] for m in new if k in m]
            d = float(np.mean(nv)) - float(np.mean(ba)) if ba else float("nan")
            sd = float(np.std(ba, ddof=1)) if len(ba) > 1 else float("nan")
            print(f"{LABEL[fold] if first else '':<30}{name:<7}{_msd(ba):>18}{_msd(nv):>18}{d:>+16.4f}{(d / sd if sd else float('nan')):>8.2f}")
            first = False
            row[name] = dict(interp_mean=float(np.mean(ba)) if ba else None, interp_sd=sd,
                             step_mean=float(np.mean(nv)), step_sd=(float(np.std(nv, ddof=1)) if len(nv) > 1 else None),
                             n_seed_step=len(nv))
        if "skew_actual" in new[0]:
            print(f"{'':<30}{'(실측 skew':<7}{new[0]['skew_actual']:>18.4f})")
        out[fold] = row
        print("-" * len(hdr))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"saved {OUT}")


if __name__ == "__main__":
    print(f"[run_m2step] seeds={SEEDS} dry={DRY} folds_dir={R.FOLDS_DIR}")
    make_folds(load_m2ns())
    if not DRY:
        train_all()
        compare()
