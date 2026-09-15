# -*- coding: utf-8 -*-
"""R1#7 강건성: 학습 구간을 1981-01-05(주간 M2 WM2NS 시작) 이후로 잘라 다시 학습 → 전 구간 학습본과 비교.

왜
--
1981-01 이전 M2 는 월간 M2NS 를 주간으로 선형 보간한 것이다 (data/extend_to_1971.py:6-22).
리뷰어는 그 10 년치가 학습에 인공물을 남겼는지 물었다.  시험 폴드는 2006 년 이후라
검증·시험 자료는 그대로이고 학습 시작만 바뀐다.

무엇을
------
  1. 폴드 CSV 의 train 만 date >= 1981-01-05 로 잘라 새 폴드 이름 {fold}1981 로 저장
     (val/test 는 복사).  폴드 이름을 바꾸는 이유: rawstd_preprocess_fold 가 out_dir 에
     {fold}_{split}_rawvol.csv 를 폴드 이름으로만 쓰므로, 같은 이름이면 본모형 전처리 파일을
     덮어쓴다 (rawvol_helpers.py:72).
  2. 본모형과 같은 spec(run_full_fpath.LOCKED, fpath_novol d2)으로 4 폴드 × 시드 R81_SEEDS
     (기본 2026 하나) 학습.  태그 rvAbl_full_fpath_novol_from1981_d2_s{seed}.
  3. 시험 지표(CRPS, cov80, cov95, 생성 왜도)를 전 구간 학습본
     (rvAbl_full_fpath_novol_d2, 같은 시드 + 5 시드 평균±sd)과 나란히 출력, result/robust_from1981.json 저장.

사용 (Colab)
------------
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/run_from1981.py
  시드를 늘리려면 R81_SEEDS=2026,2027,2028,2029,2030
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

import run_full_fpath as R                               # noqa: E402
import train_garch_flow as T                             # noqa: E402
from train_garch_flow import main_worker                 # noqa: E402

CUT = pd.Timestamp(os.environ.get("R81_CUT", "1981-01-05"))
SEEDS = [int(s) for s in os.environ.get("R81_SEEDS", "2026").split(",") if s.strip()]
FOLDS = R.FOLDS
DIM = int(os.environ["FPATH_DIM"])
SUFFIX = "1981"
BASE_TAG = f"rvAbl_full_fpath_novol_d{DIM}"
NEW_TAG = f"rvAbl_full_fpath_novol_from1981_d{DIM}"
OUT = os.path.join(R.RESULT_DIR, "robust_from1981.json")
KEYS = [("crps_pooled", "CRPS"), ("coverage_80", "cov80"), ("coverage_95", "cov95"), ("skew_sim", "skew")]
LABEL = {"F_gfc": "Financial crisis (2006-2010)", "F_long_A": "Recovery (2011-2015)",
         "F_long_B_origin": "COVID (2016-2020)", "F_long": "Tightening (2021-2025)"}


def make_folds():
    """train 을 CUT 이후로 자른 {fold}1981_{split}.csv 를 같은 폴드 디렉터리에 만든다."""
    for fold in FOLDS:
        for split in ("train", "val", "test"):
            src = os.path.join(R.FOLDS_DIR, f"{fold}_{split}.csv")
            dst = os.path.join(R.FOLDS_DIR, f"{fold}{SUFFIX}_{split}.csv")
            if os.path.exists(dst):
                continue
            if split == "train":
                df = pd.read_csv(src, parse_dates=["date"])
                n0 = len(df)
                df = df[df["date"] >= CUT].reset_index(drop=True)
                df.to_csv(dst, index=False)
                print(f"  {fold}{SUFFIX}_train: {n0} → {len(df)} rows  ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
            else:
                shutil.copy2(src, dst)


def train_all():
    R.set_cond_cols(R.ENC_FULL, "tbill_wr")
    T.MASK_FUTURE_TBILL = False
    T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
    T.ENCODER_MASK_SP = False
    for seed in SEEDS:
        tag = f"{NEW_TAG}_s{seed}"
        for fold in FOLDS:
            f81 = f"{fold}{SUFFIX}"
            summ = os.path.join(R.RESULT_DIR, f"garch_flow_ar_{tag}_{f81}_summary.json")
            if os.path.exists(summ):
                print(f"  [skip] {os.path.basename(summ)}"); continue
            spec = dict(R.LOCKED); spec.update(fold=f81, seed=seed, tag=tag, folds_dir=R.FOLDS_DIR)
            print(f"  [run] {tag} fold={f81}"); t0 = time.time()
            try:
                main_worker(spec); print(f"    done ({time.time() - t0:.0f}s)")
            except Exception as e:                       # noqa: BLE001
                print(f"    [FAIL] {tag} {f81}: {e!r}")


def _test_eval(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    return d.get("test_eval")


def compare():
    print("\n" + "#" * 108)
    print(f"# 학습 시작 1971 (전 구간) vs {CUT.date()} 이후 — 같은 spec, 시험 폴드 동일.  값은 시험셋 전수 원점 기준.")
    print("#" * 108)
    out = {}
    hdr = f"{'Test period':<30}{'metric':<7}{'full s2026':>12}{'full 5-seed':>18}{'from1981':>12}{'Δ(1981−full)':>14}"
    print(hdr); print("-" * len(hdr))
    for fold in FOLDS:
        base_same = _test_eval(os.path.join(R.RESULT_DIR, f"garch_flow_ar_{BASE_TAG}_s{SEEDS[0]}_{fold}_summary.json"))
        base_all = [m for m in (_test_eval(p) for p in sorted(glob.glob(
            os.path.join(R.RESULT_DIR, f"garch_flow_ar_{BASE_TAG}_s*_{fold}_summary.json")))) if m]
        new = [m for m in (_test_eval(os.path.join(R.RESULT_DIR, f"garch_flow_ar_{NEW_TAG}_s{s}_{fold}{SUFFIX}_summary.json"))
                           for s in SEEDS) if m]
        if not new:
            print(f"{LABEL[fold]:<30}(from1981 결과 없음)"); continue
        row = {}
        first = True
        for k, name in KEYS:
            b1 = base_same.get(k) if base_same else None
            ba = [m[k] for m in base_all if k in m]
            nv = [m[k] for m in new if k in m]
            n_mean = float(np.mean(nv))
            ba_txt = f"{np.mean(ba):.4f}±{np.std(ba, ddof=1):.4f}" if len(ba) > 1 else (f"{ba[0]:.4f}" if ba else "—")
            d = (n_mean - float(np.mean(ba))) if ba else float("nan")
            print(f"{LABEL[fold] if first else '':<30}{name:<7}{(b1 if b1 is not None else float('nan')):>12.4f}"
                  f"{ba_txt:>18}{n_mean:>12.4f}{d:>+14.4f}")
            first = False
            row[name] = dict(full_same_seed=b1, full_5seed_mean=(float(np.mean(ba)) if ba else None),
                             full_5seed_sd=(float(np.std(ba, ddof=1)) if len(ba) > 1 else None),
                             from1981=n_mean, n_seed_1981=len(nv))
        if new and "skew_actual" in new[0]:
            print(f"{'':<30}{'(실측 skew':<7}{new[0]['skew_actual']:>12.4f})")
        out[fold] = row
        print("-" * len(hdr))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"saved {OUT}")
    print("Δ 가 전 구간 5 시드 sd 안이면 1981 이전 보간 M2 는 결과를 바꾸지 않은 것이다.")


if __name__ == "__main__":
    print(f"[run_from1981] cut={CUT.date()} seeds={SEEDS} folds_dir={R.FOLDS_DIR}")
    make_folds()
    train_all()
    compare()
