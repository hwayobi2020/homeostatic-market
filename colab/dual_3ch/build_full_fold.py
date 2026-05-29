"""전 기간 통합 데이터셋 'full' 빌드 — 시나리오 분석 전용 단일 모델용.

시나리오 분석(9-grid counterfactual)은 OOS 예측 검증이 아니라 조건부 구조 탐색이므로
walk-forward fold(train/test 시간분할)가 맞지 않는다.  대신 전 기간을 통합한 단일
데이터셋 'full' 로 모델 하나를 학습해 거시 시나리오에 어떻게 반응하는지 본다.

F_long fold(가장 긴 시계열: train 1971~2020 + val + test 2021~25)를 합쳐 전 기간
시계열로 만들고 full_train/val/test 로 저장한다.
  - full_train : 앞 구간 (GARCH fit + 모델 학습)
  - full_val   : 끝 N_VAL 주 (early stop 용)
  - full_test  : full_val 복사 (main_worker 통과용 — 시나리오 분석은 OOS 아님)

OOS 성능 검증은 기존 walk-forward fold 로 따로 (목적 분리).

Usage (Colab): !python colab/dual_3ch/build_full_fold.py
"""
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")

SRC_FOLD = "F_long"      # 가장 긴 fold (전 기간 1971~2025 포함)
N_VAL = 260              # early stop 용 끝 구간 (기존 fold val 크기와 동일)


def main():
    parts = []
    for sp in ("train", "val", "test"):
        p = os.path.join(FOLDS_DIR, f"{SRC_FOLD}_{sp}.csv")
        if not os.path.exists(p):
            sys.exit(f"[FATAL] missing source csv: {p}")
        parts.append(pd.read_csv(p, parse_dates=["date"]))

    df = (pd.concat(parts, ignore_index=True)
            .drop_duplicates("date").sort_values("date").reset_index(drop=True))
    n = len(df)
    if n <= N_VAL + 65:
        sys.exit(f"[FATAL] full 데이터가 너무 짧음: n={n}")

    train = df.iloc[:n - N_VAL].copy()
    val = df.iloc[n - N_VAL:].copy()

    out = {}
    for name, d in [("train", train), ("val", val), ("test", val)]:
        p = os.path.join(FOLDS_DIR, f"full_{name}.csv")
        d.to_csv(p, index=False)
        out[name] = p

    print(f"[full] n={n} 주  ({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")
    print(f"  full_train : {len(train)} 주  "
          f"({train['date'].iloc[0].date()} ~ {train['date'].iloc[-1].date()})")
    print(f"  full_val   : {len(val)} 주  "
          f"({val['date'].iloc[0].date()} ~ {val['date'].iloc[-1].date()})")
    print(f"  full_test  : full_val 복사 (시나리오 분석은 OOS 아님 — main_worker 통과용)")
    print(f"  saved: {[os.path.basename(v) for v in out.values()]}")
    print(f"  컬럼 수 = {len(df.columns)} (원본 fold csv 컬럼 그대로 보존)")


if __name__ == "__main__":
    main()
