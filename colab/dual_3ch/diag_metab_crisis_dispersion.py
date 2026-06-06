"""진단 — 시나리오 metab 경로 진폭(σ0.31 ramp / σ0.50 step)이 *실제 위기*와 비교해 현실적인가.

시나리오 metab 경로 std(z) = SHAPE_STD × (p90−p10 of z_metab).  (ramp=0.31×, step=0.50×)
실제 13주 metab 경로 std(z) = 각 origin의 13주 미래 z_metab 궤적의 std.
→ 리만(2008-09)·COVID(2020-02) 시점의 실제값과 시나리오 진폭을 같은 z-단위로 비교.

cond_stats(=fold train 표준화)로 z 변환.  inference 불필요, 데이터만.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/diag_metab_crisis_dispersion.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_pathshape_rawvol as PS                                # noqa: E402  (patch_rawvol + cond cols)
import train_garch_flow as T                                        # noqa: E402

FOLDS = PS.FOLDS
SEED0 = PS.SEEDS[0]
ENC = PS.ENC_COLS
MI = ENC.index("metab_13w")
FUTURE_LEN = PS.FUTURE_LEN
RAMP_SIG = 0.31     # SHAPE_STD['ramp_up']
STEP_SIG = 0.50     # SHAPE_STD['step_up']

CRISES = {"Lehman(2008-09)": "2008-09-15", "COVID(2020-02)": "2020-02-20"}
CHANNELS = [("tbill_wr", "금리(tbill)"), ("metab_13w", "유동성(metab)")]   # 금리 먼저


def diag_channel(col, label, df, df_tr, cmu, csd, dates):
    """한 채널(금리/유동성)의 시나리오 진폭 vs 실제 위기 진폭 (z-단위) 출력."""
    z_all = (df[col].to_numpy(float) - cmu) / csd
    z_tr = (df_tr[col].to_numpy(float) - cmu) / csd
    rng = float(np.percentile(z_tr, 90) - np.percentile(z_tr, 10))   # 시나리오 진폭 기준(train)
    ramp_std = RAMP_SIG * rng
    step_std = STEP_SIG * rng

    # 실제 13주 경로 std(z): 모든 origin 의 미래 13주 궤적 std
    path_std = np.array([z_all[t:t + FUTURE_LEN].std(ddof=0)
                         for t in range(len(z_all) - FUTURE_LEN + 1)])

    print(f"  --- {label}  (train μ={cmu:.4f} σ={csd:.4f}, p90-p10={rng:.3f})")
    print(f"      [시나리오] 13주 경로 std(z):  ramp(σ0.31)={ramp_std:.3f}   step(σ0.50)={step_std:.3f}")
    print(f"      [실제 전구간] 13주 경로 std(z):  median={np.median(path_std):.3f}  "
          f"p95={np.percentile(path_std,95):.3f}  max={path_std.max():.3f}")
    for name, dstr in CRISES.items():
        d0 = pd.Timestamp(dstr)
        if not (dates.min() <= np.datetime64(d0) <= dates.max()):
            continue
        t = int(np.argmin(np.abs(dates - np.datetime64(d0))))
        t = min(t, len(z_all) - FUTURE_LEN)
        cstd = float(z_all[t:t + FUTURE_LEN].std(ddof=0))
        zlevel = float(z_all[t])
        ratio_step = cstd / step_std if step_std > 1e-9 else float("nan")
        print(f"        {name:<16} @ {pd.Timestamp(dates[t]).date()}: "
              f"13주 std(z)={cstd:.3f}  (level z={zlevel:+.2f})  "
              f"→ step 시나리오의 {ratio_step:.2f}배")


def main():
    print("#" * 96)
    print("# 진단: 시나리오 진폭 vs 실제 위기 13주 변동 (z-단위, 같은 척도) — 금리·유동성")
    print("#" * 96)

    for fold in FOLDS:
        # cond_stats (채널별 표준화) — 체크포인트 meta 에서
        bp = os.path.join(PS.RESULT_DIR, f"garch_flow_ar_{PS.TAG_PREFIX}_s{SEED0}_{fold}_best.pt")
        if not os.path.exists(bp):
            print(f"\n=== {fold}: [missing ckpt] {os.path.basename(bp)}"); continue
        meta = torch.load(bp, map_location="cpu")["meta"]
        cmean = np.asarray(meta["cond_stats"]["mean"], float)
        cstd_all = np.asarray(meta["cond_stats"]["std"], float)

        gp = T.garch_preprocess_fold(PS.FOLDS_DIR, fold, PS.RESULT_DIR)
        df_tr = pd.read_csv(gp["train"], parse_dates=["date"])
        # 전체 구간(train+val+test)에서 위기 탐색
        dfs = []
        for k in ("train", "val", "test"):
            d = pd.read_csv(gp[k], parse_dates=["date"]); d["_sp"] = k; dfs.append(d)
        df = pd.concat(dfs).drop_duplicates("date").sort_values("date").reset_index(drop=True)
        dates = df["date"].to_numpy()

        print(f"\n=== {fold} ===")
        for col, label in CHANNELS:
            ci = ENC.index(col)
            diag_channel(col, label, df, df_tr,
                         float(cmean[ci]), float(cstd_all[ci]), dates)

    print("\n[해석] 실제 위기 std(z) > step(σ0.50) 이면 → 내 시나리오가 위기보다 *약함* (효과는 보수적 하한).")
    print("       실제 위기 std(z) < step(σ0.50) 이면 → 시나리오가 이미 위기 진폭 이상(=충분).")
    print("       level z(위기시 수준)도 |z|>1.3(≈p90) 이면 → p10-p90 span 밖(꼬리/OOD).")


if __name__ == "__main__":
    main()
