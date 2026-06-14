"""실현(actual) 13주 수익률 통계 — fold별 pooled (모델 skew/CVaR과 동일 정의).

ablation/경로 표의 *모델* skew·CVaR이 현실적인지 판단할 ground-truth.
sim_metrics(=모델)는 raw 수익률 pooled(n_orig×n_sim×T)에 _skew/_exkurt/compute_cvar 적용.
여기선 *실현* raw 수익률(= sp_return(표준화 z) × garch_sigma + garch_mu 로 복원)을 같은 origin-window(미래 13주)로 pool해 동일 함수 적용.
주의: 실현은 rolling σ(실제 변동성), 모델 sim_raw는 origin-frozen σ — skew 비교는 유효(분포 모양), CVaR 절대크기 비교는 σ구조 차이 감안.
  · skew/exkurt: scale-invariant → 모델과 정의 동일.
  · cvar1: compute_cvar(pooled returns, 0.01) — 모델 CVaR1%와 동일.
  · uw_mean/uw_cvar1: 누적-진입대비 최저(underwater) — sim_metrics와 동일 식. (uw_cvar1은 n_orig 1%라 노이즈)
모델 추론·rescale 불필요 (raw 실현수익률 직접) → 빠름. ckpt는 cond_stats(윈도잉용)만 사용.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/realized_return_stats.py
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
sys.path.insert(0, HERE)

import analyze_pathshape_rawvol as PS                                   # noqa: E402

LABELS = {"F_gfc": "금융위기", "F_long_A": "회복기",
          "F_long_B_origin": "코로나", "F_long": "긴축기"}


def realized_stats(fold):
    seed = PS.SEEDS[0]
    bp = os.path.join(PS.RESULT_DIR, f"garch_flow_ar_{PS.TAG_PREFIX}_s{seed}_{fold}_best.pt")
    if not os.path.exists(bp):
        print(f"[missing best.pt] {os.path.basename(bp)} — cond_stats 로드 불가"); return None
    ckpt = torch.load(bp, map_location="cpu")
    cond_stats = ckpt["meta"]["cond_stats"]

    gp = PS.garch_preprocess_fold(PS.FOLDS_DIR, fold, PS.RESULT_DIR)
    origin_csv = gp["test"]
    valid_mask, z_te, df_te = PS.compute_valid_mask(origin_csv, cond_stats)

    # ★ df_te["sp_return"]은 *표준화 잔차 z* (sim_metrics가 _gz로 사용). raw 수익률로 복원:
    #   ε = z × σ,  raw r = ε + μ  →  r = sp_return × garch_sigma + garch_mu
    sp_z = df_te["sp_return"].to_numpy(float)
    sig = df_te["garch_sigma"].to_numpy(float)               # σ = rolling sp_std_13w (실현 변동성)
    mu_c = float(df_te["garch_mu"].iloc[0])
    sp = sp_z * sig + mu_c                                    # 실제 raw 주간수익률 (return units)
    n_w = z_te.shape[0] - PS.PAST_LEN - PS.FUTURE_LEN + 1
    real = np.stack([sp[w + PS.PAST_LEN: w + PS.PAST_LEN + PS.FUTURE_LEN]
                     for w in range(n_w)])[valid_mask]        # (n_orig, FUTURE_LEN) 미래 13주 실현
    n_orig0 = real.shape[0]
    if n_orig0 > PS.N_ORIGIN_MAX:                             # 모델과 동일 subsample
        idx = np.linspace(0, n_orig0 - 1, PS.N_ORIGIN_MAX).astype(int)
        real = real[idx]
    n_orig = real.shape[0]

    f = real.ravel()
    f = f[~np.isnan(f)]
    cum = np.cumsum(real, axis=1)
    cum0 = np.concatenate([np.zeros((real.shape[0], 1)), cum], axis=1)
    underw = cum0.min(axis=1)                                 # intra-horizon loss (진입 대비)

    return dict(n_orig=n_orig, n_obs=len(f),
                skew=PS._skew(f), exkurt=PS._exkurt(f),
                cvar1=PS.compute_cvar(f, 0.01),
                uw_mean=float(underw.mean()),
                uw_cvar1=PS.compute_cvar(underw, 0.01))


def main():
    print("#" * 100)
    print("# 실현(actual) 13주 수익률 통계 — fold별 pooled (모델 sim_metrics와 동일 정의)")
    print("#   skew·exkurt = scale-invariant(모델과 동일).  cvar1 = pooled 1% CVaR.  uw_cvar1 = underwater 1%(노이즈)")
    print("#" * 100)
    print(f"{'fold':<18}{'n_orig':>7}{'n_obs':>8}{'skew':>9}{'exkurt':>9}{'cvar1':>10}{'uw_mean':>10}{'uw_cvar1':>10}")
    for fold in PS.FOLDS:
        r = realized_stats(fold)
        if r is None:
            continue
        lbl = f"{LABELS.get(fold, fold)}[{fold}]"
        print(f"{lbl:<18}{r['n_orig']:>7}{r['n_obs']:>8}{r['skew']:>+9.3f}{r['exkurt']:>+9.3f}"
              f"{r['cvar1']:>+10.4f}{r['uw_mean']:>+10.4f}{r['uw_cvar1']:>+10.4f}")
    print("\n[비교법] 위 실현 skew vs 표 4.9의 *모델* skew(full/maskall 등).")
    print("  실현보다 모델이 더 깊으면(더 음수) = 모델 좌꼬리 과대, 얕으면 과소.")


if __name__ == "__main__":
    main()
