"""raw-vol Flow ablation: NF-GARCH 의 GARCH(1,1) 표준화 대신 raw 13w stdev 로 표준화.

기존 train_garch_flow.py 의 garch_preprocess_fold / forward_garch_rescale 를
monkey-patch 하여 GARCH 동학 없이 raw stdev 만으로 z 표준화 + origin-frozen σ
복원으로 교체.  기존 코드 0 수정 — 되돌리려면 본 파일을 import 안 하면 된다.

NF-GARCH 와의 차이:
  - z = (r - μ_train) / sp_std_13w_raw         (GARCH σ_t 대신 raw 13w rolling std)
  - μ_train  = train sp_return 의 단순 평균    (시변 mean 없음, 누수 없음)
  - forward σ = origin frozen                  (13주 내내 origin 시점 raw 13w std 유지)

Usage:
    from rawvol_helpers import patch_rawvol
    patch_rawvol()                              # GARCH 함수 → raw 버전 교체
    from train_garch_flow import main_worker
    main_worker(spec)
"""
import os
import sys

import numpy as np
import pandas as pd


def rawstd_preprocess_fold(folds_dir, fold, out_dir, scale=100.0):
    """GARCH fit 없이 raw 13w stdev 로 z 표준화.  csv 컬럼은 garch_preprocess_fold
    와 동일하게 채워 train_garch_flow 의 역변환 코드(garch_sigma/mu/omega/alpha/beta
    참조)와 시그니처가 호환되게 한다.  raw 모드에선 omega/alpha/beta = 0 (forward
    동학 없음 → origin-frozen σ 효과).
    """
    src = {sp: os.path.join(folds_dir, f"{fold}_{sp}.csv")
           for sp in ("train", "val", "test")}
    for p in src.values():
        if not os.path.exists(p):
            sys.exit(f"[FATAL] rawstd_preprocess: missing csv {p}")
    dfs = {sp: pd.read_csv(p, parse_dates=["date"]) for sp, p in src.items()}

    # train sp_return 평균 (단순 constant mean, train 내부만 보므로 누수 없음).
    mu_train = float(dfs["train"]["sp_return"].astype(float).mean())

    # 전체 series 합쳐서 13주 rolling skew 계산 (date 순, 과거 정보만 보므로 누수 없음).
    # 시작부 NaN(처음 12개)은 0 으로 채움 — "정보 없음" 신호.
    full = (pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
              .drop_duplicates("date").sort_values("date").reset_index(drop=True))
    full["sp_skew_13w"] = (full["sp_return"].astype(float)
                                .rolling(window=13, min_periods=13).skew().fillna(0.0))
    skew_map = dict(zip(full["date"], full["sp_skew_13w"].to_numpy(dtype=float)))

    eps = 1e-12
    out = {}
    os.makedirs(out_dir, exist_ok=True)
    s_med_last = s_min_last = s_max_last = None
    sk_min = sk_med = sk_max = None
    for sp in ("train", "val", "test"):
        d = dfs[sp].copy()
        r = d["sp_return"].astype(float).to_numpy()
        s_raw = d["sp_std_13w"].astype(float).to_numpy()
        if np.any(np.isnan(s_raw)):
            sys.exit(f"[FATAL] rawvol: sp_std_13w NaN in split {sp}")
        sk_raw = d["date"].map(skew_map).to_numpy(dtype=float)         # 13w rolling skew
        if np.any(np.isnan(sk_raw)):
            sys.exit(f"[FATAL] rawvol: sp_skew_13w NaN/미매핑 in split {sp}")
        d["garch_mu"]       = mu_train
        d["garch_sigma"]    = s_raw                       # vol slot = raw 13w std
        d["garch_omega"]    = 0.0                         # raw 모드: forward 동학 없음
        d["garch_alpha"]    = 0.0
        d["garch_beta"]     = 0.0
        d["sp_return"]      = (r - mu_train) / (s_raw + eps)    # z_t (raw-std 표준화)
        d["sp_std_13w"]     = s_raw                       # raw 13w std 유지(덮어쓰기 없음)
        d["sp_skew_13w"]    = sk_raw                      # raw 13w rolling skew (신규)
        d["sp_log_std_13w"] = np.log(s_raw + eps)
        outp = os.path.join(out_dir, f"{fold}_{sp}_rawvol.csv")
        d.to_csv(outp, index=False)
        out[sp] = outp
        s_min_last, s_med_last, s_max_last = (
            float(np.min(s_raw)), float(np.median(s_raw)), float(np.max(s_raw)))
        sk_min, sk_med, sk_max = (
            float(np.min(sk_raw)), float(np.median(sk_raw)), float(np.max(sk_raw)))

    print(f"  [Raw-Vol] no GARCH fit; train mu = {mu_train:+.5f}")
    print(f"  [Raw-Vol] sp_std_13w  (raw, return units) min/median/max = "
          f"{s_min_last:.5f}/{s_med_last:.5f}/{s_max_last:.5f}")
    print(f"  [Raw-Vol] sp_skew_13w (raw 13w rolling)   min/median/max = "
          f"{sk_min:+.3f}/{sk_med:+.3f}/{sk_max:+.3f}  (시작부 12개 NaN→0)")
    return out


def forward_rawvol_rescale(sim_paths_zt, s2_orig, e2_orig, om, al, be, mu):
    """raw-vol 모드 역변환: σ origin-frozen (13주 내내 origin σ 상수).
    forward_garch_rescale 와 시그니처 동일 (drop-in).  GARCH 동학 무시 — 변동성
    클러스터링 가정 없음 (raw stdev 의 본래 단순화)."""
    sigma_orig = np.sqrt(np.maximum(s2_orig, 0.0))            # (n_origins,)
    return sim_paths_zt * sigma_orig[:, None, None] + float(mu)


# ---------------------------------------------------------------------------
# 자기회귀 변동성 역변환 (GARCH 모수를 쓰지 않는다)
# ---------------------------------------------------------------------------
# 학습 표준화는 이미 시점별이다:
#   data/extend_to_1971.py:396  df["sp_std_13w"] = rolling(13).std(ddof=1).shift(1)
# 즉 σ_t 는 r_{t-13..t-1} 만 쓰고 한 주 시프트돼 있어 인과적이다.  그래서 생성
# 경로를 그대로 이어 붙여 σ 를 갱신할 수 있다 — 원점에서 얼리는 것은 역변환의
# 단순화일 뿐 학습의 제약이 아니다.
#
# 이 함수를 쓰면 지평 안에서 변동성이 움직이되 GARCH(ω, α, β)를 빌리지 않는다.
_ROLL_W = 13


def forward_rollvol_rescale(sim_paths_zt, tail_raw, mu):
    """자기회귀 롤링 변동성 역변환.

    σ_{h} = std(직전 13 주 수익률, ddof=1) 로 매 스텝 갱신한다.  첫 스텝의 창은
    실측 꼬리(원점까지 13 주)이고, 이후로는 그 창이 생성 수익률로 한 칸씩
    밀린다.  학습 때의 표준화와 같은 정의(직전 13 주, 한 주 시프트)다.

    Args:
      sim_paths_zt : (n_orig, n_sim, T)  표준화 수익률 z
      tail_raw     : (n_orig, 13)        원점까지의 실측 수익률 (raw, μ 포함)
      mu           : 상수 평균 (raw 단위)
    Returns: (n_orig, n_sim, T) raw return.
    """
    z = np.asarray(sim_paths_zt, dtype=np.float64)
    n_orig, n_sim, T = z.shape
    tail = np.asarray(tail_raw, dtype=np.float64)
    if tail.shape != (n_orig, _ROLL_W):
        raise ValueError(f"tail_raw shape {tail.shape} != {(n_orig, _ROLL_W)}")
    out = np.empty_like(z)
    for i in range(n_orig):
        win = np.repeat(tail[i][None, :], n_sim, axis=0)       # (n_sim, 13)
        for h in range(T):
            sig = win.std(axis=1, ddof=1)                      # (n_sim,)
            sig = np.maximum(sig, 1e-12)
            r = z[i, :, h] * sig + float(mu)
            out[i, :, h] = r
            win = np.concatenate([win[:, 1:], r[:, None]], axis=1)
    return out


def patch_rawvol():
    """train_garch_flow 의 GARCH 함수들을 raw-vol 버전으로 monkey-patch.

    main_worker 와 evaluate_test 가 모듈 global 로 garch_preprocess_fold /
    forward_garch_rescale 를 참조하므로 monkey-patch 가 작동한다.  기존 파일 0 수정.
    """
    import train_garch_flow as T
    T.garch_preprocess_fold = rawstd_preprocess_fold
    T.forward_garch_rescale = forward_rawvol_rescale
    print("[patch_rawvol] train_garch_flow.garch_preprocess_fold / "
          "forward_garch_rescale → rawstd / origin-frozen σ 로 교체")
