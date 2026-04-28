"""Dual-Mamba 데이터 로더 — weekly_ppbond_{train,test}.csv → 윈도우 텐서.

윈도우 정의 (P=52, F=52, L=104, W=26):
    valid 윈도우 시작 index i ∈ [W-1, T-L]   (cumulative 계산 위해 i ≥ 25 필요)
    past   range : [i, i+P)
    future range : [i+P, i+P+F)
    past_buffer  : [i+P-W+1, i+P)            = 마지막 25w of past
    past_cum     : 26w 누적, t ∈ [i, i+P) 시점에서 끝나는 sum

표준화:
    7개 컬럼 z-score (train 통계로 train+test 모두 적용):
        sp_return, m2_growth, m2v, cpi_yoy, vix, tbill_wr, excess_liq_yoy
    누적 채널 (past_cum, future_cum 의 raw input) 도 같은 standardized 값 사용.

학습 batch 텐서:
    past_cond            [N, 52, 6]    sp + 5 macros (z-scored)
    future_tbill         [N, 52, 1]    tbill_wr scenario (z-scored)
    past_buffer_raw      [N, 25, 2]    [tbill_wr, excess_liq_yoy] (z-scored, Bridge input)
    past_excess_liq_obs  [N, 52, 1]    excess_liq_yoy (z-scored)  ← Stage 1 target past
    future_excess_liq_obs[N, 52, 1]    excess_liq_yoy (z-scored)  ← Stage 1 target future
    past_sp_obs          [N, 52, 1]    sp_return (z-scored)       ← Stage 2 target past
    future_sp_obs        [N, 52, 1]    sp_return (z-scored)       ← Stage 2 target future
    past_cum_observed    [N, 52, 2]    [tbill_26w, excess_liq_26w] (z-scored 합)
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Windows MKL DLL 충돌 방지: torch 를 numpy/pandas 보다 먼저 import
import torch
from torch.utils.data import TensorDataset, DataLoader

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


# 컬럼 정의
CHANNELS_PAST_COND  = ["sp_return", "m2_growth", "m2v", "cpi_yoy", "vix", "tbill_wr"]   # 6채널 (Stage 1 cond)
CHANNELS_MACRO4     = ["m2_growth", "m2v", "cpi_yoy", "vix"]                            # Stage 2 cond (4 macro)
CHANNELS_RAW_2      = ["tbill_wr", "excess_liq_yoy"]                                    # Bridge 입력 + cumulative
COL_TARGET_X1       = "excess_liq_yoy"                                                  # Stage 1 target
COL_TARGET_X2       = "sp_return"                                                       # Stage 2 target

# 윈도우 default
P_DEFAULT = 52
F_DEFAULT = 52
W_DEFAULT = 26


# ══════════════════════════════════════════════════════════════════
# 1. 통계 계산 (train 만으로)
# ══════════════════════════════════════════════════════════════════

def compute_stats(df_train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """train DataFrame 에서 z-score 통계 계산.

    Return:
        stats : {col: {'mean': float, 'std': float}}
    """
    stats: Dict[str, Dict[str, float]] = {}
    cols = sorted(set(CHANNELS_PAST_COND + CHANNELS_RAW_2 + [COL_TARGET_X1, COL_TARGET_X2]))
    for col in cols:
        arr = df_train[col].values.astype(np.float64)
        mu  = float(arr.mean())
        sd  = float(arr.std() + 1e-8)
        stats[col] = {"mean": mu, "std": sd}
    return stats


def save_stats(stats: dict, path: str | Path):
    Path(path).write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_stats(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════
# 2. 윈도우 빌더
# ══════════════════════════════════════════════════════════════════

def _zscore(arr: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return ((arr - mu) / sd).astype(np.float32)


def build_windows(df: pd.DataFrame,
                  stats: dict,
                  P: int = P_DEFAULT,
                  F: int = F_DEFAULT,
                  W: int = W_DEFAULT,
                  return_dates: bool = False) -> Dict[str, torch.Tensor]:
    """DataFrame → 윈도우 텐서 dict.

    Args:
        df    : weekly_ppbond_{train,test}.csv loaded DataFrame (sorted by date)
        stats : compute_stats() 출력 (train 통계)
        P, F  : past / future 길이
        W     : 누적 윈도우 (default 26)

    Return:
        dict of tensors (모두 [N, ..., ...] float32)
        + (옵션) "dates" : list[str], 각 윈도우의 future-start 날짜
    """
    T = len(df)
    L = P + F
    Wm1 = W - 1

    # 표준화 시계열 (raw 컬럼만)
    arrs: Dict[str, np.ndarray] = {}
    for col in sorted(set(CHANNELS_PAST_COND + CHANNELS_RAW_2 + [COL_TARGET_X1, COL_TARGET_X2])):
        s = stats[col]
        arrs[col] = _zscore(df[col].values.astype(np.float64), s["mean"], s["std"])

    # 26w cumulative — rolling MEAN (sum/W) — z-scored 입력 std ~ 1 안정 스케일
    # (sum 으로 하면 std=25 폭주, autocorr 강해서 sqrt(W) 보정 안 들어감)
    cum_arrs: Dict[str, np.ndarray] = {}
    kernel = np.ones(W, dtype=np.float32) / float(W)   # mean kernel
    for col in CHANNELS_RAW_2:
        # np.convolve mode='valid' → length T - W + 1
        # cum_arrs[col][k] = mean( arrs[col][k : k+W] ),  대응 시점 t = k + W - 1
        cum_arrs[col] = np.convolve(arrs[col], kernel, mode="valid").astype(np.float32)
        assert len(cum_arrs[col]) == T - W + 1

    # valid 윈도우 시작 index
    valid_i = list(range(Wm1, T - L + 1))
    n_w = len(valid_i)
    if n_w <= 0:
        raise ValueError(f"데이터 부족: T={T}, L={L}, Wm1={Wm1}")

    # 출력 텐서 사전 할당
    out_np = {
        "past_cond"            : np.zeros((n_w, P, len(CHANNELS_PAST_COND)), dtype=np.float32),
        "past_macro4"          : np.zeros((n_w, P, len(CHANNELS_MACRO4)),     dtype=np.float32),
        "future_tbill"         : np.zeros((n_w, F, 1), dtype=np.float32),
        "future_macro4"        : np.zeros((n_w, F, len(CHANNELS_MACRO4)),     dtype=np.float32),
        "past_buffer_raw"      : np.zeros((n_w, Wm1, len(CHANNELS_RAW_2)), dtype=np.float32),
        "past_excess_liq_obs"  : np.zeros((n_w, P, 1), dtype=np.float32),
        "future_excess_liq_obs": np.zeros((n_w, F, 1), dtype=np.float32),
        "past_sp_obs"          : np.zeros((n_w, P, 1), dtype=np.float32),
        "future_sp_obs"        : np.zeros((n_w, F, 1), dtype=np.float32),
        "past_cum_observed"    : np.zeros((n_w, P, len(CHANNELS_RAW_2)), dtype=np.float32),
    }
    dates: list = []

    for w_idx, i in enumerate(valid_i):
        past_s, past_e   = i, i + P                     # 과거 [past_s, past_e)
        future_s, future_e = past_e, past_e + F         # 미래 [future_s, future_e)
        buf_s            = past_e - Wm1                 # buffer [buf_s, past_e)

        # past_cond [P, 6]
        for c_idx, col in enumerate(CHANNELS_PAST_COND):
            out_np["past_cond"][w_idx, :, c_idx] = arrs[col][past_s:past_e]

        # past_macro4 [P, 4] (Stage 2 condition macro 부분)
        for c_idx, col in enumerate(CHANNELS_MACRO4):
            out_np["past_macro4"][w_idx, :, c_idx] = arrs[col][past_s:past_e]

        # future_tbill [F, 1]
        out_np["future_tbill"][w_idx, :, 0] = arrs["tbill_wr"][future_s:future_e]

        # future_macro4 [F, 4] (Stage 2 oracle 모드용 — 미래 macro 관측 정답)
        for c_idx, col in enumerate(CHANNELS_MACRO4):
            out_np["future_macro4"][w_idx, :, c_idx] = arrs[col][future_s:future_e]

        # past_buffer_raw [Wm1, 2]
        for c_idx, col in enumerate(CHANNELS_RAW_2):
            out_np["past_buffer_raw"][w_idx, :, c_idx] = arrs[col][buf_s:past_e]

        # past / future targets [P or F, 1]
        out_np["past_excess_liq_obs"  ][w_idx, :, 0] = arrs[COL_TARGET_X1][past_s:past_e]
        out_np["future_excess_liq_obs"][w_idx, :, 0] = arrs[COL_TARGET_X1][future_s:future_e]
        out_np["past_sp_obs"          ][w_idx, :, 0] = arrs[COL_TARGET_X2][past_s:past_e]
        out_np["future_sp_obs"        ][w_idx, :, 0] = arrs[COL_TARGET_X2][future_s:future_e]

        # past_cum_observed [P, 2]
        # cum_arrs[col][k] = rolling MEAN at t = k + Wm1 (z-scored 입력 평균, std ~ 1).
        # 우리 원하는 t ∈ [past_s, past_e), k = t - Wm1 ∈ [past_s - Wm1, past_e - Wm1).
        k_s = past_s - Wm1
        k_e = past_e - Wm1
        for c_idx, col in enumerate(CHANNELS_RAW_2):
            out_np["past_cum_observed"][w_idx, :, c_idx] = cum_arrs[col][k_s:k_e]

        if return_dates:
            dates.append(str(df["date"].iloc[future_s]))

    out: Dict[str, torch.Tensor] = {k: torch.from_numpy(v) for k, v in out_np.items()}
    if return_dates:
        out["dates"] = dates
    return out


# ══════════════════════════════════════════════════════════════════
# 3. DataLoader 헬퍼 — TensorDataset + DataLoader
# ══════════════════════════════════════════════════════════════════

WINDOW_KEYS = [
    "past_cond", "past_macro4", "future_tbill", "future_macro4",
    "past_buffer_raw",
    "past_excess_liq_obs", "future_excess_liq_obs",
    "past_sp_obs", "future_sp_obs", "past_cum_observed",
]


def build_dataloader(window_dict: Dict[str, torch.Tensor],
                     batch_size: int = 32,
                     shuffle: bool = True,
                     drop_last: bool = False,
                     num_workers: int = 0) -> DataLoader:
    """윈도우 dict → DataLoader.

    Iterator 가 batch dict 를 yield 하도록 collate_fn 사용.
    """
    tensors = [window_dict[k] for k in WINDOW_KEYS]
    ds = TensorDataset(*tensors)

    def _collate(batch_list):
        # batch_list : list of tuples (length WINDOW_KEYS)
        out = {}
        for i, k in enumerate(WINDOW_KEYS):
            out[k] = torch.stack([item[i] for item in batch_list], dim=0)
        return out

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers,
                      collate_fn=_collate)


def time_split_train_val(window_dict: Dict[str, torch.Tensor],
                          val_fraction: float = 0.15) -> tuple[Dict, Dict]:
    """시간순 마지막 val_fraction 을 validation 으로 분리."""
    n = window_dict["past_cond"].shape[0]
    n_val = max(int(n * val_fraction), 1)
    tr = {k: v[:-n_val] for k, v in window_dict.items() if torch.is_tensor(v)}
    va = {k: v[-n_val:] for k, v in window_dict.items() if torch.is_tensor(v)}
    return tr, va


# ══════════════════════════════════════════════════════════════════
# 4. CLI / smoke
# ══════════════════════════════════════════════════════════════════

def _smoke():
    here = Path(__file__).resolve().parent
    train_csv = here / "data" / "weekly_ppbond_train.csv"
    test_csv  = here / "data" / "weekly_ppbond_test.csv"

    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)
    print(f"train rows: {len(df_tr)}  ({df_tr['date'].iloc[0]} ~ {df_tr['date'].iloc[-1]})")
    print(f"test  rows: {len(df_te)}  ({df_te['date'].iloc[0]} ~ {df_te['date'].iloc[-1]})")

    stats = compute_stats(df_tr)
    print(f"\nStats (train):")
    for col, s in stats.items():
        print(f"  {col:20s}: mean={s['mean']:+.6f}, std={s['std']:.6f}")

    win_tr = build_windows(df_tr, stats, return_dates=True)
    win_te = build_windows(df_te, stats, return_dates=True)
    print(f"\nTrain windows: {win_tr['past_cond'].shape[0]}, "
          f"first future_start = {win_tr['dates'][0]}, last = {win_tr['dates'][-1]}")
    print(f"Test  windows: {win_te['past_cond'].shape[0]}, "
          f"first future_start = {win_te['dates'][0]}, last = {win_te['dates'][-1]}")

    print(f"\nShape check:")
    for k in WINDOW_KEYS:
        v = win_tr[k]
        print(f"  {k:25s}: {tuple(v.shape)}, mean={v.mean().item():+.4f}, std={v.std().item():.4f}")

    # 일관성 검증: past_cum_observed[w, P-1] 와 그 다음 위치 (Bridge 한 번 시점) 비교
    print(f"\nConsistency check (past_cum vs Bridge output, t=past_end-1 vs future_start):")
    from dual_mamba import CumulativeBridge
    bridge = CumulativeBridge(window=W_DEFAULT)
    # 한 윈도우 만 — Bridge 의 future_seq 입력에 future raw (관측, teacher_forcing 와 같음)
    # Bridge 는 future raw 의 cum 을 output, t=0 위치는 "past 25 + future 0" 의 26w sum
    # 직전 시점 past_cum_observed[w, P-1] = i+P-1 시점 26w sum
    w = 0
    bridge_out = bridge(
        win_tr["past_buffer_raw"][w:w+1],
        torch.cat([win_tr["future_tbill"][w:w+1],
                   win_tr["future_excess_liq_obs"][w:w+1]], dim=-1),
    )                                                  # [1, 52, 2]
    print(f"  past_cum_observed[w, P-1, 0] (t=past_end-1) : {win_tr['past_cum_observed'][w, -1, 0].item():+.4f}")
    print(f"  bridge_out[w, 0, 0]          (t=past_end)   : {bridge_out[0, 0, 0].item():+.4f}")
    print(f"    (둘이 비슷한 스케일이면 OK — t는 1 step 차이)")

    # DataLoader smoke
    loader = build_dataloader(win_tr, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"\nBatch dict (B=4):")
    for k in WINDOW_KEYS:
        print(f"  {k:25s}: {tuple(batch[k].shape)}")


if __name__ == "__main__":
    _smoke()
