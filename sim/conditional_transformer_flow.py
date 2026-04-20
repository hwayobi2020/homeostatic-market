"""ConditionalTransformerFlow — Phase 1 skeleton (v2, weekly + real data).

구조
----
    Target     X   [B, L, 1]   S&P 500 log-cumulative trajectory (윈도우 내 누적)
    Condition  C   [B, L, 3]   [m2_growth_cum, metab_max_cum, metab_min_cum] (z-score)
    Latent     z   [B, L, 1]   isotropic Gaussian latent

Flow (Masked Autoregressive, causal affine):
    x_t = exp(s_t) * z_t + t_t
    (s_t, t_t) = f( x_{<t}, c_{<=t} )      ← causal shift로 가역성 보장

    forward : x, c  →  z,  log|det J|
    inverse : z, c  →  x                   (시간축 sequential)

Phase 1 범위
----
    - 데이터 v29 minimal : M2 + Stock + Metabolism(max, min)
    - 30 features·일별 계층은 Phase 2/3에서 확장. 뼈대는 slot만 열어둠.

이 파일은 "뼈대 + shape 검증" 전용.
    1) 모델 클래스 정의
    2) CSV → tensor 윈도우 변환 유틸
    3) 스모크 테스트 (dummy shape + 실데이터 로딩 + 가역성)
훈련 루프·log-likelihood loss·시나리오 생성은 후속 파일.
"""

from __future__ import annotations

# torch must be imported BEFORE numpy/pandas on Windows to avoid Intel MKL DLL conflict
# (numpy's MKL libs can pre-empt torch's if numpy loads first)
import torch
import torch.nn as nn

from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# 1. Building blocks
# ═══════════════════════════════════════════════════════════════════

class CausalTransformerBlock(nn.Module):
    """causal self-attention stack.
    [B, L, d_in] → [B, L, d_model], 각 시점이 과거만 본다.
    """

    def __init__(self, d_in: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.0, max_len: int = 512):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        # enable_nested_tensor=False : norm_first=True와 호환 안 됨 (warning 제거용)
        self.core = nn.TransformerEncoder(layer, num_layers=n_layers,
                                          enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.proj_in(x) + self.pos[:, :L]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        return self.core(h, mask=mask, is_causal=True)


class ConditionalTransformerFlow(nn.Module):
    """Single causal affine flow step (skeleton).

    Args:
        d_cond    : condition channel count (Phase 1 = 3)
        d_model   : transformer hidden size
        n_heads   : attention heads
        n_layers  : transformer layers per block
    """

    def __init__(self, d_cond: int = 3, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_cond = d_cond

        # 1) Condition encoder: c[B,L,d_cond] → c_feat[B,L,D]
        self.c_encoder = CausalTransformerBlock(
            d_in=d_cond, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers,
        )

        # 2) Parameter network: (x_shift, c_feat) → (log_scale, translation)
        self.xc_proj = nn.Linear(1 + d_model, d_model)
        self.param_net = CausalTransformerBlock(
            d_in=d_model, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers,
        )
        self.param_head = nn.Linear(d_model, 2)

    # ── internal ──
    def _params_from(self, x_shift: torch.Tensor, c_feat: torch.Tensor):
        h = torch.cat([x_shift, c_feat], dim=-1)       # [B, L, 1+D]
        h = self.xc_proj(h)
        h = self.param_net(h)
        p = self.param_head(h)                          # [B, L, 2]
        log_scale, translation = p.chunk(2, dim=-1)     # each [B, L, 1]
        # |s| ≤ 4 : exp(s) ∈ [0.018, 54.6] 표현 가능.
        # 기존 ±2는 Stock log-cum의 시점별 표준편차(≈0.15) 맞추기에 타이트.
        log_scale = torch.tanh(log_scale) * 4.0
        return log_scale, translation

    @staticmethod
    def _right_shift(x: torch.Tensor) -> torch.Tensor:
        x_shift = torch.zeros_like(x)
        x_shift[:, 1:, :] = x[:, :-1, :]
        return x_shift

    # ── forward (학습 방향) ──
    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        x : [B, L, 1]
        c : [B, L, d_cond]
        return:
            z           : [B, L, 1]
            log_det_J   : [B]       (시점 합산, forward 기준)
            log_scale_t : [B, L]    (시점별 s_t — past-mask NLL 계산용)
        """
        c_feat = self.c_encoder(c)
        x_shift = self._right_shift(x)
        log_scale, translation = self._params_from(x_shift, c_feat)
        z = (x - translation) * torch.exp(-log_scale)
        log_scale_t = log_scale.squeeze(-1)                  # [B, L]
        log_det_J = log_scale_t.sum(dim=1)                   # [B]
        return z, log_det_J, log_scale_t

    # ── inverse (생성 방향) ──
    @torch.no_grad()
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        z : [B, L, 1]
        c : [B, L, d_cond]
        return x [B, L, 1]   — causal AR이라 시간축 순차 루프.
        """
        B, L, _ = z.shape
        c_feat = self.c_encoder(c)         # 한 번만
        x = torch.zeros_like(z)
        for t in range(L):
            x_shift = self._right_shift(x)
            log_scale, translation = self._params_from(x_shift, c_feat)
            x[:, t:t + 1, :] = (
                z[:, t:t + 1, :] * torch.exp(log_scale[:, t:t + 1, :])
                + translation[:, t:t + 1, :]
            )
        return x


# ═══════════════════════════════════════════════════════════════════
# 1b. Time-reversed wrapper + Multi-step stack
# ═══════════════════════════════════════════════════════════════════

class TimeReversedFlow(nn.Module):
    """시간축 flip → inner flow → 다시 flip.

    Inner flow는 causal(정방향)이라 좌→우만 의존. flip하면 우→좌 의존
    관점에서 정보 섞임 → 같은 구조의 step끼리 다른 시점 관계 학습.
    Jacobian의 log-det는 flip에 invariant이므로 inner의 값 그대로 사용.
    """

    def __init__(self, inner: ConditionalTransformerFlow):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        x_rev = x.flip(dims=[1])
        c_rev = c.flip(dims=[1])
        z_rev, log_det_J, log_scale_rev_t = self.inner(x_rev, c_rev)
        z = z_rev.flip(dims=[1])
        # flipped 시간축에서 계산된 log_scale_t를 원래 시간축으로 복원
        log_scale_t = log_scale_rev_t.flip(dims=[1])         # [B, L]
        return z, log_det_J, log_scale_t

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z_rev = z.flip(dims=[1])
        c_rev = c.flip(dims=[1])
        x_rev = self.inner.inverse(z_rev, c_rev)
        return x_rev.flip(dims=[1])


class MultiStepFlow(nn.Module):
    """K개의 causal affine step을 쌓는다. 홀수 step은 time-reversed.

    forward  : x → y1 → y2 → ... → yK = z       (composition)
    inverse  : z → yK-1 → ... → y0 = x          (역 composition)
    log|det J_total| = Σ_k log|det J_k|

    Args:
        K        : flow step 개수 (단일 step이면 ConditionalTransformerFlow를 직접 쓰면 됨)
        d_cond   : condition channel 수
        d_model  : transformer hidden size
        n_heads  : attention heads (per step)
        n_layers : transformer layers per block (per step)
    """

    def __init__(self, K: int = 3, d_cond: int = 3, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 time_reverse: bool = True):
        super().__init__()
        self.K = K
        self.time_reverse = time_reverse
        steps = []
        for k in range(K):
            inner = ConditionalTransformerFlow(
                d_cond=d_cond, d_model=d_model,
                n_heads=n_heads, n_layers=n_layers,
            )
            if time_reverse and (k % 2 == 1):
                step = TimeReversedFlow(inner)
            else:
                step = inner
            steps.append(step)
        self.steps = nn.ModuleList(steps)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        B, L, _ = x.shape
        log_det_total    = x.new_zeros(B)
        log_scale_t_total = x.new_zeros(B, L)
        for step in self.steps:
            x, ld, ls_t = step(x, c)
            log_det_total    = log_det_total    + ld
            log_scale_t_total = log_scale_t_total + ls_t
        return x, log_det_total, log_scale_t_total   # x is now z

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = z
        for step in reversed(self.steps):
            x = step.inverse(x, c)
        return x


# ═══════════════════════════════════════════════════════════════════
# 1c. Conditional forecast helper (past-fixed, future-generated)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def conditional_generate(model, x_past: torch.Tensor, c: torch.Tensor,
                         L: int, P: int) -> torch.Tensor:
    """
    x_past : [B, P, 1]       관측된 past target (예: past sp_return)
    c      : [B, L, d_cond]  전 구간 조건 (past + future)
    L      : 전체 윈도우 길이
    P      : past 길이.  F = L − P

    Past z swap 기법 (time-reverse와 호환):
        1. x_full = [x_past, 0_padding] 로 forward → z_full
           (z_past 구간은 정확, z_future 구간은 의미 없음 — 곧 버림)
        2. z_future_new ~ N(0, I) 샘플
        3. z_new = [z_full[:P], z_future_new]
        4. inverse(z_new, c) → x_gen
        5. 가역성 덕에 x_gen[:P] ≈ x_past, x_gen[P:] = 새 시나리오

    Returns: x_gen [B, L, 1] — past는 관측 복원, future는 생성
    """
    B = x_past.shape[0]
    F = L - P
    # 1. Past-filled tensor
    x_full = torch.cat([x_past, x_past.new_zeros(B, F, 1)], dim=1)   # [B, L, 1]
    # 2. Forward → z_past 얻기 (z_future는 버림)
    z_full, _, _ = model(x_full, c)
    # 3. Future z 새로 샘플
    z_future = torch.randn(B, F, 1, device=x_past.device, dtype=x_past.dtype)
    z_new = torch.cat([z_full[:, :P, :], z_future], dim=1)           # [B, L, 1]
    # 4. Inverse
    x_gen = model.inverse(z_new, c)
    return x_gen


# ═══════════════════════════════════════════════════════════════════
# 2. Data utilities (CSV → tensor windows) — ALL RAW (no cumulation)
# ═══════════════════════════════════════════════════════════════════
#
# v30 스키마 (MICH 제거, γ 4채널):
#   Target    X : [sp_return]                                  raw weekly log-return
#   Condition C : [m2_growth, tbill_wr,                        raw
#                  metabolism_max, metabolism_min]             raw
#
# cum 변환 제거: 모든 채널을 그 시점의 raw 값(주간 log-diff 또는 rate level)
# 그대로 slide. flow는 각 시점의 "이번 주" 값 시퀀스를 학습.

COL_TARGET = 'sp_return'
COLS_COND  = ['m2_growth', 'tbill_wr', 'metabolism_max', 'metabolism_min']


def level_windows(series: np.ndarray, L: int) -> np.ndarray:
    """
    Raw value series [N] → sliding windows [N-L+1, L], 누적 없음.
    window[s, i] = series[s + i]
    """
    if series.ndim != 1:
        raise ValueError(f'series must be 1D, got shape {series.shape}')
    N = len(series)
    starts = np.arange(N - L + 1)
    idx = starts[:, None] + np.arange(L)[None, :]
    return series[idx]


def load_windows(csv_path: str | Path, L: int = 104,
                 stats: dict | None = None):
    """
    Weekly v30 CSV → (X, C, stats), **모든 채널 raw**.

    Args:
        csv_path : data/weekly_v30_train.csv 등
        L        : 윈도우 길이 (Phase 1 기본 104)
        stats    : condition 표준화 통계 (mean, std). None이면 계산.

    Returns:
        X     : [N_w, L, 1]   raw 주간 log-return, 표준화 안 함
        C     : [N_w, L, 4]   raw 4채널, z-score 표준화 적용
        stats : {'mean': [4], 'std': [4]} — test 시 재사용
    """
    df = pd.read_csv(csv_path)
    needed = [COL_TARGET] + COLS_COND
    df = df.dropna(subset=needed).reset_index(drop=True)
    if len(df) < L:
        raise ValueError(f'Not enough rows: {len(df)} < L={L}')

    # Target X
    sp = df[COL_TARGET].to_numpy(dtype=np.float32)
    X_np = level_windows(sp, L)                              # [N_w, L]

    # Condition C (4채널 raw)
    c_arrs = [level_windows(df[col].to_numpy(dtype=np.float32), L)
              for col in COLS_COND]
    C_raw = np.stack(c_arrs, axis=-1)                         # [N_w, L, 4]

    if stats is None:
        C_flat = C_raw.reshape(-1, C_raw.shape[-1])
        stats = {
            'mean': C_flat.mean(axis=0).astype(np.float32),
            'std':  C_flat.std(axis=0).astype(np.float32) + 1e-6,
        }
    C_norm = (C_raw - stats['mean']) / stats['std']           # [N_w, L, 4]

    X = torch.from_numpy(X_np[..., None]).float()             # [N_w, L, 1]
    C = torch.from_numpy(C_norm).float()                      # [N_w, L, 4]
    return X, C, stats


# ═══════════════════════════════════════════════════════════════════
# 3. Smoke test
# ═══════════════════════════════════════════════════════════════════

def _smoke_dummy():
    print('─' * 70)
    print('  (a) Dummy tensor smoke test')
    print('─' * 70)
    torch.manual_seed(0)
    B, L, d_cond, D = 4, 104, 4, 64
    x = torch.randn(B, L, 1)
    c = torch.randn(B, L, d_cond)
    model = ConditionalTransformerFlow(d_cond=d_cond, d_model=D,
                                       n_heads=4, n_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'    input   x: {tuple(x.shape)},  c: {tuple(c.shape)}')
    print(f'    params  total = {n_params:,}')

    z, logdet, ls_t = model(x, c)
    print(f'    forward z: {tuple(z.shape)},  log_det_J: {tuple(logdet.shape)}, '
          f'log_scale_t: {tuple(ls_t.shape)}')

    x_rec = model.inverse(z, c)
    err = (x - x_rec).abs().max().item()
    print(f'    inverse x_rec: {tuple(x_rec.shape)}')
    print(f'    max |x - x_rec| = {err:.3e}   '
          f'{"PASS" if err < 1e-3 else "FAIL"} (가역성)')
    return err


def _smoke_multistep(K: int = 3):
    print('─' * 70)
    print(f'  (a2) MultiStepFlow (K={K}) dummy smoke test')
    print('─' * 70)
    torch.manual_seed(1)
    B, L, d_cond, D = 4, 104, 4, 64
    x = torch.randn(B, L, 1)
    c = torch.randn(B, L, d_cond)
    # (i) Time-reverse 다양화 포함 — full invertibility만 확인
    model_rev = MultiStepFlow(K=K, d_cond=d_cond, d_model=D,
                              n_heads=4, n_layers=2, time_reverse=True)
    n_params_rev = sum(p.numel() for p in model_rev.parameters())
    print(f'    [time_reverse=True]  K={K}  params = {n_params_rev:,}')
    z, logdet, ls_t = model_rev(x, c)
    x_rec = model_rev.inverse(z, c)
    err_rev = (x - x_rec).abs().max().item()
    print(f'      forward z: {tuple(z.shape)}  ld: {tuple(logdet.shape)}  '
          f'ls_t: {tuple(ls_t.shape)}')
    print(f'      max |x - x_rec| = {err_rev:.3e}   '
          f'{"PASS" if err_rev < 1e-3 else "FAIL"} (full invertibility)')

    # (ii) Pure causal (forecast용) — past 고정 가능성까지 확인
    model_cau = MultiStepFlow(K=K, d_cond=d_cond, d_model=D,
                              n_heads=4, n_layers=2, time_reverse=False)
    n_params_cau = sum(p.numel() for p in model_cau.parameters())
    print(f'    [time_reverse=False] K={K}  params = {n_params_cau:,}')
    z2, _, _ = model_cau(x, c)
    x_rec2 = model_cau.inverse(z2, c)
    err_cau = (x - x_rec2).abs().max().item()
    print(f'      max |x - x_rec| = {err_cau:.3e}   '
          f'{"PASS" if err_cau < 1e-3 else "FAIL"} (full invertibility)')

    P = 52
    x_past = x[:, :P, :].clone()
    x_gen = conditional_generate(model_cau, x_past, c, L=L, P=P)
    past_err = (x_past - x_gen[:, :P, :]).abs().max().item()
    print(f'      conditional_generate past-preservation: '
          f'max |x_past - x_gen[:P]| = {past_err:.3e}   '
          f'{"PASS" if past_err < 1e-3 else "FAIL"}')
    return err_rev


def _smoke_real(train_csv: Path, test_csv: Path, L: int = 104):
    print('─' * 70)
    print('  (b) Real weekly v29 data smoke test')
    print('─' * 70)

    X_tr, C_tr, stats = load_windows(train_csv, L=L)
    X_te, C_te, _     = load_windows(test_csv,  L=L, stats=stats)
    print(f'    train  X: {tuple(X_tr.shape)},  C: {tuple(C_tr.shape)}')
    print(f'    test   X: {tuple(X_te.shape)},  C: {tuple(C_te.shape)}')
    print(f'    cond stats (train)   mean: {stats["mean"]}')
    print(f'                          std: {stats["std"]}')
    print(f'    X range  train: [{X_tr.min():.3f}, {X_tr.max():.3f}]   '
          f'test: [{X_te.min():.3f}, {X_te.max():.3f}]')
    print(f'    C range (normalized) train: [{C_tr.min():.3f}, {C_tr.max():.3f}]   '
          f'test: [{C_te.min():.3f}, {C_te.max():.3f}]')

    # 모델 forward/inverse 실데이터로 가역성 확인 (작은 배치)
    torch.manual_seed(0)
    model = ConditionalTransformerFlow(d_cond=4, d_model=64,
                                       n_heads=4, n_layers=2)
    B = 8
    x = X_tr[:B]
    c = C_tr[:B]
    z, logdet, ls_t = model(x, c)
    x_rec = model.inverse(z, c)
    err = (x - x_rec).abs().max().item()
    print(f'    real-data forward→inverse  max |x - x_rec| = {err:.3e}   '
          f'{"PASS" if err < 1e-3 else "FAIL"}')
    return err


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    repo = here.parent
    train_csv = repo / 'data' / 'weekly_v30_train.csv'
    test_csv  = repo / 'data' / 'weekly_v30_test.csv'

    err_dummy = _smoke_dummy()
    print()
    err_multi = _smoke_multistep(K=3)
    print()
    err_real  = _smoke_real(train_csv, test_csv, L=104)

    print('\n' + '=' * 70)
    print(f'  SUMMARY  dummy err={err_dummy:.3e}  |  real err={err_real:.3e}')
    print('=' * 70)
