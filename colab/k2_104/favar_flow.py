"""FAVAR Flow — Phase 2 Joint Target Conditional Normalizing Flow.

Phase 15 conditional_transformer_flow.py의 일반화:
    - Target 3D joint:  (tbill_wr, mich_wr, sp_return)
    - Condition 1D   :  m2_growth  ("정책 조이스틱")
    - Past 52w 전체 target이 teacher forcing 으로 causal attn에 노출
    - Future 52w 는 target 생성 (조건부 flow 역변환)

구조
----
    Target     X   [B, L, 3]   (i, π, sp)
    Condition  C   [B, L, 1]   (m2_growth)
    Latent     z   [B, L, 3]   isotropic Gaussian

Flow (Masked Autoregressive, causal affine):
    x_t(d) = exp(s_t^d) * z_t^d + t_t^d                for d=1..3
    (s_t^d, t_t^d) = f_d(x_{<t}, c_{<=t})              ← causal (시간축)
                                                          ※ 동일 t 내에서 d 사이는
                                                            independent given past.
                                                            Cross-sectional dependency
                                                            는 K step stacking 으로
                                                            학습.

MultiStepFlow(K=3, time_reverse=False) 로 감싸서 causal-only stack.
time_reverse=False 이유: past-fixing (conditional_generate)을 가능하게 하려면 causal.

데이터 레이어
------------
    weekly_v31 기준.
    Target columns : sp_return, tbill_wr, mich_wr
        → 반환 텐서 순서 고정 [tbill_wr, mich_wr, sp_return]  (i, π, sp)
    Condition columns: m2_growth

    Raw (누적 안 함). Condition z-score (훈련 통계 저장, test에 재사용).

이 파일이 하는 것
-----------------
    1. FAVARFlow  : 3D target causal affine flow step
    2. MultiStepFAVARFlow : K step stack, optional time-reverse
    3. conditional_generate_favar : past-z swap (past=4D state을 통해 3D target과 1D cond)
    4. load_windows_favar : v31 CSV → (X, C, stats) 텐서
    5. smoke test : 가역성 확인

PINN / 학습 루프 / 평가는 별도 파일 (train_favar_phase2.py 등)에서.
"""

from __future__ import annotations

# torch must be imported BEFORE numpy/pandas on Windows (MKL DLL)
import torch
import torch.nn as nn

from pathlib import Path

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# 1. Building blocks
# ══════════════════════════════════════════════════════════════════

class WaveletActivation(nn.Module):
    """Mexican Hat (Ricker) wavelet activation — compact support 활성화 함수.

    ψ(t) = (1 − t²) · exp(−t²/2),  t = (x − μ)/σ

    특성:
      - Compact support: |t| > 3 에서 ψ ≈ 0, gradient ≈ 0
      - Zero-mean: ∫ψ(t) dt = 0 (편향 차단)
      - Localized: |t| ≤ 1 영역에서만 강한 발화

    목적 (PINN mean shift 차단):
      - 신경망이 페널티 회피 위해 translation 을 +수십 까지 밀려 해도
      - 그 출력이 wavelet 의 |t| > 3 영역으로 가면 ψ ≈ 0 → gradient 차단
      - 모델이 mean shift 추구 의 수학적 동력 자체 사라짐 (compact support)

    학습 가능 파라미터:
      μ (translation): 위기 영역의 위치를 모델이 자동 학습
      log_sigma → σ (scale): 발화 영역의 폭, softplus 로 양수 보장
    """

    def __init__(self, num_features: int, learnable: bool = True):
        super().__init__()
        if learnable:
            self.mu        = nn.Parameter(torch.zeros(num_features))
            self.log_sigma = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer('mu',        torch.zeros(num_features))
            self.register_buffer('log_sigma', torch.zeros(num_features))
        self.learnable = learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.nn.functional.softplus(self.log_sigma) + 1e-3   # 양수 + collapse 방지
        t = (x - self.mu) / sigma
        return (1.0 - t.pow(2)) * torch.exp(-t.pow(2) / 2.0)


class CausalTransformerBlock(nn.Module):
    """causal self-attention stack. [B, L, d_in] → [B, L, d_model]."""

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
        self.core = nn.TransformerEncoder(layer, num_layers=n_layers,
                                          enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.proj_in(x) + self.pos[:, :L]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        return self.core(h, mask=mask, is_causal=True)


class FAVARFlow(nn.Module):
    """Single causal affine flow step on 3D target + 1D condition.

    Args:
        d_cond    : condition channel count (FAVAR = 1)
        d_target  : target channel count (FAVAR = 3)
        d_model   : transformer hidden size
        n_heads   : attention heads
        n_layers  : transformer layers per block
    """

    def __init__(self, d_cond: int = 1, d_target: int = 3,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 use_wavelet: bool = False):
        super().__init__()
        self.d_cond   = d_cond
        self.d_target = d_target
        self.use_wavelet = use_wavelet

        # 1) Condition encoder: c[B,L,d_cond] → c_feat[B,L,D]
        self.c_encoder = CausalTransformerBlock(
            d_in=d_cond, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers,
        )

        # 2) Parameter network: (x_shift, c_feat) → (log_scale, translation)
        #    x_shift ∈ R^{d_target}, c_feat ∈ R^{d_model}
        self.xc_proj = nn.Linear(d_target + d_model, d_model)
        self.param_net = CausalTransformerBlock(
            d_in=d_model, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers,
        )
        # output: 2 * d_target per step (log_scale_d + translation_d for each d)
        self.param_head = nn.Linear(d_model, 2 * d_target)
        # Wavelet activation gate — param_net 와 param_head 사이에 삽입.
        # mean shift 차단 효과 (translation 폭주 시 compact support 가 gradient 0 으로).
        if use_wavelet:
            self.wavelet_gate = WaveletActivation(d_model, learnable=True)

    # ── internal ──
    def _params_from(self, x_shift: torch.Tensor, c_feat: torch.Tensor):
        h = torch.cat([x_shift, c_feat], dim=-1)           # [B, L, d_target + d_model]
        h = self.xc_proj(h)
        h = self.param_net(h)
        if self.use_wavelet:
            h = self.wavelet_gate(h)                        # compact support 게이트
        p = self.param_head(h)                              # [B, L, 2*d_target]
        log_scale, translation = p.chunk(2, dim=-1)         # each [B, L, d_target]
        # log_scale clamp — d_target 차원 모두에 동일하게 tanh*4 적용 (|s|≤4)
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
        x : [B, L, d_target]
        c : [B, L, d_cond]
        return:
            z           : [B, L, d_target]
            log_det_J   : [B]                   (L, d_target 모두 합산)
            log_scale   : [B, L, d_target]      (진단/past-mask loss용)
        """
        c_feat = self.c_encoder(c)
        x_shift = self._right_shift(x)
        log_scale, translation = self._params_from(x_shift, c_feat)
        z = (x - translation) * torch.exp(-log_scale)
        # log|det J_{z→x}| = Σ_t Σ_d s_t^d
        log_det_J = log_scale.sum(dim=(1, 2))                # [B]
        return z, log_det_J, log_scale

    # ── inverse (생성 방향, eval 전용 no_grad) ──
    @torch.no_grad()
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        z : [B, L, d_target]
        c : [B, L, d_cond]
        return x [B, L, d_target]   — causal AR이라 시간축 순차 루프.
        """
        B, L, D = z.shape
        assert D == self.d_target, f'z target dim {D} != model d_target {self.d_target}'
        c_feat = self.c_encoder(c)              # 한 번만 (c 전체)
        x = torch.zeros_like(z)
        for t in range(L):
            x_shift = self._right_shift(x)
            log_scale, translation = self._params_from(x_shift, c_feat)
            x[:, t:t + 1, :] = (
                z[:, t:t + 1, :] * torch.exp(log_scale[:, t:t + 1, :])
                + translation[:, t:t + 1, :]
            )
        return x

    # ── inverse_training (gradient 허용, PINN 학습 전용) ──
    def inverse_training(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        PINN 학습용 differentiable inverse.

        원본 inverse 는 `x[:, t:t+1, :] = ...` 로 in-place 수정 + @torch.no_grad 라서
        gradient 가 막힘. 여기서는:
          - @torch.no_grad 제거 → forward graph 통해 gradient 흐름
          - in-place 대신 **list 누적 + torch.cat** 으로 새 텐서 생성
          - 모든 중간 텐서 `z.new_zeros(...)` 로 device/dtype 자동 일치

        복잡도: O(L) transformer forward + O(L²) cat 비용.
        causal mask 가 각 t 단계에서 "future=zeros" 영향 차단 → 수치적으로 원본 inverse와
        동일한 x 재현 (가역성 smoke test 로 확인).

        z : [B, L, d_target]
        c : [B, L, d_cond]
        return x : [B, L, d_target]
        """
        B, L, D = z.shape
        assert D == self.d_target, f'z target dim {D} != model d_target {self.d_target}'
        c_feat = self.c_encoder(c)
        x_list: list = []
        for t in range(L):
            if t == 0:
                x_so_far = z.new_zeros(B, L, D)
            else:
                past_gen = torch.cat(x_list, dim=1)                 # [B, t, D]
                future_pad = z.new_zeros(B, L - t, D)
                x_so_far = torch.cat([past_gen, future_pad], dim=1)  # [B, L, D]
            x_shift = self._right_shift(x_so_far)
            log_scale, translation = self._params_from(x_shift, c_feat)
            x_t = (
                z[:, t:t + 1, :] * torch.exp(log_scale[:, t:t + 1, :])
                + translation[:, t:t + 1, :]
            )
            x_list.append(x_t)
        return torch.cat(x_list, dim=1)                              # [B, L, D]


# ══════════════════════════════════════════════════════════════════
# 1b. Time-reversed wrapper + Multi-step stack
# ══════════════════════════════════════════════════════════════════

class TimeReversedFAVARFlow(nn.Module):
    """시간축 flip → inner flow → 다시 flip. Jacobian flip-invariant."""

    def __init__(self, inner: FAVARFlow):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        x_rev = x.flip(dims=[1])
        c_rev = c.flip(dims=[1])
        z_rev, log_det_J, log_scale_rev = self.inner(x_rev, c_rev)
        z = z_rev.flip(dims=[1])
        log_scale = log_scale_rev.flip(dims=[1])
        return z, log_det_J, log_scale

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z_rev = z.flip(dims=[1])
        c_rev = c.flip(dims=[1])
        x_rev = self.inner.inverse(z_rev, c_rev)
        return x_rev.flip(dims=[1])

    def inverse_training(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z_rev = z.flip(dims=[1])
        c_rev = c.flip(dims=[1])
        x_rev = self.inner.inverse_training(z_rev, c_rev)
        return x_rev.flip(dims=[1])


class MultiStepFAVARFlow(nn.Module):
    """K개 causal affine step stacking. 홀수 step time-reversed (optional)."""

    def __init__(self, K: int = 3, d_cond: int = 1, d_target: int = 3,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 time_reverse: bool = False, use_wavelet: bool = False):
        super().__init__()
        self.K = K
        self.d_target = d_target
        self.time_reverse = time_reverse
        self.use_wavelet = use_wavelet
        steps = []
        for k in range(K):
            inner = FAVARFlow(
                d_cond=d_cond, d_target=d_target, d_model=d_model,
                n_heads=n_heads, n_layers=n_layers,
                use_wavelet=use_wavelet,
            )
            if time_reverse and (k % 2 == 1):
                step = TimeReversedFAVARFlow(inner)
            else:
                step = inner
            steps.append(step)
        self.steps = nn.ModuleList(steps)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        B, L, D = x.shape
        log_det_total   = x.new_zeros(B)
        log_scale_total = x.new_zeros(B, L, D)
        for step in self.steps:
            x, ld, ls = step(x, c)
            log_det_total   = log_det_total   + ld
            log_scale_total = log_scale_total + ls
        return x, log_det_total, log_scale_total    # x is now z

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = z
        for step in reversed(self.steps):
            x = step.inverse(x, c)
        return x

    def inverse_training(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Differentiable inverse for PINN training (gradient flows through)."""
        x = z
        for step in reversed(self.steps):
            x = step.inverse_training(x, c)
        return x


# ══════════════════════════════════════════════════════════════════
# 1b'. Cross-channel coupling flow (RealNVP style + causal time AR)
# ══════════════════════════════════════════════════════════════════
#
# 기존 FAVARFlow 의 한계:
#   "동일 t 내에서 d 사이는 independent given past" → cross-channel dependency
#   는 K step stacking 으로만 학습. K=2/3 에서 부족 (부호 일치 38-47%).
#
# CrossChannelCausalFlow: RealNVP 식 mask 변환 + 시간축 causal AR.
#   - mask=1 채널: identity (z = x). 한 step 안에서 그대로.
#   - mask=0 채널: affine 변환. (s, t) 는
#       (mask=1 채널의 현재 + mask=0 채널의 past) + condition c 로부터 계산.
#   - 시간축 causal: mask=0 채널의 (s, t) 가 자기 채널 의 과거에만 의존.
#
# K step alternating mask: step 0 mask=[1,0,1,...], step 1 mask=[0,1,0,...]
#   → K=2 만으로도 두 방향 (sp→margin, margin→sp) 모두 학습 가능.
#
# Inverse 는 시간축 sequential (mask=0 채널의 과거가 inverse 결과 참조).
#   GPU/CPU 모두 sequential time loop 비용 있음.

class CrossChannelCausalFlow(nn.Module):
    """RealNVP-style cross-channel coupling + causal time AR.

    Args:
        d_cond, d_target, d_model, n_heads, n_layers : FAVARFlow 와 동일
        mask_pattern : len = d_target. mask=1 채널 = identity, mask=0 채널 = affine 변환.
            None 이면 alternating [(i+1)%2 for i] (D=2 → [1, 0])
    """

    def __init__(self, d_cond: int = 4, d_target: int = 2,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 mask_pattern: list[int] | None = None):
        super().__init__()
        self.d_cond   = d_cond
        self.d_target = d_target
        if mask_pattern is None:
            mask_pattern = [(i + 1) % 2 for i in range(d_target)]
        if len(mask_pattern) != d_target:
            raise ValueError(f'mask_pattern length {len(mask_pattern)} != d_target {d_target}')
        self.register_buffer('mask', torch.tensor(mask_pattern, dtype=torch.float32))

        self.c_encoder = CausalTransformerBlock(
            d_in=d_cond, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        )
        self.xc_proj = nn.Linear(d_target + d_model, d_model)
        self.param_net = CausalTransformerBlock(
            d_in=d_model, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        )
        self.param_head = nn.Linear(d_model, 2 * d_target)

    @staticmethod
    def _right_shift(x: torch.Tensor) -> torch.Tensor:
        x_shift = torch.zeros_like(x)
        x_shift[:, 1:, :] = x[:, :-1, :]
        return x_shift

    def _make_input(self, x: torch.Tensor) -> torch.Tensor:
        """input 만들기: mask=1 채널은 현재값, mask=0 채널은 past (right_shift)."""
        mask_2d = self.mask.view(1, 1, -1)
        x_past = self._right_shift(x)
        return x * mask_2d + x_past * (1.0 - mask_2d)

    def _params_from(self, x_input: torch.Tensor, c_feat: torch.Tensor):
        h = torch.cat([x_input, c_feat], dim=-1)
        h = self.xc_proj(h)
        h = self.param_net(h)
        p = self.param_head(h)                                # [B, L, 2*d_target]
        log_scale, translation = p.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * 4.0
        # mask=1 채널 (identity) 의 (s, t) 0 으로 강제
        mask_2d = self.mask.view(1, 1, -1)
        log_scale   = log_scale   * (1.0 - mask_2d)
        translation = translation * (1.0 - mask_2d)
        return log_scale, translation

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        c_feat = self.c_encoder(c)
        x_input = self._make_input(x)
        log_scale, translation = self._params_from(x_input, c_feat)
        z = (x - translation) * torch.exp(-log_scale)
        log_det_J = log_scale.sum(dim=(1, 2))
        return z, log_det_J, log_scale

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c_feat = self.c_encoder(c)
        B, L, D = z.shape
        mask_2d = self.mask.view(1, 1, -1).to(z.device).to(z.dtype)
        # mask=1 채널: x = z (identity, 모든 시점)
        x = z * mask_2d
        # mask=0 채널: time-sequential
        for t in range(L):
            x_input = self._make_input(x)
            log_scale, translation = self._params_from(x_input, c_feat)
            x_t_unmasked = z[:, t:t+1, :] * torch.exp(log_scale[:, t:t+1, :]) + translation[:, t:t+1, :]
            x[:, t:t+1, :] = x[:, t:t+1, :] * mask_2d + x_t_unmasked * (1.0 - mask_2d)
        return x

    def inverse_training(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Differentiable inverse — list 누적 + cat (no in-place 수정)."""
        c_feat = self.c_encoder(c)
        B, L, D = z.shape
        mask_2d = self.mask.view(1, 1, -1).to(z.device).to(z.dtype)
        x_list: list = []
        x_init = z * mask_2d                                  # [B, L, D]
        for t in range(L):
            if t == 0:
                x_so_far = x_init.clone()
            else:
                past_built = torch.cat(x_list, dim=1)         # [B, t, D]
                future_init = x_init[:, t:, :]                # [B, L-t, D]
                x_so_far = torch.cat([past_built, future_init], dim=1)
            x_input = self._make_input(x_so_far)
            log_scale, translation = self._params_from(x_input, c_feat)
            x_t_unmasked = z[:, t:t+1, :] * torch.exp(log_scale[:, t:t+1, :]) + translation[:, t:t+1, :]
            x_t_full = x_init[:, t:t+1, :] * mask_2d + x_t_unmasked * (1.0 - mask_2d)
            x_list.append(x_t_full)
        return torch.cat(x_list, dim=1)


class MultiStepCrossChannelFlow(nn.Module):
    """K step alternating mask cross-channel coupling.

    K=2, D=2 → step 0 mask=[1,0], step 1 mask=[0,1] → 두 방향 양방향 coupling.
    K 짝수 면 균형, K 홀수 면 한쪽이 한 번 더 변환됨.
    """

    def __init__(self, K: int = 2, d_cond: int = 4, d_target: int = 2,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.K = K
        self.d_target = d_target
        steps = []
        for k in range(K):
            mask_pattern = [(k + i) % 2 for i in range(d_target)]   # alternating per step
            inner = CrossChannelCausalFlow(
                d_cond=d_cond, d_target=d_target, d_model=d_model,
                n_heads=n_heads, n_layers=n_layers, mask_pattern=mask_pattern,
            )
            steps.append(inner)
        self.steps = nn.ModuleList(steps)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        B, L, D = x.shape
        log_det_total   = x.new_zeros(B)
        log_scale_total = x.new_zeros(B, L, D)
        for step in self.steps:
            x, ld, ls = step(x, c)
            log_det_total   = log_det_total   + ld
            log_scale_total = log_scale_total + ls
        return x, log_det_total, log_scale_total

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = z
        for step in reversed(self.steps):
            x = step.inverse(x, c)
        return x

    def inverse_training(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = z
        for step in reversed(self.steps):
            x = step.inverse_training(x, c)
        return x


# ══════════════════════════════════════════════════════════════════
# 1c. Conditional forecast helper — past-z swap (3D target 일반화)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def conditional_generate_favar(model, x_past: torch.Tensor, c: torch.Tensor,
                               L: int, P: int) -> torch.Tensor:
    """
    x_past : [B, P, d_target]      관측된 past target (i_past, π_past, sp_past)
    c      : [B, L, d_cond]        전 구간 조건 (past + future m2)
    L      : 전체 윈도우 길이
    P      : past 길이.  F = L - P

    Past-z swap (time_reverse=False 일 때만 유효):
        1. x_full = [x_past, 0_padding]  → forward → z_full
        2. z_future ~ N(0, I) 새로 샘플   (shape [B, F, d_target])
        3. z_new = [z_full[:P], z_future]
        4. x_gen = inverse(z_new, c)     past는 가역성으로 복원, future는 새 시나리오

    Returns: x_gen [B, L, d_target]
    """
    B, P_in, D = x_past.shape
    assert P_in == P, f'x_past length {P_in} != P {P}'
    F = L - P
    x_full = torch.cat([x_past, x_past.new_zeros(B, F, D)], dim=1)      # [B, L, D]
    z_full, _, _ = model(x_full, c)
    z_future = torch.randn(B, F, D, device=x_past.device, dtype=x_past.dtype)
    z_new = torch.cat([z_full[:, :P, :], z_future], dim=1)              # [B, L, D]
    x_gen = model.inverse(z_new, c)
    return x_gen


def conditional_generate_favar_training(model, x_past: torch.Tensor, c: torch.Tensor,
                                         L: int, P: int) -> torch.Tensor:
    """
    PINN 학습용 differentiable 버전.
    No @torch.no_grad → gradient 가 model 파라미터까지 흐름.

    x_past : [B, P, d_target]     관측된 past target (gradient 필요 없음, detach 가능)
    c      : [B, L, d_cond]       full condition

    Returns x_gen [B, L, d_target]  — past 구간은 x_past 복원, future는 새 시나리오.
    Gradient 는 future 생성 경로의 flow parameter 를 따라 흐름.
    """
    B, P_in, D = x_past.shape
    assert P_in == P, f'x_past length {P_in} != P {P}'
    F = L - P
    x_full = torch.cat([x_past, x_past.new_zeros(B, F, D)], dim=1)      # [B, L, D]
    # Forward pass (학습 중 autograd 자동 켜짐)
    z_full, _, _ = model(x_full, c)
    # Future z 샘플링은 stop-grad (random noise, parameter 와 무관)
    z_future = torch.randn(B, F, D, device=x_past.device, dtype=x_past.dtype).detach()
    z_new = torch.cat([z_full[:, :P, :], z_future], dim=1)              # [B, L, D]
    # Differentiable inverse
    x_gen = model.inverse_training(z_new, c)
    return x_gen


# ══════════════════════════════════════════════════════════════════
# 2. Data utilities — v31 CSV → tensor windows
# ══════════════════════════════════════════════════════════════════
#
# FAVAR schema:
#   Target    X : [tbill_wr, mich_wr, sp_return]    3 channels, raw
#   Condition C : [m2_growth]                       1 channel, z-score
#
# 모든 채널 raw (log-diff 또는 rate level 그대로).
# Target은 표준화 안 함 (weekly raw 단위 유지 — 스케일이 작아 학습 안정성 OK).
# Condition (m2_growth) 만 훈련 통계로 z-score.

COLS_TARGET = ['tbill_wr', 'mich_wr', 'sp_return']
COLS_COND   = ['m2_growth']


def level_windows(series: np.ndarray, L: int) -> np.ndarray:
    if series.ndim != 1:
        raise ValueError(f'series must be 1D, got shape {series.shape}')
    N = len(series)
    starts = np.arange(N - L + 1)
    idx = starts[:, None] + np.arange(L)[None, :]
    return series[idx]


def load_windows_favar(csv_path: str | Path, L: int = 104,
                        stats: dict | None = None):
    """
    Weekly v31 CSV → (X, C, stats).

    Args:
        csv_path : data/weekly_v31_train.csv 등
        L        : 윈도우 길이 (기본 104)
        stats    : condition 표준화 stats. None이면 계산.

    Returns:
        X     : [N_w, L, 3]   (tbill_wr, mich_wr, sp_return)  raw
        C     : [N_w, L, 1]   (m2_growth)  z-scored (train stats)
        stats : {'mean': [1], 'std': [1]}   — target 쪽은 표준화 안 함
    """
    df = pd.read_csv(csv_path)
    needed = COLS_TARGET + COLS_COND
    df = df.dropna(subset=needed).reset_index(drop=True)
    if len(df) < L:
        raise ValueError(f'Not enough rows: {len(df)} < L={L}')

    # Target X : 3 channel raw
    tgt_arrs = [level_windows(df[c].to_numpy(dtype=np.float32), L)
                for c in COLS_TARGET]
    X_np = np.stack(tgt_arrs, axis=-1)                              # [N_w, L, 3]

    # Condition C : 1 channel raw
    cond_arrs = [level_windows(df[c].to_numpy(dtype=np.float32), L)
                 for c in COLS_COND]
    C_raw = np.stack(cond_arrs, axis=-1)                             # [N_w, L, 1]

    if stats is None:
        C_flat = C_raw.reshape(-1, C_raw.shape[-1])
        stats = {
            'mean': C_flat.mean(axis=0).astype(np.float32),
            'std':  C_flat.std(axis=0).astype(np.float32) + 1e-6,
        }
    C_norm = (C_raw - stats['mean']) / stats['std']

    X = torch.from_numpy(X_np).float()                               # [N_w, L, 3]
    C = torch.from_numpy(C_norm).float()                             # [N_w, L, 1]
    return X, C, stats


# ══════════════════════════════════════════════════════════════════
# 3. Smoke test
# ══════════════════════════════════════════════════════════════════

def _smoke_dummy():
    print('─' * 72)
    print('  (a) Dummy tensor smoke test')
    print('─' * 72)
    torch.manual_seed(0)
    B, L, D_cond, D_tgt, D_model = 4, 104, 1, 3, 64
    x = torch.randn(B, L, D_tgt)
    c = torch.randn(B, L, D_cond)
    model = FAVARFlow(d_cond=D_cond, d_target=D_tgt,
                      d_model=D_model, n_heads=4, n_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'    input  x: {tuple(x.shape)},  c: {tuple(c.shape)}')
    print(f'    params total = {n_params:,}')

    z, ld, ls = model(x, c)
    print(f'    forward z: {tuple(z.shape)},  log_det_J: {tuple(ld.shape)},  '
          f'log_scale: {tuple(ls.shape)}')

    x_rec = model.inverse(z, c)
    err = (x - x_rec).abs().max().item()
    print(f'    inverse x_rec: {tuple(x_rec.shape)}')
    print(f'    max |x - x_rec| = {err:.3e}   '
          f'{"PASS" if err < 1e-3 else "FAIL"} (single-step invertibility)')
    return err


def _smoke_multistep(K: int = 3):
    print('─' * 72)
    print(f'  (b) MultiStepFAVARFlow (K={K}) smoke test')
    print('─' * 72)
    torch.manual_seed(1)
    B, L, D_cond, D_tgt, D_model = 4, 104, 1, 3, 64
    x = torch.randn(B, L, D_tgt)
    c = torch.randn(B, L, D_cond)

    # causal-only (forecast 용)
    model = MultiStepFAVARFlow(K=K, d_cond=D_cond, d_target=D_tgt,
                               d_model=D_model, n_heads=4, n_layers=2,
                               time_reverse=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'    [time_reverse=False]  K={K}  params = {n_params:,}')

    z, ld, ls = model(x, c)
    x_rec = model.inverse(z, c)
    err = (x - x_rec).abs().max().item()
    print(f'      forward z: {tuple(z.shape)}  ld: {tuple(ld.shape)}  ls: {tuple(ls.shape)}')
    print(f'      max |x - x_rec| = {err:.3e}   '
          f'{"PASS" if err < 1e-3 else "FAIL"} (K-step invertibility)')

    # Conditional generate (past-z swap)
    P = 52
    x_past = x[:, :P, :].clone()
    x_gen = conditional_generate_favar(model, x_past, c, L=L, P=P)
    past_err = (x_past - x_gen[:, :P, :]).abs().max().item()
    print(f'      conditional_generate past-preservation: '
          f'max |x_past - x_gen[:P]| = {past_err:.3e}   '
          f'{"PASS" if past_err < 1e-3 else "FAIL"}')

    # time_reverse=True 체크 (학습용, past-fixing 불가능)
    model_tr = MultiStepFAVARFlow(K=K, d_cond=D_cond, d_target=D_tgt,
                                  d_model=D_model, n_heads=4, n_layers=2,
                                  time_reverse=True)
    print(f'    [time_reverse=True]   K={K}  params = {sum(p.numel() for p in model_tr.parameters()):,}')
    z2, _, _ = model_tr(x, c)
    x_rec2 = model_tr.inverse(z2, c)
    err_tr = (x - x_rec2).abs().max().item()
    print(f'      max |x - x_rec| = {err_tr:.3e}   '
          f'{"PASS" if err_tr < 1e-3 else "FAIL"} (invertibility)')
    return err


def _smoke_real(train_csv: Path, test_csv: Path, L: int = 104):
    print('─' * 72)
    print('  (c) Real weekly v31 data smoke test')
    print('─' * 72)

    X_tr, C_tr, stats = load_windows_favar(train_csv, L=L)
    X_te, C_te, _     = load_windows_favar(test_csv,  L=L, stats=stats)
    print(f'    train  X: {tuple(X_tr.shape)},  C: {tuple(C_tr.shape)}')
    print(f'    test   X: {tuple(X_te.shape)},  C: {tuple(C_te.shape)}')
    print(f'    cond stats (train)  mean: {stats["mean"]}')
    print(f'                         std: {stats["std"]}')
    print(f'    X range train  '
          f'[tbill:{X_tr[...,0].min():+.5f}~{X_tr[...,0].max():+.5f}]  '
          f'[mich:{X_tr[...,1].min():+.5f}~{X_tr[...,1].max():+.5f}]  '
          f'[sp:{X_tr[...,2].min():+.5f}~{X_tr[...,2].max():+.5f}]')

    torch.manual_seed(0)
    model = MultiStepFAVARFlow(K=3, d_cond=1, d_target=3,
                               d_model=64, n_heads=4, n_layers=2,
                               time_reverse=False)
    B = 8
    x = X_tr[:B]
    c = C_tr[:B]
    z, ld, ls = model(x, c)
    x_rec = model.inverse(z, c)
    err = (x - x_rec).abs().max().item()
    print(f'    real forward→inverse  max |x - x_rec| = {err:.3e}   '
          f'{"PASS" if err < 1e-3 else "FAIL"}')

    # conditional_generate with real past (3D)
    P = 52
    x_past = x[:, :P, :].clone()
    x_gen = conditional_generate_favar(model, x_past, c, L=L, P=P)
    past_err = (x_past - x_gen[:, :P, :]).abs().max().item()
    print(f'    conditional gen past-preservation  max err = {past_err:.3e}   '
          f'{"PASS" if past_err < 1e-3 else "FAIL"}')
    return err


if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    here = Path(__file__).resolve().parent
    repo = here.parent
    train_csv = repo / 'data' / 'weekly_v31_train.csv'
    test_csv  = repo / 'data' / 'weekly_v31_test.csv'

    e1 = _smoke_dummy()
    print()
    e2 = _smoke_multistep(K=3)
    print()
    e3 = _smoke_real(train_csv, test_csv, L=104)

    print('\n' + '=' * 72)
    print(f'  SUMMARY  dummy={e1:.3e}  multistep={e2:.3e}  real={e3:.3e}')
    print('=' * 72)
