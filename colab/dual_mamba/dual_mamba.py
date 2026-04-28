"""Hierarchical Dual-Mamba Pipeline (Stage 1 + Cumulative Bridge + Stage 2).

설계 요약
========
3단계 파이프라인:
    Stage 1   (Mamba A, MacroExpander) :
        Input  past series (sp_return + 5 macros) + future tbill scenario
        Output future 52w excess_liq (raw weekly, 1ch)
    Stage 1.5 (Bridge, Deterministic 26w rolling sum, learnable params 0개)
        Input  past 25w buffer + future 52w (tbill scenario, Stage 1 출력)
        Output future 52w (tbill_26w, excess_liq_26w)
    Stage 2   (Mamba B, PriceGenerator) :
        Input  past sp_return + Stage 1.5 cumulative condition
        Output future 52w sp_return

Causal autoregressive normalizing flow (K affine step):
    x_t^d = exp(s_t^d) * z_t^d + t_t^d
    (s_t^d, t_t^d) = head( Mamba( concat( x_{<=t-1}, c_{<=t} ) ) )

Backbone: mambapy.mamba.Mamba (selective state-space, O(L) memory).
모드     : teacher_forcing — Bridge 입력에 관측 future_excess_liq 사용. Stage 간 gradient 단절.

작성: 2026-04-28 dual_mamba project
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mambapy.mamba import Mamba, MambaConfig

LOG2PI = math.log(2.0 * math.pi)


# ══════════════════════════════════════════════════════════════════
# 0. Cumulative Bridge — Deterministic, Differentiable
# ══════════════════════════════════════════════════════════════════

class CumulativeBridge(nn.Module):
    """과거 (W-1) 주 buffer + 미래 F 주 → 미래 F 주 W주 rolling MEAN.

    학습 파라미터 0개. Conv1d (kernel = ones/W, depthwise) 로 구현.
    Sum 대신 mean 사용 — z-scored 입력의 26w 합은 std~26으로 폭주 (autocorr 강함).
    Mean 으로 나누면 std ~ 1 안정 스케일.

    Shape:
        past_buffer : [B, W-1, D]
        future_seq  : [B, F,   D]
        Output      : [B, F,   D]   (W=26 일 때, 미래 t 시점에서 끝나는 26w 평균)

    Output index t (0..F-1) 의미:
        t=0   : mean( past_buffer[-25:] + future_seq[0] )    = past 25 + future 1
        t=W-1 : mean( future_seq[0:W] )                      = 전부 future
        t=F-1 : mean( future_seq[F-W:F] )                    = 마지막 W주 future
    """

    def __init__(self, window: int = 26):
        super().__init__()
        self.window = window

    def forward(self,
                past_buffer: torch.Tensor,
                future_seq: torch.Tensor) -> torch.Tensor:
        B, Wm1, D = past_buffer.shape
        Bf, F_len, Df = future_seq.shape
        assert B == Bf and D == Df, "batch / channel 불일치"
        assert Wm1 == self.window - 1, f"buffer 길이 {Wm1} != window-1 {self.window-1}"

        # 1) Concat   [B, W-1+F, D]   e.g. [B, 77, 2]
        full = torch.cat([past_buffer, future_seq], dim=1)
        # 2) [B, D, L] 형태 (conv1d 입력 규약)
        full_t = full.transpose(1, 2)
        # 3) Depthwise mean kernel — 채널별 독립 mean (sum / W)
        kernel = torch.ones(D, 1, self.window,
                            device=full.device, dtype=full.dtype) / float(self.window)
        # 4) groups=D 로 채널별 평균
        out_t = F.conv1d(full_t, kernel, groups=D)            # [B, D, F_len]
        # 5) 원형 [B, F, D] 복귀
        return out_t.transpose(1, 2).contiguous()


# ══════════════════════════════════════════════════════════════════
# 1. Causal Mamba Affine Flow Step
# ══════════════════════════════════════════════════════════════════

def _right_shift(x: torch.Tensor) -> torch.Tensor:
    """[B, L, D] → [B, L, D] with x_shift[:, 0, :] = 0, x_shift[:, t, :] = x[:, t-1, :]."""
    x_shift = torch.zeros_like(x)
    x_shift[:, 1:, :] = x[:, :-1, :]
    return x_shift


class MambaCausalAffineFlow(nn.Module):
    """1 affine step. Causal autoregressive flow on x[B, L, d_target] given c[B, L, d_cond].

    x_t^d = exp(s_t^d) * z_t^d + t_t^d
    (s_t, t_t) = head( Mamba( concat( x_{<=t-1}, c_{<=t} ) ) )

    Mamba block은 left-to-right selective scan이라 자연스럽게 causal.
    log_scale clamp 로 |s|≤4 (수치 안정성).
    """

    def __init__(self, d_cond: int, d_target: int,
                 d_model: int = 64, n_layers: int = 2,
                 log_scale_clamp: float = 2.0):
        super().__init__()
        self.d_cond     = d_cond
        self.d_target   = d_target
        self.d_model    = d_model
        self.log_scale_clamp = log_scale_clamp

        # 1) Condition encoder: c[B, L, d_cond] → c_feat[B, L, d_model]
        self.c_proj  = nn.Linear(d_cond, d_model)
        self.c_mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))

        # 2) Parameter network: (x_shift, c_feat) → (log_scale, translation)
        self.xc_proj    = nn.Linear(d_target + d_model, d_model)
        self.param_mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
        self.param_head = nn.Linear(d_model, 2 * d_target)

    def _params(self, x_shift: torch.Tensor, c_feat: torch.Tensor):
        h = torch.cat([x_shift, c_feat], dim=-1)              # [B, L, d_target + d_model]
        h = self.xc_proj(h)                                   # [B, L, d_model]
        h = self.param_mamba(h)                               # [B, L, d_model]
        p = self.param_head(h)                                # [B, L, 2*d_target]
        log_scale, translation = p.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * self.log_scale_clamp
        return log_scale, translation

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """학습 forward (x → z + log_det_J).

        x : [B, L, d_target]
        c : [B, L, d_cond]
        """
        c_feat = self.c_mamba(self.c_proj(c))                 # [B, L, d_model]
        x_shift = _right_shift(x)
        log_scale, translation = self._params(x_shift, c_feat)
        z = (x - translation) * torch.exp(-log_scale)
        log_det_J = log_scale.sum(dim=(1, 2))                  # [B]
        return z, log_det_J, log_scale

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """추론 inverse (z → x). Causal AR sequential time loop."""
        c_feat = self.c_mamba(self.c_proj(c))
        B, L, D = z.shape
        x = torch.zeros_like(z)
        for t in range(L):
            x_shift = _right_shift(x)
            log_scale, translation = self._params(x_shift, c_feat)
            x[:, t:t + 1, :] = (
                z[:, t:t + 1, :] * torch.exp(log_scale[:, t:t + 1, :])
                + translation[:, t:t + 1, :]
            )
        return x


class MultiStepMambaFlow(nn.Module):
    """K개 MambaCausalAffineFlow 스택."""

    def __init__(self, K: int, d_cond: int, d_target: int,
                 d_model: int = 64, n_layers: int = 2,
                 log_scale_clamp: float = 2.0):
        super().__init__()
        self.K = K
        self.d_target = d_target
        self.steps = nn.ModuleList([
            MambaCausalAffineFlow(
                d_cond=d_cond, d_target=d_target,
                d_model=d_model, n_layers=n_layers,
                log_scale_clamp=log_scale_clamp,
            )
            for _ in range(K)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        B, L, D = x.shape
        log_det_total   = x.new_zeros(B)
        log_scale_total = x.new_zeros(B, L, D)
        z = x
        for step in self.steps:
            z, ld, ls = step(z, c)
            log_det_total   = log_det_total   + ld
            log_scale_total = log_scale_total + ls
        return z, log_det_total, log_scale_total

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = z
        for step in reversed(self.steps):
            x = step.inverse(x, c)
        return x


# ══════════════════════════════════════════════════════════════════
# 2. Stage 1 — Macro Expander (excess_liq generator)
# ══════════════════════════════════════════════════════════════════

class MacroExpander(nn.Module):
    """미래 tbill 시나리오 + 과거 시계열 → 미래 excess_liq 생성.

    구조:
        Target    X1_full [B, L=104, 1]   excess_liq (past observed + future to learn/generate)
        Condition C1_full [B, L=104, 6]   조건 6채널
            Past 52w  : [sp_return, m2_growth, m2v, cpi_yoy, vix, tbill_wr]  (실측)
            Future 52w: [0,         0,         0,   0,       0,   tbill_wr_scenario] (사용자 주입 + 0 padding)

    NLL은 future 52w에 한정 (past portion은 teacher forcing).
    """

    D_COND_PAST   = 6   # sp_return, m2_growth, m2v, cpi_yoy, vix, tbill_wr
    D_COND_FUTURE = 1   # tbill_wr (사용자 시나리오, 다른 5채널은 0 padding)
    D_TARGET      = 1   # excess_liq_yoy

    def __init__(self, K: int = 2, d_model: int = 64, n_layers: int = 2,
                 log_scale_clamp: float = 2.0):
        super().__init__()
        self.flow = MultiStepMambaFlow(
            K=K, d_cond=self.D_COND_PAST, d_target=self.D_TARGET,
            d_model=d_model, n_layers=n_layers,
            log_scale_clamp=log_scale_clamp,
        )

    @staticmethod
    def build_condition(past_cond: torch.Tensor,
                        future_tbill: torch.Tensor) -> torch.Tensor:
        """past 6채널 + future 1채널 (tbill만, 나머지 0 padding) → C1_full [B, L, 6]."""
        B, P, D_past = past_cond.shape
        Bf, F_len, _ = future_tbill.shape
        assert B == Bf and D_past == MacroExpander.D_COND_PAST, \
            f"past_cond shape {past_cond.shape} != [B, P, 6]"
        # tbill_wr 위치 = past_cond[:, :, 5] 와 동일 (마지막 채널)
        future_cond = past_cond.new_zeros(B, F_len, D_past)
        future_cond[:, :, 5] = future_tbill[:, :, 0]
        c_full = torch.cat([past_cond, future_cond], dim=1)             # [B, L, 6]
        return c_full

    def encode_condition(self, c_full: torch.Tensor) -> torch.Tensor:
        """Stage 1 의 첫 번째 affine step c_encoder 출력 (Stage 2 export 용).

        c_full : [B, L, 6]
        return c_feat : [B, L, d_model]   (Mamba가 학습한 압축된 macro 컨텍스트)

        Note: flow forward 안에서 동일 계산이 또 일어나므로 약간 중복 (Mamba 1회 추가).
        """
        first_step = self.flow.steps[0]
        return first_step.c_mamba(first_step.c_proj(c_full))

    def forward(self,
                past_cond: torch.Tensor,
                future_tbill: torch.Tensor,
                X1_full: torch.Tensor):
        """학습 forward.

        past_cond     : [B, P=52, 6]    (관측 past 6 channel)
        future_tbill  : [B, F=52, 1]    (사용자 주입 future tbill_wr)
        X1_full       : [B, L=104, 1]   (관측 excess_liq_yoy past+future, NLL 학습 target)

        Return:
            z          : [B, L, 1]
            log_det_J  : [B]
            log_scale  : [B, L, 1]
            c_full     : [B, L, 6]      (Stage 1 condition, 진단용)
            c_feat     : [B, L, d_model] (encoded condition, Stage 2 export 용)
        """
        c_full = self.build_condition(past_cond, future_tbill)
        c_feat = self.encode_condition(c_full)
        z, log_det_J, log_scale = self.flow(X1_full, c_full)
        return z, log_det_J, log_scale, c_full, c_feat

    @torch.no_grad()
    def generate(self,
                 past_cond: torch.Tensor,
                 future_tbill: torch.Tensor,
                 past_excess_liq: torch.Tensor,
                 z_future: torch.Tensor | None = None) -> torch.Tensor:
        """추론용 (past 관측 + future tbill 시나리오 → future excess_liq).

        Past-z swap 방식:
            1) X1_full = [past_excess_liq, 0_pad] → flow forward → z_full
            2) z_future ~ N(0, I) 새로 샘플 (또는 사용자 주입)
            3) z_new = [z_full[:P], z_future]
            4) inverse(z_new, c_full) → x_gen
        """
        B, P, _ = past_excess_liq.shape
        F_len = future_tbill.shape[1]
        L = P + F_len

        c_full = self.build_condition(past_cond, future_tbill)

        # Past-z swap
        x_init = torch.cat(
            [past_excess_liq,
             past_excess_liq.new_zeros(B, F_len, self.D_TARGET)], dim=1
        )
        z_full, _, _ = self.flow(x_init, c_full)
        if z_future is None:
            z_future = torch.randn(B, F_len, self.D_TARGET,
                                    device=past_excess_liq.device,
                                    dtype=past_excess_liq.dtype)
        z_new = torch.cat([z_full[:, :P, :], z_future], dim=1)
        x_gen = self.flow.inverse(z_new, c_full)
        return x_gen   # [B, L, 1]


# ══════════════════════════════════════════════════════════════════
# 3. Stage 2 — Price Generator (sp_return generator)
# ══════════════════════════════════════════════════════════════════

class PriceGenerator(nn.Module):
    """과거 sp + raw macro/tbill 시퀀스 + 누적 항상성 → 미래 sp_return 생성.

    구조 (8채널 condition):
        Target    X2_full [B, L=104, 1]                  sp_return (past 관측 + future)
        Condition C2_full [B, L=104, 6 + 2 (+d_export)]  = raw 6 + cumulative 2 (+ Stage 1 c_feat 4)
            past 52w  : raw 6채널 (sp,m2_growth,m2v,cpi,vix,tbill_wr 관측) + cumulative 2채널 + (옵션 c_feat)
            future 52w: raw 6채널 (5채널 0 padding + tbill_wr 사용자) + Bridge cumulative + (옵션 c_feat)

    Stage 1 의 build_condition 과 동일한 방식으로 raw past_cond + future_tbill 사용.
    d_export = 0 이면 raw 6 + cumulative 2 = 8채널 (default).
    d_export > 0 이면 + import_proj(c_feat) 추가.
    """

    # past_cond 6채널 중 sp_return (채널 0) 제외 — Target X autoregressive 와 중복 방지
    D_COND_RAW = 5   # m2_growth, m2v, cpi_yoy, vix, tbill_wr
    D_COND_CUM = 2   # tbill_26w, excess_liq_26w
    D_TARGET   = 1   # sp_return

    def __init__(self, K: int = 2, d_model: int = 64, n_layers: int = 2,
                 log_scale_clamp: float = 2.0,
                 d_stage1_feat: int = 64,    # Stage 1 c_feat 입력 차원
                 d_export: int = 0,          # 0 = raw+cum 만, >0 = c_feat 추가
                 ):
        super().__init__()
        self.d_export = d_export
        if d_export > 0:
            self.import_proj = nn.Linear(d_stage1_feat, d_export)
        else:
            self.import_proj = None

        d_cond_total = self.D_COND_RAW + self.D_COND_CUM + d_export
        self.D_COND = d_cond_total
        self.flow = MultiStepMambaFlow(
            K=K, d_cond=d_cond_total, d_target=self.D_TARGET,
            d_model=d_model, n_layers=n_layers,
            log_scale_clamp=log_scale_clamp,
        )

    @staticmethod
    def build_raw_condition(past_cond: torch.Tensor,
                             future_tbill: torch.Tensor,
                             future_macro4: torch.Tensor | None = None,
                             oracle_macro: bool = False) -> torch.Tensor:
        """past_cond 6채널 [sp, m2_growth, m2v, cpi_yoy, vix, tbill_wr] 에서
        sp_return (채널 0) 제외한 5채널만 사용. Target X 의 autoregressive 경로와 중복 방지.

        oracle_macro=False (기본):
            past 5ch (관측) + future (4 zero pad + tbill_wr 시나리오) → [B, L=104, 5]
        oracle_macro=True (Stage 2 upper bound 측정용):
            past 5ch (관측) + future (m2_growth, m2v, cpi_yoy, vix obs + tbill_wr) → [B, L, 5]
            future_macro4 [B, F, 4] 가 m2_growth/m2v/cpi_yoy/vix 순서로 관측 정답 leak.
        """
        B, P, D_past_orig = past_cond.shape
        Bf, F_len, _ = future_tbill.shape
        assert B == Bf and D_past_orig == 6, f"past_cond 채널 6 기대, got {D_past_orig}"
        # sp_return (채널 0) 제외 → 5채널 (m2_growth, m2v, cpi_yoy, vix, tbill_wr)
        past_no_sp = past_cond[:, :, 1:]                            # [B, P, 5]
        if oracle_macro:
            assert future_macro4 is not None, \
                "future_macro4 required when oracle_macro=True"
            assert future_macro4.shape == (B, F_len, 4), \
                f"future_macro4 expected [B, F, 4], got {future_macro4.shape}"
            future_cond = torch.cat([
                future_macro4,                                       # [B, F, 4] (m2g, m2v, cpi, vix)
                future_tbill,                                        # [B, F, 1] (tbill_wr)
            ], dim=-1)                                               # [B, F, 5]
        else:
            future_cond = past_cond.new_zeros(B, F_len, PriceGenerator.D_COND_RAW)
            # tbill_wr 은 past_no_sp 의 마지막 채널 (인덱스 4) → future 도 동일 위치
            future_cond[:, :, 4] = future_tbill[:, :, 0]
        return torch.cat([past_no_sp, future_cond], dim=1)          # [B, L, 5]

    def _build_cond(self, past_cond, future_tbill, past_cum, future_cum, c_feat,
                    future_macro4=None, oracle_macro: bool = False):
        """[B, L=104, 6 + 2 (+d_export)] condition 빌드."""
        c_raw = self.build_raw_condition(past_cond, future_tbill,
                                          future_macro4=future_macro4,
                                          oracle_macro=oracle_macro)        # [B, L, 5]
        c_cum = torch.cat([past_cum, future_cum], dim=1)                    # [B, L, 2]
        c_full = torch.cat([c_raw, c_cum], dim=-1)                          # [B, L, 7]
        if self.d_export > 0:
            assert c_feat is not None, "c_feat required when d_export > 0"
            c_extra = self.import_proj(c_feat)                              # [B, L, d_export]
            c_full = torch.cat([c_full, c_extra], dim=-1)                   # [B, L, 7+d_export]
        return c_full

    def forward(self,
                past_cond: torch.Tensor,
                future_tbill: torch.Tensor,
                past_cum: torch.Tensor,
                future_cum: torch.Tensor,
                X2_full: torch.Tensor,
                c_feat: torch.Tensor | None = None,
                future_macro4: torch.Tensor | None = None,
                oracle_macro: bool = False):
        """학습 forward.

        past_cond    : [B, P, 6]   raw 6채널 관측 (Stage 1 과 동일)
        future_tbill : [B, F, 1]   tbill_wr 사용자 시나리오
        past_cum     : [B, P, 2]   관측 26w mean
        future_cum   : [B, F, 2]   Bridge 출력
        X2_full      : [B, L, 1]   sp_return target
        c_feat       : [B, L, d_stage1_feat] or None (d_export>0 시 필수)
        future_macro4: [B, F, 4]   oracle_macro=True 시 m2g/m2v/cpi/vix 관측 정답 (leak baseline)
        oracle_macro : bool        True 면 future portion 4 zero pad → 관측 정답으로 swap
        """
        c_full = self._build_cond(past_cond, future_tbill, past_cum, future_cum, c_feat,
                                   future_macro4=future_macro4,
                                   oracle_macro=oracle_macro)
        z, log_det_J, log_scale = self.flow(X2_full, c_full)
        return z, log_det_J, log_scale, c_full

    @torch.no_grad()
    def generate(self,
                 past_cond: torch.Tensor,
                 future_tbill: torch.Tensor,
                 past_cum: torch.Tensor,
                 future_cum: torch.Tensor,
                 past_sp: torch.Tensor,
                 c_feat: torch.Tensor | None = None,
                 z_future: torch.Tensor | None = None) -> torch.Tensor:
        """추론. Past-z swap."""
        B, P, _ = past_sp.shape
        F_len = future_cum.shape[1]

        c_full = self._build_cond(past_cond, future_tbill, past_cum, future_cum, c_feat)
        x_init = torch.cat(
            [past_sp, past_sp.new_zeros(B, F_len, self.D_TARGET)], dim=1
        )
        z_full, _, _ = self.flow(x_init, c_full)
        if z_future is None:
            z_future = torch.randn(B, F_len, self.D_TARGET,
                                    device=past_sp.device, dtype=past_sp.dtype)
        z_new = torch.cat([z_full[:, :P, :], z_future], dim=1)
        x_gen = self.flow.inverse(z_new, c_full)
        return x_gen


# ══════════════════════════════════════════════════════════════════
# 4. Wrapper — Hierarchical Dual-Mamba Pipeline
# ══════════════════════════════════════════════════════════════════

class DualMambaPipeline(nn.Module):
    """Stage 1 + Bridge + Stage 2 통합 wrapper.

    학습 모드:
        bridge_input_mode="obs"           : Bridge 입력에 관측 future_excess_liq 사용
                                            (기존 동작, Stage 1/2 학습-추론 분포 mismatch 발생).
        bridge_input_mode="sample_detach" : Bridge 입력에 Stage 1 generate sample 사용
                                            (detach 로 Stage 1 gradient 단절, Stage 2 만 sample 분포에 적응).
    """

    def __init__(self, K: int = 2, d_model: int = 64, n_layers: int = 2,
                 window: int = 26, log_scale_clamp: float = 4.0,
                 d_export: int = 0,
                 bridge_input_mode: str = "obs",
                 stage2_oracle_macro: bool = False):
        """d_export: Stage 1 c_feat → Stage 2 import_proj 출력 차원.
                     0 = raw 6 + cumulative 2 = 8ch (default), >0 = + c_feat 추가.
        bridge_input_mode: "obs" (기존) | "sample_detach" (Stage 1 sample → Bridge, gradient 단절).
        stage2_oracle_macro: True 면 Stage 2 condition future portion 4 zero pad → 관측 정답
                             (m2_growth, m2v, cpi_yoy, vix) leak. Stage 2 capacity 상한 측정용.
                             추론 시 사용 불가 — 학습/eval baseline 전용.
        """
        super().__init__()
        if bridge_input_mode not in ("obs", "sample_detach"):
            raise ValueError(
                f"bridge_input_mode must be 'obs' or 'sample_detach', got {bridge_input_mode!r}"
            )
        self.stage1 = MacroExpander(
            K=K, d_model=d_model, n_layers=n_layers,
            log_scale_clamp=log_scale_clamp,
        )
        self.bridge = CumulativeBridge(window=window)
        self.stage2 = PriceGenerator(
            K=K, d_model=d_model, n_layers=n_layers,
            log_scale_clamp=log_scale_clamp,
            d_stage1_feat=d_model, d_export=d_export,
        )
        self.window = window
        self.d_export = d_export
        self.bridge_input_mode = bridge_input_mode
        self.stage2_oracle_macro = stage2_oracle_macro

    def forward(self, batch: dict) -> dict:
        """teacher_forcing 학습 forward.

        batch keys:
            past_cond            : [B, P=52, 6]   (sp_return, m2_growth, m2v, cpi_yoy, vix, tbill_wr) — Stage 1 cond
            future_tbill         : [B, F=52, 1]   (사용자 시나리오)
            past_buffer_raw      : [B, 25, 2]     (마지막 25w [tbill_wr, excess_liq_yoy])
            past_excess_liq_obs  : [B, P=52, 1]   (관측 excess_liq past)
            future_excess_liq_obs: [B, F=52, 1]   (관측 excess_liq future, NLL_1 + Bridge)
            past_sp_obs          : [B, P=52, 1]   (관측 sp_return past)
            future_sp_obs        : [B, F=52, 1]   (관측 sp_return future, NLL_2)
            past_cum_observed    : [B, P=52, 2]   (관측 [tbill_26w, excess_liq_26w] past) — Stage 2 cond
            past_macro4          : [B, P=52, 4]   (현재 미사용, 6ch 확장 옵션 대비 보존)

        Return dict:
            nll1            : scalar (Stage 1 future NLL, mean over batch+future timesteps)
            nll2            : scalar (Stage 2)
            total_loss      : nll1 + nll2
            log_det_J_1, log_det_J_2 : [B]
            future_cum      : [B, F, 2]    (Bridge 출력, 진단용)
            log_scale_1, log_scale_2 : [B, L, 1]
        """
        B, P, _ = batch["past_cond"].shape
        F_len = batch["future_tbill"].shape[1]
        L = P + F_len   # 104

        # ── Stage 1 ──────────────────────────────────────────────
        X1_full = torch.cat(
            [batch["past_excess_liq_obs"], batch["future_excess_liq_obs"]], dim=1
        )                                                           # [B, L, 1]
        z1, log_det_J_1, log_scale_1, c1_full, c_feat = self.stage1(
            past_cond    = batch["past_cond"],
            future_tbill = batch["future_tbill"],
            X1_full      = X1_full,
        )                                                           # c_feat: [B, L, d_model]

        # NLL_1 (future portion only)
        z1_future = z1[:, P:, :]                                   # [B, F, 1]
        ls1_future = log_scale_1[:, P:, :]                         # [B, F, 1]
        # nll_t,d = 0.5 z² + 0.5 log(2π) + log_scale_total
        # log_det_J_1 sums over (L, D), but for "future-only" NLL we use per-timestep log_scale.
        # NLL formula: 0.5*z² + 0.5*log2π + log_scale (per (t,d))
        nll1_per = 0.5 * z1_future.pow(2) + 0.5 * LOG2PI + ls1_future   # [B, F, 1]
        nll1 = nll1_per.mean()                                      # scalar (B*F*D 평균)

        # ── Stage 1.5 (Bridge) ───────────────────────────────────
        # Bridge 입력 future_excess_liq 결정:
        #   - "obs"          : 관측 ground truth (학습-추론 분포 mismatch 발생)
        #   - "sample_detach": Stage 1 generate sample, detach (Stage 2 만 학습)
        if self.bridge_input_mode == "obs":
            future_excess_liq_for_bridge = batch["future_excess_liq_obs"]
        elif self.bridge_input_mode == "sample_detach":
            # Stage 1 generate (no_grad, sequential AR inverse). 매 step random z_future ~ N(0,1).
            x1_sample = self.stage1.generate(
                past_cond       = batch["past_cond"],
                future_tbill    = batch["future_tbill"],
                past_excess_liq = batch["past_excess_liq_obs"],
                z_future        = None,
            )                                                       # [B, L, 1] (no_grad context)
            future_excess_liq_for_bridge = x1_sample[:, P:, :].detach()
        else:
            raise ValueError(f"Unknown bridge_input_mode: {self.bridge_input_mode}")

        future_raw_2ch = torch.cat([
            batch["future_tbill"],                # [B, F, 1] tbill scenario
            future_excess_liq_for_bridge,         # [B, F, 1] obs or Stage 1 sample (detach)
        ], dim=-1)                                # [B, F, 2]
        future_cum = self.bridge(
            past_buffer = batch["past_buffer_raw"],   # [B, 25, 2]
            future_seq  = future_raw_2ch,             # [B, F, 2]
        )                                              # [B, F, 2]   (rolling mean, std~1)

        # ── Stage 2 ──────────────────────────────────────────────
        X2_full = torch.cat(
            [batch["past_sp_obs"], batch["future_sp_obs"]], dim=1
        )                                                           # [B, L, 1]
        z2, log_det_J_2, log_scale_2, c2_full = self.stage2(
            past_cond    = batch["past_cond"],
            future_tbill = batch["future_tbill"],
            past_cum     = batch["past_cum_observed"],
            future_cum   = future_cum,
            X2_full      = X2_full,
            c_feat       = c_feat if self.d_export > 0 else None,
            future_macro4 = batch.get("future_macro4") if self.stage2_oracle_macro else None,
            oracle_macro  = self.stage2_oracle_macro,
        )

        z2_future  = z2[:, P:, :]
        ls2_future = log_scale_2[:, P:, :]
        nll2_per = 0.5 * z2_future.pow(2) + 0.5 * LOG2PI + ls2_future
        nll2 = nll2_per.mean()

        return {
            "nll1": nll1, "nll2": nll2, "total_loss": nll1 + nll2,
            "z1": z1, "log_det_J_1": log_det_J_1, "log_scale_1": log_scale_1,
            "future_cum": future_cum,
            "z2": z2, "log_det_J_2": log_det_J_2, "log_scale_2": log_scale_2,
        }

    @torch.no_grad()
    def generate(self,
                 past_cond: torch.Tensor,
                 future_tbill: torch.Tensor,
                 past_buffer_raw: torch.Tensor,
                 past_excess_liq: torch.Tensor,
                 past_sp: torch.Tensor,
                 past_cum_observed: torch.Tensor,
                 z1: torch.Tensor | None = None,
                 z2: torch.Tensor | None = None) -> dict:
        """추론용 시나리오 생성.

        흐름:
            1) Stage 1: past + future_tbill → future_excess_liq sample
            2) Bridge:  past_buffer + (future_tbill, future_excess_liq sample) → future_cum
            3) Stage 2: past + future_cum → future_sp sample
        """
        B, P, _ = past_cond.shape
        F_len = future_tbill.shape[1]

        # Stage 1
        x1_gen = self.stage1.generate(
            past_cond=past_cond, future_tbill=future_tbill,
            past_excess_liq=past_excess_liq, z_future=z1,
        )                                                           # [B, L, 1]
        future_excess_liq_gen = x1_gen[:, P:, :]                    # [B, F, 1]

        # Bridge
        future_raw_2ch = torch.cat(
            [future_tbill, future_excess_liq_gen], dim=-1
        )                                                            # [B, F, 2]
        future_cum = self.bridge(past_buffer_raw, future_raw_2ch)   # [B, F, 2]

        # Stage 1 c_feat (for Stage 2 import_proj)
        c_feat = None
        if self.d_export > 0:
            c1_full = self.stage1.build_condition(past_cond, future_tbill)
            c_feat = self.stage1.encode_condition(c1_full)           # [B, L, d_model]

        # Stage 2 (raw 6 + cumulative 2 + optional c_feat)
        x2_gen = self.stage2.generate(
            past_cond=past_cond, future_tbill=future_tbill,
            past_cum=past_cum_observed, future_cum=future_cum,
            past_sp=past_sp, z_future=z2, c_feat=c_feat,
        )                                                           # [B, L, 1]
        future_sp_gen = x2_gen[:, P:, :]                            # [B, F, 1]

        return {
            "future_excess_liq": future_excess_liq_gen,             # [B, F, 1]
            "future_cum": future_cum,                               # [B, F, 2]
            "future_sp": future_sp_gen,                             # [B, F, 1]
        }


# ══════════════════════════════════════════════════════════════════
# 5. Smoke test
# ══════════════════════════════════════════════════════════════════

def _smoke():
    print("─" * 72)
    print("  Dual-Mamba pipeline smoke test (CPU)")
    print("─" * 72)
    torch.manual_seed(0)
    B, P, F_len = 2, 52, 52
    L = P + F_len

    pipe = DualMambaPipeline(K=2, d_model=32, n_layers=1, window=26)
    n_params = sum(p.numel() for p in pipe.parameters())
    print(f"  Total params (K=2, d_model=32, n_layers=1): {n_params:,}")

    batch = {
        "past_cond":             torch.randn(B, P, 6),
        "past_macro4":           torch.randn(B, P, 4),       # m2_growth, m2v, cpi_yoy, vix
        "future_tbill":          torch.randn(B, F_len, 1) * 0.01,
        "past_buffer_raw":       torch.randn(B, 25, 2) * 0.01,
        "past_excess_liq_obs":   torch.randn(B, P, 1) * 0.01,
        "future_excess_liq_obs": torch.randn(B, F_len, 1) * 0.01,
        "past_sp_obs":           torch.randn(B, P, 1) * 0.02,
        "future_sp_obs":         torch.randn(B, F_len, 1) * 0.02,
        "past_cum_observed":     torch.randn(B, P, 2),       # normalized
    }

    out = pipe(batch)
    print(f"\n  Forward outputs:")
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"    {k:18s}: {tuple(v.shape) if v.dim() else 'scalar'}  "
                  f"value={v.item():.4f}" if v.dim() == 0 else
                  f"    {k:18s}: {tuple(v.shape)}")

    # Backward
    loss = out["total_loss"]
    loss.backward()
    grad_norm = sum(p.grad.pow(2).sum().item() for p in pipe.parameters() if p.grad is not None) ** 0.5
    print(f"\n  Loss = {loss.item():.4f}")
    print(f"  Total gradient L2 norm = {grad_norm:.4f}")

    # Generate
    pipe.eval()
    with torch.no_grad():
        gen = pipe.generate(
            past_cond=batch["past_cond"],
            future_tbill=batch["future_tbill"],
            past_buffer_raw=batch["past_buffer_raw"],
            past_excess_liq=batch["past_excess_liq_obs"],
            past_sp=batch["past_sp_obs"],
            past_cum_observed=batch["past_cum_observed"],
        )
    print(f"\n  Generated shapes:")
    for k, v in gen.items():
        print(f"    {k:18s}: {tuple(v.shape)}")


if __name__ == "__main__":
    _smoke()
