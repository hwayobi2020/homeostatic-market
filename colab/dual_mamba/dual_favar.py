"""Hierarchical Dual-FAVAR Pipeline (Stage 1 + Cumulative Bridge + Stage 2).

dual_mamba.py 와 동일한 구조 — backbone 만 Mamba selective scan → Transformer causal attention 으로 교체.
인터페이스 호환 (DualFAVARPipeline.forward(batch) → dict).

설계 요약
========
3단계 파이프라인:
    Stage 1 (FAVAR A, MacroExpander) :
        Input  past series (sp_return + 5 macros) + future tbill scenario
        Output future 52w excess_liq (raw weekly, 1ch)
    Stage 1.5 (Bridge, Deterministic 26w rolling mean, learnable params 0개)
    Stage 2 (FAVAR B, PriceGenerator) :
        Input  past sp_return + Stage 1.5 cumulative condition
        Output future 52w sp_return

Causal autoregressive normalizing flow (K affine step):
    x_t^d = exp(s_t^d) * z_t^d + t_t^d
    (s_t^d, t_t^d) = head( CausalTransformerBlock( concat( x_{<=t-1}, c_{<=t} ) ) )

Backbone: nn.TransformerEncoder + causal mask (sim/favar_flow.py 의 CausalTransformerBlock 형식).
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

LOG2PI = math.log(2.0 * math.pi)


# ══════════════════════════════════════════════════════════════════
# 0. Cumulative Bridge — Deterministic, Differentiable
# ══════════════════════════════════════════════════════════════════

class CumulativeBridge(nn.Module):
    """과거 (W-1) 주 buffer + 미래 F 주 → 미래 F 주 W주 rolling MEAN.

    학습 파라미터 0개. Conv1d (kernel = ones/W, depthwise) 로 구현.
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

        full = torch.cat([past_buffer, future_seq], dim=1)
        full_t = full.transpose(1, 2)
        kernel = torch.ones(D, 1, self.window,
                            device=full.device, dtype=full.dtype) / float(self.window)
        out_t = F.conv1d(full_t, kernel, groups=D)
        return out_t.transpose(1, 2).contiguous()


# ══════════════════════════════════════════════════════════════════
# 1. Causal Transformer Backbone + Affine Flow Step
# ══════════════════════════════════════════════════════════════════

def _right_shift(x: torch.Tensor) -> torch.Tensor:
    """[B, L, D] → [B, L, D] with x_shift[:, 0, :] = 0, x_shift[:, t, :] = x[:, t-1, :]."""
    x_shift = torch.zeros_like(x)
    x_shift[:, 1:, :] = x[:, :-1, :]
    return x_shift


class CausalTransformerBlock(nn.Module):
    """causal self-attention stack. [B, L, d_in] → [B, L, d_model].

    sim/favar_flow.py 의 동명 클래스와 동일 구조.
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
        self.core = nn.TransformerEncoder(layer, num_layers=n_layers,
                                          enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.proj_in(x) + self.pos[:, :L]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        return self.core(h, mask=mask, is_causal=True)


class FAVARCausalAffineFlow(nn.Module):
    """1 affine step. Causal autoregressive flow on x[B, L, d_target] given c[B, L, d_cond].

    x_t^d = exp(s_t^d) * z_t^d + t_t^d
    (s_t, t_t) = head( CausalTransformerBlock( concat( x_{<=t-1}, c_{<=t} ) ) )

    dual_mamba.MambaCausalAffineFlow 의 backbone swap 버전 (Mamba → Transformer).
    """

    def __init__(self, d_cond: int, d_target: int,
                 d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4,
                 log_scale_clamp: float = 4.0):
        super().__init__()
        self.d_cond     = d_cond
        self.d_target   = d_target
        self.d_model    = d_model
        self.log_scale_clamp = log_scale_clamp

        # 1) Condition encoder
        self.c_encoder = CausalTransformerBlock(
            d_in=d_cond, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers,
        )

        # 2) Parameter network (xc → params)
        self.xc_proj = nn.Linear(d_target + d_model, d_model)
        self.param_net = CausalTransformerBlock(
            d_in=d_model, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers,
        )
        self.param_head = nn.Linear(d_model, 2 * d_target)

    def _params(self, x_shift: torch.Tensor, c_feat: torch.Tensor):
        h = torch.cat([x_shift, c_feat], dim=-1)
        h = self.xc_proj(h)
        h = self.param_net(h)
        p = self.param_head(h)
        log_scale, translation = p.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * self.log_scale_clamp
        return log_scale, translation

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """학습 forward (x → z + log_det_J)."""
        c_feat = self.c_encoder(c)
        x_shift = _right_shift(x)
        log_scale, translation = self._params(x_shift, c_feat)
        z = (x - translation) * torch.exp(-log_scale)
        log_det_J = log_scale.sum(dim=(1, 2))
        return z, log_det_J, log_scale

    @torch.no_grad()
    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """추론 inverse (z → x). Causal AR sequential time loop."""
        c_feat = self.c_encoder(c)
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


class MultiStepFAVARFlow(nn.Module):
    """K개 FAVARCausalAffineFlow 스택."""

    def __init__(self, K: int, d_cond: int, d_target: int,
                 d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4,
                 log_scale_clamp: float = 4.0):
        super().__init__()
        self.K = K
        self.d_target = d_target
        self.steps = nn.ModuleList([
            FAVARCausalAffineFlow(
                d_cond=d_cond, d_target=d_target,
                d_model=d_model, n_layers=n_layers,
                n_heads=n_heads,
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

class MacroExpanderFAVAR(nn.Module):
    """미래 tbill 시나리오 + 과거 시계열 → 미래 excess_liq 생성.

    dual_mamba.MacroExpander 와 동일 인터페이스, backbone 만 FAVAR Transformer.
    """

    D_COND_PAST   = 6   # sp_return, m2_growth, m2v, cpi_yoy, vix, tbill_wr
    D_COND_FUTURE = 1   # tbill_wr (사용자 시나리오)
    D_TARGET      = 1   # excess_liq_yoy

    def __init__(self, K: int = 2, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, log_scale_clamp: float = 4.0):
        super().__init__()
        self.flow = MultiStepFAVARFlow(
            K=K, d_cond=self.D_COND_PAST, d_target=self.D_TARGET,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            log_scale_clamp=log_scale_clamp,
        )

    @staticmethod
    def build_condition(past_cond: torch.Tensor,
                        future_tbill: torch.Tensor) -> torch.Tensor:
        B, P, D_past = past_cond.shape
        Bf, F_len, _ = future_tbill.shape
        assert B == Bf and D_past == MacroExpanderFAVAR.D_COND_PAST
        future_cond = past_cond.new_zeros(B, F_len, D_past)
        future_cond[:, :, 5] = future_tbill[:, :, 0]
        return torch.cat([past_cond, future_cond], dim=1)

    def encode_condition(self, c_full: torch.Tensor) -> torch.Tensor:
        """Stage 1 의 첫 번째 affine step c_encoder 출력 (Stage 2 export 용)."""
        first_step = self.flow.steps[0]
        return first_step.c_encoder(c_full)

    def forward(self, past_cond, future_tbill, X1_full):
        c_full = self.build_condition(past_cond, future_tbill)
        c_feat = self.encode_condition(c_full)
        z, log_det_J, log_scale = self.flow(X1_full, c_full)
        return z, log_det_J, log_scale, c_full, c_feat

    @torch.no_grad()
    def generate(self, past_cond, future_tbill, past_excess_liq, z_future=None):
        B, P, _ = past_excess_liq.shape
        F_len = future_tbill.shape[1]
        L = P + F_len
        c_full = self.build_condition(past_cond, future_tbill)
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
        return self.flow.inverse(z_new, c_full)


# ══════════════════════════════════════════════════════════════════
# 3. Stage 2 — Price Generator (sp_return generator)
# ══════════════════════════════════════════════════════════════════

class PriceGeneratorFAVAR(nn.Module):
    """과거 sp + raw macro/tbill + 누적 항상성 → 미래 sp_return 생성.

    dual_mamba.PriceGenerator 와 동일 인터페이스, backbone 만 FAVAR Transformer.
    """

    D_COND_RAW = 5   # m2_growth, m2v, cpi_yoy, vix, tbill_wr (sp_return 제외)
    D_COND_CUM = 2   # tbill_26w, excess_liq_26w
    D_TARGET   = 1   # sp_return

    def __init__(self, K: int = 2, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, log_scale_clamp: float = 4.0,
                 d_stage1_feat: int = 64, d_export: int = 0):
        super().__init__()
        self.d_export = d_export
        if d_export > 0:
            self.import_proj = nn.Linear(d_stage1_feat, d_export)
        else:
            self.import_proj = None

        d_cond_total = self.D_COND_RAW + self.D_COND_CUM + d_export
        self.D_COND = d_cond_total
        self.flow = MultiStepFAVARFlow(
            K=K, d_cond=d_cond_total, d_target=self.D_TARGET,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            log_scale_clamp=log_scale_clamp,
        )

    @staticmethod
    def build_raw_condition(past_cond: torch.Tensor,
                             future_tbill: torch.Tensor,
                             future_macro4: torch.Tensor | None = None,
                             oracle_macro: bool = False) -> torch.Tensor:
        B, P, D_past_orig = past_cond.shape
        Bf, F_len, _ = future_tbill.shape
        assert B == Bf and D_past_orig == 6
        past_no_sp = past_cond[:, :, 1:]
        if oracle_macro:
            assert future_macro4 is not None
            future_cond = torch.cat([future_macro4, future_tbill], dim=-1)
        else:
            future_cond = past_cond.new_zeros(B, F_len, PriceGeneratorFAVAR.D_COND_RAW)
            future_cond[:, :, 4] = future_tbill[:, :, 0]
        return torch.cat([past_no_sp, future_cond], dim=1)

    def _build_cond(self, past_cond, future_tbill, past_cum, future_cum, c_feat,
                    future_macro4=None, oracle_macro: bool = False):
        c_raw = self.build_raw_condition(past_cond, future_tbill,
                                          future_macro4=future_macro4,
                                          oracle_macro=oracle_macro)
        c_cum = torch.cat([past_cum, future_cum], dim=1)
        c_full = torch.cat([c_raw, c_cum], dim=-1)
        if self.d_export > 0:
            assert c_feat is not None
            c_extra = self.import_proj(c_feat)
            c_full = torch.cat([c_full, c_extra], dim=-1)
        return c_full

    def forward(self, past_cond, future_tbill, past_cum, future_cum, X2_full,
                c_feat=None, future_macro4=None, oracle_macro: bool = False):
        c_full = self._build_cond(past_cond, future_tbill, past_cum, future_cum, c_feat,
                                   future_macro4=future_macro4, oracle_macro=oracle_macro)
        z, log_det_J, log_scale = self.flow(X2_full, c_full)
        return z, log_det_J, log_scale, c_full

    @torch.no_grad()
    def generate(self, past_cond, future_tbill, past_cum, future_cum,
                 past_sp, c_feat=None, z_future=None):
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
        return self.flow.inverse(z_new, c_full)


# ══════════════════════════════════════════════════════════════════
# 4. Wrapper — Hierarchical Dual-FAVAR Pipeline
# ══════════════════════════════════════════════════════════════════

class DualFAVARPipeline(nn.Module):
    """Stage 1 + Bridge + Stage 2 통합. dual_mamba.DualMambaPipeline 와 동일 인터페이스."""

    def __init__(self, K: int = 2, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4,
                 window: int = 26, log_scale_clamp: float = 4.0,
                 d_export: int = 0,
                 bridge_input_mode: str = "obs",
                 stage2_oracle_macro: bool = False):
        super().__init__()
        if bridge_input_mode not in ("obs", "sample_detach"):
            raise ValueError(f"bridge_input_mode invalid: {bridge_input_mode!r}")
        self.stage1 = MacroExpanderFAVAR(
            K=K, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            log_scale_clamp=log_scale_clamp,
        )
        self.bridge = CumulativeBridge(window=window)
        self.stage2 = PriceGeneratorFAVAR(
            K=K, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            log_scale_clamp=log_scale_clamp,
            d_stage1_feat=d_model, d_export=d_export,
        )
        self.window = window
        self.d_export = d_export
        self.bridge_input_mode = bridge_input_mode
        self.stage2_oracle_macro = stage2_oracle_macro

    def forward(self, batch: dict) -> dict:
        B, P, _ = batch["past_cond"].shape
        F_len = batch["future_tbill"].shape[1]
        L = P + F_len

        # ── Stage 1 ──────────────────────────────────────────────
        X1_full = torch.cat(
            [batch["past_excess_liq_obs"], batch["future_excess_liq_obs"]], dim=1
        )
        z1, log_det_J_1, log_scale_1, c1_full, c_feat = self.stage1(
            past_cond=batch["past_cond"],
            future_tbill=batch["future_tbill"],
            X1_full=X1_full,
        )

        z1_future = z1[:, P:, :]
        ls1_future = log_scale_1[:, P:, :]
        nll1_per = 0.5 * z1_future.pow(2) + 0.5 * LOG2PI + ls1_future
        nll1 = nll1_per.mean()

        # ── Stage 1.5 (Bridge) ───────────────────────────────────
        if self.bridge_input_mode == "obs":
            future_excess_liq_for_bridge = batch["future_excess_liq_obs"]
        elif self.bridge_input_mode == "sample_detach":
            x1_sample = self.stage1.generate(
                past_cond=batch["past_cond"],
                future_tbill=batch["future_tbill"],
                past_excess_liq=batch["past_excess_liq_obs"],
                z_future=None,
            )
            future_excess_liq_for_bridge = x1_sample[:, P:, :].detach()
        else:
            raise ValueError(f"Unknown bridge_input_mode: {self.bridge_input_mode}")

        future_raw_2ch = torch.cat([
            batch["future_tbill"],
            future_excess_liq_for_bridge,
        ], dim=-1)
        future_cum = self.bridge(
            past_buffer=batch["past_buffer_raw"],
            future_seq=future_raw_2ch,
        )

        # ── Stage 2 ──────────────────────────────────────────────
        X2_full = torch.cat(
            [batch["past_sp_obs"], batch["future_sp_obs"]], dim=1
        )
        z2, log_det_J_2, log_scale_2, c2_full = self.stage2(
            past_cond=batch["past_cond"],
            future_tbill=batch["future_tbill"],
            past_cum=batch["past_cum_observed"],
            future_cum=future_cum,
            X2_full=X2_full,
            c_feat=c_feat if self.d_export > 0 else None,
            future_macro4=batch.get("future_macro4") if self.stage2_oracle_macro else None,
            oracle_macro=self.stage2_oracle_macro,
        )

        z2_future = z2[:, P:, :]
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
    def generate(self, past_cond, future_tbill, past_buffer_raw,
                 past_excess_liq, past_sp, past_cum_observed,
                 z1=None, z2=None) -> dict:
        B, P, _ = past_cond.shape
        F_len = future_tbill.shape[1]

        x1_gen = self.stage1.generate(
            past_cond=past_cond, future_tbill=future_tbill,
            past_excess_liq=past_excess_liq, z_future=z1,
        )
        future_excess_liq_gen = x1_gen[:, P:, :]

        future_raw_2ch = torch.cat(
            [future_tbill, future_excess_liq_gen], dim=-1
        )
        future_cum = self.bridge(past_buffer_raw, future_raw_2ch)

        c_feat = None
        if self.d_export > 0:
            c1_full = self.stage1.build_condition(past_cond, future_tbill)
            c_feat = self.stage1.encode_condition(c1_full)

        x2_gen = self.stage2.generate(
            past_cond=past_cond, future_tbill=future_tbill,
            past_cum=past_cum_observed, future_cum=future_cum,
            past_sp=past_sp, z_future=z2, c_feat=c_feat,
        )
        future_sp_gen = x2_gen[:, P:, :]

        return {
            "future_excess_liq": future_excess_liq_gen,
            "future_cum": future_cum,
            "future_sp": future_sp_gen,
        }
