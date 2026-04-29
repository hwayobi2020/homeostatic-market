"""Hierarchical Mamba Pipeline — Dual-Mamba + Cumulative Bridge skeleton.

전체 구조:
    Stage 1   (Mamba A)     : 과거 + 미래 tbill 시나리오 → 미래 52주 excess_liq (raw)
    Stage 1.5 (Bridge)      : 과거 25주 buffer + Stage 1 출력 → 26주 rolling sum (미분 가능)
    Stage 2   (Mamba B)     : 과거 sp_return + Stage 1.5 출력 → 미래 52주 sp_return

설계 근거:
    어제(2026-04-28) 결과 — Raw cumulative (tbill_26w, excess_liq_26w) NN 효과 -0.134 nat.
    pp_bond 산식 가공은 collinearity. → Bridge 의 역할은 "산식 brige" 가 아니라
    "26w 적분 부담을 코드가 대신 처리해서 모델은 누적 데이터를 직접 받음".

이 파일의 범위:
    forward() 의 텐서 shape 흐름과 Stage 1.5 미분 가능성만 명시.
    Stage 1, Stage 2 의 Mamba 내부는 stub (TODO 표시).
"""

from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
# Stage 1.5 — Cumulative Bridge (Deterministic, Differentiable)
# ══════════════════════════════════════════════════════════════════

class CumulativeBridge(nn.Module):
    """과거 (W-1) 주 buffer + 미래 F 주 → 미래 F 주 W주 rolling sum.

    학습 파라미터 0개. 순수 conv1d (kernel = ones) 로 구현.
    Conv1d 가 differentiable view + 선형 연산이라 gradient 자동 흐름.

    수치 검증:
        과거 W-1 = 25주 buffer + 미래 52주 = 77 input 시퀀스
        Kernel size = W = 26, stride 1, padding 0
        Output length = 77 - 26 + 1 = 52  (미래 52주 정합)

    Output index t (0..51) 의미:
        future_cum[t] = sum( full[t : t+26] )
                      = "미래 t 주 시점에서 끝나는 26주 누적합"

        t=0  : past[-25:] + future[0]  (25 past + 1 future)
        t=25 : future[0:26]            (전부 future)
        t=51 : future[26:52]           (마지막 26주 future)

    미분 가능성:
        full = cat([past_buffer.detach(), future_seq], dim=1)
            → past_buffer 는 관측값이라 gradient 불필요 (detach 가능)
            → future_seq (Stage 1 출력) 는 gradient 보존
        F.conv1d(full, kernel) 의 backward:
            ∂out[i,d] / ∂full[j,d] = kernel[d, 0, i-j]  (window 안일 때만 1)
        결과: future_seq[t] 의 gradient 는 영향받은 모든 output 에 더해져 흐름.
    """

    def __init__(self, window: int = 26):
        super().__init__()
        self.window = window

    def forward(self,
                past_buffer: torch.Tensor,
                future_seq: torch.Tensor) -> torch.Tensor:
        """
        past_buffer : [B, W-1, D]   (관측, gradient 불필요)
                       e.g. [B, 25, 2]  마지막 25주 [tbill_wr, excess_liq_raw]
        future_seq  : [B, F, D]     (Stage 1 출력, gradient 필요)
                       e.g. [B, 52, 2]  미래 52주 [tbill_user, excess_liq_gen]

        Return:
            future_cum : [B, F, D]  e.g. [B, 52, 2]   미래 26주 누적합
        """
        B, Wm1, D = past_buffer.shape
        Bf, F_len, Df = future_seq.shape
        assert B == Bf and D == Df, "batch / channel 불일치"
        assert Wm1 == self.window - 1, f"buffer 길이 {Wm1} != window-1 {self.window-1}"

        # 1) Concat   [B, W-1+F, D]  e.g. [B, 77, 2]
        full = torch.cat([past_buffer, future_seq], dim=1)

        # 2) Conv1d 에 맞게 transpose [B, D, L]
        full_t = full.transpose(1, 2)                       # [B, D, 77]

        # 3) Depthwise sum kernel — 채널별 독립 sum
        #    weight shape: [out_ch, in_ch/groups, kernel_size] = [D, 1, W]
        kernel = torch.ones(D, 1, self.window,
                            device=full.device, dtype=full.dtype)

        # 4) Apply conv1d (groups=D 로 채널별 독립 sum)
        out_t = F.conv1d(full_t, kernel, groups=D)          # [B, D, F]

        # 5) Back to [B, F, D]
        future_cum = out_t.transpose(1, 2).contiguous()      # [B, 52, 2]
        return future_cum


# ══════════════════════════════════════════════════════════════════
# Stage 1 — Macro Expander (Mamba A)
# ══════════════════════════════════════════════════════════════════

class MacroExpander(nn.Module):
    """미래 tbill 시나리오 + 과거 시계열 → 미래 excess_liq 생성.

    Stub: Mamba 내부 구현은 별도. 여기선 입출력 shape 만.

    학습 (forward):
        관측 future_excess_liq → z (NLL 학습)
    추론 (inverse):
        z ~ N(0,I) + condition → generated future_excess_liq
    """

    def __init__(self, d_past: int, d_future_cond: int, d_target: int = 1,
                 d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.d_past         = d_past         # 과거 채널 수 (sp_return + macros)
        self.d_future_cond  = d_future_cond  # 미래 조건 채널 (tbill 시나리오 → 1)
        self.d_target       = d_target       # 미래 생성 채널 (excess_liq → 1)
        self.d_model        = d_model
        # TODO: mambapy.mamba.Mamba 또는 비슷한 SSM block 두 개 (past encoder + flow param net)
        # TODO: affine flow head — Linear(d_model → 2 * d_target) for (log_scale, translation)
        # TODO: K=2 affine step stack

    def forward(self,
                past_series: torch.Tensor,
                future_cond: torch.Tensor,
                future_target_observed: torch.Tensor):
        """학습용 forward (관측 미래 → z).

        past_series             : [B, P, d_past]         e.g. [B, 52, 6]
                                   (sp_return, m2_growth, m2v, cpi_yoy, vix, tbill_wr)
        future_cond             : [B, F, d_future_cond]  e.g. [B, 52, 1]
                                   (사용자 주입 tbill_wr 시나리오)
        future_target_observed  : [B, F, d_target]       e.g. [B, 52, 1]
                                   (관측 future excess_liq, NLL 계산용)

        Return:
            z          : [B, F, d_target]               e.g. [B, 52, 1]
            log_det_J  : [B]
            log_scale  : [B, F, d_target]   (진단용)
        """
        # TODO: Stage 1 본구현
        B, F_len, _ = future_cond.shape
        z         = torch.zeros(B, F_len, self.d_target, device=past_series.device)
        log_det_J = torch.zeros(B,                 device=past_series.device)
        log_scale = torch.zeros(B, F_len, self.d_target, device=past_series.device)
        return z, log_det_J, log_scale

    def inverse(self,
                past_series: torch.Tensor,
                future_cond: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """추론용 (z + 조건 → 생성 future_excess_liq). 미분 불필요시 @no_grad."""
        # TODO: Stage 1 inverse 본구현
        return torch.zeros_like(z)


# ══════════════════════════════════════════════════════════════════
# Stage 2 — Price Generator (Mamba B)
# ══════════════════════════════════════════════════════════════════

class PriceGenerator(nn.Module):
    """과거 주가 + 미래 누적 항상성 condition → 미래 sp_return 생성.

    Stub. 학습/추론 인터페이스만 정의.

    Condition 채널 = Stage 1.5 출력 (tbill_26w, excess_liq_26w) 2채널
    Past 채널 = sp_return 1채널 (또는 + 과거 누적 condition 도 같이)
    """

    def __init__(self, d_past_target: int = 1, d_cond: int = 2, d_target: int = 1,
                 d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.d_past_target = d_past_target  # 과거 sp_return → 1
        self.d_cond        = d_cond         # tbill_26w + excess_liq_26w → 2
        self.d_target      = d_target       # 미래 sp_return → 1
        self.d_model       = d_model
        # TODO: Mamba B (causal autoregressive flow), K=2 affine step

    def forward(self,
                past_sp: torch.Tensor,
                past_cum: torch.Tensor,
                future_cum: torch.Tensor,
                future_sp_observed: torch.Tensor):
        """학습용 forward (관측 future sp → z).

        past_sp             : [B, P, 1]            e.g. [B, 52, 1]
        past_cum            : [B, P, 2]            e.g. [B, 52, 2]   (관측 26w cum)
        future_cum          : [B, F, 2]            e.g. [B, 52, 2]   (Stage 1.5 출력)
        future_sp_observed  : [B, F, 1]            e.g. [B, 52, 1]   (NLL 계산용)

        Return:
            z          : [B, F, 1]
            log_det_J  : [B]
            log_scale  : [B, F, 1]
        """
        # TODO: Stage 2 본구현
        B, F_len, _ = future_cum.shape
        z         = torch.zeros(B, F_len, self.d_target, device=past_sp.device)
        log_det_J = torch.zeros(B,                 device=past_sp.device)
        log_scale = torch.zeros(B, F_len, self.d_target, device=past_sp.device)
        return z, log_det_J, log_scale

    def inverse(self,
                past_sp: torch.Tensor,
                past_cum: torch.Tensor,
                future_cum: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        """추론용 (z + 조건 → 생성 future_sp)."""
        # TODO: Stage 2 inverse 본구현
        return torch.zeros_like(z)


# ══════════════════════════════════════════════════════════════════
# Wrapper — Dual-Mamba Pipeline
# ══════════════════════════════════════════════════════════════════

class HierarchicalMambaPipeline(nn.Module):
    """End-to-end wrapper. forward() 가 학습 모드, generate() 가 추론 모드.

    학습 모드 (joint NLL):
        Stage 1  : 관측 future_excess_liq → z_1 (NLL_1)
        Stage 1.5: bridge( past_buffer, future_raw_2ch )
                   future_raw_2ch = [user_tbill_scenario, observed OR generated excess_liq]
                   * 관측 사용 = "teacher forcing bridge" — sequential 학습 / 분리 평가
                   * Stage 1 sample 사용 = "joint" — gradient end-to-end (differentiable inverse 필요)
        Stage 2  : 관측 future_sp_observed → z_2 (NLL_2)
        Total NLL = NLL_1 + λ · NLL_2
    """

    def __init__(self,
                 d_past_macro: int = 6, d_future_tbill: int = 1,
                 d_excess_liq: int = 1, d_cum: int = 2,
                 window: int = 26, d_model: int = 64, n_layers: int = 2,
                 mode: str = "teacher_forcing"):
        """
        mode:
            "teacher_forcing" : Bridge 입력에 관측 future_raw 사용 (sequential 등가, 안정적)
            "joint"           : Stage 1 inverse 로 sample → bridge → Stage 2 (gradient e2e)
        """
        super().__init__()
        self.stage1 = MacroExpander(
            d_past=d_past_macro, d_future_cond=d_future_tbill,
            d_target=d_excess_liq, d_model=d_model, n_layers=n_layers,
        )
        self.bridge = CumulativeBridge(window=window)
        self.stage2 = PriceGenerator(
            d_past_target=1, d_cond=d_cum, d_target=1,
            d_model=d_model, n_layers=n_layers,
        )
        self.mode = mode

    def forward(self, batch: dict) -> dict:
        """
        batch keys (학습용):
            past_macros          : [B, P, d_past_macro]      e.g. [B, 52, 6]
                                    (sp_return, m2_growth, m2v, cpi_yoy, vix, tbill_wr)
            future_tbill_scenario: [B, F, 1]                  e.g. [B, 52, 1]
                                    (사용자 주입 미래 금리)
            past_buffer_raw      : [B, W-1, 2]                e.g. [B, 25, 2]
                                    (마지막 25주 [tbill_wr, excess_liq_raw] 관측)
            past_sp              : [B, P, 1]                  e.g. [B, 52, 1]
            past_cum_observed    : [B, P, 2]                  e.g. [B, 52, 2]
                                    (과거 [tbill_26w, excess_liq_26w] 관측)
            future_excess_liq_obs: [B, F, 1]                  e.g. [B, 52, 1]   (NLL_1 용)
            future_sp_obs        : [B, F, 1]                  e.g. [B, 52, 1]   (NLL_2 용)

        Return dict:
            z1, log_det_J_1, log_scale_1   ← Stage 1 출력
            future_cum                      ← Stage 1.5 출력 [B, F, 2]
            z2, log_det_J_2, log_scale_2   ← Stage 2 출력
        """
        P = batch["past_macros"].shape[1]
        F_len = batch["future_tbill_scenario"].shape[1]

        # ── Stage 1 ──────────────────────────────────────────────
        z1, log_det_J_1, log_scale_1 = self.stage1(
            past_series           = batch["past_macros"],            # [B, 52, 6]
            future_cond           = batch["future_tbill_scenario"],  # [B, 52, 1]
            future_target_observed= batch["future_excess_liq_obs"],  # [B, 52, 1]
        )
        # z1 : [B, 52, 1]

        # ── Stage 1.5 (Bridge, differentiable) ───────────────────
        # 미래 raw 2채널 = [user_tbill_scenario, future_excess_liq]
        # mode 따라 future_excess_liq 출처 분기:
        if self.mode == "teacher_forcing":
            future_excess_liq_for_bridge = batch["future_excess_liq_obs"]      # 관측 사용
            #   ↑ Stage 1 의 출력은 NLL 학습에만 쓰이고, Bridge 입력은 관측값.
            #     → Stage 2 학습이 Stage 1 noise 에 영향 받지 않음. 분리 진단 용이.
        elif self.mode == "joint":
            # Stage 1 의 inverse_training 으로 z1 을 다시 x 로 복원 (sample 아님, observed 와 같지만
            # gradient 가 흐르는 경로를 만들기 위함). 또는 z ~ N(0,I) 새 샘플 + inverse_training.
            future_excess_liq_for_bridge = self.stage1.inverse(
                past_series  = batch["past_macros"],
                future_cond  = batch["future_tbill_scenario"],
                z            = torch.randn_like(z1),
            )                                                                  # [B, 52, 1]
        else:
            raise ValueError(self.mode)

        # 2채널 concat
        future_raw_2ch = torch.cat([
            batch["future_tbill_scenario"],   # [B, 52, 1]  (사용자 시나리오, 그대로)
            future_excess_liq_for_bridge,     # [B, 52, 1]  (관측 또는 Stage 1 출력)
        ], dim=-1)                            # [B, 52, 2]

        future_cum = self.bridge(
            past_buffer = batch["past_buffer_raw"],  # [B, 25, 2]   (관측, detach 가능)
            future_seq  = future_raw_2ch,            # [B, 52, 2]   (gradient 흐름)
        )                                            # [B, 52, 2]

        # ── Stage 2 ──────────────────────────────────────────────
        z2, log_det_J_2, log_scale_2 = self.stage2(
            past_sp            = batch["past_sp"],            # [B, 52, 1]
            past_cum           = batch["past_cum_observed"],  # [B, 52, 2]
            future_cum         = future_cum,                  # [B, 52, 2]
            future_sp_observed = batch["future_sp_obs"],      # [B, 52, 1]
        )
        # z2 : [B, 52, 1]

        return {
            "z1": z1, "log_det_J_1": log_det_J_1, "log_scale_1": log_scale_1,
            "future_cum": future_cum,
            "z2": z2, "log_det_J_2": log_det_J_2, "log_scale_2": log_scale_2,
        }

    @torch.no_grad()
    def generate(self, past_macros, future_tbill_scenario, past_buffer_raw,
                 past_sp, past_cum_observed,
                 z1: torch.Tensor | None = None, z2: torch.Tensor | None = None):
        """추론용 시나리오 생성 (사용자 시점).

        past_macros, past_buffer_raw, past_sp, past_cum_observed : 관측
        future_tbill_scenario : 사용자가 주입한 미래 금리 시나리오

        반환:
            future_excess_liq : [B, F, 1]  Stage 1 sample
            future_cum        : [B, F, 2]
            future_sp         : [B, F, 1]  Stage 2 sample
        """
        B = past_macros.shape[0]
        F_len = future_tbill_scenario.shape[1]
        d_eq, d_sp = self.stage1.d_target, self.stage2.d_target
        if z1 is None:
            z1 = torch.randn(B, F_len, d_eq, device=past_macros.device)
        if z2 is None:
            z2 = torch.randn(B, F_len, d_sp, device=past_macros.device)

        future_excess_liq = self.stage1.inverse(past_macros, future_tbill_scenario, z1)
        future_raw_2ch    = torch.cat([future_tbill_scenario, future_excess_liq], dim=-1)
        future_cum        = self.bridge(past_buffer_raw, future_raw_2ch)
        future_sp         = self.stage2.inverse(past_sp, past_cum_observed, future_cum, z2)
        return future_excess_liq, future_cum, future_sp


# ══════════════════════════════════════════════════════════════════
# Smoke test — shape 확인 + Bridge 미분 가능성 검증
# ══════════════════════════════════════════════════════════════════

def _smoke_bridge():
    print("─" * 72)
    print("  CumulativeBridge differentiability test")
    print("─" * 72)
    torch.manual_seed(0)
    B, W, F_len, D = 4, 26, 52, 2

    past_buffer = torch.randn(B, W - 1, D)                         # 관측
    future_seq  = torch.randn(B, F_len, D, requires_grad=True)     # Stage 1 출력 흉내

    bridge = CumulativeBridge(window=W)
    future_cum = bridge(past_buffer, future_seq)
    print(f"  past_buffer shape  : {tuple(past_buffer.shape)}")
    print(f"  future_seq  shape  : {tuple(future_seq.shape)}")
    print(f"  future_cum  shape  : {tuple(future_cum.shape)}")
    assert future_cum.shape == (B, F_len, D), "출력 shape 오류"

    # gradient 흐름 확인 — future_seq 에 대해 backward 호출
    loss = future_cum.sum()
    loss.backward()
    grad = future_seq.grad
    print(f"  grad on future_seq : {tuple(grad.shape)}")
    print(f"  grad mean / max    : {grad.mean().item():.4f} / {grad.max().item():.4f}")
    print(f"  expected = 26 (각 future 위치가 평균 26개 output 에 영향)  "
          f"→ 안쪽 위치 grad ≈ {grad[:, 25:-25].mean().item():.4f}")

    # 산식 검증 — output[0] = sum(past_buffer[0,:,d]) + future_seq[0,0,d]
    manual_t0 = past_buffer[0, :, 0].sum() + future_seq[0, 0, 0]
    print(f"  manual t=0 d=0     : {manual_t0.item():+.4f}")
    print(f"  conv1d t=0 d=0     : {future_cum[0, 0, 0].item():+.4f}")


def _smoke_pipeline():
    print("\n" + "─" * 72)
    print("  HierarchicalMambaPipeline shape flow")
    print("─" * 72)
    torch.manual_seed(1)
    B, P, F_len = 4, 52, 52
    pipe = HierarchicalMambaPipeline(
        d_past_macro=6, d_future_tbill=1, d_excess_liq=1, d_cum=2,
        window=26, d_model=64, n_layers=2, mode="teacher_forcing",
    )
    batch = {
        "past_macros":             torch.randn(B, P, 6),
        "future_tbill_scenario":   torch.randn(B, F_len, 1),
        "past_buffer_raw":         torch.randn(B, 25, 2),
        "past_sp":                 torch.randn(B, P, 1),
        "past_cum_observed":       torch.randn(B, P, 2),
        "future_excess_liq_obs":   torch.randn(B, F_len, 1),
        "future_sp_obs":           torch.randn(B, F_len, 1),
    }
    out = pipe(batch)
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k:20s}: {tuple(v.shape)}")


if __name__ == "__main__":
    _smoke_bridge()
    _smoke_pipeline()
