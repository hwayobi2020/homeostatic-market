"""full_fpath 모델 — 기존 MAC-Flow(train_garch_flow.MambaFlowAR)에 *미래경로 전체 요약*을
Flow context 로 추가한 변형.  train_garch_flow.py 는 한 줄도 안 바꾼다(격리).

배경
----
기존 모델은 미래 거시(tbill/metab)를 AR rollout 에서 *각 시점 그 시점 값* 으로만 본다
(per-step contemporaneous, look-ahead 없음 — encode 의 마지막 토큰).  반사실 시나리오에선
미래 거시 경로를 *이미 다 알고* 부과하므로, 한 스텝씩 가릴 이유가 없다.  → 미래경로 전체
(tbill 13 + unmask macro 13 = 26)를 작은 MLP 로 요약(out=16)해 *모든 생성 스텝의 Flow
context 에 broadcast* 한다.  기존 per-step 경로는 그대로 두고 *추가*만 한다(통제: 두 모델
차이 = 전체경로 요약의 순수 기여 = look-ahead 효과).

사용
----
  import train_garch_flow as T
  import fpath_model
  fpath_model.FUTURE_SUMMARY_DIM = 16      # flow 로 들어가는 미래요약 차원
  fpath_model.FUTURE_SUMMARY_HIDDEN = 16   # bottleneck
  T.MambaFlowAR = fpath_model.MambaFlowARFpath   # monkeypatch (train()/evaluate_test 가 이걸 씀)

기존 full/maskall 학습/추론에는 영향 없음(이 모듈을 import·patch 하지 않으면 됨).
checkpoint meta 에 future 설정을 안 적으므로, *재현/별도 rebuild* 가 필요한 분석 스크립트는
같은 방식으로 patch 한 뒤 로드해야 한다(방향성 1차 검증은 main_worker in-process eval 로 충분).
"""
import torch
import torch.nn as nn

import train_garch_flow as T

# 런너가 덮어쓰는 전역 설정 (기본값).
FUTURE_SUMMARY_DIM = 16        # Flow context 에 더해지는 미래요약 차원 (out)
FUTURE_SUMMARY_HIDDEN = 16     # 요약기 bottleneck hidden
# summary_only: per-step 미래 거시는 *모델 내부에서* 마스킹(인코더가 못 봄)하고,
# 미래 인코더(요약 코드)로만 조건화.  데이터는 full 로 로드해야 인코더가 실제 경로를 받음.
FPATH_SUMMARY_ONLY = False


class FutureMLPSummary(nn.Module):
    """미래 거시경로(flatten) → bottleneck → 요약 벡터.  PastMLPSummary 와 동일 구조.

    in_dim = FUTURE_LEN * n_future_macro_ch  (예: 13 * (tbill 1 + metab 1) = 26)
    (B, in_dim) -> (B, out_dim)
    """
    def __init__(self, in_dim, out_dim, hidden=16, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                 # (B, in_dim)
        return self.net(x)


class MambaFlowARFpath(T.MambaFlowAR):
    """MambaFlowAR + 미래경로 전체 요약(Flow context 에 broadcast).

    super().__init__ 가 *옛* flow_context_dim 으로 self.flow 를 만들므로,
    future 차원을 더한 뒤 self.flow 를 *재구성* 한다(미래요약 포함 context).
    """

    def __init__(self, *args, **kwargs):
        # flow 재구성에 필요한 인자 (super 가 attribute 로 안 남기는 것들) 선캡처.
        self._fp_dropout = float(kwargs.get("dropout", 0.0))
        self._fp_n_flow_layers = int(kwargs.get("n_flow_layers", T.N_FLOW_LAYERS))
        self._fp_n_flow_hidden = int(kwargs.get("n_flow_hidden", T.N_FLOW_HIDDEN))
        super().__init__(*args, **kwargs)

        # 미래 거시 채널 = 금리(tbill) + unmask macro(예: metab_13w), COND_COLS 인덱스.
        um = [T.COND_COLS.index(c) for c in T.FUTURE_UNMASK_MACRO_COLS
              if c in T.COND_COLS]
        self._fp_cols = [T.TBILL_CH] + um            # 시각 t 의 [tbill, metab,...] 채널
        n_ch = len(self._fp_cols)
        self._fp_n_ch = n_ch
        in_dim = T.FUTURE_LEN * n_ch
        self.future_summary_dim = int(FUTURE_SUMMARY_DIM)
        # summary_only: per-step 미래 거시(금리+모든 macro)를 메인 인코더 입력에서 0 마스킹.
        # (미래 인코더는 마스킹 전 실제 경로를 받음 → 요약 코드로만 조건화)
        self._summary_only = bool(FPATH_SUMMARY_ONLY)
        self._fp_mask_cols = [T.TBILL_CH] + list(T.MACRO_CH)

        self.future_encoder = FutureMLPSummary(
            in_dim=in_dim, out_dim=self.future_summary_dim,
            hidden=int(FUTURE_SUMMARY_HIDDEN), dropout=self._fp_dropout)

        # Flow context 확장 + flow 재구성 (미래요약 포함).
        self.flow_context_dim = self.flow_context_dim + self.future_summary_dim
        base = T.SkewStudentT(shape=[1], df=T.FLOW_BASE_DF)
        transforms = []
        for _ in range(self._fp_n_flow_layers):
            transforms.append(
                T.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=1,
                    hidden_features=self._fp_n_flow_hidden,
                    context_features=self.flow_context_dim,
                    num_blocks=T.N_FLOW_BLOCKS,
                    num_bins=T.N_FLOW_BINS,
                    tails="linear",
                    tail_bound=T.FLOW_TAIL_BOUND,
                    dropout_probability=self._fp_dropout,
                ))
        self.flow = T.Flow(T.CompositeTransform(transforms), base)

    # ── 미래경로 요약 (학습/샘플 공통 레이아웃: (B, FUTURE_LEN, n_ch) → flatten) ──
    def _future_summary_from_seq(self, fut_seq):
        """fut_seq: (B, FUTURE_LEN, n_ch) [tbill, metab,...] → (B, future_summary_dim)."""
        B = fut_seq.shape[0]
        return self.future_encoder(fut_seq.reshape(B, -1))

    def log_prob(self, x_input, target_path, extra_context=None):
        """teacher-forced log p — 원본과 동일 + 미래경로 요약 broadcast(맨 뒤)."""
        B = x_input.shape[0]
        Tn = target_path.shape[1]
        # 미래경로 요약은 *마스킹 전* 실제 경로에서 뽑는다 (아래). 메인 인코더 입력만 마스킹.
        if self._summary_only:
            x_enc = x_input.clone()
            x_enc[:, self.past_len:, self._fp_mask_cols] = 0.0   # per-step 미래 거시 제거
        else:
            x_enc = x_input
        h = self.encode(x_enc)                                  # (B, L, d_model)
        h_future = h[:, self.past_len:, :]                      # (B, T, d_model)
        ctx_parts = [h_future]
        if self.use_past_summary:
            past_sum = self.encode_past(x_input)
            ctx_parts.append(past_sum.unsqueeze(1).expand(-1, Tn, -1))
        if self.direct_prev_dim > 0:
            prev_ret = x_input[:, self.past_len:, T.SP_CH:T.SP_CH + 1]
            ctx_parts.append(prev_ret)
        if self.direct_future_dim > 0:
            _dir_idx = [T.COND_COLS.index(c) for c in T.DIRECT_FUTURE_COLS
                        if c in T.COND_COLS]
            ctx_parts.append(x_input[:, self.past_len:, _dir_idx])
        if self.extra_context_dim > 0:
            if extra_context is None:
                raise ValueError("extra_context required when extra_context_dim>0")
            ctx_parts.append(extra_context.unsqueeze(1).expand(-1, Tn, -1))
        # ── 신규: 미래경로 전체 요약 (origin/scenario 고정 → 전 스텝 동일) ──
        fut_seq = x_input[:, self.past_len:, self._fp_cols]      # (B, T, n_ch)
        fut_sum = self._future_summary_from_seq(fut_seq)         # (B, dim)
        ctx_parts.append(fut_sum.unsqueeze(1).expand(-1, Tn, -1))

        ctx_seq = torch.cat(ctx_parts, dim=-1)
        y_flat = target_path.reshape(B * Tn, 1)
        ctx_flat = ctx_seq.reshape(B * Tn, self.flow_context_dim)
        log_p_flat = self.flow.log_prob(inputs=y_flat, context=ctx_flat)
        return log_p_flat.reshape(B, Tn).sum(dim=1)

    @torch.no_grad()
    def ar_sample(self, x_past, future_tbill_z, last_past_sp_z, n_sim,
                  extra_context=None, future_macro_z=None):
        """원본 ar_sample + 미래경로 전체 요약 broadcast(ctx 맨 뒤)."""
        device = x_past.device
        B = x_past.shape[0]
        N_CH = x_past.shape[-1]
        seq = x_past.unsqueeze(1).expand(B, n_sim, -1, -1).contiguous()
        seq = seq.view(B * n_sim, T.PAST_LEN, N_CH)
        tbill_fut = future_tbill_z.unsqueeze(1).expand(B, n_sim, -1).contiguous()
        tbill_fut = tbill_fut.view(B * n_sim, T.FUTURE_LEN)
        last_sp = last_past_sp_z.unsqueeze(1).expand(B, n_sim).contiguous().view(-1)
        if self.extra_context_dim > 0:
            if extra_context is None:
                raise ValueError("extra_context required when extra_context_dim>0")
            extra_rep = extra_context.unsqueeze(1).expand(
                B, n_sim, -1).contiguous().view(B * n_sim, self.extra_context_dim)
        else:
            extra_rep = None

        if self.use_past_summary:
            past_sum = self.encode_past(x_past)
            past_sum_rep = past_sum.unsqueeze(1).expand(
                B, n_sim, -1).contiguous().view(B * n_sim, -1)
        else:
            past_sum_rep = None

        _unmask_idx = [T.COND_COLS.index(c) for c in T.FUTURE_UNMASK_MACRO_COLS
                       if c in T.COND_COLS]
        if _unmask_idx and future_macro_z is not None:
            fm = future_macro_z.unsqueeze(1).expand(B, n_sim, -1, -1).contiguous()
            fm = fm.view(B * n_sim, T.FUTURE_LEN, len(_unmask_idx))
        else:
            fm = None

        # ── 신규: 미래경로 전체 요약 (1회 계산 → 전 스텝/전 sim broadcast) ──
        #   레이아웃 [tbill, metab,...] = log_prob 의 self._fp_cols 순서와 일치.
        fut3 = future_tbill_z.unsqueeze(-1)                      # (B, T, 1)
        if future_macro_z is not None and future_macro_z.shape[-1] > 0:
            fut3 = torch.cat([fut3, future_macro_z], dim=-1)     # (B, T, n_ch)
        fut_sum0 = self._future_summary_from_seq(fut3)           # (B, dim)
        fut_sum_rep = fut_sum0.unsqueeze(1).expand(
            B, n_sim, -1).contiguous().view(B * n_sim, -1)

        sampled = []
        for tau in range(T.FUTURE_LEN):
            next_input = torch.zeros(B * n_sim, 1, T.N_CHANNELS,
                                     device=device, dtype=seq.dtype)
            next_input[:, 0, T.SP_CH] = last_sp
            if not self._summary_only:
                # full_fpath: per-step 미래 거시 주입(인코더가 봄).
                next_input[:, 0, T.TBILL_CH] = tbill_fut[:, tau]
                if fm is not None:
                    for _j, _ch in enumerate(_unmask_idx):
                        next_input[:, 0, _ch] = fm[:, tau, _j]
            # summary_only: per-step 미래 거시 안 채움(0=마스크) → 요약 코드로만 조건화

            seq = torch.cat([seq, next_input], dim=1)
            h_seq = self.encode(seq)
            h_tau = h_seq[:, -1, :]
            ctx_parts = [h_tau]
            if self.use_past_summary:
                ctx_parts.append(past_sum_rep)
            if self.direct_prev_dim > 0:
                ctx_parts.append(last_sp.unsqueeze(-1))
            if self.direct_future_dim > 0:
                _dir_idx = [T.COND_COLS.index(c) for c in T.DIRECT_FUTURE_COLS
                            if c in T.COND_COLS]
                ctx_parts.append(seq[:, -1, _dir_idx])
            if self.extra_context_dim > 0:
                ctx_parts.append(extra_rep)
            ctx_parts.append(fut_sum_rep)                        # 신규: 미래경로 요약

            ctx_tau = torch.cat(ctx_parts, dim=-1)
            sp_z = self.flow.sample(1, context=ctx_tau).squeeze(-1).squeeze(-1)
            sampled.append(sp_z)
            last_sp = sp_z

        sampled = torch.stack(sampled, dim=1)
        return sampled.view(B, n_sim, T.FUTURE_LEN)
