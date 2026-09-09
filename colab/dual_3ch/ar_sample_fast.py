# -*- coding: utf-8 -*-
"""AR 롤아웃 가속 — 스텝마다 마지막 토큰만 인코딩.

무엇이 느렸나
-------------
`ar_sample` 은 스텝마다 시퀀스를 한 칸 늘려 **전체를 다시 인코딩**하고 마지막
토큰만 쓴다 (`seq = torch.cat(...)` → `self.encode(seq)` → `h_seq[:, -1, :]`).
13 스텝 동안 길이 53~65 를 통째로 통과시키므로 토큰 수가 Sum(53..65) = 767 이다.
실제로 필요한 건 13 개다 (59.0 배).

왜 마지막 토큰만으로 충분한가
-----------------------------
본모형 인코더는 per-step MLP 다 (`train_garch_flow.MLPEncoder.forward` =
`self.net(x)`, 시점별 독립).  `encode()` 는 `input_proj(x) + pos_emb[:L_cur]` 뒤
그 MLP 를 통과시키는데 위치 임베딩도 인덱스별이라 시점 간 섞임이 없다.
실측: MLPEncoder 마지막토큰 차이 0.0 / LSTM 0.1159 / Transformer 2.1765.

그래서 아래 두 조건을 **둘 다** 만족할 때만 가속 경로를 탄다. 아니면 원본 위임.
  · 인코더가 `MLPEncoder`
  · `ENCODER_MASK_SP` 가 False — True 면 원본이 미래 sp 를 0 마스크하는데
    가속판은 그 분기를 못 타서 최대 절대차 2.36 으로 갈린다
    (`run_macroenc.py:51` 이 True 를 쓴다)

두 클래스를 따로 패치한다
-------------------------
`fpath_model.MambaFlowARFpath` 가 `ar_sample` 을 오버라이드하고(:150) ctx 맨 뒤에
미래경로 요약 `fut_sum_rep` 을 붙인다.  베이스만 패치하면 rebind 순서에 따라
가속이 조용히 안 먹거나(경고 없음) 차원이 안 맞아 크래시한다.  그래서 베이스와
Fpath 를 각각의 가속판으로 교체한다.

실측 배속 (CPU, torch 2.11.0+cpu, 합성 텐서)
--------------------------------------------
4.3~6.1x.  토큰 수 비 59 배가 그대로 오지 않는 이유는 `encode` 가 원본 실행시간의
74~75% 이고 `flow.sample` 이 25% 라 Amdahl 상한이 3.96x 이기 때문이다.  상한을
넘는 건 `torch.cat` 재할당 비용까지 같이 사라져서다.  GPU 에서는 큰 배치 인코딩이
더 잘 병렬화되므로 배속이 이보다 낮을 수 있다 (폴드 CSV 가 없어 미실측).

사용
----
    import ar_sample_fast
    ar_sample_fast.patch()      # 적중한 클래스를 출력한다
    ar_sample_fast.unpatch()

검증
----
    !python colab/dual_3ch/ar_sample_fast.py
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import train_garch_flow as T                                      # noqa: E402

_ORIG_BASE = None
_ORIG_FPATH = None


def _fast_ok(self):
    """가속 경로를 탈 수 있는가.  per-step MLP 이고 sp 마스킹이 꺼져 있을 때만."""
    return (isinstance(getattr(self, "mamba", None), T.MLPEncoder)
            and not T.ENCODER_MASK_SP)


def _rollout(self, x_past, future_tbill_z, last_past_sp_z, n_sim,
             extra_context, future_macro_z, fut_sum_rep=None,
             summary_only=False):
    """공통 롤아웃.  fut_sum_rep 이 주어지면 ctx 맨 뒤에 붙인다 (Fpath 판)."""
    device = x_past.device
    B = x_past.shape[0]
    N_CH = x_past.shape[-1]
    PAST_LEN, FUTURE_LEN = T.PAST_LEN, T.FUTURE_LEN

    tbill_fut = future_tbill_z.unsqueeze(1).expand(B, n_sim, -1).contiguous()
    tbill_fut = tbill_fut.view(B * n_sim, FUTURE_LEN)
    last_sp = last_past_sp_z.unsqueeze(1).expand(B, n_sim).contiguous().view(-1)

    if self.extra_context_dim > 0:
        if extra_context is None:
            raise ValueError("extra_context required when extra_context_dim > 0")
        extra_rep = extra_context.unsqueeze(1).expand(
            B, n_sim, -1).contiguous().view(B * n_sim, self.extra_context_dim)
    else:
        extra_rep = None

    # 과거 요약은 원점 고정 — 루프 밖 1 회 (원본과 동일).
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
        fm = fm.view(B * n_sim, FUTURE_LEN, len(_unmask_idx))
    else:
        fm = None

    _dir_idx = ([T.COND_COLS.index(c) for c in T.DIRECT_FUTURE_COLS
                 if c in T.COND_COLS] if self.direct_future_dim > 0 else [])

    sampled = []
    for tau in range(FUTURE_LEN):
        next_input = torch.zeros(B * n_sim, 1, N_CH,
                                 device=device, dtype=x_past.dtype)
        next_input[:, 0, T.SP_CH] = last_sp
        if not summary_only:
            next_input[:, 0, T.TBILL_CH] = tbill_fut[:, tau]
            if fm is not None:
                for _j, _ch in enumerate(_unmask_idx):
                    next_input[:, 0, _ch] = fm[:, tau, _j]

        # ── 원본과 다른 유일한 지점: 전체 seq 대신 이 토큰 하나만 인코딩.
        #    원본은 cat 후 L_cur = PAST_LEN+tau+1 이고 pos_emb[:L_cur] 의 마지막
        #    인덱스가 PAST_LEN+tau 이므로 아래와 같다.
        pos = PAST_LEN + tau
        h_tau = self.mamba(self.input_proj(next_input)
                           + self.pos_emb[pos:pos + 1].unsqueeze(0))[:, 0, :]

        ctx_parts = [h_tau]
        if self.use_past_summary:
            ctx_parts.append(past_sum_rep)
        if self.direct_prev_dim > 0:
            ctx_parts.append(last_sp.unsqueeze(-1))
        if self.direct_future_dim > 0:
            ctx_parts.append(next_input[:, 0, _dir_idx])
        if self.extra_context_dim > 0:
            ctx_parts.append(extra_rep)
        if fut_sum_rep is not None:
            ctx_parts.append(fut_sum_rep)

        ctx_tau = torch.cat(ctx_parts, dim=-1) if len(ctx_parts) > 1 else h_tau
        sp_z = self.flow.sample(1, context=ctx_tau).squeeze(-1).squeeze(-1)
        sampled.append(sp_z)
        last_sp = sp_z

    return torch.stack(sampled, dim=1).view(B, n_sim, FUTURE_LEN)


@torch.no_grad()
def ar_sample_base(self, x_past, future_tbill_z, last_past_sp_z, n_sim,
                   extra_context=None, future_macro_z=None):
    """베이스 MambaFlowAR 용 가속판."""
    if not _fast_ok(self):
        return _ORIG_BASE(self, x_past, future_tbill_z, last_past_sp_z, n_sim,
                          extra_context=extra_context,
                          future_macro_z=future_macro_z)
    return _rollout(self, x_past, future_tbill_z, last_past_sp_z, n_sim,
                    extra_context, future_macro_z)


@torch.no_grad()
def ar_sample_fpath(self, x_past, future_tbill_z, last_past_sp_z, n_sim,
                    extra_context=None, future_macro_z=None):
    """MambaFlowARFpath 용 가속판.  ctx 맨 뒤 미래경로 요약을 그대로 유지한다."""
    if not _fast_ok(self):
        return _ORIG_FPATH(self, x_past, future_tbill_z, last_past_sp_z, n_sim,
                           extra_context=extra_context,
                           future_macro_z=future_macro_z)
    B = x_past.shape[0]
    fut3 = future_tbill_z.unsqueeze(-1)
    if future_macro_z is not None and future_macro_z.shape[-1] > 0:
        fut3 = torch.cat([fut3, future_macro_z], dim=-1)
    fut_sum0 = self._future_summary_from_seq(fut3)
    fut_sum_rep = fut_sum0.unsqueeze(1).expand(
        B, n_sim, -1).contiguous().view(B * n_sim, -1)
    return _rollout(self, x_past, future_tbill_z, last_past_sp_z, n_sim,
                    extra_context, future_macro_z,
                    fut_sum_rep=fut_sum_rep,
                    summary_only=self._summary_only)


def patch():
    """베이스와 Fpath 양쪽의 ar_sample 을 가속판으로 교체하고 적중 클래스를 찍는다."""
    global _ORIG_BASE, _ORIG_FPATH
    hit = []
    done = set()          # 같은 클래스를 두 번 잡지 않는다.

    # Fpath 를 먼저 잡는다.  run_lr_sweep 이 T.MambaFlowAR 을 Fpath 로 덮어쓰므로
    # 베이스 경로가 같은 클래스를 가리킬 수 있고, 그때 베이스 가속판을 걸면
    # fut_sum_rep 이 빠져 차원이 안 맞는다.
    try:
        import fpath_model
        fp = fpath_model.MambaFlowARFpath
        if "ar_sample" in fp.__dict__:
            if _ORIG_FPATH is None:
                _ORIG_FPATH = fp.__dict__["ar_sample"]
            fp.ar_sample = ar_sample_fpath
            hit.append(fp.__name__); done.add(id(fp))
    except Exception as e:                                        # noqa: BLE001
        print(f"[ar_sample_fast] fpath_model 패치 실패: {e!r}")

    base_cls = T.MambaFlowAR
    if id(base_cls) not in done and "ar_sample" in base_cls.__dict__:
        if _ORIG_BASE is None:
            _ORIG_BASE = base_cls.__dict__["ar_sample"]
        base_cls.ar_sample = ar_sample_base
        hit.append(base_cls.__name__)
    print(f"[ar_sample_fast] 패치 적중: {hit or '없음'}  "
          f"(MLP 인코더 + ENCODER_MASK_SP=False 일 때만 가속)")
    return hit


def unpatch():
    global _ORIG_BASE, _ORIG_FPATH
    if _ORIG_FPATH is not None:
        import fpath_model
        fpath_model.MambaFlowARFpath.ar_sample = _ORIG_FPATH
    if _ORIG_BASE is not None:
        T.MambaFlowAR.ar_sample = _ORIG_BASE
    print("[ar_sample_fast] 원복")


# =====================================================================
# 회귀 테스트 — 원본과 값이 같은지 + 마스킹 켠 경우 위임하는지 + 속도
# =====================================================================
def _mk(cls, extra_dim=1):
    cols = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
    T.COND_COLS = cols; T.N_CHANNELS = len(cols)
    T.SP_CH = cols.index("sp_return"); T.TBILL_CH = cols.index("tbill_wr")
    T.MACRO_CH = [i for i in range(len(cols)) if i not in (T.SP_CH, T.TBILL_CH)]
    T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
    torch.manual_seed(2026)
    return cls(d_input=len(cols), d_model=128, mlp_num_layers=4,
               encoder_type="mlp", n_flow_layers=4, n_flow_hidden=128,
               dropout=0.0, extra_context_dim=extra_dim,
               direct_prev_return=True, use_past_summary=True,
               past_encoder_type="mlp", past_summary_dim=64).eval()


def _case(name, cls, orig_fn, fast_fn, mask_sp, B=4, n_sim=300, seed=2026):
    T.ENCODER_MASK_SP = mask_sp
    m = _mk(cls)
    x = torch.randn(B, T.PAST_LEN, T.N_CHANNELS)
    tb = torch.randn(B, T.FUTURE_LEN); lsp = torch.randn(B)
    ex = torch.randn(B, 1); fmz = torch.randn(B, T.FUTURE_LEN, 1)

    def run(fn):
        torch.manual_seed(seed)
        t0 = time.time()
        with torch.no_grad():
            out = fn(m, x, tb, lsp, n_sim, extra_context=ex, future_macro_z=fmz)
        return out, time.time() - t0

    a, ta = run(orig_fn)
    b, tbb = run(fast_fn)
    d = (a - b).abs().max().item()
    print(f"  {name:34} mask_sp={str(mask_sp):5}  차이 {d:.3e}  "
          f"원본 {ta:5.2f}s → {tbb:5.2f}s ({ta / max(tbb, 1e-9):4.2f}x)  "
          f"{'OK' if d == 0.0 else '불일치'}")
    return d


def _verify():
    import fpath_model
    fpath_model.FUTURE_SUMMARY_DIM = int(os.environ.get("FPATH_DIM", "2"))
    fpath_model.FUTURE_SUMMARY_HIDDEN = int(os.environ.get("FPATH_HIDDEN", "16"))
    fpath_model.FPATH_SUMMARY_ONLY = False

    global _ORIG_BASE, _ORIG_FPATH
    _ORIG_BASE = T.MambaFlowAR.__dict__["ar_sample"]
    _ORIG_FPATH = fpath_model.MambaFlowARFpath.__dict__["ar_sample"]

    print("=" * 78)
    print("AR 롤아웃 가속 회귀 테스트 — 원본과 값이 같아야 하고, 마스킹 켜면 위임")
    print("=" * 78)
    ds = []
    for mask in (False, True):
        ds.append(_case("MambaFlowAR (베이스)", T.MambaFlowAR,
                        _ORIG_BASE, ar_sample_base, mask))
        ds.append(_case("MambaFlowARFpath", fpath_model.MambaFlowARFpath,
                        _ORIG_FPATH, ar_sample_fpath, mask))
    T.ENCODER_MASK_SP = False
    bad = [d for d in ds if d != 0.0]
    print(f"\n{'전부 일치' if not bad else f'{len(bad)} 건 불일치'} "
          f"(mask_sp=True 는 원본에 위임하므로 차이 0 이어야 정상)")
    return ds


if __name__ == "__main__":
    _verify()
