# -*- coding: utf-8 -*-
"""flow.sample 내부 시간 분해 + 난수 배치화 등가성 확인 (GPU).

배경
----
셀 시간의 대부분이 검증 스크리닝이다 (ar_sample_fast 적용 후 실측):
  FLOW_VAL_EVERY=2  (스크리닝 16 회) → 540 초
  FLOW_VAL_EVERY=30 (스크리닝  2 회) → 184 초
스크리닝 간격은 선택 에폭을 바꾸므로 못 건드린다.

flow.sample 은 두 구간이다.
  · _distribution.sample → SkewStudentT._inv_cdf (train_garch_flow.py:207-220)
      매 호출 u 를 CPU 로 내리고 scipy.stats.t.ppf 를 2 회 부른 뒤 GPU 로 올린다.
      AR 13 스텝 x 33 청크 = 429 회 왕복.
  · _transform.inverse → RQ-NSF (torch, GPU)

CPU 측정에서 _inv_cdf 가 flow.sample 의 36.9~40.3% 였다.  GPU 에서는 torch 쪽만
빨라지므로 비중이 더 커질 수 있다.  실제 비중을 재야 다음 수를 정할 수 있다.

무엇을 재나
-----------
1. GPU 에서 _inv_cdf 대 _transform.inverse 시간 비중
2. GPU↔CPU 왕복 횟수를 줄일 수 있는지 — u 는 context 와 무관하므로
   (`_sample`:227 이 torch.rand 로 뽑고 _inv_cdf 는 u 만 받는다) 13 스텝치를
   한 번에 뽑아 scipy 를 1 회만 부를 수 있다.  단 그러려면 난수 소비가
   같아야 한다.  CPU 에서는 torch.rand(8000) x13 == torch.rand(104000) 이
   최대차 0.0 으로 확인됐다.  CUDA philox 는 블록 단위 오프셋이라 다를 수
   있으므로 GPU 에서 다시 확인한다.
3. t.ppf 를 13 회 부르는 것과 1 회 부르는 것의 시간 차 (CPU 연산 자체)

사용
----
    !python colab/dual_3ch/prof_flow_sample.py
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("PS_BASE", "fpath_novol")
os.environ.setdefault("FPATH_DIM", "2")

from rawvol_helpers import patch_rawvol                            # noqa: E402
patch_rawvol()
import run_lr_sweep as LRS                                         # noqa: E402
import train_garch_flow as T                                       # noqa: E402
import fpath_model                                                 # noqa: E402

N_SIM = int(os.environ.get("PF_NSIM", "1000"))
CHUNK = int(os.environ.get("PF_CHUNK", "8"))
STEPS = 13
SEED = 2026


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def build():
    LRS.set_cond_cols(LRS.ENC_FULL, "tbill_wr")
    T.MASK_FUTURE_TBILL = False
    T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
    T.ENCODER_MASK_SP = False
    torch.manual_seed(SEED)
    s = dict(LRS.LOCKED)
    m = fpath_model.MambaFlowARFpath(
        d_input=T.N_CHANNELS, d_model=s["d_model"],
        mlp_num_layers=s["mlp_num_layers"], encoder_type=s["encoder_type"],
        n_flow_layers=s["n_flow_layers"], n_flow_hidden=s["n_flow_hidden"],
        dropout=s["dropout"], extra_context_dim=1,
        direct_prev_return=s["direct_prev_return"],
        use_past_summary=s["use_past_summary"],
        past_encoder_type=s["past_encoder_type"],
        past_summary_dim=s["past_summary_dim"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), dev


def main():
    m, dev = build()
    rows = CHUNK * N_SIM
    print("=" * 78)
    print(f"flow.sample 시간 분해  device={dev}  rows={rows:,} (chunk {CHUNK} "
          f"x n_sim {N_SIM})  steps={STEPS}")
    print("=" * 78)

    ctx = torch.randn(rows, m.flow_context_dim, device=dev)
    dist = m.flow._distribution
    trans = m.flow._transform
    emb = m.flow._embedding_net

    # ---------- 1. 구간별 시간 ----------
    with torch.no_grad():
        for _ in range(2):                      # 워밍업
            _ = m.flow.sample(1, context=ctx)
        sync(dev)

        t_inv = t_tr = t_all = 0.0
        for _ in range(STEPS):
            sync(dev); t0 = time.time()
            out = m.flow.sample(1, context=ctx)
            sync(dev); t_all += time.time() - t0

            ec = emb(ctx)
            sync(dev); t0 = time.time()
            noise = dist.sample(1, context=ec)
            sync(dev); t_inv += time.time() - t0

            from nflows.utils import torchutils
            n2 = torchutils.merge_leading_dims(noise, num_dims=2)
            e2 = torchutils.repeat_rows(ec, num_reps=1)
            sync(dev); t0 = time.time()
            _ = trans.inverse(n2, context=e2)
            sync(dev); t_tr += time.time() - t0

    print(f"  flow.sample 전체        {t_all:7.3f}s  ({STEPS} 스텝)")
    print(f"  _distribution.sample    {t_inv:7.3f}s  ({t_inv / t_all * 100:5.1f}%)"
          f"  ← scipy _inv_cdf, GPU↔CPU 왕복")
    print(f"  _transform.inverse      {t_tr:7.3f}s  ({t_tr / t_all * 100:5.1f}%)"
          f"  ← RQ-NSF, GPU")

    # ---------- 2. 난수 배치화가 같은 수열인가 ----------
    torch.manual_seed(SEED)
    a = torch.cat([torch.rand(rows, 1, device=dev) for _ in range(STEPS)])
    torch.manual_seed(SEED)
    b = torch.rand(STEPS * rows, 1, device=dev)
    d = (a - b).abs().max().item()
    print(f"\n  torch.rand {STEPS}회 vs 1회 최대차 = {d:.3e}  "
          f"{'같은 수열 — 배치화 가능' if d == 0.0 else '다른 수열 — 배치화 불가'}")

    # ---------- 3. scipy 호출 13회 vs 1회 ----------
    from scipy.stats import t as _t
    un = np.clip(np.random.default_rng(0).random(STEPS * rows), 1e-6, 1 - 1e-6)
    t0 = time.time()
    for i in range(STEPS):
        _ = _t.ppf(un[i * rows:(i + 1) * rows], 7.0)
    t_many = time.time() - t0
    t0 = time.time()
    _ = _t.ppf(un, 7.0)
    t_one = time.time() - t0
    print(f"  scipy t.ppf  {STEPS}회 {t_many:.3f}s / 1회 {t_one:.3f}s  "
          f"({t_many / max(t_one, 1e-9):.2f}x)")

    print("\n[읽는 법]")
    print("  · _distribution.sample 비중이 크고 난수 배치화가 가능하면,")
    print("    13 스텝치 u 를 미리 뽑아 scipy 를 1 회만 불러 왕복을 13→1 로 줄인다.")
    print("    난수 수열이 같으므로 값은 보존된다.")
    print("  · 비중이 작으면 남은 이득이 없다.")


if __name__ == "__main__":
    main()
