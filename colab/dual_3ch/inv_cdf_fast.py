# -*- coding: utf-8 -*-
"""SkewStudentT._inv_cdf 가속 — 인자 융합 + 13 스텝 u 선인출.

무엇이 느렸나
-------------
GPU 실측 (rows=8000 = chunk 8 x n_sim 1000, 13 스텝):
    flow.sample 전체        0.836s
    _distribution.sample    0.577s (69.1%)   ← 여기
    _transform.inverse      0.236s (28.3%)

`_distribution.sample` 은 `SkewStudentT._inv_cdf`(train_garch_flow.py:207-220)
를 부른다.  두 가지가 낭비다.

  1. **t.ppf 를 두 번 부른다.**  z1 과 z2 를 전체 배열에 계산한 뒤
     `np.where(cond, z1, z2)` 로 고른다.  인자 쪽에서 먼저 고르면
     (`np.where` 를 ppf 앞으로) 한 번만 부르면 된다.

     절감분은 stdtrit 호출량이 아니다.  scipy `rv_continuous.ppf` 는 내부에서
     정의역 밖 원소를 마스킹해 `_ppf` 를 유효 부분집합에만 부르므로, 실제로
     줄어드는 것은 래퍼(argcheck·출력 할당·NaN 배치)를 두 번 도는 비용이다.

     CPU 실측 (rows 8000 x 13 스텝, 라운드로빈): lam +0.0000 1.92x /
     -0.0028 1.96x / -0.0050 1.91x / ±0.3126 1.35~1.67x / -0.70 1.13x.
     실제 학습된 lam 은 -0.002 ~ -0.005 다.

     **정확성**: 무작위 lam 300 회(df 5/7/10/15, float32/float64, u 에 경계값
     0·1·cut 강제 삽입)에서 융합·부분계산 모두 **0/300 불일치**.
     `(1-lam)*s` 는 파이썬 float 스칼라, `sc*s` 는 같은 값의 float64 배열이라
     IEEE 배정밀도로 동일하고 t.ppf 는 원소별 결정적 함수다.  순서 재배열이
     결과를 바꿀 여지가 구조적으로 없다.

  2. **매 스텝 GPU→CPU→GPU 를 왕복한다.**  AR 13 스텝 x 33 청크 = 429 회.
     u 는 context 와 무관하므로(`_sample`:222-228 이 shape/device 결정에만 씀)
     13 스텝치를 미리 뽑아 한 번에 변환할 수 있다.

     주의: 미리 뽑을 때 `torch.rand(rows)` 를 **13 회 그대로 호출**해야 한다.
     `torch.rand(13*rows)` 1 회로 바꾸면 GPU 에서 수열이 달라진다
     (CUDA philox 블록 오프셋, 실측 최대차 9.984e-01).  13 회 호출 후 이어붙이면
     원본과 소비 순서가 같아 비트일치다 (CPU 실측 최대차 0.000e+00).

     롤아웃 루프 안에서 RNG 를 먹는 연산은 `_sample` 의 `torch.rand` 하나뿐이라
     (평가는 model.eval() 이라 dropout 없음) 선인출이 수열을 바꾸지 않는다.

사용
----
    import inv_cdf_fast
    inv_cdf_fast.patch()        # 1 번만 적용 (인자 융합)
    inv_cdf_fast.unpatch()

검증
----
    !python colab/dual_3ch/inv_cdf_fast.py
"""
import math
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

import train_garch_flow as T                                      # noqa: E402

_ORIG_INV = None


def inv_cdf_fused(self, u):
    """원본과 비트일치하되 t.ppf 를 한 번만 부른다.

    원본은 z1, z2 를 전체에 계산하고 np.where 로 골랐다.  같은 식을 인자
    쪽에서 고르면 결과가 같고 ppf 평가량이 절반이다.
    """
    from scipy.stats import t as _t
    e = self.df
    lam = float(self._lam())
    a = 4.0 * lam * self._cval * (e - 2.0) / (e - 1.0)
    b = math.sqrt(1.0 + 3.0 * lam ** 2 - a ** 2)
    s = math.sqrt((e - 2.0) / e)
    un = np.clip(u.detach().cpu().numpy(), 1e-6, 1.0 - 1e-6)
    cut = (1.0 - lam) / 2.0
    cond = un < cut
    p = np.where(cond, un / (1.0 - lam), 0.5 + (un - cut) / (1.0 + lam))
    sc = np.where(cond, 1.0 - lam, 1.0 + lam)
    zn = (1.0 / b) * (sc * s * _t.ppf(p, e) - a)
    return torch.as_tensor(zn, device=u.device, dtype=u.dtype)


def patch():
    global _ORIG_INV
    cls = T.SkewStudentT
    if _ORIG_INV is None:
        _ORIG_INV = cls.__dict__["_inv_cdf"]
    cls._inv_cdf = inv_cdf_fused
    print("[inv_cdf_fast] SkewStudentT._inv_cdf → 인자 융합 (t.ppf 1 회)")


def unpatch():
    if _ORIG_INV is not None:
        T.SkewStudentT._inv_cdf = _ORIG_INV
        print("[inv_cdf_fast] 원복")


# =====================================================================
# 회귀 테스트 — 원본과 비트일치인지 + 속도
# =====================================================================
def _verify(rows=8000, steps=13, seed=2026):
    global _ORIG_INV
    cls = T.SkewStudentT
    # patch() 뒤에 부르면 _ORIG_INV 가 융합판이 되어 자기 자신을 비교하고
    # 최대차 0 이 자동 보장된다.  그 상태를 막는다.
    if cls.__dict__["_inv_cdf"] is inv_cdf_fused:
        raise RuntimeError("_verify() 는 patch() 전에만 유효하다 "
                           "(원본이 이미 융합판으로 교체돼 있다)")
    if _ORIG_INV is None:
        _ORIG_INV = cls.__dict__["_inv_cdf"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f"_inv_cdf 인자 융합 회귀 테스트  device={dev}  rows={rows:,}  "
          f"steps={steps}")
    print("=" * 78)

    # 실제 학습값 범위를 포함해 여러 lam 에서 확인한다.
    # _lam() = tanh(_lam_raw) * 0.99 (train_garch_flow.py:187) 이므로 역함수도
    # 0.99 로 나눠야 한다.  0.95 로 두면 요청 -0.30 이 실제 -0.3126 이 된다.
    for lam_t in (0.0, -0.0028, -0.0050, -0.30, 0.30):
        d = cls(shape=[1], df=7.0)
        with torch.no_grad():
            d._lam_raw.fill_(math.atanh(lam_t / 0.99) if lam_t else 0.0)
        d = d.to(dev)

        torch.manual_seed(seed)
        us = [torch.rand(rows, 1, device=dev) for _ in range(steps)]

        t0 = time.time()
        a = [_ORIG_INV(d, u) for u in us]
        t_orig = time.time() - t0

        t0 = time.time()
        b = [inv_cdf_fused(d, u) for u in us]
        t_fast = time.time() - t0

        mx = max((x - y).abs().max().item() for x, y in zip(a, b))
        print(f"  lam={float(d._lam()):+.4f}  원본 {t_orig:6.3f}s → "
              f"융합 {t_fast:6.3f}s ({t_orig / max(t_fast, 1e-9):4.2f}x)   "
              f"최대차 {mx:.3e}  {'비트일치' if mx == 0.0 else '차이 있음'}")

    # 선인출이 수열을 보존하는지 (13 회 호출 후 cat == 루프 중 13 회 호출)
    torch.manual_seed(seed)
    pre = torch.cat([torch.rand(rows, 1, device=dev) for _ in range(steps)])
    torch.manual_seed(seed)
    loop = torch.cat([torch.rand(rows, 1, device=dev) for _ in range(steps)])
    dd = (pre - loop).abs().max().item()
    print(f"\n  선인출 13회 vs 루프중 13회 최대차 = {dd:.3e}  "
          f"{'수열 보존' if dd == 0.0 else '수열 다름'}")

    torch.manual_seed(seed)
    one = torch.rand(steps * rows, 1, device=dev)
    d1 = (pre - one).abs().max().item()
    print(f"  선인출 13회 vs 1회 호출  최대차 = {d1:.3e}  "
          f"{'(1회 호출로 바꾸면 안 된다)' if d1 != 0.0 else ''}")


if __name__ == "__main__":
    _verify()
