# -*- coding: utf-8 -*-
"""MAC-Flow flow base 의 Hansen skew-t lambda 점검.

확인 목적: 학습 후 base lambda 가 0 근처인가.
0 이면 왜도는 spline 이 만들고 있는 것이고, base lambda 는 AdamW weight decay 에
눌렸을 수 있다 (train_garch_flow.py:1086 이 파라미터 그룹을 나누지 않아
스칼라 _lam_raw 에도 decay 가 걸린다.  Table 9 의 MLP 인코더 weight_decay = 0.5).

사용:  !python colab/check_lam.py
"""
import glob
import math
import os
import sys

import torch

_ROOTS = ["result", "colab/dual_3ch/result",
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")]
_PATS = ["*fpath_novol*best.pt", "*garch_flow_ar*best.pt", "*best.pt"]
CAND = [os.path.join(r, p) for p in _PATS for r in _ROOTS]


def find_files():
    for pat in CAND:
        fs = sorted(glob.glob(pat))
        if fs:
            print(f"패턴: {pat}   파일수: {len(fs)}")
            return fs
    return []


def find_wd(ck):
    """체크포인트 meta 어딘가에 저장된 weight_decay 를 찾는다."""
    if not isinstance(ck, dict):
        return None
    for k, v in ck.items():
        if isinstance(k, str) and "decay" in k:
            return v
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(k2, str) and "decay" in k2:
                    return v2
    return None


def main():
    files = find_files()
    if not files:
        print("체크포인트 없음 — 경로 확인 필요")
        return 1

    rows = []
    for f in files:
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:                                    # noqa: BLE001
            print(f"{os.path.basename(f)}  로드실패  {e}")
            continue
        sd = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
        if not isinstance(sd, dict):
            continue
        hits = {k: v for k, v in sd.items()
                if "lam" in k.lower() and hasattr(v, "reshape")}
        if not hits:
            continue
        wd = find_wd(ck)
        for k, v in hits.items():
            raw = float(v.reshape(-1)[0])
            rows.append((os.path.basename(f), k, raw, math.tanh(raw) * 0.99, wd))

    if not rows:
        print("'lam' 을 포함한 파라미터를 가진 체크포인트가 없다.")
        return 1

    hdr = "{:56} {:26} {:>10} {:>10}  {}".format(
        "checkpoint", "param", "raw", "lambda", "wd")
    print()
    print(hdr)
    print("-" * len(hdr))
    for fn, k, raw, lam, wd in rows:
        print("{:56} {:26} {:+10.6f} {:+10.6f}  {}".format(fn, k, raw, lam, wd))

    lams = [r[3] for r in rows]
    print()
    print("lambda  n={}  min={:+.6f}  max={:+.6f}  mean={:+.6f}".format(
        len(lams), min(lams), max(lams), sum(lams) / len(lams)))
    print("|lambda| < 0.01 이면 decay 에 눌린 것으로 본다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
