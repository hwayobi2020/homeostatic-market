# -*- coding: utf-8 -*-
"""flow base skew 파라미터(_lam_raw)를 weight decay 에서 제외하고 재학습.

배경
----
train_garch_flow.py:1086 이 AdamW 에 model.parameters() 를 통째로 넘겨
파라미터 그룹을 나누지 않는다.  그래서 flow base 의 Hansen skew-t 왜도
파라미터 _lam_raw (726,947 개 중 스칼라 1개, 초기값 0) 에도 weight decay 가
그대로 걸린다.  본모형 weight_decay = 0.5.

실측(check_lam.py, 체크포인트 60개 = 4 fold x 5 seed x fpath_dim 3종):
    |lambda| 최대 0.009158,  평균 -0.001863,  60개 전부 |lambda| < 0.01
즉 초기값 0 에서 사실상 움직이지 않았다.  60개 중 음수 46개로 왼쪽 꼬리
방향의 힘은 있으나 decay 가 상쇄한다.

이 스크립트
-----------
train_garch_flow.py 를 수정하지 않는다.  torch.optim.AdamW 를 몽키패치해
0차원(스칼라) 파라미터만 weight_decay=0 그룹으로 분리한다.
run_full_fpath.py 와 동일한 LOCKED 본모형 spec 으로 학습하고,
태그에 _lamnd 를 붙여 기존 결과를 보존한다.

사용
----
    !python colab/dual_3ch/run_lam_nodecay.py
    # 폴드/시드 한정:
    !LAMND_FOLDS=F_gfc LAMND_SEEDS=2026 python colab/dual_3ch/run_lam_nodecay.py

비교
----
    check_lam.py 로 *_lamnd_* 체크포인트의 lambda 를 다시 찍고,
    agg_section4 / tail_ablation 에서 왜도를 기존 태그와 대조한다.
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from rawvol_helpers import patch_rawvol                              # noqa: E402
patch_rawvol()
import train_garch_flow as T                                        # noqa: E402
import fpath_model                                                  # noqa: E402

# 본모형 = rvAbl_full_fpath_novol_d2 (MODEL_SPEC 확인).
fpath_model.FUTURE_SUMMARY_DIM = int(os.environ.get("FPATH_DIM", "2"))
fpath_model.FUTURE_SUMMARY_HIDDEN = int(os.environ.get("FPATH_HIDDEN", "16"))
fpath_model.FPATH_SUMMARY_ONLY = False
T.MambaFlowAR = fpath_model.MambaFlowARFpath                        # ★ monkeypatch

from train_garch_flow import main_worker                            # noqa: E402

# ---------------------------------------------------------------------------
# AdamW 몽키패치 — 스칼라 파라미터만 no-decay 그룹으로 분리
# ---------------------------------------------------------------------------
_ORIG_ADAMW = torch.optim.AdamW


def _adamw_split(params, **kw):
    ps = list(params)
    if ps and isinstance(ps[0], dict):          # 이미 그룹이면 손대지 않는다
        return _ORIG_ADAMW(ps, **kw)
    wd = kw.pop("weight_decay", 0.0)
    scalar = [p for p in ps if getattr(p, "ndim", 1) == 0]
    other = [p for p in ps if getattr(p, "ndim", 1) != 0]
    print(f"    [lam_nodecay] no-decay 스칼라 {len(scalar)}개 / "
          f"decay {len(other)}개 (wd={wd})")
    if len(scalar) != 1:
        print(f"    [warn] 0차원 파라미터가 {len(scalar)}개다. "
              f"_lam_raw 하나만 나와야 한다.")
    return _ORIG_ADAMW(
        [{"params": other, "weight_decay": wd},
         {"params": scalar, "weight_decay": 0.0}], **kw)


torch.optim.AdamW = _adamw_split
T.torch.optim.AdamW = _adamw_split

# ---------------------------------------------------------------------------
RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = [f for f in os.environ.get(
    "LAMND_FOLDS", "F_gfc,F_long_A,F_long_B_origin,F_long").split(",") if f.strip()]
SEEDS = [int(s) for s in os.environ.get(
    "LAMND_SEEDS", "2026,2027,2028,2029,2030").split(",") if s.strip()]

ENC_FULL = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]

# run_full_fpath.py 의 LOCKED + NOVOL=1 과 동일 (본모형 spec).
LOCKED = dict(encoder_type="mlp", d_model=128, mlp_num_layers=4,
              extra_context_channels="sp_skew_13w",
              direct_prev_return=True, use_past_summary=True,
              past_encoder_type="mlp", past_summary_dim=64,
              dropout=0.2, n_flow_layers=4, n_flow_hidden=128, weight_decay=0.5)


def set_cond_cols(cols, rate_col):
    cols = list(cols)
    T.COND_COLS = cols
    T.N_CHANNELS = len(cols)
    T.SP_CH = cols.index("sp_return")
    T.TBILL_CH = cols.index(rate_col)
    T.MACRO_CH = [c for c in range(len(cols)) if c not in (T.SP_CH, T.TBILL_CH)]
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass


def main():
    print("#" * 96)
    print("# lam_nodecay — flow base _lam_raw 를 weight decay 에서 제외하고 재학습")
    print(f"#  본모형 spec + fpath_dim={fpath_model.FUTURE_SUMMARY_DIM}  "
          f"folds={FOLDS}  seeds={SEEDS}")
    print("#  비교 기준: rvAbl_full_fpath_novol_d2_s{seed}  (기존, decay 걸린 것)")
    print("#" * 96)

    set_cond_cols(ENC_FULL, "tbill_wr")
    T.MASK_FUTURE_TBILL = False
    T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
    T.ENCODER_MASK_SP = False

    for fold in FOLDS:
        if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                   for s in ("train", "val", "test")):
            print(f"  [skip {fold}] fold CSV 없음")
            continue
        for seed in SEEDS:
            tag = (f"rvAbl_full_fpath_novol_lamnd"
                   f"_d{fpath_model.FUTURE_SUMMARY_DIM}_s{seed}")
            sp = os.path.join(RESULT_DIR,
                              f"garch_flow_ar_{tag}_{fold}_summary.json")
            if os.path.exists(sp):
                print(f"  [skip] {os.path.basename(sp)}")
                continue
            spec = dict(LOCKED)
            spec.update(fold=fold, seed=seed, tag=tag)
            print(f"  [run] {tag} fold={fold}")
            t0 = time.time()
            try:
                main_worker(spec)
                print(f"    done ({time.time() - t0:.0f}s)")
            except Exception as e:                                   # noqa: BLE001
                print(f"    [FAIL] {tag} {fold}: {e!r}")

    print("\n[done] lam_nodecay 학습 완료.")
    print("  lambda 확인:  !python colab/dual_3ch/check_lam.py")
    print("  왜도 대조  :  agg_section4 에 rvAbl_full_fpath_novol_lamnd_d2 태그 추가")


if __name__ == "__main__":
    main()
