# -*- coding: utf-8 -*-
"""MAC-Flow 학습 전용 — 지정한 설정으로 4 폴드 x N 시드 체크포인트를 만든다.

튜닝(`run_tuning_all.py`)이나 실험(`run_thin_compare.py`)과 분리한다.
이 스크립트는 격자 탐색도 지표 비교도 하지 않는다.  학습만 한다.

산출: `result/garch_flow_ar_{tag}_s{seed}_{fold}_best.pt` 와 `_summary.json`.
태그는 `run_thin_compare.py` 의 `FLOW_TAG` 로 그대로 넘길 수 있는 형태다.

사용
----
    # 게재판(4 층 / hidden 128) 5 시드
    !python colab/dual_3ch/train_flow_seeds.py

    # 8 층 / hidden 32 5 시드
    !TF_LAYERS=8 TF_HIDDEN=32 python colab/dual_3ch/train_flow_seeds.py

    # 그 결과로 §4.1 비교 (게재판 결과는 안 덮는다)
    !FLOW_TAG=rvAbl_full_fpath_novol_fl8fh32_d2 OUT_SUFFIX=_fl8fh32 \
        python colab/dual_3ch/run_thin_compare.py

환경변수
--------
    TF_LAYERS  n_flow_layers   (기본 4  = 게재판)
    TF_HIDDEN  n_flow_hidden   (기본 128 = 게재판)
    TF_LR      학습률          (기본 1e-4)
    TF_WD      weight_decay    (기본 0.5)
    TF_SEEDS   시드 목록       (기본 2026~2030)
    TF_FOLDS   폴드 목록       (기본 4 폴드 전부)
    TF_MAX_EPOCH  최대 에폭    (기본 60 = 게재판)
    FLOW_VAL_CRPS  1 이면 체크포인트를 val CRPS 로 고른다 (기본 0 = val NLL)
    TF_RAWVOL  1(기본)=원점 고정 σ (게재판) / 0=GARCH 재귀 변동성 (태그에 _grec)

기본값은 게재판 그대로다.  인자 없이 돌리면 논문 설정 5 시드가 만들어진다.
"""
import os
import sys
import time

import torch                                                       # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("FPATH_DIM", "2")

# TF_RAWVOL=0 이면 patch 를 걸지 않는다 = train_garch_flow 원본의 GARCH 재귀 변동성
# (σ²_h = ω + α·ε²_{h-1} + β·σ²_{h-1}) 을 그대로 쓴다.  §4.1.2 공정성 점검용이고
# 게재판(§4.2·§4.3)은 TF_RAWVOL=1 (기본, 원점 고정 σ) 이다.
RAWVOL = os.environ.get("TF_RAWVOL", "1") == "1"
from rawvol_helpers import patch_rawvol                            # noqa: E402
if RAWVOL:
    patch_rawvol()
else:
    print("[TF_RAWVOL=0] patch_rawvol 미적용 — GARCH 재귀 변동성 판으로 학습한다")
import train_garch_flow as T                                       # noqa: E402
import fpath_model                                                 # noqa: E402

fpath_model.FUTURE_SUMMARY_DIM = int(os.environ.get("FPATH_DIM", "2"))
fpath_model.FUTURE_SUMMARY_HIDDEN = int(os.environ.get("FPATH_HIDDEN", "16"))
fpath_model.FPATH_SUMMARY_ONLY = False
T.MambaFlowAR = fpath_model.MambaFlowARFpath                       # monkeypatch

from train_garch_flow import main_worker                           # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")

LAYERS = int(os.environ.get("TF_LAYERS", "4"))
HIDDEN = int(os.environ.get("TF_HIDDEN", "128"))
LR = float(os.environ.get("TF_LR", "1e-4"))
WD = float(os.environ.get("TF_WD", "0.5"))
MAX_EPOCH = int(os.environ.get("TF_MAX_EPOCH", "60"))
SEEDS = [int(s) for s in os.environ.get(
    "TF_SEEDS", "2026,2027,2028,2029,2030").split(",") if s.strip()]
FOLDS = [f for f in os.environ.get(
    "TF_FOLDS", "F_gfc,F_long_A,F_long_B_origin,F_long").split(",") if f.strip()]
if not SEEDS or not FOLDS:
    sys.exit("[FATAL] TF_SEEDS / TF_FOLDS 가 비었다")

ENC_FULL = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]

# 게재판 spec (run_full_fpath.LOCKED + FPATH_NOVOL=1).  흐름 헤드 크기만 바꾼다.
LOCKED = dict(encoder_type="mlp", d_model=128, mlp_num_layers=4,
              extra_context_channels="sp_skew_13w",
              direct_prev_return=True, use_past_summary=True,
              past_encoder_type="mlp", past_summary_dim=64,
              dropout=0.2, weight_decay=WD,
              n_flow_layers=LAYERS, n_flow_hidden=HIDDEN, lr=LR,
              max_epoch=MAX_EPOCH)


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


def make_tag(seed):
    """게재판 설정이면 run_full_fpath 와 같은 태그, 하나라도 다르면 접미사를 붙인다.

    lr·wd·max_epoch 도 인코딩한다.  안 그러면 그것만 바꿔 4/128 로 돌렸을 때
    태그가 게재판과 글자 그대로 같아져 게재판 체크포인트를 덮어쓴다.
    """
    sfx = ""
    if LAYERS != 4 or HIDDEN != 128:
        sfx += f"_fl{LAYERS}fh{HIDDEN}"
    if abs(LR - 1e-4) > 1e-12:
        sfx += "_lr" + ("%g" % LR).replace("-", "m").replace(".", "d")
    if abs(WD - 0.5) > 1e-12:
        sfx += "_wd" + ("%g" % WD).replace("-", "m").replace(".", "d")
    if MAX_EPOCH != 60:
        sfx += f"_ep{MAX_EPOCH}"
    if os.environ.get("FLOW_VAL_CRPS") == "1":
        sfx += "_crps"
    if not RAWVOL:
        sfx += "_grec"                       # GARCH recursive vol (게재판과 다른 판본)
    return f"rvAbl_full_fpath_novol{sfx}_d{fpath_model.FUTURE_SUMMARY_DIM}_s{seed}"


def main():
    print("#" * 92)
    print("# MAC-Flow 학습 — 체크포인트만 만든다 (격자 탐색·지표 비교 없음)")
    print(f"#  layers={LAYERS} hidden={HIDDEN} lr={LR:g} wd={WD:g} "
          f"max_epoch={MAX_EPOCH}")
    print(f"#  folds={FOLDS}  seeds={SEEDS}")
    print(f"#  ckpt 기준 = {'val CRPS' if os.environ.get('FLOW_VAL_CRPS') == '1' else 'val NLL 조기종료'}")
    print(f"#  태그 예시: {make_tag(SEEDS[0])}")
    print("#" * 92)

    set_cond_cols(ENC_FULL, "tbill_wr")
    T.MASK_FUTURE_TBILL = False
    T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
    T.ENCODER_MASK_SP = False

    done = skipped = failed = 0
    for fold in FOLDS:
        missing = [s for s in ("train", "val", "test")
                   if not os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))]
        if missing:
            print(f"  [skip {fold}] CSV 없음: {missing}")
            continue
        for seed in SEEDS:
            tag = make_tag(seed)
            sp = os.path.join(RESULT_DIR,
                              f"garch_flow_ar_{tag}_{fold}_summary.json")
            if os.path.exists(sp):
                print(f"  [skip] {os.path.basename(sp)}")
                skipped += 1
                continue
            spec = dict(LOCKED)
            spec.update(fold=fold, seed=seed, tag=tag)
            print(f"\n  [run] {tag}  fold={fold}")
            t0 = time.time()
            try:
                main_worker(spec)
                print(f"    done ({time.time() - t0:.0f}s)")
                done += 1
            except Exception as e:                                # noqa: BLE001
                print(f"    [FAIL] {e!r}")
                failed += 1

    print(f"\n학습 {done} · 건너뜀 {skipped} · 실패 {failed}")
    print(f"§4.1 비교로 넘기려면: FLOW_TAG={make_tag(SEEDS[0]).rsplit('_s', 1)[0]}")


if __name__ == "__main__":
    main()
