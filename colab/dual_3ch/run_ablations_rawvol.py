"""§4.3.1 / §4.3.3 ablation 재학습 — LOCKED MAC-Flow(rvP2mainMlp) 와 *동일 spec*, 한 축만 변형.

ablation (각각 4 fold × 3 seed 재학습, main_worker):
  metab_drop : ENC 에서 metab_13w 제거 (+ 미래 unmask 없음)        — 유동성 신호 기여
  maskall    : 미래 거시 전부 마스킹(MASK_FUTURE_TBILL + unmask=[]) — path 조건화 효과(=path-mask)
  fedrate    : tbill_wr → fedfunds_wr 스왑 (prep_fedfunds_column.py 선행)
  win26/win52: 목표 horizon FUTURE_LEN=26/52 (windowing 기본값 rebind 필요)

기존 main 모델(rvP2mainMlp)은 그대로 비교 기준.  LOCKED spec 정확 복제 — ablation 외 모든 것 동일(통제).

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/prep_fedfunds_column.py   # fedrate 용 (1회)
    !python colab/dual_3ch/run_ablations_rawvol.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from rawvol_helpers import patch_rawvol                              # noqa: E402
patch_rawvol()
import train_garch_flow as T                                        # noqa: E402
from train_garch_flow import main_worker                            # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEEDS = [2026, 2027, 2028]                # ablation 3-seed (mean±std)
ENC_FULL = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
ORIG_FL = int(T.FUTURE_LEN)               # 13 (rebind 복원용)
ORIG_PL = int(T.PAST_LEN)                 # 52

# LOCKED 본모형 spec (rvP2mainMlp 와 동일 — ablation 축만 변경)
LOCKED = dict(encoder_type="mlp", d_model=128, mlp_num_layers=4,
              extra_context_channels="sp_std_13w,sp_skew_13w",
              direct_prev_return=True, use_past_summary=True,
              past_encoder_type="mlp", past_summary_dim=64,
              dropout=0.2, n_flow_layers=4, n_flow_hidden=128, weight_decay=0.5)

# (이름, ENC_COLS, rate_col, FUTURE_UNMASK_MACRO_COLS, MASK_FUTURE_TBILL, future_len)
ABLATIONS = [
    ("metab_drop", ["sp_return", "tbill_wr", "ads_lag", "wti_wr"], "tbill_wr", [], False, 13),
    ("maskall",    ENC_FULL,                                       "tbill_wr", [], True,  13),
    ("fedrate",    ["sp_return", "fedfunds_wr", "ads_lag", "wti_wr", "metab_13w"], "fedfunds_wr", ["metab_13w"], False, 13),
    ("win26",      ENC_FULL,                                       "tbill_wr", ["metab_13w"], False, 26),
    ("win52",      ENC_FULL,                                       "tbill_wr", ["metab_13w"], False, 52),
]


def set_cond_cols(cols, rate_col):
    cols = list(cols)
    T.COND_COLS = cols
    T.N_CHANNELS = len(cols)
    T.SP_CH = cols.index("sp_return")
    T.TBILL_CH = cols.index(rate_col)                 # 금리(rate) 채널 — fedrate 면 fedfunds_wr
    T.MACRO_CH = [c for c in range(len(cols)) if c not in (T.SP_CH, T.TBILL_CH)]
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass


_WIN_FNAMES = ("cached_load_windows_seq", "load_windows_seq", "load_extra_context",
               "cond_valid_mask", "compute_valid_mask", "extract_derived_origin")
_ORIG_DEFAULTS = {}   # fname -> *원본* __defaults__ 스냅샷 (import 시점, future_len=13 기준)


def set_window(fl):
    """windowing 함수들의 future_len 기본값(=13)을 rebind — main_worker 가 인자 없이 호출하므로.

    ★ *원본* 기본값 스냅샷에서 매번 rebind 한다.  (이전 버그: 현재(이미 변형된) 기본값에서
    d==13 만 교체 → win26 이 26 으로 바꾼 뒤 13 복원/52 재설정이 d==26 을 못 잡아 26 에 고착
    → FUTURE_LEN 과 불일치(78 vs 65).  스냅샷 기준 rebind 로 idempotent 하게 해결.)
    """
    T.FUTURE_LEN = fl
    T.L = ORIG_PL + fl
    for fname in _WIN_FNAMES:
        fn = getattr(T, fname, None)
        if fn is None:
            continue
        if fname not in _ORIG_DEFAULTS:
            _ORIG_DEFAULTS[fname] = fn.__defaults__     # 최초 1회: 원본 보존
        orig = _ORIG_DEFAULTS[fname]
        if orig is None:
            continue
        fn.__defaults__ = tuple(fl if d == ORIG_FL else d for d in orig)
    # ★ 모델 MambaFlowAR.__init__ 의 future_len 기본값(=13)도 rebind.
    #   main_worker 가 future_len 을 안 넘기고 기본값을 쓰므로(→ self.L, pos_emb 크기 결정),
    #   이걸 안 바꾸면 win26/52 에서 데이터 길이(78/104)와 pos_emb(65) 가 불일치한다.
    #   __init__ 기본값 중 13 은 future_len 뿐(d_model=128, bins=16, past_len=52 등) → 값매칭 안전.
    try:
        init_fn = T.MambaFlowAR.__init__
        if "MambaFlowAR.__init__" not in _ORIG_DEFAULTS:
            _ORIG_DEFAULTS["MambaFlowAR.__init__"] = init_fn.__defaults__
        orig_init = _ORIG_DEFAULTS["MambaFlowAR.__init__"]
        if orig_init is not None:
            init_fn.__defaults__ = tuple(fl if d == ORIG_FL else d for d in orig_init)
    except Exception as e:
        print(f"  [warn] MambaFlowAR.__init__ future_len rebind 실패: {e!r}")
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass


def main():
    print("#" * 96)
    print("# §4.3 ablation 재학습 (LOCKED spec, 한 축만 변형) — 4 fold × 3 seed × 5 ablation")
    print(f"#  ablations: {[a[0] for a in ABLATIONS]}")
    print("#" * 96)

    for name, enc, rate, unmask, mask_tbill, fl in ABLATIONS:
        # fedrate 는 fedfunds_wr 컬럼 필요 (prep_fedfunds_column.py 선행)
        miss = [c for c in unmask if c not in enc]
        if miss:
            sys.exit(f"[FATAL] {name}: unmask {miss} not in ENC")
        print(f"\n{'='*92}\n[ablation {name}] ENC={enc} rate={rate} unmask={unmask} "
              f"mask_tbill={mask_tbill} future_len={fl}\n{'='*92}")
        set_cond_cols(enc, rate)
        T.MASK_FUTURE_TBILL = mask_tbill
        T.FUTURE_UNMASK_MACRO_COLS = list(unmask)
        T.ENCODER_MASK_SP = False
        set_window(fl)

        for fold in FOLDS:
            if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                       for s in ("train", "val", "test")):
                print(f"  [skip {fold}] fold CSV 없음"); continue
            # fedrate: fedfunds_wr 컬럼 존재 확인
            if "fedfunds_wr" in enc:
                import pandas as pd
                cols0 = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_train.csv"), nrows=1).columns
                if "fedfunds_wr" not in cols0:
                    print(f"  [skip {fold}] fedfunds_wr 없음 → prep_fedfunds_column.py 먼저"); continue
            for seed in SEEDS:
                tag = f"rvAbl_{name}_s{seed}"
                sp = os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{fold}_summary.json")
                if os.path.exists(sp):
                    print(f"  [skip] {os.path.basename(sp)}"); continue
                spec = dict(LOCKED); spec.update(fold=fold, seed=seed, tag=tag)
                print(f"  [run] {tag} fold={fold}")
                t0 = time.time()
                try:
                    main_worker(spec)
                    print(f"    done ({time.time()-t0:.0f}s)")
                except Exception as e:
                    print(f"    [FAIL] {tag} {fold}: {e!r}")

        set_window(ORIG_FL)   # 다음 ablation 위해 window 복원

    print("\n[done] ablation 재학습 완료 — 집계는 agg(rvAbl_* summary) 또는 별도 표.")
    print("  metab_drop/maskall/fedrate = full(rvP2mainMlp) 대비 적합도·skew 변화로 기여 판단.")
    print("  win26/win52 = horizon 별 적합도 (13w 본모형과 비교).")


if __name__ == "__main__":
    main()
