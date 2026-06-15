"""no-vol 진단 — full(LOCKED) 에서 명시적 변동성 feature(sp_std_13w)만 제거.

가설(채널-공유): 표 4.11 에서 sp_std_13w 가 압도적 1위(Δuw_cvar1 −0.142)이고 metab/tbill 은
≈0 인데, 이는 거시 주도 변동성 레짐을 sp_std_13w 가 이미 담아 거시 marginal 을 가리기 때문.
→ sp_std_13w 를 빼면 (가격에서 vol 복원 + 거시로 보완) 거시 중요도가 오르나? 를 본다.

주의: σ(스케일)는 여전히 rolling std(rawvol scaffolding, 가격 파생) — 이 테스트는 "vol 정보 전부
제거"가 아니라 "flow context 의 *명시적* sp_std_13w feature 제거"다.  (sp_skew_13w 는 유지)

기존 full(rvP2mainMlp, sp_std 포함)은 그대로 비교 기준.  tag = rvAbl_novol_s{seed}.

탐색은 1 fold 권장: env NOVOL_FOLDS=F_long_B_origin (기본 4 fold).  seeds: NOVOL_SEEDS=2026.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/run_novol.py
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
ALL_FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
FOLDS = [f for f in os.environ.get("NOVOL_FOLDS", ",".join(ALL_FOLDS)).split(",") if f.strip()]
SEEDS = [int(s) for s in os.environ.get("NOVOL_SEEDS", "2026").split(",") if s.strip()]
MASKALL = os.environ.get("NOVOL_MASKALL", "0") == "1"   # 1=maskall+sp_std제거 (거시 통로 차단)

ENC_FULL = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]

# LOCKED full spec — 단 extra_context 에서 sp_std_13w 제거(sp_skew_13w 만 유지).
LOCKED = dict(encoder_type="mlp", d_model=128, mlp_num_layers=4,
              extra_context_channels="sp_skew_13w",          # ★ sp_std_13w 제거
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
    _mode = "novol_maskall(미래거시 없음 + sp_std 없음)" if MASKALL else "novol(full + sp_std 없음)"
    print("#" * 96)
    print(f"# {_mode} — LOCKED, extra_context 에서 sp_std_13w 제거(sp_skew_13w 유지)")
    print(f"#  ENC={ENC_FULL}  maskall={MASKALL}  seeds={SEEDS}  folds={FOLDS}")
    print("#  ※ σ(스케일)는 여전히 rolling std — 명시적 vol feature 만 제거")
    print("#" * 96)

    set_cond_cols(ENC_FULL, "tbill_wr")
    if MASKALL:
        # maskall + sp_std 제거: 미래 거시 없음 + 명시 vol 없음 → 레짐 복원 통로 차단
        T.MASK_FUTURE_TBILL = True
        T.FUTURE_UNMASK_MACRO_COLS = []
        _tagbase = "novol_maskall"
    else:
        T.MASK_FUTURE_TBILL = False
        T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
        _tagbase = "novol"
    T.ENCODER_MASK_SP = False

    for fold in FOLDS:
        if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                   for s in ("train", "val", "test")):
            print(f"  [skip {fold}] fold CSV 없음"); continue
        for seed in SEEDS:
            tag = f"rvAbl_{_tagbase}_s{seed}"
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

    print("\n[done] no-vol 학습 완료.")
    print("  비교: full(sp_std 포함) vs novol — test skew/CVaR/cov 변화, 그리고 permutation 중요도에서")
    print("        metab/tbill 이 sp_std 빠진 뒤 올라오는지(채널-공유 가설 검증).")


if __name__ == "__main__":
    main()
