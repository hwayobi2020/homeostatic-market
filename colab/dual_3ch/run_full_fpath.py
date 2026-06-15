"""full_fpath 학습 — 기존 MAC-Flow(LOCKED full spec) + 미래경로 전체 요약(fpath_model).

기존 full(rvP2mainMlp)과 *동일 spec*에 미래경로 요약(Flow context out=16)만 추가.
train_garch_flow.py 는 안 건드리고 fpath_model.MambaFlowARFpath 로 monkeypatch 만 한다.
→ 기존 full/maskall/metab_drop(어블리션)과 깨끗이 비교.

tag = rvAbl_full_fpath_s{seed}  (기존 tag 와 충돌 없음)

방향성 1차: 기본 1 seed × 4 fold.  5 seed 확대 시 env FPATH_SEEDS=2026,2027,2028,2029,2030.
미래요약 차원 sweep: env FPATH_DIM (기본 16), FPATH_HIDDEN (기본 16).

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/run_full_fpath.py
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
import fpath_model                                                  # noqa: E402

# 미래요약 설정 (env override 가능) → monkeypatch *전에* 세팅.
fpath_model.FUTURE_SUMMARY_DIM = int(os.environ.get("FPATH_DIM", "16"))
fpath_model.FUTURE_SUMMARY_HIDDEN = int(os.environ.get("FPATH_HIDDEN", "16"))
T.MambaFlowAR = fpath_model.MambaFlowARFpath                        # ★ monkeypatch

from train_garch_flow import main_worker                            # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEEDS = [int(s) for s in os.environ.get("FPATH_SEEDS", "2026").split(",") if s.strip()]

ENC_FULL = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]

# LOCKED 본모형 spec (rvP2mainMlp / run_ablations LOCKED 와 동일).
LOCKED = dict(encoder_type="mlp", d_model=128, mlp_num_layers=4,
              extra_context_channels="sp_std_13w,sp_skew_13w",
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
    print("# full_fpath 학습 — LOCKED full spec + 미래경로 전체 요약(Flow context)")
    print(f"#  future_summary_dim={fpath_model.FUTURE_SUMMARY_DIM} "
          f"hidden={fpath_model.FUTURE_SUMMARY_HIDDEN}  seeds={SEEDS}")
    print(f"#  ENC={ENC_FULL}  unmask=[metab_13w]  mask_tbill=False")
    print("#" * 96)

    # full 과 동일 조건: 미래 tbill 보임 + metab unmask, sp 미마스크.
    set_cond_cols(ENC_FULL, "tbill_wr")
    T.MASK_FUTURE_TBILL = False
    T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
    T.ENCODER_MASK_SP = False

    for fold in FOLDS:
        if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                   for s in ("train", "val", "test")):
            print(f"  [skip {fold}] fold CSV 없음"); continue
        for seed in SEEDS:
            tag = f"rvAbl_full_fpath_s{seed}"
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

    print("\n[done] full_fpath 학습 완료.")
    print("  per-step 비교: agg_section4 에 full_fpath tag 추가 / IHL 비교: tail_ablation 에 full_fpath 추가.")


if __name__ == "__main__":
    main()
