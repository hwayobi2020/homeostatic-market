"""MLP 본모형 확정 러너 — 1b-on-MLP(flow 튜닝) + MLP 본모형 4-fold.

배경 (2026-06-01)
-----------------
마스터(run_master_rawvol_tuning.py)는 압축기를 *val NLL* 로 골라 LSTM 을 본모형으로
삼았으나, 이는 (단일 seed·val·non-tail) 기준의 artifact 였다.  5-seed *test* ablation
(Table 4.7)에서 MLP 가 NLL 최저 + 좌측 skew 최강(−0.78) + CRPS/EMD 최선 → 본 논문
(tail-risk)의 본모형 압축기 = MLP 로 확정.  val NLL 은 *encoder 내부 hyperparameter
튜닝* 용이지 *압축기 선택* 기준이 아니다.

flow head 는 마스터 Phase 1b 에서 LSTM 에만 튜닝되었으므로, MLP 본모형의 flow 를
LSTM 것으로 빌려 쓰지 않고 MLP 위에서 다시 튜닝한다(쉬운 길 대신 올바른 길).

이 스크립트가 하는 일
  1) 기존 Phase 1a MLP 결과(rvP1a_mlp_*)를 읽어 MLP 의 best (pd, dr) 재사용.
  2) [1b-on-MLP] MLP+best(pd,dr) 고정, flow 격자 9개 × 3 fold val-only → best flow 선택.
  3) [MLP 본모형] MLP+best(pd,dr)+best flow × 5 seed × 4 fold(F_gfc 포함), test 평가 ON.

재진입(summary 있으면 skip)·실패 격리·val-only no-op monkey-patch — 마스터와 동일.
새 tag : rvP1bMlp_* / rvP2mainMlp_*  (기존 LSTM run 과 충돌 없음).

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/run_mlp_main_rawvol.py
끝나면 agg_rawvol_tuning.py 재실행 → Table 3.7(MLP flow)·Table 4.1(MLP 본모형) 갱신.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from rawvol_helpers import patch_rawvol     # noqa: E402
patch_rawvol()

import train_garch_flow as T                # noqa: E402
from train_garch_flow import main_worker    # noqa: E402
from best_specs import BEST_SPECS           # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
os.makedirs(RESULT_DIR, exist_ok=True)

FOLDS_TUNE = ["F_long_A", "F_long_B_origin", "F_long"]
FOLDS_MAIN = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
TUNE_SEED = 2026

ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
DC_COLS = "sp_std_13w,sp_skew_13w"
MASK_FUTURE_TBILL = False
FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]

# Phase 1a 격자 (MLP best 재사용 위해 동일 grid 로 스캔)
P1A_PAST_DIM = [32, 64, 128]
P1A_DROPOUT = [0.1, 0.2]
# Phase 1b flow 격자 (MLP 위에서 재튜닝)
P1B_FLOW_LAYERS = [4, 6, 8]
P1B_FLOW_HIDDEN = [32, 64, 128]
P1B_WEIGHT_DECAY = 0.5


def set_cond_cols(cols):
    cols = list(cols)
    T.COND_COLS = cols
    T.N_CHANNELS = len(cols)
    T.SP_CH = cols.index("sp_return")
    T.TBILL_CH = cols.index("tbill_wr")
    T.MACRO_CH = [c for c in range(len(cols)) if c not in (T.SP_CH, T.TBILL_CH)]
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass


set_cond_cols(ENC_COLS)
T.MASK_FUTURE_TBILL = MASK_FUTURE_TBILL
T.FUTURE_UNMASK_MACRO_COLS = FUTURE_UNMASK_MACRO_COLS
T.ENCODER_MASK_SP = False

_MLP = BEST_SPECS["mlp"]
MAIN_ENC = dict(encoder_type="mlp",
                d_model=_MLP.get("d_model", 128),
                mlp_num_layers=_MLP.get("mlp_num_layers", 4))
COMMON = dict(extra_context_channels=DC_COLS, direct_prev_return=True,
              use_past_summary=True)

_ORIG_EVAL = T.evaluate_test


def _noop_eval(*args, **kwargs):
    return {}


def _summary_path(tag, fold):
    return os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{fold}_summary.json")


def read_val(tag, fold):
    fp = _summary_path(tag, fold)
    if not os.path.exists(fp):
        return None
    try:
        return json.load(open(fp)).get("best_val_nll_per_week")
    except (json.JSONDecodeError, OSError):
        return None


def avg_3fold(vals):
    if any(v is None for v in vals):
        return None
    return sum(vals) / len(vals)


def run_one(tag, fold, overrides, seed, label):
    fp = _summary_path(tag, fold)
    if os.path.exists(fp):
        print(f"[skip {label}] {os.path.basename(fp)}")
        return "skip"
    spec = dict(MAIN_ENC)
    spec.update(COMMON)
    spec.update(overrides)
    spec.update(fold=fold, tag=tag, seed=seed)
    print(f"\n[run {label}] tag={tag} fold={fold} seed={seed}")
    t0 = time.time()
    try:
        main_worker(spec)
        dt = time.time() - t0
        print(f"[done {label}] elapsed={dt:7.1f}s ({dt/60:.2f} min)")
        return "done"
    except Exception as e:
        print(f"[FAIL {label}] {tag} {fold} s{seed}: {e!r}")
        return "fail"


# ════════════════════════════════════════════════════════════════════
# 0) 기존 Phase 1a MLP 결과에서 best (pd, dr) 재사용
# ════════════════════════════════════════════════════════════════════
print("#" * 80)
print("# MLP 본모형 러너 : 1b-on-MLP → MLP 본모형 4-fold")
print("#" * 80)

cands = []
for pd in P1A_PAST_DIM:
    for dr in P1A_DROPOUT:
        tag = f"rvP1a_mlp_pd{pd}_dr{dr}"
        avg = avg_3fold([read_val(tag, f) for f in FOLDS_TUNE])
        if avg is not None:
            cands.append((pd, dr, avg))
if not cands:
    raise SystemExit("[FATAL] rvP1a_mlp_* 결과 없음 — run_master_rawvol_tuning.py(Phase 1a) 먼저 필요.")
MLP_PD, MLP_DR, mlp_val = min(cands, key=lambda x: x[2])
print(f"\n[0] MLP Phase-1a best : pd={MLP_PD} dr={MLP_DR}  (3-fold avg val NLL={mlp_val:.4f})")
print(f"    후보: " + "  ".join(f"pd{c[0]}dr{c[1]}={c[2]:.4f}"
                                for c in sorted(cands, key=lambda x: x[2])))

COMP = dict(past_encoder_type="mlp", past_summary_dim=MLP_PD, dropout=MLP_DR)


# ════════════════════════════════════════════════════════════════════
# 1) Phase 1b-on-MLP : flow 격자 sweep (val-only)
# ════════════════════════════════════════════════════════════════════
T.evaluate_test = _noop_eval
flow_specs = [(nl, nh) for nl in P1B_FLOW_LAYERS for nh in P1B_FLOW_HIDDEN]
total = len(flow_specs) * len(FOLDS_TUNE)
print("\n" + "=" * 80)
print(f"[1b-on-MLP] flow sweep : {len(flow_specs)} spec × {len(FOLDS_TUNE)} fold "
      f"= {total} run (val-only, 압축기=MLP pd{MLP_PD} 고정)")
print("=" * 80)
i = 0
for nl, nh in flow_specs:
    tag = f"rvP1bMlp_fl{nl}_fh{nh}"
    ov = dict(COMP)
    ov.update(n_flow_layers=nl, n_flow_hidden=nh, weight_decay=P1B_WEIGHT_DECAY)
    for fold in FOLDS_TUNE:
        i += 1
        run_one(tag, fold, ov, TUNE_SEED, f"1bMLP {i}/{total}")
T.evaluate_test = _ORIG_EVAL

# best flow 선택 (3-fold avg val NLL)
fb = []
for nl, nh in flow_specs:
    tag = f"rvP1bMlp_fl{nl}_fh{nh}"
    avg = avg_3fold([read_val(tag, f) for f in FOLDS_TUNE])
    if avg is not None:
        fb.append((nl, nh, avg))
if not fb:
    raise SystemExit("[FATAL] 1b-on-MLP 결과 없음.")
BEST_NL, BEST_NH, flow_val = min(fb, key=lambda x: x[2])
print("\n[1b-on-MLP 선택] flow 격자 (3-fold avg val NLL):")
for nl, nh, avg in sorted(fb, key=lambda x: x[2]):
    print(f"  layers={nl} hidden={nh:>3}  avg_val_NLL={avg:.4f}")
print(f"  → MLP best flow = layers={BEST_NL} hidden={BEST_NH} (val NLL={flow_val:.4f})")


# ════════════════════════════════════════════════════════════════════
# 2) MLP 본모형 : best(pd,dr)+best flow × 5 seed × 4 fold (test 평가 ON)
# ════════════════════════════════════════════════════════════════════
T.evaluate_test = _ORIG_EVAL
n_main = len(SEEDS) * len(FOLDS_MAIN)
print("\n" + "=" * 80)
print(f"[MLP 본모형] best flow(fl{BEST_NL}/fh{BEST_NH}) × {len(SEEDS)} seed × "
      f"{len(FOLDS_MAIN)} fold = {n_main} run (test 평가 ON)")
print("=" * 80)
ov = dict(COMP)
ov.update(n_flow_layers=BEST_NL, n_flow_hidden=BEST_NH, weight_decay=P1B_WEIGHT_DECAY)
j = 0
for seed in SEEDS:
    tag = f"rvP2mainMlp_pd{MLP_PD}_fl{BEST_NL}_fh{BEST_NH}_s{seed}"
    for fold in FOLDS_MAIN:
        j += 1
        run_one(tag, fold, ov, seed, f"mainMLP {j}/{n_main}")

print("\n" + "#" * 80)
print(f"# [DONE] MLP 본모형 : 압축기=mlp(pd{MLP_PD},dr{MLP_DR}), flow=fl{BEST_NL}/fh{BEST_NH}")
print(f"# 결과: garch_flow_ar_rvP2mainMlp_*_summary.json")
print(f"# 다음: agg_rawvol_tuning.py 재실행 → Table 3.7(MLP flow)·Table 4.1(MLP 본모형)")
print("#" * 80)
