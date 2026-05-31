"""raw-vol 순차 튜닝 master — Phase 1a(압축기) → 1b(flow) → 2(멀티시드) 한 번에.

[설계 = 2026-05-31 세션, 플랜 B 순차 튜닝]
  메인 per-step 인코더 = MLP 고정 (§3.3.2).  ablation 차원 = 과거 요약 압축기
  (past_encoder_type) — 논문 §3.4.1 "동일 압축기 자리에 LSTM/Transformer/Mamba 결합".

  압축기 구조 실측(train_garch_flow.py 710~727): 종류 불문 n_layers=1 고정,
  유일한 capacity 노브 = past_summary_dim.  → 1a 격자 = 종류 × past_summary_dim × dropout.

  Phase 1a (압축기 sweep, flow=Heavy 고정, val-only)
      past_encoder_type ∈ {mlp, lstm, transformer, mamba}
      past_summary_dim  ∈ {32, 64, 128}
      dropout           ∈ {0.1, 0.2}
      = 4 × 3 × 2 = 24 spec × 3 fold = 72 run (AR 롤아웃 스킵)
      → 자동 선택: 종류별 best(3-fold avg val NLL) = ablation 후보 4개,
                   전체 best 종류 = 본모형 압축기

  Phase 1b (flow 격자 sweep, best 압축기 고정, val-only)
      n_flow_layers ∈ {4, 6, 8}
      n_flow_hidden ∈ {32, 64, 128}
      (n_flow_bins=16, blocks=2, tail_bound=10, weight_decay=0.5 고정 — 논문 §3.3.4 / Table 3.3)
      = 9 spec × 3 fold = 27 run
      → 자동 선택: best flow(3-fold avg val NLL)

  Phase 2 (멀티시드, test 평가 ON = AR 롤아웃)
      ablation : 압축기 4종 winner (flow=Heavy) × 5 seed × 3 fold = 60 run  → Table 4.7/4.8
      main     : best 압축기 + best flow × 5 seed × 4 fold(F_gfc 포함) = 20 run → Table 4.1(본모형)/4.6/4.9
      = 80 run

  채우는 표 : 3.7(encoder) · 4.1(본모형) · 4.7 · 4.8.
  (제외 — 다음 세션: #4 path-mask, #5 perm, #6 반사실, #7 baseline, #8 DM.)

무인 실행 안전장치
  - 재진입: 각 run 의 summary.json 이 있으면 skip → Colab 끊겨도 !python 재실행으로 이어받기.
  - 실패 격리: 한 run 이 예외나도 다음 진행 (try/except).
  - 자동 선택은 매 실행 시 디스크 summary 에서 재계산 → 재시작 안전.
  - val-only(1a/1b)는 evaluate_test no-op monkey-patch (AR 롤아웃 스킵, core 파일 0 수정).
  - 진행률 [P1a i/72] 형식 출력.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/run_master_rawvol_tuning.py
  → 첫 1~2 run 이 성공하는지(수 분) 확인 후 퇴근 권장.  끊기면 같은 명령 재실행.
"""
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

# ── raw-vol monkey-patch 먼저 ──
from rawvol_helpers import patch_rawvol     # noqa: E402
patch_rawvol()

import train_garch_flow as T                # noqa: E402
from train_garch_flow import main_worker    # noqa: E402
from best_specs import BEST_SPECS           # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
os.makedirs(RESULT_DIR, exist_ok=True)

# ── 공통 설정 ──
FOLDS_TUNE = ["F_long_A", "F_long_B_origin", "F_long"]   # 튜닝/ablation 3 fold
FOLDS_MAIN = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]  # 본모형 4 fold
SEEDS = [2026, 2027, 2028, 2029, 2030]
TUNE_SEED = 2026

# raw-vol 채널 구성 (run_rawvol_macroenc.py 와 동일)
ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
DC_COLS = "sp_std_13w,sp_skew_13w"
MASK_FUTURE_TBILL = False
FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]

# Heavy flow head (1a 고정 / ablation 공통) — 논문 Table 3.3
FLOW_HEAVY = dict(n_flow_layers=6, n_flow_hidden=64, weight_decay=0.5)

# 1a 격자
P1A_PAST_ENC = ["mlp", "lstm", "transformer", "mamba"]
P1A_PAST_DIM = [32, 64, 128]
P1A_DROPOUT = [0.1, 0.2]
# 1b 격자 (flow)
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
for _c in FUTURE_UNMASK_MACRO_COLS:
    if _c not in ENC_COLS:
        raise SystemExit(f"[FATAL] FUTURE_UNMASK_MACRO_COLS {_c!r} not in ENC_COLS")

# 메인 per-step 인코더 = MLP 고정 (BEST_SPECS['mlp'] 의 d_model / mlp_num_layers)
_MLP = BEST_SPECS["mlp"]
MAIN_ENC = dict(encoder_type="mlp",
                d_model=_MLP.get("d_model", 128),
                mlp_num_layers=_MLP.get("mlp_num_layers", 4))

# 모든 run 공통 cond (raw-vol macroenc 정신)
COMMON = dict(extra_context_channels=DC_COLS, direct_prev_return=True,
              use_past_summary=True)

# val-only 토글용
_ORIG_EVAL = T.evaluate_test


def _noop_eval(*args, **kwargs):
    return {}


# ─────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────────────────────────────────
def _summary_path(tag, fold):
    return os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{fold}_summary.json")


def read_val_nll(tag, fold):
    """summary.json 의 best_val_nll_per_week (선택 키, test 미사용)."""
    fp = _summary_path(tag, fold)
    if not os.path.exists(fp):
        return None
    try:
        return json.load(open(fp)).get("best_val_nll_per_week")
    except (json.JSONDecodeError, OSError):
        return None


def run_one(tag, fold, overrides, seed, label):
    """단일 (tag, fold, seed) 학습.  재진입 skip + 실패 격리 + 타이밍."""
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


def avg_3fold(val_by_fold, folds):
    """folds 전부 존재하면 평균, 아니면 None."""
    vals = [val_by_fold.get(f) for f in folds]
    if any(v is None for v in vals):
        return None
    return sum(vals) / len(vals)


# ═════════════════════════════════════════════════════════════════════════
# PHASE 1a — 압축기 sweep (flow=Heavy 고정, val-only)
# ═════════════════════════════════════════════════════════════════════════
def phase1a():
    T.evaluate_test = _noop_eval     # val-only
    specs = [(pet, pd, dr) for pet in P1A_PAST_ENC
             for pd in P1A_PAST_DIM for dr in P1A_DROPOUT]
    total = len(specs) * len(FOLDS_TUNE)
    print("\n" + "=" * 80)
    print(f"[PHASE 1a] 압축기 sweep : {len(specs)} spec × {len(FOLDS_TUNE)} fold "
          f"= {total} run (val-only, flow=Heavy, 메인=MLP 고정)")
    print("=" * 80)
    i = 0
    for pet, pd, dr in specs:
        tag = f"rvP1a_{pet}_pd{pd}_dr{dr}"
        ov = dict(FLOW_HEAVY)
        ov.update(past_encoder_type=pet, past_summary_dim=pd, dropout=dr)
        for fold in FOLDS_TUNE:
            i += 1
            run_one(tag, fold, ov, TUNE_SEED, f"P1a {i}/{total}")
    T.evaluate_test = _ORIG_EVAL


def select_1a():
    """종류별 best (pd, dr) + 전체 best 종류 반환."""
    # key=(pet,pd,dr) -> {fold: val}
    vals = defaultdict(dict)
    for pet in P1A_PAST_ENC:
        for pd in P1A_PAST_DIM:
            for dr in P1A_DROPOUT:
                tag = f"rvP1a_{pet}_pd{pd}_dr{dr}"
                for fold in FOLDS_TUNE:
                    v = read_val_nll(tag, fold)
                    if v is not None:
                        vals[(pet, pd, dr)][fold] = v
    type_winner = {}   # pet -> dict(past_summary_dim, dropout, avg)
    for (pet, pd, dr), fv in vals.items():
        a = avg_3fold(fv, FOLDS_TUNE)
        if a is None:
            continue
        if pet not in type_winner or a < type_winner[pet]["avg"]:
            type_winner[pet] = dict(past_summary_dim=pd, dropout=dr, avg=a)
    print("\n[PHASE 1a 선택] 압축기 종류별 best (3-fold avg val NLL):")
    for pet in P1A_PAST_ENC:
        if pet in type_winner:
            w = type_winner[pet]
            print(f"  {pet:12s} pd={w['past_summary_dim']:>3} dr={w['dropout']} "
                  f"avg_val_nll={w['avg']:.4f}")
        else:
            print(f"  {pet:12s} [3-fold 미완 — ablation 제외]")
    complete = {p: w for p, w in type_winner.items()}
    if not complete:
        raise SystemExit("[FATAL] Phase 1a 완료된 압축기 없음 — 1a 먼저 끝까지.")
    main_pet = min(complete, key=lambda p: complete[p]["avg"])
    print(f"  → 본모형 압축기 = {main_pet} "
          f"(pd={complete[main_pet]['past_summary_dim']}, "
          f"dr={complete[main_pet]['dropout']}, avg={complete[main_pet]['avg']:.4f})")
    return type_winner, main_pet


# ═════════════════════════════════════════════════════════════════════════
# PHASE 1b — flow 격자 sweep (best 압축기 고정, val-only)
# ═════════════════════════════════════════════════════════════════════════
def phase1b(main_pet, type_winner):
    T.evaluate_test = _noop_eval
    w = type_winner[main_pet]
    comp = dict(past_encoder_type=main_pet,
                past_summary_dim=w["past_summary_dim"], dropout=w["dropout"])
    specs = [(nl, nh) for nl in P1B_FLOW_LAYERS for nh in P1B_FLOW_HIDDEN]
    total = len(specs) * len(FOLDS_TUNE)
    print("\n" + "=" * 80)
    print(f"[PHASE 1b] flow sweep : {len(specs)} spec × {len(FOLDS_TUNE)} fold "
          f"= {total} run (val-only, 압축기={main_pet} 고정)")
    print("=" * 80)
    i = 0
    for nl, nh in specs:
        tag = f"rvP1b_fl{nl}_fh{nh}"
        ov = dict(comp)
        ov.update(n_flow_layers=nl, n_flow_hidden=nh,
                  weight_decay=P1B_WEIGHT_DECAY)
        for fold in FOLDS_TUNE:
            i += 1
            run_one(tag, fold, ov, TUNE_SEED, f"P1b {i}/{total}")
    T.evaluate_test = _ORIG_EVAL


def select_1b():
    vals = defaultdict(dict)
    for nl in P1B_FLOW_LAYERS:
        for nh in P1B_FLOW_HIDDEN:
            tag = f"rvP1b_fl{nl}_fh{nh}"
            for fold in FOLDS_TUNE:
                v = read_val_nll(tag, fold)
                if v is not None:
                    vals[(nl, nh)][fold] = v
    best = None   # (nl, nh, avg)
    print("\n[PHASE 1b 선택] flow 격자 (3-fold avg val NLL):")
    for (nl, nh), fv in sorted(vals.items()):
        a = avg_3fold(fv, FOLDS_TUNE)
        if a is None:
            continue
        print(f"  layers={nl} hidden={nh:>3}  avg_val_nll={a:.4f}")
        if best is None or a < best[2]:
            best = (nl, nh, a)
    if best is None:
        raise SystemExit("[FATAL] Phase 1b 완료된 flow spec 없음.")
    print(f"  → best flow = layers={best[0]}, hidden={best[1]} (avg={best[2]:.4f})")
    return dict(n_flow_layers=best[0], n_flow_hidden=best[1],
                weight_decay=P1B_WEIGHT_DECAY)


# ═════════════════════════════════════════════════════════════════════════
# PHASE 2 — 멀티시드 (test 평가 ON)
# ═════════════════════════════════════════════════════════════════════════
def phase2(type_winner, main_pet, best_flow):
    T.evaluate_test = _ORIG_EVAL     # AR 롤아웃 ON
    # ── 2-ablation : 압축기 4종 winner (flow=Heavy) × 5 seed × 3 fold ──
    abl = [p for p in P1A_PAST_ENC if p in type_winner]
    n_abl = len(abl) * len(SEEDS) * len(FOLDS_TUNE)
    # ── 2-main : best 압축기 + best flow × 5 seed × 4 fold ──
    n_main = len(SEEDS) * len(FOLDS_MAIN)
    print("\n" + "=" * 80)
    print(f"[PHASE 2] 멀티시드 (test 평가 ON): ablation {n_abl} run + "
          f"main {n_main} run = {n_abl + n_main}")
    print("=" * 80)

    i = 0
    for pet in abl:
        w = type_winner[pet]
        ov = dict(FLOW_HEAVY)
        ov.update(past_encoder_type=pet, past_summary_dim=w["past_summary_dim"],
                  dropout=w["dropout"])
        for seed in SEEDS:
            tag = f"rvP2abl_{pet}_pd{w['past_summary_dim']}_dr{w['dropout']}_s{seed}"
            for fold in FOLDS_TUNE:
                i += 1
                run_one(tag, fold, ov, seed, f"P2-abl {i}/{n_abl}")

    w = type_winner[main_pet]
    ov = dict(best_flow)
    ov.update(past_encoder_type=main_pet,
              past_summary_dim=w["past_summary_dim"], dropout=w["dropout"])
    j = 0
    for seed in SEEDS:
        tag = (f"rvP2main_{main_pet}_pd{w['past_summary_dim']}"
               f"_fl{best_flow['n_flow_layers']}_fh{best_flow['n_flow_hidden']}_s{seed}")
        for fold in FOLDS_MAIN:
            j += 1
            run_one(tag, fold, ov, seed, f"P2-main {j}/{n_main}")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("#" * 80)
    print("# raw-vol 순차 튜닝 master : Phase 1a → 1b → 2")
    print(f"#   메인=MLP(d{MAIN_ENC['d_model']}/nl{MAIN_ENC['mlp_num_layers']}) 고정")
    print(f"#   채널={ENC_COLS}  context={DC_COLS}")
    print(f"#   미래 unmask: tbill(mask={MASK_FUTURE_TBILL}) + {FUTURE_UNMASK_MACRO_COLS}")
    print("#" * 80)

    # fold csv 사전 점검 (없으면 즉시 알림 — 밤새 헛도는 것 방지)
    miss = []
    for fold in FOLDS_MAIN:
        for s in ("train", "val", "test"):
            p = os.path.join(FOLDS_DIR, f"{fold}_{s}.csv")
            if not os.path.exists(p):
                miss.append(os.path.basename(p))
    if miss:
        print(f"[WARN] 누락 fold csv: {miss}  (해당 fold run 은 [FATAL]로 종료될 수 있음)")

    phase1a()
    type_winner, main_pet = select_1a()

    phase1b(main_pet, type_winner)
    best_flow = select_1b()

    phase2(type_winner, main_pet, best_flow)

    dt = time.time() - t0
    print("\n" + "#" * 80)
    print(f"# [ALL DONE] 총 경과 {dt/60:.1f} min ({dt/3600:.2f} h)")
    print(f"# 본모형: 압축기={main_pet}, flow=layers{best_flow['n_flow_layers']}"
          f"/hidden{best_flow['n_flow_hidden']}")
    print(f"# 결과 summary: {RESULT_DIR}/garch_flow_ar_rvP2*_summary.json")
    print(f"# 다음: 결과 집계 → Table 3.7/4.1/4.7/4.8 갱신, 그 후 #4~#8")
    print("#" * 80)


if __name__ == "__main__":
    main()
