"""macroenc-garch — NF-GARCH 위에 macroenc 구조 (encoder=거시 / flow head=주가·변동성).

5/23 macroenc(train_mamba_flow_ar + run_macroenc) 구조를 train_garch_flow(NF-GARCH)로
이식.  사용자 확정(2026-05-25):
  - encoder(MLP) 입력 = 과거 sp + 거시: sp_return(과거), tbill_wr, ads_lag, wti_wr, metab_13w
      (ENCODER_MASK_SP=True 는 *미래 sp 만* 0 마스크한다.  과거 sp 시퀀스는 인코더가 봐야
       좌측 skew(폭락 비대칭)를 학습 — 2026-05-25 진단: 과거까지 마스크 시 sim skew 부호가
       뒤집힘(+1.11).  미래 sp 는 AR 의 prevret 으로 flow head 에 전달 = "미래는 거시 중심".)
  - flow head 직접 입력:
      prevret = direct_prev_return=True            (직전 주 표준화수익률 z_t, 1회)
      volDC   = extra_context "sp_std_13w"          (=GARCH σ 하나; log σ 는 같은 σ 의 단조변환
                                                     이라 정보 중복 → 제외)
  - σ 누수 수정(forward GARCH σ 예측)은 evaluate_test 공통이라 자동 적용.

★ 미래 거시 경로 = 조건부 시나리오 입력 (예측 아님):
  - MASK_FUTURE_TBILL=False  → 미래 tbill(단기금리) 경로 unmask.  (원형은 True=마스크였음)
  - FUTURE_UNMASK_MACRO_COLS=["metab_13w"] → 미래 metab(화폐가치절하) 경로도 unmask.
  둘 다 인코더 조건으로 들어가 "이런 금리·화폐가치절하 경로라면 주가 분포는?" 을 생성한다
  (운영 시 미래 거시를 안다는 게 아니라, 가정된 경로를 조건으로 주는 시나리오 생성기).

Usage (Colab): !python colab/dual_3ch/run_garch_macroenc.py
DM 비교: dm_gen_compare.py 의 garch-flow prefix 를 'garch_flow_ar_macroenc_s2026' 로 맞춰 실행.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import train_garch_flow as T              # noqa: E402
from train_garch_flow import main_worker  # noqa: E402
from best_specs import BEST_SPECS         # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]   # skew(base lambda 수정) 4 fold 검증.
SEEDS = [2026]            # garch-flow 패턴(fold별 single seed + per-origin DM).  늘리려면 추가.
# 과거요약 = MLP(deterministic, calibration best), d=64.  skew-t base _lam_raw 를 weight_decay
#   에서 제외(버그 수정) 후, 4 fold 에서 좌측 skew 가 안정적으로 잡히는지 확인.
PAST_ENCODER_TYPES = ["mlp"]
PAST_SUMMARY_DIM = 64

# encoder 가 보는 채널(거시) + sp_return(마스크되어 prevret/teacher-forcing 용으로만 잔류)
ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
DC_COLS = "sp_std_13w"            # volDC = GARCH σ 하나만 (log σ 제외, 사용자 확정)
MASK_FUTURE_TBILL = False         # thesis: 미래 단기금리 경로 조건(unmask).  원형 마스크는 True.
FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]   # 미래 metab(화폐가치절하) 경로도 인코더 조건(unmask).
                                  #   tbill(MASK_FUTURE_TBILL=False) 과 대칭 = 미래 거시 경로 둘 다 조건.


def set_cond_cols(cols):
    """train_garch_flow 의 채널 전역을 ENC_COLS 로 monkey-patch (run_macroenc 과 동일)."""
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
T.FUTURE_UNMASK_MACRO_COLS = FUTURE_UNMASK_MACRO_COLS   # 미래 metab 경로 unmask (조건)
T.ENCODER_MASK_SP = False         # sp 를 인코더가 다시 보게 (MLP 는 per-step 독립이라 sp 마스크
                                  #   시 좌측 skew 신호인 직전 수익률을 잃음 → False 로 복원)
# metab_13w 가 encoder 입력(ENC_COLS)에 있어야 미래 unmask 가 작동 (silent-zero 방지)
for _c in FUTURE_UNMASK_MACRO_COLS:
    if _c not in ENC_COLS:
        raise SystemExit(f"[FATAL] FUTURE_UNMASK_MACRO_COLS {_c!r} 가 ENC_COLS 에 없음")
spec_base = dict(BEST_SPECS["mlp"])

print(f"[macroenc-garch] {len(FOLDS)} fold x {len(PAST_ENCODER_TYPES)} past_enc x {len(SEEDS)} seed")
print(f"  encoder(main): sp + 거시(tbill,ads,wti,metab_13w)  (ENCODER_MASK_SP=False, sp 봄)")
print(f"  past summary : {PAST_ENCODER_TYPES} (dim={PAST_SUMMARY_DIM}, n=1)")
print(f"  flow head 직접: prevret(sp_return z_t) + volDC({DC_COLS}=GARCH σ)")
print(f"  미래 unmask(조건): tbill(MASK_FUTURE_TBILL={MASK_FUTURE_TBILL}) + metab{FUTURE_UNMASK_MACRO_COLS}")
print(f"  ENCODER_MASK_SP=True / NF-GARCH σ-leak fix: evaluate_test forward-σ 자동 적용")

for fold in FOLDS:
    if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
               for s in ("train", "val", "test")):
        print(f"[skip {fold}] fold CSV 없음")
        continue
    for past_enc in PAST_ENCODER_TYPES:
        for seed in SEEDS:
            tag = f"macroenc_past{past_enc.capitalize()}_d{PAST_SUMMARY_DIM}_s{seed}"
            sp = os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{fold}_summary.json")
            if os.path.exists(sp):
                print(f"[skip] {os.path.basename(sp)}")
                continue
            spec = dict(spec_base)
            spec.update(fold=fold, seed=seed, tag=tag,
                        extra_context_channels=DC_COLS, direct_prev_return=True,
                        use_past_summary=True, past_encoder_type=past_enc,
                        past_summary_dim=PAST_SUMMARY_DIM)
            print(f"\n[garch-flow macroenc] {tag} fold={fold} past_enc={past_enc}")
            try:
                main_worker(spec)
            except Exception as e:
                print(f"[FAIL] {tag} {fold}: {e!r}")

print("\n[done] macroenc-garch 학습 완료.")
print("  per-origin CRPS npy: garch_flow_ar_macroenc_s{seed}_{fold}_crps_per_origin.npy")
