# -*- coding: utf-8 -*-
"""세 모델 학습률 통합 스윕 — 선택은 검증셋 지표로만 한다.

배경
----
리뷰어 1 #5 는 베이스라인이 본모형과 같은 정도로 튜닝됐는지 묻는다.  코드를
확인한 결과 두 가지가 사실이었다.

  · MAC-Flow  : lr = 1e-4 가 스윕으로 고른 값이 아니다.  train_garch_flow.py:116
                에서 모듈 상수로 고정돼 있고 스윕 격자에 lr 차원이 없다.
  · Cond VAE/GAN : 250 에폭 고정에 마지막 가중치를 썼다.  검증 기반 체크포인트
                선택이 아예 없었다 (val_csv 를 읽기만 하고 안 썼다).

즉 한쪽은 lr 이, 다른 쪽은 모델 선택이 빠져 있었다.  §3.7 은 "모든 모델의
하이퍼파라미터를 검증 분할에서 스윕해 골랐다"고 적고 있는데 그대로가 아니다.
이 스크립트는 세 모델을 같은 절차로 한 번에 튜닝해 그 서술을 사실로 만든다.

절차 (세 모델 동일)
-------------------
  1. lr 격자 × 폴드 × 시드로 학습한다.
  2. 학습 중 검증 지표로 체크포인트를 고른다.
       MAC-Flow      : val NLL 최저 (train_garch_flow 기존 동작)
       Cond VAE/GAN  : val CRPS(z 공간) 최저 (train_vae_gan_baseline 에 추가)
  3. 고른 체크포인트로 *검증셋* 을 시험셋과 똑같은 롤아웃·재스케일·지표
     정의로 평가한다.
  4. 폴드 평균 val CRPS 가 가장 낮은 lr 을 고른다.  시험셋은 선택에 쓰지 않는다.

폴드 평균으로 고르는 이유는 §3.7 이 "4-fold 평균 검증 지표"라고 적기 때문이다.
스윕은 시드 1 개로 돌리고, 고른 lr 에 대해서만 5 시드 재학습으로 안정성을
확인한다 (§3.7 의 "multi-seed (5-seed) retraining" 이 가리키는 단계).

이미 계산된 셀은 summary json 이 있으면 건너뛴다.  MAC-Flow 태그 규약은
run_lr_sweep.py 와 같으므로 거기서 돌린 결과가 그대로 재사용된다.

사용
----
    !python colab/dual_3ch/run_tuning_all.py
    # 격자/범위 조정
    !TUNE_MODELS=vae,gan TUNE_FOLDS=F_gfc python colab/dual_3ch/run_tuning_all.py
    # 고른 lr 로 5 시드 확인
    !TUNE_LR_FLOW=3e-4 TUNE_SEEDS=2026,2027,2028,2029,2030 \
        python colab/dual_3ch/run_tuning_all.py
"""
import json
import os
import sys
import time
from types import SimpleNamespace

# torch 는 다른 무엇보다 먼저 import 한다.  Windows 에서 다른 패키지가 먼저
# 로드되면 c10.dll 초기화가 실패한다 (OSError WinError 1114).
import torch                                                             # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

# 논문 본모형(fpath_novol d2) 을 쓰도록 PS import 전에 지정한다.
os.environ.setdefault("PS_BASE", "fpath_novol")
os.environ.setdefault("FPATH_DIM", "2")
# 베이스라인도 검증셋을 시험셋과 같은 정의로 평가하게 한다 (lr 선택 지표).
os.environ["VG_EVAL_VAL"] = "1"
# 베이스라인 원점 고정 추가 맥락.  MAC-Flow 는 sp_skew_13w 를 extra_context 로
# 받아 flow 헤드에 직접 붙인다(run_lr_sweep LOCKED: extra_context_channels).
# 한쪽만 최근 실현 왜도를 입력으로 받으면 "왜도 재현" 비교가 성립하지 않으므로
# 같은 값을 같은 형태(원점 행 z-스코어)로 준다.  VG_EXTRA_COLS="" 로 끌 수 있다.
os.environ.setdefault("VG_EXTRA_COLS", "sp_skew_13w")
# 미래 구간 조건 경로도 맞춘다.  flow_setup 이 flow 에
# FUTURE_UNMASK_MACRO_COLS=["metab_13w"] 를 주므로 flow 는 미래 metab 경로까지
# 조건으로 쓴다 (Table 3 의 반사실 주입 경로 = tbill + Excess_liq_13w).
# 베이스라인만 tbill 만 받으면 조건 정보가 달라 비교가 동등하지 않다.
os.environ.setdefault("VG_FUTURE_UNMASK", "metab_13w")

# 베이스라인 조건 채널 — 논문 Table 3 과 같게 맞춘다.
#   예전에는 train_flow_seq.COND_COLS(6채널)를 썼는데 그 목록에는
#   Excess_liq_13w(metab_13w)가 빠져 있고 대신 sp_std_13w / sp_log_std_13w 가
#   들어 있었다.  즉 베이스라인만 논문의 핵심 조건 변수를 못 받고, MAC-Flow 만
#   받는 상태였다.  §3.5.1/§3.5.2 는 베이스라인이 "the past 52-week window" 를
#   받는다고만 적어 이 차이를 밝히지 않는다 → CRPS 비교가 동등하지 않았다.
#   Table 3: sp_return, tbill_wr, Excess_liq_13w(metab_13w), ads_lag, wti_wr
VG_COND_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]

from rawvol_helpers import (patch_rawvol, rawstd_preprocess_fold,        # noqa: E402
                            forward_rawvol_rescale)
patch_rawvol()

# run_lr_sweep 은 import 시점에 (a) fpath monkeypatch (b) evaluate_test 를 감싸
# 검증셋 롤아웃을 추가한다.  MAC-Flow 쪽 설정을 그대로 재사용한다.
import run_lr_sweep as LRS                                               # noqa: E402
import analyze_pathshape_rawvol as PS                                    # noqa: E402
import train_garch_flow as T                                             # noqa: E402
import train_vae_gan_baseline as VG                                      # noqa: E402

# AR 롤아웃 가속.  run_lr_sweep 이 T.MambaFlowAR 을 MambaFlowARFpath 로 바꾼
# *뒤* 에 걸어야 두 클래스가 모두 잡힌다.  per-step MLP 인코더 + ENCODER_MASK_SP
# False 일 때만 가속하고 아니면 원본에 위임한다 (회귀 테스트에서 네 경우 모두
# 원본과 차이 0.000e+00, CPU 배속 Fpath 2.70x).  ARFAST=0 으로 끌 수 있다.
if os.environ.get("ARFAST", "1") == "1":
    import ar_sample_fast                                                # noqa: E402
    ar_sample_fast.patch()

VG.garch_preprocess_fold = rawstd_preprocess_fold      # import-bound 이름 교체
VG.forward_garch_rescale = forward_rawvol_rescale

FPATH_DIM = LRS.fpath_model.FUTURE_SUMMARY_DIM
RESULT_DIR = PS.RESULT_DIR
FOLDS_DIR = PS.FOLDS_DIR


def _env_list(key, default):
    return [x.strip() for x in os.environ.get(key, default).split(",") if x.strip()]


MODELS = _env_list("TUNE_MODELS", "flow,vae,gan")
FOLDS = _env_list("TUNE_FOLDS", ",".join(PS.FOLDS))
SEEDS = [int(s) for s in _env_list("TUNE_SEEDS", "2026")]
N_SIM = int(os.environ.get("TUNE_NSIM", str(PS.N_SIM)))
# VAL summary 가 없는 flow 셀을 다시 돌릴지.  0 이면 빈칸인 채로 둔다.
REDO_MISSING_VAL = os.environ.get("TUNE_REDO_MISSING_VAL", "1") == "1"
# MAC-Flow 최대 에폭.  기본 60 은 스크리닝 비용이 크다 — F_gfc lr1e-4 실측
# 1748 초/셀 이고 36 셀이면 17.5 시간이다.  같은 로그에서 CRPS 최저는 ep14,
# ep16~60 검증점 22 개가 모두 그 값을 못 넘겼고 NLL 은 ep6 부터 단조 상승한다.
# 30 으로 자르면 학습·검증이 절반이 되고 ep1~30 해상도는 2 간격 그대로다.
MAX_EPOCH = int(os.environ.get("TUNE_MAX_EPOCH", "30"))

# 기존 값이 격자 안에 들어가도록 잡았다 (flow 1e-4, vae 5e-4, gan 1e-4).
LR_GRID = {
    "flow": [float(x) for x in _env_list("TUNE_LR_FLOW", "1e-4,3e-4,1e-3")],
    "vae": [float(x) for x in _env_list("TUNE_LR_VAE", "1e-4,5e-4,1e-3")],
    "gan": [float(x) for x in _env_list("TUNE_LR_GAN", "5e-5,1e-4,3e-4")],
}
BASE_LR = {"flow": 1e-4, "vae": 5e-4, "gan": 1e-4}      # 게재 판본 값

# MAC-Flow 는 weight_decay 도 같이 흔든다.  lr 을 10 배 올려도 best epoch 이
# 3~4 에 머무는 것은 학습률이 병목이 아니라는 뜻이고, LOCKED 의 weight_decay=0.5
# 는 train_garch_flow CLI 기본값 0.01 의 50 배다.
WD_GRID = [float(x) for x in _env_list("TUNE_WD_FLOW", "0.5,0.1,0.01")]
BASE_WD = 0.5                                           # 게재 판본 값

# 흐름 헤드 hidden.  MAC-Flow 726,947 파라미터 중 흐름 헤드가 약 59 만이고
# (MODEL_SPEC.md §8 count_params.py 실측 총계 − 인코더 쪽 산술 135,344),
# CondVAE 177,210 / CondGAN 204,558 보다 3.5~4.1 배 크다.  리뷰어 1 #5 의
# 튜닝 공정성 지적에 답하려면 용량을 맞춘 비교가 필요한데 그 기록이 없다.
# 흐름 헤드는 hidden 의 제곱에 비례하므로 이 축 하나로 베이스라인 수준까지 내려간다.
HID_GRID = [int(x) for x in _env_list("TUNE_HID_FLOW", "128")]
BASE_HID = 128                                          # 게재 판본 값 (Table 5)

# 흐름 헤드 층수.  원래 튜닝(run_mlp_main_rawvol, Phase 1b-on-MLP)이
# layers {4,6,8} x hidden {32,64,128} 9 조합을 3 폴드 val NLL 로 돌아 4/128 을
# 골랐다.  그 뒤 모델이 fpath_novol 로 바뀌었는데 재튜닝을 안 했고, 선택 기준도
# 지금은 val CRPS 다.  같은 격자를 현재 구성·현재 기준으로 다시 돈다.
FL_GRID = [int(x) for x in _env_list("TUNE_LAYERS_FLOW", "4")]
BASE_FL = 4                                             # 게재 판본 값 (Table 5)

# 베이스라인 용량 축.  지금까지 스윕된 적이 없어 D_CTX=128 / HID=128 고정이었다.
# flow 는 n_flow_layers x n_flow_hidden 9 조합을 돌았으므로 §3.7 의 "모든 모델을
# 스윕했다"가 사실이 되려면 베이스라인도 같은 축을 열어야 한다.  파라미터 수도
# flow 726,947 대 CondVAE 177,210 / CondGAN 169,741 로 4 배 차이가 난다.
DCTX_GRID = [int(x) for x in _env_list("TUNE_DCTX_VG", "128")]
HID_GRID_VG = [int(x) for x in _env_list("TUNE_HID_VG", "128")]
BASE_DCTX, BASE_HID_VG = 128, 128                       # 게재 판본 값 (Table 6/7)


def vg_combos(mk):
    """베이스라인 격자 조합 (lr, D_CTX, HID)."""
    return [(lr, dc, h) for lr in LR_GRID[mk]
            for dc in DCTX_GRID for h in HID_GRID_VG]


def flow_combos():
    """MAC-Flow 격자 조합 목록.  축이 넷이라 중첩 루프 대신 목록으로 다룬다."""
    return [(lr, wd, hid, fl)
            for lr in LR_GRID["flow"] for fl in FL_GRID
            for hid in HID_GRID for wd in WD_GRID]

OUT = os.path.join(RESULT_DIR, "tuning_all.json")


# =====================================================================
# MAC-Flow
# =====================================================================
def flow_setup():
    """run_lr_sweep.main() 의 채널/마스킹 설정과 동일하게 맞춘다."""
    LRS.set_cond_cols(LRS.ENC_FULL, "tbill_wr")
    T.MASK_FUTURE_TBILL = False
    T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
    T.ENCODER_MASK_SP = False


def flow_cell(fold, seed, lr, wd=BASE_WD, hid=BASE_HID, fl=BASE_FL):
    """MAC-Flow 한 셀.  wd=0.5 일 때 태그가 run_lr_sweep 과 같아 결과가 재사용된다.

    시험셋 summary 만 있고 VAL summary 가 없는 셀은 다시 돌린다.  lr 선택 기준이
    val 지표라, 그게 없는 셀은 재사용해도 표에서 빈칸으로 남는다 (VAL 롤아웃이
    run_lr_sweep 에 추가되기 전 판본으로 돌린 셀이 이 경우다).
    """
    wd_sfx = "" if abs(wd - BASE_WD) < 1e-12 else f"_wd{LRS.lr_tag(wd)}"
    hid_sfx = "" if hid == BASE_HID else f"_fh{hid}"
    fl_sfx = "" if fl == BASE_FL else f"_fl{fl}"
    tag = (f"rvAbl_full_fpath_novol_lr{LRS.lr_tag(lr)}{wd_sfx}{hid_sfx}{fl_sfx}"
           f"_d{FPATH_DIM}_s{seed}")
    sp = os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{fold}_summary.json")
    vp = os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{fold}_VAL_summary.json")
    have = os.path.exists(sp)
    if have and not os.path.exists(vp) and REDO_MISSING_VAL:
        print(f"    [redo] summary 는 있으나 VAL 없음 → 재학습 "
              f"({os.path.basename(vp)})")
        have = False
    if not have:
        spec = dict(LRS.LOCKED)
        spec.update(fold=fold, seed=seed, tag=tag, lr=lr, weight_decay=wd,
                    n_flow_hidden=hid, n_flow_layers=fl, n_sim=N_SIM)
        spec["max_epoch"] = MAX_EPOCH
        t0 = time.time()
        LRS.main_worker(spec)
        print(f"    done ({time.time() - t0:.0f}s)")
    else:
        print(f"    [skip] {os.path.basename(sp)}")

    with open(sp, encoding="utf-8") as fh:
        d = json.load(fh)
    ve = {}
    if os.path.exists(vp):
        with open(vp, encoding="utf-8") as fh:
            ve = json.load(fh) or {}
    else:
        print(f"    [warn] VAL summary 없음 → val 지표 빈칸: "
              f"{os.path.basename(vp)}")
    # 실제로 평가에 쓰인 가중치의 에폭을 보고한다.  FLOW_VAL_CRPS=1 이면
    # best_state 를 CRPS 로만 갱신하므로(train_garch_flow:1219-1222, :1235-1236)
    # best_epoch(=NLL 기준)이 아니라 best_crps_epoch 이 맞다.
    crit = d.get("ckpt_criterion", "val_nll")
    ep_used = (d.get("best_crps_epoch") if crit == "val_crps_z"
               else d.get("best_epoch"))
    if crit == "val_crps_z":
        print(f"    [체크포인트] CRPS 기준 ep{ep_used} "
              f"(NLL 기준이었다면 ep{d.get('best_epoch')})")
    if d.get("n_params"):
        print(f"    [용량] {int(d['n_params']):,} 파라미터  "
              f"(CondVAE 177,210 / CondGAN 204,558 — MODEL_SPEC.md §8)")
    return dict(val=ve, test=(d.get("test_eval") or {}),
                n_params=d.get("n_params"),
                best_epoch=ep_used, ckpt_criterion=crit,
                best_val_nll=d.get("best_val_nll"))


# =====================================================================
# CondVAE / CondGAN
# =====================================================================
def baseline_cell(mk, fold, seed, lr, device, dctx=BASE_DCTX, hid=BASE_HID_VG):
    """VAE/GAN 한 셀.  조건 채널을 Table 3 목록으로 맞춘 뒤 학습한다.

    태그에 t3 를 넣어 예전 6 채널(train_flow_seq.COND_COLS) 결과와 파일을
    분리한다.  안 그러면 채널이 다른 옛 결과를 "이미 있음"으로 건너뛴다.
    sp_skew_13w 를 주입하면 sk, 미래 metab 을 주면 fu 를 더 붙인다.
    용량이 게재 판본과 다르면 c{D_CTX}h{HID} 도 붙여 분리한다.
    """
    sk_sfx = "sk" if os.environ.get("VG_EXTRA_COLS") else ""
    fu_sfx = "fu" if os.environ.get("VG_FUTURE_UNMASK") else ""
    cap_sfx = ("" if (dctx == BASE_DCTX and hid == BASE_HID_VG)
               else f"_c{dctx}h{hid}")
    tag = f"_t3{sk_sfx}{fu_sfx}{cap_sfx}_lr{LRS.lr_tag(lr)}_s{seed}"
    sp = os.path.join(RESULT_DIR, f"{mk}_baseline{tag}_{fold}_summary.json")
    if not os.path.exists(sp):
        args = SimpleNamespace(model=mk, fold=fold, folds_dir=FOLDS_DIR,
                               out_dir=RESULT_DIR, n_sim=N_SIM, seed=seed,
                               lr=lr, tag=tag)
        saved = list(T.COND_COLS)
        saved_cap = (VG.D_CTX, VG.HID)
        PS.set_cond_cols(VG_COND_COLS)
        VG.COND_COLS = list(T.COND_COLS); VG.N_CH = len(VG.COND_COLS)
        VG.SP_CH, VG.TBILL_CH = T.SP_CH, T.TBILL_CH
        # 모델 층은 __init__ 에서 모듈 전역을 읽으므로 여기서 바꾸면 반영된다.
        VG.D_CTX, VG.HID = dctx, hid
        t0 = time.time()
        try:
            VG.run_fold(mk, fold, args, device)
            print(f"    done ({time.time() - t0:.0f}s)")
        finally:
            PS.set_cond_cols(saved)
            VG.D_CTX, VG.HID = saved_cap
    else:
        print(f"    [skip] {os.path.basename(sp)}")

    with open(sp, encoding="utf-8") as fh:
        d = json.load(fh)
    return dict(val=(d.get("val_eval") or {}), test=(d.get("test_eval") or {}),
                best_epoch=d.get("best_epoch"), n_params=d.get("n_params"),
                best_val_nll=d.get("best_val_crps_z"))


# =====================================================================
def _mean(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(v) / len(v) if v else None


def _f(v, fmt, w):
    return fmt.format(v) if isinstance(v, (int, float)) else "-".rjust(w)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 108)
    print("# 학습률 통합 스윕 — MAC-Flow / CondVAE / CondGAN, 선택은 val 지표만")
    print(f"#  models={MODELS}  folds={FOLDS}  seeds={SEEDS}  n_sim={N_SIM}")
    print(f"#  flow max_epoch={MAX_EPOCH}  (베이스라인은 250 고정)")
    for m in MODELS:
        print(f"#  {m:5} lr grid = {LR_GRID[m]}   (게재 판본 {BASE_LR[m]:g})")
    print(f"#  device={device}  fpath_dim={FPATH_DIM}")
    print("#" * 108)

    rows = []
    for mk in MODELS:
        if mk == "flow":
            flow_setup()
        for fold in FOLDS:
            missing = [s for s in ("train", "val", "test")
                       if not os.path.exists(os.path.join(FOLDS_DIR,
                                                          f"{fold}_{s}.csv"))]
            if missing:
                print(f"  [skip {fold}] CSV 없음: {missing}")
                continue
            for seed in SEEDS:
                # weight_decay 는 MAC-Flow 에서만 흔든다 (베이스라인은 미사용).
                # flow: (lr, wd, hid, fl) / 베이스라인: (lr, dctx, hid, None)
                combos = (flow_combos() if mk == "flow"
                          else [(lr, dc, h, None) for lr, dc, h in vg_combos(mk)])
                for lr, wd, hid, fl in combos:
                        label = (f"lr={lr:g} wd={wd:g} layers={fl} hid={hid}"
                                 if mk == "flow"
                                 else f"lr={lr:g} d_ctx={wd} hid={hid}")
                        print(f"\n  [{mk}] fold={fold} seed={seed} {label}")
                        try:
                            if mk == "flow":
                                c = flow_cell(fold, seed, lr, wd, hid, fl)
                            else:
                                c = baseline_cell(mk, fold, seed, lr, device,
                                                  wd, hid)
                        except Exception as e:                    # noqa: BLE001
                            print(f"    [FAIL] {e!r}")
                            continue
                        v = c["val"] or {}
                        rows.append(dict(
                            model=mk, fold=fold, seed=seed, lr=lr, wd=wd,
                            hid=hid, fl=fl, n_params=c.get("n_params"),
                            best_epoch=c.get("best_epoch"),
                            v_crps=v.get("crps_pooled"),
                            v_cov80=v.get("coverage_80"),
                            v_cov95=v.get("coverage_95"),
                            v_skew_a=v.get("skew_actual"),
                            v_skew_s=v.get("skew_sim"),
                            _test=c["test"]))

    if not rows:
        print("\n결과 없음")
        return

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"\n저장: {os.path.basename(OUT)}  ({len(rows)} 셀)")

    print("\n" + "=" * 108)
    print("[셀별 — 검증셋 지표만.  시험셋은 lr 선택에 쓰지 않는다]")
    print("=" * 108)
    hdr = ("{:6} {:16} {:>6} {:>8} {:>7} {:>4} {:>5} {:>10} {:>8} {:>10} "
           "{:>8} {:>8} {:>11} {:>11}"
           .format("model", "fold", "seed", "lr", "wd", "lyr", "hid", "params",
                   "best_ep", "val_CRPS", "v_cov80", "v_cov95",
                   "v_skew실측", "v_skew모형"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("{:6} {:16} {:>6} {:>8.0e} {:>7} {:>4} {:>5} {:>10} {:>8} {:>10} "
              "{:>8} {:>8} {:>11} {:>11}"
              .format(r["model"], r["fold"], r["seed"], r["lr"],
                      _f(r.get("wd"), "{:g}", 7),
                      _f(r.get("fl"), "{:d}", 4),
                      _f(r.get("hid"), "{:d}", 5),
                      (f"{int(r['n_params']):,}"
                       if isinstance(r.get("n_params"), (int, float))
                       else "-".rjust(10)),
                      r["best_epoch"] if r["best_epoch"] is not None else "-",
                      _f(r["v_crps"], "{:.5f}", 10),
                      _f(r["v_cov80"], "{:.3f}", 8),
                      _f(r["v_cov95"], "{:.3f}", 8),
                      _f(r["v_skew_a"], "{:+.4f}", 11),
                      _f(r["v_skew_s"], "{:+.4f}", 11)))

    print("\n" + "=" * 108)
    print("[모델 × lr — 폴드·시드 평균 (선택 기준: 평균 val_CRPS 최소)]")
    print("=" * 108)
    hdr2 = ("{:6} {:>8} {:>7} {:>4} {:>5} {:>10} {:>4} {:>8} {:>10} {:>8} "
            "{:>8} {:>11}"
            .format("model", "lr", "wd", "lyr", "hid", "params", "n", "best_ep",
                    "val_CRPS", "v_cov80", "v_cov95", "v_skew모형"))
    print(hdr2)
    print("-" * len(hdr2))
    chosen = {}
    for mk in MODELS:
        cand = []
        combos = (flow_combos() if mk == "flow"
                  else [(lr, dc, h, None) for lr, dc, h in vg_combos(mk)])
        for lr, wd, hid, fl in combos:
                g = [r for r in rows if r["model"] == mk and r["lr"] == lr
                     and r.get("wd") == wd and r.get("hid") == hid
                     and r.get("fl") == fl]
                if not g:
                    continue
                mc = _mean(g, "v_crps")
                cand.append((mc, lr, wd, hid, fl))
                # 베이스라인은 wd 칸에 D_CTX 가 들어간다 (Table 6/7 기본 128/128).
                base = (abs(lr - BASE_LR[mk]) < 1e-12 and (
                    (wd == BASE_DCTX and hid == BASE_HID_VG) if mk != "flow"
                    else (abs(wd - BASE_WD) < 1e-12 and hid == BASE_HID
                          and fl == BASE_FL)))
                npar = _mean(g, "n_params")
                print("{:6} {:>8.0e} {:>7} {:>4} {:>5} {:>10} {:>4} {:>8} "
                      "{:>10} {:>8} {:>8} {:>11}{}"
                      .format(mk, lr, _f(wd, "{:g}", 7), _f(fl, "{:d}", 4),
                              _f(hid, "{:d}", 5),
                              (f"{int(npar):,}"
                               if isinstance(npar, (int, float))
                               else "-".rjust(10)),
                              len(g),
                              _f(_mean(g, "best_epoch"), "{:.1f}", 8),
                              _f(mc, "{:.5f}", 10),
                              _f(_mean(g, "v_cov80"), "{:.3f}", 8),
                              _f(_mean(g, "v_cov95"), "{:.3f}", 8),
                              _f(_mean(g, "v_skew_s"), "{:+.4f}", 11),
                              "  (게재 판본)" if base else ""))
        cand = [c for c in cand if isinstance(c[0], (int, float))]
        if cand:
            _, lr, wd, hid, fl = min(cand, key=lambda c: c[0])
            chosen[mk] = (lr, wd, hid, fl)

    print("\n[선택 결과 — 평균 val CRPS 기준]")
    for mk, (lr, wd, hid, fl) in chosen.items():
        if mk == "flow":
            same = (abs(lr - BASE_LR[mk]) < 1e-12 and abs(wd - BASE_WD) < 1e-12
                    and hid == BASE_HID and fl == BASE_FL)
            cur = (f"lr = {lr:g}, weight_decay = {wd:g}, "
                   f"flow layers = {fl}, flow hidden = {hid}")
            old = (f"lr {BASE_LR[mk]:g}, weight_decay {BASE_WD:g}, "
                   f"flow layers {BASE_FL}, flow hidden {BASE_HID}")
        else:
            same = (abs(lr - BASE_LR[mk]) < 1e-12
                    and wd == BASE_DCTX and hid == BASE_HID_VG)
            cur = f"lr = {lr:g}, d_ctx = {wd}, hidden = {hid}"
            old 	= (f"lr {BASE_LR[mk]:g}, d_ctx {BASE_DCTX}, "
                       f"hidden {BASE_HID_VG}")
        print(f"  {mk:5} {cur}"
              + ("  (게재 판본과 같음)" if same else f"  ← 게재 판본 {old} 에서 변경"))
    if len(SEEDS) == 1:
        print("\n시드 1 개 결과다.  고른 값으로 5 시드 재학습해 안정성을 확인해야 한다:")
        for mk, (lr, wd, hid, fl) in chosen.items():
            cmd = (f"  TUNE_MODELS={mk} TUNE_LR_{mk.upper()}={lr:g} "
                   f"TUNE_SEEDS=2026,2027,2028,2029,2030")
            if mk == "flow":
                cmd += (f" TUNE_WD_FLOW={wd:g} TUNE_LAYERS_FLOW={fl}"
                        f" TUNE_HID_FLOW={hid}")
            else:
                cmd += f" TUNE_DCTX_VG={wd} TUNE_HID_VG={hid}"
            print(cmd)

    if os.environ.get("TUNE_SHOW_TEST", "0") == "1":
        print("\n[시험셋 — lr 확정 후 확인용]")
        for r in rows:
            t = r["_test"] or {}
            print("  {:5} {:16} s{} lr={:.0e}  CRPS={} cov80={} skew={}".format(
                r["model"], r["fold"], r["seed"], r["lr"],
                _f(t.get("crps_pooled"), "{:.5f}", 9),
                _f(t.get("coverage_80"), "{:.3f}", 6),
                _f(t.get("skew_sim"), "{:+.4f}", 8)))


if __name__ == "__main__":
    main()
