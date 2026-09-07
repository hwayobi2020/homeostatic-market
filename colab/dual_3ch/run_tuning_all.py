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

# VAE/GAN 의 원래 조건 채널(6개)을 PS import 전에 확보한다.
#   analyze_pathshape_rawvol 은 import 시점에 COND_COLS 를 5 채널로 덮어쓴다.
import train_flow_seq as _TFS                                            # noqa: E402
VG_COND_COLS = list(_TFS.COND_COLS)

from rawvol_helpers import (patch_rawvol, rawstd_preprocess_fold,        # noqa: E402
                            forward_rawvol_rescale)
patch_rawvol()

# run_lr_sweep 은 import 시점에 (a) fpath monkeypatch (b) evaluate_test 를 감싸
# 검증셋 롤아웃을 추가한다.  MAC-Flow 쪽 설정을 그대로 재사용한다.
import run_lr_sweep as LRS                                               # noqa: E402
import analyze_pathshape_rawvol as PS                                    # noqa: E402
import train_garch_flow as T                                             # noqa: E402
import train_vae_gan_baseline as VG                                      # noqa: E402

VG.garch_preprocess_fold = rawstd_preprocess_fold      # import-bound 이름 교체
VG.forward_garch_rescale = forward_rawvol_rescale

import torch                                                             # noqa: E402

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


def flow_cell(fold, seed, lr, wd=BASE_WD):
    """MAC-Flow 한 셀.  wd=0.5 일 때 태그가 run_lr_sweep 과 같아 결과가 재사용된다.

    시험셋 summary 만 있고 VAL summary 가 없는 셀은 다시 돌린다.  lr 선택 기준이
    val 지표라, 그게 없는 셀은 재사용해도 표에서 빈칸으로 남는다 (VAL 롤아웃이
    run_lr_sweep 에 추가되기 전 판본으로 돌린 셀이 이 경우다).
    """
    wd_sfx = "" if abs(wd - BASE_WD) < 1e-12 else f"_wd{LRS.lr_tag(wd)}"
    tag = (f"rvAbl_full_fpath_novol_lr{LRS.lr_tag(lr)}{wd_sfx}"
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
        spec.update(fold=fold, seed=seed, tag=tag, lr=lr, weight_decay=wd)
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
    return dict(val=ve, test=(d.get("test_eval") or {}),
                best_epoch=d.get("best_epoch"), best_val_nll=d.get("best_val_nll"))


# =====================================================================
# CondVAE / CondGAN
# =====================================================================
def baseline_cell(mk, fold, seed, lr, device):
    """VAE/GAN 한 셀.  조건 채널을 원래 6 채널로 되돌린 뒤 학습한다."""
    tag = f"_lr{LRS.lr_tag(lr)}_s{seed}"
    sp = os.path.join(RESULT_DIR, f"{mk}_baseline{tag}_{fold}_summary.json")
    if not os.path.exists(sp):
        args = SimpleNamespace(model=mk, fold=fold, folds_dir=FOLDS_DIR,
                               out_dir=RESULT_DIR, n_sim=N_SIM, seed=seed,
                               lr=lr, tag=tag)
        saved = list(T.COND_COLS)
        PS.set_cond_cols(VG_COND_COLS)
        VG.COND_COLS = list(T.COND_COLS); VG.N_CH = len(VG.COND_COLS)
        VG.SP_CH, VG.TBILL_CH = T.SP_CH, T.TBILL_CH
        t0 = time.time()
        try:
            VG.run_fold(mk, fold, args, device)
            print(f"    done ({time.time() - t0:.0f}s)")
        finally:
            PS.set_cond_cols(saved)
    else:
        print(f"    [skip] {os.path.basename(sp)}")

    with open(sp, encoding="utf-8") as fh:
        d = json.load(fh)
    return dict(val=(d.get("val_eval") or {}), test=(d.get("test_eval") or {}),
                best_epoch=d.get("best_epoch"),
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
                wds = WD_GRID if mk == "flow" else [None]
                for lr in LR_GRID[mk]:
                    for wd in wds:
                        label = (f"lr={lr:g}" if wd is None
                                 else f"lr={lr:g} wd={wd:g}")
                        print(f"\n  [{mk}] fold={fold} seed={seed} {label}")
                        try:
                            if mk == "flow":
                                c = flow_cell(fold, seed, lr, wd)
                            else:
                                c = baseline_cell(mk, fold, seed, lr, device)
                        except Exception as e:                    # noqa: BLE001
                            print(f"    [FAIL] {e!r}")
                            continue
                        v = c["val"] or {}
                        rows.append(dict(
                            model=mk, fold=fold, seed=seed, lr=lr, wd=wd,
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
    hdr = ("{:6} {:16} {:>6} {:>8} {:>7} {:>8} {:>10} {:>8} {:>8} {:>11} {:>11}"
           .format("model", "fold", "seed", "lr", "wd", "best_ep", "val_CRPS",
                   "v_cov80", "v_cov95", "v_skew실측", "v_skew모형"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("{:6} {:16} {:>6} {:>8.0e} {:>7} {:>8} {:>10} {:>8} {:>8} "
              "{:>11} {:>11}"
              .format(r["model"], r["fold"], r["seed"], r["lr"],
                      _f(r.get("wd"), "{:g}", 7),
                      r["best_epoch"] if r["best_epoch"] is not None else "-",
                      _f(r["v_crps"], "{:.5f}", 10),
                      _f(r["v_cov80"], "{:.3f}", 8),
                      _f(r["v_cov95"], "{:.3f}", 8),
                      _f(r["v_skew_a"], "{:+.4f}", 11),
                      _f(r["v_skew_s"], "{:+.4f}", 11)))

    print("\n" + "=" * 108)
    print("[모델 × lr — 폴드·시드 평균 (선택 기준: 평균 val_CRPS 최소)]")
    print("=" * 108)
    hdr2 = ("{:6} {:>8} {:>7} {:>4} {:>8} {:>10} {:>8} {:>8} {:>11}"
            .format("model", "lr", "wd", "n", "best_ep", "val_CRPS", "v_cov80",
                    "v_cov95", "v_skew모형"))
    print(hdr2)
    print("-" * len(hdr2))
    chosen = {}
    for mk in MODELS:
        cand = []
        wds = WD_GRID if mk == "flow" else [None]
        for lr in LR_GRID[mk]:
            for wd in wds:
                g = [r for r in rows if r["model"] == mk and r["lr"] == lr
                     and r.get("wd") == wd]
                if not g:
                    continue
                mc = _mean(g, "v_crps")
                cand.append((mc, lr, wd))
                base = (abs(lr - BASE_LR[mk]) < 1e-12
                        and (wd is None or abs(wd - BASE_WD) < 1e-12))
                print("{:6} {:>8.0e} {:>7} {:>4} {:>8} {:>10} {:>8} {:>8} {:>11}{}"
                      .format(mk, lr, _f(wd, "{:g}", 7), len(g),
                              _f(_mean(g, "best_epoch"), "{:.1f}", 8),
                              _f(mc, "{:.5f}", 10),
                              _f(_mean(g, "v_cov80"), "{:.3f}", 8),
                              _f(_mean(g, "v_cov95"), "{:.3f}", 8),
                              _f(_mean(g, "v_skew_s"), "{:+.4f}", 11),
                              "  (게재 판본)" if base else ""))
        cand = [c for c in cand if isinstance(c[0], (int, float))]
        if cand:
            _, lr, wd = min(cand, key=lambda c: c[0])
            chosen[mk] = (lr, wd)

    print("\n[선택 결과 — 평균 val CRPS 기준]")
    for mk, (lr, wd) in chosen.items():
        same = (abs(lr - BASE_LR[mk]) < 1e-12
                and (wd is None or abs(wd - BASE_WD) < 1e-12))
        cur = f"lr = {lr:g}" + ("" if wd is None else f", weight_decay = {wd:g}")
        old = (f"lr {BASE_LR[mk]:g}"
               + ("" if wd is None else f", weight_decay {BASE_WD:g}"))
        print(f"  {mk:5} {cur}"
              + ("  (게재 판본과 같음)" if same else f"  ← 게재 판본 {old} 에서 변경"))
    if len(SEEDS) == 1:
        print("\n시드 1 개 결과다.  고른 값으로 5 시드 재학습해 안정성을 확인해야 한다:")
        for mk, (lr, wd) in chosen.items():
            cmd = (f"  TUNE_MODELS={mk} TUNE_LR_{mk.upper()}={lr:g} "
                   f"TUNE_SEEDS=2026,2027,2028,2029,2030")
            if wd is not None:
                cmd += f" TUNE_WD_FLOW={wd:g}"
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
