# -*- coding: utf-8 -*-
"""학습률 스윕 — 본모형 spec 그대로, lr 만 바꿔 재학습.

배경
----
lr = 1e-4 는 튜닝으로 고른 값이 아니다.  train_garch_flow.py:31-32 주석이
"matched to train_flow_seq.py for fair comparison" 이라 적고, :116 에서
모듈 상수 LR = 1e-4 로 고정한다.  run_master_rawvol_tuning.py 의 스윕 격자에는
lr 차원이 없다(인코더 구조 + flow layers/hidden 만, weight_decay·bins·blocks·
tail_bound 는 고정).  과거 세션 기록에서도 lr 은 스윕 후보에 올랐다가 제외됐고
이후 본모형에 대해 다시 시도된 적이 없다.

관찰
----
F_gfc / seed 2026 실행에서 best val NLL 이 ep3 에 나오고 ep33 조기종료.
train windows 1395 / batch 32 = 44 step/epoch 이므로 채택된 체크포인트는
약 132 optimizer step 이다.  스칼라 파라미터(_lam_raw)의 이동 상한이
132 x 1e-4 = 0.013 수준이고, 실측 |lambda| 최대가 0.009158 로 그 안에 있다.

사용
----
    !python colab/dual_3ch/run_lr_sweep.py
    # 기본: F_gfc, seed 2026, lr = 1e-4(기준) / 3e-4 / 1e-3
    !LRS_LRS=1e-4,1e-3 LRS_FOLDS=F_gfc LRS_SEEDS=2026 python colab/dual_3ch/run_lr_sweep.py

읽는 법
-------
    best val NLL 이 lr=1e-4 보다 낮아지면 지금 값은 과소학습이다.
    best epoch 이 더 이르고 val 이 더 높으면 lr 이 과하다.
    두 경우 다 아니면 lr 은 원인이 아니다.
"""
import json
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

fpath_model.FUTURE_SUMMARY_DIM = int(os.environ.get("FPATH_DIM", "2"))
fpath_model.FUTURE_SUMMARY_HIDDEN = int(os.environ.get("FPATH_HIDDEN", "16"))
fpath_model.FPATH_SUMMARY_ONLY = False
T.MambaFlowAR = fpath_model.MambaFlowARFpath                        # ★ monkeypatch

from train_garch_flow import main_worker                            # noqa: E402

# ---------------------------------------------------------------------------
# evaluate_test 래핑 — 시험셋 평가 뒤 *검증셋*에도 같은 롤아웃을 한 번 더 돌린다.
# lr 선택을 시험 지표로 하면 test-set selection 이 된다.  검증셋에서 논문과
# 같은 정의의 CRPS/커버리지/왜도를 뽑아 그것으로 고른다.
# ---------------------------------------------------------------------------
_ORIG_EVAL = T.evaluate_test
VAL_EVAL = {}          # tag -> val 지표 dict


def _eval_with_val(model, best_state, test_csv, cond_stats, target_stats,
                   result_prefix, n_sim, seed, device, **kw):
    out = _ORIG_EVAL(model, best_state, test_csv, cond_stats, target_stats,
                     result_prefix, n_sim, seed, device, **kw)
    val_csv = test_csv.replace("_test", "_val")
    if val_csv != test_csv and os.path.exists(val_csv):
        print("\n[VAL] 같은 롤아웃을 검증셋에 재실행 (lr 선택용)")
        _pf, _ph = T.plot_fanchart, T.plot_histogram
        T.plot_fanchart = lambda *a, **k: None      # 그림 생략(시간 절약)
        T.plot_histogram = lambda *a, **k: None
        try:
            v = _ORIG_EVAL(model, best_state, val_csv, cond_stats, target_stats,
                           result_prefix + "_VAL", n_sim, seed, device, **kw)
            VAL_EVAL[os.path.basename(result_prefix)] = v
            with open(result_prefix + "_VAL_summary.json", "w",
                      encoding="utf-8") as fh:                       # skip 대비 저장
                json.dump(v, fh, indent=2, default=str)
        except Exception as e:                                       # noqa: BLE001
            print(f"[VAL] 실패 {e!r}")
        finally:
            T.plot_fanchart, T.plot_histogram = _pf, _ph
    else:
        print(f"[VAL] 검증 csv 없음: {val_csv}")
    return out


T.evaluate_test = _eval_with_val

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")

FOLDS = [f for f in os.environ.get("LRS_FOLDS", "F_gfc").split(",") if f.strip()]
SEEDS = [int(s) for s in os.environ.get("LRS_SEEDS", "2026").split(",") if s.strip()]
LRS = [float(x) for x in os.environ.get("LRS_LRS", "1e-4,3e-4,1e-3").split(",")
       if x.strip()]

ENC_FULL = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]

# run_full_fpath.py 의 LOCKED + NOVOL=1 (본모형 spec).  lr 만 바꾼다.
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


def lr_tag(lr):
    """1e-4 -> '1em4' 같은 파일명 안전 태그."""
    return ("%g" % lr).replace("-", "m").replace("+", "p").replace(".", "d")


def main():
    print("#" * 96)
    print("# lr 스윕 — 본모형 spec 고정, lr 만 변경")
    print(f"#  folds={FOLDS}  seeds={SEEDS}  lrs={LRS}  "
          f"fpath_dim={fpath_model.FUTURE_SUMMARY_DIM}")
    print("#  기준: lr=1e-4 (게재 판본).  best val NLL 과 best epoch 을 비교한다.")
    print("#" * 96)

    set_cond_cols(ENC_FULL, "tbill_wr")
    T.MASK_FUTURE_TBILL = False
    T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
    T.ENCODER_MASK_SP = False

    rows = []
    for fold in FOLDS:
        if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                   for s in ("train", "val", "test")):
            print(f"  [skip {fold}] fold CSV 없음")
            continue
        for seed in SEEDS:
            for lr in LRS:
                tag = (f"rvAbl_full_fpath_novol_lr{lr_tag(lr)}"
                       f"_d{fpath_model.FUTURE_SUMMARY_DIM}_s{seed}")
                sp = os.path.join(RESULT_DIR,
                                  f"garch_flow_ar_{tag}_{fold}_summary.json")
                if not os.path.exists(sp):
                    spec = dict(LOCKED)
                    spec.update(fold=fold, seed=seed, tag=tag, lr=lr)
                    print(f"\n  [run] lr={lr:g}  fold={fold}  seed={seed}")
                    t0 = time.time()
                    try:
                        main_worker(spec)
                        print(f"    done ({time.time() - t0:.0f}s)")
                    except Exception as e:                           # noqa: BLE001
                        print(f"    [FAIL] {tag} {fold}: {e!r}")
                        continue
                else:
                    print(f"  [skip] {os.path.basename(sp)} (이미 있음)")
                try:
                    with open(sp, encoding="utf-8") as fh:
                        d = json.load(fh)
                    vp = os.path.join(
                        RESULT_DIR,
                        f"garch_flow_ar_{tag}_{fold}_VAL_summary.json")
                    ve = {}
                    if os.path.exists(vp):
                        with open(vp, encoding="utf-8") as fh:
                            ve = json.load(fh) or {}
                    rows.append(dict(
                        fold=fold, seed=seed, lr=lr,
                        best_val=d.get("best_val_nll"),
                        best_ep=d.get("best_epoch"),
                        v_crps=ve.get("crps_pooled"),
                        v_cov80=ve.get("coverage_80"),
                        v_cov95=ve.get("coverage_95"),
                        v_skew_a=ve.get("skew_actual"),
                        v_skew_s=ve.get("skew_sim"),
                        _test=(d.get("test_eval", {}) or {}),
                    ))
                except Exception as e:                               # noqa: BLE001
                    print(f"    [warn] summary 읽기 실패 {e!r}")

    if not rows:
        print("\n결과 없음")
        return

    def _f(v, fmt, w=9):
        return fmt.format(v) if isinstance(v, (int, float)) else "-".rjust(w)

    print("\n" + "=" * 104)
    print("[선택 기준]  검증셋(val) 지표만.  시험셋은 lr 선택에 쓰지 않는다.")
    print("=" * 104)
    hdr = ("{:16} {:>6} {:>8} {:>10} {:>8} {:>10} {:>8} {:>8} {:>10} {:>10}"
           .format("fold", "seed", "lr", "val_NLL", "best_ep",
                   "val_CRPS", "v_cov80", "v_cov95", "v_skew실측", "v_skew모형"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("{:16} {:>6} {:>8.0e} {:>10} {:>8} {:>10} {:>8} {:>8} {:>10} {:>10}"
              .format(r["fold"], r["seed"], r["lr"],
                      _f(r["best_val"], "{:.4f}", 10),
                      r["best_ep"] if r["best_ep"] else "-",
                      _f(r["v_crps"], "{:.5f}", 10),
                      _f(r["v_cov80"], "{:.3f}", 8),
                      _f(r["v_cov95"], "{:.3f}", 8),
                      _f(r["v_skew_a"], "{:+.4f}", 10),
                      _f(r["v_skew_s"], "{:+.4f}", 10)))

    # lr 별 시드 평균(val 기준)
    print("\n[lr 별 시드 평균 — val]")
    print("{:>8} {:>4} {:>10} {:>8} {:>10} {:>8} {:>8} {:>10}".format(
        "lr", "n", "val_NLL", "best_ep", "val_CRPS", "v_cov80", "v_cov95",
        "v_skew모형"))
    for lr in sorted({r["lr"] for r in rows}):
        g = [r for r in rows if r["lr"] == lr]

        def _m(key):
            v = [r[key] for r in g if isinstance(r[key], (int, float))]
            return sum(v) / len(v) if v else None

        print("{:>8.0e} {:>4} {:>10} {:>8} {:>10} {:>8} {:>8} {:>10}".format(
            lr, len(g), _f(_m("best_val"), "{:.4f}", 10),
            _f(_m("best_ep"), "{:.1f}", 8), _f(_m("v_crps"), "{:.5f}", 10),
            _f(_m("v_cov80"), "{:.3f}", 8), _f(_m("v_cov95"), "{:.3f}", 8),
            _f(_m("v_skew_s"), "{:+.4f}", 10)))

    print("\nlr 은 위 val 표로만 고른다.  고른 뒤에 시험셋을 본다.")
    if os.environ.get("LRS_SHOW_TEST", "0") == "1":
        print("\n[시험셋 — 선택 후 확인용]")
        for r in rows:
            t = r["_test"]
            print("  lr={:.0e} seed={}  CRPS={} cov80={} skew={}".format(
                r["lr"], r["seed"], t.get("crps_pooled"),
                t.get("coverage_80"), t.get("skew_sim")))


if __name__ == "__main__":
    main()
