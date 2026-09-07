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
                    te = d.get("test_eval", {}) or {}
                    rows.append(dict(
                        fold=fold, seed=seed, lr=lr,
                        best_val=d.get("best_val_nll"),
                        best_ep=d.get("best_epoch"),
                        crps=te.get("crps_pooled"),
                        cov80=te.get("coverage_80"),
                        cov95=te.get("coverage_95"),
                        skew_a=te.get("skew_actual"),
                        skew_s=te.get("skew_sim"),
                    ))
                except Exception as e:                               # noqa: BLE001
                    print(f"    [warn] summary 읽기 실패 {e!r}")

    if not rows:
        print("\n결과 없음")
        return

    print("\n" + "=" * 104)
    print("[요약]  본모형 spec 고정, lr 만 변경")
    print("=" * 104)
    hdr = ("{:16} {:>6} {:>8} {:>10} {:>8} {:>9} {:>8} {:>8} {:>9} {:>9}"
           .format("fold", "seed", "lr", "best_val", "best_ep",
                   "CRPS", "cov80", "cov95", "skew실측", "skew모형"))
    print(hdr)
    print("-" * len(hdr))

    def _f(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float)) else "{:>9}".format("-")

    for r in rows:
        print("{:16} {:>6} {:>8.0e} {:>10} {:>8} {:>9} {:>8} {:>8} {:>9} {:>9}".format(
            r["fold"], r["seed"], r["lr"],
            _f(r["best_val"], "{:.4f}"), r["best_ep"] if r["best_ep"] else "-",
            _f(r["crps"], "{:.5f}"), _f(r["cov80"], "{:.3f}"),
            _f(r["cov95"], "{:.3f}"), _f(r["skew_a"], "{:+.4f}"),
            _f(r["skew_s"], "{:+.4f}")))

    base = [r for r in rows if abs(r["lr"] - 1e-4) < 1e-12]
    if base and len(rows) > len(base):
        b = base[0]
        print(f"\n기준 lr=1e-4 : best_val={b['best_val']} @ep{b['best_ep']}")
        print("best_val 이 더 낮아지면 과소학습, 더 높아지면 lr 과다.")


if __name__ == "__main__":
    main()
