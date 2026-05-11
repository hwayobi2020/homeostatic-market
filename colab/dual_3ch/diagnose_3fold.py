"""Short diagnostic — scans result/ and prints minimal status per fold (≈10 lines/fold).

각 fold 의:
  - v13 ckpt 5개 존재 여부
  - v13 test_pearson / best_epoch 분포 (요약 통계만)
  - Flow B (Skew-t) ckpt + sanity check 결과
  - Sensitivity CSV 존재 여부

출력 30-40줄 내외. 긴 training log 안 보고 핵심 상태만 확인용.

Usage:
  python colab/dual_3ch/diagnose_3fold.py
"""
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result")


def summarize(arr, fmt="+.3f"):
    arr = [x for x in arr if x is not None]
    if not arr:
        return "—"
    if len(arr) == 1:
        return f"{arr[0]:{fmt}}"
    return f"median {statistics.median(arr):{fmt}}, range [{min(arr):{fmt}}, {max(arr):{fmt}}]"


def diagnose_fold(fold):
    print(f"\n=== {fold} ===")

    # ----- v13 Model A -----
    ckpts = sorted(glob.glob(os.path.join(
        RESULT, f"vol_pilot_3m_psel_vix_v13_*_{fold}_seed*_best.pt")))
    summaries = sorted(glob.glob(os.path.join(
        RESULT, f"vol_pilot_3m_psel_vix_v13_*_{fold}_seed*_summary.json")))
    seeds_found = sorted([int(p.split("seed")[-1].split("_")[0]) for p in ckpts])
    print(f"  v13 ckpts        : {len(ckpts)}/5  seeds={seeds_found}")

    pearsons, r2s, best_eps, val_p = [], [], [], []
    for p in summaries:
        try:
            with open(p) as f:
                s = json.load(f)
            pearsons.append(s.get("test_pearson_std"))
            r2s.append(s.get("test_r2"))
            best_eps.append(s.get("best_epoch"))
            val_p.append(s.get("val_pearson"))
        except Exception:
            pass
    print(f"  v13 test_pearson : {summarize(pearsons)}")
    print(f"  v13 test_r2      : {summarize(r2s)}")
    print(f"  v13 best_epoch   : {summarize(best_eps, '.0f')}")
    print(f"  v13 val_pearson  : {summarize(val_p)}")

    # ----- Flow B -----
    flow_ckpt = os.path.join(RESULT, f"scenario_3m_flow_1d_{fold}_skewt_df5.pt")
    flow_meta = os.path.join(RESULT, f"scenario_3m_flow_1d_{fold}_skewt_df5_meta.json")
    print(f"  Flow B ckpt      : {'✓' if os.path.exists(flow_ckpt) else '✗'}   meta: {'✓' if os.path.exists(flow_meta) else '✗'}")
    if os.path.exists(flow_meta):
        try:
            with open(flow_meta) as f:
                fm = json.load(f)
            sanity = fm.get("sanity", {})
            print(f"  Flow train ε emp : skew {fm.get('train_eps_skew'):+.3f}, "
                  f"ex_kurt {fm.get('train_eps_ex_kurt'):+.3f}")
            print(f"  Flow gen sanity  : skew {sanity.get('gen_skew'):+.3f}, "
                  f"ex_kurt {sanity.get('gen_ex_kurt'):+.3f}, "
                  f"Δkurt {sanity.get('delta_ex_kurt'):+.3f}")
            if fm.get("alpha_final") is not None:
                print(f"  Flow α (skew_t)  : {fm.get('alpha_init')} → {fm.get('alpha_final'):+.3f}")
        except Exception as e:
            print(f"  Flow meta read err: {e}")

    # ----- Sensitivity -----
    sens_csv = os.path.join(RESULT, f"sensitivity_v13_summary_{fold}_global.csv")
    hist_csv = os.path.join(RESULT, f"sensitivity_v13_histogram_stats_{fold}_global.csv")
    print(f"  Sensitivity csvs : summary {'✓' if os.path.exists(sens_csv) else '✗'},  "
          f"hist {'✓' if os.path.exists(hist_csv) else '✗'}")

    # ----- Figures -----
    figs = [
        f"sensitivity_v13_curves_{fold}_global.png",
        f"sensitivity_v13_fanchart_{fold}_global.png",
        f"sensitivity_v13_histogram_{fold}_global.png",
    ]
    fig_status = "".join("✓" if os.path.exists(os.path.join(RESULT, f)) else "✗" for f in figs)
    print(f"  Figures (3)      : {fig_status}  (curves/fanchart/histogram)")


def main():
    print("=" * 72)
    print(" 3-fold holdout diagnostic")
    print("=" * 72)
    if not os.path.isdir(RESULT):
        print(f"[FATAL] result dir not found: {RESULT}")
        return
    print(f"  result_dir: {RESULT}")
    for fold in ["F1", "F2", "F3"]:
        diagnose_fold(fold)
    print()
    # Final aggregate (only if 3fold aggregate exists)
    agg = os.path.join(RESULT, "sensitivity_v13_summary_3fold_global.csv")
    if os.path.exists(agg):
        print(f"  3-fold aggregate CSV exists: {agg}")
    else:
        print(f"  3-fold aggregate CSV NOT yet — orchestrator did not finish all folds.")


if __name__ == "__main__":
    main()
