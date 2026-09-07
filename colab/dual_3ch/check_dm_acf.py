"""DM 검정 진단 — d 의 자기상관과 HAC 보정 효과 확인.

run_thin_compare.py 가 저장한 dm_compare*.json 에는 검정 결과만 있고 원점별
CRPS 계열은 없다.  여기서는 그 계열을 다시 만들지 않고, 저장된 요약과 함께
다음을 점검한다.

  1. d = CRPS_baseline − CRPS_MACFlow 의 자기상관 (lag 1..20)
  2. HAC se 와 단순 se 의 비율 — 보정이 얼마나 표준오차를 키웠는지
  3. HAC se 가 유독 작은 셀(F_long_A vs CondGAN 등)의 d 분포

원점별 CRPS 계열이 필요하므로 run_thin_compare 의 수집 부분을 재사용한다.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/check_dm_acf.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import run_thin_compare as R                                             # noqa: E402
import torch                                                             # noqa: E402
from types import SimpleNamespace                                        # noqa: E402


def acf(x, nlags=20):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    d0 = float(np.dot(x, x)) / n
    return [float(np.dot(x[k:], x[:-k]) / n / d0) for k in range(1, nlags + 1)]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print("# DM 진단 — d 의 자기상관과 HAC 보정 효과")
    print("#" * 100)

    for fold in R.FOLDS:
        print(f"\n===== {fold}")
        per_model = {}

        for mk in ("vae", "gan"):
            runs = []
            for sd in R.SEEDS:
                try:
                    spv = os.path.join(R.PS.RESULT_DIR,
                                       f"{mk}_baseline_{fold}_summary.json")
                    bakv = R._keep_summary(spv)
                    args = SimpleNamespace(model=mk, fold=fold,
                                           folds_dir=R.PS.FOLDS_DIR,
                                           out_dir=R.PS.RESULT_DIR,
                                           n_sim=R.N_SIM, seed=sd)
                    _saved = list(R.T.COND_COLS)
                    R.PS.set_cond_cols(R.VG_COND_COLS)
                    R.VG.COND_COLS = list(R.T.COND_COLS)
                    R.VG.N_CH = len(R.VG.COND_COLS)
                    R.VG.SP_CH, R.VG.TBILL_CH = R.T.SP_CH, R.T.TBILL_CH
                    try:
                        runs.append(R.VG.run_fold(mk, fold, args, device))
                    finally:
                        R.PS.set_cond_cols(_saved)
                    R._restore_summary(spv, bakv)
                except Exception as e:
                    print(f"  [FAIL {mk} s{sd}] {e!r}")
            if runs:
                per_model[f"Cond{mk.upper()}"] = runs

        runs = []
        for sd in R.SEEDS:
            try:
                d = R.macflow_arrays(fold, sd, device)
                if d is not None:
                    runs.append(d)
            except Exception as e:
                print(f"  [FAIL MAC-Flow s{sd}] {e!r}")
        if runs:
            per_model["MAC-Flow"] = runs

        if "MAC-Flow" not in per_model or len(per_model) < 2:
            print("  [skip]"); continue

        rep = {k: v[0] for k, v in per_model.items()}
        _, common = R.align(rep)
        order = np.argsort(common)

        loss = {}
        for name, rs in per_model.items():
            ps_list = []
            for d in rs:
                ps = np.asarray(d["pred_start"], int)
                pos = {int(v): i for i, v in enumerate(ps)}
                sel = np.array([pos[v] for v in common], int)
                ps_list.append(R.crps_per_origin(np.asarray(d["sim"])[sel],
                                                 np.asarray(d["act"])[sel]))
            loss[name] = np.mean(ps_list, axis=0)[order]

        base = loss["MAC-Flow"]
        for name in ("CondVAE", "CondGAN"):
            if name not in loss:
                continue
            d = loss[name] - base
            n = len(d)
            se_iid = float(d.std(ddof=1) / np.sqrt(n))
            import dm_test as DM
            se_hac = DM.newey_west_se(d, R.DM_LAG)
            a = acf(d, 20)
            print(f"\n  MAC-Flow vs {name}  n={n}")
            print(f"    d: mean={d.mean():+.5f}  sd={d.std(ddof=1):.5f}  "
                  f"min={d.min():+.5f}  max={d.max():+.5f}  양수 {int((d>0).sum())}/{n}")
            print(f"    se: iid={se_iid:.5f}  HAC(lag{R.DM_LAG})={se_hac:.5f}  "
                  f"비율={se_hac/se_iid:.2f}배")
            print("    ACF lag  1-6 : " + " ".join(f"{v:+.3f}" for v in a[:6]))
            print("    ACF lag  7-12: " + " ".join(f"{v:+.3f}" for v in a[6:12]))
            print("    ACF lag 13-20: " + " ".join(f"{v:+.3f}" for v in a[12:20]))
            if abs(a[12]) > 0.2:
                print("    [주의] lag 13 이후에도 자기상관이 남아 lag=12 가 부족할 수 있다.")

    print("\n[읽는 법] 비율이 1 에 가까우면 HAC 보정이 거의 안 먹은 것이다.")
    print("  겹침이 실재하면 비율이 1 보다 뚜렷이 커야 한다.")
    print("  ACF 가 lag 12 안에서 0 으로 죽으면 lag=12 설정이 적절하다.")


if __name__ == "__main__":
    main()
