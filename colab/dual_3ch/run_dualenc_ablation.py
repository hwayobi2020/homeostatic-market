# -*- coding: utf-8 -*-
"""R4#1 이중 인코더 ablation — 미래경로 인코더와 스텝별 주입의 한계 기여를 분리한다.

무엇을 묻는가
--------------
논문은 미래 가상 경로를 두 경로로 넣는다 (Figure 1):
  (A) 미래 맥락 인코더 : 13 주 경로 전체를 요약해 모든 스텝에 같은 값으로 브로드캐스트
  (B) 스텝별 인코더    : 시점 τ 의 경로 값을 그 스텝에만 주입
리뷰어는 둘 다 필요한지, 각각의 한계 기여가 무엇인지 묻는다.  그래서 2x2 중 세 칸을 돌린다
(네 번째 칸 = 미래 경로를 아예 안 주는 것은 조건부 모형이 아니므로 제외).

  both      A+B   본모형            tag rvAbl_full_fpath_novol_d2          (이미 학습됨 → 재사용)
  summary   A만   스텝별 미래 마스킹  tag rvAbl_summary_only_novol_d2        fpath_model.FPATH_SUMMARY_ONLY=True
  perstep   B만   미래 요약 인코더 제거 tag rvAbl_perstep_only_novol           베이스 MambaFlowAR (fpath 패치 해제)

셋은 같은 LOCKED spec, 같은 폴드, 같은 시드, 같은 학습 절차(검증 NLL 최저 복원, patience 30)를 쓴다.
차이는 미래 경로가 맥락에 들어가는 경로뿐이다.

무엇을 내놓는가
----------------
폴드 × 변형별 시험셋 CRPS / cov80 / cov95 / 생성 왜도 (5 시드 평균 ± 표준편차) 와
본모형 대비 차이, 그리고 시드 짝지은 t 검정 (같은 시드끼리 짝 → 학습 잡음 상쇄).
result/dualenc_ablation.json 에 저장.

사용 (Colab)
------------
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/run_dualenc_ablation.py
  시드/변형을 줄이려면  DE_SEEDS=2026,2027  DE_VARIANTS=summary,perstep
"""
import json
import os
import sys
import time

os.environ["FPATH_NOVOL"] = "1"
os.environ["FPATH_DIM"] = os.environ.get("FPATH_DIM", "2")
os.environ["FPATH_SUMMARY_ONLY"] = "0"          # 변형별로 런타임에 바꾼다
os.environ.pop("FPATH_HEAD", None)              # 흐름 헤드 (본모형)

import torch                                     # noqa: E402  (pandas 보다 먼저)
import numpy as np                               # noqa: E402
from scipy import stats                          # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_full_fpath as R                       # noqa: E402  (T.MambaFlowAR 를 Fpath 로 패치)
import fpath_model                               # noqa: E402
import train_garch_flow as T                     # noqa: E402
from train_garch_flow import main_worker         # noqa: E402

# 패치 전 원본 클래스 = Fpath 의 부모.  "미래 요약 없음(perstep)" 변형에 쓴다.
BASE_CLS = fpath_model.MambaFlowARFpath.__mro__[1]
FPATH_CLS = fpath_model.MambaFlowARFpath

DIM = int(os.environ["FPATH_DIM"])
SEEDS = [int(s) for s in os.environ.get("DE_SEEDS", "2026,2027,2028,2029,2030").split(",") if s.strip()]
WANT = [v for v in os.environ.get("DE_VARIANTS", "both,summary,perstep").split(",") if v]
FOLDS = R.FOLDS
OUT = os.path.join(R.RESULT_DIR, "dualenc_ablation.json")
LABEL = {"F_gfc": "Financial crisis (2006-2010)", "F_long_A": "Recovery (2011-2015)",
         "F_long_B_origin": "COVID (2016-2020)", "F_long": "Tightening (2021-2025)"}
KEYS = [("crps_pooled", "CRPS"), ("coverage_80", "cov80"), ("coverage_95", "cov95"), ("skew_sim", "skew")]

VARIANTS = {
    "both":    dict(tag=f"rvAbl_full_fpath_novol_d{DIM}",    cls=FPATH_CLS, summary_only=False,
                    desc="미래 요약 + 스텝별 주입 (본모형)"),
    "summary": dict(tag=f"rvAbl_summary_only_novol_d{DIM}",  cls=FPATH_CLS, summary_only=True,
                    desc="미래 요약만 (스텝별 미래 마스킹)"),
    "perstep": dict(tag="rvAbl_perstep_only_novol",          cls=BASE_CLS,  summary_only=False,
                    desc="스텝별 주입만 (미래 요약 인코더 없음)"),
}


def summary_path(tag, seed, fold):
    return os.path.join(R.RESULT_DIR, f"garch_flow_ar_{tag}_s{seed}_{fold}_summary.json")


def train_variant(name):
    v = VARIANTS[name]
    print("\n" + "=" * 96)
    print(f"[{name}] {v['desc']}   tag={v['tag']}_s*")
    print("=" * 96)
    R.set_cond_cols(R.ENC_FULL, "tbill_wr")
    T.MASK_FUTURE_TBILL = False
    T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
    T.ENCODER_MASK_SP = False
    T.MambaFlowAR = v["cls"]                       # ★ 변형별 모델 클래스
    fpath_model.FPATH_SUMMARY_ONLY = v["summary_only"]
    for seed in SEEDS:
        for fold in FOLDS:
            sp = summary_path(v["tag"], seed, fold)
            if os.path.exists(sp):
                print(f"  [skip] {os.path.basename(sp)}"); continue
            spec = dict(R.LOCKED); spec.update(fold=fold, seed=seed, tag=f"{v['tag']}_s{seed}",
                                               folds_dir=R.FOLDS_DIR)
            print(f"  [run] {v['tag']}_s{seed} {fold}"); t0 = time.time()
            try:
                main_worker(spec); print(f"    done ({time.time() - t0:.0f}s)")
            except Exception as e:                  # noqa: BLE001
                print(f"    [FAIL] {name} s{seed} {fold}: {e!r}")


def load(name, fold):
    """변형·폴드의 시드별 test_eval (시드 순서 유지, 없으면 None)."""
    v = VARIANTS[name]
    out = []
    for s in SEEDS:
        p = summary_path(v["tag"], s, fold)
        m = json.load(open(p, encoding="utf-8")).get("test_eval") if os.path.exists(p) else None
        out.append(m)
    return out


def report():
    print("\n" + "#" * 118)
    print("# 이중 인코더 ablation — 같은 spec·폴드·시드, 미래 경로가 맥락에 들어가는 경로만 다르다.")
    print(f"#   시드 {SEEDS}.  값은 5 시드 평균±표준편차.  p 는 본모형과 시드를 짝지은 t 검정(양측).")
    print("#" * 118)
    out = {}
    hdr = (f"{'Test period':<30}{'metric':<7}{'both(본모형)':>18}{'summary only':>18}{'Δ':>10}{'p':>8}"
           f"{'per-step only':>18}{'Δ':>10}{'p':>8}")
    print(hdr); print("-" * len(hdr))
    for fold in FOLDS:
        data = {n: load(n, fold) for n in VARIANTS}
        if not any(m for m in data["both"]):
            print(f"{LABEL[fold]:<30}(both 결과 없음)"); continue
        row = {}
        first = True
        for k, name in KEYS:
            base = [m[k] for m in data["both"] if m and k in m]
            cells, stats_row = [], {}
            for other in ("summary", "perstep"):
                pair = [(b[k], o[k]) for b, o in zip(data["both"], data[other]) if b and o and k in b and k in o]
                if not pair:
                    cells.append(("—", float("nan"), float("nan"))); continue
                bv = np.array([p[0] for p in pair]); ov = np.array([p[1] for p in pair])
                t = stats.ttest_rel(ov, bv) if len(pair) > 1 else None
                cells.append((f"{ov.mean():.4f}±{ov.std(ddof=1):.4f}" if len(pair) > 1 else f"{ov.mean():.4f}",
                              float(ov.mean() - bv.mean()),
                              float(t.pvalue) if t is not None else float("nan")))
                stats_row[other] = dict(mean=float(ov.mean()),
                                        sd=float(ov.std(ddof=1)) if len(pair) > 1 else None,
                                        delta=float(ov.mean() - bv.mean()),
                                        p_paired=float(t.pvalue) if t is not None else None, n_pair=len(pair))
            b_txt = f"{np.mean(base):.4f}±{np.std(base, ddof=1):.4f}" if len(base) > 1 else f"{np.mean(base):.4f}"
            print(f"{LABEL[fold] if first else '':<30}{name:<7}{b_txt:>18}"
                  f"{cells[0][0]:>18}{cells[0][1]:>+10.4f}{cells[0][2]:>8.4f}"
                  f"{cells[1][0]:>18}{cells[1][1]:>+10.4f}{cells[1][2]:>8.4f}")
            first = False
            stats_row["both"] = dict(mean=float(np.mean(base)),
                                     sd=float(np.std(base, ddof=1)) if len(base) > 1 else None)
            row[name] = stats_row
        m0 = next((m for m in data["both"] if m), None)
        if m0 and "skew_actual" in m0:
            print(f"{'':<30}{'(실측 skew':<7}{m0['skew_actual']:>18.4f})")
        out[fold] = row
        print("-" * len(hdr))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"saved {OUT}")
    print("Δ = 변형 − 본모형.  CRPS 는 낮을수록, 커버리지는 명목(0.80/0.95)에 가까울수록, 왜도는 실측 부호에 가까울수록 좋다.")


if __name__ == "__main__":
    print(f"[run_dualenc_ablation] variants={WANT} seeds={SEEDS} dim={DIM}")
    for name in WANT:
        if name not in VARIANTS:
            sys.exit(f"[FATAL] 모르는 변형: {name}")
        train_variant(name)
    report()
