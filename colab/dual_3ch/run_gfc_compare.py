"""F_gfc (리만/GFC 위기 fold) 에서 best 구성(metab_both, MLP, 미래금리 마스크) vs GARCH.

사용자 요청 (2026-05-23):
  "가장 성공적인거(=CH7+both, metab embed+DC, MLP)랑 garch만 돌려서 비교 일단."
  → 이후 "마스크가 GARCH 하고 비교하기엔 맞는거네, 마스크로 비교하자."

왜 마스크인가 (GARCH 와 정보 대칭):
  train_mamba_flow_ar.load_windows_seq 는 미래 구간[past_len:]의 모든 macro 채널을
  이미 0 마스킹한다 (metab embed 포함). 미래에 실제값을 보는 채널은 tbill_wr 하나뿐.
  → MASK_FUTURE_TBILL=True 로 그 tbill 마저 가리면 Flow 는 미래 exogenous 정보를
  전혀 안 본다 = GARCH(과거 sp_return 만) 와 정보 대칭. metab 은 (과거 embed +
  origin 시점 DC) 로만 작동 → 미래 leak 없음. 공정한 head-to-head.

전제: data/build_gfc_fold.py 가 먼저 실행돼 F_gfc_{train,val,test}.csv 존재해야 함.

대상:
  metab_both (MLP, MASK)  : CH6 + metab_13w(embed) + metab_13w(DC), 미래 tbill 마스크
                            태그 metab_both_mask_mlp  (기존 논마스크 abl_metab_both_mlp 와 분리)
  GARCH-N / GARCH-t       : ConstantMean + GARCH(1,1) + Normal/Student-t  (floor)
  (ratemask_mlp = 마스크 ch6 가 F_gfc 에 있으면 baseline row 자동 포함)

NLL 주의: Flow=teacher-forced joint, GARCH=multi-step marginal → 직접 비교 불가
(참고용). 의미축: CRPS / cov95 / CVaR5Δ / std_ratio.

⚠️ GARCH 는 이 스크립트가 직접 안 띄운다 (메모리 겹침 방지). 부모(torch + 학습된
   Flow) 안에서 subprocess 로 train_garch_ar.py(=train_flow_seq import → torch+nflows
   재로드)를 띄우면 메모리 2배 → Colab 런타임 OOM. GARCH 는 아래처럼 독립 셀로 먼저.

Usage (Colab):
    # ① GARCH (독립 셀 — 깨끗한 프로세스, 가벼움)
    !python colab/dual_3ch/train_garch_ar.py --fold F_gfc --use-arx 0 --dist normal
    !python colab/dual_3ch/train_garch_ar.py --fold F_gfc --use-arx 0 --dist t
    # ② Flow 학습(이미 있으면 skip) + 비교표
    !python colab/dual_3ch/run_gfc_compare.py
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import train_mamba_flow_ar as T            # noqa: E402
from train_mamba_flow_ar import main_worker  # noqa: E402
from best_specs import BEST_SPECS           # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLD = "F_gfc"
ALL_SEEDS = [2026, 2027, 2028, 2029, 2030]
SEEDS = ALL_SEEDS          # quick peek 만 원하면 [2026] 로.
CH6 = ["sp_return", "tbill_wr", "ads_lag", "sp_std_13w", "wti_wr", "sp_log_std_13w"]
M = "metab_13w"


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


# ── 가드: F_gfc fold CSV 존재 확인 ──
for sp in ("train", "val", "test"):
    p = os.path.join(FOLDS_DIR, f"{FOLD}_{sp}.csv")
    if not os.path.exists(p):
        sys.exit(f"[FATAL] missing {p}\n  → 먼저 실행: !python data/build_gfc_fold.py")

print(f"[GFC compare] fold={FOLD}  metab_both(MLP, MASK) {len(SEEDS)} seed + GARCH-N/t")

# ── 1) metab_both (MLP) 학습 — 미래 tbill 마스크 (GARCH 와 정보 대칭) ──
set_cond_cols(CH6 + [M])
T.MASK_FUTURE_TBILL = True          # <-- 미래 금리경로 숨김 = GARCH 와 공정 비교
spec_base = dict(BEST_SPECS["mlp"])    # encoder_type="mlp" 내장
for seed in SEEDS:
    tag = f"metab_both_mask_mlp_s{seed}"
    p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{FOLD}_summary.json")
    if os.path.exists(p):
        print(f"[skip flow] {os.path.basename(p)}")
        continue
    spec = dict(spec_base)
    spec.update(fold=FOLD, seed=seed, tag=tag, extra_context_channels=M)
    print(f"\n[flow] {tag} fold={FOLD}  (MASK_FUTURE_TBILL=True)")
    try:
        main_worker(spec)
    except Exception as e:
        print(f"[FLOW FAIL] {tag} {FOLD}: {e!r}")

# ── 2) GARCH 는 독립 셀로 (메모리 겹침 방지) — 여기선 존재만 점검 ──
#     없으면 아래 명령을 별도 셀에서 먼저 돌릴 것 (subprocess 로 안 띄움).
for dist in ("normal", "t"):
    gtag = "pure" if dist == "normal" else "pure_t"
    p = os.path.join(RESULT_DIR, f"garch_{gtag}_{FOLD}_summary.json")
    if not os.path.exists(p):
        print(f"[GARCH 없음] {os.path.basename(p)} — 먼저 독립 셀로 실행:")
        print(f"  !python colab/dual_3ch/train_garch_ar.py "
              f"--fold {FOLD} --use-arx 0 --dist {dist}")


# ── 3) 비교표 ──
def flow_metrics(tag):
    p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{FOLD}_summary.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("test_eval", {})


def garch_metrics(gtag):
    p = os.path.join(RESULT_DIR, f"garch_{gtag}_{FOLD}_summary.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))   # GARCH 는 top-level


KEYS = ["crps_pooled", "emd", "coverage_95", "cvar_5pct_diff", "std_ratio",
        "per_week_nll_z"]


def pool_flow(tag_fmt):
    acc = {k: [] for k in KEYS}
    for s in SEEDS:
        te = flow_metrics(tag_fmt.format(s=s))
        if not te:
            continue
        for k in KEYS:
            if te.get(k) is not None:
                acc[k].append(te[k])
    return acc


def fmt(vals):
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:+.4f}"
    return f"{st.mean(vals):+.4f}±{st.stdev(vals):.3f}"


# 원점 수(origin parity) 확인용
def flow_n_origin(tag_fmt):
    for s in SEEDS:
        p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag_fmt.format(s=s)}_{FOLD}_summary.json")
        if os.path.exists(p):
            te = json.load(open(p)).get("test_eval", {})
            return te.get("n_test_origins")
    return None


print("\n" + "=" * 104)
print(f"F_gfc (리만/GFC 위기, test 2006-2010) — best Flow(metab_both, MASK) vs GARCH floor")
print("  미래 금리경로 마스크 = Flow·GARCH 모두 미래 exogenous 안 봄 (정보 대칭, 공정)")
print("  CRPS/EMD 낮을수록 | cov95 0.95 근접 | CVaR5Δ 0 근접 | std_ratio 1.0 근접 | "
      "NLL 은 model-class 간 비교 불가(참고)")
print("=" * 104)
hdr = (f"{'model':<28}{'n':>3} {'CRPS':>14}{'EMD':>13}{'cov95':>13}"
       f"{'CVaR5Δ':>13}{'std_ratio':>13}{'NLL(비교불가)':>16}")
print(hdr)

rows = []
base = pool_flow("ratemask_mlp_s{s}")          # 마스크 ch6 (있으면 baseline)
if any(base.values()):
    rows.append(("ch6_mlp (mask, no metab)", base, len(base["crps_pooled"])))
both = pool_flow("metab_both_mask_mlp_s{s}")
rows.append(("metab_both MLP (mask, best)", both, len(both["crps_pooled"])))

for label, acc, n in rows:
    print(f"{label:<28}{n:>3} {fmt(acc['crps_pooled']):>14}{fmt(acc['emd']):>13}"
          f"{fmt(acc['coverage_95']):>13}{fmt(acc['cvar_5pct_diff']):>13}"
          f"{fmt(acc['std_ratio']):>13}{fmt(acc['per_week_nll_z']):>16}")

for label, gtag in [("GARCH-N (floor)", "pure"), ("GARCH-t (floor)", "pure_t")]:
    g = garch_metrics(gtag)
    if not g:
        print(f"{label:<28}{'0':>3} {'n/a':>14}")
        continue
    nll = g.get("per_week_nll_z")
    nll_s = f"{nll:+.4f}" if nll is not None else "marg(skip)"
    print(f"{label:<28}{'1':>3} {g['crps_pooled']:+.4f}      "
          f"{g['emd']:+.4f}     {g['coverage_95']:+.4f}     "
          f"{g['cvar_5pct_diff']:+.4f}     {g['std_ratio']:+.4f}     {nll_s:>12}")

# origin parity 점검 출력
fo = flow_n_origin("metab_both_mask_mlp_s{s}")
g = garch_metrics("pure")
go = g.get("n_test_origins") if g else None
print(f"\n[origin parity] Flow n_test_origins={fo}  GARCH valid origins={go}  "
      f"{'OK(일치)' if fo == go else '!!! 불일치 — 비교 주의'}")
print()
