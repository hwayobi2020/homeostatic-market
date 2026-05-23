"""F_gfc — 자기통계량 재배분(self-stat DC) vs metab_both vs GARCH.

가설 (사용자 2026-05-23):
  sp_return 분포의 충분통계량 — 위치(직전 수익률)·폭(직전 변동성) — 은 encoder 가
  거시변수와 섞어 압축할 게 아니라 Flow head 에 '직접' 줘야 한다. 거시(metab)는 encoder.
  → GARCH 의 자기회귀 구조(직전 ε·σ² → 다음 분포)를 Flow head 에 명시적으로 옮긴 구성.
  목적: GARCH 가 이긴 F_gfc(위기)의 under-dispersion(std_ratio 0.64)을 끌어올리나?

배분 (사용자 확정):
  encoder(MLP, 5ch) : sp_return, tbill_wr, ads_lag, wti_wr, metab_13w
  Flow head 직접     : sp_return (per-step, direct_prev_return=True)
                       sp_std_13w, sp_log_std_13w (DC, origin 변동성)
  마스크             : MASK_FUTURE_TBILL=True  (GARCH 와 정보 대칭, 공정)
  변동성은 encoder 에서 제거(=DC 전용), metab 은 DC 제거(=encoder embed 전용).

비교 (전부 F_gfc, 마스크 = 정보 대칭):
  selfstat_mask_mlp   : 이 새 구성 (5 seed)
  metab_both_mask_mlp : 기존 마스크 best (이미 있음, 대조)
  GARCH-N / GARCH-t   : floor (이미 있음)

전제: F_gfc fold + metab_both_mask + GARCH 결과가 이미 있어야 비교표가 채워짐.
NLL 주의: Flow=joint, GARCH=marginal → 직접 비교 불가(참고).

Usage (Colab):
    !python colab/dual_3ch/run_gfc_selfstat.py
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
SEEDS = ALL_SEEDS          # quick peek 만 원하면 [2026]

# encoder(MLP) 채널 — 변동성 빼고 metab 넣음
ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
# Flow head DC — 직전 변동성(origin)
DC_COLS = "sp_std_13w,sp_log_std_13w"


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


# ── 가드: F_gfc fold CSV ──
for sp in ("train", "val", "test"):
    p = os.path.join(FOLDS_DIR, f"{FOLD}_{sp}.csv")
    if not os.path.exists(p):
        sys.exit(f"[FATAL] missing {p}\n  → 먼저: !python data/build_gfc_fold.py")

print(f"[GFC selfstat] fold={FOLD}  encoder={ENC_COLS}")
print(f"               Flow head: direct_prev_return=True, DC={DC_COLS}  MASK=True")

# ── 학습 (self-stat 재배분, 마스크) ──
set_cond_cols(ENC_COLS)
T.MASK_FUTURE_TBILL = True
spec_base = dict(BEST_SPECS["mlp"])    # encoder_type="mlp", mlp_num_layers=4 ...
for seed in SEEDS:
    tag = f"selfstat_mask_mlp_s{seed}"
    p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{FOLD}_summary.json")
    if os.path.exists(p):
        print(f"[skip] {os.path.basename(p)}")
        continue
    spec = dict(spec_base)
    spec.update(fold=FOLD, seed=seed, tag=tag,
                extra_context_channels=DC_COLS, direct_prev_return=True)
    print(f"\n[flow] {tag} fold={FOLD}  (5ch enc + prevret + volDC, MASK)")
    try:
        main_worker(spec)
    except Exception as e:
        print(f"[FLOW FAIL] {tag} {FOLD}: {e!r}")


# ── 비교표 ──
KEYS = ["crps_pooled", "emd", "coverage_95", "cvar_5pct_diff", "std_ratio",
        "per_week_nll_z"]


def flow_te(tag):
    p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{FOLD}_summary.json")
    return json.load(open(p)).get("test_eval", {}) if os.path.exists(p) else None


def garch(gtag):
    p = os.path.join(RESULT_DIR, f"garch_{gtag}_{FOLD}_summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def pool(tag_fmt):
    acc = {k: [] for k in KEYS}
    for s in SEEDS:
        te = flow_te(tag_fmt.format(s=s))
        if not te:
            continue
        for k in KEYS:
            if te.get(k) is not None:
                acc[k].append(te[k])
    return acc


def fmt(v):
    if not v:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:+.4f}"
    return f"{st.mean(v):+.4f}±{st.stdev(v):.3f}"


def n_orig(tag_fmt):
    for s in SEEDS:
        te = flow_te(tag_fmt.format(s=s))
        if te:
            return te.get("n_test_origins")
    return None


print("\n" + "=" * 108)
print(f"F_gfc (GFC 위기, 2006-2010) — 자기통계량 재배분 vs metab_both vs GARCH  (전부 마스크=정보대칭)")
print("  CRPS/EMD 낮을수록 | cov95 0.95 근접 | CVaR5Δ 0 근접 | std_ratio 1.0 근접 | NLL 비교불가(참고)")
print("=" * 108)
print(f"{'model':<30}{'n':>3} {'CRPS':>14}{'EMD':>13}{'cov95':>13}"
      f"{'CVaR5Δ':>13}{'std_ratio':>13}{'NLL':>15}")

for label, fmtstr in [("metab_both (mask, 기존)", "metab_both_mask_mlp_s{s}"),
                      ("selfstat (prevret+volDC)", "selfstat_mask_mlp_s{s}")]:
    a = pool(fmtstr)
    print(f"{label:<30}{len(a['crps_pooled']):>3} {fmt(a['crps_pooled']):>14}"
          f"{fmt(a['emd']):>13}{fmt(a['coverage_95']):>13}{fmt(a['cvar_5pct_diff']):>13}"
          f"{fmt(a['std_ratio']):>13}{fmt(a['per_week_nll_z']):>15}")

for label, gtag in [("GARCH-N (floor)", "pure"), ("GARCH-t (floor)", "pure_t")]:
    g = garch(gtag)
    if not g:
        print(f"{label:<30}{'0':>3} {'n/a':>14}")
        continue
    nll = g.get("per_week_nll_z")
    nll_s = f"{nll:+.4f}" if nll is not None else "marg(skip)"
    print(f"{label:<30}{'1':>3} {g['crps_pooled']:+.4f}      {g['emd']:+.4f}     "
          f"{g['coverage_95']:+.4f}     {g['cvar_5pct_diff']:+.4f}     "
          f"{g['std_ratio']:+.4f}    {nll_s:>12}")

fo = n_orig("selfstat_mask_mlp_s{s}")
g = garch("pure")
go = g.get("n_test_origins") if g else None
print(f"\n[origin parity] selfstat n_origins={fo}  GARCH={go}  "
      f"{'OK' if fo == go else '!!! 불일치 — 비교 주의'}")
print()
