"""self-stat cond 구성 하이퍼파라미터 재튜닝 (gap29 + metab_26w + 미래 tbill/metab unmask).

BEST_SPECS["mlp"] 는 옛 setup(8채널 base, gap15, metab_13w) 튜닝이라 현재 구성엔 안 맞음.
→ 현재 구성(self-stat cond)에서 val NLL 로 하이퍼파라미터 재선택. test 는 안 봄(누출 방지).

고정: self-stat cond — encoder MLP 5채널(sp_return,tbill,ads,wti,metab_26w),
      Flow head = prevret + volDC(sp_std_13w,sp_log_std_13w),
      미래 unmask = tbill + metab_26w.
그리드: d_model{128,256} x mlp_num_layers{3,4} x dropout{0.1,0.2} x Flow{Light,Heavy} = 16.
선택: best_val_nll_per_week (fold 평균) 최소.  test sampling 은 n_sim=100 으로 가볍게.

Usage (Colab): !python colab/dual_3ch/run_tune_cond.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import train_mamba_flow_ar as T            # noqa: E402
from train_mamba_flow_ar import main_worker  # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEED = 2026
ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_26w"]
DC_COLS = "sp_std_13w,sp_log_std_13w"
N_SIM_TUNE = 100   # val NLL 로 고르므로 test sampling 은 가볍게

FLOWS = {
    "Light": dict(n_flow_layers=2, n_flow_hidden=32, weight_decay=0.1),
    "Heavy": dict(n_flow_layers=6, n_flow_hidden=64, weight_decay=0.5),
}


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


GRID = []
for dm in (128, 256):
    for nl in (3, 4):
        for dr in (0.1, 0.2):
            for fname, fkw in FLOWS.items():
                cap = f"dm{dm}_nl{nl}_dr{dr}_{fname}"
                hp = dict(d_model=dm, mlp_num_layers=nl, dropout=dr, **fkw)
                GRID.append((cap, hp))

set_cond_cols(ENC_COLS)
T.MASK_FUTURE_TBILL = False
T.FUTURE_UNMASK_MACRO_COLS = ["metab_26w"]
print(f"[tune cond] {len(GRID)} grid x {len(FOLDS)} fold = {len(GRID) * len(FOLDS)} run "
      f"(self-stat cond, val NLL 기준)")

t0 = time.time()
for cap, hp in GRID:
    for fold in FOLDS:
        tag = f"tunecond_{cap}_s{SEED}"
        p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{fold}_summary.json")
        if os.path.exists(p):
            print(f"[skip] {os.path.basename(p)}")
            continue
        spec = dict(encoder_type="mlp", fold=fold, seed=SEED, tag=tag,
                    extra_context_channels=DC_COLS, direct_prev_return=True,
                    n_sim=N_SIM_TUNE, **hp)
        print(f"\n[run] {tag} fold={fold}")
        try:
            main_worker(spec)
        except Exception as e:
            print(f"[FAIL] {tag} {fold}: {e!r}")

# ── val NLL (best_val_nll_per_week) fold 평균으로 best 선택 ──
print(f"\n[tune done] elapsed={(time.time() - t0) / 60:.1f} min")
print("\n" + "=" * 70)
print("self-stat cond 하이퍼파라미터 — best_val_nll/week (fold 평균, 낮을수록)")
print("=" * 70)
results = {}
for cap, hp in GRID:
    vals = []
    for fold in FOLDS:
        p = os.path.join(RESULT_DIR, f"mamba_flow_ar_tunecond_{cap}_s{SEED}_{fold}_summary.json")
        if os.path.exists(p):
            s = json.load(open(p))
            v = s.get("best_val_nll_per_week")
            if v is not None:
                vals.append(v)
    if vals:
        results[cap] = sum(vals) / len(vals)
for cap, v in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {cap:<26} {v:.4f}  (n={sum(1 for f in FOLDS if os.path.exists(os.path.join(RESULT_DIR, f'mamba_flow_ar_tunecond_{cap}_s{SEED}_{f}_summary.json')))})")
if results:
    best = min(results, key=results.get)
    print(f"\nbest: {best}  (val_nll/wk = {results[best]:.4f})")
    print("  → 이 hp 로 BEST_SPECS['mlp'] 갱신 후 cond/mask 본실험 재학습.")
