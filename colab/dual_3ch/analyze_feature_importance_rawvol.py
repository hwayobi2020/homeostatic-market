"""§4.3.2 변수 중요도 — Permutation Importance (teacher-forced NLL 악화폭).

각 조건 채널을 origin 간 셔플(관계 파괴) → teacher-forced NLL(per-week) 증가폭 = 중요도.
샘플링 없이 model.log_prob 만 → inference 중 가장 쌈.  LOCKED MLP.

채널: tbill_wr, ads_lag, wti_wr, metab_13w (입력+미래unmask 전체 섞음) +
      extra context sp_std_13w, sp_skew_13w.  (sp_return=목표라 제외)
ΔNLL/week > 0 클수록 중요.  n_perm=5 평균(셔플 노이즈 완화).

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/analyze_feature_importance_rawvol.py
"""
import json
import os
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rawvol_helpers import patch_rawvol                              # noqa: E402
patch_rawvol()
import train_garch_flow as T                                        # noqa: E402
from train_garch_flow import (                                      # noqa: E402
    MambaFlowAR, cached_load_windows_seq, load_extra_context,
    garch_preprocess_fold, PAST_LEN, FUTURE_LEN,
)

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(os.path.normpath(os.path.join(HERE, "..", "..")), "data", "folds_v33_vix_expanding")
CACHE_DIR = os.path.join(RESULT_DIR, "feat_importance_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

ENC_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_13w"]
DC_COLS_LIST = ["sp_std_13w", "sp_skew_13w"]
PERMUTE_ENC = ["tbill_wr", "ads_lag", "wti_wr", "metab_13w"]         # sp_return 제외(목표)
LK = dict(d_model=128, mlp_layers=4, flow_layers=4, flow_hidden=128, pd=64, dropout=0.2)
TAG_PREFIX = f"rvP2mainMlp_pd{LK['pd']}_fl{LK['flow_layers']}_fh{LK['flow_hidden']}"
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEEDS = [2026, 2027, 2028]
N_PERM = 5


def set_cond_cols(cols):
    cols = list(cols)
    T.COND_COLS = cols; T.N_CHANNELS = len(cols)
    T.SP_CH = cols.index("sp_return"); T.TBILL_CH = cols.index("tbill_wr")
    T.MACRO_CH = [c for c in range(len(cols)) if c not in (T.SP_CH, T.TBILL_CH)]
    try:
        T._DATA_CACHE.clear()
    except Exception:
        pass


set_cond_cols(ENC_COLS)
T.MASK_FUTURE_TBILL = False
T.FUTURE_UNMASK_MACRO_COLS = ["metab_13w"]
T.ENCODER_MASK_SP = False


def rebuild_model(ckpt, device):
    m = MambaFlowAR(
        d_input=len(ENC_COLS), d_model=LK["d_model"],
        n_flow_layers=LK["flow_layers"], n_flow_hidden=LK["flow_hidden"],
        dropout=LK["dropout"], extra_context_dim=len(DC_COLS_LIST),
        encoder_type="mlp", mlp_num_layers=LK["mlp_layers"],
        direct_prev_return=True, use_past_summary=True,
        past_encoder_type="mlp", past_summary_dim=LK["pd"],
    ).to(device)
    m.load_state_dict(ckpt["model_state"]); m.eval()
    return m


@torch.no_grad()
def run_fold_seed(fold, seed, device):
    bp = os.path.join(RESULT_DIR, f"garch_flow_ar_{TAG_PREFIX}_s{seed}_{fold}_best.pt")
    if not os.path.exists(bp):
        print(f"[missing] {os.path.basename(bp)}"); return None
    ckpt = torch.load(bp, map_location=device); meta = ckpt["meta"]
    cond_stats = meta["cond_stats"]; target_stats = meta["target_stats"]
    extra_stats = meta.get("extra_stats")
    model = rebuild_model(ckpt, device)

    test_csv = garch_preprocess_fold(FOLDS_DIR, fold, RESULT_DIR)["test"]
    Xte, Yte, _, _, _ = cached_load_windows_seq(test_csv, cond_stats=cond_stats, target_stats=target_stats)
    Xte_dev = Xte.to(device); Yte_dev = Yte.to(device)
    Xte_extra, _, n_ex, _ = load_extra_context(test_csv, DC_COLS_LIST, extra_stats=extra_stats)
    if n_ex != Xte.shape[0]:
        sys.exit(f"[FATAL] extra {n_ex} != {Xte.shape[0]} ({fold},{seed})")
    Xe_dev = Xte_extra.to(device)
    n = Xte.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    def nll(X, E):
        return float((-model.log_prob(X, Yte_dev, extra_context=E)).mean().item()) / FUTURE_LEN

    base = nll(Xte_dev, Xe_dev)
    imp = {}
    # ENC 채널 셔플 (입력+미래 전 위치)
    for name in PERMUTE_ENC:
        ci = ENC_COLS.index(name)
        deltas = []
        for _ in range(N_PERM):
            perm = torch.randperm(n, generator=g)
            Xp = Xte_dev.clone()
            Xp[:, :, ci] = Xte_dev[perm][:, :, ci]
            deltas.append(nll(Xp, Xe_dev) - base)
        imp[name] = float(np.mean(deltas))
    # extra context 채널 셔플
    for j, name in enumerate(DC_COLS_LIST):
        deltas = []
        for _ in range(N_PERM):
            perm = torch.randperm(n, generator=g)
            Ep = Xe_dev.clone()
            Ep[:, j] = Xe_dev[perm][:, j]
            deltas.append(nll(Xte_dev, Ep) - base)
        imp[name] = float(np.mean(deltas))
    return dict(n_origin=int(n), base_nll_wk=base, importance=imp)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 92)
    print(f"# §4.3.2 Permutation Importance — ΔNLL/week (LOCKED {TAG_PREFIX}, n_perm={N_PERM})")
    print("#  ΔNLL/wk > 0 클수록 중요 (셔플 시 적합도 악화).  음수 ≈ 기여 없음")
    print("#" * 92)

    all_feats = PERMUTE_ENC + DC_COLS_LIST
    for fold in FOLDS:
        for seed in SEEDS:
            c = os.path.join(CACHE_DIR, f"{fold}_s{seed}.json")
            if os.path.exists(c):
                print(f"[skip] {fold} s{seed}"); continue
            r = run_fold_seed(fold, seed, device)
            if r is None:
                continue
            json.dump(r, open(c, "w"), indent=2)
            print(f"[done] {fold} s{seed}  base NLL/wk={r['base_nll_wk']:+.4f}")

    print("\n" + "=" * 92)
    print("[집계] Permutation Importance ΔNLL/week — 4 fold × 3 seed 평균±std")
    print(f"  {'feature':>14}" + "".join(f"{f:>14}" for f in all_feats))
    for fold in FOLDS:
        rs = [json.load(open(os.path.join(CACHE_DIR, f"{fold}_s{s}.json")))
              for s in SEEDS if os.path.exists(os.path.join(CACHE_DIR, f"{fold}_s{s}.json"))]
        if not rs:
            print(f"  {fold:>14}  (결과 없음)"); continue
        row = f"  {fold:>14}"
        for f in all_feats:
            vals = [r["importance"][f] for r in rs]
            row += f"  {np.mean(vals):+.4f}±{np.std(vals):.3f}"
        print(row)
    # 전체 pooled 순위
    pooled = {f: [] for f in all_feats}
    for fold in FOLDS:
        for s in SEEDS:
            p = os.path.join(CACHE_DIR, f"{fold}_s{s}.json")
            if os.path.exists(p):
                imp = json.load(open(p))["importance"]
                for f in all_feats:
                    pooled[f].append(imp[f])
    rank = sorted(all_feats, key=lambda f: -np.mean(pooled[f]))
    print("\n  [중요도 순위 (전체 평균 ΔNLL/wk)]")
    for f in rank:
        print(f"    {f:>14}: {np.mean(pooled[f]):+.4f}")
    print("\n[판정] ΔNLL/wk 큰 순 = 적합도 기여 큰 변수.  tbill·metab(thesis 핵심)이 상위면 정합.")


if __name__ == "__main__":
    main()
