"""origin-level 격리: 금리가 horizon 동안 >=25bp 변할 때 vs 평탄할 때,
미래 금리경로 조건화가 도움인가?  (fold 혼합 없이 직접 판정)

학습 없음 — 저장된 MLP ckpt 를 forward 1회(no sampling) 해서 per-origin
teacher-forced NLL 만 뽑는다.
  ON  = abl_ch6_mlp_s2026   (미래 tbill unmask)
  OFF = ratemask_mlp_s2026  (미래 tbill mask)
정의: d = NLL_off - NLL_on.  d > 0 이면 ON(금리)이 더 좋음 = 금리 도움.
origin 을 horizon 내 3M T-bill 변화 |Δ| >= 25bp 로 stratify.

가설: 변동(>=25bp) origin 에선 d>0(금리 도움), 평탄 origin 에선 d<=0(금리 noise).

Usage (Colab):  !python colab/dual_3ch/analyze_rate_stratify_mlp.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_mamba_flow_ar as T            # noqa: E402
from train_mamba_flow_ar import (          # noqa: E402
    MambaFlowAR, cached_load_windows_seq, compute_valid_mask)
from best_specs import FOLDS as BASE_FOLDS, BEST_SPECS   # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
RESULT_DIR = os.path.join(HERE, "result")
CH6 = ["sp_return", "tbill_wr", "ads_lag", "sp_std_13w", "wti_wr", "sp_log_std_13w"]
PAST, FUT = 52, 13
SEED = 2026
BP_THRESH = 0.25   # 25bp = 0.25 연율 percentage point
FOLDS = list(BASE_FOLDS) + ["F_gfc"]   # 기존 3 fold + GFC 위기 fold


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


def build_mlp_model(meta):
    # meta 에 encoder_type/mlp_num_layers/dropout 가 없으므로 학습 spec(BEST_SPECS["mlp"])
    # 으로 재구성. ⚠️ dropout 은 반드시 학습값(0.2)과 같아야 함: MLPEncoder 는 dropout>0
    # 일 때만 nn.Dropout 을 net Sequential 에 끼우므로, dropout=0 으로 만들면 인덱스가
    # 통째로 밀려 state_dict key 가 어긋난다 (net.4 ↔ net.3, Linear ↔ LayerNorm).
    mlp = BEST_SPECS["mlp"]
    return MambaFlowAR(
        d_input=len(meta["cond_cols"]), d_model=meta["d_model"],
        n_flow_layers=meta["n_flow_layers"], n_flow_hidden=meta["n_flow_hidden"],
        n_flow_blocks=meta["n_flow_blocks"], n_flow_bins=meta["n_flow_bins"],
        flow_tail_bound=meta["flow_tail_bound"], dropout=mlp["dropout"],
        extra_context_dim=meta.get("extra_context_dim", 0),
        encoder_type="mlp", mlp_num_layers=mlp["mlp_num_layers"],
        direct_prev_return=False)


def per_origin_nll(tag, fold, mask_tbill, dev):
    p = os.path.join(RESULT_DIR, f"mamba_flow_ar_{tag}_{fold}_best.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    ck = torch.load(p, map_location=dev)
    meta = ck["meta"]
    set_cond_cols(meta["cond_cols"])
    T.MASK_FUTURE_TBILL = mask_tbill
    csv = os.path.join(FOLDS_DIR, f"{fold}_test.csv")
    Xte, Yte, _, _, _ = cached_load_windows_seq(
        csv, cond_stats=meta["cond_stats"], target_stats=meta["target_stats"])
    m = build_mlp_model(meta).to(dev)
    m.load_state_dict(ck["model_state"])
    m.eval()
    with torch.no_grad():
        lp = m.log_prob(Xte.to(dev), Yte.to(dev))    # (n_valid,) sum over weeks
    return (-lp.cpu().numpy() / FUT)                 # per-origin per-week NLL


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for i, fold in enumerate(FOLDS):
        on_ck = os.path.join(RESULT_DIR, f"mamba_flow_ar_abl_ch6_mlp_s{SEED}_{fold}_best.pt")
        if not os.path.exists(on_ck):
            print(f"[skip {fold}] missing ON ckpt: {os.path.basename(on_ck)}")
            continue
        meta = torch.load(on_ck, map_location="cpu")["meta"]
        set_cond_cols(meta["cond_cols"])
        vmask, _, _ = compute_valid_mask(
            os.path.join(FOLDS_DIR, f"{fold}_test.csv"), meta["cond_stats"])
        vt = np.where(vmask)[0]
        try:
            nll_on = per_origin_nll(f"abl_ch6_mlp_s{SEED}", fold, False, dev)
            nll_off = per_origin_nll(f"ratemask_mlp_s{SEED}", fold, True, dev)
        except FileNotFoundError as e:
            print(f"[skip {fold}] missing ckpt: {e}")
            continue
        if len(nll_on) != len(vt) or len(nll_off) != len(vt):
            print(f"[WARN {fold}] length mismatch on={len(nll_on)} off={len(nll_off)} vt={len(vt)}")
        df = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_test.csv"))
        if i == 0:
            print("[csv cols]", list(df.columns))
        # tbill_wr = DTB3 연율%를 주간복리수익률로 변환한 레벨 (extend_to_1971 L305).
        # 25bp(연율 %p) 비교 위해 연율 %p 로 환산: ((1+wr)^52 - 1)*100  (0.000938 → 5.00).
        rate_w = pd.to_numeric(df["tbill_wr"], errors="coerce").values
        rate_pp = ((1.0 + rate_w) ** 52 - 1.0) * 100.0
        chg = np.array([abs(rate_pp[t + PAST + FUT - 1] - rate_pp[t + PAST - 1])
                        for t in vt])
        date = (df["date"].values[vt + PAST - 1] if "date" in df.columns
                else np.array([f"{fold}_{t}" for t in vt]))
        print(f"[{fold}] rate=tbill_wr→연율%p  origins={len(vt)}  "
              f"|Δrate|pp min/med/max = {np.nanmin(chg):.3f}/{np.nanmedian(chg):.3f}"
              f"/{np.nanmax(chg):.3f}  (>=25bp: {int((chg >= BP_THRESH).sum())})")
        for j in range(len(vt)):
            rows.append(dict(fold=fold, date=str(date[j]), chg_pp=float(chg[j]),
                             nll_on=float(nll_on[j]), nll_off=float(nll_off[j])))

    if not rows:
        print("[FATAL] no rows (ckpt 누락?)")
        return
    R = pd.DataFrame(rows)
    R["changing"] = R["chg_pp"] >= BP_THRESH
    R["d"] = R["nll_off"] - R["nll_on"]   # >0 => ON(rate) better = 금리 도움

    print("\n" + "=" * 82)
    print("금리 변동(≥25bp) vs 평탄 — per-origin teacher-forced NLL")
    print("  d = NLL_off − NLL_on   ( >0 = 금리 ON 이 더 좋음 = 금리 도움 )")
    print("=" * 82)
    for lab, sub in [("평탄 (<25bp)", R[~R.changing]), ("변동 (≥25bp)", R[R.changing])]:
        if len(sub) == 0:
            print(f"{lab:<14} n=0")
            continue
        print(f"{lab:<14} n={len(sub):>4}  NLL_on={sub.nll_on.mean():+.4f}  "
              f"NLL_off={sub.nll_off.mean():+.4f}  d={sub.d.mean():+.4f}")
    print("\n[fold × regime]  (d>0 = 금리 도움)")
    for fold in FOLDS:
        for lab, mask in [("flat", ~R.changing), ("chg ", R.changing)]:
            sub = R[(R.fold == fold) & mask]
            if len(sub):
                print(f"  {fold:<16}{lab} n={len(sub):>4}  d={sub.d.mean():+.4f}")
    out = os.path.join(RESULT_DIR, "rate_stratify_mlp.csv")
    R.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
