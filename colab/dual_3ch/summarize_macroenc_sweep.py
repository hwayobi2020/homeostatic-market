"""macroenc-garch past_encoder sweep 결과 요약 표.

result/garch_flow_ar_macroenc_past{Mamba,Lstm,Transformer,Mlp}_s2026_F_gfc_summary.json
4개를 읽어 한 표로 출력.  skew/exkurt/lambda 가 summary 에 있으면 함께 출력하고,
없으면 'n/a' (옛 summary — eval 재실행 시 채워짐) 로 표시.

Usage (Colab): !python colab/dual_3ch/summarize_macroenc_sweep.py
"""
import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "result")

FOLD = "F_gfc"
SEED = 2026
PAST_ENCODERS = ["Mamba", "Lstm", "Transformer", "Mlp"]


def _g(d, *keys, default="n/a"):
    """nested get with fallback."""
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def _fmt(v, spec=".4f"):
    if isinstance(v, (int, float)):
        return format(v, spec)
    return str(v)


def main():
    rows = []
    for pe in PAST_ENCODERS:
        tag = f"macroenc_past{pe}_s{SEED}"
        fp = os.path.join(RESULT_DIR, f"garch_flow_ar_{tag}_{FOLD}_summary.json")
        if not os.path.exists(fp):
            print(f"[missing] {os.path.basename(fp)}")
            continue
        with open(fp) as f:
            s = json.load(f)
        te = s.get("test_eval", {})
        rows.append(dict(
            past_enc = pe,
            params   = s.get("n_params", "n/a"),
            val_nll  = s.get("best_val_nll_per_week", "n/a"),
            ep       = s.get("best_epoch", "n/a"),
            nll      = _g(te, "per_week_nll_z"),
            crps     = _g(te, "crps_pooled"),
            emd      = _g(te, "emd"),
            std_r    = _g(te, "std_ratio"),
            skew_a   = _g(te, "skew_actual"),
            skew_s   = _g(te, "skew_sim"),
            exk_a    = _g(te, "exkurt_actual"),
            exk_s    = _g(te, "exkurt_sim"),
            cov50    = _g(te, "coverage_50"),
            cov80    = _g(te, "coverage_80"),
            cov95    = _g(te, "coverage_95"),
            lam      = _g(te, "skewt_lambda"),
            cvar1_d  = _g(te, "cvar_1pct_diff"),
        ))

    if not rows:
        print("[FATAL] no summary json found in", RESULT_DIR)
        return

    has_skew = any(isinstance(r["skew_s"], (int, float)) for r in rows)

    print(f"\n=== macroenc-garch past_encoder sweep ({FOLD}, seed={SEED}) ===")
    print("과거 시퀀스 요약 부품 4종 비교 (capacity d=64, n=1 동일; 메인 인코더=MLP 고정)\n")

    hdr = (f"{'past_enc':<12} {'params':>8} {'val_nll':>8} {'ep':>4} "
           f"{'test_nll':>9} {'CRPS':>8} {'EMD':>9} {'std_r':>7} "
           f"{'skew_a/s':>16} {'exkurt_a/s':>18} "
           f"{'cov50':>6} {'cov80':>6} {'cov95':>6} {'baseλ':>8} {'cvar1Δ':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        skew_pair  = f"{_fmt(r['skew_a'], '+.3f')}/{_fmt(r['skew_s'], '+.3f')}"
        exk_pair   = f"{_fmt(r['exk_a'], '+.2f')}/{_fmt(r['exk_s'], '+.2f')}"
        print(f"{r['past_enc']:<12} {str(r['params']):>8} "
              f"{_fmt(r['val_nll'], '+.4f'):>8} {str(r['ep']):>4} "
              f"{_fmt(r['nll'], '+.4f'):>9} {_fmt(r['crps'], '.5f'):>8} "
              f"{_fmt(r['emd'], '.6f'):>9} {_fmt(r['std_r'], '.3f'):>7} "
              f"{skew_pair:>16} {exk_pair:>18} "
              f"{_fmt(r['cov50'], '.3f'):>6} {_fmt(r['cov80'], '.3f'):>6} "
              f"{_fmt(r['cov95'], '.3f'):>6} {_fmt(r['lam'], '+.4f'):>8} "
              f"{_fmt(r['cvar1_d'], '+.5f'):>9}")

    if not has_skew:
        print("\n[note] skew/exkurt/baseλ 가 옛 summary 에는 없음 (eval_metrics 에 "
              "방금 추가됨).  채우려면 summary 지우고 eval 재실행:")
        print("       !rm colab/dual_3ch/result/garch_flow_ar_macroenc_past*_"
              f"s{SEED}_{FOLD}_summary.json")
        print("       !python colab/dual_3ch/run_garch_macroenc.py")


if __name__ == "__main__":
    main()
