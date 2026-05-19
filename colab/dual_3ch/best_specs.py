"""Best hyperparameter spec for each encoder (Clean Run Phase 1/2 결과 확정).

Selection criterion : 3-fold avg val NLL/week (Phase 1, seed=2026).
Verification        : Phase 2 multi-seed (5 seed x 3 fold = 15 run pooled).
Source              : colab/dual_3ch/run_phase1_clean.py + run_phase2_clean.py.

Window / Channel / Mamba-SSM / Flow 내부 고정값은 train_mamba_flow_ar.py
글로벌 상수와 동일.  본 파일은 *encoder 별로 sweep 한 변수만* 확정한다.

Usage :
    from best_specs import BEST_SPECS, get_best_spec
    spec = get_best_spec("mamba", fold="F_long_A", seed=2026,
                         tag_suffix="_baseline_compare")
    main_worker(spec)
"""

# --------------------------------------------------------------------------- #
# Flow head preset  (run_phase1_clean.py:41-44 와 동일)
# --------------------------------------------------------------------------- #
FLOW_HEAVY = dict(n_flow_layers=6, n_flow_hidden=64, weight_decay=0.5)
FLOW_LIGHT = dict(n_flow_layers=2, n_flow_hidden=32, weight_decay=0.1)


# --------------------------------------------------------------------------- #
# Encoder 별 best capacity spec  (Phase 1 sweep top1, val 기준)
# --------------------------------------------------------------------------- #
# 주의 : key 이름 `n_mamba_layers` 는 LSTM/Transformer 에서도 그대로 reuse 된다.
#   - LSTM        : LSTM 의 num_layers
#   - Transformer : Transformer block 수
#   - Mamba       : Mamba block 수
# train_mamba_flow_ar.py 와 run_phase1_clean.py 의 호출 관례와 일치.
BEST_SPECS = {
    "lstm": dict(
        encoder_type="lstm",
        d_model=128,
        n_mamba_layers=2,        # LSTM num_layers
        dropout=0.1,
        **FLOW_HEAVY,
    ),
    "mlp": dict(
        encoder_type="mlp",
        d_model=128,
        mlp_num_layers=4,
        dropout=0.2,
        **FLOW_HEAVY,
    ),
    "transformer": dict(
        encoder_type="transformer",
        d_model=128,
        transformer_n_heads=8,
        n_mamba_layers=3,        # Transformer block 수
        dropout=0.2,
        **FLOW_LIGHT,
    ),
    "mamba": dict(
        encoder_type="mamba",
        d_model=128,
        n_mamba_layers=3,
        dropout=0.2,
        **FLOW_HEAVY,
    ),
}


# --------------------------------------------------------------------------- #
# Reference metrics (paper table / sanity check 용)
# --------------------------------------------------------------------------- #
# Phase 1 (seed=2026 단일) — 3-fold avg val NLL/week (낮을수록 좋음)
PHASE1_BEST_VAL_PER_WEEK = {
    "lstm":        +1.3629,    # hd128_nl2_dr0.1 / Heavy
    "mlp":         +1.3496,    # hd128_nl4_dr0.2 / Heavy
    "transformer": +1.3872,    # dm128_nh8_nl3_dr0.2 / Light
    "mamba":       +1.3865,    # dm128_nl3_dr0.2 / Heavy
}

# Phase 2 (5 seed x 3 fold = 15 run pooled) — test NLL/week (z-score units)
PHASE2_TEST_PER_WEEK_Z = {
    # (mean, std)
    "lstm":        (+1.3220, 0.1096),
    "mlp":         (+1.2919, 0.1140),
    "transformer": (+1.3715, 0.1313),
    "mamba":       (+1.3385, 0.1159),
}

# Phase 2 추가 scenario metrics (15 run pooled mean)
PHASE2_SCENARIO_METRICS = {
    # encoder : dict(emd, cvar_5pct_diff, coverage_80)
    "lstm":        dict(emd=0.00363, cvar_5pct_diff=+0.00967, coverage_80=0.730),
    "mlp":         dict(emd=0.00397, cvar_5pct_diff=+0.00135, coverage_80=0.798),
    "transformer": dict(emd=0.00463, cvar_5pct_diff=+0.00822, coverage_80=0.725),
    "mamba":       dict(emd=0.00426, cvar_5pct_diff=+0.00067, coverage_80=0.815),
}


# --------------------------------------------------------------------------- #
# Fold / Seed 상수 (Clean Run 와 동일)
# --------------------------------------------------------------------------- #
FOLDS = ["F_long_A", "F_long_B_origin", "F_long"]
PHASE1_SEED = 2026
PHASE2_SEEDS = [2026, 2027, 2028, 2029, 2030]

# Paper main encoder — Tail-Risk thesis 정합 (CVaR5d / cov80 best)
MAIN_ENCODER = "mamba"


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def get_best_spec(encoder, fold=None, seed=None, tag_suffix=""):
    """Return a fresh spec dict ready for main_worker().

    Args:
      encoder    : one of {"lstm", "mlp", "transformer", "mamba"}.
      fold       : one of FOLDS (optional).
      seed       : integer seed (optional).
      tag_suffix : extra tag suffix.  Final tag form =
                   "best_{encoder}{tag_suffix}_s{seed}" (seed 있을 때).

    Returns:
      dict suitable for main_worker(spec).  Includes all sweep variables
      (encoder_type, d_model, n_*_layers, dropout, flow params, weight_decay)
      and optionally {fold, seed, tag}.
    """
    if encoder not in BEST_SPECS:
        raise ValueError(
            f"unknown encoder {encoder!r}; choose from {list(BEST_SPECS)}"
        )
    spec = dict(BEST_SPECS[encoder])     # shallow copy (값 모두 immutable)
    if fold is not None:
        spec["fold"] = fold
    if seed is not None:
        spec["seed"] = int(seed)
        spec["tag"] = f"best_{encoder}{tag_suffix}_s{seed}"
    else:
        spec["tag"] = f"best_{encoder}{tag_suffix}"
    return spec


def all_encoders():
    """Convenience : iterate encoder names in canonical order."""
    return list(BEST_SPECS.keys())


# --------------------------------------------------------------------------- #
# Sanity print (`python best_specs.py` 로 실행 시 요약 출력)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Encoder-wise BEST spec (Clean Run Phase 1/2 확정)")
    print("=" * 88)
    for enc in all_encoders():
        spec = BEST_SPECS[enc]
        val = PHASE1_BEST_VAL_PER_WEEK[enc]
        tm, ts = PHASE2_TEST_PER_WEEK_Z[enc]
        sm = PHASE2_SCENARIO_METRICS[enc]
        print(f"\n[{enc}]")
        print(f"  spec                : {spec}")
        print(f"  Phase 1 val/week    : {val:+.4f}")
        print(f"  Phase 2 test/week_z : {tm:+.4f} +/- {ts:.4f}")
        print(f"  EMD / CVaR5d / cov80: {sm['emd']:.5f} / "
              f"{sm['cvar_5pct_diff']:+.5f} / {sm['coverage_80']:.3f}")
