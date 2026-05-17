"""Phase 1 (Clean Run) -- Unified encoder x Flow head sweep, single-seed.

[Master design -- Gemini Clean Run pipeline]

  Encoders (28 specs total):
      LSTM         (4): hd in {64,128}            x nl in {2,3} x dr in {0.1}
      MLP          (8): hd in {128,256}           x nl in {3,4} x dr in {0.1,0.2}
      Transformer  (8): (dm,nh) in {(64,4),(128,8)} x nl in {2,3} x dr in {0.1,0.2}
      Mamba        (8): dm in {64,128}            x nl in {2,3} x dr in {0.1,0.2}

  Flow heads (2):
      Light : n_flow_layers=2, n_flow_hidden=32, weight_decay=0.1
      Heavy : n_flow_layers=6, n_flow_hidden=64, weight_decay=0.5

  Seed   : 2026 (single).
  Folds  : F_long_A / F_long_B_origin / F_long.
  Total  : 28 specs x 2 flows x 3 folds = 168 runs.

  Globally fixed (= train_mamba_flow_ar.py defaults):
      lr=1e-4, batch=32, max_epoch=60, patience=30,
      n_flow_blocks=2, n_flow_bins=16, tail_bound=10.0.

  Tag : sweep_{enc}_{capspec}_dr{R}_flow{Heavy|Light}_s2026
        e.g.  sweep_mamba_dm128_nl2_dr0.1_flowHeavy_s2026.

  Re-entrant : (spec, fold) pairs whose summary.json already exists are skipped.

Usage (Colab) :
    !python colab/dual_3ch/run_phase1_clean.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_mamba_flow_ar import main_worker  # noqa: E402

SEED = 2026
FOLDS = ["F_long_A", "F_long_B_origin", "F_long"]
FLOWS = {
    "Light": dict(n_flow_layers=2, n_flow_hidden=32, weight_decay=0.1),
    "Heavy": dict(n_flow_layers=6, n_flow_hidden=64, weight_decay=0.5),
}
RESULT_DIR = os.path.join(HERE, "result")


def _encoder_specs():
    """List of (enc_name, cap_str, encoder_kwargs, dropout)."""
    specs = []
    # LSTM 4 : hd x nl x dr={0.1}
    for hd in (64, 128):
        for nl in (2, 3):
            for dr in (0.1,):
                specs.append((
                    "lstm",
                    f"hd{hd}_nl{nl}",
                    dict(encoder_type="lstm",
                         d_model=hd, n_mamba_layers=nl),
                    dr,
                ))
    # MLP 8 : hd x nl x dr={0.1, 0.2}
    for hd in (128, 256):
        for nl in (3, 4):
            for dr in (0.1, 0.2):
                specs.append((
                    "mlp",
                    f"hd{hd}_nl{nl}",
                    dict(encoder_type="mlp",
                         d_model=hd, mlp_num_layers=nl),
                    dr,
                ))
    # Transformer 8 : (dm,nh) paired x nl x dr={0.1, 0.2}
    for dm, nh in ((64, 4), (128, 8)):
        for nl in (2, 3):
            for dr in (0.1, 0.2):
                specs.append((
                    "transformer",
                    f"dm{dm}_nh{nh}_nl{nl}",
                    dict(encoder_type="transformer",
                         d_model=dm, transformer_n_heads=nh,
                         n_mamba_layers=nl),
                    dr,
                ))
    # Mamba 8 : dm x nl x dr={0.1, 0.2}
    for dm in (64, 128):
        for nl in (2, 3):
            for dr in (0.1, 0.2):
                specs.append((
                    "mamba",
                    f"dm{dm}_nl{nl}",
                    dict(encoder_type="mamba",
                         d_model=dm, n_mamba_layers=nl),
                    dr,
                ))
    return specs


SPECS = _encoder_specs()
assert len(SPECS) == 28, f"expected 28 encoder specs, got {len(SPECS)}"

N_TOTAL = len(SPECS) * len(FLOWS) * len(FOLDS)
print(f"[Phase 1 clean] {len(SPECS)} encoder specs x {len(FLOWS)} flow heads "
      f"x {len(FOLDS)} folds = {N_TOTAL} runs")
print(f"[Phase 1 clean] result dir : {RESULT_DIR}")

done = 0
skipped = 0
failed = 0
t0 = time.time()

# fold outer (csv cache via _DATA_CACHE), encoder/flow inner.
for fold in FOLDS:
    for enc, cap, enc_kw, dropout in SPECS:
        for flow_name, flow_kw in FLOWS.items():
            tag = (f"sweep_{enc}_{cap}_dr{dropout}_flow{flow_name}_s{SEED}")
            summary_path = os.path.join(
                RESULT_DIR,
                f"mamba_flow_ar_{tag}_{fold}_summary.json",
            )
            idx = done + skipped + failed + 1
            if os.path.exists(summary_path):
                print(f"[skip {idx:3d}/{N_TOTAL}] "
                      f"{os.path.basename(summary_path)}")
                skipped += 1
                continue
            # Build spec dict.  Dict takes precedence over train script
            # defaults (main_worker merges {**defaults, **args}); encoder
            # kwargs and flow kwargs do not collide with each other.
            spec = dict(
                fold=fold,
                tag=tag,
                seed=SEED,
                dropout=dropout,
                **enc_kw,
                **flow_kw,
            )
            print(f"\n[run  {idx:3d}/{N_TOTAL}] {tag}  fold={fold}")
            try:
                main_worker(spec)
                done += 1
            except Exception as e:
                print(f"[FAIL {idx:3d}/{N_TOTAL}] {tag} {fold}: {e!r}")
                failed += 1

dt = time.time() - t0
print(f"\n[Phase 1 done] done={done} skipped={skipped} failed={failed}  "
      f"elapsed={dt / 60:.1f} min")
