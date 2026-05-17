"""Mamba encoder Phase 1 sweep -- multi-seed (5 seeds x 3 folds x 8 specs).

Mirrors the LSTM / MLP / Transformer Phase 1 sweep structure but with 5 seeds
per (spec, fold) so that seed dispersion is measured at sweep time.

Spec grid (8 specs):
    d_model          in {64, 128}
    n_mamba_layers   in {2, 3}
    dropout          in {0.1, 0.2}
Seeds (5):           {42, 123, 777, 0, 99}        -- same as Stage 1 multi-seed.
Folds (3):           F_long_A / F_long_B_origin / F_long.
Total = 8 specs * 5 seeds * 3 folds = 120 runs.

Output:
    result/mamba_flow_ar_sweep_mamba_dm{D}_nl{N}_dr{R}_s{S}_{fold}_summary.json
The 'sweep_mamba_dm... _s{seed}' tag matches the existing sweep-result
parsing regex used for LSTM/MLP/Transformer sweeps, with '_s{seed}' suffix so
the 5 seeds do not overwrite one another at the same (spec, fold).

Re-entrancy:
    Already-completed (spec, fold, seed) runs (detected by existing
    *_summary.json) are skipped, so the script can be safely re-run after a
    Colab session disconnect.

Iteration order:
    fold outer, then spec, then seed -- _DATA_CACHE in train_mamba_flow_ar.py
    keeps the per-fold csv in memory, so spec/seed switches inside one fold
    reuse the cache.

Usage (Colab):
    !python colab/dual_3ch/run_sweep_mamba.py
"""
import itertools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_mamba_flow_ar import main_worker  # noqa: E402

FOLDS = ["F_long_A", "F_long_B_origin", "F_long"]
SEEDS = [42, 123, 777, 0, 99]
GRID = list(itertools.product(
    [64, 128],   # d_model
    [2, 3],      # n_mamba_layers
    [0.1, 0.2],  # dropout
))
RESULT_DIR = os.path.join(HERE, "result")

n_total = len(GRID) * len(SEEDS) * len(FOLDS)
print(f"[sweep] mamba  |  {len(GRID)} specs x {len(SEEDS)} seeds x "
      f"{len(FOLDS)} folds = {n_total} runs")
print(f"[sweep] result dir   : {RESULT_DIR}")
print(f"[sweep] grid         : d_model x n_mamba_layers x dropout")
for d_m, n_l, dr in GRID:
    print(f"                       dm={d_m:3d}  nl={n_l}  dr={dr}")

done = 0
skipped = 0
failed = 0
t0 = time.time()

for fold in FOLDS:
    for d_model, n_mamba_layers, dropout in GRID:
        for seed in SEEDS:
            tag = (f"sweep_mamba_dm{d_model}_nl{n_mamba_layers}"
                   f"_dr{dropout}_s{seed}")
            summary_path = os.path.join(
                RESULT_DIR,
                f"mamba_flow_ar_{tag}_{fold}_summary.json",
            )
            if os.path.exists(summary_path):
                print(f"[skip {done+skipped+failed+1:3d}/{n_total}] exists: "
                      f"{os.path.basename(summary_path)}")
                skipped += 1
                continue
            spec = dict(
                fold=fold,
                tag=tag,
                seed=seed,
                d_model=d_model,
                n_mamba_layers=n_mamba_layers,
                dropout=dropout,
                encoder_type="mamba",
            )
            idx = done + skipped + failed + 1
            print(f"\n[run  {idx:3d}/{n_total}] tag={tag}  fold={fold}")
            try:
                main_worker(spec)
                done += 1
            except Exception as e:
                print(f"[FAIL {idx:3d}/{n_total}] {tag} {fold}: {e!r}")
                failed += 1

dt = time.time() - t0
print(f"\n[sweep done] done={done}  skipped={skipped}  failed={failed}  "
      f"elapsed={dt/60:.1f} min")
