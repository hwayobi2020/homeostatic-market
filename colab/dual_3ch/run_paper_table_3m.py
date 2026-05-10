"""Run paper table 3M — 12 variants × 5 seeds = 60 runs, then aggregate.

Paper table mapping (user's xlsx 표_3 시트):
  #1  base                          → train_vol_pilot_3m     v1   (base)
  #2  base + excess_liq (cond)      → train_vol_pilot_3m     v13  (base_el_only)
  #3  base bondpp                   → train_vol_pilot_3m     v2   (base_bp)
  #4  base bondsum                  → train_vol_pilot_3m     v10  (base_bondsum)
  #5  base stockpp                  → train_vol_pilot_3m     v3   (base_sp)
  #6  base stocksum                 → train_vol_pilot_3m     v11  (base_stocksum)
  #7  base bondpp stockpp           → train_vol_pilot_3m     v4   (base_bp_sp)
  #8  mtl bondpp                    → train_vol_pilot_mtl_3m v5   (mtl_bp)
  #9  mtl stockpp                   → train_vol_pilot_mtl_3m v6   (mtl_sp)
  #10 mtl bondpp stockpp            → train_vol_pilot_mtl_3m v7   (mtl_bp_sp)
  #11 excess_liq mtl(base+bp+sp)    → train_vol_pilot_mtl_3m v8   (mtl_el_bp_sp_cond)
  #12 excess_liq mtl(base)          → train_vol_pilot_mtl_3m v9   (mtl_el)

Behavior:
  - Skip if {tag}_best.pt + {tag}_summary.json already exist (resumable).
  - After all runs complete, aggregate per variant (5 seed median ± std)
    and print 12-row table to stdout + save CSV.

Usage (Colab):
  !cd /content/drive/MyDrive/.../homeostatic-market \
   && python data/build_pilot_split.py \
   && python colab/dual_3ch/run_paper_table_3m.py
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from train_vol_pilot_3m import (
    run as run_scalar,
    build_spec as build_spec_scalar,
)
from train_vol_pilot_mtl_3m import (
    run as run_mtl,
    build_spec as build_spec_mtl,
    LAMBDA_AUX,
)


# (paper_row, label, kind, code_variant_id)
PAPER_TABLE = [
    (1,  "base",                          "scalar", 1),
    (2,  "base + excess_liq (cond)",      "scalar", 13),
    (3,  "base bondpp",                   "scalar", 2),
    (4,  "base bondsum",                  "scalar", 10),
    (5,  "base stockpp",                  "scalar", 3),
    (6,  "base stocksum",                 "scalar", 11),
    (7,  "base bondpp stockpp",           "scalar", 4),
    (8,  "mtl bondpp",                    "mtl",    5),
    (9,  "mtl stockpp",                   "mtl",    6),
    (10, "mtl bondpp stockpp",            "mtl",    7),
    (11, "excess_liq mtl(base+bp+sp)",    "mtl",    8),
    (12, "excess_liq mtl(base)",          "mtl",    9),
]

DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def ensure_pilot_split(repo_root):
    split_dir = os.path.join(repo_root, "data", "pilot_split")
    train_csv = os.path.join(split_dir, "train.csv")
    val_csv   = os.path.join(split_dir, "val.csv")
    test_csv  = os.path.join(split_dir, "test.csv")
    if all(os.path.exists(p) for p in [train_csv, val_csv, test_csv]):
        print(f"[pilot_split] OK: {split_dir}")
        return train_csv, val_csv, test_csv

    print(f"[pilot_split] missing under {split_dir} — auto-building...")
    build_script = os.path.join(repo_root, "data", "build_pilot_split.py")
    if not os.path.exists(build_script):
        sys.exit(f"[FATAL] build_pilot_split.py not found at {build_script}")
    r = subprocess.run([sys.executable, build_script], cwd=repo_root)
    if r.returncode != 0:
        sys.exit(f"[FATAL] build_pilot_split.py failed (exit {r.returncode})")
    if not all(os.path.exists(p) for p in [train_csv, val_csv, test_csv]):
        sys.exit(f"[FATAL] still missing after build: {split_dir}")
    print(f"[pilot_split] built: {split_dir}")
    return train_csv, val_csv, test_csv


def summary_filename(kind, code_v, name, fold_tag, seed):
    if kind == "scalar":
        return f"vol_pilot_3m_psel_v{code_v}_{name}_{fold_tag}_seed{seed}_summary.json"
    elif kind == "mtl":
        return f"vol_pilot_mtl_3m_psel_v{code_v}_{name}_{fold_tag}_seed{seed}_summary.json"
    raise ValueError(kind)


def load_summary(out_dir, kind, code_v, name, fold_tag, seed):
    fn = summary_filename(kind, code_v, name, fold_tag, seed)
    p = os.path.join(out_dir, fn)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def aggregate_table(out_dir, seeds, fold_tag="pilot"):
    rows = []
    for paper_row, label, kind, code_v in PAPER_TABLE:
        spec = build_spec_scalar(code_v, with_vix=False) if kind == "scalar" else build_spec_mtl(code_v)
        name = spec["name"]

        per_pearson  = []
        per_r2       = []
        per_spearman = []
        per_best_ep  = []
        ckpt_n = 0
        for seed in seeds:
            s = load_summary(out_dir, kind, code_v, name, fold_tag, seed)
            if s is None:
                continue
            ckpt_n += 1
            per_pearson.append(s.get("test_pearson_std"))
            per_r2.append(s.get("test_r2"))
            per_spearman.append(s.get("test_spearman_std"))
            per_best_ep.append(s.get("best_epoch"))

        def med_std(arr):
            arr_clean = [x for x in arr if (x is not None) and np.isfinite(x)]
            if len(arr_clean) == 0:
                return (float("nan"), float("nan"))
            a = np.array(arr_clean, dtype=np.float64)
            med = float(np.median(a))
            std = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
            return (med, std)

        med_p,  std_p  = med_std(per_pearson)
        med_r2, std_r2 = med_std(per_r2)
        med_sp, std_sp = med_std(per_spearman)
        med_be, _      = med_std(per_best_ep)

        rows.append(dict(
            paper_row=paper_row,
            label=label,
            kind=kind,
            code_variant=code_v,
            spec_name=name,
            n_seeds=ckpt_n,
            pearson_med=med_p, pearson_std=std_p,
            r2_med=med_r2,     r2_std=std_r2,
            spearman_med=med_sp, spearman_std=std_sp,
            best_ep_med=med_be,
        ))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "paper_table_3m_median.csv")
    df.to_csv(csv_path, index=False)
    return df, csv_path


def print_table(df):
    sep = "=" * 130
    print(f"\n{sep}")
    print(f"PAPER TABLE 3M — 12 variants × seeds (median ± std on test set)")
    print(f"{sep}")
    print(f"{'#':>3s} {'label':36s} {'kind':6s} {'cv':>3s} {'n':>2s}  "
          f"{'Pearson(med ± std)':>22s}  {'R²(med ± std)':>22s}  {'Spear(med ± std)':>22s}  {'best_ep_med':>11s}")
    print("-" * 130)
    best_p = df["pearson_med"].max() if df["pearson_med"].notna().any() else float("nan")
    for _, r in df.iterrows():
        marker = " ★" if (np.isfinite(r["pearson_med"]) and r["pearson_med"] == best_p) else "  "
        print(f"{r['paper_row']:>3d} {r['label'][:36]:36s} {r['kind']:6s} {r['code_variant']:>3d} {r['n_seeds']:>2d}  "
              f"{r['pearson_med']:+7.3f} ± {r['pearson_std']:5.3f}{marker:>3s}      "
              f"{r['r2_med']:+7.3f} ± {r['r2_std']:5.3f}      "
              f"{r['spearman_med']:+7.3f} ± {r['spearman_std']:5.3f}      "
              f"{r['best_ep_med']:>11.1f}")
    print(f"{sep}")
    print(f"  ★ = best Pearson among completed variants")


def main():
    ap = argparse.ArgumentParser(
        description="Run paper table 3M (12 variants × 5 seeds) + aggregate")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience",   type=int, default=30)
    ap.add_argument("--batch",      type=int, default=32)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--lambda-aux", type=float, default=LAMBDA_AUX)
    ap.add_argument("--rows", nargs="+", type=int, default=None,
                    help="Specific paper rows to run (default: all 12)")
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip training, just aggregate existing summary.json files")
    args = ap.parse_args()

    fold_tag = "pilot"
    train_csv, val_csv, test_csv = ensure_pilot_split(ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    rows_to_run = PAPER_TABLE if args.rows is None else [r for r in PAPER_TABLE if r[0] in args.rows]
    n_total = len(rows_to_run) * len(args.seeds)

    if not args.aggregate_only:
        print(f"\n{'#'*72}")
        print(f"# PAPER TABLE 3M — {len(rows_to_run)} rows × {len(args.seeds)} seeds = {n_total} runs")
        print(f"# Output dir : {args.out_dir}")
        print(f"# Skip-if-exists enabled (resumable).  fold_tag={fold_tag}")
        print(f"# Seeds      : {args.seeds}")
        print(f"{'#'*72}\n")

        t0 = time.time()
        for ri, (paper_row, label, kind, code_v) in enumerate(rows_to_run, 1):
            print(f"\n{'='*72}")
            print(f"# [{ri}/{len(rows_to_run)}] Paper row {paper_row}: {label}  ({kind} v{code_v})")
            elapsed = time.time() - t0
            print(f"#   elapsed so far: {elapsed/60:.1f} min")
            print(f"{'='*72}")

            if kind == "scalar":
                spec = build_spec_scalar(code_v, with_vix=False)
                for seed in args.seeds:
                    run_scalar(spec, train_csv, val_csv, test_csv, args.out_dir, seed,
                               max_epochs=args.max_epochs, patience=args.patience,
                               batch=args.batch, lr=args.lr, fold_tag=fold_tag)
            elif kind == "mtl":
                spec = build_spec_mtl(code_v)
                for seed in args.seeds:
                    run_mtl(spec, train_csv, val_csv, test_csv, args.out_dir, seed,
                            max_epochs=args.max_epochs, patience=args.patience,
                            batch=args.batch, lr=args.lr, fold_tag=fold_tag,
                            lambda_aux=args.lambda_aux)
            else:
                print(f"[SKIP] unknown kind={kind}")

        total_min = (time.time() - t0) / 60.0
        print(f"\n{'#'*72}")
        print(f"# All training runs completed in {total_min:.1f} min")
        print(f"{'#'*72}")

    # === Aggregate ===
    print(f"\n{'#'*72}")
    print(f"# AGGREGATING ALL {len(PAPER_TABLE)} ROWS  (seeds={args.seeds})")
    print(f"{'#'*72}")
    df, csv_path = aggregate_table(args.out_dir, args.seeds, fold_tag=fold_tag)
    print_table(df)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
