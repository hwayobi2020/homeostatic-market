"""Causal Transformer matrix batch runner — 9 variants × 3 folds × N seeds.

train_matrix_transformer.py 호출 자동화 — 변종별 정규화 옵션 자동 적용.
SKIP 로직: 이미 끝난 (variant, fold, seed) 는 자동 건너뜀 (재실행 안전).

Usage (Colab T4):
  cd /content/drive/MyDrive/Colab Notebooks/homeostatic-market
  python colab/run_matrix_transformer.py --variants 1 2 3 4 5 6 7 8 9 \
                                          --folds F1 F2 F3 \
                                          --seeds 42 43 44 45 46 \
                                          --continue-on-fail
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(HERE, "dual_3ch", "train_matrix_transformer.py")

# 변종별 정규화 자동 적용 (target 자리에 보조 채널 있는 변종)
NORMBP_VARIANTS = {5, 7}    # mtl_bp, mtl_bp_sp
NORMSP_VARIANTS = {6, 7}    # mtl_sp, mtl_bp_sp
NORMEX_VARIANTS = {8, 9}    # best_base_mtl, best_excess_liq_mtl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants",   nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument("--folds",      nargs="+", default=["F1", "F2", "F3"])
    ap.add_argument("--seeds",      nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience",   type=int, default=30)
    ap.add_argument("--batch",      type=int, default=32)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--lambda-aux", type=float, default=0.1)
    ap.add_argument("--out-dir",    default=os.path.join(HERE, "dual_3ch", "result"))
    ap.add_argument("--dry-run",            action="store_true")
    ap.add_argument("--continue-on-fail",   action="store_true")
    args = ap.parse_args()

    if not os.path.exists(TRAIN_SCRIPT):
        print(f"[FATAL] train script not found: {TRAIN_SCRIPT}")
        sys.exit(1)

    total = len(args.variants) * len(args.folds)
    done  = 0
    failed = []

    print(f"\n{'#' * 72}")
    print(f"# Causal Transformer matrix batch")
    print(f"# variants={args.variants}, folds={args.folds}, seeds={args.seeds}")
    print(f"# total combos = {total} (variant × fold), total runs = {total * len(args.seeds)}")
    print(f"# lambda_aux = {args.lambda_aux}")
    print(f"# out_dir = {args.out_dir}")
    print(f"{'#' * 72}\n")

    for variant in args.variants:
        for fold in args.folds:
            cmd = [sys.executable, TRAIN_SCRIPT,
                   "--variant",     str(variant),
                   "--fold",        fold,
                   "--seeds",       *map(str, args.seeds),
                   "--max-epochs",  str(args.max_epochs),
                   "--patience",    str(args.patience),
                   "--batch",       str(args.batch),
                   "--lr",          str(args.lr),
                   "--lambda-aux",  str(args.lambda_aux),
                   "--out-dir",     args.out_dir]
            if variant in NORMBP_VARIANTS:
                cmd.append("--normalize-bondpp")
            if variant in NORMSP_VARIANTS:
                cmd.append("--normalize-stockpp")
            if variant in NORMEX_VARIANTS:
                cmd.append("--normalize-excess")
            done += 1
            print(f"\n{'=' * 72}")
            print(f"[{done}/{total}] variant={variant}, fold={fold}, seeds={args.seeds}")
            print(f"  cmd = {' '.join(cmd)}")
            print(f"{'=' * 72}")
            if args.dry_run:
                continue
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                msg = f"variant={variant}, fold={fold} FAILED (returncode {ret.returncode})"
                print(f"  ✗ {msg}")
                failed.append(msg)
                if not args.continue_on_fail:
                    print(f"  abort. (--continue-on-fail 로 다음 조합 진행)")
                    sys.exit(ret.returncode)

    print(f"\n{'#' * 72}")
    print(f"# DONE: {done}/{total} combos")
    if failed:
        print(f"# FAILED: {len(failed)}")
        for m in failed:
            print(f"  - {m}")
    print(f"{'#' * 72}")


if __name__ == "__main__":
    main()
