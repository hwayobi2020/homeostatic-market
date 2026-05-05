"""Paper matrix batch runner — 7 variants × 3 folds × N seeds.

paper_plan.txt 매트릭스 전체 학습 자동화. 각 (variant, fold) 조합당 1회의
train_matrix.py 호출 (내부에서 multi-seed loop). 변종별 정규화 옵션 자동 적용.

Usage (Colab T4):
  cd /content/drive/MyDrive/Colab\ Notebooks/homeostatic-market
  python colab/run_matrix.py --variants 1 2 3 4 5 6 7 \
                              --folds F1 F2 F3 \
                              --seeds 42 43 44 45 46

Usage (Windows local smoke test):
  python colab/run_matrix.py --variants 1 --folds F1 --seeds 42 \
                              --max-epochs 2 --patience 100 --batch 8 --dry-run
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(HERE, "dual_3ch", "train_matrix.py")

# 변종별 정규화 자동 적용 — target 위치에 bondpp/stockpp 있는 변종만
NORMBP_VARIANTS = {5, 7}   # mtl_bp, mtl_bp_sp
NORMSP_VARIANTS = {6, 7}   # mtl_sp, mtl_bp_sp


def main():
    ap = argparse.ArgumentParser(description="Paper matrix batch runner (7 variants × 3 folds × N seeds)")
    ap.add_argument("--variants", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--folds",    nargs="+", default=["F1", "F2", "F3"])
    ap.add_argument("--seeds",    nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience",   type=int, default=30)
    ap.add_argument("--batch",      type=int, default=32)
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--out-dir",    default=os.path.join(HERE, "dual_3ch", "result"))
    ap.add_argument("--dry-run",    action="store_true",
                    help="명령만 출력, 실제 실행 X")
    ap.add_argument("--continue-on-fail", action="store_true",
                    help="개별 (variant,fold) 실패 시 다음 조합 진행")
    args = ap.parse_args()

    if not os.path.exists(TRAIN_SCRIPT):
        print(f"[FATAL] train script not found: {TRAIN_SCRIPT}")
        sys.exit(1)

    total = len(args.variants) * len(args.folds)
    done = 0
    failed = []

    print(f"\n{'#' * 72}")
    print(f"# Paper matrix batch: variants={args.variants}, folds={args.folds}, "
          f"seeds={args.seeds}")
    print(f"# total (variant×fold) combos = {total}, "
          f"total runs = {total * len(args.seeds)}")
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
                   "--out-dir",     args.out_dir]
            if variant in NORMBP_VARIANTS:
                cmd.append("--normalize-bondpp")
            if variant in NORMSP_VARIANTS:
                cmd.append("--normalize-stockpp")
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
                    print(f"  abort. (--continue-on-fail 로 다음 조합 진행 가능)")
                    sys.exit(ret.returncode)

    print(f"\n{'#' * 72}")
    print(f"# DONE: {done}/{total} (variant×fold) combos")
    if failed:
        print(f"# FAILED: {len(failed)}")
        for m in failed:
            print(f"  - {m}")
    print(f"{'#' * 72}")


if __name__ == "__main__":
    main()
