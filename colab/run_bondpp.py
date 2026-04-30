"""Batch runner — bondpp_3m 추가 6 변종 한 번에 5 시드로 순차 실행.

6 변종:
  1. MTL 2ch (sp + bondpp), bondpp 정규화 X
  2. MTL 2ch (sp + bondpp), bondpp 정규화 O
  3. MTL 3ch (liq + sp + bondpp), bondpp 정규화 X
  4. MTL 3ch (liq + sp + bondpp), bondpp 정규화 O
  5. MTL 3ch (sp + bondpp + vix), bondpp 정규화 X
  6. MTL 3ch (sp + bondpp + vix), bondpp 정규화 O

사용법 (Colab):
    !python /content/drive/MyDrive/Colab\\ Notebooks/homeostatic-market/colab/run_bondpp.py
    !python ... --seeds 42                # 1 시드만 빠르게 검증
    !python ... --skip 1 3                # 정규화 O 만 돌리기

예상 시간: T4 약 18 분 (3 분 × 6), A100 약 9 분.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# (folder, script, extra_args, log_name, label)
EXPERIMENTS = [
    ("dual_3ch", "train_mtl_bondpp2.py",   [],                       "mtl_bp2_run.log",            "1. MTL 2ch (sp + bondpp), 정규화 X"),
    ("dual_3ch", "train_mtl_bondpp2.py",   ["--normalize-bondpp"],   "mtl_bp2_normbp_run.log",     "2. MTL 2ch (sp + bondpp), 정규화 O"),
    ("dual_3ch", "train_mtl_bondpp3.py",   [],                       "mtl_bp3_run.log",            "3. MTL 3ch (liq + sp + bondpp), 정규화 X"),
    ("dual_3ch", "train_mtl_bondpp3.py",   ["--normalize-bondpp"],   "mtl_bp3_normbp_run.log",     "4. MTL 3ch (liq + sp + bondpp), 정규화 O"),
    ("dual_3ch", "train_mtl_bondpp_vix.py", [],                      "mtl_bp_vix_run.log",         "5. MTL 3ch (sp + bondpp + vix), 정규화 X"),
    ("dual_3ch", "train_mtl_bondpp_vix.py", ["--normalize-bondpp"],  "mtl_bp_vix_normbp_run.log",  "6. MTL 3ch (sp + bondpp + vix), 정규화 O"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["42", "123", "777", "0", "99"])
    ap.add_argument("--with-26w", action="store_true",
                    help="모든 실험에 tbill_26w_lag 포함 (legacy 3ch cond reproduce)")
    ap.add_argument("--skip",  nargs="+", default=[],
                    help="특정 실험 번호 스킵 (예: --skip 2 4)")
    args = ap.parse_args()

    print(f"\n{'#' * 70}")
    print(f"# Batch run (bondpp) — {len(EXPERIMENTS)} variants")
    print(f"# seeds = {args.seeds}")
    print(f"# with-26w = {args.with_26w}")
    print(f"{'#' * 70}\n")

    skipped = set(args.skip)
    t_start = time.time()
    statuses = []
    for i, (folder, script, extra_args, log_name, label) in enumerate(EXPERIMENTS, start=1):
        if str(i) in skipped:
            print(f"\n--- skipped: {label} ---")
            statuses.append((label, "skipped"))
            continue

        cwd = os.path.join(ROOT, folder)
        result_dir = os.path.join(cwd, "result")
        os.makedirs(result_dir, exist_ok=True)
        log_path = os.path.join(result_dir, log_name)

        cmd = [sys.executable, script, "--seeds"] + list(args.seeds) + list(extra_args)
        if args.with_26w:
            cmd.append("--with-26w")

        ts = time.time()
        print(f"\n{'=' * 70}")
        print(f"{label}")
        print(f"  cwd: {cwd}")
        print(f"  cmd: {' '.join(cmd)}")
        print(f"  log: {log_path}")
        print(f"{'=' * 70}")

        with open(log_path, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - ts
        status = "OK" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
        print(f"  [{elapsed/60:.1f} min] {status}")
        statuses.append((label, status))

    total = time.time() - t_start
    print(f"\n{'#' * 70}")
    print(f"# DONE. Total: {total/60:.1f} min")
    print(f"{'#' * 70}")
    for label, st in statuses:
        marker = "✓" if st == "OK" else ("·" if st == "skipped" else "✗")
        print(f"  {marker} {st:<20s} {label}")


if __name__ == "__main__":
    main()
