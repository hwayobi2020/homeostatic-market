"""Batch runner — frbsf_wr target 추가 ablation (변종 A).

검증 목적:
  ★ best (sp + bondpp 2ch) sp_return median = -2.199 위에 frbsf_wr 를 target 채널로 추가.
  - frbsf_wr 직교성 통과 (VIF 1.04, weekly nz_diff_pct 100%)
  - cond/target gradient 위치 효과 (bondpp_3m breakthrough) 의 변수 확장 검증
  - 같은 채널 추가 패턴 (bondpp3 sp+liq+bp 시 negative transfer -2.16) 이 frbsf 에도 발생하는지 비교

2 변종:
  1. MTL 3ch (sp + bondpp + frbsf_wr), bondpp 정규화 X
  2. MTL 3ch (sp + bondpp + frbsf_wr), bondpp 정규화 O

사전 준비 (로컬 또는 Colab 양쪽):
    python data/build_frbsf_weekly.py
    → colab/dual_3ch/data/weekly_ppbond_frbsf_{train,test}.csv 생성됨

Colab 사용법:
    !python /content/drive/MyDrive/Colab\\ Notebooks/homeostatic-market/colab/run_frbsf_ablation.py
    !python ... --seeds 42                # 1 시드만 빠르게 검증
    !python ... --skip 2                  # 정규화 X 만 돌리기

예상 시간: T4 약 6분 (3분 × 2), A100 약 3분.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# (folder, script, extra_args, log_name, label)
EXPERIMENTS = [
    ("dual_3ch", "train_mtl_bondpp_frbsf.py", [],                       "mtl_bp_frbsf_run.log",         "1. MTL 3ch (sp + bondpp + frbsf_wr), 정규화 X"),
    ("dual_3ch", "train_mtl_bondpp_frbsf.py", ["--normalize-bondpp"],   "mtl_bp_frbsf_normbp_run.log",  "2. MTL 3ch (sp + bondpp + frbsf_wr), 정규화 O"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["42", "123", "777", "0", "99"])
    ap.add_argument("--with-26w", action="store_true",
                    help="모든 실험에 tbill_26w_lag 포함")
    ap.add_argument("--only",  nargs="+", default=[],
                    help="특정 실험 번호만 실행 (예: --only 1)")
    ap.add_argument("--skip",  nargs="+", default=[],
                    help="특정 실험 번호 스킵 (예: --skip 2)")
    args = ap.parse_args()

    only = set(args.only)
    skipped = set(args.skip) if not only else set()
    print(f"\n{'#' * 70}")
    print(f"# Batch run (frbsf ablation) — {len(EXPERIMENTS)} variants")
    print(f"# seeds = {args.seeds}")
    print(f"# with-26w = {args.with_26w}")
    if only:
        print(f"# only = {sorted(only, key=int)}")
    elif skipped:
        print(f"# skip = {sorted(skipped, key=int)}")
    print(f"{'#' * 70}\n")

    # 사전 점검: weekly_ppbond_frbsf csv 가 dual_3ch/data 에 있는지
    dual_data = os.path.join(ROOT, "dual_3ch", "data")
    for split in ("train", "test"):
        p = os.path.join(dual_data, f"weekly_ppbond_frbsf_{split}.csv")
        if not os.path.exists(p):
            print(f"⚠ {p} 없음")
            print(f"  먼저 'python data/build_frbsf_weekly.py' 실행 필요.")
            sys.exit(1)
    print(f"✓ csv 확인: {dual_data}/weekly_ppbond_frbsf_{{train,test}}.csv\n")

    t_start = time.time()
    statuses = []
    for i, (folder, script, extra_args, log_name, label) in enumerate(EXPERIMENTS, start=1):
        if only and str(i) not in only:
            print(f"\n--- not in --only: {label} ---")
            statuses.append((label, "skipped"))
            continue
        if str(i) in skipped:
            print(f"\n--- skipped: {label} ---")
            statuses.append((label, "skipped"))
            continue

        cwd = os.path.join(ROOT, folder)
        result_dir = os.path.join(cwd, "result")
        os.makedirs(result_dir, exist_ok=True)

        cmd = [sys.executable, "-u", script, "--seeds"] + list(args.seeds) + list(extra_args)
        if args.with_26w:
            cmd.append("--with-26w")

        ts = time.time()
        print(f"\n{'=' * 70}")
        print(f"{label}")
        print(f"  cwd: {cwd}")
        print(f"  cmd: {' '.join(cmd)}")
        print(f"{'=' * 70}", flush=True)

        # stdout/stderr 미지정 → 부모 (Colab cell) 로 그대로 흘러감 (실시간 출력)
        proc = subprocess.run(cmd, cwd=cwd)
        elapsed = time.time() - ts
        status = "OK" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
        print(f"\n  [{elapsed/60:.1f} min] {status}", flush=True)
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
