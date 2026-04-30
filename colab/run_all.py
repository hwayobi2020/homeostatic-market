"""Batch runner — 모든 실험을 한 번에 5 시드로 순차 실행.

사용법 (Colab):
    !python /content/drive/MyDrive/Colab\\ Notebooks/homeostatic-market/colab/run_all.py
    # tbill_26w_lag 포함하여 reproduce 시:
    !python ... --with-26w
    # 1 시드만 빠르게 검증:
    !python ... --seeds 42

실행 순서:
  1. Base K2_104 (cond → sp_return)
  2. Base x liquidity (ablation: excess_liq_wr 제거)
  3. Stage 1 MacroExpander (target = excess_liq_wr)
  4. Stage 1 (vix) (target = vix_wr) — OOS 폭발 알려진 실패 케이스
  5. Stage 2 PriceGenerator (paired with Stage 1 ckpt)
  6. Stage 2 (대조군) Single-stage with mask
  7. MTL 2ch joint (excess_liq_wr + sp_return)
  8. MTL 3ch joint (+ vix_wr)
  9. MTL 2ch (sp + bondpp_3m), 정규화 X
 10. MTL 2ch (sp + bondpp_3m), 정규화 O
 11. MTL 3ch (liq + sp + bondpp_3m), 정규화 X
 12. MTL 3ch (liq + sp + bondpp_3m), 정규화 O
 13. MTL 3ch (sp + bondpp_3m + vix), 정규화 X
 14. MTL 3ch (sp + bondpp_3m + vix), 정규화 O
 15. MTL 2ch (sp + vix), 정규화 X
 16. MTL 2ch (sp + vix), 정규화 O (vix_wr 만)
 17. MTL 2ch (sp + stockpp_3m), 정규화 X (ablation: bondpp 와 1대1 비교)

각 실험 5 시드 (42, 123, 777, 0, 99) 기본. T4 ~12시간, A100 ~7시간 (17 실험 기준).
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# (folder, script, extra_args, log_name, label)
EXPERIMENTS = [
    ("k2_104",   "train.py",              [],                       "K2_104_multi_run.log",      "1. Base K2_104"),
    ("k2_104",   "train_ablation.py",     [],                       "K2_104_ablation_run.log",   "2. Base x liquidity (ablation)"),
    ("dual_3ch", "train_stage1.py",       [],                       "stage1_multi_run.log",      "3. Stage 1 (MacroExpander)"),
    ("dual_3ch", "train_stage1_vix.py",   [],                       "stage1vix_multi_run.log",   "4. Stage 1 (vix) — known OOS broken"),
    ("dual_3ch", "train_stage2.py",       [],                       "stage2_multi_run.log",      "5. Stage 2 (PriceGenerator) — needs Stage 1 ckpt"),
    ("dual_3ch", "train_singlestage.py",  [],                       "singlestage_multi_run.log", "6. Stage 2 (대조군 single-stage)"),
    ("dual_3ch", "train_joint.py",        [],                       "joint_multi_run.log",       "7. MTL 2ch joint (excess_liq_wr + sp_return)"),
    ("dual_3ch", "train_mtl_3ch.py",      [],                       "mtl3_multi_run.log",        "8. MTL 3ch joint (+ vix_wr)"),
    ("dual_3ch", "train_mtl_bondpp2.py",  [],                       "mtl_bp2_run.log",           "9. MTL 2ch (sp + bondpp_3m), 정규화 X"),
    ("dual_3ch", "train_mtl_bondpp2.py",  ["--normalize-bondpp"],   "mtl_bp2_normbp_run.log",   "10. MTL 2ch (sp + bondpp_3m), 정규화 O"),
    ("dual_3ch", "train_mtl_bondpp3.py",  [],                       "mtl_bp3_run.log",          "11. MTL 3ch (liq + sp + bondpp_3m), 정규화 X"),
    ("dual_3ch", "train_mtl_bondpp3.py",  ["--normalize-bondpp"],   "mtl_bp3_normbp_run.log",   "12. MTL 3ch (liq + sp + bondpp_3m), 정규화 O"),
    ("dual_3ch", "train_mtl_bondpp_vix.py", [],                     "mtl_bp_vix_run.log",       "13. MTL 3ch (sp + bondpp_3m + vix), 정규화 X"),
    ("dual_3ch", "train_mtl_bondpp_vix.py", ["--normalize-bondpp"], "mtl_bp_vix_normbp_run.log","14. MTL 3ch (sp + bondpp_3m + vix), 정규화 O"),
    ("dual_3ch", "train_mtl_2ch_vix.py",   [],                      "mtl_2ch_vix_run.log",       "15. MTL 2ch (sp + vix), 정규화 X"),
    ("dual_3ch", "train_mtl_2ch_vix.py",   ["--normalize-vix"],     "mtl_2ch_vix_normvix_run.log","16. MTL 2ch (sp + vix), 정규화 O (vix_wr)"),
    ("dual_3ch", "train_mtl_ppstock2.py",  [],                      "mtl_pps2_run.log",          "17. MTL 2ch (sp + stockpp_3m)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["42", "123", "777", "0", "99"])
    ap.add_argument("--with-26w", action="store_true",
                    help="모든 실험에 tbill_26w_lag 포함 (legacy 결과 reproduce)")
    ap.add_argument("--only", nargs="+", default=[],
                    help="특정 실험 번호만 실행 (예: --only 13). --skip 보다 우선")
    ap.add_argument("--skip", nargs="+", default=[],
                    help="특정 실험 번호 스킵 (예: --skip 4 7). --only 지정 시 무시됨")
    args = ap.parse_args()

    only = set(args.only)
    skipped = set(args.skip) if not only else set()
    print(f"\n{'#' * 70}")
    print(f"# Batch run — {len(EXPERIMENTS)} experiments")
    print(f"# seeds = {args.seeds}")
    print(f"# with-26w = {args.with_26w}")
    if only:
        print(f"# only = {sorted(only, key=int)}")
    elif skipped:
        print(f"# skip = {sorted(skipped, key=int)}")
    print(f"{'#' * 70}\n")

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

        # 학습 후 multi-seed 집계 라인 자동 출력 (성공한 경우)
        if proc.returncode == 0:
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # train script 의 multi-seed 집계 출력 ("multi-seed (n=" 으로 시작) 부터 끝까지
                start = None
                for k, ln in enumerate(lines):
                    if "multi-seed (n=" in ln:
                        start = k
                        break
                if start is not None:
                    print("  --- multi-seed 집계 ---")
                    for ln in lines[start:]:
                        text = ln.rstrip()
                        if text:
                            print("  " + text)
            except Exception as e:
                print(f"  log read error: {e}")
        else:
            # 실패 시 마지막 15 라인만 (디버깅용)
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                print("  --- log tail (실패) ---")
                for ln in lines[-15:]:
                    text = ln.rstrip()
                    if text:
                        print("  " + text)
            except Exception as e:
                print(f"  log read error: {e}")

    total = time.time() - t_start
    print(f"\n{'#' * 70}")
    print(f"# DONE. Total: {total/60:.1f} min")
    print(f"{'#' * 70}")
    for label, st in statuses:
        marker = "✓" if st == "OK" else ("·" if st == "skipped" else "✗")
        print(f"  {marker} {st:<20s} {label}")


if __name__ == "__main__":
    main()
