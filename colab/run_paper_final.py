"""Paper 표 전체 재학습 master — 모든 변종 × 3 fold × 5 seed.

각 train script 의 skip 분기가 ckpt 존재 시 자동 스킵하므로 재실행 안전.
중간 끊김 후 재실행해도 이미 끝난 변종/fold/seed 는 자동 건너뜀.

ckpt 저장 폴더: colab/<folder>/result_paper_final/  (이전 result/ 와 분리)

학습 대상 (paper 표 4-17, dual_3ch only — k2_104 의 행 1·2·3 은 별도 처리):
    4   stage1                                   (sp target 없음, fold 만)
    5   stage1_vix --normalize-vix               (sp target 없음, fold + vix정규)
    6   stage2 --normalize-sp                    (paired stage1 ckpt 사용)
    7   singlestage --normalize-sp
    8   joint --normalize-sp
    9   mtl_2ch_vix --no-liq --normalize-vix --normalize-sp
    10  mtl_bondpp2 --no-liq --normalize-bondpp --normalize-sp
    11  mtl_ppstock2 --no-liq --normalize-stockpp --normalize-sp
    12  mtl_bondpp_vix --no-liq --normalize-bondpp --normalize-vix --normalize-sp
    13  mtl_bp_stockpp --no-liq --normalize-sp   (bondpp/stockpp 정규화 default)
    14  mtl_bondpp3 --normalize-bondpp --normalize-sp
    15  mtl_3ch --normalize-vix --normalize-sp
    16  mtl_bondpp3 --no-liq --normalize-bondpp --normalize-sp
    17  mtl_bondpp_vix --no-liq --normalize-bondpp --normalize-vix --normalize-sp  (= 행 12 와 동일 ckpt — skip 됨)
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SEEDS = ["42", "123", "777", "0", "99"]
FOLDS = ["F1", "F2", "F3"]

# (paper_row, train_script_relative, args)
# train_script_relative: colab/ 기준 상대 경로
COMMANDS = [
    ( 1, "k2_104/train.py",                 "--normalize-sp"),
    ( 2, "k2_104/train.py",                 "--no-liq --normalize-sp"),
    ( 3, "k2_104/train_addbp.py",           "--normalize-bondpp --normalize-sp"),
    ( 4, "dual_3ch/train_stage1.py",        ""),
    ( 5, "dual_3ch/train_stage1_vix.py",    "--normalize-vix"),
    ( 6, "dual_3ch/train_stage2.py",        "--normalize-sp"),
    ( 7, "dual_3ch/train_singlestage.py",   "--normalize-sp"),
    ( 8, "dual_3ch/train_joint.py",         "--normalize-sp"),
    ( 9, "dual_3ch/train_mtl_2ch_vix.py",   "--no-liq --normalize-vix --normalize-sp"),
    (10, "dual_3ch/train_mtl_bondpp2.py",   "--no-liq --normalize-bondpp --normalize-sp"),
    (11, "dual_3ch/train_mtl_ppstock2.py",  "--no-liq --normalize-stockpp --normalize-sp"),
    (12, "dual_3ch/train_mtl_bondpp_vix.py","--no-liq --normalize-bondpp --normalize-vix --normalize-sp"),
    (13, "dual_3ch/train_mtl_bp_stockpp.py","--no-liq --normalize-sp"),  # bondpp/stockpp 정규화 default True
    # 행 14: cond=1ch (no-liq) 로 학습. target 에 excess_liq 있어 cond redundancy 제거.
    # ckpt 명 = mtl_bp3_noliq_normbp_normsr → 행 16 ckpt 와 동일 (paper 표 라벨만 다름).
    (14, "dual_3ch/train_mtl_bondpp3.py",   "--no-liq --normalize-bondpp --normalize-sp"),
    (15, "dual_3ch/train_mtl_3ch.py",       "--normalize-vix --normalize-sp"),
    (16, "dual_3ch/train_mtl_bondpp3.py",   "--no-liq --normalize-bondpp --normalize-sp"),
    # 17 은 12 와 동일 명령어 (skip 분기에서 자동 처리) — 명시 X
]

OUT_SUBDIR_NAME = "result_paper_final"


def run_one(script_rel, args, fold, seeds, out_dir):
    full_script = os.path.join(HERE, script_rel)
    cmd = ["python", full_script, "--fold", fold,
           "--seeds", *seeds,
           "--out-dir", out_dir]
    if args.strip():
        cmd.extend(args.strip().split())
    # stage2 의 경우 --stage1-dir 이 같은 폴더라 default 와 다름 → 명시
    if "train_stage2.py" in script_rel:
        cmd.extend(["--stage1-dir", out_dir])
    print(f"\n>>> [{fold}] {os.path.basename(script_rel)}  {args}")
    print(f"    {' '.join(cmd)}")
    sys.stdout.flush()
    return subprocess.call(cmd)


def main():
    # paper_row 별 fold 별 seed 5개 일괄 실행
    t0 = time.time()
    for paper_row, script_rel, args in COMMANDS:
        folder = script_rel.split("/")[0]
        out_dir = os.path.join(HERE, folder, OUT_SUBDIR_NAME)
        os.makedirs(out_dir, exist_ok=True)
        for fold in FOLDS:
            rc = run_one(script_rel, args, fold, SEEDS, out_dir)
            if rc != 0:
                print(f"!!! 명령어 실패 (returncode={rc}): row={paper_row} fold={fold}")
            elapsed = (time.time() - t0) / 60
            print(f"--- 누적 경과시간: {elapsed:.1f}분")

    total = (time.time() - t0) / 60
    print(f"\n=== 전체 학습 완료. 총 {total:.1f}분 ===")
    print(f"\n  평가:  python {os.path.join(HERE, 'eval_paper_table_3fold.py')} "
          f"--out-subdir {OUT_SUBDIR_NAME}")


if __name__ == "__main__":
    main()
