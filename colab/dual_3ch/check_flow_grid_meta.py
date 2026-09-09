# -*- coding: utf-8 -*-
"""용량 격자 9조합의 summary 메타 대조.

게재판(4층/hidden 128, wd 0.5)은 태그에 접미사가 없어 run_lr_sweep 결과와
파일명이 같다.  run_tuning_all.flow_cell 은 summary 가 있으면 [skip] 하고
그 파일을 읽으므로, 옛 실행(max_epoch 60, 조기종료 있음, val NLL 기준)의
결과를 나머지 8조합(max_epoch 30, 조기종료 없음, val CRPS 기준)과 나란히
비교했을 수 있다.  그러면 조합 간 CRPS 차이(0.00001~0.00007)는 비교가
성립하지 않는다.

각 조합의 체크포인트 기준·에폭·파라미터 수·파일 시각을 찍어 확인한다.

사용
----
    !python colab/dual_3ch/check_flow_grid_meta.py
"""
import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "result")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    pat = os.path.join(RESULT, "garch_flow_ar_rvAbl_full_fpath_novol_lr0d0001*_summary.json")
    files = [f for f in sorted(glob.glob(pat)) if "_VAL_" not in f]
    if not files:
        print(f"결과 없음: {pat}")
        return

    hdr = ("{:44} {:>4} {:>5} {:>10} {:>13} {:>8} {:>8} {:>12}"
           .format("tag", "lyr", "hid", "params", "ckpt_criterion",
                   "crps_ep", "nll_ep", "mtime"))
    print(hdr)
    print("-" * len(hdr))
    crits = set()
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:                                    # noqa: BLE001
            print(f"{os.path.basename(f)[:44]:44} [읽기 실패] {e!r}")
            continue
        name = os.path.basename(f).replace("garch_flow_ar_", "")
        name = name.replace("rvAbl_full_fpath_novol_", "").replace("_summary.json", "")
        crit = d.get("ckpt_criterion", "(없음)")
        crits.add(crit)

        def g(k, w=10):
            v = d.get(k)
            return f"{v:,}" if isinstance(v, int) else str(v) if v is not None else "-"

        print("{:44} {:>4} {:>5} {:>10} {:>13} {:>8} {:>8} {:>12}".format(
            name[:44], g("n_flow_layers"), g("n_flow_hidden"), g("n_params"),
            str(crit), g("best_crps_epoch"), g("best_epoch"),
            datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%m-%d %H:%M")))

    print(f"\n파일 {len(files)} 개.  ckpt_criterion 종류 = {sorted(crits)}")
    if len(crits) > 1:
        print("→ 조합마다 체크포인트 기준이 다르다.  같은 표에서 CRPS 를 비교할 수 없다.")
    else:
        print("→ 기준이 하나다.  비교 조건은 일치한다.")
    print("\n[읽는 법] ckpt_criterion 이 val_crps_z 면 새 판본(조기종료 없음),")
    print("  val_nll 이거나 '(없음)' 이면 옛 판본이다.  mtime 이 크게 벌어져도 의심한다.")


if __name__ == "__main__":
    main()
