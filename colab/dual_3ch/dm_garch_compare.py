# -*- coding: utf-8 -*-
"""MAC-Flow vs GARCH-ST(재적합) DM 검정 — §4.1.2 (표 12) 대응.

§4.1.3(VAE/GAN)에는 `run_thin_compare.py` 가 DM·비중첩·통합 검정을 싣는데
§4.1.2 에는 검정이 없었다.  리뷰어 1 #1("no standard errors, no intervals,
no test") 대응을 GARCH-ST 비교에도 맞춘다.

짝짓기
------
원점별(13 주 평균) CRPS 를 조건 시점 날짜로 교집합해 정렬한다.
  MAC-Flow  : garch_flow_ar_rvAbl_full_fpath_novol_d2_s{seed}_{fold}_crps_per_origin.npy
              5 시드 평균 (학습 변동을 평균으로 흡수, DM 은 원점 축 검정)
  GARCH-ST  : garch_xpast_refit_{fold}_crps_per_origin.npy
              재적합 파라미터 판.  MLE 라 시드 개념이 없다(결정론적).

부호 규약은 `run_thin_compare.py` 와 같다:
  diff = CRPS(GARCH-ST) − CRPS(MAC-Flow),  **diff > 0 이면 MAC-Flow 우위**.

검정
----
  · 폴드별      : 원점 전수, HAC lag 12 (지평 13 주 → 이웃 원점 최대 12 주 겹침)
  · 비중첩      : 원점 13 간격, 오프셋 13 개를 각각 HAC lag 0 으로 검정
  · 4 폴드 통합 : 폴드를 이어붙여 한 번, HAC lag 12

사용
----
    !python colab/dual_3ch/refit_garch_xpast.py       # 파라미터
    !python colab/dual_3ch/eval_garch_xpast_refit.py  # per-origin CRPS 생성
    !python colab/dual_3ch/dm_garch_compare.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dm_test import dm_test                                        # noqa: E402

RESULT_DIR = os.path.join(HERE, "result")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
# maskall 어블리션(`run_ablations_rawvol.py`)의 기본 시드는 2026~2028 3 개다.
# 5 개를 기본으로 두면 없는 시드가 조용히 빠진 채 meta 에는 5 개로 기록된다.
SEEDS = [int(s) for s in os.environ.get(
    "DMG_SEEDS", "2026,2027,2028").split(",") if s.strip()]
# §4.1.2 는 "we excluded MAC-Flow's future path" 라고 쓴다.  즉 표 12 의 MAC-Flow 는
# 미래 경로를 받지 않는 maskall 판이고, GARCH-X(past) 와 정보가 맞다.
# full 판(`rvAbl_full_fpath_novol_d2`)을 넣으면 MAC-Flow 만 미래 거시 경로를 보는
# 불공정 비교가 된다 — 그건 표 13(§4.1.3) 의 설정이다.
FLOW_TAG = os.environ.get("FLOW_TAG", "rvAbl_maskall")
GARCH_PREFIX = os.environ.get("GARCH_PREFIX", "garch_xpast_refit")
HAC_LAG = 12
STRIDE = 13
OUT = os.path.join(RESULT_DIR, os.environ.get(
    "DMG_OUT", "dm_garch_compare_%s.json" % FLOW_TAG))


def _load(prefix):
    """(원점별 13주평균 CRPS, 날짜) 를 돌려준다.  없으면 (None, None)."""
    c = os.path.join(RESULT_DIR, f"{prefix}_crps_per_origin.npy")
    d = os.path.join(RESULT_DIR, f"{prefix}_origin_dates.npy")
    if not (os.path.exists(c) and os.path.exists(d)):
        return None, None
    arr = np.load(c)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return np.asarray(arr, float), np.load(d, allow_pickle=True).astype(str)


def flow_losses(fold):
    """5 시드 각각의 (손실, 날짜).  하나라도 없으면 그 시드는 건너뛴다."""
    out = []
    for s in SEEDS:
        loss, dates = _load(f"garch_flow_ar_{FLOW_TAG}_s{s}_{fold}")
        if loss is None:
            print(f"    [없음] seed {s}")
            continue
        out.append((s, loss, dates))
    return out


def align(fold):
    """MAC-Flow 5 시드 평균과 GARCH-ST 를 날짜 교집합으로 정렬한다."""
    g_loss, g_dates = _load(f"{GARCH_PREFIX}_{fold}")
    if g_loss is None:
        print(f"  [skip {fold}] {GARCH_PREFIX}_{fold}_crps_per_origin.npy 없음"
              "  — eval_garch_xpast_refit.py 를 먼저 돌려라")
        return None
    seeds = flow_losses(fold)
    if not seeds:
        print(f"  [skip {fold}] MAC-Flow per-origin CRPS 없음")
        return None

    common = set(g_dates)
    for _, _, d in seeds:
        common &= set(d)
    if not common:
        print(f"  [skip {fold}] 날짜 교집합 0 개")
        return None
    order = sorted(common)                       # 시간순 (ISO 날짜 문자열)

    def pick(loss, dates):
        idx = {dt: i for i, dt in enumerate(dates)}
        return np.array([loss[idx[dt]] for dt in order], float)

    F = np.mean([pick(l, d) for _, l, d in seeds], axis=0)
    G = pick(g_loss, g_dates)
    print(f"  원점 교집합 {len(order)} 개  MAC-Flow:{len(seeds)}시드 "
          f"GARCH-ST:재적합 1판")
    return dict(dates=order, flow=F, garch=G,
                n_seed=len(seeds), seeds=[s for s, _, _ in seeds])


def show(tag, G, F, lag):
    r = dm_test(G, F, lag)                       # d = GARCH − Flow  (>0 이면 Flow 우위)
    print(f"    {tag:<26}{r['n']:>5}{r['d_mean']:>12.5f}{r['se']:>11.5f}"
          f"{r['dm_hln']:>10.3f}{r['p_two_sided']:>9.4f}")
    return r


def main():
    print("#" * 108)
    print("# MAC-Flow vs GARCH-ST(재적합) · 원점별 13 주 평균 CRPS · "
          f"MAC-Flow 는 시드 {SEEDS} 평균")
    print(f"# diff = CRPS(GARCH-ST) − CRPS(MAC-Flow)  →  diff > 0 이면 MAC-Flow 우위")
    print(f"# FLOW_TAG={FLOW_TAG}  GARCH={GARCH_PREFIX}  HAC lag={HAC_LAG}")
    print("#" * 108)

    data = {}
    for fold in FOLDS:
        print(f"\n===== {fold}")
        a = align(fold)
        if a:
            data[fold] = a
    if not data:
        print("\n비교할 자료가 없다.")
        return

    hdr = (f"{'fold':<18}{'비교':<26}{'n':>5}{'mean diff':>12}"
           f"{'HAC se':>11}{'DM(HLN)':>10}{'p':>9}")
    print("\n" + "=" * len(hdr))
    print(f"[DM 검정] 원점 시간순 · HAC lag={HAC_LAG} · HLN 보정 · 양측")
    print("=" * len(hdr))
    print(hdr)
    res = {"per_fold": {}, "thin": {}, "pooled": {}}
    for fold, a in data.items():
        print(f"{fold:<18}", end="")
        res["per_fold"][fold] = show("MAC-Flow vs GARCH-ST",
                                     a["garch"], a["flow"], HAC_LAG)

    print("\n" + "=" * len(hdr))
    print(f"[비중첩 검정] 원점 {STRIDE} 간격 · 오프셋 {STRIDE} 개 각각 "
          "(부분표본 안에서는 예측구간 겹침 없음, HAC lag=0)")
    print("=" * len(hdr))
    print(f"{'fold':<18}{'n/오프셋':>9}{'mean diff':>12}{'DM 평균':>10}"
          f"{'DM 범위':>20}{'p 중앙값':>10}{'p<.05':>7}{'양수':>7}")
    for fold, a in data.items():
        stats_ = []
        for off in range(STRIDE):
            G, F = a["garch"][off::STRIDE], a["flow"][off::STRIDE]
            if len(G) < 3:
                continue
            stats_.append(dm_test(G, F, 0))
        if not stats_:
            continue
        dm = np.array([s["dm_hln"] for s in stats_], float)
        pv = np.array([s["p_two_sided"] for s in stats_], float)
        dmean = np.array([s["d_mean"] for s in stats_], float)
        k = len(stats_)
        print(f"{fold:<18}{stats_[0]['n']:>9}{dmean.mean():>12.5f}"
              f"{np.nanmean(dm):>10.3f}"
              f"{f'[{np.nanmin(dm):.2f}, {np.nanmax(dm):.2f}]':>20}"
              f"{np.nanmedian(pv):>10.4f}{f'{int((pv < .05).sum())}/{k}':>7}"
              f"{f'{int((dmean > 0).sum())}/{k}':>7}")
        res["thin"][fold] = dict(n_offset=k, n_per_offset=stats_[0]["n"],
                                 d_mean=float(dmean.mean()),
                                 dm_mean=float(np.nanmean(dm)),
                                 p_median=float(np.nanmedian(pv)),
                                 n_sig=int((pv < .05).sum()),
                                 n_pos=int((dmean > 0).sum()))
    print("  '양수' 는 MAC-Flow 가 이긴 오프셋 수.")

    print("\n" + "=" * len(hdr))
    print(f"[4 폴드 통합] 폴드를 이어붙여 한 번 · HAC lag={HAC_LAG}")
    print("=" * len(hdr))
    print(hdr)
    G = np.concatenate([a["garch"] for a in data.values()])
    F = np.concatenate([a["flow"] for a in data.values()])
    print(f"{'ALL':<18}", end="")
    res["pooled"] = show("MAC-Flow vs GARCH-ST", G, F, HAC_LAG)
    print("  폴드 경계에서 시간이 끊기고 폴드마다 변동성 수준이 다르다.")
    print("  폴드별 표와 같이 봐야 한다.")

    # 요청 시드가 아니라 **실제로 쓴 시드**를 폴드별로 적는다.  없는 시드가
    # 조용히 빠진 채 meta 만 요청값으로 남으면 나중에 시드 수를 오독한다.
    res["meta"] = dict(flow_tag=FLOW_TAG, garch_prefix=GARCH_PREFIX,
                       seeds_requested=SEEDS, hac_lag=HAC_LAG, stride=STRIDE,
                       seeds_used={f: a["seeds"] for f, a in data.items()},
                       n_seed_used={f: a["n_seed"] for f, a in data.items()},
                       n_origin={f: len(a["dates"]) for f, a in data.items()})
    missing = {f: sorted(set(SEEDS) - set(a["seeds"]))
               for f, a in data.items() if set(SEEDS) - set(a["seeds"])}
    if missing:
        print(f"\n[주의] 요청했으나 없어서 빠진 시드: {missing}")
        res["meta"]["seeds_missing"] = missing
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
