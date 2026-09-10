# -*- coding: utf-8 -*-
"""원점별 IHL 반응 집계 — 풀링 CVaR 가 왜 덜 움직이는지 갈라낸다.

배경.
  §4.3 표(16/18/19)의 IHL CVaR10% 는 원점 182개 × 경로 1,000개를 한 풀로 합친 뒤
  최악 10% 를 자른다.  그 10% 에 어느 (원점,경로) 쌍이 들어가는지는 그 원점의
  변동성이 거의 결정하고, 거시 경로 형태를 바꿔도 "어느 원점이 위험한가"는 별로
  안 바뀐다.  그래서 꼬리 구성은 유지된 채 값만 조금 움직인다 = 희석 가설.
  경쟁 가설: 원점 안에서도 꼬리가 조건에 원래 둔감하다.

읽는 법 (감사 지적 반영).
  - Δ 는 부호를 유지한다.  절대값을 쓰면 추정잡음이 그대로 신호로 둔갑한다.
  - 희석 배율 = (원점별 Δ 평균) / (풀링 Δ).  시드 평균을 먼저 낸 뒤 비를 만든다
    (시드별 비의 평균은 분모가 0 근처인 시드에 지배된다).
  - 스케일 배율 = |풀링 flat CVaR| / |원점별 flat CVaR 평균|.  풀링 쪽이 항상 더
    깊으므로 두 정의는 기계적으로 수준이 다르다.  희석 배율을 읽을 때 이 열을
    같이 봐야 한다 — 배율이 스케일 배율 근처면 정의 차이일 뿐이다.
  - 잡음 바닥: 시드 간 Δ 흩어짐(sd)을 원점 간 Δ 퍼짐(sd)과 나란히 찍는다.
    원점 퍼짐이 시드 흩어짐보다 크지 않으면 원점별 결과를 해석하면 안 된다.
  - 원점별 CVaR 평균 ≠ 풀링 CVaR.  전자는 원점 등가중, 후자는 고변동 원점이
    꼬리를 독점한다.  서로 다른 양이므로 논문 표에 정의를 명시할 것.

Table 19 는 논문과 같은 Diff*(상호작용)로 낸다:
    Diff* = (joint - flat) - [(rate_only - flat) + (liq_only - flat_zm)]

입력: pathshape_*_cache{SUFFIX}{PS.ORIGIN_SUFFIX}/  (원점별 필드를 담은 새 캐시)
"""
import os
import sys
import csv
import math

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_pathshape_rawvol as PS                       # noqa: E402
import agg_seed_se_pathshape as AG                          # noqa: E402  (_load/_t_sf2/JOINT_KEY)
import model_on_realized_rawvol as MR                       # noqa: E402  (§4.2 국면 분할 재사용)

FOLDS = PS.FOLDS
SEEDS = PS.SEEDS
LABELS = AG.LABELS
OSF = PS.ORIGIN_SUFFIX

CACHE_ZM = AG.CACHE_ZM + OSF          # Table 16
CACHE_AN = AG.CACHE_AN + OSF          # Table 18
CACHE_JT = AG.CACHE_JT + OSF          # Table 19

# (풀링 스칼라, 원점별 배열) 쌍.  같은 정의를 두 축에서 계산한 것.
METRIC_PAIRS = [
    ("uw_cvar10", "uw_cvar10_by_origin"),
    ("uw_cvar5",  "uw_cvar5_by_origin"),
    ("uw_cvar1",  "uw_cvar1_by_origin"),
    ("uw_mean",   "uw_mean_by_origin"),
    ("mdd_cvar1", "mdd_cvar1_by_origin"),
    ("term_mean", "term_mean_by_origin"),
]
MAIN_METRIC = "uw_cvar10"
PP = 100.0                                        # %p 표기


# ----------------------------------------------------------------------------
def _walk_leaves(node, prefix=()):
    """캐시 구조가 스크립트마다 달라(k1/k3, joint 중첩) 구조를 가정하지 않는다."""
    if isinstance(node, dict):
        if "uw_cvar10" in node:
            yield ("/".join(prefix) if prefix else "flat"), node
            return
        for k, v in node.items():
            if k == "_origin":
                continue
            yield from _walk_leaves(v, prefix + (k,))


def _ms(vals):
    """평균, 표준오차, n — 시드 축 집계."""
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    if a.size == 1:
        return float(a[0]), float("nan"), 1
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(a.size)), int(a.size)


def _ratio(num, den):
    return num / den if den not in (0.0,) and np.isfinite(den) and abs(den) > 1e-12 else float("nan")


def _origin_meta(sample):
    o = sample.get("_origin") or {}
    return (np.asarray(o.get("tbill", []), float),
            np.asarray(o.get("metab", []), float),
            list(o.get("dates", [])))


def _state_bins(tb, mb):
    """§4.2 의 분할을 그대로 재사용한다 (MR.TBILL_EDGES / MR.METAB_EDGES)."""
    out = {}
    if tb.size:
        out["tbill"] = np.array([MR.abin(float(v), MR.TBILL_EDGES) for v in tb])
    if mb.size:
        out["metab"] = np.array([MR.abin(float(v), MR.METAB_EDGES) for v in mb])
    return out


# ----------------------------------------------------------------------------
def _deltas(per_seed, get_node, metric, org_key):
    """시드별 (풀링 Δ, 원점별 Δ 배열, flat 수준) 을 모은다.  부호 유지."""
    pool, org_vec, lvl_pool, lvl_org = [], [], [], []
    for s in sorted(per_seed):
        d = per_seed[s]
        node = get_node(d, s)
        if node is None:
            continue
        base = d["flat"]
        if metric not in node or metric not in base:
            continue
        pool.append(node[metric] - base[metric])
        lvl_pool.append(base[metric])
        if org_key in node and org_key in base:
            v1 = np.asarray(node[org_key], float)
            v0 = np.asarray(base[org_key], float)
            org_vec.append(v1 - v0)
            lvl_org.append(float(v0.mean()))
    return pool, org_vec, lvl_pool, lvl_org


def _report_row(label, pool, org_vec, lvl_pool, lvl_org, rows_out, meta, bins, tag=()):
    if not pool:
        return None
    dp, dp_se, n_s = _ms(pool)
    lp, _, _ = _ms(lvl_pool)

    if not org_vec:
        print(f"    {label:<26} {dp*PP:>+9.3f} ±{dp_se*PP:<7.3f} {lp*PP:>9.2f}"
              f" {'—':>9} {'—':>9} {'—':>7} {'—':>7} {'—':>8} {'—':>8}")
        return None

    seed_stack = np.stack(org_vec)                     # (n_seed, n_origin)
    per_seed_org_mean = seed_stack.mean(axis=1)        # 시드별 원점평균 Δ
    do, do_se, _ = _ms(list(per_seed_org_mean))
    lo, _, _ = _ms(lvl_org)

    dilution = _ratio(do, dp)                          # 시드평균 후 비 (mean-of-ratios 아님)
    scale = _ratio(abs(lp), abs(lo))                   # 정의 차이로 생기는 기계적 배율
    mean_org = seed_stack.mean(axis=0)                 # 시드평균 원점별 Δ
    spread = float(mean_org.std(ddof=1)) if mean_org.size > 1 else float("nan")
    noise = float(per_seed_org_mean.std(ddof=1)) if n_s > 1 else float("nan")
    pos = float(np.mean(mean_org > 0))
    flip = "  ←부호반대" if np.isfinite(dilution) and dilution < 0 else ""

    print(f"    {label:<26} {dp*PP:>+9.3f} ±{dp_se*PP:<7.3f} {lp*PP:>9.2f}"
          f" {do*PP:>+9.3f} ±{do_se*PP:<7.3f} {dilution:>7.2f} {scale:>7.2f}"
          f" {spread*PP:>8.3f} {noise*PP:>8.3f}{flip}")

    # 국면별 조건부 평균 (§4.2 분할) — 원점별 Δ 가 국면에 따라 뒤집히는지
    for axis, lab in bins.items():
        if lab.size != mean_org.size:
            continue
        cells = []
        for b in MR.BINS:
            idx = np.where(lab == b)[0]
            if idx.size:
                cells.append(f"{b}:{mean_org[idx].mean()*PP:+.3f}(n={idx.size})")
        if cells:
            print(f"      └ {axis:<6} " + "  ".join(cells))

    tb, mb, dates = meta
    for i in range(mean_org.size):
        rows_out.append(list(tag) + [label, i,
                                     dates[i] if i < len(dates) else "",
                                     f"{tb[i]:.6f}" if i < tb.size else "",
                                     f"{mb[i]:.6f}" if i < mb.size else "",
                                     f"{mean_org[i]:.8f}"])
    return dict(pool=dp, org=do, dilution=dilution, scale=scale)


def _header():
    print(f"    {'scenario':<26} {'풀링Δ':>9} {'±SE':<8} {'flat수준':>9}"
          f" {'원점Δ':>9} {'±SE':<8} {'희석배':>7} {'스케일':>7} {'원점퍼짐':>8} {'시드흩':>8}")


# ----------------------------------------------------------------------------
def group_simple(title, cache_dir, metric, rows_out):
    print("\n" + "=" * 126)
    print(f"[{title}]   metric={metric}   cache={os.path.basename(cache_dir)}")
    for fold in FOLDS:
        per_seed = AG._load(cache_dir, fold)
        if not per_seed:
            print(f"  [{LABELS[fold]}] 캐시 없음 → {cache_dir}")
            continue
        sample = next(iter(per_seed.values()))
        org_key = dict(METRIC_PAIRS).get(metric)
        if org_key not in (sample.get("flat") or {}):
            print(f"  [{LABELS[fold]}] 원점별 필드 없음 — 새 캐시를 만들어야 한다.")
            continue
        meta = _origin_meta(sample)
        bins = _state_bins(meta[0], meta[1])
        n_org = len(sample["flat"][org_key])
        print(f"\n  [{LABELS[fold]}]  seeds={sorted(per_seed)}  n_origin={n_org}")
        _header()
        for lab, _node in _walk_leaves(sample):
            if lab == "flat":
                continue
            def _get(d, s, _lab=lab):
                return dict(_walk_leaves(d)).get(_lab)
            pool, org_vec, lp, lo = _deltas(per_seed, _get, metric, org_key)
            _report_row(lab, pool, org_vec, lp, lo, rows_out, meta, bins,
                        tag=(title.split()[1], fold, metric))


def group_table19(metric, rows_out):
    """논문 Table 19 와 같은 Diff*(상호작용)를 풀링·원점별 두 축으로."""
    print("\n" + "=" * 126)
    print(f"[Table 19  joint Diff* = (joint-flat) - ((rate_only-flat) + (liq_only-flat_zm))]  metric={metric}")
    org_key = dict(METRIC_PAIRS).get(metric)
    for fold in FOLDS:
        cj = AG._load(CACHE_JT, fold)
        cz = AG._load(CACHE_ZM, fold)
        if not cj or not cz:
            print(f"  [{LABELS[fold]}] 캐시 없음 → {CACHE_JT if not cj else CACHE_ZM}")
            continue
        sj = next(iter(cj.values()))
        if org_key not in (sj.get("flat") or {}):
            print(f"  [{LABELS[fold]}] 원점별 필드 없음 — 새 캐시를 만들어야 한다.")
            continue
        meta = _origin_meta(sj)
        bins = _state_bins(meta[0], meta[1])
        print(f"\n  [{LABELS[fold]}]  seeds={sorted(cj)}")
        _header()

        for liq in ("ramp_up", "ramp_down", "step_up", "step_down"):
            pool, org_vec, lvl_p, lvl_o = [], [], [], []
            for s in sorted(cj):
                if s not in cz:
                    continue
                j, z = cj[s], cz[s]
                try:
                    jn = j["joint"][AG.JOINT_KEY[liq]]
                    rn = j["rate_only"]
                    zn = z["shape_metab"]["k1"][liq]
                except (KeyError, TypeError):
                    continue
                fj, fz = j["flat"], z["flat"]
                pool.append((jn[metric] - fj[metric])
                            - ((rn[metric] - fj[metric]) + (zn[metric] - fz[metric])))
                lvl_p.append(fj[metric])
                if org_key in jn and org_key in zn:
                    a = lambda d: np.asarray(d[org_key], float)      # noqa: E731
                    org_vec.append((a(jn) - a(fj)) - ((a(rn) - a(fj)) + (a(zn) - a(fz))))
                    lvl_o.append(float(a(fj).mean()))
            _report_row(f"uptrend x {liq}", pool, org_vec, lvl_p, lvl_o,
                        rows_out, meta, bins, tag=("19", fold, metric))


def main():
    print("#" * 126)
    print(f"# 원점별 IHL 반응 (희석 vs 꼬리둔감)  TAG_PREFIX={PS.TAG_PREFIX}  "
          f"SUFFIX='{PS.CACHE_SUFFIX}'  ORIGIN_SUFFIX='{OSF}'")
    print("# Δ 는 %p, 부호 유지.  기준선은 세 캐시 모두 res['flat'](양축 마지막값 지속).")
    print("# 희석배 = 원점Δ/풀링Δ (시드평균 후).  스케일 = |풀링 flat|/|원점 flat 평균| — 정의 차이로")
    print("#   생기는 기계적 배율이므로, 희석배가 스케일 근처면 희석 증거가 아니다.")
    print("# 원점퍼짐(원점 간 sd) 이 시드흩(시드 간 sd) 보다 크지 않으면 원점별 해석 금지.")
    print("#" * 126)

    rows = [["table", "fold", "metric", "scenario", "origin_idx",
             "date", "tbill", "metab", "d_metric"]]
    for metric in (MAIN_METRIC, "uw_mean"):
        group_simple("Table 16  zero-mean (평균·분산 일치 형태)", CACHE_ZM, metric, rows)
        group_simple("Table 18  anchored (금리 연속 vs 계단)", CACHE_AN, metric, rows)
        group_table19(metric, rows)

    out_csv = os.path.join(PS.RESULT_DIR, f"by_origin_delta{PS.CACHE_SUFFIX}{OSF}.csv")
    if len(rows) > 1:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        print(f"\n[csv] 원점별 Δ (시드평균) {len(rows)-1} 행 → {out_csv}")
    else:
        print("\n[csv] 기록할 원점별 Δ 가 없다 — 새 캐시를 먼저 만들 것.")

    print("\n[남은 것] 플라시보(같은 flat 을 다른 sim 난수로 두 번 돌린 차이)는 아직 없다.")
    print("          그것 없이는 원점퍼짐의 잡음 성분을 시드흩만으로 근사하는 셈이다.")


if __name__ == "__main__":
    main()
