"""리뷰어1 #1 대응 — §4.3 경로효과의 시드 표준오차 (재추론 없이 캐시만 읽음).

배경
----
MDPI 심사 리뷰어1 지적: Table 16/18/19 의 차이가 0.01~0.54 %p 수준인데 기준값이
10~26 %p 다.  표준오차·신뢰구간·검정이 하나도 없어 안정성을 확인할 방법이 없다.
→ "1,000 draws and five seeds 면 표준오차 재료는 이미 있다."

이 스크립트가 다루는 것 = 그 중 **시드(모델 학습) 변동** 성분.
  · §4.3 표들은 fold × seed 캐시에 개별 저장돼 있고, 표에는 5시드 평균만 실렸다.
  · 같은 seed 안에서 시나리오들이 paired 로 생성된다(torch.manual_seed(seed) 고정).
    → 시드별 차이 d_s 를 만들어 paired 통계를 낼 수 있다.  n=5.
  · 보고: mean(d), sd(d), se=sd/sqrt(n), t=mean/se, two-sided p (t, df=n-1).

이 스크립트가 다루지 *않는* 것 = **원점(origin) 표본 변동** 성분.
  · uw_cvar10 은 sim_metrics 에서 (n_orig × n_sim) 을 ravel 한 풀 분포의 CVaR 이라
    캐시엔 원점별 분해가 남아 있지 않다.
  · 원점 부트스트랩을 하려면 sim_metrics 가 원점별 IHL 을 반환하도록 고쳐 재추론해야
    한다(재학습은 불필요 — ckpt 는 이미 있음).  그리고 origin 이 주간·horizon 13주라
    13배 중첩이므로 반드시 *블록* 부트스트랩이어야 한다(리뷰어1 #2).

읽는 법
-------
  · |t| >= 2.776 (t_{0.975, df=4}) 이면 5% 수준에서 시드변동만으로는 설명 안 됨.
  · 시드 SE 가 차이보다 크면 그 셀은 볼드에서 빼고 "유의차 없음"으로 서술해야 한다.
  · 여기서 살아남아도 원점 성분이 남아 있으므로 최종 판정은 아니다(상한이 아니라 하한).

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/agg_seed_se_pathshape.py
"""
import json
import math
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_pathshape_rawvol as PS                                   # noqa: E402

RESULT_DIR = PS.RESULT_DIR
SUF = PS.CACHE_SUFFIX
FOLDS = PS.FOLDS
SEEDS = PS.SEEDS
LABELS = {"F_gfc": "Financial crisis (2006-2010)",
          "F_long_A": "Recovery (2011-2015)",
          "F_long_B_origin": "COVID (2016-2020)",
          "F_long": "Tightening (2021-2025)"}

# 캐시 3종 (각 스크립트의 CACHE_DIR 과 동일 규칙)
CACHE_ZM = os.path.join(RESULT_DIR, f"pathshape_zeromean_k1k3_cache{SUF}")      # Table 16
CACHE_AN = os.path.join(RESULT_DIR, f"pathshape_anchored_k1k3_cache{SUF}")      # Table 18
CACHE_JT = os.path.join(RESULT_DIR, f"pathshape_joint_ratestep_metabmm_k1_cache{SUF}")  # Table 19

METRICS = ["uw_cvar10", "uw_mean", "skew"]
PP = {"uw_cvar10": 100.0, "uw_mean": 100.0, "skew": 1.0}   # 논문 표기 배율(%p vs 원단위)
UNIT = {"uw_cvar10": "%p", "uw_mean": "%p", "skew": ""}

# t_{0.975, df} — df=1..8
_TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def _load(cache_dir, fold):
    """{seed: dict} — 존재하는 시드만."""
    out = {}
    for s in SEEDS:
        p = os.path.join(cache_dir, f"{fold}_s{s}.json")
        if os.path.exists(p):
            out[s] = json.load(open(p, encoding="utf-8"))
    return out


def _t_sf2(t, df):
    """two-sided p — scipy 없이 t 분포 생존함수(불완전베타)."""
    try:
        from scipy import stats
        return float(2.0 * stats.t.sf(abs(t), df))
    except Exception:
        x = df / (df + t * t)
        return float(_betainc(df / 2.0, 0.5, x))


def _betainc(a, b, x):
    """정규화 불완전베타 I_x(a,b) — 연분수(Lentz).  scipy 없을 때만 사용."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    if x >= (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def paired(diffs, metric):
    """시드별 차이 리스트 → 통계 dict.  단위는 논문 표기(%p)로 환산."""
    a = np.asarray([d for d in diffs if d is not None and np.isfinite(d)], float) * PP[metric]
    n = len(a)
    if n < 2:
        return dict(n=n, mean=float(a.mean()) if n else float("nan"),
                    sd=float("nan"), se=float("nan"), t=float("nan"),
                    p=float("nan"), sig=False, vals=a.tolist())
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se if se > 0 else float("inf") if mean != 0 else 0.0
    df = n - 1
    return dict(n=n, mean=mean, sd=sd, se=se, t=t, p=_t_sf2(t, df),
                sig=abs(t) >= _TCRIT.get(df, 2.776), vals=a.tolist())


def _row(label, st, metric):
    if not np.isfinite(st.get("sd", float("nan"))):
        return f"  {label:<44} n={st['n']}  (시드 부족 — 통계 불가)"
    mark = "**" if st["sig"] else "  "
    u = UNIT[metric]
    vals = ", ".join(f"{v:+.3f}" for v in st["vals"])
    return (f"  {label:<44} diff={st['mean']:+7.3f}{u}  sd={st['sd']:6.3f}  se={st['se']:6.3f}  "
            f"t={st['t']:+7.2f}  p={st['p']:.4f} {mark}   [seed: {vals}]")


# ---------------- Table 16: 유동성 경로 (zero-mean, k1) ramp↑ vs ramp↓ ----------------
def table16(metric):
    print(f"\n{'='*118}\n[Table 16] 유동성 경로 ramp_up vs ramp_down  (zero-mean, k=1)   metric={metric}")
    print(f"  H0: 두 경로의 {metric} 차이 = 0.  paired by seed.")
    print(f"{'='*118}")
    got = False
    for fold in FOLDS:
        c = _load(CACHE_ZM, fold)
        if not c:
            print(f"  [{LABELS[fold]}] 캐시 없음 → {CACHE_ZM}")
            continue
        got = True
        d = []
        for s, r in c.items():
            try:
                d.append(r["shape_metab"]["k1"]["ramp_up"][metric]
                         - r["shape_metab"]["k1"]["ramp_down"][metric])
            except (KeyError, TypeError):
                d.append(None)
        print(_row(LABELS[fold], paired(d, metric), metric))
    return got


# ---------------- Table 18: 금리 경로 (anchored, k1) continuous vs discrete ----------------
def table18(metric):
    print(f"\n{'='*118}\n[Table 18] 금리 경로 continuous(ramp_up) vs discrete(step_up)  (anchored, k=1)   metric={metric}")
    print(f"  H0: 점진 인상과 계단 인상의 {metric} 차이 = 0.  paired by seed.")
    print(f"{'='*118}")
    got = False
    for fold in FOLDS:
        c = _load(CACHE_AN, fold)
        if not c:
            print(f"  [{LABELS[fold]}] 캐시 없음 → {CACHE_AN}")
            continue
        got = True
        d = []
        for s, r in c.items():
            try:
                d.append(r["shape_tbill"]["k1"]["ramp_up"][metric]
                         - r["shape_tbill"]["k1"]["step_up"][metric])
            except (KeyError, TypeError):
                d.append(None)
        print(_row(LABELS[fold], paired(d, metric), metric))
    return got


# ---------------- Table 19: 조합경로 상호작용 (Diff*) ----------------
JOINT_KEY = {"ramp_up": "유동성 ramp↑", "ramp_down": "유동성 ramp↓",
             "step_up": "유동성 step↑", "step_down": "유동성 step↓"}


def table19(metric):
    """Diff* = (joint - flat) - [(rate_only - flat) + (liq_only - flat_zm)]

    joint/rate_only/flat = joint 캐시,  liq_only = zero-mean 캐시(Table 16 과 동일 셀).
    두 캐시의 flat 은 동일 계산(양축 앵커 flat, 동일 seed)이지만 각자 것을 쓴다.
    """
    print(f"\n{'='*118}\n[Table 19] 조합경로 상호작용 Diff* = joint - (rate 단독 + liquidity 단독)   metric={metric}")
    print(f"  H0: 상호작용 = 0 (금리·유동성 효과가 단순 가산).  paired by seed.")
    print(f"{'='*118}")
    got = False
    for fold in FOLDS:
        cj = _load(CACHE_JT, fold)
        cz = _load(CACHE_ZM, fold)
        if not cj or not cz:
            miss = CACHE_JT if not cj else CACHE_ZM
            print(f"  [{LABELS[fold]}] 캐시 없음 → {miss}")
            continue
        got = True
        for liq in ("ramp_up", "ramp_down"):
            d = []
            for s in SEEDS:
                if s not in cj or s not in cz:
                    continue
                try:
                    j = cj[s]
                    z = cz[s]
                    flat_j = j["flat"][metric]
                    flat_z = z["flat"][metric]
                    eff_joint = j["joint"][JOINT_KEY[liq]][metric] - flat_j
                    eff_rate = j["rate_only"][metric] - flat_j
                    eff_liq = z["shape_metab"]["k1"][liq][metric] - flat_z
                    d.append(eff_joint - (eff_rate + eff_liq))
                except (KeyError, TypeError):
                    d.append(None)
            print(_row(f"{LABELS[fold]}  |  uptrend x {liq}", paired(d, metric), metric))
    return got


def _dump_cache_status():
    print(f"\n{'='*118}\n[캐시 상태]  TAG_PREFIX={PS.TAG_PREFIX}  CACHE_SUFFIX='{SUF}'\n{'='*118}")
    for name, cd in (("zeromean(T16)", CACHE_ZM), ("anchored(T18)", CACHE_AN), ("joint(T19)", CACHE_JT)):
        if not os.path.isdir(cd):
            print(f"  {name:<16} [없음] {cd}")
            continue
        have = {f: sorted(_load(cd, f).keys()) for f in FOLDS}
        tot = sum(len(v) for v in have.values())
        print(f"  {name:<16} {tot}/{len(FOLDS)*len(SEEDS)} 파일  {cd}")
        for f, ss in have.items():
            print(f"      {f:<18} seeds={ss}")


def main():
    print("#" * 118)
    print("# 리뷰어1 #1 대응 — §4.3 경로효과의 *시드* 표준오차 (재추론 없이 캐시만 읽음)")
    print("#   n=5 시드, paired.  **=|t|>=t_{0.975,df} 로 5% 수준 유의.")
    print("#   주의: 이것은 시드(학습) 변동만이다.  원점 표본변동은 별도 재추론 필요(블록 부트스트랩).")
    print("#" * 118)
    _dump_cache_status()

    any_data = False
    for metric in METRICS:
        any_data |= table16(metric)
        any_data |= table18(metric)
        any_data |= table19(metric)

    if not any_data:
        print("\n[중단] 캐시를 하나도 찾지 못했다. RESULT_DIR 과 CACHE_SUFFIX 를 확인할 것:")
        print(f"        RESULT_DIR   = {RESULT_DIR}")
        print(f"        CACHE_SUFFIX = '{SUF}'  (analyze_pathshape_rawvol 의 모델 스위치에 따라 바뀜)")
        return

    print(f"\n{'='*118}")
    print("[해석 지침]")
    print("  · ** 없는 셀 = 5시드 변동만으로도 차이를 설명할 수 있다 → 논문 표의 볼드를 빼고")
    print("    '유의한 차이 없음'으로 서술해야 한다(리뷰어1 #1 에 정면으로 답하는 방식).")
    print("  · ** 있는 셀 = 시드변동을 넘어선다.  단 원점 성분이 남아 있어 최종 판정은 아니다.")
    print("  · 다음 단계: sim_metrics 를 원점별 IHL 반환으로 확장 → 13주 블록 부트스트랩")
    print("    (origin 이 주간이고 horizon 13주라 13배 중첩 — 리뷰어1 #2 와 같은 뿌리).")
    print("=" * 118)


if __name__ == "__main__":
    main()