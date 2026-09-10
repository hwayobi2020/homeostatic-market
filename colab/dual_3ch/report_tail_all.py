# -*- coding: utf-8 -*-
"""§4.1 꼬리위험 통합 리포트 — 네 모형, 같은 원점, 같은 지표축.

왜 이 파일인가
--------------
§4.1 의 검정이 CRPS 하나뿐이었다.  논문 주제는 꼬리위험(IHL·CVaR·VaR)이고
§4.2·§4.3 은 IHL 로 재는데 §4.1 만 다른 축이었다.  여기서 네 모형을 **같은
원점 집합**에 정렬해 꼬리 지표까지 함께 검정한다.

읽는 것 (전부 이미 저장된 배열/요약 — 학습 0 회)
------------------------------------------------
  MAC-Flow  : garch_flow_ar_{FLOW_TAG}_s{seed}_{fold}_best.pt  → 재샘플링
  CondVAE   : result/thin_vg_cache/vae_*.npz                   (run_thin_compare 캐시)
  CondGAN   : result/thin_vg_cache/gan_*.npz
  GARCH-ST  : result/{GARCH_PREFIX}_{fold}_arrays.npz          (MLE, 1 판)

검정
----
  원점별 손실 4 종 → DM(Diebold-Mariano, HAC lag 12) · 비중첩(stride 13) · 4 폴드 통합
    CRPS      : 원점별 13 주 평균 CRPS
    pinball1  : τ=0.01 핀볼 손실 (VaR 1% 의 proper scoring rule)
    pinball5  : τ=0.05
    IHL       : |모형 IHL CVaR10%(원점 내) − 실측 IHL|
  CVaR 은 단독으로 elicitable 이 아니라 DM 을 걸지 않는다.  대신 시드 t검정으로
  전역 CVaR 오차를 비교한다 (MAC-Flow·VAE·GAN 은 5 시드, GARCH 는 상수 1 판).

사용
----
    !python colab/dual_3ch/report_tail_all.py
    # GARCH 판 바꾸기:  GARCH_PREFIX=garch_xpast_matched python ...
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

os.environ.setdefault("PS_BASE", "fpath_novol")
os.environ.setdefault("FPATH_DIM", "2")
os.environ.setdefault("VG_EXTRA_COLS", "sp_skew_13w")
os.environ.setdefault("VG_FUTURE_UNMASK", "metab_13w")

from rawvol_helpers import patch_rawvol, ihl_paths                   # noqa: E402
patch_rawvol()

import analyze_pathshape_rawvol as PS                                # noqa: E402
import train_garch_flow as T                                         # noqa: E402
import dm_test as DM                                                 # noqa: E402
from scipy import stats                                              # noqa: E402

RESULT_DIR = PS.RESULT_DIR
FOLDS = [f for f in os.environ.get("RT_FOLDS", ",".join(PS.FOLDS)).split(",") if f]
SEEDS = [int(s) for s in os.environ.get(
    "RT_SEEDS", ",".join(str(s) for s in PS.SEEDS)).split(",") if s.strip()]
FLOW_TAG = os.environ.get("FLOW_TAG", PS.TAG_PREFIX)
PS.TAG_PREFIX = FLOW_TAG
GARCH_PREFIX = os.environ.get("GARCH_PREFIX", "garch_xpast_matched_fut")
VG_CACHE_DIR = os.path.join(RESULT_DIR, "thin_vg_cache")
STRIDE = 13
DM_LAG = 12
ORDER = ["MAC-Flow", "CondVAE", "CondGAN", "GARCH-ST"]


# ------------------------------------------------------------------ 손실
def crps_per_origin(sim, act):
    return np.array([np.mean([T.crps_ensemble_sample(sim[i, :, w], act[i, w])
                              for w in range(sim.shape[2])])
                     for i in range(sim.shape[0])], float)


def pinball_per_origin(sim, act, tau):
    """τ 분위 핀볼 손실.  원점별로 13 주 평균.  VaR 의 proper scoring rule."""
    q = np.quantile(sim, tau, axis=1)                    # (n_orig, T)
    d = act - q
    return np.mean(np.where(d >= 0, tau * d, (tau - 1.0) * d), axis=1)


def ihl_loss_per_origin(sim, act, alpha=0.10):
    """|모형 IHL CVaRα(원점 내 n_sim) − 실측 IHL|."""
    s = ihl_paths(sim)                                   # (n_orig, n_sim)
    a = ihl_paths(act)                                   # (n_orig,)
    k = max(1, int(alpha * s.shape[1]))
    return np.abs(np.sort(s, axis=1)[:, :k].mean(axis=1) - a)


LOSSES = [("CRPS", crps_per_origin),
          ("pinball1%", lambda s, a: pinball_per_origin(s, a, 0.01)),
          ("pinball5%", lambda s, a: pinball_per_origin(s, a, 0.05)),
          ("IHL", ihl_loss_per_origin)]


def global_metrics(sim, act):
    """시드 t검정용 전역 지표 (0 에 가까울수록 좋은 오차로 통일)."""
    from train_garch_x import cvar as _cv, var_q as _vq
    sf, af = sim.ravel(), act.ravel()
    s_ihl, a_ihl = ihl_paths(sim).ravel(), ihl_paths(act).ravel()

    def _cv10(x):
        y = np.sort(x)
        return float(y[:max(1, int(0.10 * y.size))].mean())

    return {"CVaR1%오차": abs(_cv(sf, .01) - _cv(af, .01)),
            "CVaR5%오차": abs(_cv(sf, .05) - _cv(af, .05)),
            "VaR1%오차": abs(_vq(sf, .01) - _vq(af, .01)),
            "IHL평균오차": abs(s_ihl.mean() - a_ihl.mean()),
            "IHL CVaR10오차": abs(_cv10(s_ihl) - _cv10(a_ihl))}


# ------------------------------------------------------------------ 적재
_RT = None
_DEV = None


def load_flow(fold, seed):
    global _RT, _DEV
    if _RT is None:
        import torch
        import run_thin_compare as RT                                # noqa: E402
        _RT, _DEV = RT, ("cuda" if torch.cuda.is_available() else "cpu")
    return _RT.macflow_arrays(fold, seed, _DEV)


def load_vg(fold, seed, kind):
    spec = {"vae": "c128h192_lr0d0001", "gan": "c192h128_lr0d0003"}[kind]
    p = os.path.join(VG_CACHE_DIR, f"{kind}_t3skfu_{spec}_s{seed}_{fold}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return dict(sim=d["sim"], act=d["act"], pred_start=d["pred_start"])


def load_garch(fold):
    p = os.path.join(RESULT_DIR, f"{GARCH_PREFIX}_{fold}_arrays.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return dict(sim=d["sim"], act=d["act"], pred_start=d["pred_start"])


def collect(fold):
    """모형 → 시드별 배열 목록.  GARCH 는 길이 1."""
    runs = {}
    fl = []
    for s in SEEDS:
        try:
            fl.append(load_flow(fold, s))
        except Exception as e:                                        # noqa: BLE001
            print(f"  [MAC-Flow s{s}] {e!r}")
    if fl:
        runs["MAC-Flow"] = fl
    for name, kind in (("CondVAE", "vae"), ("CondGAN", "gan")):
        got = [d for d in (load_vg(fold, s, kind) for s in SEEDS) if d]
        if got:
            runs[name] = got
    g = load_garch(fold)
    if g:
        runs["GARCH-ST"] = [g]
    return runs


def align(runs):
    """전 모형·전 시드의 pred_start 교집합으로 정렬."""
    common = None
    for lst in runs.values():
        for d in lst:
            s = set(int(x) for x in d["pred_start"])
            common = s if common is None else (common & s)
    if not common:
        return None, None
    order = np.array(sorted(common), int)
    out = {}
    for name, lst in runs.items():
        arrs = []
        for d in lst:
            pos = {int(v): i for i, v in enumerate(np.asarray(d["pred_start"], int))}
            sel = np.array([pos[v] for v in order], int)
            arrs.append((np.asarray(d["sim"])[sel], np.asarray(d["act"])[sel]))
        out[name] = arrs
    return out, order


# ------------------------------------------------------------------ 본체
def main():
    print("#" * 112)
    print("# §4.1 꼬리위험 통합 — 네 모형, 같은 원점, 학습 0 회")
    print(f"#  MAC-Flow={FLOW_TAG}  GARCH={GARCH_PREFIX}  seeds={SEEDS}")
    print(f"#  손실: {[n for n, _ in LOSSES]}  ·  DM HAC lag={DM_LAG} · 비중첩 stride={STRIDE}")
    print("#" * 112)

    res = {"dm": {}, "thin": {}, "pooled": {}, "seed_t": {}}
    pooled = {n: {m: [] for m in ORDER} for n, _ in LOSSES}
    gm_store = {m: {} for m in ORDER}

    for fold in FOLDS:
        print(f"\n===== {fold}")
        runs = collect(fold)
        if "MAC-Flow" not in runs:
            print("  [skip] MAC-Flow 없음")
            continue
        aligned, order = align(runs)
        if aligned is None:
            print("  [skip] 원점 교집합 0")
            continue
        print("  원점 " + str(len(order)) + " 개  " +
              "  ".join(f"{k}:{len(v)}시드" for k, v in aligned.items()))

        # 원점별 손실 (시드 평균)
        L = {}
        for lname, fn in LOSSES:
            L[lname] = {m: np.mean([fn(s, a) for s, a in aligned[m]], axis=0)
                        for m in aligned}
            for m in aligned:
                pooled[lname][m].append(L[lname][m])

        # 전역 지표 (시드별로 남겨 t검정에 쓴다)
        for m in aligned:
            gm_store[m][fold] = [global_metrics(s, a) for s, a in aligned[m]]

        res["dm"][fold], res["thin"][fold] = {}, {}
        for lname, _ in LOSSES:
            base = L[lname]["MAC-Flow"]
            for m in ORDER[1:]:
                if m not in L[lname]:
                    continue
                key = f"{m}|{lname}"
                res["dm"][fold][key] = DM.dm_test(L[lname][m], base, DM_LAG)
                per_off = []
                for off in range(STRIDE):
                    idx = np.arange(off, len(base), STRIDE)
                    if idx.size >= 8:
                        per_off.append(DM.dm_test(L[lname][m][idx], base[idx], 0))
                if per_off:
                    dm_ = [r["dm_hln"] for r in per_off]
                    p_ = [r["p_two_sided"] for r in per_off]
                    res["thin"][fold][key] = dict(
                        n_offset=len(per_off), n_per_offset=int(per_off[0]["n"]),
                        d_mean=float(np.mean([r["d_mean"] for r in per_off])),
                        dm_mean=float(np.mean(dm_)), p_median=float(np.median(p_)),
                        n_sig05=int(sum(1 for x in p_ if x < .05)),
                        n_pos=int(sum(1 for r in per_off if r["d_mean"] > 0)))

    if not res["dm"]:
        print("\n비교할 자료가 없다.")
        return

    # ---------------- 출력 1: 폴드별 DM ----------------
    w = 112
    print("\n" + "=" * w)
    print(f"[DM 검정] 원점 시간순 · HAC lag={DM_LAG} · HLN 보정 · 양측")
    print("  mean diff > 0 이면 MAC-Flow 의 손실이 낮다 = 우위")
    print("=" * w)
    print(f"{'fold':<17}{'손실':<11}{'비교':<12}{'n':>5}{'mean diff':>12}"
          f"{'HAC se':>11}{'DM(HLN)':>10}{'p':>9}")
    for fold in res["dm"]:
        for key, r in res["dm"][fold].items():
            m, lname = key.split("|")
            print(f"{fold:<17}{lname:<11}{'vs ' + m:<12}{r['n']:>5}"
                  f"{r['d_mean']:>12.6f}{r['se']:>11.6f}"
                  f"{r['dm_hln']:>10.3f}{r['p_two_sided']:>9.4f}")
        print()

    # ---------------- 출력 2: 비중첩 ----------------
    print("=" * w)
    print(f"[비중첩] 원점 {STRIDE} 간격 · 오프셋 {STRIDE} 개 각각 (HAC lag=0)")
    print("=" * w)
    print(f"{'fold':<17}{'손실':<11}{'비교':<12}{'n/오프셋':>9}{'mean diff':>12}"
          f"{'DM 평균':>10}{'p 중앙':>9}{'p<.05':>7}{'양수':>7}")
    for fold in res["thin"]:
        for key, r in res["thin"][fold].items():
            m, lname = key.split("|")
            print(f"{fold:<17}{lname:<11}{'vs ' + m:<12}{r['n_per_offset']:>9}"
                  f"{r['d_mean']:>12.6f}{r['dm_mean']:>10.3f}{r['p_median']:>9.4f}"
                  f"{str(r['n_sig05']) + '/' + str(r['n_offset']):>7}"
                  f"{str(r['n_pos']) + '/' + str(r['n_offset']):>7}")
        print()

    # ---------------- 출력 3: 4 폴드 통합 ----------------
    print("=" * w)
    print(f"[4 폴드 통합] 폴드를 이어붙여 한 번 · HAC lag={DM_LAG}")
    print("=" * w)
    print(f"{'손실':<11}{'비교':<12}{'n':>6}{'mean diff':>12}{'HAC se':>11}"
          f"{'DM(HLN)':>10}{'p':>9}")
    for lname, _ in LOSSES:
        base = pooled[lname]["MAC-Flow"]
        if not base:
            continue
        b = np.concatenate(base)
        for m in ORDER[1:]:
            if not pooled[lname][m] or len(pooled[lname][m]) != len(base):
                continue
            r = DM.dm_test(np.concatenate(pooled[lname][m]), b, DM_LAG)
            res["pooled"][f"{m}|{lname}"] = r
            print(f"{lname:<11}{'vs ' + m:<12}{r['n']:>6}{r['d_mean']:>12.6f}"
                  f"{r['se']:>11.6f}{r['dm_hln']:>10.3f}{r['p_two_sided']:>9.4f}")
        print()

    # ---------------- 출력 4: 시드 t검정 (CVaR 포함) ----------------
    print("=" * w)
    print("[시드 t검정 · 4 폴드 통합] 목표 대비 |오차| 를 시드별 4 폴드 평균 → 검정")
    print("  베이스라인이 5 시드면 Welch, GARCH(1 판)면 그 상수에 대한 1 표본")
    print("=" * w)
    keys = list(global_metrics(np.zeros((1, 2, 2)), np.zeros((1, 2))).keys())
    print(f"{'지표':<16}{'MAC-Flow':>20}{'비교':<12}{'베이스라인':>20}{'t':>8}{'p':>9}")
    for k in keys:
        def seed_means(model):
            n = min((len(gm_store[model][f]) for f in gm_store[model]), default=0)
            if not n or len(gm_store[model]) != len(res["dm"]):
                return None
            return np.array([np.mean([gm_store[model][f][i][k]
                                      for f in gm_store[model]])
                             for i in range(n)], float)
        X = seed_means("MAC-Flow")
        if X is None:
            continue
        for m in ORDER[1:]:
            Y = seed_means(m)
            if Y is None:
                continue
            if Y.size == 1:
                t, p = stats.ttest_1samp(X, Y[0]); btxt = f"{Y[0]:>20.6f}"
            else:
                t, p = stats.ttest_ind(X, Y, equal_var=False)
                btxt = f"{Y.mean():>12.6f}±{Y.std(ddof=1) / np.sqrt(Y.size):.6f}"
            mark = "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""
            res["seed_t"][f"{m}|{k}"] = dict(mac=float(X.mean()),
                                             base=float(Y.mean()),
                                             t=float(t), p=float(p))
            print(f"{k:<16}{X.mean():>12.6f}±{X.std(ddof=1) / np.sqrt(X.size):.6f}"
                  f"{'vs ' + m:<12}{btxt}{t:>8.2f}{p:>9.4f} {mark}")
        print()
    print("  *** p<.01  ** p<.05  * p<.10")
    print("  CVaR 은 단독 elicitable 이 아니라 DM 이 아니라 이 표로만 본다.")

    out = os.path.join(RESULT_DIR, f"tail_report_{FLOW_TAG}_{GARCH_PREFIX}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
