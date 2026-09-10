# -*- coding: utf-8 -*-
"""GARCH-X(past)-skewt, 입력 채널을 MAC-Flow 와 맞춘 판 — §4.1.2 공정성 점검.

`train_garch_xpast.py` 는 분산식에 tbill·metab 두 개만 받는데, MAC-Flow 는
과거 52 주 5 채널(sp_return·tbill_wr·ads_lag·wti_wr·metab_13w)에 원점 고정
스칼라 sp_skew_13w 까지 받는다.  리뷰어 4 #5a("GARCH-ST 는 단변량인데
MAC-Flow 는 5채널이라 불공정")에 정면으로 답하려면 기준선에도 같은 채널을
줘야 한다.

바뀐 것은 **외생 채널 수뿐**이다.

    var : σ²_t = (ω + α·ε²_{t-1} + β·σ²_{t-1}) · exp(Σ_k γ_k · x_k,t)
          x = [tbill_wr, metab_13w, ads_lag, wti_wr, sp_skew_13w]  (train 통계 표준화)

`sp_skew_13w` 도 **분산식의 외생항으로만** 넣는다.  Hansen skew-t 의 왜도 모수
λ 는 그대로 상수다 — λ 를 조건부로 만들면 GARCH 에 조건부 왜도를 주는 것이라
모형 부류 자체가 바뀐다.  여기서 맞추는 것은 *정보*지 *구조*가 아니다.

맞출 수 없는 것: MAC-Flow 는 52 주 경로를 받는데 GARCH 는 원점 값을 13 주 내내
고정한다(미래 거시 미주입).  과거 요약 방식의 차이는 남으며, 그것이 두 모형
부류의 차이다.

모수 8 개 → 11 개 (γ 가 2 → 5).
결과: `garch_xpast_matched_{fold}_summary.json` + per-origin CRPS/날짜 npy.
기존 `garch_xpast_*` / `garch_xpast_refit_*` 파일은 건드리지 않는다.

사용
----
    !python colab/dual_3ch/train_garch_xpast_matched.py
    !GARCH_PREFIX=garch_xpast_matched FLOW_TAG=rvAbl_full_fpath_novol_d2 \
        python colab/dual_3ch/dm_garch_compare.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from train_garch_xpast import (                                    # noqa: E402
    build, hansen_sample, FOLDS, FOLDS_DIR, RESULT_DIR,
    PAST_LEN, FUT, N_SIM, SEED,
)
from train_garch_x import crps_pooled, crps_ensemble, cvar, var_q, emd1d  # noqa: E402

# 분산식 외생 채널.  MAC-Flow 의 과거 채널 + 원점 고정 스칼라와 같은 집합이다
# (sp_return 은 y 자체라 제외, sp_std_13w 는 GARCH 가 σ 재귀로 내생 추정한다).
XCOLS = ["tbill_wr", "metab_13w", "ads_lag", "wti_wr", "sp_skew_13w"]
NG = len(XCOLS)                                   # γ 개수
# GX_FUTURE=1 : 시뮬레이션에 미래 실현 거시 *경로* 를 매 스텝 주입한다.
#   기본 0 은 원점 값 13 주 고정(=§4.1.2 의 정보 일치 설정).
#   1 은 MAC-Flow 가 fpath 요약으로 받는 미래 경로를 기준선에도 주는 판이며,
#   결과 파일은 `_fut` 접미사로 분리해 기존 산출물과 섞이지 않게 한다.
USE_FUTURE = os.environ.get("GX_FUTURE", "0") == "1"
TAGSFX = "_fut" if USE_FUTURE else ""
FIT_SCALE = 100.0                                 # 퍼센트 스케일 (refit 과 동일)
OPTS = dict(maxiter=5000, maxfun=100000, ftol=1e-14, gtol=1e-12)
LAM0_GRID = [-0.4, -0.2, 0.0, 0.2]                # λ 다중 출발 (refit 과 동일)


def _skew13(dfs):
    """13 주 rolling skew (date → 값).  **`.shift(1)` 을 건다.**

    `rawstd_preprocess_fold` 는 shift 없이 계산하지만, 거기서는 MAC-Flow 가 이
    값을 *원점 행 하나*만 뽑아 13 주 내내 고정해 쓰므로 r_origin 이 포함돼도
    관측된 값이라 문제가 없다.

    여기서는 다르다.  GX_FUTURE=1 이면 t+1..t+13 행의 값을 매 스텝 조건으로
    넣는데, shift 가 없으면 t+h 행의 왜도가 **예측 대상인 r_{t+h} 를 포함**한다
    — 정답을 보고 예측하는 셈이다.  `sp_std_13w` 는 원래부터
    `rolling(13).std(ddof=1).shift(1)` 이라(`data/extend_to_1971.py:396`) 왜도만
    빠져 있던 것이므로, 같은 규약으로 맞춘다.
    """
    full = (pd.concat(list(dfs.values()), ignore_index=True)
              .drop_duplicates("date").sort_values("date").reset_index(drop=True))
    sk = (full["sp_return"].astype(float)
          .rolling(window=13, min_periods=13).skew().shift(1).fillna(0.0))
    return dict(zip(full["date"], sk.to_numpy(dtype=float)))


def build_x(df, skew_map):
    """(n, NG) 외생 행렬.  없는 컬럼은 즉시 중단한다 (조용한 0 채움 금지)."""
    cols = []
    for c in XCOLS:
        if c == "sp_skew_13w":
            v = df["date"].map(skew_map).to_numpy(dtype=float)
        elif c == "metab_13w":
            _, _, v = build(df)                   # get_metab13 재사용
        else:
            if c not in df.columns:
                sys.exit(f"[FATAL] fold CSV 에 {c} 없음")
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        cols.append(v)
    return np.column_stack(cols)


# ---------- 모형 (train_garch_xpast 와 같되 γ 가 벡터) ----------
def _exp_arg(g, X_t):
    return float(np.clip(np.dot(g, X_t), -10.0, 10.0))


def _unpack(p):
    mu, om, al, be = p[0], p[1], p[2], p[3]
    g = np.asarray(p[4:4 + NG], dtype=float)
    nu, lam = p[4 + NG], p[5 + NG]
    return mu, om, al, be, g, nu, lam


def filter_var(p, y, X):
    mu, om, al, be, g, nu, lam = _unpack(p)
    eps = y - mu
    n = len(y)
    s2 = np.empty(n)
    fin = eps[np.isfinite(eps)]
    s2[0] = float(np.var(fin)) if fin.size else 1.0
    Xc = np.nan_to_num(X, nan=0.0)
    for t in range(1, n):
        prev = s2[t - 1] if np.isfinite(s2[t - 1]) else s2[0]
        e2 = eps[t - 1] ** 2 if np.isfinite(eps[t - 1]) else prev
        base = om + al * e2 + be * prev
        s2[t] = max(base * np.exp(_exp_arg(g, Xc[t])), 1e-12)
    return s2, eps


def neg_ll(p, y, X):
    from train_garch_xpast import hansen_logpdf
    mu, om, al, be, g, nu, lam = _unpack(p)
    s2, eps = filter_var(p, y, X)
    eta = eps / np.sqrt(s2)
    ll = hansen_logpdf(eta, nu, lam) - 0.5 * np.log(s2)
    if not np.all(np.isfinite(ll)):
        return 1e12
    return -float(np.sum(ll))


def fit(y, X):
    """퍼센트 스케일 + 조인 종료조건 + λ 다중 출발 (refit_garch_xpast 와 같은 처방).

    기존 적합이 L-BFGS-B 조기 종료로 초기값에 머문 문제(리뷰어 1 #4)를 피한다.
    """
    ys = y * FIT_SCALE
    v0 = float(np.var(ys))
    bnds = ([(-5, 5), (1e-10, None), (1e-6, 0.5), (1e-6, 0.999)]
            + [(-1.5, 1.5)] * NG + [(2.1, 60.0), (-0.95, 0.95)])
    best = None
    for lam0 in LAM0_GRID:
        p0 = ([float(np.mean(ys)), 0.05 * v0, 0.08, 0.88]
              + [0.0] * NG + [7.0, lam0])
        r = minimize(neg_ll, p0, args=(ys, X), method="L-BFGS-B",
                     bounds=bnds, options=OPTS)
        if best is None or r.fun < best.fun:
            best = r
    p = np.array(best.x, dtype=float)
    p[0] /= FIT_SCALE                                  # mu
    p[1] /= FIT_SCALE ** 2                             # omega
    return p, float(best.fun), int(best.nit)


def simulate(p, eps_orig, s2_orig, X_path, n_sim, rng):
    """origin 에서 13 주 forward.

    `X_path` 가 (NG,) 면 원점 값을 13 주 내내 고정한다 (GX_FUTURE=0, 기본).
    (FUT, NG) 면 **미래 실현 거시 경로**를 매 스텝 주입한다 (GX_FUTURE=1) —
    MAC-Flow 가 fpath 요약으로 받는 그 경로를 GARCH 에도 주는 판이다.
    GARCH 는 경로 전체를 한 번에 요약하지 못하고 한 스텝씩만 받을 수 있으므로,
    이것이 이 모형 부류가 미래 경로를 쓸 수 있는 최대치다.
    """
    mu, om, al, be, g, nu, lam = _unpack(p)
    Xp = np.atleast_2d(np.nan_to_num(np.asarray(X_path, float), nan=0.0))
    if Xp.shape[0] == 1:
        Xp = np.repeat(Xp, FUT, axis=0)                 # 원점 고정
    if Xp.shape != (FUT, NG):
        raise ValueError(f"X_path shape {Xp.shape} != {(FUT, NG)} or {(NG,)}")
    s2 = np.full(n_sim, s2_orig)
    e2prev = np.full(n_sim, eps_orig ** 2)
    out = np.empty((n_sim, FUT))
    for h in range(FUT):
        macro = float(np.exp(_exp_arg(g, Xp[h])))
        s2 = np.maximum((om + al * e2prev + be * s2) * macro, 1e-12)
        eps = np.sqrt(s2) * hansen_sample(nu, lam, n_sim, rng)
        out[:, h] = mu + eps
        e2prev = eps ** 2
    return out


def _sk(a):
    a = np.asarray(a, float)
    return float(np.mean(((a - a.mean()) / (a.std() + 1e-12)) ** 3))


def _ek(a):
    a = np.asarray(a, float)
    return float(np.mean(((a - a.mean()) / (a.std() + 1e-12)) ** 4) - 3.0)


def run_fold(fold):
    sp = os.path.join(RESULT_DIR, f"garch_xpast_matched{TAGSFX}_{fold}_summary.json")
    if os.path.exists(sp):
        print(f"[skip] {os.path.basename(sp)}")
        return
    paths = {s: os.path.join(FOLDS_DIR, f"{fold}_{s}.csv")
             for s in ("train", "val", "test")}
    dfs = {s: pd.read_csv(p, parse_dates=["date"])
           for s, p in paths.items() if os.path.exists(p)}
    if "train" not in dfs or "test" not in dfs:
        print(f"[skip {fold}] csv 없음")
        return
    skew_map = _skew13(dfs)

    tr, te = dfs["train"], dfs["test"]
    ytr, _, _ = build(tr)
    yte, _, _ = build(te)
    Xtr_raw, Xte_raw = build_x(tr, skew_map), build_x(te, skew_map)

    # train 통계로 표준화 (test 통계 미사용 — 누수 방지)
    mu_x = np.nanmean(Xtr_raw, axis=0)
    sd_x = np.nanstd(Xtr_raw, axis=0) + 1e-12
    Xtr = (Xtr_raw - mu_x) / sd_x
    Xte = (Xte_raw - mu_x) / sd_x

    m = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
    print(f"\n{'=' * 70}\n fold={fold}  train n={int(m.sum())}  test n={len(yte)}")
    p, nll, nit = fit(ytr[m], Xtr[m])
    mu, om, al, be, g, nu, lam = _unpack(p)
    gtxt = "  ".join(f"{c}={v:+.4f}" for c, v in zip(XCOLS, g))
    print(f"  neg_ll={nll:.4f} nit={nit}")
    print(f"  mu={mu:+.6f} om={om:.3e} al={al:.4f} be={be:.4f} nu={nu:.2f} lam={lam:+.5f}")
    print(f"  gamma: {gtxt}")

    s2_te, eps_te = filter_var(p, yte, Xte)
    rng = np.random.default_rng(SEED)
    origins = [t for t in range(PAST_LEN, len(yte) - FUT)
               if np.isfinite(yte[t]) and np.isfinite(s2_te[t])
               and np.all(np.isfinite(yte[t + 1:t + 1 + FUT]))]
    sim = np.empty((len(origins), N_SIM, FUT), dtype=np.float32)
    act = np.empty((len(origins), FUT), dtype=np.float32)
    for i, t in enumerate(origins):
        # GX_FUTURE=1 이면 t+1..t+FUT 의 실현 거시 경로를 주입한다.
        Xarg = Xte[t + 1: t + 1 + FUT] if USE_FUTURE else Xte[t]
        sim[i] = simulate(p, eps_te[t], s2_te[t], Xarg, N_SIM, rng)
        act[i] = yte[t + 1:t + 1 + FUT]

    crps_m, crps_s = crps_pooled(sim, act)
    af, sf = act.ravel(), sim.ravel()
    std_a, std_s = float(af.std(ddof=1)), float(sf.std(ddof=1))
    cov = {}
    for lvl, lo, hi in [(50, 25, 75), (80, 10, 90), (95, 2.5, 97.5)]:
        L, H = np.percentile(sf, lo), np.percentile(sf, hi)
        cov[lvl] = float(((af >= L) & (af <= H)).mean())

    # DM 검정용 per-origin CRPS + 조건 시점 (garch_xpast_refit 과 같은 규약)
    per_oc = np.array([[crps_ensemble(sim[i, :, w], act[i, w]) for w in range(FUT)]
                       for i in range(len(origins))])
    dates = np.array([str(te["date"].values[t]) for t in origins])
    pref = os.path.join(RESULT_DIR, f"garch_xpast_matched{TAGSFX}_{fold}")
    np.save(f"{pref}_crps_per_origin.npy", per_oc)
    np.save(f"{pref}_origin_dates.npy", dates)

    print(f"  origins={len(origins)}  CRPS={crps_m:.5f}  "
          f"std ratio={std_s / std_a:.3f}  "
          f"cov 50/80/95={cov[50]:.3f}/{cov[80]:.3f}/{cov[95]:.3f}")
    print(f"  skew a/s={_sk(af):+.4f}/{_sk(sf):+.4f}  "
          f"exkurt a/s={_ek(af):+.3f}/{_ek(sf):+.3f}")

    summ = dict(
        fold=fold,
        model=f"GARCH-X(past)-skewt [MAC-Flow 채널 일치: {', '.join(XCOLS)}]",
        xcols=XCOLS, n_params=int(len(p)), neg_ll=nll, n_iter=nit,
        params=dict(mu=mu, omega=om, alpha=al, beta=be, nu=nu, lambda_skew=lam,
                    **{f"gamma_{c}": float(v) for c, v in zip(XCOLS, g)}),
        n_origins=len(origins), seed=SEED,
        test_eval=dict(crps_pooled=crps_m, crps_std=crps_s, emd=emd1d(sf, af),
                       std_actual=std_a, std_sim=std_s, std_ratio=std_s / std_a,
                       coverage_50=cov[50], coverage_80=cov[80], coverage_95=cov[95],
                       cvar_5pct_diff=cvar(sf, .05) - cvar(af, .05),
                       cvar_1pct_diff=cvar(sf, .01) - cvar(af, .01),
                       var_1pct_diff=var_q(sf, .01) - var_q(af, .01),
                       skew_actual=_sk(af), skew_sim=_sk(sf),
                       exkurt_actual=_ek(af), exkurt_sim=_ek(sf)))
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(summ, fh, indent=2, default=str)
    print(f"  saved {os.path.basename(sp)}")


def main():
    print("#" * 92)
    print("# GARCH-X(past)-skewt — 입력 채널을 MAC-Flow 와 일치 (리뷰어 4 #5a)")
    print(f"#  분산식 외생: {XCOLS}   모수 {4 + NG + 2} 개")
    print("#  λ 는 그대로 상수다 (정보를 맞추되 구조는 안 바꾼다)")
    print("#  미래 거시 = " + ("실현 경로 매 스텝 주입 (GX_FUTURE=1)"
                              if USE_FUTURE else "원점 값 13 주 고정 (기본)"))
    print("#" * 92)
    for fold in FOLDS:
        try:
            run_fold(fold)
        except Exception as e:                                     # noqa: BLE001
            print(f"  [FAIL] {fold}: {e!r}")
    print("\n[done] garch_xpast_matched_*_summary.json")


if __name__ == "__main__":
    main()
