# -*- coding: utf-8 -*-
"""§4.2 수준 연관을 회귀 계수로 — 금리·유동성(·경기·유가) 수준과 IHL 꼬리의 조건부 연관, 변수별 상대 비중.

무엇을
------
원점마다 실현된 13 주 금리 수준(연율 %, 13 주 평균)과 유동성 수준(metab_13w, 13 주 평균 %),
그리고 LEVEL_EXTRA(기본 ads_lag, wti_wr — 모형에서는 미래가 마스킹된 채널) 의 13 주 평균을 설명변수로,
IHL 을 종속변수로 놓고 **expectile 회귀**(Newey–Powell, 비대칭 최소제곱)를 건다.
이것은 미래 경로를 조건으로 준 효과가 아니라 *실현된 수준과 꼬리의 조건부 연관*(상관 분석)이다.
경로 형태의 조건부 효과는 §4.3 의 주입 실험이 맡는다.  모형 쪽 ads·wti 계수는 미래 조건화가 아니라
과거값을 통한 간접 반영의 크기다 (재학습 없이 변수별 기여를 같은 단위로 비교하려는 목적).
  · 실측 : 원점당 실현 IHL 1 개 (n ≈ 181/폴드)
  · 모형 : 원점당 MAC-Flow 경로 n_sim 개 (실현 거시 경로로 조건화, §4.1 과 같은 배열) 를 풀링
계수 = "금리 1%p 당 IHL 꼬리 %p", "유동성 1% 당 IHL 꼬리 %p".  두 변수를 같이 넣으므로 서로 통제된 값이고,
표준화 계수(설명변수 1 SD 당)도 같이 내서 상대 비중을 읽는다.  τ = 0.5(평균, OLS 와 동일) / 0.25 / 0.10 / 0.05.

통제 : 원점의 실현 변동성 수준(garch_sigma, 연율 %) 을 넣은 버전도 같이 낸다 (LEVEL_CTRL=1 기본).
  수준 구간이 국면과 얽혀 있으므로, 통제 없는 계수 = 연관 크기, 통제 있는 계수 = 상태 주어진 부분 연관.

불확실성
--------
  · 실측 : 원점 이동블록 부트스트랩 (블록 13 주, B 회) — 인접 원점의 12 주 겹침을 흡수
  · 모형 : 시드 5 개 SD (재학습 불확실성) + 시드별 원점 블록 부트스트랩 SE 의 평균
  · 폴드 통합 : 폴드 더미를 넣고 폴드 안에서 블록 부트스트랩

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 python colab/dual_3ch/level_regression_4_2.py
    # LEVEL_B=300 (부트스트랩 횟수)  LEVEL_TAUS=0.5,0.25,0.1,0.05  LEVEL_CTRL=0/1
"""
import csv
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_pathshape_rawvol as PS                         # noqa: E402
import train_garch_flow as T                                  # noqa: E402
import report_tail_all as RT                                  # noqa: E402
from rawvol_helpers import ihl_paths                          # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FOLDS = PS.FOLDS
SEEDS = PS.SEEDS
LABELS = {"F_gfc": "Financial crisis", "F_long_A": "Recovery",
          "F_long_B_origin": "COVID", "F_long": "Tightening"}
TAUS = [float(x) for x in os.environ.get("LEVEL_TAUS", "0.5,0.25,0.1,0.05").split(",")]
B = int(os.environ.get("LEVEL_B", "300"))
CTRL = os.environ.get("LEVEL_CTRL", "1") == "1"
EXTRA = [v for v in os.environ.get("LEVEL_EXTRA", "ads_lag,wti_wr").split(",") if v]   # 추가 설명변수 (13주 평균, CSV 원단위)
BLOCK = 13
PP = 100.0
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ expectile 회귀
def expectile_fit(X, y, tau, w=None, iters=60, tol=1e-9):
    """비대칭 최소제곱: min Σ |τ − 1(y<Xβ)| (y−Xβ)².  w = 관측 가중 (선택)."""
    n, k = X.shape
    w = np.ones(n) if w is None else w
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        a = np.where(r < 0, 1.0 - tau, tau) * w
        Xa = X * a[:, None]
        new = np.linalg.solve(X.T @ Xa + 1e-12 * np.eye(k), Xa.T @ y)
        if np.max(np.abs(new - beta)) < tol:
            beta = new; break
        beta = new
    return beta


def block_boot(X, y, tau, groups, w=None, B=B):
    """원점 이동블록 부트스트랩.  groups = 각 행의 원점 index (0..n_orig−1, 시간순).
    폴드가 여럿이면 fold_of 로 폴드 안에서만 블록을 뽑도록 groups 를 폴드별로 나눠 부른다."""
    n_orig = groups.max() + 1
    rows_of = [np.where(groups == g)[0] for g in range(n_orig)]
    starts_max = max(1, n_orig - BLOCK + 1)
    n_blocks = int(math.ceil(n_orig / BLOCK))
    out = []
    for _ in range(B):
        st = RNG.integers(0, starts_max, size=n_blocks)
        sel_o = np.concatenate([np.arange(s, min(s + BLOCK, n_orig)) for s in st])[:n_orig]
        idx = np.concatenate([rows_of[o] for o in sel_o])
        out.append(expectile_fit(X[idx], y[idx], tau, None if w is None else w[idx]))
    return np.std(np.asarray(out), axis=0, ddof=1)


def _aloss(X, y, tau, w=None):
    """비대칭 제곱손실 합 (expectile 의 목적함수).  절편만 있는 모형과의 비율로 의사-R² 를 만든다."""
    b = expectile_fit(X, y, tau, w)
    r = y - X @ b
    a = np.where(r < 0, 1.0 - tau, tau) * (np.ones_like(r) if w is None else w)
    return float(np.sum(a * r * r))


def shapley_share(X, y, tau, var_idx, w=None):
    """설명력(의사-R² = 1 − 손실/절편손실) 을 변수별 Shapley(LMG) 로 분해 → 각 변수의 몫(합 = 전체 의사-R²).
    var_idx: 분해할 열 index 목록 (절편·폴드더미 제외).  부분집합 2^k 개 적합."""
    from itertools import combinations
    k = len(var_idx)
    base_cols = [j for j in range(X.shape[1]) if j not in var_idx]     # 절편(+더미) 는 항상 포함
    L0 = _aloss(X[:, base_cols], y, tau, w)
    cache = {}
    def loss(S):
        key = tuple(sorted(S))
        if key not in cache:
            cols = base_cols + [var_idx[i] for i in key]
            cache[key] = _aloss(X[:, cols], y, tau, w)
        return cache[key]
    shares = np.zeros(k)
    for i in range(k):
        others = [j for j in range(k) if j != i]
        for r in range(k):
            for S in combinations(others, r):
                wgt = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
                shares[i] += wgt * (loss(S) - loss(S + (i,))) / L0
    return shares, 1.0 - loss(tuple(range(k))) / L0


# ------------------------------------------------------------------ 데이터
def fold_frame(fold):
    """test CSV → date, tbill(연율%), metab(13주 %), vol(연율%) per row."""
    gp = T.garch_preprocess_fold(PS.FOLDS_DIR, fold, PS.RESULT_DIR)
    df = pd.read_csv(gp["test"])
    tb_ann = ((1.0 + df["tbill_wr"].to_numpy(float)) ** 52 - 1.0) * 100.0
    mb = df["metab_13w"].to_numpy(float) * 100.0
    vol = df["garch_sigma"].to_numpy(float) * math.sqrt(52) * 100.0 if "garch_sigma" in df.columns else None
    dates = df["date"].astype(str).to_numpy() if "date" in df.columns else np.array([str(i) for i in range(len(df))])
    extra = {c: df[c].to_numpy(float) for c in EXTRA if c in df.columns}
    return dict(tb=tb_ann, mb=mb, vol=vol, dates=dates, extra=extra)


def origin_regressors(fr, pred_start):
    """원점별 미래 13 주 평균 수준 (실현).  vol 은 원점 직전 행 (조건 시점) 의 값."""
    tb = np.array([fr["tb"][p:p + T.FUTURE_LEN].mean() for p in pred_start])
    mb = np.array([fr["mb"][p:p + T.FUTURE_LEN].mean() for p in pred_start])
    vol = np.array([fr["vol"][p - 1] for p in pred_start]) if fr["vol"] is not None else None
    dates = np.array([fr["dates"][p - 1][:10] for p in pred_start])
    ex = {c: np.array([v[p:p + T.FUTURE_LEN].mean() for p in pred_start]) for c, v in fr["extra"].items()}
    return tb, mb, vol, dates, ex


def design(tb, mb, vol, fold_id=None, n_folds=1, ex=None):
    cols = [np.ones_like(tb), tb, mb]
    names = ["const", "tbill(%p)", "metab(%)"]
    for c, v in (ex or {}).items():
        cols.append(v); names.append(c)
    if CTRL and vol is not None:
        cols.append(vol); names.append("vol(%)")
    if fold_id is not None and n_folds > 1:
        for f in range(1, n_folds):
            cols.append((fold_id == f).astype(float)); names.append(f"fold{f}")
    return np.column_stack(cols), names


# ------------------------------------------------------------------ 본체
def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 118)
    print(f"# §4.2 수준 회귀 (expectile, Newey–Powell)  τ={TAUS}  B={B}  통제={'vol' if CTRL else '없음'}  MAC-Flow={PS.TAG_PREFIX}")
    print("#   계수 단위: IHL %p per 1%p 금리(연율) / per 1% 유동성(metab_13w).  [표준화] = 설명변수 1 SD 당.")
    print("#   실측 SE = 원점 이동블록 부트스트랩(13주).  모형 = 시드 5개 평균 ± 시드 SD (괄호: 블록부트 SE 평균).")
    print("#" * 118)
    rows = [["fold", "tau", "who", "var", "beta", "se", "beta_std", "n_orig", "n_seed"]]
    pooled = {"act": [], "sim": {s: [] for s in SEEDS}}

    for fi, fold in enumerate(FOLDS):
        fr = fold_frame(fold)
        arrs = []
        for s in SEEDS:
            try:
                arrs.append(RT.load_flow(fold, s))
            except Exception as e:                                  # noqa: BLE001
                print(f"  [MAC-Flow s{s}] {e!r}")
        if not arrs:
            print(f"\n[{LABELS[fold]}] MAC-Flow 배열 없음"); continue
        common = sorted(set.intersection(*[set(int(x) for x in a["pred_start"]) for a in arrs]))
        order = np.array(common, int)
        tb, mb, vol, dates, ex = origin_regressors(fr, order)
        sd = {"tbill(%p)": tb.std(ddof=1), "metab(%)": mb.std(ddof=1), "vol(%)": (vol.std(ddof=1) if vol is not None else np.nan)}
        sd.update({c: v.std(ddof=1) for c, v in ex.items()})
        X, names = design(tb, mb, vol, ex=ex)
        # 실측 IHL
        a0 = arrs[0]; pos = {int(v): i for i, v in enumerate(a0["pred_start"])}
        act = np.asarray(a0["act"])[[pos[v] for v in order]]
        y_act = ihl_paths(act) * PP
        g_act = np.arange(len(order))
        print(f"\n=== [{LABELS[fold]}]  원점 {len(order)}  {dates[0]}~{dates[-1]}   "
              f"금리 {tb.mean():.2f}±{sd['tbill(%p)']:.2f}%  유동성 {mb.mean():+.2f}±{sd['metab(%)']:.2f}%"
              + (f"  vol {vol.mean():.1f}±{sd['vol(%)']:.1f}%" if vol is not None else ""))
        print(f"    {'τ':>5} {'who':<7}" + "".join(f"{n:>22}" for n in names[1:]))
        # 모형 IHL (시드별 풀링)
        sim_y, sim_X, sim_g = {}, {}, {}
        for s, a in zip(SEEDS, arrs):
            pos = {int(v): i for i, v in enumerate(a["pred_start"])}
            sim = np.asarray(a["sim"])[[pos[v] for v in order]]        # (n_orig, n_sim, T)
            yi = ihl_paths(sim) * PP                                    # (n_orig, n_sim)
            n_sim = yi.shape[1]
            sim_y[s] = yi.ravel()
            sim_X[s] = np.repeat(X, n_sim, axis=0)
            sim_g[s] = np.repeat(np.arange(len(order)), n_sim)
        pooled["act"].append((X, y_act, g_act, fi, tb, mb, vol, ex))
        for s in SEEDS:
            if s in sim_y:
                pooled["sim"][s].append((sim_X[s], sim_y[s], sim_g[s], fi))

        for tau in TAUS:
            b_act = expectile_fit(X, y_act, tau)
            se_act = block_boot(X, y_act, tau, g_act)
            line = f"    {tau:>5.2f} {'actual':<7}"
            for j, n in enumerate(names[1:], 1):
                line += f"{b_act[j]:>+9.3f} ±{se_act[j]:<6.3f}[{b_act[j]*sd[n]:>+5.2f}]"
                rows.append([fold, tau, "actual", n, f"{b_act[j]:.4f}", f"{se_act[j]:.4f}", f"{b_act[j]*sd[n]:.4f}", len(order), 1])
            print(line)
            vidx = list(range(1, len(names)))
            sh_a, r2_a = shapley_share(X, y_act, tau, vidx)
            tot = max(sh_a.sum(), 1e-12)
            print(f"    {'':>5} {'  share':<7}" + "".join(f"{100*sh_a[j-1]/tot:>20.0f}% " for j in vidx) + f"  (actual 의사-R² {r2_a:.3f})")
            for j in vidx:
                rows.append([fold, tau, "actual", names[j] + " share%", f"{100*sh_a[j-1]/tot:.1f}", "", f"{r2_a:.4f}", len(order), 1])
            bs, ses = [], []
            for s in SEEDS:
                if s not in sim_y:
                    continue
                bs.append(expectile_fit(sim_X[s], sim_y[s], tau))
                ses.append(block_boot(sim_X[s], sim_y[s], tau, sim_g[s], B=max(50, B // 3)))
            bs = np.asarray(bs); ses = np.asarray(ses)
            bm = bs.mean(axis=0); bsd = bs.std(axis=0, ddof=1) if len(bs) > 1 else np.full_like(bm, np.nan)
            line = f"    {'':>5} {'MAC-Flow':<7}"
            for j, n in enumerate(names[1:], 1):
                line += f"{bm[j]:>+9.3f} ±{bsd[j]:<6.3f}[{bm[j]*sd[n]:>+5.2f}]"
                rows.append([fold, tau, "MAC-Flow", n, f"{bm[j]:.4f}", f"{bsd[j]:.4f}", f"{bm[j]*sd[n]:.4f}", len(order), len(bs)])
            print(line + f"   (블록부트 SE 평균: " + ", ".join(f"{ses.mean(axis=0)[j]:.3f}" for j in range(1, len(names))) + ")")
            shs, r2s = [], []
            for s in SEEDS:
                if s in sim_y:
                    sh, r2 = shapley_share(sim_X[s], sim_y[s], tau, vidx); shs.append(sh); r2s.append(r2)
            if shs:
                sh_m = np.mean(shs, axis=0); tot = max(sh_m.sum(), 1e-12)
                print(f"    {'':>5} {'  share':<7}" + "".join(f"{100*sh_m[j-1]/tot:>20.0f}% " for j in vidx) + f"  (MAC-Flow 의사-R² {np.mean(r2s):.3f})")
                for j in vidx:
                    rows.append([fold, tau, "MAC-Flow", names[j] + " share%", f"{100*sh_m[j-1]/tot:.1f}", "", f"{np.mean(r2s):.4f}", len(order), len(shs)])

    # ---- 폴드 통합 (폴드 더미) ----
    if len(pooled["act"]) > 1:
        nF = len(pooled["act"])
        Xs, ys, gs, offs = [], [], [], 0
        tbs, mbs, vols, exs = [], [], [], []
        for (X, y, g, fi, tb, mb, vol, ex) in pooled["act"]:
            tbs.append(tb); mbs.append(mb); vols.append(vol); exs.append(ex)
        tb = np.concatenate(tbs); mb = np.concatenate(mbs)
        vol = np.concatenate(vols) if all(v is not None for v in vols) else None
        fold_id = np.concatenate([np.full(len(t), i) for i, t in enumerate(tbs)])
        ex = {c: np.concatenate([e[c] for e in exs]) for c in (exs[0] if exs else {})}
        Xp, names = design(tb, mb, vol, fold_id, nF, ex=ex)
        y_act = np.concatenate([p[1] for p in pooled["act"]])
        g_act = np.concatenate([p[2] + sum(len(q[1]) for q in pooled["act"][:i]) for i, p in enumerate(pooled["act"])])
        sd = {"tbill(%p)": tb.std(ddof=1), "metab(%)": mb.std(ddof=1), "vol(%)": (vol.std(ddof=1) if vol is not None else np.nan)}
        sd.update({c: v.std(ddof=1) for c, v in ex.items()})
        print(f"\n=== [4 폴드 통합, 폴드 더미]  원점 {len(y_act)}   금리 SD {sd['tbill(%p)']:.2f}  유동성 SD {sd['metab(%)']:.2f}")
        kshow = 3 + len(ex) + (1 if (CTRL and vol is not None) else 0)
        print(f"    {'τ':>5} {'who':<7}" + "".join(f"{n:>22}" for n in names[1:kshow]))
        for tau in TAUS:
            b = expectile_fit(Xp, y_act, tau); se = block_boot(Xp, y_act, tau, g_act)
            line = f"    {tau:>5.2f} {'actual':<7}"
            for j in range(1, kshow):
                line += f"{b[j]:>+9.3f} ±{se[j]:<6.3f}[{b[j]*sd[names[j]]:>+5.2f}]"
                rows.append(["pooled", tau, "actual", names[j], f"{b[j]:.4f}", f"{se[j]:.4f}", f"{b[j]*sd[names[j]]:.4f}", len(y_act), 1])
            print(line)
            vidx = list(range(1, kshow))
            sh_a, r2_a = shapley_share(Xp, y_act, tau, vidx); tot = max(sh_a.sum(), 1e-12)
            print(f"    {'':>5} {'  share':<7}" + "".join(f"{100*sh_a[j-1]/tot:>20.0f}% " for j in vidx) + f"  (actual 의사-R² {r2_a:.3f}, 폴드더미 제외 몫)")
            for j in vidx:
                rows.append(["pooled", tau, "actual", names[j] + " share%", f"{100*sh_a[j-1]/tot:.1f}", "", f"{r2_a:.4f}", len(y_act), 1])
            bs = []; shs_p, r2s_p = [], []
            for s in SEEDS:
                parts = pooled["sim"][s]
                if len(parts) < nF:
                    continue
                Xs_ = []; ys_ = []
                for (Xi, yi, gi, fi) in parts:
                    n_sim = len(yi) // (len(gi) // gi.max() if gi.max() else 1) if False else None
                    Xs_.append(Xi); ys_.append(yi)
                # 폴드 더미를 시드 배열에도 붙인다
                Xcat = np.vstack(Xs_); ycat = np.concatenate(ys_)
                fid = np.concatenate([np.full(len(yi), fi) for (Xi, yi, gi, fi) in parts])
                dums = np.column_stack([(fid == f).astype(float) for f in range(1, nF)])
                Xcat = np.hstack([Xcat, dums])
                bs.append(expectile_fit(Xcat, ycat, tau))
                sh, r2 = shapley_share(Xcat, ycat, tau, list(range(1, kshow))); shs_p.append(sh); r2s_p.append(r2)
            if bs:
                bs = np.asarray(bs); bm = bs.mean(axis=0); bsd = bs.std(axis=0, ddof=1)
                line = f"    {'':>5} {'MAC-Flow':<7}"
                for j in range(1, kshow):
                    line += f"{bm[j]:>+9.3f} ±{bsd[j]:<6.3f}[{bm[j]*sd[names[j]]:>+5.2f}]"
                    rows.append(["pooled", tau, "MAC-Flow", names[j], f"{bm[j]:.4f}", f"{bsd[j]:.4f}", f"{bm[j]*sd[names[j]]:.4f}", len(y_act), len(bs)])
                print(line)
                sh_m = np.mean(shs_p, axis=0); tot = max(sh_m.sum(), 1e-12)
                print(f"    {'':>5} {'  share':<7}" + "".join(f"{100*sh_m[j-1]/tot:>20.0f}% " for j in range(1, kshow)) + f"  (MAC-Flow 의사-R² {np.mean(r2s_p):.3f})")
                for j in range(1, kshow):
                    rows.append(["pooled", tau, "MAC-Flow", names[j] + " share%", f"{100*sh_m[j-1]/tot:.1f}", "", f"{np.mean(r2s_p):.4f}", len(y_act), len(bs)])

    out = os.path.join(PS.RESULT_DIR, f"level_regression_4_2{PS.CACHE_SUFFIX}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\n[csv] {len(rows)-1} 행 → {out}")
    print("  읽는 법: τ=0.5 는 평균 효과(OLS), τ 가 작을수록 왼쪽 꼬리.  [ ] 는 설명변수 1 SD 당 효과 = 상대 비중.")
    print("  실측과 모형의 계수가 같은 부호·비슷한 크기면 모형이 수준 효과를 실데이터와 같은 크기로 담고 있는 것이다.")


if __name__ == "__main__":
    main()
