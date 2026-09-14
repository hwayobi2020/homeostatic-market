# -*- coding: utf-8 -*-
"""§4.3 현미경 — 폴드마다 원점 **하나**를 규칙으로 고르고, 그 원점에서 경로 형태별 IHL 분포 전체를 본다.

무엇을
------
통계(원점 평균·최악값)가 아니라 사례다.  선택 규칙: 폴드 안에서 *실현된* 13 주 tbill 변화폭
|tb(끝) − tb(시작)| 이 가장 큰 원점 (금리가 실제로 움직이던 시점).  MS_ORIGIN_DATE=YYYY-MM-DD 로
날짜를 직접 지정할 수도 있다 (한 폴드만 돌리려면 MS_FOLD=F_gfc).

그 원점에서 다음을 각 시드(5)로 생성한다 (원점 1 개라 GPU 몇 초):
    flat                : 양 축 마지막값 유지
    rate_cont / rate_step : 금리 anchored ramp_up / step_up (Table 20 의 Continuous / Discrete)
    liq_ramp_up / liq_ramp_down : 유동성 zero-mean ramp (Table 18)
    joint_ramp_up / joint_ramp_down : 금리 step_up + 유동성 zero-mean ramp (Table 21)
    realized            : 실제로 실현된 13 주 금리·유동성 경로 (model-on-realized)
경로 수는 MS_NSIM (기본 10000 — 히스토그램이 매끈해지도록 §4.3 의 1000 보다 크게).
같은 시드 = 같은 난수라 시나리오끼리 경로가 짝지어진다.

출력
----
  result/microscope_{fold}.npz   : 시나리오별 IHL (n_seed, n_sim), 원점 날짜, 실현 경로
  result/microscope_{fold}.png   : flat 대비 히스토그램 (시드 5 개 풀링)
  표: 시나리오별 mean / VaR10 / CVaR10 / CVaR5 / CVaR1 (시드 평균 ± 시드 SD), flat 대비 Δ

사용
----
    !PS_BASE=fpath_novol FPATH_DIM=2 python colab/dual_3ch/microscope_origin.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_pathshape_rawvol as PS                          # noqa: E402
import pathshape_realized_anchored_rawvol as AN                # noqa: E402  (anchored shapes)
import pathshape_zeromean_anchored_rawvol as ZM                # noqa: E402  (zero-mean shapes, rollout)
from rawvol_helpers import ihl_paths                           # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FOLDS = [os.environ["MS_FOLD"]] if os.environ.get("MS_FOLD") else PS.FOLDS
SEEDS = PS.SEEDS
FUT = PS.FUTURE_LEN
N_SIM = int(os.environ.get("MS_NSIM", "10000"))
ORIGIN_DATE = os.environ.get("MS_ORIGIN_DATE", "")
LABELS = {"F_gfc": "Financial crisis", "F_long_A": "Recovery",
          "F_long_B_origin": "COVID", "F_long": "Tightening"}
PP = 100.0


def _annual(z, ctx):
    """tbill z → 연율 %."""
    wr = z * ctx["csd"][ctx["ti"]] + ctx["cmu"][ctx["ti"]]
    return ((1.0 + wr) ** 52 - 1.0) * 100.0


def pick_origin(ctx):
    tb = ctx["real_tb_fut"]                                    # (n, FUT) z
    dates = [str(d)[:10] for d in ctx["origin_dates"]]
    if ORIGIN_DATE:
        i = dates.index(ORIGIN_DATE)
    else:
        chg = np.abs(tb[:, -1] - tb[:, 0])
        i = int(np.argmax(chg))
    d_ann = _annual(tb[i, -1], ctx) - _annual(tb[i, 0], ctx)
    top = np.argsort(-np.abs(tb[:, -1] - tb[:, 0]))[:3]
    return i, dates[i], d_ann, [(dates[j], _annual(tb[j, -1], ctx) - _annual(tb[j, 0], ctx)) for j in top]


def subset_ctx(ctx, i):
    c = dict(ctx)
    c["Xte"] = ctx["Xte"][i:i + 1]
    c["extra"] = ctx["extra"][i:i + 1] if ctx["extra"] is not None else None
    c["last_sp"] = ctx["last_sp"][i:i + 1]
    r = dict(ctx["rescale"]); r["s2"] = r["s2"][i:i + 1]; r["e2"] = r["e2"][i:i + 1]
    c["rescale"] = r
    return c


def scenarios(ctx1, i, ctx):
    range_tb = ctx["tb_z"][90] - ctx["tb_z"][10]
    range_mb = ctx["mb_z"][90] - ctx["mb_z"][10]
    anc_tb = ctx["real_tb_fut"][i:i + 1, 0:1]; anc_mb = ctx["real_mb_fut"][i:i + 1, 0:1]
    tb_flat = anc_tb + np.zeros((1, FUT)); mb_flat = anc_mb + np.zeros((1, FUT))
    tb_cont = anc_tb + AN.SHAPES["ramp_up"][None, :] * range_tb
    tb_step = anc_tb + AN.SHAPES["step_up"][None, :] * range_tb
    mb_up = anc_mb + ZM.SHAPES["ramp_up"][None, :] * range_mb
    mb_dn = anc_mb + ZM.SHAPES["ramp_down"][None, :] * range_mb
    return {
        "flat":            (tb_flat, mb_flat),
        "rate_cont":       (tb_cont, mb_flat),
        "rate_step":       (tb_step, mb_flat),
        "liq_ramp_up":     (tb_flat, mb_up),
        "liq_ramp_down":   (tb_flat, mb_dn),
        "joint_ramp_up":   (tb_step, mb_up),
        "joint_ramp_down": (tb_step, mb_dn),
        "realized":        (ctx["real_tb_fut"][i:i + 1], ctx["real_mb_fut"][i:i + 1]),
    }


def ihl_of(ctx1, tb, mb, seed, device):
    ZM.N_SIM = N_SIM                                           # rollout_paths 가 모듈 전역을 읽는다
    sim_z = ZM.rollout_paths(ctx1, tb, mb, seed, device)       # (1, n_sim, T)
    r = ctx1["rescale"]
    sim_raw = PS.forward_garch_rescale(sim_z * r["tsd"] + r["tmu"], r["s2"], r["e2"],
                                       r["om"], r["al"], r["be"], r["mu"])
    return ihl_paths(sim_raw)[0] * PP                          # (n_sim,) %


def tail(x):
    s = np.sort(x); n = s.size
    return dict(mean=s.mean(), var10=s[max(1, int(.10 * n)) - 1],
                cvar10=s[:max(1, int(.10 * n))].mean(), cvar5=s[:max(1, int(.05 * n))].mean(),
                cvar1=s[:max(1, int(.01 * n))].mean())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 110)
    print(f"# §4.3 현미경 — 실현 13주 tbill 변화폭 최대 원점 1개, 시나리오별 IHL 분포.  n_sim={N_SIM}, device={device}")
    print("#" * 110)
    for fold in FOLDS:
        per = {}; meta = None
        for seed in SEEDS:
            ctx = PS.load_fold_seed(fold, seed, device)
            if ctx is None:
                continue
            i, date, d_ann, top = pick_origin(ctx)
            if meta is None:
                meta = dict(i=i, date=date, d_ann=d_ann, top=top,
                            real_tb=[_annual(z, ctx) for z in ctx["real_tb_fut"][i]],
                            real_mb=[float(z * ctx["csd"][ctx["mi"]] + ctx["cmu"][ctx["mi"]]) for z in ctx["real_mb_fut"][i]])
            ctx1 = subset_ctx(ctx, i)
            for name, (tb, mb) in scenarios(ctx1, i, ctx).items():
                per.setdefault(name, []).append(ihl_of(ctx1, tb, mb, seed, device))
            del ctx; torch.cuda.empty_cache() if device == "cuda" else None
        if meta is None:
            print(f"\n[{LABELS.get(fold, fold)}] 체크포인트 없음"); continue
        arr = {k: np.stack(v) for k, v in per.items()}          # (n_seed, n_sim)
        print(f"\n=== [{LABELS.get(fold, fold)}] 원점 {meta['date']}  실현 13주 tbill 변화 {meta['d_ann']:+.2f}%p (연율)")
        print(f"    후보 상위 3: " + ", ".join(f"{d} ({v:+.2f})" for d, v in meta["top"]))
        print(f"    실현 tbill 경로(연율%): " + " ".join(f"{v:.2f}" for v in meta["real_tb"]))
        print(f"    실현 metab 경로(13주%): " + " ".join(f"{v:+.2f}" for v in meta["real_mb"]))
        print(f"    {'scenario':<17}{'mean':>8}{'VaR10':>8}{'CVaR10':>8}{'CVaR5':>8}{'CVaR1':>8}   {'ΔCVaR10 vs flat (시드평균±SD)':>32}")
        fl = [tail(x) for x in arr["flat"]]
        for name, m in arr.items():
            ts = [tail(x) for x in m]
            mn = {k: np.mean([t[k] for t in ts]) for k in ts[0]}
            d = [t["cvar10"] - f["cvar10"] for t, f in zip(ts, fl)]
            dtxt = "" if name == "flat" else f"{np.mean(d):+.2f} ± {np.std(d, ddof=1):.2f}"
            print(f"    {name:<17}{mn['mean']:>8.2f}{mn['var10']:>8.2f}{mn['cvar10']:>8.2f}{mn['cvar5']:>8.2f}{mn['cvar1']:>8.2f}   {dtxt:>32}")
        out = os.path.join(PS.RESULT_DIR, f"microscope_{fold}.npz")
        np.savez(out, date=meta["date"], d_ann=meta["d_ann"], real_tb=meta["real_tb"], real_mb=meta["real_mb"],
                 **{k: v for k, v in arr.items()})
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            names = [n for n in arr if n != "flat"]
            fig, axes = plt.subplots(2, 4, figsize=(18, 7)); axes = axes.ravel()
            bins = np.linspace(min(a.min() for a in arr.values()), 0, 80)
            for ax, name in zip(axes, names):
                ax.hist(arr["flat"].ravel(), bins=bins, alpha=.45, label="flat", density=True)
                ax.hist(arr[name].ravel(), bins=bins, alpha=.45, label=name, density=True)
                ax.axvline(np.mean([tail(x)["cvar10"] for x in arr["flat"]]), color="C0", ls="--", lw=1)
                ax.axvline(np.mean([tail(x)["cvar10"] for x in arr[name]]), color="C1", ls="--", lw=1)
                ax.set_title(name); ax.legend(fontsize=8); ax.set_xlabel("IHL (%)")
            for ax in axes[len(names):]:
                ax.axis("off")
            fig.suptitle(f"{LABELS.get(fold, fold)} — origin {meta['date']} (realized 13w Δtbill {meta['d_ann']:+.2f}%p); dashed = CVaR10")
            fig.tight_layout(); png = out.replace(".npz", ".png"); fig.savefig(png, dpi=120); plt.close(fig)
            print(f"    [png] {png}")
        except Exception as e:
            print(f"    [plot skip] {e}")
        print(f"    [npz] {out}")


if __name__ == "__main__":
    main()
