# -*- coding: utf-8 -*-
"""자기회귀 롤링 변동성 역변환으로 §4.1 지표 재계산 — 재학습 없음.

무엇을 바꾸나
-------------
게재판은 생성 구간 13 주 내내 원점 σ 를 상수로 곱한다(`forward_rawvol_rescale`).
여기서는 σ 를 **생성된 경로로 갱신**한다:

    σ_h = std(직전 13 주 수익률, ddof=1)      (첫 창 = 원점까지의 실측 13 주)

학습 표준화도 같은 계열이다 — `data/extend_to_1971.py:396` 의
`rolling(13).std(ddof=1).shift(1)` 은 직전 13 주만 쓰고 한 주 시프트돼 있어
인과적이다.

**단, 정확히 같지는 않다.**  학습 σ_i 는 원점을 *제외한* std(r_{i-13..i-1}) 인데
여기 첫 창은 원점을 *포함한* std(r_{o-12..o}) 다.  한 주 시프트가 어긋나므로
이 역변환은 학습 표준화의 정확한 역이 아니다.  아래 `frozen_fresh` arm 이 그
차이를 분리한다.

역변환은 손실·체크포인트 선택(둘 다 z 공간)에 들어가지 않으므로 이 비교 자체는
기존 가중치로 성립한다.  다만 GPU 학습이 결정적이지 않아(`major revision/
실험계획.md` 미결 4번: cudnn.deterministic 미설정, 같은 시드 CRPS 10^-4 드리프트)
"다시 학습하면 같은 모델"이라고 말할 수는 없다 — 재학습은 재현성 검증이다.

GARCH 와 다르다: ω·α·β 를 적합하지 않고, 13 주 균등 가중이며, 장기 평균 회귀도
없다.  즉 남의 모형을 빌리지 않으면서 지평 안 변동성 동학을 갖는다.

출력
----
원점 고정(게재판)과 롤링을 **같은 z 표본**으로 나란히 계산해 비교한다.
`result/rollvol_compare_{TAG}.json` + 콘솔 표.

사용
----
    !python colab/dual_3ch/eval_rollvol.py
    # 시드/폴드 한정:  RV_SEEDS=2026 RV_FOLDS=F_gfc python colab/dual_3ch/eval_rollvol.py
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("PS_BASE", "fpath_novol")
os.environ.setdefault("FPATH_DIM", "2")

import analyze_pathshape_rawvol as PS                              # noqa: E402
import train_garch_flow as T                                       # noqa: E402
from train_garch_flow import (                                     # noqa: E402
    cached_load_windows_seq, load_extra_context, compute_valid_mask,
    PAST_LEN, FUTURE_LEN,
)
from rawvol_helpers import forward_rollvol_rescale                 # noqa: E402

RESULT_DIR = PS.RESULT_DIR
FOLDS_DIR = PS.FOLDS_DIR
FOLDS = [f for f in os.environ.get(
    "RV_FOLDS", ",".join(PS.FOLDS)).split(",") if f.strip()]
SEEDS = [int(s) for s in os.environ.get(
    "RV_SEEDS", "2026,2027,2028,2029,2030").split(",") if s.strip()]
N_SIM = int(os.environ.get("RV_NSIM", str(PS.N_SIM)))
CHUNK = PS.CHUNK
# 앞선 2-arm 실행 결과를 덮지 않도록 파일명을 분리한다 (RV_OUT_SUFFIX 로 재지정 가능).
OUT = os.path.join(RESULT_DIR, "rollvol_compare_%s%s.json"
                   % (PS.TAG_PREFIX, os.environ.get("RV_OUT_SUFFIX", "_3arm")))


# ---------------------------------------------------------------- 지표
def _sk(a):
    a = np.asarray(a, float)
    return float(np.mean(((a - a.mean()) / (a.std() + 1e-12)) ** 3))


def _ek(a):
    a = np.asarray(a, float)
    return float(np.mean(((a - a.mean()) / (a.std() + 1e-12)) ** 4) - 3.0)


def _crps_pooled(sim, act):
    """train_garch_x.crps_pooled 와 같은 정의 (원점×주 평균)."""
    from train_garch_x import crps_ensemble
    no, ns, Tt = sim.shape
    vals = [crps_ensemble(sim[t, :, w], act[t, w])
            for t in range(no) for w in range(Tt)]
    return float(np.mean(vals))


def _emd(a, b, nb=200):
    from train_garch_x import emd1d
    return emd1d(a, b, nb)


def metrics(sim_raw, act_raw):
    sf, af = sim_raw.ravel(), act_raw.ravel()
    std_a, std_s = float(af.std(ddof=1)), float(sf.std(ddof=1))
    cov = {}
    for lvl, lo, hi in [(50, 25, 75), (80, 10, 90), (95, 2.5, 97.5)]:
        L, H = np.percentile(sf, lo), np.percentile(sf, hi)
        cov[lvl] = float(((af >= L) & (af <= H)).mean())
    return dict(crps_pooled=_crps_pooled(sim_raw, act_raw), emd=_emd(sf, af),
                std_actual=std_a, std_sim=std_s, std_ratio=std_s / std_a,
                coverage_50=cov[50], coverage_80=cov[80], coverage_95=cov[95],
                skew_actual=_sk(af), skew_sim=_sk(sf),
                exkurt_actual=_ek(af), exkurt_sim=_ek(sf))


# ---------------------------------------------------------------- 로딩
def load_fold_seed(fold, seed, device):
    """PS.load_fold_seed 와 같은 재료 + 롤링 역변환에 필요한 실측 꼬리·미래 경로."""
    bp = os.path.join(RESULT_DIR,
                      f"garch_flow_ar_{PS.TAG_PREFIX}_s{seed}_{fold}_best.pt")
    if not os.path.exists(bp):
        print(f"  [없음] {os.path.basename(bp)}")
        return None
    ckpt = torch.load(bp, map_location=device)
    meta = ckpt["meta"]
    cond_stats, target_stats = meta["cond_stats"], meta["target_stats"]
    extra_stats = meta.get("extra_stats")
    model = PS.rebuild_model(ckpt, device)

    gp = T.garch_preprocess_fold(FOLDS_DIR, fold, RESULT_DIR)      # rawstd (patched)
    csv = gp["test"]
    Xte, Yte, _, _, _ = cached_load_windows_seq(
        csv, cond_stats=cond_stats, target_stats=target_stats)
    Xte_extra, _, n_ex, _ = load_extra_context(
        csv, PS.DC_COLS_LIST, extra_stats=extra_stats)
    if n_ex != Xte.shape[0]:
        sys.exit(f"[FATAL] extra {n_ex} != main {Xte.shape[0]} ({fold},{seed})")

    valid_mask, z_te, df_te = compute_valid_mask(csv, cond_stats)
    n_w = z_te.shape[0] - PAST_LEN - FUTURE_LEN + 1
    last_full = z_te[PAST_LEN - 1: PAST_LEN - 1 + n_w, T.SP_CH]
    last_sp = torch.from_numpy(
        last_full[valid_mask].astype(np.float32)).to(device)

    ti = PS.ENC_COLS.index("tbill_wr")
    mi = PS.ENC_COLS.index("metab_13w")
    fut_tb = np.stack([z_te[w + PAST_LEN: w + PAST_LEN + FUTURE_LEN, ti]
                       for w in range(n_w)])[valid_mask]
    fut_mb = np.stack([z_te[w + PAST_LEN: w + PAST_LEN + FUTURE_LEN, mi]
                       for w in range(n_w)])[valid_mask]

    # raw 수익률 복원: rawstd 전처리에서 sp_return = (r - mu)/sigma 이므로 r = z·σ + μ
    gsig = df_te["garch_sigma"].to_numpy(float)
    gz = df_te["sp_return"].to_numpy(float)
    gmu = float(df_te["garch_mu"].iloc[0])
    raw = gz * gsig + gmu
    oidx = np.where(valid_mask)[0]
    orow = oidx + (PAST_LEN - 1)                                   # 원점(마지막 관측) 행
    if orow.min() < 12:
        sys.exit(f"[FATAL] 원점 앞 13주가 모자란다 ({fold})")
    tail = np.stack([raw[r - 12: r + 1] for r in orow])            # (n_orig, 13)
    s2_orig = gsig[orow] ** 2

    # 실측 미래 13주 (raw) — 평가 대상
    act = np.stack([raw[r + 1: r + 1 + FUTURE_LEN] for r in orow])

    tmu, tsd = float(target_stats["mean"]), float(target_stats["std"])
    return dict(model=model, Xte=Xte.to(device), extra=Xte_extra.to(device),
                last_sp=last_sp, fut_tb=fut_tb, fut_mb=fut_mb,
                tail=tail, s2_orig=s2_orig, act=act, mu=gmu,
                tmu=tmu, tsd=tsd, n_orig=len(orow))


@torch.no_grad()
def rollout_realized(ctx, device, seed):
    """원점별 *실현* 거시 경로를 주입해 z 표본을 만든다 (§4.1 설정)."""
    torch.manual_seed(seed)
    Xte, extra, last_sp = ctx["Xte"], ctx["extra"], ctx["last_sp"]
    n = Xte.shape[0]
    out = []
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        tb = torch.tensor(ctx["fut_tb"][s:e], device=device, dtype=Xte.dtype)
        mb = torch.tensor(ctx["fut_mb"][s:e], device=device,
                          dtype=Xte.dtype).unsqueeze(-1)
        sim = ctx["model"].ar_sample(Xte[s:e, :PAST_LEN, :], tb, last_sp[s:e],
                                     N_SIM, extra_context=extra[s:e],
                                     future_macro_z=mb)
        out.append(sim.cpu())
    return torch.cat(out, dim=0).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print("# 역변환 3 판 — 같은 체크포인트·같은 z 표본, 곱하는 σ 만 다름")
    print("#  원점고정   : σ = garch_sigma[원점]  = std(r_{o-13..o-1})   (게재판)")
    print("#  고정(신선) : σ = std(r_{o-12..o})   를 13 주 상수            (정보집합만 맞춤)")
    print("#  롤링       : σ_h = std(직전 13 주), 생성분으로 갱신")
    print("#  → (고정↔고정신선) = 한 주 신선도 효과, (고정신선↔롤링) = 동학 효과")
    print(f"#  tag={PS.TAG_PREFIX}  folds={FOLDS}  seeds={SEEDS}  n_sim={N_SIM}")
    print(f"#  device={device}")
    print("#" * 100)

    res = {}
    for fold in FOLDS:
        per = {"frozen": [], "frozen_fresh": [], "rolling": []}
        for seed in SEEDS:
            ctx = load_fold_seed(fold, seed, device)
            if ctx is None:
                continue
            z = rollout_realized(ctx, device, seed)                # (n_orig, n_sim, T)
            zt = z * ctx["tsd"] + ctx["tmu"]                       # 표준화 수익률

            # (1) 게재판: σ = garch_sigma[원점] = std(r_{o-13..o-1}).  원점 수익률
            #     r_o 를 아직 안 쓴 값이라 롤링의 첫 창보다 한 주 낡았다.
            sig = np.sqrt(np.maximum(ctx["s2_orig"], 0.0))
            frozen = zt * sig[:, None, None] + ctx["mu"]

            # (2) 같은 정보로 고정: σ = std(꼬리 13주) = std(r_{o-12..o}) 를 13주 내내 상수.
            #     (3) 과 첫 스텝 정보집합이 같으므로, (2)↔(3) 차이가 순수한 *동학* 효과이고
            #     (1)↔(2) 차이가 *한 주 신선도* 효과다.  둘을 섞으면 해석이 오염된다.
            sig_fresh = ctx["tail"].std(axis=1, ddof=1)
            frozen_fresh = zt * sig_fresh[:, None, None] + ctx["mu"]

            # (3) 자기회귀 롤링
            rolling = forward_rollvol_rescale(zt, ctx["tail"], ctx["mu"])

            per["frozen"].append(metrics(frozen, ctx["act"]))
            per["frozen_fresh"].append(metrics(frozen_fresh, ctx["act"]))
            per["rolling"].append(metrics(rolling, ctx["act"]))
            print(f"  {fold} s{seed}  n_orig={ctx['n_orig']}  CRPS "
                  f"고정={per['frozen'][-1]['crps_pooled']:.5f} "
                  f"고정(신선)={per['frozen_fresh'][-1]['crps_pooled']:.5f} "
                  f"롤링={per['rolling'][-1]['crps_pooled']:.5f}")
        if not per["frozen"]:
            continue
        res[fold] = {k: {m: float(np.mean([d[m] for d in v]))
                         for m in v[0]} | {"n_seed": len(v)}
                     for k, v in per.items()}

    if not res:
        print("\n체크포인트를 못 찾았다.")
        return

    hdr = (f"{'fold':<17}{'역변환':<10}{'n':>3}{'CRPS':>10}{'EMD':>10}"
           f"{'std_r':>8}{'cov80':>8}{'cov95':>8}{'skew':>9}{'실측skew':>10}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for fold, r in res.items():
        for lab, key in (("원점고정", "frozen"), ("고정(신선)", "frozen_fresh"),
                         ("롤링", "rolling")):
            d = r[key]
            print(f"{fold:<17}{lab:<10}{d['n_seed']:>3}{d['crps_pooled']:>10.5f}"
                  f"{d['emd']:>10.6f}{d['std_ratio']:>8.3f}{d['coverage_80']:>8.3f}"
                  f"{d['coverage_95']:>8.3f}{d['skew_sim']:>9.3f}"
                  f"{d['skew_actual']:>10.3f}")
        print()

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
