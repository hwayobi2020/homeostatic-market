"""비중첩(솎아낸) 원점에서 네 모델 13주 지표 비교 — 한 번에 실행 + 요약 출력.

리뷰어1 #2
----------
  "Origins appear weekly, horizon is 13 weeks - 13x overlap.  Coverage rates are
   not independent?  Hence, the effective sample is far smaller than nominal.
   Therefore, use thin to non-overlapping windows as a check, ..."

원점이 주 단위라 이웃 원점의 13주 예측 구간이 최대 12주를 공유한다.  원점을
STRIDE(=13) 간격으로 솎으면 예측 구간이 겹치지 않아 관측치가 서로 독립이 된다.
지평은 13주 그대로 두므로 모델의 원래 설정을 바꾸지 않는다.

  · 원점: 0, 13, 26, ... (폴드당 181 → 14 개)
  · 지표: 13주 전체 (CRPS, cov50/80/95, skew, std)
  · 구간: 전역 풀링 (네 모델 동일 정의)

주의: 과거 52주 입력창은 솎은 뒤에도 원점 간 39주가 남으므로 완전히 분리되지는
않는다.  겹치지 않게 되는 것은 *평가 대상*인 예측 구간이다.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/run_thin_compare.py
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rawvol_helpers import patch_rawvol                                  # noqa: E402
patch_rawvol()                                   # MAC-Flow 와 동일 파이프라인 (rolling-std)

import analyze_pathshape_rawvol as PS                                    # noqa: E402
import pathshape_zeromean_anchored_rawvol as ZM                          # noqa: E402
import train_garch_flow as T                                             # noqa: E402
import train_garch_xpast as GX                                           # noqa: E402
import train_vae_gan_baseline as VG                                      # noqa: E402
from rawvol_helpers import rawstd_preprocess_fold, forward_rawvol_rescale  # noqa: E402

VG.garch_preprocess_fold = rawstd_preprocess_fold   # import-bound 이름 교체
VG.forward_garch_rescale = forward_rawvol_rescale

STRIDE = 13
FOLDS = PS.FOLDS
SEEDS = PS.SEEDS
N_SIM = PS.N_SIM
OUT = os.path.join(PS.RESULT_DIR, f"thin{STRIDE}_compare{PS.CACHE_SUFFIX}.json")


def metrics(sim, act):
    """전역 풀링 구간 기준 13주 지표.  sim (n, n_sim, T), act (n, T)."""
    af = np.asarray(act, float).ravel()
    sf = np.asarray(sim, float).ravel()
    cov = {}
    for lvl, lo, hi in [(50, 25.0, 75.0), (80, 10.0, 90.0), (95, 2.5, 97.5)]:
        L, H = np.percentile(sf, lo), np.percentile(sf, hi)
        cov[lvl] = float(((af >= L) & (af <= H)).mean())

    def _sk(x):
        x = np.asarray(x, float)
        return float(np.mean(((x - x.mean()) / (x.std() + 1e-12)) ** 3))

    return dict(n_origin=int(np.shape(act)[0]), n_obs=int(af.size),
                crps=float(T.crps_pooled(sim, act)[0]),
                cov50=cov[50], cov80=cov[80], cov95=cov[95],
                skew_actual=_sk(af), skew_sim=_sk(sf),
                std_actual=float(af.std(ddof=1)), std_sim=float(sf.std(ddof=1)))


def thin(sim, act):
    """원점을 STRIDE 간격으로 솎는다."""
    idx = np.arange(0, np.shape(act)[0], STRIDE)
    return np.asarray(sim)[idx], np.asarray(act)[idx], idx


def macflow_arrays(fold, seed, device):
    """저장된 ckpt 로 재추론 → (sim_raw, actual_raw).

    미래 조건 = **원점의 마지막 관측값을 13주 유지**(flat).  실현 경로를 주면
    MAC-Flow 만 미래 정보를 갖게 되어 비교가 불공정해지고, 0 을 넣으면 z-score
    기준 학습기간 평균으로 점프시켜 원점 수준과 단절된다.  과거 52주를 주는
    이상 그 마지막 값이 이어지는 것이 자연스럽다.
    §4.3 flat 기준선(pathshape_zeromean_anchored_rawvol.py:116-119)과 같은 방식.
    """
    ctx = PS.load_fold_seed(fold, seed, device)
    if ctx is None:
        return None, None

    ckpt = torch.load(os.path.join(
        PS.RESULT_DIR, f"garch_flow_ar_{PS.TAG_PREFIX}_s{seed}_{fold}_best.pt"),
        map_location="cpu")
    meta = ckpt["meta"]
    cond_stats, target_stats = meta["cond_stats"], meta["target_stats"]
    gp = T.garch_preprocess_fold(PS.FOLDS_DIR, fold, PS.RESULT_DIR)
    _, Yte, _, _, _ = T.cached_load_windows_seq(
        gp["test"], cond_stats=cond_stats, target_stats=target_stats)
    valid_mask, z_te, df_te = T.compute_valid_mask(gp["test"], cond_stats)

    # 앵커 = 과거 52주의 *마지막* 관측값 (행 w+PAST_LEN-1).  미래 첫 주가 아니다.
    #   (§4.3 의 pathshape 코드는 real_*_fut[:, 0] = 미래 첫 주를 앵커로 쓴다.
    #    주간 거시는 거의 안 움직여 값 차이는 미미하나, 여기서는 미래를 전혀
    #    쓰지 않는 쪽으로 엄격하게 잡는다.)
    n_w = z_te.shape[0] - T.PAST_LEN - T.FUTURE_LEN + 1
    _last = z_te[T.PAST_LEN - 1: T.PAST_LEN - 1 + n_w][valid_mask]      # (n_orig, N_CH)
    anc_tb = _last[:, ctx["ti"]:ctx["ti"] + 1]
    anc_mb = _last[:, ctx["mi"]:ctx["mi"] + 1]
    tb_flat = anc_tb + np.zeros((1, T.FUTURE_LEN))
    mb_flat = anc_mb + np.zeros((1, T.FUTURE_LEN))
    sim_z = ZM.rollout_paths(ctx, tb_flat, mb_flat, seed, device)
    r = ctx["rescale"]
    sim_raw = PS.forward_garch_rescale(sim_z * r["tsd"] + r["tmu"], r["s2"], r["e2"],
                                       r["om"], r["al"], r["be"], r["mu"])
    tmu, tsd = float(target_stats["mean"]), float(target_stats["std"])
    gsig = df_te["garch_sigma"].to_numpy(float)
    gmu = df_te["garch_mu"].to_numpy(float)
    oidx = np.where(valid_mask)[0]
    sig = np.array([[gsig[int(o) + T.PAST_LEN + t] for t in range(T.FUTURE_LEN)]
                    for o in oidx])
    mu = np.array([[gmu[int(o) + T.PAST_LEN + t] for t in range(T.FUTURE_LEN)]
                   for o in oidx])
    # evaluate_test 와 동일: actual 은 스텝별 실현 σ 로 복원 (정답 라벨이므로 누수 아님).
    actual_raw = (Yte.numpy() * tsd + tmu) * sig + mu
    if actual_raw.shape[0] != sim_raw.shape[0]:
        sys.exit(f"[FATAL] origin 수 불일치 {fold} s{seed}: "
                 f"actual {actual_raw.shape[0]} vs sim {sim_raw.shape[0]}")
    return sim_raw, actual_raw


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 108)
    print(f"# 비중첩 비교 — 원점 STRIDE={STRIDE} 로 솎음, 지평 13주 유지, 전역 풀링 구간")
    print(f"# device={device}  n_sim={N_SIM}  MAC-Flow tag={PS.TAG_PREFIX} ({len(SEEDS)} seed)")
    print("#" * 108)

    out = {f: {} for f in FOLDS}
    for fold in FOLDS:
        print(f"\n===== {fold}")

        # ── GARCH-ST (CPU, 적합+시뮬)
        try:
            sp = os.path.join(GX.RESULT_DIR, f"garch_xpast_{fold}_summary.json")
            bak = sp + ".thinbak"
            if os.path.exists(sp):
                os.replace(sp, bak)          # skip 회피
            a = GX.run_fold(fold)
            if os.path.exists(bak):
                os.replace(bak, sp)          # 원상복구
            s, ac, idx = thin(a["sim"], a["act"])
            out[fold]["GARCH-ST"] = metrics(s, ac)
            print(f"  [GARCH-ST] 원점 {a['act'].shape[0]} → {len(idx)}")
        except Exception as e:
            print(f"  [FAIL GARCH-ST] {e!r}")

        # ── CondVAE / CondGAN (학습 포함)
        for mk in ("vae", "gan"):
            try:
                args = SimpleNamespace(model=mk, fold=fold, folds_dir=PS.FOLDS_DIR,
                                       out_dir=PS.RESULT_DIR, n_sim=N_SIM, seed=2026)
                spv = os.path.join(PS.RESULT_DIR, f"{mk}_baseline_{fold}_summary.json")
                bakv = spv + ".thinbak"
                if os.path.exists(spv):
                    os.replace(spv, bakv)    # 기존 τ=1 결과 보존
                a = VG.run_fold(mk, fold, args, device)
                if os.path.exists(bakv):
                    os.replace(bakv, spv)
                s, ac, idx = thin(a["sim"], a["act"])
                out[fold][f"Cond{mk.upper()}"] = metrics(s, ac)
                print(f"  [Cond{mk.upper()}] 원점 {a['act'].shape[0]} → {len(idx)}")
            except Exception as e:
                print(f"  [FAIL {mk}] {e!r}")

        # ── MAC-Flow (ckpt 재추론, 시드 평균)
        per_seed = []
        for seed in SEEDS:
            try:
                sim, act = macflow_arrays(fold, seed, device)
                if sim is None:
                    continue
                s, ac, idx = thin(sim, act)
                per_seed.append(metrics(s, ac))
            except Exception as e:
                print(f"  [FAIL MAC-Flow s{seed}] {e!r}")
        if per_seed:
            keys = [k for k in per_seed[0] if k not in ("n_origin", "n_obs")]
            m = {k: float(np.mean([r[k] for r in per_seed])) for k in keys}
            m["n_origin"] = per_seed[0]["n_origin"]; m["n_obs"] = per_seed[0]["n_obs"]
            m["n_seed"] = len(per_seed)
            out[fold]["MAC-Flow"] = m
            print(f"  [MAC-Flow] 원점 {m['n_origin']}  ({len(per_seed)} seed 평균)")

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")

    # ── 요약표
    order = ["MAC-Flow", "GARCH-ST", "CondVAE", "CondGAN"]
    cols = [("n_origin", "원점", "{:>5.0f}"), ("n_obs", "관측", "{:>6.0f}"),
            ("crps", "CRPS", "{:>9.5f}"), ("cov50", "cov50", "{:>7.3f}"),
            ("cov80", "cov80", "{:>7.3f}"), ("cov95", "cov95", "{:>7.3f}"),
            ("skew_actual", "skew실측", "{:>9.4f}"), ("skew_sim", "skew모형", "{:>9.4f}")]
    print(f"\n{'='*108}\n[요약]  원점 STRIDE={STRIDE} · 지평 13주 · 전역 풀링 구간\n{'='*108}")
    print(f"{'fold':<18}{'model':<11}" + "".join(f"{c[1]:>10}" for c in cols))
    for fold in FOLDS:
        for name in order:
            r = out[fold].get(name)
            if r is None:
                continue
            row = "".join(("{:>10}".format(c[2].format(r[c[0]]).strip())) for c in cols)
            print(f"{fold:<18}{name:<11}{row}")
        print()
    print("[읽는 법] cov 는 명목 0.50/0.80/0.95 에 가까울수록 좋다. "
          "skew 는 실측과 부호·크기가 맞을수록 좋다. CRPS 는 낮을수록 좋다.")


if __name__ == "__main__":
    main()