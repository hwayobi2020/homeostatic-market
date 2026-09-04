"""MAC-Flow τ=1 (1주 앞) 지표 — 저장된 체크포인트로 평가만 재실행.

배경
----
§4.1 비교의 공정성 문제.  MAC-Flow 는 미래 13주 거시 경로를 통째로 받고
CondVAE/CondGAN 은 미래 금리만, GARCH-ST 는 origin 값 고정만 받는다.  이를
맞추려고 미래를 마스킹하면(maskall) 스텝별 주입 경로에 13주 내내 같은 값을
강제로 밀어넣는 핸디캡이 되어, 조건화를 없애는 것이 아니라 "변하지 않는
조건"을 계속 주입하는 것이 된다.

τ=1 에서는 미래 경로에서 주입되는 값이 첫 점 하나뿐이라 세 모델이 받는
미래 정보가 같아진다.  인위적 제약이 필요 없고, 원점 간 중첩도 없어
관측치가 서로 독립이다(리뷰어1 #2 의 13배 중첩이 이 축에서는 발생하지 않는다).

구간은 13주 지표와 동일하게 *전역 풀링* 으로 잡는다.  τ=1 로 잘라도
전역/원점별 선택은 남는 문제이며, 여기서는 세 모델의 정의를 맞추는 쪽을 택했다.

조건: 실현된 미래 거시 경로(real_tb_fut / real_mb_fut).  Table 10 의
"conditioning on realized macro paths" 와 같은 설정이다.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/eval_tau1_macflow.py
"""
import json
import os
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_pathshape_rawvol as PS                                   # noqa: E402
import pathshape_zeromean_anchored_rawvol as ZM                         # noqa: E402
import train_garch_flow as T                                            # noqa: E402

OUT = os.path.join(PS.RESULT_DIR, f"tau1_macflow{PS.CACHE_SUFFIX}.json")


def eval_one(fold, seed, device):
    ctx = PS.load_fold_seed(fold, seed, device)
    if ctx is None:
        return None

    # 실현 미래 거시 경로를 그대로 조건으로 주입 (origin 별 (n, 13) z 경로).
    sim_z = ZM.rollout_paths(ctx, ctx["real_tb_fut"], ctx["real_mb_fut"], seed, device)
    r = ctx["rescale"]
    sim_raw = PS.forward_garch_rescale(sim_z * r["tsd"] + r["tmu"],
                                       r["s2"], r["e2"],
                                       r["om"], r["al"], r["be"], r["mu"])   # (n, n_sim, 13)

    # ── 실측 복원 ────────────────────────────────────────────────────────
    # evaluate_test 와 동일: actual 은 *스텝별 실현 σ* 로 복원한다 (정답 라벨이므로 누수 아님).
    # sim 은 위에서 origin 고정 σ 로 복원했다 (look-ahead 제거).
    ckpt = torch.load(
        os.path.join(PS.RESULT_DIR,
                     f"garch_flow_ar_{PS.TAG_PREFIX}_s{seed}_{fold}_best.pt"),
        map_location="cpu")
    meta = ckpt["meta"]
    cond_stats, target_stats = meta["cond_stats"], meta["target_stats"]

    gp = T.garch_preprocess_fold(PS.FOLDS_DIR, fold, PS.RESULT_DIR)
    origin_csv = gp["test"]
    _, Yte, _, _, _ = T.cached_load_windows_seq(
        origin_csv, cond_stats=cond_stats, target_stats=target_stats)
    valid_mask, _, df_te = T.compute_valid_mask(origin_csv, cond_stats)

    tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
    actual_zt = Yte.numpy() * tsd + tmu                                  # (n, 13)
    gsig = df_te["garch_sigma"].to_numpy(float)
    gmu = df_te["garch_mu"].to_numpy(float)
    oidx = np.where(valid_mask)[0]
    sig = np.array([[gsig[int(o) + T.PAST_LEN + t] for t in range(T.FUTURE_LEN)]
                    for o in oidx])
    mu = np.array([[gmu[int(o) + T.PAST_LEN + t] for t in range(T.FUTURE_LEN)]
                   for o in oidx])
    actual_raw = actual_zt * sig + mu                                    # (n, 13)

    if actual_raw.shape[0] != sim_raw.shape[0]:
        sys.exit(f"[FATAL] origin 수 불일치 {fold} s{seed}: "
                 f"actual {actual_raw.shape[0]} vs sim {sim_raw.shape[0]}")

    # ── τ=1 지표 (전역 풀링 구간) ────────────────────────────────────────
    a1 = actual_raw[:, 0].ravel()
    s1 = sim_raw[:, :, 0].ravel()
    crps1 = T.crps_pooled(sim_raw[:, :, 0:1], actual_raw[:, 0:1])[0]
    cov1 = {}
    for lvl, lo, hi in [(50, 25.0, 75.0), (80, 10.0, 90.0), (95, 2.5, 97.5)]:
        L, H = np.percentile(s1, lo), np.percentile(s1, hi)
        cov1[lvl] = float(((a1 >= L) & (a1 <= H)).mean())

    def _sk(x):
        x = np.asarray(x, float)
        return float(np.mean(((x - x.mean()) / (x.std() + 1e-12)) ** 3))

    res = dict(tau1_n=int(a1.size), tau1_crps_pooled=float(crps1),
               tau1_coverage_50=cov1[50], tau1_coverage_80=cov1[80],
               tau1_coverage_95=cov1[95],
               tau1_std_actual=float(a1.std(ddof=1)),
               tau1_std_sim=float(s1.std(ddof=1)),
               tau1_skew_actual=_sk(a1), tau1_skew_sim=_sk(s1))
    print(f"  [{fold} s{seed}] n={res['tau1_n']}  CRPS={crps1:.5f}  "
          f"cov 50/80/95={cov1[50]:.3f}/{cov1[80]:.3f}/{cov1[95]:.3f}  "
          f"skew a/s={res['tau1_skew_actual']:+.4f}/{res['tau1_skew_sim']:+.4f}")
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print(f"# MAC-Flow τ=1 평가 — LOCKED {PS.TAG_PREFIX}, device={device}, "
          f"n_sim={PS.N_SIM}, 조건=실현 미래 거시 경로")
    print("#" * 100)

    out = {}
    for fold in PS.FOLDS:
        out[fold] = {}
        for seed in PS.SEEDS:
            r = eval_one(fold, seed, device)
            if r is not None:
                out[fold][str(seed)] = r

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")

    print(f"\n{'='*100}\n[폴드별 시드 평균]\n{'='*100}")
    keys = ["tau1_crps_pooled", "tau1_coverage_50", "tau1_coverage_80",
            "tau1_coverage_95", "tau1_skew_actual", "tau1_skew_sim"]
    print(f"{'fold':<18} {'n':>5} " + " ".join(f"{k.replace('tau1_',''):>12}" for k in keys))
    for fold in PS.FOLDS:
        rs = list(out[fold].values())
        if not rs:
            print(f"{fold:<18} (결과 없음)"); continue
        ns = {r["tau1_n"] for r in rs}
        n_txt = str(ns.pop()) if len(ns) == 1 else f"{sorted(ns)}"
        vals = [np.mean([r[k] for r in rs]) for k in keys]
        print(f"{fold:<18} {n_txt:>5} " + " ".join(f"{v:>12.4f}" for v in vals))
    print("\n[대조] GARCH-ST tau1_n=181, CondVAE/CondGAN tau1_n=182 (원점 조건 차이).")


if __name__ == "__main__":
    main()