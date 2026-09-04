"""비중첩 원점 · 미래 정보 없음 조건에서 네 모델 13주 지표 비교 (한 번 실행 + 요약).

리뷰어1 #2
----------
  "Origins appear weekly, horizon is 13 weeks - 13x overlap.  Coverage rates are
   not independent?  Hence, the effective sample is far smaller than nominal.
   Therefore, use thin to non-overlapping windows as a check, ..."

원점이 주 단위라 이웃 원점의 13주 예측 구간이 최대 12주를 공유한다.  원점을
STRIDE(=13) 간격으로 솎으면 예측 구간이 겹치지 않는다.  지평은 13주 그대로
두므로 모델 설정을 바꾸지 않는다.

미래 조건 (네 모델 동일)
------------------------
원점의 **마지막 관측값을 13주 유지**(flat).  누구도 미래를 보지 않는다.
  · MAC-Flow  : tbill·metab 을 flat 경로로 주입
  · CondVAE/GAN: MASK_FUTURE_FILL="last" — 미래 tbill 도 실현 경로를 되돌리지 않음
  · GARCH-ST  : 원래부터 origin 값 고정 (train_garch_xpast.py:154)
0 을 채우면 z-score 기준 학습기간 평균으로 점프해 원점 수준과 단절되므로 쓰지 않는다.

구간은 네 모델 모두 전역 풀링(train_garch_flow.evaluate_test 와 동일 정의).

한계: 예측 구간 겹침은 사라지지만 과거 52주 입력창은 원점 간 39주가 겹친다.

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

from rawvol_helpers import (patch_rawvol, rawstd_preprocess_fold,        # noqa: E402
                            forward_rawvol_rescale)
patch_rawvol()                                   # MAC-Flow 와 동일 파이프라인

import analyze_pathshape_rawvol as PS                                    # noqa: E402
import pathshape_zeromean_anchored_rawvol as ZM                          # noqa: E402
import train_garch_flow as T                                             # noqa: E402
import train_garch_xpast as GX                                           # noqa: E402
import train_vae_gan_baseline as VG                                      # noqa: E402

VG.garch_preprocess_fold = rawstd_preprocess_fold   # import-bound 이름 교체
VG.forward_garch_rescale = forward_rawvol_rescale
VG.MASK_FUTURE_FILL = "last"                        # 미래 tbill 도 flat 유지

STRIDE = 13
FOLDS = PS.FOLDS
SEEDS = PS.SEEDS
N_SIM = PS.N_SIM
VG_SEED = 2026
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


def thin_all_offsets(sim, act):
    """STRIDE 간격 부분표본을 오프셋 0..STRIDE-1 전부 만들어 지표를 평균한다.

    오프셋 0 만 쓰면 원점 14 개짜리 부분표본 하나로 판정하게 되어 어느 14 개를
    골랐느냐에 결과가 좌우된다.  13 개 오프셋을 모두 돌려 평균하면 그 임의성이
    사라지고, 각 부분표본 안에서는 예측 구간이 겹치지 않는다는 성질도 유지된다.
    """
    n = np.shape(act)[0]
    per = []
    for off in range(STRIDE):
        idx = np.arange(off, n, STRIDE)
        if idx.size < 2:
            continue
        per.append(metrics(np.asarray(sim)[idx], np.asarray(act)[idx]))
    keys = [k for k in per[0] if k not in ("n_origin", "n_obs")]
    m = {k: float(np.mean([r[k] for r in per])) for k in keys}
    m["n_origin"] = float(np.mean([r["n_origin"] for r in per]))
    m["n_obs"] = float(np.mean([r["n_obs"] for r in per]))
    m["n_offset"] = len(per)
    return m


def align(models):
    """모델별 pred_start(예측 첫 주의 test CSV 행)로 교집합을 잡아 행을 맞춘다.

    GARCH-ST 는 train_garch_xpast.py:197-199 규칙, 나머지 셋은
    train_garch_flow.py:496-498 규칙으로 원점을 고르므로 개수가 다르다(181 vs 182).
    같은 정수 인덱스에 STRIDE 를 걸면 서로 다른 주를 가리키게 되므로,
    교집합을 잡은 뒤 같은 주에 대해서만 비교한다.
    """
    common = None
    for d in models.values():
        s = set(int(x) for x in d["pred_start"])
        common = s if common is None else (common & s)
    common = np.array(sorted(common), dtype=int)
    out = {}
    for name, d in models.items():
        ps = np.asarray(d["pred_start"], dtype=int)
        pos = {int(v): i for i, v in enumerate(ps)}
        sel = np.array([pos[v] for v in common], dtype=int)
        out[name] = dict(sim=np.asarray(d["sim"])[sel], act=np.asarray(d["act"])[sel])
    return out, common


def _keep_summary(path):
    """run_fold 가 덮어쓰지 않도록 기존 summary 를 잠시 치운다."""
    bak = path + ".thinbak"
    if os.path.exists(path):
        os.replace(path, bak)
        return bak
    return None


def _restore_summary(path, bak):
    if bak and os.path.exists(bak):
        os.replace(bak, path)


def macflow_arrays(fold, seed, device):
    """저장된 ckpt 로 재추론.  미래 조건 = 과거 52주의 마지막 관측값 flat 유지."""
    ctx = PS.load_fold_seed(fold, seed, device)
    if ctx is None:
        return None

    ckpt = torch.load(os.path.join(
        PS.RESULT_DIR, f"garch_flow_ar_{PS.TAG_PREFIX}_s{seed}_{fold}_best.pt"),
        map_location="cpu")
    meta = ckpt["meta"]
    cond_stats, target_stats = meta["cond_stats"], meta["target_stats"]
    gp = T.garch_preprocess_fold(PS.FOLDS_DIR, fold, PS.RESULT_DIR)
    _, Yte, _, _, _ = T.cached_load_windows_seq(
        gp["test"], cond_stats=cond_stats, target_stats=target_stats)
    valid_mask, z_te, df_te = T.compute_valid_mask(gp["test"], cond_stats)

    # 앵커 = 과거 52주의 마지막 관측 행 (w + PAST_LEN - 1).
    #   real_*_fut[:, 0] 은 미래 첫 주이므로 쓰지 않는다.
    n_w = z_te.shape[0] - T.PAST_LEN - T.FUTURE_LEN + 1
    last = z_te[T.PAST_LEN - 1: T.PAST_LEN - 1 + n_w][valid_mask]       # (n_orig, N_CH)
    tb_flat = last[:, ctx["ti"]:ctx["ti"] + 1] + np.zeros((1, T.FUTURE_LEN))
    mb_flat = last[:, ctx["mi"]:ctx["mi"] + 1] + np.zeros((1, T.FUTURE_LEN))

    sim_z = ZM.rollout_paths(ctx, tb_flat, mb_flat, seed, device)
    r = ctx["rescale"]
    sim_raw = PS.forward_garch_rescale(sim_z * r["tsd"] + r["tmu"], r["s2"], r["e2"],
                                       r["om"], r["al"], r["be"], r["mu"])

    # actual 은 스텝별 실현 σ 로 복원 (evaluate_test 와 동일, 정답 라벨이므로 누수 아님).
    tmu, tsd = float(target_stats["mean"]), float(target_stats["std"])
    gsig = df_te["garch_sigma"].to_numpy(float)
    gmu = df_te["garch_mu"].to_numpy(float)
    oidx = np.where(valid_mask)[0]
    sig = np.array([[gsig[int(o) + T.PAST_LEN + t] for t in range(T.FUTURE_LEN)]
                    for o in oidx])
    mu = np.array([[gmu[int(o) + T.PAST_LEN + t] for t in range(T.FUTURE_LEN)]
                   for o in oidx])
    actual_raw = (Yte.numpy() * tsd + tmu) * sig + mu

    if actual_raw.shape[0] != sim_raw.shape[0]:
        sys.exit(f"[FATAL] origin 수 불일치 {fold} s{seed}: "
                 f"actual {actual_raw.shape[0]} vs sim {sim_raw.shape[0]}")
    return dict(sim=sim_raw, act=actual_raw, pred_start=oidx + T.PAST_LEN)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 112)
    print(f"# 비중첩 비교 — 원점 STRIDE={STRIDE}(오프셋 {STRIDE}개 평균), 지평 13주,")
    print(f"#   미래 조건=원점 마지막값 flat (네 모델 공통), 구간=전역 풀링, 원점=교집합")
    print(f"# device={device}  n_sim={N_SIM}  MAC-Flow={PS.TAG_PREFIX} ({len(SEEDS)} seed)")
    print("#" * 112)

    out = {f: {} for f in FOLDS}
    for fold in FOLDS:
        print(f"\n===== {fold}")
        raw = {}

        try:
            sp = os.path.join(GX.RESULT_DIR, f"garch_xpast_{fold}_summary.json")
            bak = _keep_summary(sp)
            raw["GARCH-ST"] = GX.run_fold(fold)
            _restore_summary(sp, bak)
        except Exception as e:
            print(f"  [FAIL GARCH-ST] {e!r}")

        for mk in ("vae", "gan"):
            try:
                spv = os.path.join(PS.RESULT_DIR, f"{mk}_baseline_{fold}_summary.json")
                bakv = _keep_summary(spv)
                args = SimpleNamespace(model=mk, fold=fold, folds_dir=PS.FOLDS_DIR,
                                       out_dir=PS.RESULT_DIR, n_sim=N_SIM, seed=VG_SEED)
                raw[f"Cond{mk.upper()}"] = VG.run_fold(mk, fold, args, device)
                _restore_summary(spv, bakv)
            except Exception as e:
                print(f"  [FAIL {mk}] {e!r}")

        mac = []
        for seed in SEEDS:
            try:
                d = macflow_arrays(fold, seed, device)
                if d is not None:
                    mac.append(d)
            except Exception as e:
                print(f"  [FAIL MAC-Flow s{seed}] {e!r}")
        if mac:
            raw["MAC-Flow"] = mac[0]          # 정렬용 대표 (pred_start 는 시드 공통)

        if len(raw) < 2:
            print("  [skip] 정렬할 모델이 부족하다"); continue

        aligned, common = align(raw)
        print(f"  원점 교집합 {len(common)} 개 " +
              " ".join(f"{k}:{len(v['pred_start'])}" for k, v in raw.items()))

        for name, d in aligned.items():
            if name == "MAC-Flow":
                continue
            out[fold][name] = thin_all_offsets(d["sim"], d["act"])

        # MAC-Flow: 시드별로 같은 교집합 행을 뽑아 지표를 낸 뒤 시드 평균
        if mac:
            ps = np.asarray(mac[0]["pred_start"], dtype=int)
            pos = {int(v): i for i, v in enumerate(ps)}
            sel = np.array([pos[v] for v in common], dtype=int)
            per_seed = [thin_all_offsets(np.asarray(d["sim"])[sel],
                                         np.asarray(d["act"])[sel]) for d in mac]
            keys = list(per_seed[0].keys())
            m = {k: float(np.mean([r[k] for r in per_seed])) for k in keys}
            m["n_seed"] = len(per_seed)
            out[fold]["MAC-Flow"] = m

        for name in ("MAC-Flow", "GARCH-ST", "CondVAE", "CondGAN"):
            if name in out[fold]:
                r = out[fold][name]
                print(f"  [{name:<9}] 부분표본 원점 {r['n_origin']:.1f} × "
                      f"오프셋 {int(r['n_offset'])}")

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}")

    order = ["MAC-Flow", "GARCH-ST", "CondVAE", "CondGAN"]
    cols = [("n_origin", "원점"), ("crps", "CRPS"),
            ("cov50", "cov50"), ("cov80", "cov80"), ("cov95", "cov95"),
            ("skew_actual", "skew실측"), ("skew_sim", "skew모형")]
    print(f"\n{'='*104}")
    print(f"[요약]  STRIDE={STRIDE} 오프셋 평균 · 지평 13주 · 미래=원점 마지막값 flat · 원점 교집합")
    print("=" * 104)
    print(f"{'fold':<18}{'model':<11}" + "".join(f"{c[1]:>11}" for c in cols))
    for fold in FOLDS:
        for name in order:
            r = out[fold].get(name)
            if r is None:
                continue
            cells = []
            for k, _ in cols:
                v = r[k]
                cells.append(f"{v:>11.1f}" if k == "n_origin"
                             else f"{v:>11.5f}" if k == "crps" else f"{v:>11.4f}")
            print(f"{fold:<18}{name:<11}" + "".join(cells))
        print()
    print("[읽는 법] cov 는 명목 0.50/0.80/0.95 에 가까울수록, skew 는 실측과 부호·크기가 "
          "맞을수록, CRPS 는 낮을수록 좋다.")


if __name__ == "__main__":
    main()
