"""§4.3.2 (개정) 변수 중요도 — Permutation Importance on 꼬리/시나리오 지표.

기존 analyze_feature_importance_rawvol.py 는 ΔNLL/week 로 측정했으나, NLL 은
본 논문 어디에서도 평가지표로 쓰지 않으며(논문은 CRPS·EMD·coverage·skew·CVaR·IHL 사용),
NLL 은 분포 *몸통(중심·스케일)* 밀도 적합도라 변동성 스케일(sp_std_13w)이 압도적으로 지배하여
거시(tbill·metab)의 *꼬리/경로* 기여를 잡지 못한다.

본 스크립트는 각 입력 채널을 origin 간 셔플(관계 파괴)한 뒤 *실제로 path 를 샘플링*하여,
sim_metrics 의 꼬리/시나리오 지표 변화(Δ)를 중요도로 측정한다.
  · 1차 지표: uw_cvar10 (intra-horizon loss 10%, 경로의존 꼬리) — 논문 핵심 위험지표.
  · 보조:     cvar1 (per-step 수익률 1% 꼬리), skew (좌꼬리 비대칭).
  ΔX = metric(shuffled X) − metric(base).
  uw_cvar10·cvar1·skew 모두 음수(깊을수록 −) → ΔX > 0 = 셔플 시 꼬리가 *얕아짐*
  = 그 입력이 깊은 꼬리/비대칭에 기여했음(중요).  |ΔX| 클수록 그 지표를 좌우.

샘플링: analyze_pathshape_rawvol.load_fold_seed 로 origin별 실현 미래경로
        (real_tb_fut/real_mb_fut, z) + 과거(Xte) + extra context 로딩,
        ar_sample 으로 origin별 N_SIM path 생성 → sim_metrics.
  paired seed: base 와 각 shuffle 샘플링에 동일 torch.manual_seed
               → 샘플링 난수 상쇄, Δ = 순수 조건화(셔플) 효과.
채널 셔플 위치:
  tbill_wr  : 과거 Xte[:,:,TBILL_CH] + 미래 real_tb_fut  (동일 perm — 한 origin 의 금리궤적 통째 교환)
  metab_13w : 과거 Xte[:,:,metab]    + 미래 real_mb_fut  (동일 perm)
  ads_lag/wti_wr : 과거만 (미래 마스크되므로 Xte 과거 채널만)
  sp_std_13w/sp_skew_13w : extra_context 열
  (sp_return = 목표/AR seed → 제외)

n_perm 평균(셔플 노이즈), 4 fold × 5 seed.  json 캐시로 재진입.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/analyze_feature_importance_tail_rawvol.py
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

# analyze_pathshape_rawvol 가 import 시 patch_rawvol() 적용(forward σ origin-frozen) + 본모형 상수 설정
import analyze_pathshape_rawvol as PS                                 # noqa: E402

RESULT_DIR = PS.RESULT_DIR
CACHE_DIR = os.path.join(RESULT_DIR, f"feat_importance_tail_cache{PS.CACHE_SUFFIX}")
os.makedirs(CACHE_DIR, exist_ok=True)

FOLDS = PS.FOLDS
LABELS = {"F_gfc": "금융위기", "F_long_A": "회복기",
          "F_long_B_origin": "코로나", "F_long": "긴축기"}
SEEDS = PS.SEEDS
N_SIM = PS.N_SIM
CHUNK = PS.CHUNK
PAST_LEN = PS.PAST_LEN
FUTURE_LEN = PS.FUTURE_LEN
N_PERM = 3                                  # 셔플 반복(샘플링 비용 큼; 노이즈는 mean±std 로 노출)

# (feature, kind) — kind: "enc"=인코더 채널(Xte), "extra"=extra context 열
# extra 항목은 PS.DC_COLS_LIST 에 실제 존재하는 채널만 셔플(fpath_novol 본모형은 sp_std_13w 없음 → 자동 제외).
PERMUTE = [
    ("tbill_wr", "enc"),
    ("metab_13w", "enc"),
    ("ads_lag", "enc"),
    ("wti_wr", "enc"),
]
PERMUTE += [(c, "extra") for c in ("sp_std_13w", "sp_skew_13w") if c in PS.DC_COLS_LIST]
METRIC_KEYS = ["uw_cvar10", "cvar1", "skew"]      # 1차 uw_cvar10, 보조 cvar1/skew


@torch.no_grad()
def sample_realized(model, Xte, extra, last_sp, fut_tb, fut_mb, device):
    """origin별 실현(혹은 셔플된) 미래경로로 ar_sample.

    fut_tb : (n, FUTURE_LEN) z,  fut_mb : (n, FUTURE_LEN) z.  과거/extra 는 Xte/extra 그대로.
    """
    n = Xte.shape[0]
    out = []
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        x_past = Xte[s:e, :PAST_LEN, :]
        lsp = last_sp[s:e]
        ex = extra[s:e]
        ftb = fut_tb[s:e].contiguous()                       # (k, FUTURE_LEN)
        fmac = fut_mb[s:e].unsqueeze(-1).contiguous()        # (k, FUTURE_LEN, 1)
        sim = model.ar_sample(x_past, ftb, lsp, N_SIM,
                              extra_context=ex, future_macro_z=fmac)
        out.append(sim.cpu())
    return torch.cat(out, dim=0).numpy()


@torch.no_grad()
def run_fold_seed(fold, seed, device):
    ctx = PS.load_fold_seed(fold, seed, device)
    if ctx is None:
        return None
    model = ctx["model"]; Xte = ctx["Xte"]; extra = ctx["extra"]
    last_sp = ctx["last_sp"]; rescale = ctx["rescale"]
    n = Xte.shape[0]
    fut_tb0 = torch.tensor(ctx["real_tb_fut"], device=device, dtype=Xte.dtype)   # (n, FUTURE_LEN)
    fut_mb0 = torch.tensor(ctx["real_mb_fut"], device=device, dtype=Xte.dtype)

    enc_idx = {nm: PS.ENC_COLS.index(nm) for nm, k in PERMUTE if k == "enc"}
    extra_idx = {nm: PS.DC_COLS_LIST.index(nm) for nm, k in PERMUTE if k == "extra"}

    def _metrics(Xin, exin, ftb, fmb):
        torch.manual_seed(seed)                               # paired 샘플링 난수
        sim = sample_realized(model, Xin, exin, last_sp, ftb, fmb, device)
        return PS.sim_metrics(sim, rescale)

    base = _metrics(Xte, extra, fut_tb0, fut_mb0)

    g = torch.Generator().manual_seed(seed)                   # perm 재현용(cpu)
    imp = {}
    for name, kind in PERMUTE:
        deltas = {k: [] for k in METRIC_KEYS}
        for _ in range(N_PERM):
            perm = torch.randperm(n, generator=g).to(device)
            Xp = Xte.clone(); exp_ = extra.clone()
            ftb = fut_tb0; fmb = fut_mb0
            if kind == "enc":
                ci = enc_idx[name]
                Xp[:, :, ci] = Xte[perm][:, :, ci]           # 과거 채널 셔플
                if name == "tbill_wr":
                    ftb = fut_tb0[perm]                       # 미래 금리경로 동일 perm
                elif name == "metab_13w":
                    fmb = fut_mb0[perm]                       # 미래 유동성경로 동일 perm
            else:
                j = extra_idx[name]
                exp_[:, j] = extra[perm][:, j]               # extra context 열 셔플
            m = _metrics(Xp, exp_, ftb, fmb)
            for k in METRIC_KEYS:
                deltas[k].append(m[k] - base[k])
        row = {}
        for k in METRIC_KEYS:
            arr = np.asarray(deltas[k], float)
            row[k] = float(arr.mean()); row[k + "_std"] = float(arr.std())
        imp[name] = row

    return dict(n_origin=int(n), base={k: base[k] for k in METRIC_KEYS}, importance=imp)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#" * 100)
    print(f"# §4.3.2 (개정) Permutation Importance — 꼬리/시나리오 지표 Δ (LOCKED {PS.TAG_PREFIX}, n_perm={N_PERM})")
    print("#  지표: uw_cvar10(IHL,1차) / cvar1 / skew.  ΔX>0 = 셔플 시 꼬리 얕아짐 = 그 입력이 깊은 꼬리에 기여(중요)")
    print(f"#  device={device}, N_SIM={N_SIM}, origins≤{PS.N_ORIGIN_MAX}")
    print("#" * 100)

    for fold in FOLDS:
        for seed in SEEDS:
            c = os.path.join(CACHE_DIR, f"{fold}_s{seed}.json")
            if os.path.exists(c):
                print(f"[skip] {fold} s{seed}"); continue
            r = run_fold_seed(fold, seed, device)
            if r is None:
                continue
            json.dump(r, open(c, "w"), indent=2)
            print(f"[done] {fold} s{seed}  base uw_cvar10={r['base']['uw_cvar10']:+.4f} "
                  f"cvar1={r['base']['cvar1']:+.4f} skew={r['base']['skew']:+.3f}")

    _summarize()


def _summarize():
    feats = [nm for nm, _ in PERMUTE]
    print("\n" + "=" * 100)
    print("[집계] Permutation Δ — 4 fold × 5 seed 평균.  (ΔX>0 = 셔플 시 꼬리 얕아짐 = 중요)")

    pooled = {k: {f: [] for f in feats} for k in METRIC_KEYS}
    for k in METRIC_KEYS:
        print(f"\n--- Δ{k} (per fold) " + "-" * (80 - len(k)))
        print(f"  {'fold':>10}" + "".join(f"{f:>14}" for f in feats))
        for fold in FOLDS:
            rs = [json.load(open(os.path.join(CACHE_DIR, f"{fold}_s{s}.json")))
                  for s in SEEDS if os.path.exists(os.path.join(CACHE_DIR, f"{fold}_s{s}.json"))]
            if not rs:
                print(f"  {LABELS.get(fold, fold):>10}  (결과 없음)"); continue
            line = f"  {LABELS.get(fold, fold):>10}"
            for f in feats:
                vals = [r["importance"][f][k] for r in rs]
                pooled[k][f].extend(vals)
                line += f"  {np.mean(vals):>+12.4f}"
            print(line)

    print("\n" + "=" * 100)
    print("[중요도 순위] |전체 평균 Δ| 큰 순 (입력이 해당 꼬리지표를 얼마나 좌우하나)")
    for k in METRIC_KEYS:
        rank = sorted(feats, key=lambda f: -abs(np.mean(pooled[k][f])))
        print(f"\n  [{k}]")
        for f in rank:
            m = np.mean(pooled[k][f]); sd = np.std(pooled[k][f])
            print(f"    {f:>14}: Δ={m:+.4f} ± {sd:.4f}")
    print("\n[판정] tbill_wr·metab_13w 가 uw_cvar10/skew 에서 |Δ| 상위면, NLL 에선 묻혔던")
    print("  거시 입력의 *꼬리/경로* 기여가 드러나는 것 (§4.1·§4.3.1 ablation 과 정합).")


if __name__ == "__main__":
    main()
