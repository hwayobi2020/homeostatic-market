"""Joint Flow — 13주 sp_return 경로를 13차원 conditional NSF 로 한번에 생성 (AR rollout 없음).

사용자 가설(2026-05-23): flow-AR 이 거시 신호를 못 살린 게 'AR 스텝별 희석' 탓일 수 있다.
  → 같은 flow class 를 AR 없이 13차원 joint 로 생성하면 살아나나?
  flow-AR(현재) vs flow-joint(이 파일) vs diffusion 비교로 'AR vs joint' confound 를 분리.
  flow 라 exact NLL 도 나옴 → flow-AR 과 NLL 직접 비교 가능(둘 다 log p(13주 경로|조건)).

구성 (diffusion 과 동일 conditioning, vol-adjustment 안 씀 = 평소 전역 z-score):
  조건 = MLP enc(과거 52×5=260→128) + DC(volDC 2 + 미래 tbill 13 + 미래 metab26w 13) = 156
  대상 = 미래 13주 sp_return (13차원, z-score)
  flow = 13-dim Masked RQ-NSF (13차원 autoregressive) × n_layers, 사이에 ReversePermutation.
평가: per-week NLL(exact) + CRPS/cov/std_ratio/CVaR. flow-AR/diffusion/GARCH 같은 gap29 fold 비교.

Usage (Colab): !python colab/dual_3ch/train_flow_joint.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from train_mamba_flow_ar import (crps_pooled, compute_var,   # noqa: E402
                                 compute_cvar, compute_emd_1d)
try:
    from nflows.flows.base import Flow
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms import (
        CompositeTransform,
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    )
    from nflows.transforms.permutations import ReversePermutation
except ImportError:
    sys.exit("FATAL: nflows required.  pip install nflows")

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEED = 2026

PAST_LEN = 52
FUT_LEN = 13
COND_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_26w"]  # 과거 윈도우 → MLP
VOLDC_COLS = ["sp_std_13w", "sp_log_std_13w"]
FUT_TBILL = "tbill_wr"
FUT_METAB = "metab_26w"
PAST_DIM = PAST_LEN * len(COND_COLS)            # 260
DC_DIM = len(VOLDC_COLS) + FUT_LEN + FUT_LEN    # 28
COND_DIM = PAST_DIM + DC_DIM                     # 288 (build 출력)
TARGET_DIM = FUT_LEN                             # 13

D_MODEL = 128
HIDDEN = 256
N_FLOW_LAYERS = 6
N_FLOW_HIDDEN = 64
N_FLOW_BLOCKS = 2
N_FLOW_BINS = 16
TAIL_BOUND = 10.0
DROPOUT = 0.1
MAX_EPOCH = 200
PATIENCE = 40
BATCH = 64
LR = 1e-4
WEIGHT_DECAY = 0.5
N_SIM = 1000


# ===================================================================== #
# Data — origin 별 (cond 288, target 13), 전역 z-score (train stats)
# ===================================================================== #
def col_stats(df, cols):
    out = {}
    for c in cols:
        a = pd.to_numeric(df[c], errors="coerce").values
        out[c] = (float(np.nanmean(a)), float(np.nanstd(a, ddof=1) + 1e-8))
    return out


def build(df, stats):
    Z = {c: (pd.to_numeric(df[c], errors="coerce").values - stats[c][0]) / stats[c][1]
         for c in COND_COLS + VOLDC_COLS}
    spz = Z["sp_return"]
    n = len(df)
    conds, targs = [], []
    for t in range(n - PAST_LEN - FUT_LEN + 1):
        ps = slice(t, t + PAST_LEN)
        fs = slice(t + PAST_LEN, t + PAST_LEN + FUT_LEN)
        past = np.stack([Z[c][ps] for c in COND_COLS], axis=1)          # (52, 5)
        voldc = np.array([Z[c][t + PAST_LEN - 1] for c in VOLDC_COLS])  # origin-frozen
        fut_tb = Z[FUT_TBILL][fs]
        fut_mt = Z[FUT_METAB][fs]
        tgt = spz[fs]                                                   # 미래 13주 sp_return z
        cvec = np.concatenate([past.reshape(-1), voldc, fut_tb, fut_mt])
        if np.any(np.isnan(cvec)) or np.any(np.isnan(tgt)):
            continue
        conds.append(cvec)
        targs.append(tgt)
    return (np.asarray(conds, dtype=np.float32),
            np.asarray(targs, dtype=np.float32))


# ===================================================================== #
# Model — MLP encoder(과거) + 13-dim conditional NSF (joint, AR 없음)
# ===================================================================== #
class FlowJoint(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(PAST_DIM, HIDDEN), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(HIDDEN, D_MODEL))
        cond_dim = D_MODEL + DC_DIM                 # 128 + 28 = 156
        base = StandardNormal(shape=[TARGET_DIM])
        transforms = []
        for i in range(N_FLOW_LAYERS):
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=TARGET_DIM, hidden_features=N_FLOW_HIDDEN,
                context_features=cond_dim, num_blocks=N_FLOW_BLOCKS,
                num_bins=N_FLOW_BINS, tails="linear", tail_bound=TAIL_BOUND,
                dropout_probability=dropout))
            if i < N_FLOW_LAYERS - 1:
                transforms.append(ReversePermutation(features=TARGET_DIM))
        self.flow = Flow(CompositeTransform(transforms), base)

    def _ctx(self, c):
        emb = self.encoder(c[:, :PAST_DIM])
        return torch.cat([emb, c[:, PAST_DIM:]], dim=-1)   # [enc(past), DC]

    def log_prob(self, x, c):
        return self.flow.log_prob(x, context=self._ctx(c))   # (B,) log p(13주|조건)

    @torch.no_grad()
    def sample(self, n, c):
        return self.flow.sample(n, context=self._ctx(c))     # (B, n, 13)


# ===================================================================== #
# Train + evaluate one fold
# ===================================================================== #
def run_fold(fold, device):
    summary_path = os.path.join(RESULT_DIR, f"flow_joint_{fold}_summary.json")
    if os.path.exists(summary_path):
        print(f"[skip] {os.path.basename(summary_path)}"); return
    tr = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_train.csv"))
    va = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_val.csv"))
    te = pd.read_csv(os.path.join(FOLDS_DIR, f"{fold}_test.csv"))
    stats = col_stats(tr, COND_COLS + VOLDC_COLS)
    Ctr, Ytr = build(tr, stats); Cva, Yva = build(va, stats); Cte, Yte = build(te, stats)
    print(f"\n{'='*70}\n fold={fold}  cond_dim={COND_DIM}(MLP enc {PAST_DIM}+DC {DC_DIM})  "
          f"target={TARGET_DIM}")
    print(f"  windows train/val/test = {len(Ctr)}/{len(Cva)}/{len(Cte)}")
    if min(len(Ctr), len(Cva), len(Cte)) == 0:
        print(f"[skip {fold}] empty"); return

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = FlowJoint().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"  params = {sum(p.numel() for p in model.parameters()):,}  "
          f"flow={N_FLOW_LAYERS}L/{N_FLOW_HIDDEN}h/{N_FLOW_BINS}bins (13-dim joint)")

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Ctr), torch.from_numpy(Ytr)),
                       batch_size=BATCH, shuffle=True)
    Cva_d = torch.from_numpy(Cva).to(device); Yva_d = torch.from_numpy(Yva).to(device)

    best_val = float("inf"); best_state = None; best_ep = 0; bad = 0
    for ep in range(1, MAX_EPOCH + 1):
        model.train()
        for cb, yb in tr_dl:
            cb = cb.to(device); yb = yb.to(device)
            loss = -model.log_prob(yb, cb).mean() / FUT_LEN     # per-week NLL
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vnll = (-model.log_prob(Yva_d, Cva_d).mean() / FUT_LEN).item()   # exact, 결정적
        if vnll < best_val - 1e-5:
            best_val = vnll; best_ep = ep; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep <= 5 or ep % 20 == 0:
            print(f"    ep{ep:3d}: train_nll/wk={loss.item():.4f}  val_nll/wk={vnll:.4f}  "
                  f"(best {best_val:.4f}@{best_ep})")
        if bad >= PATIENCE:
            print(f"    [early stop] ep{ep} (from {best_ep})"); break
    model.load_state_dict(best_state)

    # exact test NLL (flow-AR 과 직접 비교 가능)
    Cte_d = torch.from_numpy(Cte).to(device); Yte_d = torch.from_numpy(Yte).to(device)
    model.eval()
    with torch.no_grad():
        test_nll = (-model.log_prob(Yte_d, Cte_d).mean() / FUT_LEN).item()

    print(f"  [sample] n_sim={N_SIM}   test_nll/wk={test_nll:.4f}")
    n_org = len(Cte)
    tmu, tsd = stats["sp_return"]
    sim_z = np.empty((n_org, N_SIM, TARGET_DIM), dtype=np.float32)
    chunk = 16
    with torch.no_grad():
        for s in range(0, n_org, chunk):
            e = min(n_org, s + chunk)
            c = torch.from_numpy(Cte[s:e]).to(device)
            sim_z[s:e] = model.sample(N_SIM, c).cpu().numpy()    # (k, N_SIM, 13)

    sim_raw = sim_z * tsd + tmu
    act_raw = Yte * tsd + tmu
    crps_m, crps_s = crps_pooled(sim_raw, act_raw)
    af = act_raw.ravel(); sf = sim_raw.ravel()
    std_a = float(af.std(ddof=1)); std_s = float(sf.std(ddof=1))
    cvar5_a = compute_cvar(af, 0.05); cvar5_s = compute_cvar(sf, 0.05)
    cvar1_a = compute_cvar(af, 0.01); cvar1_s = compute_cvar(sf, 0.01)
    var1_a = compute_var(af, 0.01); var1_s = compute_var(sf, 0.01)
    emd = compute_emd_1d(sf, af)
    cov = {}
    for lvl, lo_q, hi_q in [(50, 25, 75), (80, 10, 90), (95, 2.5, 97.5)]:
        lo = np.percentile(sim_raw, lo_q, axis=1); hi = np.percentile(sim_raw, hi_q, axis=1)
        cov[lvl] = float(((act_raw >= lo) & (act_raw <= hi)).mean())
    print(f"  CRPS={crps_m:.5f}  std a/s/ratio={std_a:.5f}/{std_s:.5f}/{std_s/std_a:.3f}  "
          f"cov 50/80/95={cov[50]:.3f}/{cov[80]:.3f}/{cov[95]:.3f}")

    summary = dict(
        fold=fold, model="Joint Flow (MLP enc + DC, 13-dim conditional NSF)",
        cond_dim=COND_DIM, target_dim=TARGET_DIM,
        n_train=len(Ctr), n_val=len(Cva), n_test=len(Cte),
        best_epoch=best_ep, best_val_nll_per_week=best_val, seed=SEED,
        test_eval=dict(
            per_week_nll_z=test_nll,
            crps_pooled=crps_m, crps_std=crps_s, emd=emd,
            std_actual=std_a, std_sim=std_s, std_ratio=std_s / std_a,
            coverage_50=cov[50], coverage_80=cov[80], coverage_95=cov[95],
            var_1pct_diff=var1_s - var1_a, cvar_1pct_diff=cvar1_s - cvar1_a,
            cvar_5pct_diff=cvar5_s - cvar5_a),
    )
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  saved {os.path.basename(summary_path)}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[flow-joint] device={device}  13-dim NSF (joint, AR 없음)  {len(FOLDS)} fold")
    for fold in FOLDS:
        if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                   for s in ("train", "val", "test")):
            print(f"[skip {fold}] fold CSV 없음"); continue
        try:
            run_fold(fold, device)
        except Exception as e:
            print(f"[FAIL] {fold}: {e!r}")

    # ── 비교: flow-joint vs flow-AR(cond) vs diffusion vs GARCH ──
    print("\n" + "=" * 100)
    print("flow-joint vs flow-AR(cond) vs diffusion vs GARCH  (같은 gap29 fold)")
    print("  CRPS 낮을수록 | cov95 0.95 | std_ratio 1.0 | (NLL 은 두 flow 끼리만 비교)")
    print("=" * 100)

    def row(path, sub=None):
        if not os.path.exists(path):
            return None
        d = json.load(open(path))
        return d.get(sub, d) if sub else d

    for label, get in [
        ("flow-joint",  lambda f: row(os.path.join(RESULT_DIR, f"flow_joint_{f}_summary.json"), "test_eval")),
        ("flow-AR cond", lambda f: row(os.path.join(RESULT_DIR, f"mamba_flow_ar_selfstat_cond_m26_mlp_s2026_{f}_summary.json"), "test_eval")),
        ("diffusion",   lambda f: row(os.path.join(RESULT_DIR, f"diffusion_path_{f}_summary.json"), "test_eval")),
        ("GARCH-N",     lambda f: row(os.path.join(RESULT_DIR, f"garch_pure_{f}_summary.json"))),
    ]:
        print(f"\n[{label}]   {'fold':<16}{'NLL/wk':>9}{'CRPS':>9}{'cov95':>8}{'std_ratio':>11}{'CVaR5Δ':>10}")
        for fold in FOLDS:
            d = get(fold)
            if not d:
                print(f"{'':<6}{fold:<16}  (없음)"); continue

            def g(k):
                return d.get(k) or 0
            print(f"{'':<6}{fold:<16}{g('per_week_nll_z'):>9.4f}{g('crps_pooled'):>9.5f}"
                  f"{g('coverage_95'):>8.3f}{g('std_ratio'):>11.3f}{g('cvar_5pct_diff'):>10.5f}")
    print("\n해석: flow-joint 의 NLL/CRPS/std_ratio 가 flow-AR 보다 좋으면 → 'AR 희석'이 원인이었던 것. "
          "비슷하면 → AR 무관, 신호/모델 한계. (NLL 은 flow-joint vs flow-AR 만 같은 class 비교)")


if __name__ == "__main__":
    main()
