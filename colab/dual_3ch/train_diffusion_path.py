"""Conditional Path Diffusion — MLP 인코더 + DC(tbill, metab26w) → 13주 sp_return 한번에 생성.

설계 (사용자 2026-05-23):
  MLP 인코더로 과거 윈도우(52주 × 5채널: sp,tbill,ads,wti,metab) 를 인코딩 →
  그 출력 + DC(미래 tbill 13주 + 미래 metab26w 13주) 를 조건으로
  디퓨전(DDPM)이 미래 13주 sp_return 경로를 **AR 없이 한번에** 생성.
  ※ DC = volDC(sp_std_13w, sp_log_std_13w; scale 앵커) + 미래 tbill + 미래 metab26w.

동기: flow-AR 은 스텝별 재귀라 거시 신호가 prev_ret·per-step 에 희석돼 못 살았다.
      한번에(joint) 생성하면 거시가 13주 결합분포를 직접 빚을 수 있나?  데이터로 확인.
  주의: 위기 과소분산은 레짐 희소성 문제라 안 풀릴 수 있음.

평가: CRPS/cov/std_ratio/CVaR (디퓨전은 exact NLL 없음).  flow/GARCH 와 같은 gap29 fold.
Usage (Colab): !python colab/dual_3ch/train_diffusion_path.py
"""
import json
import math
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

RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
SEED = 2026

PAST_LEN = 52
FUT_LEN = 13
COND_COLS = ["sp_return", "tbill_wr", "ads_lag", "wti_wr", "metab_26w"]  # 과거 윈도우 → MLP
VOLDC_COLS = ["sp_std_13w", "sp_log_std_13w"]                            # DC: scale 앵커(origin)
FUT_TBILL = "tbill_wr"
FUT_METAB = "metab_26w"                                                   # DC: 미래 tbill+metab
PAST_DIM = PAST_LEN * len(COND_COLS)        # 260
DC_DIM = len(VOLDC_COLS) + FUT_LEN + FUT_LEN   # 28  (volDC 2 + 미래 tbill 13 + 미래 metab 13)
COND_DIM = PAST_DIM + DC_DIM                 # 288  (build 출력)
TARGET_DIM = FUT_LEN                         # 13

D_MODEL = 128       # MLP 인코더 출력
HIDDEN = 256
N_BLOCKS = 3
TEMB_DIM = 64
DIFF_STEPS = 200
MAX_EPOCH = 200
PATIENCE = 30
BATCH = 64
LR = 2e-4
WEIGHT_DECAY = 1e-4
N_SIM = 1000


# ===================================================================== #
# Data — origin 별 (cond 286, target 13), 전부 z-score (train stats)
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
        past = np.stack([Z[c][ps] for c in COND_COLS], axis=1)   # (52, 5)
        voldc = np.array([Z[c][t + PAST_LEN - 1] for c in VOLDC_COLS])  # origin-frozen
        fut_tb = Z[FUT_TBILL][fs]
        fut_mt = Z[FUT_METAB][fs]
        tgt = spz[fs]
        cvec = np.concatenate([past.reshape(-1), voldc, fut_tb, fut_mt])  # 260+2+13+13
        if np.any(np.isnan(cvec)) or np.any(np.isnan(tgt)):
            continue
        conds.append(cvec)
        targs.append(tgt)
    return (np.asarray(conds, dtype=np.float32),
            np.asarray(targs, dtype=np.float32))


# ===================================================================== #
# Model — MLP encoder(과거) + Denoiser(ResMLP).  eps_theta(x_t, t, [enc(past), DC])
# ===================================================================== #
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(x)


class PathDiffusionNet(nn.Module):
    """MLP 인코더(과거 윈도우) + DC(미래 tbill/metab) → ResMLP denoiser."""
    def __init__(self, dropout=0.1):
        super().__init__()
        # MLP 인코더: 과거 flatten(260) → d_model
        self.encoder = nn.Sequential(
            nn.Linear(PAST_DIM, HIDDEN), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(HIDDEN, D_MODEL))
        cond_dim = D_MODEL + DC_DIM            # 128 + 26 = 154
        self.temb_mlp = nn.Sequential(
            nn.Linear(TEMB_DIM, TEMB_DIM), nn.GELU(), nn.Linear(TEMB_DIM, TEMB_DIM))
        self.inp = nn.Linear(TARGET_DIM + cond_dim + TEMB_DIM, HIDDEN)
        self.blocks = nn.ModuleList([ResBlock(HIDDEN, dropout) for _ in range(N_BLOCKS)])
        self.out = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, TARGET_DIM))

    def forward(self, x_t, t, c):
        emb = self.encoder(c[:, :PAST_DIM])             # MLP(과거)
        cc = torch.cat([emb, c[:, PAST_DIM:]], dim=-1)   # [enc(past), DC]
        te = self.temb_mlp(timestep_embedding(t, TEMB_DIM))
        h = torch.relu(self.inp(torch.cat([x_t, cc, te], dim=-1)))
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


# ===================================================================== #
# DDPM
# ===================================================================== #
def cosine_schedule(T, s=0.008):
    x = torch.linspace(0, T, T + 1)
    acp = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    acp = acp / acp[0]
    betas = 1 - acp[1:] / acp[:-1]
    return torch.clip(betas, 1e-8, 0.999)


class Diffusion:
    def __init__(self, T=DIFF_STEPS, device="cpu"):
        self.T = T
        self.betas = cosine_schedule(T).to(device)
        self.alphas = 1.0 - self.betas
        self.acp = torch.cumprod(self.alphas, dim=0)
        self.acp_prev = torch.cat([torch.ones(1, device=device), self.acp[:-1]])
        self.sqrt_acp = torch.sqrt(self.acp)
        self.sqrt_1macp = torch.sqrt(1.0 - self.acp)
        self.device = device

    def q_sample(self, x0, t, noise):
        return self.sqrt_acp[t][:, None] * x0 + self.sqrt_1macp[t][:, None] * noise

    @torch.no_grad()
    def p_sample_loop(self, model, c, n):
        x = torch.randn(n, TARGET_DIM, device=self.device)
        for ti in reversed(range(self.T)):
            t = torch.full((n,), ti, device=self.device, dtype=torch.long)
            eps = model(x, t, c)
            alpha = self.alphas[ti]; acp = self.acp[ti]
            mean = (x - (1 - alpha) / torch.sqrt(1 - acp) * eps) / torch.sqrt(alpha)
            if ti > 0:
                var = self.betas[ti] * (1 - self.acp_prev[ti]) / (1 - acp)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
        return x


# ===================================================================== #
# Train + evaluate one fold
# ===================================================================== #
def run_fold(fold, device):
    summary_path = os.path.join(RESULT_DIR, f"diffusion_path_{fold}_summary.json")
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
    model = PathDiffusionNet().to(device)
    diff = Diffusion(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"  params = {sum(p.numel() for p in model.parameters()):,}  DDPM steps={DIFF_STEPS}")

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Ctr), torch.from_numpy(Ytr)),
                       batch_size=BATCH, shuffle=True)
    Cva_d = torch.from_numpy(Cva).to(device); Yva_d = torch.from_numpy(Yva).to(device)

    best_val = float("inf"); best_state = None; best_ep = 0; bad = 0
    for ep in range(1, MAX_EPOCH + 1):
        model.train()
        for cb, yb in tr_dl:
            cb = cb.to(device); yb = yb.to(device)
            t = torch.randint(0, diff.T, (yb.shape[0],), device=device)
            noise = torch.randn_like(yb)
            loss = ((model(diff.q_sample(yb, t, noise), t, cb) - noise) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for _ in range(4):
                t = torch.randint(0, diff.T, (Yva_d.shape[0],), device=device)
                noise = torch.randn_like(Yva_d)
                vloss += ((model(diff.q_sample(Yva_d, t, noise), t, Cva_d) - noise) ** 2).mean().item()
            vloss /= 4
        if vloss < best_val - 1e-5:
            best_val = vloss; best_ep = ep; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep <= 5 or ep % 20 == 0:
            print(f"    ep{ep:3d}: train_eps_mse={loss.item():.4f}  val={vloss:.4f}  "
                  f"(best {best_val:.4f}@{best_ep})")
        if bad >= PATIENCE:
            print(f"    [early stop] ep{ep} (from {best_ep})"); break
    model.load_state_dict(best_state)

    print(f"  [sample] n_sim={N_SIM}")
    model.eval()
    tmu, tsd = stats["sp_return"]
    n_org = len(Cte)
    sim_z = np.empty((n_org, N_SIM, TARGET_DIM), dtype=np.float32)
    chunk = 16
    with torch.no_grad():
        for s in range(0, n_org, chunk):
            e = min(n_org, s + chunk); k = e - s
            c = torch.from_numpy(Cte[s:e]).to(device)
            c_rep = c.unsqueeze(1).expand(k, N_SIM, COND_DIM).reshape(k * N_SIM, COND_DIM)
            x = diff.p_sample_loop(model, c_rep, k * N_SIM)
            sim_z[s:e] = x.reshape(k, N_SIM, TARGET_DIM).cpu().numpy()

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
        fold=fold, model="Conditional Path Diffusion (MLP enc + DC tbill/metab26w, DDPM)",
        cond_dim=COND_DIM, target_dim=TARGET_DIM, diff_steps=DIFF_STEPS,
        n_train=len(Ctr), n_val=len(Cva), n_test=len(Cte),
        best_epoch=best_ep, best_val_eps_mse=best_val, seed=SEED,
        test_eval=dict(
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
    print(f"[diffusion-path] device={device}  MLP enc + DC(tbill,metab26w)  {len(FOLDS)} fold")
    for fold in FOLDS:
        if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                   for s in ("train", "val", "test")):
            print(f"[skip {fold}] fold CSV 없음"); continue
        try:
            run_fold(fold, device)
        except Exception as e:
            print(f"[FAIL] {fold}: {e!r}")

    print("\n" + "=" * 96)
    print("path-diffusion(MLP+DC) vs flow(cond/directboth) vs GARCH  (같은 gap29 fold)")
    print("  CRPS 낮을수록 | cov95 0.95 | std_ratio 1.0 근접")
    print("=" * 96)

    def row(path, sub=None):
        if not os.path.exists(path):
            return None
        d = json.load(open(path))
        return d.get(sub, d) if sub else d

    for label, get in [
        ("diffusion",   lambda f: row(os.path.join(RESULT_DIR, f"diffusion_path_{f}_summary.json"), "test_eval")),
        ("flow cond",   lambda f: row(os.path.join(RESULT_DIR, f"mamba_flow_ar_selfstat_cond_m26_mlp_s2026_{f}_summary.json"), "test_eval")),
        ("flow direct", lambda f: row(os.path.join(RESULT_DIR, f"mamba_flow_ar_selfstat_directboth_m26_mlp_s2026_{f}_summary.json"), "test_eval")),
        ("GARCH-N",     lambda f: row(os.path.join(RESULT_DIR, f"garch_pure_{f}_summary.json"))),
    ]:
        print(f"\n[{label}]   {'fold':<16}{'CRPS':>9}{'cov95':>8}{'std_ratio':>11}{'CVaR5Δ':>10}")
        for fold in FOLDS:
            d = get(fold)
            if not d:
                print(f"{'':<6}{fold:<16}  (없음)"); continue

            def g(k):
                return d.get(k) or 0
            print(f"{'':<6}{fold:<16}{g('crps_pooled'):>9.5f}{g('coverage_95'):>8.3f}"
                  f"{g('std_ratio'):>11.3f}{g('cvar_5pct_diff'):>10.5f}")


if __name__ == "__main__":
    main()
