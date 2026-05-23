"""Conditional Path Diffusion — MLP 인코더 + DC(tbill, metab26w) → 13주 sp_return 한번에 생성.

설계 (사용자 2026-05-23):
  MLP 인코더로 과거 윈도우(52주 × 5채널: sp,tbill,ads,wti,metab) 를 인코딩 →
  그 출력 + DC(미래 tbill 13주 + 미래 metab26w 13주) 를 조건으로
  디퓨전(DDPM)이 미래 13주 sp_return 경로를 **AR 없이 한번에** 생성.
  ※ DC = volDC(sp_std_13w, sp_log_std_13w) + 미래 tbill + 미래 metab26w.
  ※ vol-adjustment(GARCH식): target = (return − drift)/최근변동성(sp_std_13w) 로 표준화.
    디퓨전은 표준화 혁신 분포만 학습, scale 은 최근변동성이 잡음 → vol-clustering 수입
    (GARCH 가 이기던 무기). 생성 후 다시 최근변동성을 곱해 raw return 복원.

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
HIDDEN = 384
N_BLOCKS = 4
TEMB_DIM = 64
DIFF_STEPS = 200
MAX_EPOCH = 200
PATIENCE = 50
BATCH = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.999
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
    sp_raw = pd.to_numeric(df["sp_return"], errors="coerce").values
    vol_raw = pd.to_numeric(df["sp_std_13w"], errors="coerce").values   # 최근 실현변동성(scale)
    mu = stats["sp_return"][0]                                          # drift(상수 평균)
    n = len(df)
    conds, targs, sigmas = [], [], []
    for t in range(n - PAST_LEN - FUT_LEN + 1):
        ps = slice(t, t + PAST_LEN)
        fs = slice(t + PAST_LEN, t + PAST_LEN + FUT_LEN)
        past = np.stack([Z[c][ps] for c in COND_COLS], axis=1)   # (52, 5)
        voldc = np.array([Z[c][t + PAST_LEN - 1] for c in VOLDC_COLS])  # origin-frozen
        fut_tb = Z[FUT_TBILL][fs]
        fut_mt = Z[FUT_METAB][fs]
        sigma = vol_raw[t + PAST_LEN - 1]                        # origin 시점 최근변동성
        # vol-adjustment(GARCH식): target = (return − drift)/최근변동성 = 표준화 혁신
        tgt = (sp_raw[fs] - mu) / (sigma + 1e-8)
        cvec = np.concatenate([past.reshape(-1), voldc, fut_tb, fut_mt])  # 260+2+13+13
        if (np.any(np.isnan(cvec)) or np.any(np.isnan(tgt))
                or np.isnan(sigma) or sigma <= 0):
            continue
        conds.append(cvec)
        targs.append(tgt)
        sigmas.append(sigma)
    return (np.asarray(conds, dtype=np.float32),
            np.asarray(targs, dtype=np.float32),
            np.asarray(sigmas, dtype=np.float32))


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
    def p_sample_loop(self, model, c, n, clip=8.0):
        """clip_denoised DDPM: eps→x0 변환 후 z-범위 clip → posterior mean 재계산.
        clip 없으면 약한 denoiser + 1/√α 증폭이 누적돼 발산(샘플 std 폭발)."""
        x = torch.randn(n, TARGET_DIM, device=self.device)
        for ti in reversed(range(self.T)):
            t = torch.full((n,), ti, device=self.device, dtype=torch.long)
            eps = model(x, t, c)
            acp = self.acp[ti]; acp_prev = self.acp_prev[ti]
            alpha = self.alphas[ti]; beta = self.betas[ti]
            # eps → 예측 x0, clip (clip_denoised) — 발산 방지
            x0 = (x - torch.sqrt(1 - acp) * eps) / torch.sqrt(acp)
            x0 = torch.clamp(x0, -clip, clip)
            # posterior q(x_{t-1}|x_t,x0) mean (Ho et al. 2020 eq.7)
            mean = (torch.sqrt(acp_prev) * beta / (1 - acp) * x0
                    + torch.sqrt(alpha) * (1 - acp_prev) / (1 - acp) * x)
            if ti > 0:
                var = beta * (1 - acp_prev) / (1 - acp)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
        return x


# ===================================================================== #
# EMA — 가중치 지수이동평균 (디퓨전 샘플 품질 안정화)
# ===================================================================== #
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())


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
    Ctr, Ytr, _ = build(tr, stats); Cva, Yva, _ = build(va, stats); Cte, Yte, Ste = build(te, stats)
    print(f"\n{'='*70}\n fold={fold}  cond_dim={COND_DIM}(MLP enc {PAST_DIM}+DC {DC_DIM})  "
          f"target={TARGET_DIM}")
    print(f"  windows train/val/test = {len(Ctr)}/{len(Cva)}/{len(Cte)}")
    if min(len(Ctr), len(Cva), len(Cte)) == 0:
        print(f"[skip {fold}] empty"); return

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = PathDiffusionNet().to(device)
    diff = Diffusion(device=device)
    ema = EMA(model, EMA_DECAY)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"  params = {sum(p.numel() for p in model.parameters()):,}  "
          f"DDPM steps={DIFF_STEPS}  EMA={EMA_DECAY}")

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(Ctr), torch.from_numpy(Ytr)),
                       batch_size=BATCH, shuffle=True)
    Cva_d = torch.from_numpy(Cva).to(device); Yva_d = torch.from_numpy(Yva).to(device)
    # 고정 val grid: t 균등분포 + 고정 noise → 노이즈 없는 안정적 early stop
    torch.manual_seed(SEED + 1)
    n_vw = Yva_d.shape[0]
    val_t = ((torch.arange(n_vw, device=device).float() * diff.T) / n_vw).long().clamp_(0, diff.T - 1)
    val_noise = torch.randn_like(Yva_d)
    val_xt = diff.q_sample(Yva_d, val_t, val_noise)

    best_val = float("inf"); best_state = None; best_ep = 0; bad = 0
    for ep in range(1, MAX_EPOCH + 1):
        model.train()
        for cb, yb in tr_dl:
            cb = cb.to(device); yb = yb.to(device)
            t = torch.randint(0, diff.T, (yb.shape[0],), device=device)
            noise = torch.randn_like(yb)
            loss = ((model(diff.q_sample(yb, t, noise), t, cb) - noise) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ema.update(model)
        # val: EMA 가중치로 고정 grid 평가 (training 가중치는 백업 후 복원)
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.shadow); model.eval()
        with torch.no_grad():
            vloss = ((model(val_xt, val_t, Cva_d) - val_noise) ** 2).mean().item()
        if vloss < best_val - 1e-5:
            best_val = vloss; best_ep = ep; bad = 0
            best_state = {k: v.detach().clone() for k, v in ema.shadow.items()}
        else:
            bad += 1
        model.load_state_dict(backup); model.train()
        if ep <= 5 or ep % 20 == 0:
            print(f"    ep{ep:3d}: train_eps_mse={loss.item():.4f}  val(ema)={vloss:.4f}  "
                  f"(best {best_val:.4f}@{best_ep})")
        if bad >= PATIENCE:
            print(f"    [early stop] ep{ep} (from {best_ep})"); break
    model.load_state_dict(best_state)

    print(f"  [sample] n_sim={N_SIM}")
    model.eval()
    mu = stats["sp_return"][0]
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

    sim_raw = mu + sim_z * Ste[:, None, None]    # vol-adjustment 역변환 (μ + z·σ_origin)
    act_raw = mu + Yte * Ste[:, None]
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
        fold=fold, model="Conditional Path Diffusion (MLP enc + DC, vol-adjusted target, DDPM)",
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
