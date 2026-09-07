"""Conditional VAE / GAN baselines on the SAME setup as train_garch_flow.py.

Fair generative-head comparison: identical GARCH filtering (NF-GARCH z_t target),
identical macro conditioning channels, identical gap29 folds, identical eval
metrics + raw rescale.  Only the generative model differs (flow -> VAE / GAN).
This deliberately *gives VAE/GAN the same GARCH boost* (harder test for us than
plain raw-return VAE/GAN, and avoids the "you handicapped the baselines" critique).

Conditioning: an MLP encoder maps the (PAST_LEN+FUTURE_LEN) x N_CH input window
(past 9 channels + future tbill unmasked, future macro masked -- exactly the
flow's input X) to a context vector c.  The VAE/GAN generate the 13-week target
path Y (= constant-z-scored GARCH residual z_t) conditioned on c.  At eval the
generated Y is rescaled to raw returns by  z_t = Y*tsd+tmu ; r = z_t*sigma_t+mu_t
-- identical to train_garch_flow.evaluate_test.

Usage (Colab):
  !python colab/dual_3ch/train_vae_gan_baseline.py --model vae --fold F_gfc --n-sim 1000
  !python colab/dual_3ch/train_vae_gan_baseline.py --model gan --fold F_gfc --n-sim 1000
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Reuse the EXACT GARCH preprocessing, data loader, alignment, and metrics that
# train_garch_flow uses -- guarantees an apples-to-apples comparison.
from train_garch_flow import (  # noqa: E402
    garch_preprocess_fold, compute_valid_mask, cached_load_windows_seq,
    COND_COLS, TBILL_CH, PAST_LEN, FUTURE_LEN, SP_CH, MASK_FUTURE_FILL,
    crps_pooled, crps_ensemble_sample, compute_var, compute_cvar, compute_emd_1d,
    forward_garch_rescale,
)

N_CH = len(COND_COLS)
L = PAST_LEN + FUTURE_LEN

# ---- compact baseline hyperparameters (small, fast) ----
D_CTX   = 128
LATENT  = 16
HID     = 128
EPOCHS  = 250
LR_VAE  = 5e-4          # 1e-3 → 5e-4: decoder logvar 폭발/불안정(std 2.3배·우편향) 완화
DEC_LOGVAR_CLAMP = (-2.0, 2.0)   # decoder logvar 범위: 하한 -2(σ floor exp(-1)=0.37 → 과확신/
                                 # variance collapse 차단, per-origin 구간 확보) · 상한 2(폭발 캡)
LR_GAN  = 1e-4
BATCH   = 64
N_CRITIC = 5          # WGAN-GP critic steps per generator step
GP_LAMBDA = 10.0
KL_ANNEAL = 60        # epochs to ramp beta 0 -> 1

# ---- validation-based checkpoint selection ----
# MAC-Flow(train_garch_flow) 는 val NLL 최저 시점의 가중치를 복원한다.  베이스라인만
# 마지막 에폭을 쓰면 모델 선택 절차가 비대칭이 되므로(리뷰어 1 #5) 여기도 val 기준
# best checkpoint 를 고른다.  GAN 은 우도가 없어 ELBO/NLL 을 쓸 수 없으므로 VAE·GAN
# 공통으로 val CRPS(z 공간, proper scoring rule)를 기준으로 삼는다.
VAL_SELECT   = os.environ.get("VG_VAL_SELECT", "1") == "1"
EVAL_VAL     = os.environ.get("VG_EVAL_VAL", "0") == "1"   # lr 선택용 val 전체평가
VAL_EVERY    = int(os.environ.get("VG_VAL_EVERY", "10"))    # 몇 에폭마다 검증할지
VAL_N_SIM    = int(os.environ.get("VG_VAL_NSIM", "200"))    # 검증용 표본 수
VAL_SIM_SEED = 12345      # 에폭 간 검증 잡음을 공통으로 만들기 위한 고정 시드


# =====================================================================
# Context encoder (shared) -- maps input window X (B,L,N_CH) -> c (B,D_CTX)
# =====================================================================
class CtxEncoder(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(L * N_CH, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, D_CTX), nn.ReLU(),
        )

    def forward(self, x):                      # x: (B, L, N_CH)
        # NO-LEAK: the flow's sp channel is 1-step shifted, so future positions
        # hold (shifted) future ACTUAL returns.  A non-causal MLP would read the
        # target straight off the context.  Mask ALL future positions except the
        # tbill scenario path (the only legitimate future conditioning).
        # 채움값은 MASK_FUTURE_FILL 로 고른다 (train_garch_flow 와 동일 규약).
        #   zero : 0 = z-score 기준 학습기간 평균 (원점 마지막값에서 평균으로 점프)
        #   last : 원점의 마지막 관측값(x[:, PAST_LEN-1, :])을 미래 구간 유지
        x = x.clone()
        if MASK_FUTURE_FILL == "last":
            # 미래를 전혀 쓰지 않는 조건.  tbill 도 실현 경로를 되돌리지 않고
            # 원점의 마지막 관측값을 유지한다 (네 모델 조건 일치용).
            # SP 채널은 1칸 shift 돼 있어 x[:, PAST_LEN-1] 이 원점 주가 아니라
            # 그 전 주 값이므로 채우지 않고 0(=학습기간 평균)으로 둔다.
            last = x[:, PAST_LEN - 1:PAST_LEN, :].clone()
            x[:, PAST_LEN:, :] = last
            x[:, PAST_LEN:, SP_CH] = 0.0
        else:
            keep_tbill = x[:, PAST_LEN:, TBILL_CH].clone()
            x[:, PAST_LEN:, :] = 0.0
            x[:, PAST_LEN:, TBILL_CH] = keep_tbill
        return self.net(x)


# =====================================================================
# Conditional VAE  (decoder outputs per-step mean + log-var => proper spread)
# =====================================================================
class CondVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_ctx = CtxEncoder()
        self.q = nn.Sequential(
            nn.Linear(FUTURE_LEN + D_CTX, HID), nn.ReLU(),
            nn.Linear(HID, 2 * LATENT),
        )
        self.dec = nn.Sequential(
            nn.Linear(LATENT + D_CTX, HID), nn.ReLU(),
            nn.Linear(HID, 2 * FUTURE_LEN),    # (mean_13, logvar_13)
        )

    def forward(self, x, y):
        c = self.enc_ctx(x)
        h = self.q(torch.cat([y, c], dim=-1))
        mu, logvar = h[:, :LATENT], h[:, LATENT:].clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        out = self.dec(torch.cat([z, c], dim=-1))
        ymean, ylogvar = out[:, :FUTURE_LEN], out[:, FUTURE_LEN:].clamp(*DEC_LOGVAR_CLAMP)
        return ymean, ylogvar, mu, logvar

    @torch.no_grad()
    def generate(self, x, n_sim):
        c = self.enc_ctx(x)                              # (B, D_CTX)
        B = c.shape[0]
        c_rep = c.repeat_interleave(n_sim, 0)            # (B*n_sim, D_CTX)
        z = torch.randn(B * n_sim, LATENT, device=c.device)
        out = self.dec(torch.cat([z, c_rep], dim=-1))
        ymean = out[:, :FUTURE_LEN]
        ylogvar = out[:, FUTURE_LEN:].clamp(*DEC_LOGVAR_CLAMP)
        y = ymean + torch.randn_like(ymean) * torch.exp(0.5 * ylogvar)
        return y.view(B, n_sim, FUTURE_LEN)


def vae_loss(ymean, ylogvar, y, mu, logvar, beta, free_bits=0.5):
    # Gaussian NLL recon (summed over 13 steps), KL to N(0,I) with FREE-BITS
    # anti-collapse: each latent dim keeps >= free_bits nats so the latent is
    # not driven to 0 (posterior collapse made the decoder over-narrow before).
    recon = 0.5 * (ylogvar + (y - ymean) ** 2 / torch.exp(ylogvar)).sum(dim=1)
    kl_dim = -0.5 * (1 + logvar - mu ** 2 - torch.exp(logvar))      # (B, LATENT)
    kl = torch.clamp(kl_dim, min=free_bits).sum(dim=1)             # free-bits floor
    return (recon + beta * kl).mean(), recon.mean().item(), float(kl_dim.sum(1).mean())


# =====================================================================
# Conditional GAN  (WGAN-GP for stability)
# =====================================================================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_ctx = CtxEncoder(dropout=0.0)
        self.net = nn.Sequential(
            nn.Linear(LATENT + D_CTX, HID), nn.ReLU(),
            nn.Linear(HID, HID), nn.ReLU(),
            nn.Linear(HID, FUTURE_LEN),
        )

    def forward(self, x, noise):
        c = self.enc_ctx(x)
        return self.net(torch.cat([noise, c], dim=-1)), c

    @torch.no_grad()
    def generate(self, x, n_sim):
        c = self.enc_ctx(x)
        B = c.shape[0]
        c_rep = c.repeat_interleave(n_sim, 0)
        noise = torch.randn(B * n_sim, LATENT, device=c.device)
        y = self.net(torch.cat([noise, c_rep], dim=-1))
        return y.view(B, n_sim, FUTURE_LEN)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FUTURE_LEN + D_CTX, HID), nn.LeakyReLU(0.2),
            nn.Linear(HID, HID), nn.LeakyReLU(0.2),
            nn.Linear(HID, 1),
        )

    def forward(self, y, c):
        return self.net(torch.cat([y, c], dim=-1))


def gradient_penalty(critic, y_real, y_fake, c, device):
    eps = torch.rand(y_real.shape[0], 1, device=device)
    inter = (eps * y_real + (1 - eps) * y_fake).requires_grad_(True)
    score = critic(inter, c)
    grad = torch.autograd.grad(score, inter,
                               grad_outputs=torch.ones_like(score),
                               create_graph=True, retain_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


# =====================================================================
# Validation criterion for checkpoint selection
# =====================================================================
@torch.no_grad()
def val_crps(gen, Xva, Yva, device, n_sim=VAL_N_SIM, seed=VAL_SIM_SEED):
    """검증 구간 CRPS (z 공간).  낮을수록 좋다.

    표본 추출 시드를 매 호출 고정해 에폭 간 비교가 표본 잡음이 아니라 모델 차이를
    반영하도록 한다.  호출 전후로 전역 RNG 상태를 보존해 학습 궤적을 건드리지 않는다.
    """
    was_training = gen.training
    gen.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() else None)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sims = []
    for s in range(0, Xva.shape[0], 16):
        sims.append(gen.generate(Xva[s:s + 16].to(device), n_sim).cpu().numpy())
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    if was_training:
        gen.train()
    m, _ = crps_pooled(np.concatenate(sims, axis=0), Yva.numpy())
    return float(m)


def _snapshot(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _sk(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 3))


def _ek(a):
    a = np.asarray(a, float); m = a.mean(); s = a.std() + 1e-12
    return float(np.mean(((a - m) / s) ** 4) - 3.0)


# =====================================================================
# Evaluate a trained generator on one split (val or test)
# =====================================================================
def evaluate_split(gen, model_kind, fold, split_csv, cond_stats, target_stats,
                   n_sim, device, label):
    """train_garch_flow.evaluate_test 와 같은 정의로 한 split 을 평가한다.

    lr 같은 상위 하이퍼파라미터를 *검증셋* 지표로 고르려면 시험셋과 동일한
    롤아웃·재스케일·지표 정의를 검증셋에도 적용해야 한다.  그래서 이 함수를
    val/test 공용으로 쓴다 (예전에는 test 전용으로 run_fold 안에 박혀 있었다).
    """
    gen.eval()
    Xs, Ys, _, _, _ = cached_load_windows_seq(
        split_csv, cond_stats=cond_stats, target_stats=target_stats)
    valid_mask, _, df_s = compute_valid_mask(split_csv, cond_stats)
    n_orig = Xs.shape[0]

    sims = []
    for s in range(0, n_orig, 16):
        sims.append(gen.generate(Xs[s:s + 16].to(device), n_sim).cpu().numpy())
    sim_paths_z = np.concatenate(sims, axis=0)                 # (n_orig, n_sim, T)

    tmu = float(target_stats["mean"]); tsd = float(target_stats["std"])
    sim_zt = sim_paths_z * tsd + tmu
    actual_zt = Ys.numpy() * tsd + tmu

    # raw rescale (same as garch_flow, look-ahead 제거):
    #   actual = filtered σ (realized) 복원 = 실현 수익률 (정답 라벨, 누수 아님)
    #   sim    = origin 부터 forward GARCH σ 예측 (미래 실현 σ 안 씀)
    for _c in ("garch_omega", "garch_alpha", "garch_beta"):
        if _c not in df_s.columns:
            raise SystemExit(f"[FATAL] {label} csv lacks {_c} -> *_garch.csv 재생성 "
                             "필요 (garch_preprocess_fold 최신 버전으로)")
    gsig = df_s["garch_sigma"].to_numpy(float)
    gmu = df_s["garch_mu"].to_numpy(float)
    gz = df_s["sp_return"].to_numpy(float)               # = z_t (표준화 잔차)
    oidx = np.where(valid_mask)[0]
    sig = np.array([[gsig[int(oidx[i]) + PAST_LEN + t] for t in range(FUTURE_LEN)]
                    for i in range(n_orig)])
    mu = np.array([[gmu[int(oidx[i]) + PAST_LEN + t] for t in range(FUTURE_LEN)]
                   for i in range(n_orig)])
    actual_raw = actual_zt * sig + mu                    # 실현 수익률 복원
    om = float(df_s["garch_omega"].iloc[0]); al = float(df_s["garch_alpha"].iloc[0])
    be = float(df_s["garch_beta"].iloc[0]); mu_c = float(gmu[0])
    orow = oidx + (PAST_LEN - 1)                         # origin (마지막 관측) 행
    s2_orig = gsig[orow] ** 2
    e2_orig = (gz[orow] * gsig[orow]) ** 2               # ε_origin = z_origin·σ_origin
    sim_paths_raw = forward_garch_rescale(sim_zt, s2_orig, e2_orig,
                                          om, al, be, mu_c)

    af = actual_raw.ravel(); sf = sim_paths_raw.ravel()
    crps_m, _ = crps_pooled(sim_paths_raw, actual_raw)
    emd = compute_emd_1d(af, sf, n_bins=200)
    std_a, std_s = float(af.std(ddof=1)), float(sf.std(ddof=1))

    # 커버리지: MAC-Flow(train_garch_flow.evaluate_test) 와 동일하게 *전역 풀링* 구간.
    #   전 origin × 전 sim × 전 시점을 합친 분포에서 백분위 한 쌍을 뽑아 전부를 판정한다.
    #   (이전에는 axis=1 원점별이라 MAC-Flow 와 정의가 달랐다.)
    cov = {}
    for lvl, lo, hi in [(50, 25, 75), (80, 10, 90), (95, 2.5, 97.5)]:
        L_, H_ = np.percentile(sf, lo), np.percentile(sf, hi)
        cov[lvl] = float(((af >= L_) & (af <= H_)).mean())
    cv1a, cv1s = compute_cvar(af, 0.01), compute_cvar(sf, 0.01)

    m = dict(crps_pooled=crps_m, emd=emd, std_ratio=std_s / std_a,
             coverage_50=cov[50], coverage_80=cov[80], coverage_95=cov[95],
             cvar_1pct_diff=cv1s - cv1a,
             skew_actual=_sk(af), skew_sim=_sk(sf),
             exkurt_actual=_ek(af), exkurt_sim=_ek(sf))

    print(f"\n[{model_kind.upper()} {label}]  fold={fold}  (n_sim={n_sim}, "
          f"origins={n_orig})")
    print(f"    CRPS pooled       = {crps_m:.5f}")
    print(f"    EMD               = {emd:.6f}")
    print(f"    std act/sim/ratio = {std_a:.5f} / {std_s:.5f} / {std_s / std_a:.3f}")
    print(f"    skew act/sim      = {m['skew_actual']:+.4f} / {m['skew_sim']:+.4f}")
    print(f"    exkurt act/sim    = {m['exkurt_actual']:+.4f} / {m['exkurt_sim']:+.4f}")
    print(f"    CVaR1 act/sim/D   = {cv1a:+.5f} / {cv1s:+.5f} / {cv1s - cv1a:+.5f}")
    print(f"    cov 50/80/95      = {cov[50]:.3f} / {cov[80]:.3f} / {cov[95]:.3f}")

    return dict(metrics=m, sim=sim_paths_raw, act=actual_raw, oidx=oidx)


# =====================================================================
# Train + eval for one fold
# =====================================================================
def run_fold(model_kind, fold, args, device):
    gp = garch_preprocess_fold(args.folds_dir, fold, args.out_dir)
    train_csv, val_csv, test_csv = gp["train"], gp["val"], gp["test"]

    Xtr, Ytr, cond_stats, target_stats, n_tr = cached_load_windows_seq(train_csv)
    print(f"  train windows = {n_tr}, X={tuple(Xtr.shape)}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # lr 은 args 로 덮어쓸 수 있다 (스윕용).  없으면 모듈 기본값.
    lr = getattr(args, "lr", None)
    if lr is None:
        lr = LR_VAE if model_kind == "vae" else LR_GAN
    lr = float(lr)
    print(f"  lr = {lr:g}  seed = {args.seed}")

    Xva = Yva = None
    if VAL_SELECT:
        Xva, Yva, _, _, n_va = cached_load_windows_seq(
            val_csv, cond_stats=cond_stats, target_stats=target_stats)
        print(f"  val   windows = {n_va}  (checkpoint 선택 기준 = val CRPS, "
              f"{VAL_EVERY} 에폭마다)")

    best = dict(crps=float("inf"), epoch=-1, state=None)

    def check_val(gen, ep):
        """val CRPS 를 재고 개선되면 가중치를 스냅샷한다."""
        if not VAL_SELECT:
            return
        c = val_crps(gen, Xva, Yva, device)
        mark = ""
        if c < best["crps"]:
            best.update(crps=c, epoch=ep, state=_snapshot(gen))
            mark = "  *"
        print(f"      val CRPS = {c:.5f}  (best {best['crps']:.5f} "
              f"@ep{best['epoch']}){mark}")

    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True,
                    drop_last=True)

    if model_kind == "vae":
        model = CondVAE().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(1, EPOCHS + 1):
            model.train(); beta = min(1.0, ep / KL_ANNEAL)
            losses = []
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                ymean, ylogvar, mu, lv = model(xb, yb)
                loss, rec, kl = vae_loss(ymean, ylogvar, yb, mu, lv, beta)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                losses.append(loss.item())
            if ep == 1 or ep % 50 == 0 or ep == EPOCHS:
                print(f"    ep{ep:>3d}  loss={np.mean(losses):+.3f} "
                      f"(recon~{rec:+.3f} kl~{kl:.3f} beta={beta:.2f})")
            if ep % VAL_EVERY == 0 or ep == EPOCHS:
                check_val(model, ep)
        gen = model

    else:  # gan (WGAN-GP)
        G, Dc = Generator().to(device), Critic().to(device)
        optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
        optD = torch.optim.Adam(Dc.parameters(), lr=lr, betas=(0.0, 0.9))
        it = 0
        for ep in range(1, EPOCHS + 1):
            G.train(); Dc.train(); dl_loss = []
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                # ---- critic ----
                noise = torch.randn(xb.shape[0], LATENT, device=device)
                with torch.no_grad():
                    y_fake, _ = G(xb, noise)
                c = G.enc_ctx(xb).detach()
                gp_term = gradient_penalty(Dc, yb, y_fake, c, device)
                d_loss = (Dc(y_fake, c).mean() - Dc(yb, c).mean()
                          + GP_LAMBDA * gp_term)
                optD.zero_grad(); d_loss.backward(); optD.step()
                it += 1
                # ---- generator every N_CRITIC ----
                if it % N_CRITIC == 0:
                    noise = torch.randn(xb.shape[0], LATENT, device=device)
                    y_fake, c2 = G(xb, noise)
                    g_loss = -Dc(y_fake, c2).mean()
                    optG.zero_grad(); g_loss.backward(); optG.step()
                    dl_loss.append((float(d_loss), float(g_loss)))
            if (ep == 1 or ep % 50 == 0 or ep == EPOCHS) and dl_loss:
                dl_, gl_ = np.mean(dl_loss, axis=0)
                print(f"    ep{ep:>3d}  D={dl_:+.3f}  G={gl_:+.3f}")
            if ep % VAL_EVERY == 0 or ep == EPOCHS:
                check_val(G, ep)
        gen = G

    # ---------- restore best-validation checkpoint ----------
    if VAL_SELECT and best["state"] is not None:
        gen.load_state_dict(best["state"])
        print(f"  [checkpoint] val CRPS 최저 ep{best['epoch']} "
              f"({best['crps']:.5f}) 복원  (마지막 에폭 {EPOCHS} 아님)")

    # ---------- evaluate ----------
    # lr 등 상위 하이퍼파라미터를 고를 때는 *검증셋* 지표만 본다 (시험셋으로 고르면
    # test-set selection).  VG_EVAL_VAL=1 이면 시험셋과 같은 정의로 검증셋도 평가한다.
    val_metrics = None
    if EVAL_VAL:
        val_metrics = evaluate_split(gen, model_kind, fold, val_csv, cond_stats,
                                     target_stats, args.n_sim, device,
                                     "VAL")["metrics"]

    te = evaluate_split(gen, model_kind, fold, test_csv, cond_stats,
                        target_stats, args.n_sim, device, "TEST")
    sim_paths_raw, actual_raw, oidx = te["sim"], te["act"], te["oidx"]
    n_orig = sim_paths_raw.shape[0]

    tag = getattr(args, "tag", "") or ""
    os.makedirs(args.out_dir, exist_ok=True)

    # per-(origin,step) CRPS for paired DM test vs garch-flow (same origins/order)
    per_oc = np.array([[crps_ensemble_sample(sim_paths_raw[i, :, t], actual_raw[i, t])
                        for t in range(FUTURE_LEN)] for i in range(n_orig)])
    np.save(os.path.join(args.out_dir,
            f"{model_kind}_baseline{tag}_{fold}_crps_per_origin.npy"), per_oc)

    summ = dict(model=f"cond-{model_kind}", fold=fold, n_sim=args.n_sim,
                seed=int(args.seed), lr=float(lr),
                val_select=bool(VAL_SELECT),
                best_epoch=int(best["epoch"]),
                best_val_crps_z=(float(best["crps"])
                                 if np.isfinite(best["crps"]) else None),
                epochs_max=EPOCHS,
                val_eval=val_metrics,
                test_eval=te["metrics"])
    sp = os.path.join(args.out_dir,
                      f"{model_kind}_baseline{tag}_{fold}_summary.json")
    json.dump(summ, open(sp, "w"), indent=2, default=str)
    print(f"    saved {os.path.basename(sp)}")
    # pred_start = 예측 대상 첫 주의 test CSV 행 번호 (창 w 의 past 52 주 다음).
    return dict(sim=sim_paths_raw, act=actual_raw,
                pred_start=oidx + PAST_LEN, summary=summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vae", "gan"])
    ap.add_argument("--fold", required=True)
    ap.add_argument("--folds-dir",
                    default=os.path.join(os.path.normpath(os.path.join(HERE, "..", "..")),
                                         "data", "folds_v33_vix_expanding"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--n-sim", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--lr", type=float, default=None,
                    help="없으면 모듈 기본값 (vae 5e-4 / gan 1e-4)")
    ap.add_argument("--tag", default="",
                    help="결과 파일명 접미사 (스윕에서 셀 구분용)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.model}-baseline] fold={args.fold} device={device}")
    run_fold(args.model, args.fold, args, device)


if __name__ == "__main__":
    main()
