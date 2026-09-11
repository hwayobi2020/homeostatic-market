"""MAC-VAE — MAC-Flow(fpath_novol)의 *흐름 헤드만* 조건부 VAE 헤드로 바꾼 통제 모형.

무엇을 위한 것인가 (R2#3)
--------------------------
리뷰어 2 는 "CondVAE/CondGAN 은 13 주를 일괄 생성하고 MAC-Flow 는 자기회귀라서, 차이가
흐름 헤드 때문인지 생성 구조 때문인지 구분할 수 없다"고 했다.  이 모형은 그 둘을 가른다:
인코더(과거 52 주 MLP + 과거 요약) · 미래경로 전체 요약(전역 주입) · 스텝별 재주입 ·
AR 롤아웃 · 조건 채널 · 시드 · 학습 절차(val NLL 최저 복원)는 MambaFlowARFpath 그대로이고,
스텝별 스칼라 y_t 를 내는 헤드만  RQ-NSF 흐름 → 조건부 가우시안 VAE  로 바뀐다.

  (헤드, 주입)     흐름           VAE
  이중 주입      MAC-Flow      ★ 이 파일 (MAC-VAE)
  일괄           —             CondVAE (train_vae_gan_baseline)

헤드 인터페이스
---------------
MambaFlowAR(Fpath) 는 헤드를 두 호출로만 쓴다 (fpath_model.py:146, :223):
    self.flow.log_prob(inputs=y_flat, context=ctx_flat)   # (N,1),(N,C) -> (N,)
    self.flow.sample(1, context=ctx_tau)                   # -> (N, 1, 1)
VAEHead 는 같은 두 메서드를 제공한다.  log_prob 은 단일 표본 ELBO(= log p 의 하한)를
돌려주므로, train() 의 NLL 손실·val NLL 체크포인트 선택이 그대로 -ELBO 로 작동한다.

헤드 구조 (train_vae_gan_baseline.CondVAE 와 같은 부류로 맞춤)
------------------------------------------------------------
  q(z | y_t, c_t) : MLP([y, c]) -> (mu, logvar), z ∈ R^LATENT
  p(y_t | z, c_t) : MLP([z, c]) -> (mean, logvar), logvar ∈ [-2, 2]   (CondVAE 의 DEC_LOGVAR_CLAMP)
  prior N(0, I).  학습 시 free-bits 0.5 nat/dim (CondVAE 와 동일, posterior collapse 방지).
  평가(eval) 시에는 free-bits 없이 진짜 ELBO 를 돌려준다 (val NLL 이 실제 하한이 되도록).
  CondVAE 의 KL β 램프(60 에폭)는 train() 이 에폭을 헤드에 넘기지 않아 넣지 않았다.
  hidden 192 = 튜닝된 CondVAE(t3skfu_c128h192) 의 HID.   LATENT 16 = CondVAE 와 동일.

사용
----
  학습 :  FPATH_HEAD=vae FPATH_NOVOL=1 FPATH_DIM=2 FPATH_SEEDS=2026,... python colab/dual_3ch/run_full_fpath.py
          → tag rvAbl_full_fpath_novol_vaehead_d2_s{seed}
  평가 :  PS_HEAD=vae PS_BASE=fpath_novol FPATH_DIM=2 python colab/dual_3ch/<§4 스크립트>.py
          (analyze_pathshape_rawvol 이 PS_HEAD 로 클래스·태그를 바꾼다)
env :  VH_LATENT(16)  VH_HID(192)  VH_FREE_BITS(0.5)
"""
import math
import os

import torch
import torch.nn as nn

import fpath_model

LATENT = int(os.environ.get("VH_LATENT", "16"))
HID = int(os.environ.get("VH_HID", "192"))
FREE_BITS = float(os.environ.get("VH_FREE_BITS", "0.5"))
DEC_LOGVAR_CLAMP = (-2.0, 2.0)          # train_vae_gan_baseline.DEC_LOGVAR_CLAMP 와 동일
_LOG2PI = math.log(2.0 * math.pi)


class VAEHead(nn.Module):
    """스텝별 스칼라 y | context 의 조건부 가우시안 VAE.  nflows.Flow 의 log_prob/sample 호환."""

    def __init__(self, context_dim, latent=LATENT, hidden=HID, free_bits=FREE_BITS,
                 dropout=0.0):
        super().__init__()
        self.context_dim = int(context_dim)
        self.latent = int(latent)
        self.free_bits = float(free_bits)
        drop = [nn.Dropout(dropout)] if dropout > 0 else []
        self.q = nn.Sequential(nn.Linear(1 + self.context_dim, hidden), nn.ReLU(), *drop,
                               nn.Linear(hidden, 2 * self.latent))
        self.dec = nn.Sequential(nn.Linear(self.latent + self.context_dim, hidden), nn.ReLU(), *drop,
                                 nn.Linear(hidden, 2))

    def _decode(self, z, context):
        out = self.dec(torch.cat([z, context], dim=-1))
        return out[:, 0:1], out[:, 1:2].clamp(*DEC_LOGVAR_CLAMP)          # mean, logvar (N,1)

    def log_prob(self, inputs, context):
        """단일 표본 ELBO (N,).  inputs (N,1), context (N,C)."""
        h = self.q(torch.cat([inputs, context], dim=-1))
        mu, logvar = h[:, :self.latent], h[:, self.latent:].clamp(-8.0, 8.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        ymean, ylogvar = self._decode(z, context)
        recon = -0.5 * (ylogvar + (inputs - ymean) ** 2 / torch.exp(ylogvar) + _LOG2PI)
        kl_dim = -0.5 * (1.0 + logvar - mu ** 2 - torch.exp(logvar))       # (N, latent)
        if self.training and self.free_bits > 0:
            kl_dim = torch.clamp(kl_dim, min=self.free_bits)
        return recon.squeeze(-1) - kl_dim.sum(dim=-1)

    @torch.no_grad()
    def sample(self, num_samples, context):
        """(N, num_samples, 1) — nflows.Flow.sample 과 같은 모양."""
        N = context.shape[0]
        c = context.repeat_interleave(num_samples, dim=0)
        z = torch.randn(N * num_samples, self.latent, device=context.device, dtype=context.dtype)
        ymean, ylogvar = self._decode(z, c)
        y = ymean + torch.randn_like(ymean) * torch.exp(0.5 * ylogvar)
        return y.view(N, num_samples, 1)


class MambaVAEARFpath(fpath_model.MambaFlowARFpath):
    """MambaFlowARFpath 에서 self.flow 만 VAEHead 로 교체.  나머지 전부 상속."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow = VAEHead(self.flow_context_dim, dropout=self._fp_dropout)
