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
  평가(eval) 시에는 free-bits 없이 진짜 ELBO 를 돌려주고(val NLL 이 실제 하한이 되도록),
  z 잡음은 고정 시드라 흐름 헤드처럼 결정적이다(에폭 간 val NLL 비교가 같은 draw 위에서 됨).
  CondVAE 의 KL β 램프(60 에폭)는 train() 이 에폭을 헤드에 넘기지 않아 넣지 않았다.
  hidden 192 = 튜닝된 CondVAE(run_thin_compare.VG_SPEC["vae"] hid=192) 의 HID.   LATENT 16 = CondVAE 와 동일.

논문에 적어야 할 두 가지 (흐름 헤드와 같지 않은 점)
--------------------------------------------------
  · 이 헤드의 log_prob 은 단일 표본 ELBO 라 −ELBO ≥ 진짜 NLL.  MAC-Flow 의 test NLL 은 정확값이므로
    두 모형의 NLL 을 직접 비교하면 안 된다 — 비교는 표본 기반 지표(CRPS·커버리지·왜도·IHL)로 한다.
  · 디코더 logvar 를 [-2, 2] 로 묶어 표준화 공간에서 σ ≥ exp(-1) = 0.37 바닥이 있다 (CondVAE 와 동일한
    분산 붕괴 방지 장치).  흐름 헤드에는 이 제약이 없다.
  · 학습 로그의 train_nll 에는 free-bits 바닥(최대 0.5×LATENT×13 nat/origin)이 얹혀 있어 val_nll 과 같은
    축에서 읽으면 안 된다.
  · VH_LATENT / VH_HID / VH_FREE_BITS 는 체크포인트 meta 에 안 남고 버퍼 head_hparams 에 기록되며,
    MambaVAEARFpath.load_state_dict 가 그 값으로 헤드를 다시 만들어 적재한다 (평가 쪽 env 불필요).

한 번에 다 돌리기 (튜닝 → 5시드 학습 → MAC-Flow 대 MAC-VAE 짝 비교표 → §4.3 경로형태):
  python colab/dual_3ch/run_macvae_all.py

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
VAL_EPS_SEED = 12345                     # 평가 모드 log_prob 의 z 잡음 고정 시드 (CondVAE VAL_SIM_SEED 와 같은 값)
_LOG2PI = math.log(2.0 * math.pi)


class VAEHead(nn.Module):
    """스텝별 스칼라 y | context 의 조건부 가우시안 VAE.  nflows.Flow 의 log_prob/sample 호환."""

    def __init__(self, context_dim, latent=None, hidden=None, free_bits=None,
                 dropout=0.0):
        super().__init__()
        # None 이면 *호출 시점* 의 모듈 전역을 쓴다 (튜닝 루프가 vae_head.LATENT/HID 를 바꿔 가며 돈다).
        latent = LATENT if latent is None else latent
        hidden = HID if hidden is None else hidden
        free_bits = FREE_BITS if free_bits is None else free_bits
        self.context_dim = int(context_dim)
        self.latent = int(latent)
        self.hidden = int(hidden)
        self.free_bits = float(free_bits)
        drop = [nn.Dropout(dropout)] if dropout > 0 else []
        self.q = nn.Sequential(nn.Linear(1 + self.context_dim, hidden), nn.ReLU(), *drop,
                               nn.Linear(hidden, 2 * self.latent))
        self.dec = nn.Sequential(nn.Linear(self.latent + self.context_dim, hidden), nn.ReLU(), *drop,
                                 nn.Linear(hidden, 2))
        # 체크포인트에 헤드 설정을 남긴다 (train_garch_flow 의 meta 는 흐름 인자만 적는다).
        # latent/hidden 이 다르면 load_state_dict 가 모양 불일치로 터지지만, free_bits 는
        # 모양이 같아 조용히 다른 목적함수가 되므로 값 자체를 기록해 사후 식별이 되게 한다.
        self.register_buffer("head_hparams",
                             torch.tensor([float(self.latent), float(hidden), self.free_bits]))

    def _decode(self, z, context):
        out = self.dec(torch.cat([z, context], dim=-1))
        return out[:, 0:1], out[:, 1:2].clamp(*DEC_LOGVAR_CLAMP)          # mean, logvar (N,1)

    def log_prob(self, inputs, context):
        """단일 표본 ELBO (N,).  inputs (N,1), context (N,C)."""
        h = self.q(torch.cat([inputs, context], dim=-1))
        mu, logvar = h[:, :self.latent], h[:, self.latent:].clamp(-8.0, 8.0)
        if self.training:
            eps = torch.randn_like(mu)
        else:
            # 평가(val NLL 체크포인트 선택·test NLL)는 흐름 헤드처럼 결정적이어야 한다.
            # z 재표본 잡음을 고정 시드로 묶어 에폭 간 비교가 같은 draw 위에서 되게 한다.
            g = torch.Generator(device=mu.device).manual_seed(VAL_EPS_SEED)
            eps = torch.randn(mu.shape, generator=g, device=mu.device, dtype=mu.dtype)
        z = mu + eps * torch.exp(0.5 * logvar)
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

    def load_state_dict(self, state_dict, strict=True):
        """체크포인트의 head_hparams 로 헤드를 다시 만든 뒤 적재.

        analyze_pathshape_rawvol.rebuild_model 은 흐름 인자(meta)만 알고 VAE 헤드 크기는
        모르므로, 튜닝으로 고른 latent/hidden 이 모듈 기본값과 다르면 여기서 맞춘다.
        """
        hp = state_dict.get("flow.head_hparams")
        if hp is not None:
            lat, hid, fb = int(round(float(hp[0]))), int(round(float(hp[1]))), float(hp[2])
            if (lat, hid) != (self.flow.latent, self.flow.hidden) or fb != self.flow.free_bits:
                self.flow = VAEHead(self.flow_context_dim, latent=lat, hidden=hid,
                                    free_bits=fb, dropout=self._fp_dropout).to(hp.device)
        return super().load_state_dict(state_dict, strict=strict)
