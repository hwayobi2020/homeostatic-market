"""1-step Macro-Gated FiLM Transformer — sp_return[t+1] (variant 8 only).

매트릭스 multi-step 모델의 1-step 특화 학습. 표준 시계열 forecasting 패턴.
1주 예측 자체가 학습 가능한지 검증 (multi-step 학습 부담 분리).

Architecture: CausalTransformer1Step
  Encoder: train_matrix_transformer.py 와 동일 (Causal Transformer, d_model=128, n_layers=3)
  Heads:   분리된 1-step Gaussian (μ, log σ²) per target channel
  Input:   cond past 52w + target past 52w + cond future 1w (tbill 시나리오)
  Output:  target[t+1] Gaussian per channel

Loss = NLL(sp[t+1]) + λ_aux × Σ NLL(aux[t+1])    (λ_aux = 0.1)
Best ckpt: sp_return val 1-step NLL (raw)

Window 길이 L=53 (past 52 + future 1). 매트릭스 L=104 보다 짧아 데이터 점 더 많음.
ckpt prefix: m1step_film_v* — matrix multi-step + 일반 1-step (m1step_v*) 과 격리.
SKIP 로직: ckpt + summary 이미 존재 시 학습 건너뜀.
"""
import torch  # MUST be first

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch.nn as nn

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

L = 53
PAST_LEN = 52
FUTURE_LEN = 1
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 3
LR = 1e-4
BATCH = 32
MAX_EPOCHS = 60
PATIENCE = 30
GRAD_CLIP = 1.0
LAMBDA_AUX = 0.1

LOG2PI = math.log(2 * math.pi)

COLS_COND_BASE = ["tbill_wr", "m2_yoy_lag", "gdp_yoy_lag", "cpi_yoy_lag"]

VARIANTS = {
    1: dict(name="base",                cond_extra=[],                                                    target_extra=[]),
    2: dict(name="base_bp",             cond_extra=["bondpp_13w_lag"],                                    target_extra=[]),
    3: dict(name="base_sp",             cond_extra=["stockpp_13w_lag"],                                   target_extra=[]),
    4: dict(name="base_bp_sp",          cond_extra=["bondpp_13w_lag", "stockpp_13w_lag"],                 target_extra=[]),
    5: dict(name="mtl_bp",              cond_extra=[],                                                    target_extra=["bondpp_13w_lag"]),
    6: dict(name="mtl_sp",              cond_extra=[],                                                    target_extra=["stockpp_13w_lag"]),
    7: dict(name="mtl_bp_sp",           cond_extra=[],                                                    target_extra=["bondpp_13w_lag", "stockpp_13w_lag"]),
    8: dict(name="best_base_filmgate",  cond_extra=["bondpp_13w_lag", "stockpp_13w_lag"],                 target_extra=[]),  # FiLM: Drop MTL — sp only
    9: dict(name="best_excess_liq_mtl", cond_extra=[],                                                    target_extra=["excess_liq_yoy_lag"]),
    # 변종 10/11: bondpp/stockpp 의 metab 분모 없는 단순 합 ablation (사용자 의문: homeostatic 산식 vs simple cumulative)
    10: dict(name="base_bondsum",       cond_extra=["tbill_13w_cum"],                                     target_extra=[]),
    11: dict(name="base_stocksum",      cond_extra=["sp_13w_cum"],                                        target_extra=[]),
}


def build_spec(variant_id):
    if variant_id not in VARIANTS:
        raise ValueError(f"variant must be in 1..9, got {variant_id}")
    v = VARIANTS[variant_id]
    cols_cond = COLS_COND_BASE + v["cond_extra"]
    cols_target = ["sp_return"] + v["target_extra"]
    mask_future_ch = list(range(1, len(cols_cond)))  # tbill (idx 0) 만 future 활성
    return dict(
        variant_id=variant_id,
        name=v["name"],
        cols_cond=cols_cond,
        cols_target=cols_target,
        mask_future_ch=mask_future_ch,
    )


# =====================================================================
# Model
# =====================================================================

class MacroGatedCausalTransformer1Step(nn.Module):
    """FiLM-based Two-Stream Transformer (Gemini 제안):
      - Market Backbone: Causal Transformer on x_market = [past sp, tbill]
      - Macro Modulator: MLP on x_macro = [m2, gdp, cpi, bondpp, stockpp] → γ, β
      - FiLM Fusion: h_modulated = (1 + γ) * h_market + β
      - Output Head: sp_return Gaussian only (Drop MTL)

    Variant 8 cond 기반 (cond 6ch). target = sp_return 1ch only.

    Hard rules:
      - γ, β projection layers Zero-init (학습 초기 발산 방지)
      - Drop MTL (no excess_liq target)

    forward(cond, target_past): 시그니처 동일. 내부에서 슬라이스.
    """
    MARKET_DIM = 2  # sp_return (target_past) + tbill_wr (cond idx 0)
    MACRO_DIM  = 6  # tbill_wr + m2_yoy_lag, gdp_yoy_lag, cpi_yoy_lag, bondpp_13w_lag, stockpp_13w_lag (cond idx 0-5)
                    # tbill 도 modulator 입력 (paper main thesis "tbill 시나리오 input" 일관성)

    def __init__(self, d_cond, target_cols, d_model, n_heads, n_layers,
                 past_len, future_len=1, dropout=0.1):
        super().__init__()
        # Variant 8 spec 강제 검증
        assert len(target_cols) == 1 and target_cols[0] == "sp_return", \
            f"FiLM model expects target_cols=['sp_return'] only (Drop MTL), got {target_cols}"
        assert d_cond == 6, \
            f"FiLM model expects d_cond=6 (tbill + 5 macro), got {d_cond}"

        self.past_len = past_len
        self.future_len = future_len
        self.target_cols = target_cols
        self.total_len = past_len + future_len  # 53

        # === Market Backbone (Causal Transformer) ===
        self.market_proj = nn.Linear(self.MARKET_DIM, d_model)
        self.market_pos_emb = nn.Embedding(self.total_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.market_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        causal_mask = torch.triu(
            torch.ones(self.total_len, self.total_len), diagonal=1
        ).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # === Macro Modulator (MLP, bottleneck = d_model // 4) ===
        bottleneck = d_model // 4   # 32 when d_model=128
        self.macro_mlp = nn.Sequential(
            nn.Linear(self.MACRO_DIM, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
        )
        # γ, β projection — Zero-init (Hard rule 1)
        self.macro_gamma = nn.Linear(bottleneck, d_model)
        self.macro_beta  = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.macro_gamma.weight)
        nn.init.zeros_(self.macro_gamma.bias)
        nn.init.zeros_(self.macro_beta.weight)
        nn.init.zeros_(self.macro_beta.bias)

        # === Output Head (sp_return only — Hard rule 2: Drop MTL) ===
        self.sp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),  # (μ, log σ²)
        )
        # 호환성: heads 필드 (기존 train loop 호환)
        self.heads = nn.ModuleDict({"sp_return": self.sp_head})

    def forward(self, cond, target_past):
        """
        cond:        (B, total_len=53, d_cond=6) — [tbill, m2, gdp, cpi, bondpp, stockpp]
                                                   future 부분 idx 1-5 = mask=0 (외부 mask_future_ch)
        target_past: (B, past_len=52, 1)         — sp_return past

        Returns:
          outputs: dict {"sp_return": (mu, log_var)}, mu/log_var 는 (B,)
        """
        B = cond.shape[0]
        device = cond.device

        # === 슬라이싱: x_market (past sp + tbill), x_macro (tbill + 5 macro) ===
        tbill_full = cond[:, :, 0:1]                                # (B, 53, 1)

        # target_full sp: past 52 + future 1 mask (=0)
        target_full = torch.zeros(B, self.total_len, 1, dtype=cond.dtype, device=device)
        target_full[:, :self.past_len, :] = target_past             # (B, 52, 1)

        x_market = torch.cat([target_full, tbill_full], dim=-1)     # (B, 53, 2)
        x_macro  = cond[:, :, 0:6]                                  # (B, 53, 6) — tbill 포함 (modulator 도 tbill 시나리오 봄)

        # === Market Backbone (Causal Transformer) ===
        h = self.market_proj(x_market)                              # (B, 53, d_model)
        pos = torch.arange(self.total_len, device=device)
        h = h + self.market_pos_emb(pos).unsqueeze(0)
        h_market = self.market_encoder(h, mask=self.causal_mask, is_causal=True)

        # === Macro Modulator ===
        m_hidden = self.macro_mlp(x_macro)                          # (B, 53, bottleneck)
        gamma = self.macro_gamma(m_hidden)                          # (B, 53, d_model) — zero-init
        beta  = self.macro_beta(m_hidden)                           # (B, 53, d_model) — zero-init

        # === FiLM Fusion ===
        h_modulated = (1.0 + gamma) * h_market + beta               # (B, 53, d_model)

        # === Output Head ===
        h_t1 = h_modulated[:, -1, :]                                # (B, d_model) — last step (t+1)
        out = self.sp_head(h_t1)                                    # (B, 2)
        mu = out[..., 0]                                            # (B,)
        log_var = out[..., 1].clamp(min=-10.0, max=10.0)

        outputs = {"sp_return": (mu, log_var)}
        return outputs


# =====================================================================
# Loss
# =====================================================================

def gaussian_nll_1step(target, mu, log_var):
    """Gaussian NLL for 1 step — target/mu/log_var: (B,)"""
    return 0.5 * (LOG2PI + log_var + (target - mu).pow(2) * torch.exp(-log_var))


def compute_per_channel_nll(target_t1, outputs, target_cols):
    """target_t1: (B, d_target). 채널별 평균 NLL (scalar)."""
    nll_per_col = {}
    for i, col in enumerate(target_cols):
        mu, log_var = outputs[col]
        nll = gaussian_nll_1step(target_t1[..., i], mu, log_var)
        nll_per_col[col] = nll.mean()
    return nll_per_col


def total_loss(target_t1, outputs, target_cols, lambda_aux):
    nll_per_col = compute_per_channel_nll(target_t1, outputs, target_cols)
    L_main = nll_per_col["sp_return"]
    L_aux = sum(nll_per_col[c] for c in target_cols if c != "sp_return")
    if isinstance(L_aux, int):
        L_aux = torch.tensor(0.0, device=L_main.device)
    return L_main + lambda_aux * L_aux, nll_per_col


# =====================================================================
# Data loading
# =====================================================================

def load_windows(csv_path, cols_cond, cols_target, L=53,
                 cond_stats=None, target_stats_map=None, normalize_target_idxs=()):
    """L=53 sliding window — past 52 + future 1."""
    df = pd.read_csv(csv_path)
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        return None, None, None, None
    X = np.zeros((n_w, L, len(cols_target)), dtype=np.float32)
    C = np.zeros((n_w, L, len(cols_cond)),   dtype=np.float32)
    for i in range(n_w):
        X[i] = df[cols_target].iloc[i : i + L].values
        C[i] = df[cols_cond].iloc[i : i + L].values

    if cond_stats is None:
        cmu = C.reshape(-1, len(cols_cond)).mean(axis=0)
        csd = C.reshape(-1, len(cols_cond)).std(axis=0) + 1e-8
        cond_stats_out = {"mean": cmu.tolist(), "std": csd.tolist()}
    else:
        cmu = np.asarray(cond_stats["mean"], dtype=np.float32)
        csd = np.asarray(cond_stats["std"],  dtype=np.float32)
        cond_stats_out = cond_stats
    C = (C - cmu) / csd

    target_stats_map_out = {}
    for idx in normalize_target_idxs:
        if target_stats_map is not None and idx in target_stats_map:
            tmu = target_stats_map[idx]["mean"]
            tsd = target_stats_map[idx]["std"]
        else:
            tmu = float(X[..., idx].mean())
            tsd = float(X[..., idx].std()) + 1e-8
        X[..., idx] = (X[..., idx] - tmu) / tsd
        target_stats_map_out[idx] = {"mean": tmu, "std": tsd,
                                     "channel": int(idx),
                                     "channel_name": cols_target[idx]}
    return torch.from_numpy(X), torch.from_numpy(C), cond_stats_out, target_stats_map_out


def mask_future_channels(C, past_len, mask_channels):
    C = C.clone()
    for ch in mask_channels:
        C[:, past_len:, ch] = 0.0
    return C


# =====================================================================
# Train loop
# =====================================================================

def run(spec, train_csv, val_csv, test_csv, save_dir, seed,
        max_epochs, patience, batch, lr, fold_tag,
        normalize_bondpp, normalize_stockpp, normalize_excess, lambda_aux):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cols_cond   = spec["cols_cond"]
    cols_target = spec["cols_target"]
    mask_future = spec["mask_future_ch"]

    norm_idxs = []
    apply_normbp = bool(normalize_bondpp  and "bondpp_13w_lag"     in cols_target)
    apply_normsp = bool(normalize_stockpp and "stockpp_13w_lag"    in cols_target)
    apply_normex = bool(normalize_excess  and "excess_liq_yoy_lag" in cols_target)
    if apply_normbp: norm_idxs.append(cols_target.index("bondpp_13w_lag"))
    if apply_normsp: norm_idxs.append(cols_target.index("stockpp_13w_lag"))
    if apply_normex: norm_idxs.append(cols_target.index("excess_liq_yoy_lag"))

    sp_idx = cols_target.index("sp_return")

    fold_str = f"_{fold_tag}" if fold_tag else ""
    norm_tag = (("_normbp" if apply_normbp else "")
                + ("_normsp" if apply_normsp else "")
                + ("_normex" if apply_normex else ""))
    tag_full = f"m1step_film_v{spec['variant_id']}_{spec['name']}{norm_tag}{fold_str}_seed{seed}"
    ckpt_path    = os.path.join(save_dir, f"{tag_full}_best.pt")
    summary_path = os.path.join(save_dir, f"{tag_full}_summary.json")
    log_path     = os.path.join(save_dir, f"{tag_full}_trainlog.csv")

    if os.path.exists(ckpt_path) and os.path.exists(summary_path):
        print(f"[SKIP] {tag_full} — ckpt + summary 이미 존재")
        with open(summary_path) as f:
            return json.load(f)

    print(f"\n{'='*72}")
    print(f"[Variant {spec['variant_id']} = {spec['name']}{norm_tag}{fold_str}] seed={seed}  (1-step Gaussian)")
    print(f"  cond   = {cols_cond}    (D_COND={len(cols_cond)})")
    print(f"  target = {cols_target}  (D_TARGET={len(cols_target)})")
    print(f"  mask_future_ch = {mask_future}, λ_aux={lambda_aux}, L={L} (past={PAST_LEN} + future={FUTURE_LEN})")
    print(f"{'='*72}")

    Xtr, Ctr, stats_c, stats_t_map = load_windows(
        train_csv, cols_cond, cols_target, L=L, normalize_target_idxs=norm_idxs)
    if Xtr is None:
        print(f"[FAIL] not enough train windows: {train_csv}")
        return None
    Xte, Cte, _, _ = load_windows(
        test_csv, cols_cond, cols_target, L=L,
        cond_stats=stats_c, normalize_target_idxs=norm_idxs,
        target_stats_map=stats_t_map)
    if Xte is None:
        print(f"[FAIL] not enough test windows: {test_csv}")
        return None

    Ctr = mask_future_channels(Ctr, PAST_LEN, mask_future)
    Cte = mask_future_channels(Cte, PAST_LEN, mask_future)

    if val_csv is not None and os.path.exists(val_csv):
        Xv, Cv, _, _ = load_windows(
            val_csv, cols_cond, cols_target, L=L,
            cond_stats=stats_c, normalize_target_idxs=norm_idxs,
            target_stats_map=stats_t_map)
        Cv = mask_future_channels(Cv, PAST_LEN, mask_future)
        Xtr_, Ctr_ = Xtr, Ctr
        print(f"  train_w={Xtr_.shape[0]} (full), val_w={Xv.shape[0]} (val_csv), test_w={Xte.shape[0]}")
    else:
        n_w_tr = Xtr.shape[0]
        n_val = max(int(n_w_tr * 0.15), 1)
        Xtr_, Ctr_ = Xtr[:-n_val], Ctr[:-n_val]
        Xv,  Cv    = Xtr[-n_val:], Ctr[-n_val:]
        print(f"  train_w={Xtr_.shape[0]}, val_w={Xv.shape[0]} (auto 15%), test_w={Xte.shape[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MacroGatedCausalTransformer1Step(
        d_cond=len(cols_cond), target_cols=cols_target,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        past_len=PAST_LEN, future_len=FUTURE_LEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_sp = float("inf")
    best_state = None
    best_epoch = -1
    best_test_per_ch = None
    best_val_per_ch  = None
    pat = 0
    log = []

    def eval_split(X, C):
        model.eval()
        with torch.no_grad():
            target_past = X[:, :PAST_LEN, :].to(device)
            target_t1 = X[:, PAST_LEN, :].to(device)  # (B, d_target) — t+1 ground truth
            outputs = model(C.to(device), target_past)
            nll_per_col = {}
            for i, col in enumerate(cols_target):
                mu, log_var = outputs[col]
                nll = gaussian_nll_1step(target_t1[..., i], mu, log_var)
                nll_per_col[col] = float(nll.mean())
        return nll_per_col

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(Xtr_.shape[0])
        losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            xb = Xtr_[idx].to(device)
            cb = Ctr_[idx].to(device)
            target_past = xb[:, :PAST_LEN, :]
            target_t1 = xb[:, PAST_LEN, :]
            outputs = model(cb, target_past)
            loss, _ = total_loss(target_t1, outputs, cols_target, lambda_aux)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_per_col  = eval_split(Xv,  Cv)
        test_per_col = eval_split(Xte, Cte)

        per_ch_str = " ".join([f"{c}={v:+.3f}" for c, v in test_per_col.items()])
        print(f"  ep{epoch:>3d}  train_loss={train_loss:+.4f}  "
              f"val_sp={val_per_col['sp_return']:+.4f}  test_sp={test_per_col['sp_return']:+.4f}  | {per_ch_str}")

        row = dict(epoch=epoch, train_loss=train_loss)
        for c, v in val_per_col.items():  row[f"val_{c}"]  = v
        for c, v in test_per_col.items(): row[f"test_{c}"] = v
        log.append(row)

        val_sp = val_per_col["sp_return"]
        if not np.isfinite(val_sp):
            print(f"  ⚠ val_sp non-finite — best ckpt 갱신 차단")
            pat += 1
        elif val_sp < best_val_sp - 1e-4:
            best_val_sp = val_sp
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test_per_ch = dict(test_per_col)
            best_val_per_ch  = dict(val_per_col)
            best_epoch = epoch
            pat = 0
        else:
            pat += 1
        test_sp = test_per_col["sp_return"]
        if (not np.isfinite(test_sp)) or test_sp > 100.0:
            print(f"  ⚠ divergence detected (test_sp={test_sp:+.2f}) — early termination")
            break
        if pat >= patience:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val_sp={best_val_sp:+.4f}")
    if best_test_per_ch is not None:
        print(f"    test per channel: " + ", ".join([f"{c}={v:+.4f}" for c, v in best_test_per_ch.items()]))

    os.makedirs(save_dir, exist_ok=True)
    if best_state is not None:
        torch.save({
            "model_state":       best_state,
            "model_type":        "macro_gated_film_1step",
            "cond_cols":         cols_cond,
            "target_cols":       cols_target,
            "stats_cond":        stats_c,
            "stats_target_map":  {int(k): v for k, v in stats_t_map.items()},
            "mask_future_ch":    mask_future,
            "normalize_bondpp":  apply_normbp,
            "normalize_stockpp": apply_normsp,
            "normalize_excess":  apply_normex,
            "lambda_aux":        lambda_aux,
            "config": dict(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                           d_cond=len(cols_cond), d_target=len(cols_target),
                           past_len=PAST_LEN, future_len=FUTURE_LEN, total_len=L),
            "variant": dict(id=spec["variant_id"], name=spec["name"]),
        }, ckpt_path)
    pd.DataFrame(log).to_csv(log_path, index=False)
    summary = dict(
        model_type="macro_gated_film_1step",
        variant_id=spec["variant_id"],
        variant_name=spec["name"],
        fold=fold_tag,
        seed=seed,
        cond_cols=cols_cond,
        target_cols=cols_target,
        mask_future_ch=mask_future,
        normalize_bondpp=apply_normbp,
        normalize_stockpp=apply_normsp,
        normalize_excess=apply_normex,
        lambda_aux=lambda_aux,
        target_stats_map={int(k): v for k, v in stats_t_map.items()},
        best_epoch=best_epoch,
        val_sp_return=best_val_sp,
        val_per_channel  =best_val_per_ch  if best_val_per_ch  is not None else {},
        test_per_channel =best_test_per_ch if best_test_per_ch is not None else {},
        n_params=n_params,
        device=str(device),
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: {ckpt_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="1-step ahead matrix trainer (v33)")
    ap.add_argument("--variant", type=int, required=True, choices=list(VARIANTS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--fold", default=None, choices=["F1", "F2", "F3"])
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--val-csv",   default=None)
    ap.add_argument("--test-csv",  default=None)
    ap.add_argument("--out-dir",   default=os.path.join(HERE, "result"))
    ap.add_argument("--normalize-bondpp",  action="store_true")
    ap.add_argument("--normalize-stockpp", action="store_true")
    ap.add_argument("--normalize-excess",  action="store_true")
    ap.add_argument("--lambda-aux", type=float, default=LAMBDA_AUX)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience",   type=int, default=PATIENCE)
    ap.add_argument("--batch",      type=int, default=BATCH)
    ap.add_argument("--lr",         type=float, default=LR)
    args = ap.parse_args()

    spec = build_spec(args.variant)

    if args.fold is not None:
        repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
        folds_dir = os.path.join(repo_root, "data", "folds_v33")
        train_csv = os.path.join(folds_dir, f"{args.fold}_train.csv")
        val_csv   = os.path.join(folds_dir, f"{args.fold}_val.csv")
        test_csv  = os.path.join(folds_dir, f"{args.fold}_test.csv")
        print(f"[--fold {args.fold}]")
    else:
        train_csv = os.path.join(HERE, "data", "weekly_v33_train.csv")
        val_csv   = None
        test_csv  = os.path.join(HERE, "data", "weekly_v33_test.csv")

    if args.train_csv: train_csv = args.train_csv
    if args.val_csv:   val_csv   = args.val_csv
    if args.test_csv:  test_csv  = args.test_csv

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for seed in args.seeds:
        r = run(spec, train_csv, val_csv, test_csv, args.out_dir, seed,
                max_epochs=args.max_epochs, patience=args.patience,
                batch=args.batch, lr=args.lr, fold_tag=args.fold,
                normalize_bondpp=args.normalize_bondpp,
                normalize_stockpp=args.normalize_stockpp,
                normalize_excess=args.normalize_excess,
                lambda_aux=args.lambda_aux)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        apply_normbp = bool(args.normalize_bondpp  and "bondpp_13w_lag"     in spec["cols_target"])
        apply_normsp = bool(args.normalize_stockpp and "stockpp_13w_lag"    in spec["cols_target"])
        apply_normex = bool(args.normalize_excess  and "excess_liq_yoy_lag" in spec["cols_target"])
        norm_tag = (("_normbp" if apply_normbp else "")
                    + ("_normsp" if apply_normsp else "")
                    + ("_normex" if apply_normex else ""))
        fold_str = f"_{args.fold}" if args.fold else ""
        df_rows = []
        for r in results:
            row = dict(seed=r["seed"], best_epoch=r["best_epoch"],
                       val_sp_return=r["val_sp_return"])
            for c, v in r["test_per_channel"].items():
                row[f"test_{c}"] = v
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        out_csv = os.path.join(
            args.out_dir,
            f"m1step_film_v{spec['variant_id']}_{spec['name']}{norm_tag}{fold_str}_multiseed.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[Variant {spec['variant_id']} {spec['name']}{norm_tag}{fold_str}] multi-seed (n={len(df)})")
        print(f"  val_sp_return: median={df.val_sp_return.median():+.4f}, mean={df.val_sp_return.mean():+.4f} ± {df.val_sp_return.std():.4f}")
        for col_name in spec["cols_target"]:
            col = f"test_{col_name}"
            if col in df.columns:
                print(f"  test {col_name:<22s}: median={df[col].median():+.4f}, mean={df[col].mean():+.4f} ± {df[col].std():.4f}")


if __name__ == "__main__":
    main()
