"""Unconditional 1D Neural Spline Flow on globally-standardized ε
with selectable base distribution (StandardNormal or Student-t).

ε 정의 (불변):
  ε = (r - r̄_train) / σ_train       (global, 단일 scalar 정규화)

Base 분포 선택:
  - normal     : StandardNormal(shape=[1])     — 기존, fat-tail inductive bias 없음
  - student_t  : StudentT(shape=[1], df=N)     — df=5 면 excess_kurt=6 (empirical +6.75 근처)
                                                  Bollerslev 1987 since equity return 표준

왜 Student-t 가 유의미한가:
  empirical train ε 가 skew −0.85, ex_kurt +6.75. Normal base + Spline transform 만으로는
  bulk-dominated loss landscape 때문에 tail 학습 saturate (v1: kurt 1.31, v2: kurt 1.64).
  Student-t base 는 fat-tail 을 base 자체에 inductive bias 로 주입 → Flow 의 spline
  transform 은 bulk 모양만 조정. df=5 면 base 자체가 excess_kurt=6 부터 시작.

Architecture defaults (v2 hyperparameter expansion 유지):
  num_layers=6, num_bins=16, tail_bound=10.0
  base=student_t, df=5.0

Output ckpt path (auto-named):
  base=student_t: result/scenario_3m_flow_1d_global_student_df{N}.pt
  base=normal   : result/scenario_3m_flow_1d_global_v2.pt

Ckpt 형식 (호환성 + 향후 자동 architecture 재구성):
  {
    "model_state": flow.state_dict(),
    "meta": {
        "base_kind": "student_t" or "normal",
        "df": 5.0,
        "num_layers": 6, "num_bins": 16, "tail_bound": 10.0,
        "n_train": 574,
        "r_mean_train": ..., "r_std_train": ...,
    }
  }

Usage:
  python colab/dual_3ch/train_flow_b_global.py                       # student_t, df=5 (default)
  python colab/dual_3ch/train_flow_b_global.py --base normal         # 기존 normal base
  python colab/dual_3ch/train_flow_b_global.py --base student_t --df 4.5
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

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    from nflows.flows.base import Flow
    from nflows.distributions.base import Distribution
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms import CompositeTransform, PiecewiseRationalQuadraticCDF
except ImportError:
    print("FATAL: nflows required.  pip install nflows")
    sys.exit(1)


# =====================================================================
# Student-t base distribution (nflows.Distribution interface wrap of torch.distributions.StudentT)
# =====================================================================

class StudentTBase(Distribution):
    """1D Student-t base distribution for nflows.

    Args:
      shape: distribution shape (e.g., [1] for 1D)
      df   : degrees of freedom (>0). df=5 → excess_kurt=6; df=4 → ∞ (avoid);
             df=6 → excess_kurt=3. For df ≤ 4, excess_kurt is undefined/∞.

    Implementation note:
      torch.distributions.StudentT 가 location/scale 을 안 받아서 (df 만), 표준 t (loc=0, scale=1)
      를 사용. _shape=[1] 에서 (batch, 1) input → log_prob (batch, 1) → sum last dim → (batch,).
    """
    def __init__(self, shape, df=5.0):
        super().__init__()
        if df <= 0:
            raise ValueError(f"Student-t df must be > 0, got {df}")
        self._shape = torch.Size(shape)
        self.register_buffer("_df", torch.tensor(float(df)))
        # Dummy buffer for device tracking (similar to StandardNormal._log_z)
        self.register_buffer("_device_anchor", torch.zeros(1))

    @property
    def df(self):
        return float(self._df.item())

    def _log_prob(self, inputs, context=None):
        # inputs: (batch, *self._shape) → for shape=[1]: (batch, 1)
        # torch.distributions.StudentT.log_prob applies elementwise; sum over event dims.
        dist = torch.distributions.StudentT(self._df)
        lp = dist.log_prob(inputs)                       # (batch, *self._shape)
        return lp.view(inputs.shape[0], -1).sum(dim=-1)  # (batch,)

    def _sample(self, num_samples, context):
        device = self._device_anchor.device
        dist = torch.distributions.StudentT(self._df)
        if context is None:
            # Return (num_samples, *self._shape)
            samples = dist.sample(sample_shape=torch.Size([num_samples, *self._shape]))
            return samples.to(device)
        else:
            # Return (n_ctx, num_samples, *self._shape)
            n_ctx = context.shape[0]
            samples = dist.sample(
                sample_shape=torch.Size([n_ctx, num_samples, *self._shape])
            )
            return samples.to(device)

    def _mean(self, context):
        device = self._device_anchor.device
        # Mean of Student-t is 0 (for df > 1). Lower df → undefined; we don't care for sampling.
        if context is None:
            return torch.zeros(*self._shape, device=device)
        return torch.zeros(context.shape[0], *self._shape, device=device)


# =====================================================================
# Build flow with configurable base
# =====================================================================

def build_flow(base_kind, df, num_layers, num_bins, tail_bound):
    """Build 1D Flow with selectable base distribution.

    base_kind ∈ {"normal", "student_t"}.  df is only used if base_kind=="student_t".
    """
    if base_kind == "normal":
        base = StandardNormal(shape=[1])
    elif base_kind == "student_t":
        base = StudentTBase(shape=[1], df=df)
    else:
        raise ValueError(f"unknown base_kind={base_kind}")
    transforms = []
    for _ in range(num_layers):
        transforms.append(PiecewiseRationalQuadraticCDF(
            shape=[1], num_bins=num_bins, tail_bound=tail_bound, tails="linear",
        ))
    return Flow(CompositeTransform(transforms), base)


# =====================================================================
# Default save-name auto-generation
# =====================================================================

def auto_save_name(base_kind, df):
    if base_kind == "normal":
        return "scenario_3m_flow_1d_global_v2.pt"
    if base_kind == "student_t":
        if df == int(df):
            df_str = str(int(df))
        else:
            df_str = f"{df:.1f}".replace(".", "p")
        return f"scenario_3m_flow_1d_global_student_df{df_str}.pt"
    raise ValueError(base_kind)


# =====================================================================
# Extract ε with global standardization (unchanged from prior versions)
# =====================================================================

def extract_eps_global(train_csv):
    df = pd.read_csv(train_csv)
    if "sp_return" not in df.columns:
        raise KeyError("train CSV missing 'sp_return' column")
    r = df["sp_return"].values.astype(np.float64)
    n = len(r)
    if n < 30:
        raise RuntimeError(f"too few train rows: {n}")
    r_mean = float(r.mean())
    r_std  = float(r.std(ddof=1))
    if r_std < 1e-12:
        raise RuntimeError(f"degenerate r_std={r_std}")
    eps = ((r - r_mean) / r_std).astype(np.float32)

    def moments(x):
        x = np.asarray(x, dtype=np.float64)
        m, s = float(x.mean()), float(x.std(ddof=1))
        if s < 1e-12:
            return m, s, float("nan"), float("nan")
        sk = float(((x - m) ** 3).mean() / (s ** 3))
        kt = float(((x - m) ** 4).mean() / (s ** 4) - 3.0)
        return m, s, sk, kt

    rm, rs, rsk, rkt = moments(r)
    em, es, esk, ekt = moments(eps)
    print(f"  sp_return:  n={n}, mean={rm:+.6f}, std={rs:.6f}, "
          f"skew={rsk:+.3f}, ex_kurt={rkt:+.3f}, min/max=[{r.min():+.4f},{r.max():+.4f}]")
    print(f"  ε (global): mean={em:+.4f}, std={es:.4f}, "
          f"skew={esk:+.3f}, ex_kurt={ekt:+.3f}, min/max=[{eps.min():+.3f},{eps.max():+.3f}]")
    print(f"    (skew/kurt scale-invariant — ε vs raw should match)")

    return torch.from_numpy(eps), dict(
        n=n, r_mean=rm, r_std=rs,
        raw_skew=rsk, raw_ex_kurt=rkt,
        eps_skew=esk, eps_ex_kurt=ekt,
        eps_min=float(eps.min()), eps_max=float(eps.max()),
    )


# =====================================================================
# Train loop with metadata-aware save / load
# =====================================================================

def train_flow_local(eps, save_path, base_kind, df, num_layers, num_bins, tail_bound,
                     epochs, batch, lr, device, log_every=20):
    """Train flow with given base. Save with metadata. Auto-load if exists."""
    if os.path.exists(save_path):
        print(f"  [SKIP] flow ckpt 이미 존재: {save_path}")
        state = torch.load(save_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state" in state and "meta" in state:
            meta = state["meta"]
            # Rebuild flow exactly per saved metadata (architecture must match)
            flow = build_flow(
                meta.get("base_kind", base_kind),
                meta.get("df", df),
                meta.get("num_layers", num_layers),
                meta.get("num_bins", num_bins),
                meta.get("tail_bound", tail_bound),
            ).to(device)
            flow.load_state_dict(state["model_state"])
            print(f"    [load] base={meta.get('base_kind','?')}, df={meta.get('df','?')}, "
                  f"layers={meta.get('num_layers','?')}, bins={meta.get('num_bins','?')}, "
                  f"tail={meta.get('tail_bound','?')}")
        else:
            # Legacy plain state_dict — assume current CLI args' architecture
            flow = build_flow(base_kind, df, num_layers, num_bins, tail_bound).to(device)
            sd = state["model_state"] if (isinstance(state, dict) and "model_state" in state) else state
            flow.load_state_dict(sd)
            print(f"    [load legacy plain state_dict — assuming current CLI architecture]")
        return flow

    flow = build_flow(base_kind, df, num_layers, num_bins, tail_bound).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    base_desc = f"student_t(df={df})" if base_kind == "student_t" else "normal"
    print(f"  Flow 1D NSF: base={base_desc}, layers={num_layers}, bins={num_bins}, "
          f"tail={tail_bound}, params={n_params:,}")

    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    eps_t = eps.unsqueeze(-1).to(device)   # (N, 1)
    N = eps_t.shape[0]
    print(f"  N (ε) = {N:,},  batch={batch},  epochs={epochs},  lr={lr}")

    best_nll = float("inf")
    for ep in range(1, epochs + 1):
        flow.train()
        perm = torch.randperm(N)
        losses = []
        for i in range(0, N, batch):
            xb = eps_t[perm[i : i + batch]]
            loss = -flow.log_prob(inputs=xb).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=5.0)
            opt.step()
            losses.append(loss.item())
        ep_loss = float(np.mean(losses))
        if ep_loss < best_nll:
            best_nll = ep_loss
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            print(f"    flow ep{ep:>3d}: nll = {ep_loss:+.4f}   (best so far {best_nll:+.4f})")

    torch.save({
        "model_state": flow.state_dict(),
        "meta": dict(
            base_kind=base_kind, df=float(df),
            num_layers=num_layers, num_bins=num_bins, tail_bound=tail_bound,
            n_params=n_params,
            final_train_nll=ep_loss,
            best_train_nll=best_nll,
        ),
    }, save_path)
    print(f"  saved: {save_path}   (final train NLL = {ep_loss:+.4f})")
    return flow


# =====================================================================
# Sanity check — generated moments vs train ε empirical
# =====================================================================

def sanity_check(flow, train_meta, device, n_sample=5000, seed=42):
    flow.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        eps_gen = flow.sample(n_sample).cpu().numpy().reshape(-1)

    def moments(x):
        m, s = float(np.mean(x)), float(np.std(x, ddof=1))
        if s < 1e-12:
            return m, s, float("nan"), float("nan")
        sk = float(np.mean((x - m) ** 3) / (s ** 3))
        kt = float(np.mean((x - m) ** 4) / (s ** 4) - 3.0)
        return m, s, sk, kt

    m, s, sk, kt = moments(eps_gen)
    print(f"\n  Sanity — generated ε moments (n={n_sample}):")
    print(f"    {'source':22s}  {'mean':>10s}  {'std':>8s}  {'skew':>8s}  {'ex_kurt':>10s}")
    print(f"    {'train ε empirical':22s}  "
          f"{0.0:+10.4f}  {1.0:>8.4f}  {train_meta['eps_skew']:+8.3f}  {train_meta['eps_ex_kurt']:+10.3f}")
    print(f"    {'Flow generated':22s}  {m:+10.4f}  {s:>8.4f}  {sk:+8.3f}  {kt:+10.3f}")

    se_skew = math.sqrt(6.0 / n_sample)
    se_kurt = math.sqrt(24.0 / n_sample)
    print(f"    SE(skew)≈{se_skew:.3f},  SE(ex_kurt)≈{se_kurt:.3f}")

    d_sk = sk - train_meta['eps_skew']
    d_kt = kt - train_meta['eps_ex_kurt']
    print(f"    Δskew (gen − emp)    = {d_sk:+.3f}   "
          f"({'OK' if abs(d_sk) < 3*se_skew else 'MISMATCH'} at 3·SE)")
    print(f"    Δex_kurt (gen − emp) = {d_kt:+.3f}   "
          f"({'OK' if abs(d_kt) < 3*se_kurt else 'MISMATCH'} at 3·SE)")
    return dict(
        gen_mean=m, gen_std=s, gen_skew=sk, gen_ex_kurt=kt,
        delta_skew=d_sk, delta_ex_kurt=d_kt,
        se_skew=se_skew, se_kurt=se_kurt,
    )


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Train UNCONDITIONAL 1D NSF on globally-standardized ε "
                    "with selectable base (normal or student_t)")
    ap.add_argument("--pilot-split", action="store_true", default=True)
    ap.add_argument("--train-csv",  default=None)
    ap.add_argument("--result-dir", default=os.path.join(HERE, "result"))
    ap.add_argument("--save-name",  default=None,
                    help="auto-generated from base+df if not specified")
    ap.add_argument("--base", choices=["normal", "student_t"], default="student_t",
                    help="base distribution (default: student_t — natural fat-tail inductive bias)")
    ap.add_argument("--df", type=float, default=5.0,
                    help="Student-t degrees of freedom (default 5 → excess_kurt=6 ≈ empirical +6.75)")
    ap.add_argument("--num-layers", type=int,   default=6)
    ap.add_argument("--num-bins",   type=int,   default=16)
    ap.add_argument("--tail-bound", type=float, default=10.0)
    ap.add_argument("--epochs",     type=int,   default=200)
    ap.add_argument("--batch",      type=int,   default=256)
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--seed",       type=int,   default=2026)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
    train_csv = args.train_csv or os.path.join(repo_root, "data", "pilot_split", "train.csv")
    if not os.path.exists(train_csv):
        sys.exit(f"[FATAL] missing train CSV: {train_csv}")
    os.makedirs(args.result_dir, exist_ok=True)

    save_name = args.save_name or auto_save_name(args.base, args.df)
    save_path = os.path.join(args.result_dir, save_name)

    print("=" * 78)
    print(" Train UNCONDITIONAL 1D NSF — globally-standardized ε")
    print("=" * 78)
    print(f"  device     : {device}")
    print(f"  train_csv  : {train_csv}")
    print(f"  save_path  : {save_path}")
    print(f"  ε def      : ε = (r - r̄_train) / σ_train    (global scalar)")
    base_desc = (f"Student-t(df={args.df})  — excess_kurt={6.0/(args.df-4):+.2f}"
                 if args.base == "student_t" and args.df > 4 else
                 ("Student-t (df ≤ 4: kurt undefined/∞)" if args.base == "student_t" else
                  "StandardNormal"))
    print(f"  base       : {base_desc}")
    print(f"  flow       : layers={args.num_layers}, bins={args.num_bins}, "
          f"tail_bound={args.tail_bound}")

    # 1. Extract ε
    print("\n[1] Extract ε with global standardization")
    eps, train_meta = extract_eps_global(train_csv)

    # 2. Train
    print("\n[2] Train unconditional 1D NSF")
    flow = train_flow_local(
        eps, save_path,
        base_kind=args.base, df=args.df,
        num_layers=args.num_layers, num_bins=args.num_bins, tail_bound=args.tail_bound,
        epochs=args.epochs, batch=args.batch, lr=args.lr, device=device,
    )

    # 3. Sanity
    print("\n[3] Sanity check — generated vs empirical moments")
    sanity = sanity_check(flow, train_meta, device, n_sample=5000)

    # 4. Meta json
    meta_json_path = save_path.replace(".pt", "_meta.json")
    meta_json = dict(
        model_type="unconditional_1d_nsf_global_eps",
        eps_definition="global: (r - r_mean_train) / r_std_train",
        train_csv=os.path.abspath(train_csv),
        base_kind=args.base,
        df=args.df,
        r_mean_train=train_meta["r_mean"],
        r_std_train=train_meta["r_std"],
        n_train=train_meta["n"],
        train_eps_skew=train_meta["eps_skew"],
        train_eps_ex_kurt=train_meta["eps_ex_kurt"],
        train_eps_min=train_meta["eps_min"],
        train_eps_max=train_meta["eps_max"],
        sanity=sanity,
        flow_config=dict(
            num_layers=args.num_layers, num_bins=args.num_bins,
            tail_bound=args.tail_bound, epochs=args.epochs,
            batch=args.batch, lr=args.lr,
        ),
    )
    with open(meta_json_path, "w") as f:
        json.dump(meta_json, f, indent=2)
    print(f"\n  meta saved: {meta_json_path}")

    # 5. Summary
    print("\n" + "=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    print(f"  Base distribution:  {base_desc}")
    print(f"  Train return:    skew={train_meta['raw_skew']:+.3f}, ex_kurt={train_meta['raw_ex_kurt']:+.3f}")
    print(f"  Train ε (=raw):  skew={train_meta['eps_skew']:+.3f}, ex_kurt={train_meta['eps_ex_kurt']:+.3f}")
    print(f"  Flow generated:  skew={sanity['gen_skew']:+.3f}, ex_kurt={sanity['gen_ex_kurt']:+.3f}")
    print(f"  → Δskew={sanity['delta_skew']:+.3f}  Δex_kurt={sanity['delta_ex_kurt']:+.3f}")
    print(f"  → 5000-path SE: skew ≈ ±{math.sqrt(6/5000):.3f}, ex_kurt ≈ ±{math.sqrt(24/5000):.3f}")
    print(f"\n  Next: python colab/dual_3ch/sensitivity_v13.py "
          f"--flow-mode global --global-flow-ckpt {save_name}")


if __name__ == "__main__":
    main()
