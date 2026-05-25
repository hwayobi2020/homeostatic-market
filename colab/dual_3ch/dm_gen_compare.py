"""Diebold-Mariano test: garch-flow (A) vs baselines (VAE / GAN / GARCH-FHS).

Paired on per-origin mean CRPS (loaded from each model's *_crps_per_origin.npy,
which are aligned: same fold -> same load_windows_seq + valid_mask -> same origins
in the same order).  HAC (Newey-West) variance handles the overlapping-window
autocorrelation of the per-origin loss differential.

H0: equal predictive accuracy.  dCRPS = mean(loss_flow) - mean(loss_baseline):
  dCRPS < 0  => garch-flow has lower CRPS (better).  p<0.05 => significant.

Usage (Colab): !python colab/dual_3ch/dm_gen_compare.py
Requires the *_crps_per_origin.npy for garch_flow_ar / vae_baseline /
gan_baseline / garch_fhs (re-run those scripts after the per-origin save was added).
"""
import os

import numpy as np

try:
    from scipy.stats import norm
    _cdf = norm.cdf
except Exception:                       # tiny fallback if scipy absent
    import math
    def _cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "result")
FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
BASELINES = [("VAE", "vae_baseline"), ("GAN", "gan_baseline"),
             ("GARCH-FHS", "garch_fhs")]
MAXLAG = 13                              # overlapping 13-week windows


def _load(prefix, fold):
    p = os.path.join(RES, f"{prefix}_{fold}_crps_per_origin.npy")
    return np.load(p) if os.path.exists(p) else None


def dm_test(a, b, maxlag=MAXLAG):
    """a, b: (n_origin, T) per-origin CRPS.  Paired DM on per-origin mean loss."""
    la, lb = a.mean(axis=1), b.mean(axis=1)
    d = la - lb
    n = len(d)
    dbar = float(d.mean())
    dc = d - dbar
    var = float(np.mean(dc * dc))
    for k in range(1, min(maxlag, n - 1) + 1):
        w = 1.0 - k / (maxlag + 1.0)
        var += 2.0 * w * float(np.mean(dc[k:] * dc[:-k]))
    se = np.sqrt(var / n) if var > 0 else float("nan")
    stat = dbar / se if se and se == se and se > 0 else float("nan")
    p = 2.0 * (1.0 - _cdf(abs(stat))) if stat == stat else float("nan")
    return dbar, stat, p


def main():
    print("Diebold-Mariano: garch-flow (A) vs baseline (B)")
    print("  dCRPS = mean_A - mean_B  (<0 => garch-flow better);  HAC maxlag=%d" % MAXLAG)
    print("=" * 78)
    for fold in FOLDS:
        gf = _load("garch_flow_ar", fold)
        if gf is None:
            print(f"\n[{fold}] garch_flow_ar per-origin npy 없음 → garch-flow 재실행 필요")
            continue
        print(f"\n[{fold}]  n_origin = {gf.shape[0]}")
        for name, pref in BASELINES:
            b = _load(pref, fold)
            if b is None:
                print(f"   {name:10s}: (npy 없음 — 해당 모델 재실행 필요)")
                continue
            if b.shape != gf.shape:
                print(f"   {name:10s}: shape mismatch {b.shape} vs {gf.shape} (정렬 불가)")
                continue
            dbar, stat, p = dm_test(gf, b)
            sig = "** SIG" if (p == p and p < 0.05) else "   n.s."
            verd = "flow better" if dbar < 0 else "baseline better"
            print(f"   {name:10s}: dCRPS={dbar:+.5f}  DM={stat:+.2f}  p={p:.4f}  {sig}  ({verd})")
    print("\n해석:")
    print("  VAE/GAN: dCRPS<0 & p<0.05 → 'VAE/GAN 대비 유의미한 향상' 정당.")
    print("  GARCH-FHS: p>0.05 (n.s.) → '대등' 정당 (이김 주장 안 함의 근거).")


if __name__ == "__main__":
    main()
