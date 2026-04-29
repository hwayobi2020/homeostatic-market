"""K2 Flow 학습 — condition channel 비교 (A: 4ch, B: 6ch, C: 7ch).

A: m2_growth, m2v, cpi_yoy, vix                                        (baseline 4)
B: A + tbill_26w_lag + excess_liq_26w_lag                              (+raw cumulative, 6)
C: B + pp_bond_26w_lag                                                 (+homeostatic stress, 7)

Target: sp_return, margin_chg (v33/v34 와 동일 2채널)
Architecture: MultiStepFAVARFlow K=2, d_model=64, n_heads=4, n_layers=2 (K2_pure)
Loss: NLL only (no wavelet, no PINN penalty)
"""
import torch
import sys, os, math, json, argparse
import numpy as np, pandas as pd
import warnings
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim.favar_flow import MultiStepFAVARFlow

# Constants
L = 104; PAST_LEN = 52; K_STEPS = 2
D_MODEL = 64; N_HEADS = 4; N_LAYERS = 2
LR = 5e-4; BATCH = 32; MAX_EPOCHS = 60; PATIENCE = 15

COLS_TARGET = ["sp_return"]   # D_TARGET=1 (margin_chg sparse trivial collapse 회피)
COND_SETS = {
    "A_4ch": ["m2_growth", "m2v", "cpi_yoy", "vix"],
    "B_6ch": ["m2_growth", "m2v", "cpi_yoy", "vix", "tbill_26w_lag", "excess_liq_26w_lag"],
    "C_7ch": ["m2_growth", "m2v", "cpi_yoy", "vix", "tbill_26w_lag", "excess_liq_26w_lag", "pp_bond_26w_lag"],
    "Cprime_5ch": ["m2_growth", "m2v", "cpi_yoy", "vix", "pp_bond_26w_lag"],   # pp_bond only, raw cumulative 제거 (collinearity fix)
    "K2_104": ["excess_liq_yoy", "tbill_wr", "tbill_26w_lag", "excess_liq_26w_lag", "vix"],   # liquidity + vix condition (no m2/cpi)
}

LOG2PI = math.log(2 * math.pi)


def load_windows(csv_path, cols_cond, cols_target, L=104):
    df = pd.read_csv(csv_path)
    n = len(df)
    n_w = n - L + 1
    if n_w <= 0:
        return None, None, None
    X = np.zeros((n_w, L, len(cols_target)), dtype=np.float32)
    C = np.zeros((n_w, L, len(cols_cond)), dtype=np.float32)
    for i in range(n_w):
        X[i] = df[cols_target].iloc[i:i+L].values
        C[i] = df[cols_cond].iloc[i:i+L].values
    # z-score condition (per channel)
    mu = C.reshape(-1, len(cols_cond)).mean(axis=0)
    sd = C.reshape(-1, len(cols_cond)).std(axis=0) + 1e-8
    C = (C - mu) / sd
    return torch.from_numpy(X), torch.from_numpy(C), {"mean": mu.tolist(), "std": sd.tolist()}


def nll_per_step_channel(X, C, model, past_len, device):
    """Mean NLL per step per channel over future portion."""
    B, L_, D = X.shape
    F = L_ - past_len
    z, log_det_J, log_scale = model(X.to(device), C.to(device))
    nll_td = 0.5 * z.pow(2) + 0.5 * LOG2PI + log_scale        # [B, L, D] — v33 정합 (sign fix 2026-04-28)
    nll_future = nll_td[:, past_len:, :].sum(dim=(1,2))        # [B]
    return nll_future / (F * D)


def run(condition_name, train_csv, test_csv, save_dir, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    cols_cond = COND_SETS[condition_name]
    print(f"\n{'='*70}\n[{condition_name}] seed={seed} cond={cols_cond} (D_COND={len(cols_cond)})\n{'='*70}")

    Xtr, Ctr, stats_tr = load_windows(train_csv, cols_cond, COLS_TARGET, L=L)
    Xte, Cte, _ = load_windows(test_csv, cols_cond, COLS_TARGET, L=L)
    if Xtr is None or Xte is None:
        print("[FAIL] not enough windows")
        return None
    n_w_tr = Xtr.shape[0]
    n_val = max(int(n_w_tr * 0.15), 1)
    Xtr_, Ctr_ = Xtr[:-n_val], Ctr[:-n_val]
    Xv,  Cv  = Xtr[-n_val:], Ctr[-n_val:]
    print(f"  train_windows={Xtr_.shape[0]}, val_windows={Xv.shape[0]}, test_windows={Xte.shape[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiStepFAVARFlow(K=K_STEPS, d_cond=len(cols_cond), d_target=len(COLS_TARGET),
                                d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                                time_reverse=False, use_wavelet=False).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    best_val = float('inf'); best_state = None; pat = 0
    log = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(Xtr_.shape[0])
        losses = []
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            loss = nll_per_step_channel(Xtr_[idx], Ctr_[idx], model, PAST_LEN, device).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_nll = nll_per_step_channel(Xv, Cv, model, PAST_LEN, device).mean().item()
            test_nll = nll_per_step_channel(Xte, Cte, model, PAST_LEN, device).mean().item()
        train_nll = float(np.mean(losses))
        print(f"  ep{epoch:>3d}  train={train_nll:+.4f}  val={val_nll:+.4f}  test={test_nll:+.4f}")
        log.append(dict(epoch=epoch, train=train_nll, val=val_nll, test=test_nll))

        if val_nll < best_val - 1e-4:
            best_val = val_nll; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_test = test_nll; best_epoch = epoch; pat = 0
        else:
            pat += 1
        if pat >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    print(f"\n  best epoch {best_epoch}: val={best_val:+.4f}, test={best_test:+.4f}")
    return dict(condition=condition_name, seed=seed, n_cond=len(cols_cond), best_epoch=best_epoch,
                val=best_val, test=best_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["A_4ch","B_6ch","C_7ch"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777, 0, 99])
    args = ap.parse_args()

    train_csv = os.path.join(ROOT, "data", "weekly_ppbond_train.csv")
    test_csv  = os.path.join(ROOT, "data", "weekly_ppbond_test.csv")
    save_dir  = os.path.join(ROOT, "result")
    os.makedirs(save_dir, exist_ok=True)

    results = []
    for c in args.conditions:
        for seed in args.seeds:
            r = run(c, train_csv, test_csv, save_dir, seed=seed)
            if r is not None:
                results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "favar_ppbond_multiseed_results.csv"), index=False)

    print(f"\n{'='*70}\nMULTI-SEED SUMMARY (n_seeds={len(args.seeds)})\n{'='*70}")
    print(f"{'cond':>10s} | {'n_cond':>7s} | {'val_NLL':>22s} | {'test_NLL':>22s}")
    agg = {}
    for c in args.conditions:
        sub = df[df.condition == c]
        v_mean, v_std = sub.val.mean(), sub.val.std()
        t_mean, t_std = sub.test.mean(), sub.test.std()
        agg[c] = (v_mean, v_std, t_mean, t_std)
        n_cond = sub.n_cond.iloc[0]
        print(f"  {c:>8s} | {n_cond:>5d}   | {v_mean:+.4f} +- {v_std:.4f} | {t_mean:+.4f} +- {t_std:.4f}")

    if "A_4ch" in agg:
        print(f"\n=== Delta vs A_4ch (test NLL mean, multi-seed) ===")
        base_t = agg["A_4ch"][2]
        for c in args.conditions:
            if c != "A_4ch":
                d = agg[c][2] - base_t
                print(f"  {c:>10s} - A_4ch: {d:+.4f}  (negative = improvement)")

        # Pairwise deltas (per-seed) for paired-t-test
        from scipy import stats as sst
        print(f"\n=== Paired t-test (test NLL, per-seed) ===")
        for c in args.conditions:
            if c == "A_4ch": continue
            base_test = df[df.condition=="A_4ch"].sort_values("seed")["test"].values
            comp_test = df[df.condition==c].sort_values("seed")["test"].values
            tstat, pval = sst.ttest_rel(comp_test, base_test)
            print(f"  {c:>10s} vs A_4ch: t={tstat:+.3f}, p={pval:.4f}, mean Δ={np.mean(comp_test-base_test):+.4f}")


if __name__ == "__main__":
    main()
