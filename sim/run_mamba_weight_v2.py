"""Mamba weight learner v2: log-sum loss + survival penalty
Loss = -sum(log_pp_chg) + lambda * survival_penalty
"""
import sys, torch, torch.nn as nn
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass
from mambapy.mamba import Mamba, MambaConfig
import yfinance as yf

# ── Data ──
sp_daily = yf.download("^GSPC", start="1989-01-01", end="2026-04-01", interval="1d", progress=False)
sp_close = sp_daily["Close"]
if isinstance(sp_close, pd.DataFrame): sp_close = sp_close.iloc[:,0]
sp_ret_daily = sp_close.pct_change().dropna()

vix_daily = yf.download("^VIX", start="1989-01-01", end="2026-04-01", interval="1d", progress=False)
vix_d = vix_daily["Close"]
if isinstance(vix_d, pd.DataFrame): vix_d = vix_d.iloc[:,0]

full = pd.concat([
    pd.read_csv("data/monthly_noleak_v28_train.csv"),
    pd.read_csv("data/monthly_noleak_v28_test.csv"),
]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full = full.sort_values("date").reset_index(drop=True)
N = len(full)
full["metab"] = np.maximum(np.maximum(full["m2_growth"], full["tbill"]/12), full["mich"]/100/12)

SEQ_LEN = 63
daily_sequences = {}
for idx in range(N):
    month_end = full.loc[idx, "date"]
    rets_before = sp_ret_daily[sp_ret_daily.index <= month_end].tail(SEQ_LEN)
    vix_before = vix_d[vix_d.index <= month_end].tail(SEQ_LEN)
    if len(rets_before) >= SEQ_LEN:
        common = rets_before.index.intersection(vix_before.index)
        if len(common) >= SEQ_LEN:
            daily_sequences[idx] = np.stack([
                rets_before.loc[common[-SEQ_LEN:]].values,
                vix_before.loc[common[-SEQ_LEN:]].values / 100
            ], axis=1)

EXCLUDE_MF = {"date", "sp_next_return", "metab"}
MF = [c for c in full.columns if c not in EXCLUDE_MF]

class MambaWeight(nn.Module):
    def __init__(self, seq_in=2, d_model=16, n_layers=2, mf_dim=36, hid=32):
        super().__init__()
        self.proj = nn.Linear(seq_in, d_model)
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
        self.head = nn.Sequential(
            nn.Linear(d_model + mf_dim, hid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hid, 1), nn.Sigmoid())
    def forward(self, seq, mf):
        x = self.mamba(self.proj(seq))
        return self.head(torch.cat([x[:,-1,:], mf], 1)).squeeze(1)

WINDOWS = {
    "W1": (("1991-01-01","2010-12-31"),("2011-07-01","2015-12-31")),
    "W2": (("1996-01-01","2015-12-31"),("2016-07-01","2020-12-31")),
    "W3": (("2001-01-01","2020-12-31"),("2021-07-01","2025-12-31")),
}

def sharpe(rets, tb):
    if rets.std() < 1e-10: return 0.0
    return (rets.mean() - tb.mean()) / rets.std() * np.sqrt(12)

def calc_mdd(rets):
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())

# ── Sweep: loss variants ──
configs = [
    # log-sum only (no survival penalty)
    {"survival_w": 0.0, "d_model": 16, "hid": 32, "lr": 1e-3, "label": "log-sum only"},
    {"survival_w": 0.0, "d_model": 32, "hid": 64, "lr": 1e-3, "label": "log-sum,big"},
    # log-sum + survival penalty
    {"survival_w": 1.0, "d_model": 16, "hid": 32, "lr": 1e-3, "label": "surv=1"},
    {"survival_w": 5.0, "d_model": 16, "hid": 32, "lr": 1e-3, "label": "surv=5"},
    {"survival_w": 1.0, "d_model": 32, "hid": 64, "lr": 1e-3, "label": "surv=1,big"},
    {"survival_w": 5.0, "d_model": 32, "hid": 64, "lr": 1e-3, "label": "surv=5,big"},
    # slower lr
    {"survival_w": 1.0, "d_model": 32, "hid": 64, "lr": 3e-4, "label": "surv=1,big,lr3e-4"},
]

print(f"{'='*130}")
print(f"Mamba Weight v2: log-sum loss + survival penalty")
print(f"  loss = -sum(log(1+port_ret) - log(1+metab)) + surv_w * mean(relu(1-PP)^2)")
print(f"{'='*130}")

print(f"\n{'Config':<22}", end="")
for fn in WINDOWS:
    print(f"  {fn+' Ret':>8} {fn+' Sh':>7} {fn+' Cal':>7} {fn+' MDD':>7} {fn+' W':>6}", end="")
print()
print("-" * 130)

for cfg in configs:
    results = {}
    for fn, ((trs,tre),(tes,tee)) in WINDOWS.items():
        tr_mask = (full["date"]>=trs)&(full["date"]<=tre)
        te_mask = (full["date"]>=tes)&(full["date"]<=tee)
        tr_i = [i for i in np.where(tr_mask)[0]
                if i in daily_sequences and pd.notna(full.loc[i,"sp_next_return"])]
        te_i = [i for i in np.where(te_mask)[0] if i in daily_sequences]

        Xs_tr = torch.tensor(np.array([daily_sequences[i] for i in tr_i]), dtype=torch.float32)
        mf_tr_raw = full.loc[tr_i, MF].values.astype(np.float32)
        sp_next_tr = torch.tensor(full.loc[tr_i,"sp_next_return"].values, dtype=torch.float32)
        tb_tr = torch.tensor((full.loc[tr_i,"tbill"].values/12).astype(np.float32))
        metab_tr = torch.tensor(full.loc[tr_i,"metab"].values.astype(np.float32))

        Xs_te = torch.tensor(np.array([daily_sequences[i] for i in te_i]), dtype=torch.float32)
        mf_te_raw = full.loc[te_i, MF].values.astype(np.float32)
        sp_next_te = full.loc[te_i,"sp_next_return"].values
        tb_te_np = full.loc[te_i,"tbill"].values/12
        metab_te_np = full.loc[te_i,"metab"].values

        mu, sd = mf_tr_raw.mean(0), mf_tr_raw.std(0)+1e-8
        Xm_tr = torch.tensor((mf_tr_raw-mu)/sd, dtype=torch.float32)
        Xm_te = torch.tensor((mf_te_raw-mu)/sd, dtype=torch.float32)

        torch.manual_seed(42)
        model = MambaWeight(seq_in=2, d_model=cfg["d_model"], n_layers=2, mf_dim=len(MF), hid=cfg["hid"])
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

        model.train()
        best_loss = 1e9; best_st = None; pat = 0
        for ep in range(300):
            opt.zero_grad()
            w = model(Xs_tr, Xm_tr)

            # Portfolio return
            port_ret = w * sp_next_tr + (1-w) * tb_tr

            # Loss 1: log-sum (maximizes log terminal PP)
            log_pp_chg = torch.log(1 + port_ret + 1e-8) - torch.log(1 + metab_tr + 1e-8)
            loss_growth = -log_pp_chg.sum()

            # Loss 2: survival penalty (PP trajectory below 1.0)
            # log space cumsum으로 계산 — cumprod/for loop보다 gradient 안정적
            if cfg["survival_w"] > 0:
                log_pp_cumsum = torch.cumsum(log_pp_chg, dim=0)
                pp_traj = torch.exp(log_pp_cumsum)  # PP trajectory (1.0 기준)
                loss_survival = torch.relu(1.0 - pp_traj).pow(2).mean() * cfg["survival_w"]
            else:
                loss_survival = torch.tensor(0.0)

            loss = loss_growth + loss_survival
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                best_st = {k:v.clone() for k,v in model.state_dict().items()}
                pat=0
            else:
                pat+=1
            if pat>=40: break

        model.load_state_dict(best_st); model.eval()
        with torch.no_grad():
            w_te = model(Xs_te, Xm_te).numpy()

        T = len(te_i)
        rets = w_te * sp_next_te + (1-w_te) * tb_te_np
        eq = np.cumprod(1+rets)
        ann = eq[-1]**(12/T)-1
        mdd = calc_mdd(rets)
        sh = sharpe(rets, np.full(T, tb_te_np.mean()))
        cal = ann/abs(mdd) if mdd < 0 else -999
        results[fn] = (ann, sh, cal, mdd, w_te.mean())

    print(f"{cfg['label']:<22}", end="")
    for fn in WINDOWS:
        ann, sh, cal, mdd, mw = results[fn]
        print(f"  {ann*100:>+7.2f}% {sh:>6.3f} {cal:>6.3f} {mdd*100:>6.1f}% {mw:>5.2f}", end="")
    print()

# B&H
print(f"\n{'B&H':<22}", end="")
for fn, ((trs,tre),(tes,tee)) in WINDOWS.items():
    te_mask = (full["date"]>=tes)&(full["date"]<=tee)
    te_i = np.where(te_mask)[0]
    sp = full.loc[te_i,"sp_next_return"].values
    tb = full.loc[te_i,"tbill"].values/12
    eq = np.cumprod(1+sp); T=len(sp)
    ann=eq[-1]**(12/T)-1; mdd=calc_mdd(sp)
    sh=sharpe(sp,np.full(T,tb.mean())); cal=ann/abs(mdd) if mdd<0 else -999
    print(f"  {ann*100:>+7.2f}% {sh:>6.3f} {cal:>6.3f} {mdd*100:>6.1f}% {1.0:>5.2f}", end="")
print()
