"""Mamba 하락 예측기: 하위(일별 63일 시퀀스) + 상위(γ, vrp 등 월별 피쳐)
Target: 3M forward return <= -5%
"""
import sys, torch, torch.nn as nn  # torch must be imported before numpy on Windows
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass
from mambapy.mamba import Mamba, MambaConfig
import yfinance as yf
from sklearn.metrics import roc_auc_score

# ── 1. Daily data ──
sp_daily = yf.download("^GSPC", start="1989-01-01", end="2026-04-01", interval="1d", progress=False)
sp_close = sp_daily["Close"]
if isinstance(sp_close, pd.DataFrame): sp_close = sp_close.iloc[:,0]
sp_ret_daily = sp_close.pct_change().dropna()

vix_daily = yf.download("^VIX", start="1989-01-01", end="2026-04-01", interval="1d", progress=False)
vix_d = vix_daily["Close"]
if isinstance(vix_d, pd.DataFrame): vix_d = vix_d.iloc[:,0]

print(f"Daily: {sp_ret_daily.index[0].date()} ~ {sp_ret_daily.index[-1].date()}, {len(sp_ret_daily)} days")

# ── 2. Monthly data ──
full = pd.concat([
    pd.read_csv("data/monthly_noleak_v28_train.csv"),
    pd.read_csv("data/monthly_noleak_v28_test.csv"),
]).reset_index(drop=True)
full["date"] = pd.to_datetime(full["date"])
full = full.sort_values("date").reset_index(drop=True)
N = len(full)

r = full["sp_next_return"].fillna(0).values
y3 = np.full(N, np.nan)
for t in range(N - 3):
    y3[t] = (1 + r[t]) * (1 + r[t+1]) * (1 + r[t+2]) - 1.0
full["y3"] = y3
full["label_dn"] = (full["y3"] <= -0.05).astype(float)
full["label_up"] = (full["y3"] >= 0.10).astype(float)

# ── 3. Extract 63-day sequences ──
SEQ_LEN = 63
daily_sequences = {}
for idx in range(N):
    month_end = full.loc[idx, "date"]
    rets_before = sp_ret_daily[sp_ret_daily.index <= month_end].tail(SEQ_LEN)
    vix_before = vix_d[vix_d.index <= month_end].tail(SEQ_LEN)
    if len(rets_before) >= SEQ_LEN:
        common = rets_before.index.intersection(vix_before.index)
        if len(common) >= SEQ_LEN:
            r_seq = rets_before.loc[common[-SEQ_LEN:]].values
            v_seq = vix_before.loc[common[-SEQ_LEN:]].values / 100
            daily_sequences[idx] = np.stack([r_seq, v_seq], axis=1)

print(f"Sequences: {len(daily_sequences)}/{N}")

# ── 4. Model ──
class MambaDown(nn.Module):
    def __init__(self, seq_in=2, d_model=16, n_layers=2, mf_dim=10, hid=32):
        super().__init__()
        self.proj = nn.Linear(seq_in, d_model)
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
        self.head = nn.Sequential(
            nn.Linear(d_model + mf_dim, hid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hid, 1))
    def forward(self, seq, mf):
        x = self.mamba(self.proj(seq))
        return self.head(torch.cat([x[:,-1,:], mf], 1)).squeeze(1)

EXCLUDE_MF = {"date", "sp_next_return", "y3", "label_dn", "label_up"}
MF = [c for c in full.columns if c not in EXCLUDE_MF]

WINDOWS = {
    "W1": (("1991-01-01","2010-12-31"),("2011-07-01","2015-12-31")),
    "W2": (("1996-01-01","2015-12-31"),("2016-07-01","2020-12-31")),
    "W3": (("2001-01-01","2020-12-31"),("2021-07-01","2025-12-31")),
}

def run_experiment(target_col, target_label, threshold_str):
    print(f"\n{'='*100}")
    print(f"Mamba(63d daily) + all monthly features -> {target_label}")
    print(f"Monthly features: {len(MF)}")
    print(f"{'='*100}")

    for fn, ((trs,tre),(tes,tee)) in WINDOWS.items():
        tr_mask = (full["date"]>=trs) & (full["date"]<=tre)
        te_mask = (full["date"]>=tes) & (full["date"]<=tee)
        tr_i = [i for i in np.where(tr_mask)[0] if i in daily_sequences and not np.isnan(full.loc[i,"y3"])]
        te_i = [i for i in np.where(te_mask)[0] if i in daily_sequences]

        Xs_tr = torch.tensor(np.array([daily_sequences[i] for i in tr_i]), dtype=torch.float32)
        mf_tr_raw = full.loc[tr_i, MF].values.astype(np.float32)
        y_tr = torch.tensor(full.loc[tr_i, target_col].values, dtype=torch.float32)

        Xs_te = torch.tensor(np.array([daily_sequences[i] for i in te_i]), dtype=torch.float32)
        mf_te_raw = full.loc[te_i, MF].values.astype(np.float32)
        y_te = full.loc[te_i, target_col].values
        y3_te = full.loc[te_i, "y3"].values

        mu, sd = mf_tr_raw.mean(0), mf_tr_raw.std(0) + 1e-8
        Xm_tr = torch.tensor((mf_tr_raw - mu) / sd, dtype=torch.float32)
        Xm_te = torch.tensor((mf_te_raw - mu) / sd, dtype=torch.float32)

        n_events = int(y_tr.sum())
        print(f"\n{'_'*80}")
        print(f"  {fn}: train {len(tr_i)}, test {len(te_i)}, events={n_events} ({n_events/len(tr_i)*100:.1f}%)")

        model = MambaDown(seq_in=2, d_model=16, n_layers=2, mf_dim=len(MF), hid=32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        pw = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)])
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)

        model.train()
        best_loss = 1e9
        best_st = None
        pat = 0
        for ep in range(200):
            opt.zero_grad()
            loss = crit(model(Xs_tr, Xm_tr), y_tr)
            loss.backward()
            opt.step()
            if loss.item() < best_loss - 1e-5:
                best_loss = loss.item()
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
            if pat >= 30:
                break
        print(f"  Epochs: {ep+1}, best_loss={best_loss:.4f}")

        model.load_state_dict(best_st)
        model.eval()
        with torch.no_grad():
            p_out = torch.sigmoid(model(Xs_te, Xm_te)).numpy()

        vld = ~np.isnan(y3_te)
        if y_te[vld].sum() > 0 and y_te[vld].sum() < vld.sum():
            auc = roc_auc_score(y_te[vld], p_out[vld])
        else:
            auc = np.nan
        print(f"  Mamba AUC: {auc:.3f}" if not np.isnan(auc) else "  Mamba AUC: N/A")
        print(f"  p: min={p_out.min():.4f} med={np.median(p_out):.4f} max={p_out.max():.4f}")

        events = [i for i in range(len(te_i)) if y_te[i] == 1 and vld[i]]
        if events:
            print(f"  Actual events:")
            for i in events:
                d = full.loc[te_i[i], "date"]
                print(f"    {d.date()}: p={p_out[i]:.4f} 3M={y3_te[i]*100:+.1f}% g={full.loc[te_i[i],'gamma']:.2f}")

        top5 = np.argsort(p_out)[-5:][::-1]
        print(f"  Top 5 p:")
        for i in top5:
            d = full.loc[te_i[i], "date"]
            y3s = f"{y3_te[i]*100:+.1f}%" if not np.isnan(y3_te[i]) else "N/A"
            h = "HIT" if y_te[i] == 1 else ""
            print(f"    {d.date()}: p={p_out[i]:.4f} 3M={y3s} g={full.loc[te_i[i],'gamma']:.2f} {h}")


# Run both
run_experiment("label_up", "3M >= +10%", "+10")
run_experiment("label_dn", "3M <= -5%", "-5")
