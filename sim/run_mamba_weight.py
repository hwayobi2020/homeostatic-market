"""Mamba 비중 학습: 일별 시퀀스 + 월별 피쳐 → w (portfolio weight)
Loss: PP 변화율 기반 — 항상성 유지를 직접 최적화
"""
import sys, torch, torch.nn as nn
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass
from mambapy.mamba import Mamba, MambaConfig
import yfinance as yf

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

# metabolism
full["metab"] = np.maximum(
    np.maximum(full["m2_growth"], full["tbill"] / 12),
    full["mich"] / 100 / 12
)

# ── 3. 63-day sequences ──
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

EXCLUDE_MF = {"date", "sp_next_return", "metab"}
MF = [c for c in full.columns if c not in EXCLUDE_MF]

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

print(f"\n{'='*100}")
print("Mamba Weight Learner: daily seq + monthly features -> w")
print(f"Loss: -(PP change) + drawdown penalty")
print(f"Monthly features: {len(MF)}")
print(f"{'='*100}")

for fn, ((trs,tre),(tes,tee)) in WINDOWS.items():
    tr_mask = (full["date"]>=trs) & (full["date"]<=tre)
    te_mask = (full["date"]>=tes) & (full["date"]<=tee)
    tr_i = [i for i in np.where(tr_mask)[0]
            if i in daily_sequences and pd.notna(full.loc[i,"sp_next_return"])]
    te_i = [i for i in np.where(te_mask)[0] if i in daily_sequences]

    Xs_tr = torch.tensor(np.array([daily_sequences[i] for i in tr_i]), dtype=torch.float32)
    mf_tr_raw = full.loc[tr_i, MF].values.astype(np.float32)
    sp_next_tr = torch.tensor(full.loc[tr_i, "sp_next_return"].values, dtype=torch.float32)
    tb_tr = torch.tensor((full.loc[tr_i, "tbill"].values / 12).astype(np.float32))
    metab_tr = torch.tensor(full.loc[tr_i, "metab"].values.astype(np.float32))

    Xs_te = torch.tensor(np.array([daily_sequences[i] for i in te_i]), dtype=torch.float32)
    mf_te_raw = full.loc[te_i, MF].values.astype(np.float32)
    sp_next_te = full.loc[te_i, "sp_next_return"].values
    tb_te_np = full.loc[te_i, "tbill"].values / 12
    metab_te_np = full.loc[te_i, "metab"].values

    mu, sd = mf_tr_raw.mean(0), mf_tr_raw.std(0) + 1e-8
    Xm_tr = torch.tensor((mf_tr_raw - mu) / sd, dtype=torch.float32)
    Xm_te = torch.tensor((mf_te_raw - mu) / sd, dtype=torch.float32)

    print(f"\n{'_'*80}")
    print(f"  {fn}: train {len(tr_i)}, test {len(te_i)}")

    model = MambaWeight(seq_in=2, d_model=16, n_layers=2, mf_dim=len(MF), hid=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    best_loss = 1e9
    best_st = None
    pat = 0

    for ep in range(300):
        opt.zero_grad()
        w = model(Xs_tr, Xm_tr)  # (batch,) in [0,1]

        # Portfolio return
        port_ret = w * sp_next_tr + (1 - w) * tb_tr

        # PP change: (1 + port_ret) / (1 + metabolism) - 1
        pp_chg = (1 + port_ret) / (1 + metab_tr) - 1

        # Loss: maximize PP change, penalize negative PP changes (drawdown)
        loss_return = -pp_chg.mean()
        loss_drawdown = torch.relu(-pp_chg).pow(2).mean() * 5.0  # 하락 시 추가 패널티
        loss_reg = (w * (1 - w)).mean() * 0.1  # w가 0 또는 1에 가깝도록 유도 (결정적)

        loss = loss_return + loss_drawdown + loss_reg

        loss.backward()
        opt.step()

        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= 40:
            break

    print(f"  Epochs: {ep+1}, best_loss={best_loss:.4f}")

    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        w_te = model(Xs_te, Xm_te).numpy()

    T = len(te_i)
    rets_mamba = w_te * sp_next_te + (1 - w_te) * tb_te_np
    rets_bh = sp_next_te

    # PP trajectory
    pp = 1.0
    pp_hist = [pp]
    for t in range(T):
        pp = pp * (1 + rets_mamba[t]) / (1 + metab_te_np[t])
        pp_hist.append(pp)

    for label, rets, ws in [
        ("Mamba weight", rets_mamba, w_te),
        ("B&H", rets_bh, np.ones(T)),
    ]:
        eq = np.cumprod(1 + rets)
        ann = eq[-1] ** (12/T) - 1
        mdd = calc_mdd(rets)
        sh = sharpe(rets, np.full(T, tb_te_np.mean()))
        cal = ann / abs(mdd) if mdd < 0 else -999
        mw = ws.mean() if isinstance(ws, np.ndarray) else ws
        print(f"  {label:<20} Ret={ann*100:>+7.2f}%  Sharpe={sh:>6.3f}  Calmar={cal:>6.3f}  MDD={mdd*100:>6.1f}%  W={mw:.2f}")

    print(f"\n  w distribution: min={w_te.min():.3f} p25={np.percentile(w_te,25):.3f} "
          f"med={np.median(w_te):.3f} p75={np.percentile(w_te,75):.3f} max={w_te.max():.3f}")
    print(f"  Final PP = {pp_hist[-1]:.3f}")

    # 분기별 상세 (3개월 간격)
    print(f"\n  {'Date':>12}  {'w':>6}  {'SP%':>7}  {'Port%':>7}  {'γ':>6}")
    for t in range(T):
        if t % 3 == 0:
            d = full.loc[te_i[t], "date"]
            print(f"  {d.date():>12}  {w_te[t]:>.3f}  {sp_next_te[t]*100:>+6.1f}%  "
                  f"{rets_mamba[t]*100:>+6.1f}%  {full.loc[te_i[t],'gamma']:>+5.2f}")
