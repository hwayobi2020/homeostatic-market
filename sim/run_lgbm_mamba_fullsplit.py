"""LGBM + Mamba 비교 — 새 single split (1991-2015 train / 2016-2025 test, gap 6M)
Target: 3M forward return >= +10% (Phase 11 LGBM 과 동일)
Features: v28 36 monthly (LGBM) + 추가 63일 daily (Mamba)

기존 Phase 11 (W1/W2/W3) → 단일 25년 train + 9.5년 test 로 변경
"""
import sys, torch, torch.nn as nn
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
try: sys.stdout.reconfigure(encoding="utf-8")
except: pass
from mambapy.mamba import Mamba, MambaConfig
import yfinance as yf
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

TRAIN_START = "1991-01-01"
TRAIN_END   = "2015-12-31"
TEST_START  = "2016-07-01"   # 6M gap
TEST_END    = "2025-12-31"

# ── 1. Monthly data (v28) ──
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
full["label_up"] = (full["y3"] >= 0.10).astype(float)

EXCLUDE_MF = {"date", "sp_next_return", "y3", "label_up"}
MF = [c for c in full.columns if c not in EXCLUDE_MF]
print(f"Monthly features: {len(MF)}")

tr_mask = (full["date"] >= TRAIN_START) & (full["date"] <= TRAIN_END)
te_mask = (full["date"] >= TEST_START) & (full["date"] <= TEST_END)
tr_i_all = np.where(tr_mask & ~np.isnan(full["y3"]))[0]
te_i_all = np.where(te_mask & ~np.isnan(full["y3"]))[0]

print(f"\nSplit:")
print(f"  Train: {full.loc[tr_i_all[0],'date'].date()} ~ {full.loc[tr_i_all[-1],'date'].date()}, n={len(tr_i_all)}")
print(f"  Test:  {full.loc[te_i_all[0],'date'].date()} ~ {full.loc[te_i_all[-1],'date'].date()}, n={len(te_i_all)}")
print(f"  Train events (3M>=+10%): {int(full.loc[tr_i_all,'label_up'].sum())} ({full.loc[tr_i_all,'label_up'].mean()*100:.1f}%)")
print(f"  Test  events (3M>=+10%): {int(full.loc[te_i_all,'label_up'].sum())} ({full.loc[te_i_all,'label_up'].mean()*100:.1f}%)")

# ── 2. LGBM ──
print(f"\n{'='*80}\n[LGBM] v28 {len(MF)} features, 3M >= +10%\n{'='*80}")

X_tr = full.loc[tr_i_all, MF].values.astype(np.float32)
y_tr = full.loc[tr_i_all, "label_up"].values
X_te = full.loc[te_i_all, MF].values.astype(np.float32)
y_te = full.loc[te_i_all, "label_up"].values
y3_te = full.loc[te_i_all, "y3"].values

scale_pos = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
lgbm_aucs = []
for seed in [42, 123, 777, 0, 99]:
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_child_samples=10, scale_pos_weight=scale_pos,
        random_state=seed, verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    p_lgbm = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p_lgbm)
    lgbm_aucs.append(auc)
    print(f"  seed={seed}: AUC={auc:.4f}")
print(f"  Mean AUC: {np.mean(lgbm_aucs):.4f} ± {np.std(lgbm_aucs):.4f}")

# 마지막 seed 로 detail
imp = pd.Series(clf.feature_importances_, index=MF).sort_values(ascending=False)
print(f"\n  Top 8 feature importance:")
for f, v in imp.head(8).items():
    print(f"    {f}: {v}")

# ── 3. Mamba ──
print(f"\n{'='*80}\n[Mamba] v28 {len(MF)} monthly + 63d daily, 3M >= +10%\n{'='*80}")

# Daily download
sp_daily = yf.download("^GSPC", start="1989-01-01", end="2026-04-01", interval="1d", progress=False)
sp_close = sp_daily["Close"]
if isinstance(sp_close, pd.DataFrame): sp_close = sp_close.iloc[:,0]
sp_ret_daily = sp_close.pct_change().dropna()

vix_daily = yf.download("^VIX", start="1989-01-01", end="2026-04-01", interval="1d", progress=False)
vix_d = vix_daily["Close"]
if isinstance(vix_d, pd.DataFrame): vix_d = vix_d.iloc[:,0]
print(f"  Daily: {sp_ret_daily.index[0].date()} ~ {sp_ret_daily.index[-1].date()}")

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
print(f"  Sequences: {len(daily_sequences)}/{N}")

class MambaUp(nn.Module):
    def __init__(self, seq_in=2, d_model=16, n_layers=2, mf_dim=36, hid=32):
        super().__init__()
        self.proj = nn.Linear(seq_in, d_model)
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))
        self.head = nn.Sequential(
            nn.Linear(d_model + mf_dim, hid), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hid, 1))
    def forward(self, seq, mf):
        x = self.mamba(self.proj(seq))
        return self.head(torch.cat([x[:,-1,:], mf], 1)).squeeze(1)

# Filter indices that have sequences
tr_i = [i for i in tr_i_all if i in daily_sequences]
te_i = [i for i in te_i_all if i in daily_sequences]
print(f"  After daily sequence filter: train={len(tr_i)}, test={len(te_i)}")

Xs_tr = torch.tensor(np.array([daily_sequences[i] for i in tr_i]), dtype=torch.float32)
mf_tr_raw = full.loc[tr_i, MF].values.astype(np.float32)
y_tr_t = torch.tensor(full.loc[tr_i, "label_up"].values, dtype=torch.float32)

Xs_te = torch.tensor(np.array([daily_sequences[i] for i in te_i]), dtype=torch.float32)
mf_te_raw = full.loc[te_i, MF].values.astype(np.float32)
y_te_arr = full.loc[te_i, "label_up"].values
y3_te_arr = full.loc[te_i, "y3"].values

mu, sd = mf_tr_raw.mean(0), mf_tr_raw.std(0) + 1e-8
Xm_tr = torch.tensor((mf_tr_raw - mu) / sd, dtype=torch.float32)
Xm_te = torch.tensor((mf_te_raw - mu) / sd, dtype=torch.float32)

mamba_aucs = []
for seed in [42, 123, 777]:
    torch.manual_seed(seed); np.random.seed(seed)
    model = MambaUp(seq_in=2, d_model=16, n_layers=2, mf_dim=len(MF), hid=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor([(len(y_tr_t) - y_tr_t.sum()) / max(y_tr_t.sum(), 1)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    model.train()
    best_loss = 1e9; best_st = None; pat = 0
    for ep in range(200):
        opt.zero_grad()
        loss = crit(model(Xs_tr, Xm_tr), y_tr_t)
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

    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        p_mamba = torch.sigmoid(model(Xs_te, Xm_te)).numpy()
    auc = roc_auc_score(y_te_arr, p_mamba)
    mamba_aucs.append(auc)
    print(f"  seed={seed}: epochs={ep+1}, best_loss={best_loss:.4f}, AUC={auc:.4f}")

print(f"  Mean AUC: {np.mean(mamba_aucs):.4f} ± {np.std(mamba_aucs):.4f}")

# ── 4. Summary ──
print(f"\n{'='*80}\nSummary (Train 1991-2015, Test 2016-2025, 3M >= +10%)\n{'='*80}")
print(f"  LGBM  AUC = {np.mean(lgbm_aucs):.4f} ± {np.std(lgbm_aucs):.4f}  (5 seeds)")
print(f"  Mamba AUC = {np.mean(mamba_aucs):.4f} ± {np.std(mamba_aucs):.4f}  (3 seeds)")
print(f"  Test events: {int(y_te.sum())}/{len(y_te)} ({y_te.mean()*100:.1f}%)")

# Save predictions
out = pd.DataFrame({
    "date": full.loc[te_i_all, "date"].values,
    "y3": y3_te,
    "label_up": y_te,
    "p_lgbm_last_seed": p_lgbm,
})
out.to_csv("result/lgbm_fullsplit_2016_2025.csv", index=False)
print(f"\nsaved: result/lgbm_fullsplit_2016_2025.csv")
