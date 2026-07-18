# MAC-Flow — Monetary Debasement and Equity Tail-Risk

Code for the paper:

> **Monetary Debasement and Equity Tail-Risk: Counterfactual Scenario Generation via Global/Local Conditioning on Macroeconomic Paths**

MAC-Flow is an autoregressive (AR) **Conditional Normalizing Flow** (RQ-NSF, Durkan et al. 2019; skew Student-t base, Hansen 1994) that generates 13-week-ahead S&P 500 return scenarios conditioned on **counterfactual paths** of the 3-month T-bill rate and an excess-liquidity indicator (M2 − INDPRO − CPI). The future path is injected through a **dual mechanism**: a Future Context Encoder that summarizes the whole path and broadcasts it to every generation step (*global*), and a per-step encoder that injects the path point by point (*local*). Tail risk is measured with the **intra-horizon loss** (worst cumulative loss within the holding period), which is order-sensitive and therefore path-dependent.

---

## Repository layout

```
colab/dual_3ch/     model, runners, baselines, and all §4 analysis scripts
data/               data build script (FRED + Yahoo Finance → weekly folds)
notebooks/          Colab notebooks to reproduce the paper (see below)
result/             caches produced by the analysis scripts (committed selectively)
```

Legacy research code from earlier project phases is preserved in the branch **`archive-20260707`** and is not part of the paper.

## Reproducing the paper (`notebooks/`)

To reproduce all paper tables, run `notebooks/01_reproduce_tables.ipynb` (aggregates from the committed caches, no GPU needed). For a full retraining from scratch, run `03_train_macflow.ipynb`.

| Notebook | What it does | Needs |
|---|---|---|
| `01_reproduce_tables.ipynb` | Re-aggregates every §4 table from the committed caches — no inference | caches |
| `02_inference_from_checkpoints.ipynb` | Re-runs inference from trained checkpoints and rebuilds the caches | checkpoints (GPU) |
| `03_train_macflow.ipynb` | Full retraining of the main model and ablations (4 folds × 5 seeds) | GPU |
| `04_baselines.ipynb` | Trains the Conditional VAE / GAN and GARCH baselines | GPU |
| `05_counterfactual_paths.ipynb` | Runs the §4.2 counterfactual path experiments | checkpoints (GPU) |

Seeds are fixed (`2026–2030`) and sampling uses paired seeds, so re-runs reproduce the reported numbers.

**Checkpoints and caches.** Model checkpoints (`*.pt`) are too large for the repository and live on the training drive; the JSON caches that the tables aggregate from are committed under `result/` (add new ones with `git add -f result/<cache>`). Notebook `01` therefore runs anywhere; notebooks `02`/`05` require the checkpoints produced by `03`.

## Data

Raw series are public: FRED (`TB3MS`, `M2SL`, `INDPRO`, `CPIAUCSL`, `DCOILWTICO`), the Philadelphia Fed ADS index, and Yahoo Finance (S&P 500). We do not redistribute the raw data; `data/extend_to_1971.py` downloads and rebuilds the weekly dataset and the four walk-forward folds, with publication-lag alignment as described in §3.1 of the paper.

## Quick start (Colab)

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab Notebooks/homeostatic-market
!git pull
!pip -q install nflows arch yfinance pandas-datareader
# reproduce all §4 tables from caches:
!PS_BASE=fpath_novol FPATH_DIM=2 python colab/dual_3ch/agg_section4.py
```
