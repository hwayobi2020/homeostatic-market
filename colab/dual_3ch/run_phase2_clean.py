"""Phase 2 (Clean Run) -- Multi-seed validation of Phase 1 per-encoder Top 1.

[Master design -- Gemini Clean Run pipeline]

  1) Scan Phase 1 summary.json files (tag prefix 'sweep_..._s2026').
  2) For each encoder in {lstm, mlp, transformer, mamba}, pick the spec
     (Flow head included) with the smallest 3-fold average val NLL
     (best_val_nll_per_week, averaged over F_long_A / F_long_B_origin / F_long).
  3) Re-run those 4 best specs with seeds [2026, 2027, 2028, 2029, 2030]
     x folds {F_long_A, F_long_B_origin, F_long} = 60 runs.

  Re-entrant : (spec, fold, seed) triples whose summary.json already exists
  are skipped.

  Selection key : val NLL only (test metrics never used to pick specs).

Usage (Colab) :
    !python colab/dual_3ch/run_phase2_clean.py
"""
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_mamba_flow_ar import main_worker  # noqa: E402

PHASE1_SEED = 2026
SEEDS = [2026, 2027, 2028, 2029, 2030]
FOLDS = ["F_long_A", "F_long_B_origin", "F_long"]
ENCODERS = ["lstm", "mlp", "transformer", "mamba"]
RESULT_DIR = os.path.join(HERE, "result")


# ---------- Phase 1 결과 scan -> encoder 별 Top 1 ----------
fold_re = "|".join(re.escape(f) for f in FOLDS)
phase1_pat = re.compile(
    rf"^mamba_flow_ar_(sweep_(?P<enc>{'|'.join(ENCODERS)})_.+?_s{PHASE1_SEED})"
    rf"_(?P<fold>{fold_re})_summary\.json$"
)

files = sorted(glob.glob(os.path.join(
    RESULT_DIR, f"mamba_flow_ar_sweep_*_s{PHASE1_SEED}_*_summary.json")))
print(f"[Phase 2] scanning Phase 1 results : {len(files)} files matching "
      f"'mamba_flow_ar_sweep_*_s{PHASE1_SEED}_*_summary.json'")

# key = (enc, base_tag) ; value = {fold: best_val_nll_per_week}
fold_vals = defaultdict(dict)
for f in files:
    m = phase1_pat.match(os.path.basename(f))
    if not m:
        continue
    enc = m.group("enc")
    base_tag = m.group(1)
    fold = m.group("fold")
    try:
        v = json.load(open(f)).get("best_val_nll_per_week")
    except (json.JSONDecodeError, OSError):
        continue
    if v is not None:
        fold_vals[(enc, base_tag)][fold] = v

# 3-fold complete 만 후보, encoder 별 argmin
best_per_enc = {}
for (enc, base_tag), fv in fold_vals.items():
    if len(fv) < len(FOLDS):
        continue
    avg = sum(fv.values()) / len(fv)
    if enc not in best_per_enc or avg < best_per_enc[enc][1]:
        best_per_enc[enc] = (base_tag, avg, fv)

print(f"\n[Phase 2] Top 1 per encoder (3-fold avg val NLL, selection key) :")
for enc in ENCODERS:
    if enc in best_per_enc:
        base_tag, avg, fv = best_per_enc[enc]
        per = "  ".join(f"{k.split('_')[-1]}={v:+.4f}" for k, v in fv.items())
        print(f"  {enc:12s}  avg={avg:+.4f}  tag={base_tag}    | {per}")
    else:
        print(f"  {enc:12s}  [no spec with complete 3-fold set found]")

missing = [e for e in ENCODERS if e not in best_per_enc]
if missing:
    sys.exit(f"\n[FATAL] missing encoders : {missing}.  "
             f"Phase 1 (run_phase1_clean.py) 를 먼저 끝까지 돌리세요.")


# ---------- tag -> spec dict 복원 ----------
def parse_tag_to_spec(base_tag):
    """Recover the spec dict from a Phase 1 base_tag (with _s2026 suffix)."""
    # strip 'sweep_' prefix and trailing '_s{seed}'
    body = base_tag[len("sweep_"):]
    body = re.sub(rf"_s{PHASE1_SEED}$", "", body)
    # tail '_flow{Heavy|Light}'
    head, _, flow = body.rpartition("_flow")
    if flow not in ("Heavy", "Light"):
        raise ValueError(f"flow parse fail : {base_tag!r}")
    # tail '_dr{R}'
    head, _, dr_str = head.rpartition("_dr")
    dropout = float(dr_str)
    # remaining : '{enc}_{capspec}'
    enc, _, cap = head.partition("_")
    spec = dict(encoder_type=enc, dropout=dropout)
    if flow == "Light":
        spec.update(n_flow_layers=2, n_flow_hidden=32, weight_decay=0.1)
    else:
        spec.update(n_flow_layers=6, n_flow_hidden=64, weight_decay=0.5)
    toks = cap.split("_")
    if enc == "lstm":
        spec["d_model"] = int(toks[0][2:])         # hd{H}
        spec["n_mamba_layers"] = int(toks[1][2:])  # nl{N}
    elif enc == "mlp":
        spec["d_model"] = int(toks[0][2:])         # hd{H}
        spec["mlp_num_layers"] = int(toks[1][2:])  # nl{N}
    elif enc == "transformer":
        spec["d_model"] = int(toks[0][2:])              # dm{D}
        spec["transformer_n_heads"] = int(toks[1][2:])  # nh{N}
        spec["n_mamba_layers"] = int(toks[2][2:])       # nl{L}
    elif enc == "mamba":
        spec["d_model"] = int(toks[0][2:])         # dm{D}
        spec["n_mamba_layers"] = int(toks[1][2:])  # nl{N}
    else:
        raise ValueError(f"unknown encoder : {enc!r}")
    return spec


# ---------- 4 best x 5 seed x 3 fold = 60 runs ----------
runs = []
for enc in ENCODERS:
    base_tag = best_per_enc[enc][0]
    tag_noseed = re.sub(rf"_s{PHASE1_SEED}$", "", base_tag)
    base_spec = parse_tag_to_spec(base_tag)
    for seed in SEEDS:
        for fold in FOLDS:
            runs.append((enc, tag_noseed, base_spec, seed, fold))

N_TOTAL = len(runs)
print(f"\n[Phase 2] {N_TOTAL} runs ({len(ENCODERS)} encoders x "
      f"{len(SEEDS)} seeds x {len(FOLDS)} folds)")

done = 0
skipped = 0
failed = 0
t0 = time.time()

# fold-outer ordering for csv cache reuse: reorder runs accordingly.
runs_sorted = sorted(runs, key=lambda r: (FOLDS.index(r[4]), ENCODERS.index(r[0]), r[3]))

for enc, tag_noseed, base_spec, seed, fold in runs_sorted:
    new_tag = f"{tag_noseed}_s{seed}"
    summary_path = os.path.join(
        RESULT_DIR, f"mamba_flow_ar_{new_tag}_{fold}_summary.json")
    idx = done + skipped + failed + 1
    if os.path.exists(summary_path):
        print(f"[skip {idx:3d}/{N_TOTAL}] {os.path.basename(summary_path)}")
        skipped += 1
        continue
    spec = dict(base_spec)
    spec.update(fold=fold, tag=new_tag, seed=seed)
    print(f"\n[run  {idx:3d}/{N_TOTAL}] {new_tag}  fold={fold}")
    try:
        main_worker(spec)
        done += 1
    except Exception as e:
        print(f"[FAIL {idx:3d}/{N_TOTAL}] {new_tag} {fold}: {e!r}")
        failed += 1

dt = time.time() - t0
print(f"\n[Phase 2 done] done={done} skipped={skipped} failed={failed}  "
      f"elapsed={dt / 60:.1f} min")
