"""§4.1.2 baseline — CondVAE / CondGAN 4 fold 학습+eval (train_vae_gan_baseline.run_fold 루프).

summary 가 seed-태그 안 되므로(=`{model}_baseline_{fold}_summary.json`) baseline 1-seed(2026).
재진입: summary 있으면 skip.

Usage (Colab):
    %cd '/content/drive/MyDrive/Colab Notebooks/homeostatic-market'
    !git pull
    !python colab/dual_3ch/run_vae_gan_all.py
"""
import os
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import train_vae_gan_baseline as VG                                 # noqa: E402

FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
SEED = 2026
N_SIM = 1000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[vae/gan baseline] {len(FOLDS)} fold × (vae,gan)  device={device}  seed={SEED}")
    for model in ("vae", "gan"):
        for fold in FOLDS:
            sp = os.path.join(RESULT_DIR, f"{model}_baseline_{fold}_summary.json")
            if os.path.exists(sp):
                print(f"[skip] {os.path.basename(sp)}"); continue
            if not all(os.path.exists(os.path.join(FOLDS_DIR, f"{fold}_{s}.csv"))
                       for s in ("train", "val", "test")):
                print(f"[skip {fold}] fold CSV 없음"); continue
            args = SimpleNamespace(model=model, fold=fold, folds_dir=FOLDS_DIR,
                                   out_dir=RESULT_DIR, n_sim=N_SIM, seed=SEED)
            print(f"\n[{model}] fold={fold}")
            try:
                VG.run_fold(model, fold, args, device)
            except Exception as e:
                print(f"  [FAIL] {model} {fold}: {e!r}")

    print("\n[done] vae/gan baseline 완료 → {model}_baseline_{fold}_summary.json")


if __name__ == "__main__":
    main()
