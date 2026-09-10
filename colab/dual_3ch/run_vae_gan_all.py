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

# 조건 입력을 MAC-Flow 와 일치시킨다 (VG import 전에 설정해야 한다).
#   과거 52 주 5 채널(논문 Table 3) + 미래 tbill·metab + 원점 고정 sp_skew_13w.
#   예전에는 train_flow_seq.COND_COLS(6채널, metab 없이 변동성 2채널)를 그대로
#   썼다.  그러면 MAC-Flow 만 논문의 핵심 조건 변수(metab)를 받아 비교가
#   성립하지 않는다.  run_tuning_all.py / run_thin_compare.py 와 같은 설정이다.
os.environ.setdefault("VG_EXTRA_COLS", "sp_skew_13w")
os.environ.setdefault("VG_FUTURE_UNMASK", "metab_13w")

import analyze_pathshape_rawvol as PS                              # noqa: E402
import train_garch_flow as T                                       # noqa: E402
import train_vae_gan_baseline as VG                                 # noqa: E402

# ── raw-vol 모드 일치 (MAC-Flow 와 동일 파이프라인) ──────────────────────────────
# train_vae_gan_baseline 은 `from train_garch_flow import garch_preprocess_fold,
# forward_garch_rescale` 로 *import 시점에 이름을 복사*하므로, patch_rawvol() 이
# train_garch_flow.* 만 바꿔서는 baseline 에 전달되지 않는다(여전히 원본 GARCH 사용).
# → baseline 모듈(VG)의 import-bound 이름을 raw-vol 버전으로 직접 덮어쓴다.
#   이로써 VAE/GAN 도 MAC-Flow 와 동일하게 rolling-std 표준화 + origin-frozen σ 로 평가됨.
from rawvol_helpers import (patch_rawvol, rawstd_preprocess_fold,        # noqa: E402
                            forward_rawvol_rescale)
patch_rawvol()                                   # train_garch_flow.* (모듈-참조 사용처용)
VG.garch_preprocess_fold = rawstd_preprocess_fold   # ★ baseline import-bound 이름 교체
VG.forward_garch_rescale = forward_rawvol_rescale   # ★
print("[run_vae_gan_all] raw-vol 모드 적용 — VAE/GAN 도 MAC-Flow 와 동일 파이프라인(rolling-std + origin-frozen σ)")

FOLDS = ["F_gfc", "F_long_A", "F_long_B_origin", "F_long"]
RESULT_DIR = os.path.join(HERE, "result")
FOLDS_DIR = os.path.join(ROOT, "data", "folds_v33_vix_expanding")
SEED = 2026
N_SIM = 1000

# 튜닝 결과 (27 조합 × 4 폴드 val CRPS 최소).  run_tuning_all.py / run_thin_compare.py
# 와 같은 값이어야 한다.
VG_SPEC = {"vae": dict(lr=1e-4, dctx=128, hid=192),
           "gan": dict(lr=3e-4, dctx=192, hid=128)}


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
            sp_ = VG_SPEC[model]
            args = SimpleNamespace(model=model, fold=fold, folds_dir=FOLDS_DIR,
                                   out_dir=RESULT_DIR, n_sim=N_SIM, seed=SEED,
                                   lr=sp_["lr"], tag="")
            print(f"\n[{model}] fold={fold}  lr={sp_['lr']:g} "
                  f"d_ctx={sp_['dctx']} hid={sp_['hid']}")
            _saved = list(T.COND_COLS)
            _saved_cap = (VG.D_CTX, VG.HID)
            PS.set_cond_cols(PS.ENC_COLS)          # Table 3 의 5 채널
            VG.COND_COLS = list(T.COND_COLS); VG.N_CH = len(VG.COND_COLS)
            VG.SP_CH, VG.TBILL_CH = T.SP_CH, T.TBILL_CH
            VG.D_CTX, VG.HID = sp_["dctx"], sp_["hid"]
            try:
                VG.run_fold(model, fold, args, device)
            except Exception as e:
                print(f"  [FAIL] {model} {fold}: {e!r}")
            finally:
                PS.set_cond_cols(_saved)
                VG.D_CTX, VG.HID = _saved_cap

    print("\n[done] vae/gan baseline 완료 → {model}_baseline_{fold}_summary.json")


if __name__ == "__main__":
    main()
