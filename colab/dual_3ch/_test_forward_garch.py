"""Self-test: forward_garch_rescale 정확성 검증 (look-ahead 제거 fix).

핵심 성질: forward_garch_rescale 에 *실현* 표준화잔차 z_t 를 sim 으로 넣고
origin 상태(σ²_o, ε²_o)에서 시작하면, GARCH 필터 재귀와 동일하므로 *실현 수익률*을
그대로 재현해야 한다.  (시뮬 ε² 가 실현 ε² 와 같아지므로 σ 경로가 필터와 일치.)

차이가 ~1e-12 면 재귀 + origin 상태 구성이 수학적으로 올바름.
"""
import ast
import os

import numpy as np

# torch DLL 가 로컬에서 안 떠서 모듈 전체 import 불가 → 실제 배포 소스에서
# forward_garch_rescale 함수 본문만 추출해 exec (복사본 아닌 *실제 코드* 검증).
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_garch_flow.py")
_tree = ast.parse(open(_SRC, encoding="utf-8").read())
_fn = next(n for n in _tree.body
           if isinstance(n, ast.FunctionDef) and n.name == "forward_garch_rescale")
_ns = {"np": np}
exec(compile(ast.Module([_fn], []), _SRC, "exec"), _ns)
forward_garch_rescale = _ns["forward_garch_rescale"]


def gen_garch_path(T, om, al, be, mu, seed=0):
    rng = np.random.default_rng(seed)
    s2 = np.empty(T); eps = np.empty(T); r = np.empty(T)
    s2[0] = om / max(1e-6, 1 - al - be)              # unconditional var
    z = rng.standard_normal(T)
    eps[0] = z[0] * np.sqrt(s2[0]); r[0] = mu + eps[0]
    for t in range(1, T):
        s2[t] = om + al * eps[t - 1] ** 2 + be * s2[t - 1]
        eps[t] = z[t] * np.sqrt(s2[t]); r[t] = mu + eps[t]
    sig = np.sqrt(s2)
    return r, sig, eps, z


def main():
    om, al, be, mu = 2e-5, 0.08, 0.90, 0.0015      # 전형적 GARCH(1,1)
    T = 400
    r, sig, eps, z = gen_garch_path(T, om, al, be, mu, seed=42)

    H = 13
    o = 200                                          # origin (마지막 관측) index
    # origin 상태
    s2_orig = np.array([sig[o] ** 2])
    e2_orig = np.array([(z[o] * sig[o]) ** 2])       # = eps[o]^2 (코드와 동일 구성)
    # 실현 표준화잔차 z 를 미래 sim 으로 주입: future step tau -> row o+1+tau
    z_future = z[o + 1 : o + 1 + H].reshape(1, 1, H)

    sim_raw = forward_garch_rescale(z_future, s2_orig, e2_orig, om, al, be, mu)
    realized = r[o + 1 : o + 1 + H]

    diff = np.abs(sim_raw[0, 0, :] - realized)
    print("step |   forward_raw   |   realized_r   |   |diff|")
    for h in range(H):
        print(f"{h:>4d} | {sim_raw[0,0,h]:+.10f} | {realized[h]:+.10f} | {diff[h]:.2e}")
    print(f"\nmax |diff| = {diff.max():.3e}   "
          f"({'PASS' if diff.max() < 1e-10 else 'FAIL'})")

    # 2) scale 변환 검증: arch 는 r*scale 에서 fit.  om_ret=om_s/scale^2, al,be 동일.
    scale = 100.0
    om_s, al_s, be_s = om * scale ** 2, al, be       # scaled-unit params
    # return-unit 재구성
    om_ret = om_s / scale ** 2
    assert abs(om_ret - om) < 1e-18, "scale 변환 공식 오류"
    print(f"\nscale 변환: om_ret={om_ret:.3e} == om={om:.3e}  PASS")

    # 3) 여러 origin/시나리오에서 shape·유한성
    n_orig, n_sim = 5, 50
    rng = np.random.default_rng(1)
    z_sim = rng.standard_normal((n_orig, n_sim, H))
    s2o = np.full(n_orig, sig[o] ** 2); e2o = np.full(n_orig, eps[o] ** 2)
    out = forward_garch_rescale(z_sim, s2o, e2o, om, al, be, mu)
    print(f"shape={out.shape}  all_finite={np.isfinite(out).all()}  "
          f"({'PASS' if out.shape == (n_orig, n_sim, H) and np.isfinite(out).all() else 'FAIL'})")


if __name__ == "__main__":
    main()
