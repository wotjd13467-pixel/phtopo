#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""40 — P_ab(T) 결과 -> 비선형 열홀 전류 j, 홀 전도도 κ_H, 횡온도차 ΔT⊥.

Luo 2023 식8:  j = (τ/(V ħ²)) ∇T × (Pᵀ∇T).  npz 의 P 는 이미 1/V 포함(nphe_vec norm).
부호는 위 식 그대로 (SM ε 배치 미대조 — 01_current 열린질문 2).
우리 셀 극성축 = y -> 허용 P_xz, P_zx. ∇T∥x 이면 j∥y (극성축), ∇T∥z 여도 j∥y.

두 모드:
  --tau 5          상수 τ[ps]. P 는 nphe_*.npz (전 성분, shift 평균±SE). shear 등 아무 npz 가능.
  --tau omega      모드별 τ(ω,T): spectral_P_nq60.npz(parent, xz·zx 주파수 분해) 의 bin 마다
                   τ(ω_bin,T)(39_tau_omega.py -> tau_parent.npz) 곱해 합 = Q_ab = Σ τ_n P_n.
                   tau_parent 의 κ 로 ΔT⊥ 도 냄.
출력 단위: j [W/m²], κ_H = |j|/|∇T| [W/mK], ΔT⊥ = j W/κ⊥ [K], θ = κ_H/κ⊥.
사용 예:
  python 40_kappa_nphe.py --tau 5 --grad 1e6 --dir x
  python 40_kappa_nphe.py --tau omega --grad 1e6 --dir x --width 10e-6
  python 40_kappa_nphe.py --tau 5 --P results/nphe_shear_nq60.npz --dir z"""
import os, argparse
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(PROJ, "results")
HBAR, KB, E_CHG = 1.054571817e-34, 1.380649e-23, 1.602176634e-19
EVKB2 = E_CHG * KB ** 2          # eV*kB^2 -> J^3/K^2
AX = {"x": 0, "y": 1, "z": 2}
VOIGT = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (0, 2): 4, (0, 1): 5}


def current(Q, g):
    """Q[T,3,3] in J^3/K^2 * s (τ 포함), g[3] K/m -> j[T,3] W/m²."""
    w = np.einsum("tdc,d->tc", Q, g)               # (Pᵀ∇T)_c = Σ_d P_dc g_d
    return np.cross(g[None, :], w) / HBAR ** 2


def tau_interp(ctr_spec, it_T, tp):
    """tau_parent bin 표 -> spectral bin 중심에서 τ [s]. 표 밖은 끝값 고정."""
    t = tp["tau_bin"][it_T]; ok = np.isfinite(t)
    return np.interp(ctr_spec, tp["ctr"][ok], t[ok]) * 1e-12


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau", default="omega", help="상수 τ [ps] 숫자 또는 'omega'")
    ap.add_argument("--P", default=os.path.join(R, "nphe_parent_nq60.npz"), help="상수 τ 용 P npz")
    ap.add_argument("--eta", type=float, default=0.08, help="THz, npz 의 etas 중 선택")
    ap.add_argument("--grad", type=float, default=1e6, help="|∇T| [K/m]  (1e6 = 1 K/μm)")
    ap.add_argument("--dir", default="x", help="∇T 방향: x, y, z 또는 'gx,gy,gz'")
    ap.add_argument("--width", type=float, default=10e-6, help="시료 폭 W [m], ΔT⊥ 용")
    ap.add_argument("--T", type=float, nargs="*", default=[30, 80, 150, 300])
    a = ap.parse_args()

    g = np.zeros(3)
    if a.dir in AX: g[AX[a.dir]] = 1.0
    else: g = np.array([float(s) for s in a.dir.split(",")])
    g = g / np.linalg.norm(g) * a.grad

    tp = np.load(os.path.join(R, "tau_parent.npz")) if os.path.exists(os.path.join(R, "tau_parent.npz")) else None
    if a.tau == "omega":
        if tp is None: raise SystemExit("results/tau_parent.npz 없음 -> 먼저 python 39_tau_omega.py")
        s = np.load(os.path.join(R, "spectral_P_nq60.npz"))
        TS, ie = s["TS"], int(np.argmin(np.abs(s["ETAS"] - a.eta)))
        ctr = 0.5 * (s["EDGES"][:-1] + s["EDGES"][1:])
        assert np.allclose(TS, tp["TS"]), "tau_parent 온도 != spectral 온도"
        Hs = s["Hs"][:, :, ie] * 1e-3 * EVKB2          # [shift, comp(xz,zx), T, bin] J^3/K^2
        Qs = np.zeros((Hs.shape[0], len(TS), 3, 3))
        for it in range(len(TS)):
            q = Hs[:, :, it] @ tau_interp(ctr, it, tp)  # Σ_bin τ P  [shift, comp]
            Qs[:, it, 0, 2], Qs[:, it, 2, 0] = q[:, 0], q[:, 1]
        P1 = np.zeros((len(TS), 3, 3)); Pm = Hs.sum(-1).mean(0)   # τ=1s 의 P [T,3,3], τ_eff 용
        P1[:, 0, 2], P1[:, 2, 0] = Pm[0], Pm[1]
        j1 = current(P1, g)
        src = f"spectral_P_nq60 (parent, xz/zx만), τ(ω,T) from tau_parent mesh{int(tp['mesh'])}"
    else:
        d = np.load(a.P)
        TS, ie = d["TS"], int(np.argmin(np.abs(d["etas"] - a.eta)))
        Qs = d["P"][:, ie] * EVKB2 * float(a.tau) * 1e-12   # [shift, T, 3, 3]
        src = f"{os.path.basename(a.P)}, 상수 τ={float(a.tau)} ps"
    Q, Qse = Qs.mean(0), Qs.std(0, ddof=1) / np.sqrt(len(Qs)) if len(Qs) > 1 else 0 * Qs[0]
    j = current(Q, g)
    jse = np.abs(current(Q + Qse, g) - j)               # 대략적 오차(성분별 SE 동시 이동)

    gh = g / a.grad
    print(f"[{src}]  η={a.eta} THz  ∇T={a.grad:.1e} K/m along {np.round(gh, 3)}  W={a.width*1e6:.1f} μm")
    print(f"{'T':>5} {'j_x':>10} {'j_y':>10} {'j_z':>10} {'κ_H[W/mK]':>11} {'±':>8} {'τ_eff[ps]':>9}"
          f" {'κ⊥[W/mK]':>9} {'θ':>9} {'ΔT⊥[K]':>9}")
    for T in a.T:
        it = int(np.argmin(np.abs(TS - T)))
        jj = j[it]; jn = np.linalg.norm(jj); kH = jn / a.grad
        teff = ""
        if a.tau == "omega":
            n1 = np.linalg.norm(j1[it])              # 같은 방향 상수τ 전류 대비 -> 유효 τ
            teff = f"{jn / n1 * 1e12:9.2f}" if n1 else ""
        kp = th = dT = ""
        if tp is not None and jn > 0:
            e = jj / jn; kt = tp["kappa"][int(np.argmin(np.abs(tp["TS"] - T)))]
            K = np.zeros((3, 3))
            for (m, n), v in VOIGT.items(): K[m, n] = K[n, m] = kt[v]
            kperp = e @ K @ e
            kp, th, dT = f"{kperp:9.2f}", f"{kH/kperp:9.2e}", f"{jn*a.width/kperp:9.2e}"
        print(f"{TS[it]:>5.0f} {jj[0]:>10.3e} {jj[1]:>10.3e} {jj[2]:>10.3e} {kH:>11.3e} "
              f"{np.linalg.norm(jse[it])/a.grad:>8.1e} {teff:>9} {kp:>9} {th:>9} {dT:>9}")
    print("주의: j ∝ (∇T)² -> κ_H ∝ |∇T|. 부호는 Luo 식8 표기 기준(SM 미대조). "
          "κ⊥·θ·ΔT⊥ 는 parent tau_parent 의 κ 사용(shear 에도 같은 κ).")
    if a.tau == "omega":
        print("주의: τ(ω) 는 bin 평균(밴드·q 차이 무시). 표 밖(최저 bin 아래 음향)은 끝값 고정 -> 30K 이하 신뢰 낮음.")


if __name__ == "__main__":
    main()
