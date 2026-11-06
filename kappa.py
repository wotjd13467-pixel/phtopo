"""포논 열홀. 기본 = 비선형 (NPHE) κ_H,  --linear = 자기장 선형 κ_xy(B).

기본 (NPHE):  j = (τ/(V ħ²)) ∇T × (Pᵀ∇T)  (Luo 2023 식8), P 는 Cartesian 3×3 전 성분 (공간군·셀 방향 무관).
    pt kappa --P nphe_example_nq100_sc222.npz --tau 5 --direction x -T 30,80,150,300
    --tau omega: τ(ω,T) — --spec (pt spectral 의 npz) 의 bin 마다 τ(ω_bin,T) (--tau-file, pt tau 의 npz) 를 곱해 Q = Σ τ P.
                 옛 2성분 spectral 파일은 그 comps 자리에만 넣음.
    κ⊥·θ·ΔT⊥ 는 --tau-file 이 있으면 그 κ 로.
    본체 phtopo/kappa_nphe.py = C:\\Users\\techw\\Desktop\\cjs\\1_Projects\\NPHE_HfO2\\nphe\\scripts\\40_kappa_nphe.py 사본 (수정 없음).
    본체 main 은 xz/zx 고정·파일 이름 고정이라 계산·표는 여기서 하고, current / tau_interp / VOIGT 는 본체 것.
--linear (자기장 선형 열홀, Qin–Zhou–Shi / Sun–Gao–Wang 6N + Born 전하 Lorentz 결합, B 홀수 부분):
    pt kappa --linear --B 10 --axis z --nq 16 -T 5,10,20,50,100,200,300
    --verify 로 내부 검산 8항목 먼저 (새 구조).
    본체 phtopo/aphe.py = C:\\Users\\techw\\Desktop\\cjs\\4_Archives\\HfO2_shear_Weyl_PHE\\aphe.py 사본 (수정 없음)
"""
import os
import sys

import numpy as np

from .core import load

NAME, HELP = "kappa", "열홀: 기본 비선형 NPHE κ_H (nlhall/spectral npz 에서), --linear 는 자기장 선형 κ_xy(B)"
AX = {"x": 0, "y": 1, "z": 2}


def t_list(s):
    """'30,80,300' 또는 '10:300:10' -> [T...]."""
    if ":" in s:
        lo, hi, dt = (float(x) for x in s.split(":"))
        return list(np.arange(lo, hi + 1e-9, dt))
    return [float(x) for x in s.split(",")]


def add_args(p):
    p.add_argument("--linear", action="store_true", help="자기장 선형 κ_xy(B) (기본은 비선형 NPHE)")
    p.add_argument("-T", dest="T", type=t_list, default=None,
                   help="온도 '30,80,150,300' 또는 '10:300:10'. 기본: NPHE 30,80,150,300 / linear 5..300")
    p.add_argument("--eta", type=float, default=None, help="Kubo 정규화 η (THz). 기본: NPHE 0.08 (npz 안에서 선택) / linear 0")
    g = p.add_argument_group("NPHE (기본)")
    g.add_argument("--P", help="pt nlhall 의 npz (상수 τ 용). 기본 <data>/nphe_parent_nq60.npz")
    g.add_argument("--tau", default="5", help="상수 τ [ps] 또는 'omega'. 기본 5")
    g.add_argument("--spec", help="--tau omega 용 pt spectral npz. 기본 <data>/spectral_P_nq60.npz")
    g.add_argument("--tau-file", help="pt tau 의 npz (τ(ω,T), κ). 기본 <data>/tau_parent.npz")
    g.add_argument("--grad", type=float, default=1e6, help="|∇T| [K/m] (1e6 = 1 K/μm)")
    g.add_argument("--direction", default="x", help="∇T 방향 x, y, z 또는 'gx,gy,gz' (Cartesian, POSCAR 축)")
    g.add_argument("--width", type=float, default=10e-6, help="시료 폭 W [m] (ΔT⊥ 용)")
    g.add_argument("--data", default=".", help="--P/--spec/--tau-file 기본 파일이 있는 폴더. 기본 현재 폴더")
    g = p.add_argument_group("--linear")
    g.add_argument("--B", type=float, default=10.0, help="자기장 [T]. 기본 10")
    g.add_argument("--axis", default="z", help="z|xy → κ_xy, x|yz → κ_yz, y|zx → κ_zx, xz. 기본 z")
    g.add_argument("--nq", type=int, default=16, help="격자 nq³. 기본 16")
    g.add_argument("--nqz", type=int, default=None, help="k3 격자 (기본 nq). 1 은 진짜 2D 일 때만")
    g.add_argument("--bands", action="store_true", help="밴드별 κ_n(T)")
    g.add_argument("--verify", action="store_true", help="내부 검산 (새 구조는 먼저)")
    g.add_argument("--no-nac", action="store_true")
    g.add_argument("--save", help="결과 json")
    g.add_argument("--plot", help="그림 png")


def _call(main, prog, argv):
    old = sys.argv
    sys.argv = [prog] + [str(x) for x in argv]
    try:
        main()
    finally:
        sys.argv = old


def spec9(s):
    """spectral npz -> [shift, η, T, bin, 3, 3] meV·kB². 새 파일 = H9, 옛 파일 = Hs 두 성분을 comps 자리에."""
    if "H9" in s:
        return s["H9"]
    Hs = s["Hs"]  # [shift, comp, η, T, bin]
    H = np.zeros(Hs.shape[:1] + Hs.shape[2:] + (3, 3))
    for ic, c in enumerate(s["comps"]):
        H[..., AX[str(c)[0]], AX[str(c)[1]]] = Hs[:, ic]
    return H


def nphe(a):
    from . import kappa_nphe as kn

    T = a.T or [30, 80, 150, 300]
    eta = 0.08 if a.eta is None else a.eta
    g = np.zeros(3)
    if a.direction in AX:
        g[AX[a.direction]] = 1.0
    else:
        g = np.array([float(x) for x in a.direction.split(",")])
    g = g / np.linalg.norm(g) * a.grad
    tfile = a.tau_file or os.path.join(a.data, "tau_parent.npz")
    tp = np.load(tfile) if os.path.exists(tfile) else None
    if a.tau == "omega":
        if tp is None:
            raise SystemExit(f"{tfile} 없음 -> 먼저 pt tau (또는 --tau-file)")
        sfile = a.spec or os.path.join(a.data, "spectral_P_nq60.npz")
        s = np.load(sfile)
        TS, ie = s["TS"], int(np.argmin(np.abs(s["ETAS"] - eta)))
        if not np.allclose(TS, tp["TS"]):
            raise SystemExit(f"{tfile} 온도 != {sfile} 온도 -> pt tau --spec {sfile}")
        ctr = 0.5 * (s["EDGES"][:-1] + s["EDGES"][1:])
        H = spec9(s)[:, ie] * 1e-3 * kn.EVKB2  # [shift, T, bin, 3, 3] J^3/K^2
        tau = np.array([kn.tau_interp(ctr, it, tp) for it in range(len(TS))])  # [T, bin] s
        Qs = np.einsum("stbij,tb->stij", H, tau)  # Σ_bin τ P
        j1 = kn.current(H.sum(2).mean(0), g)  # τ=1s 전류 → τ_eff
        src = f"{os.path.basename(sfile)} (3×3), τ(ω,T) from {os.path.basename(tfile)} mesh{int(tp['mesh'])}"
    else:
        pfile = a.P or os.path.join(a.data, "nphe_parent_nq60.npz")
        d = np.load(pfile)
        TS, ie = d["TS"], int(np.argmin(np.abs(d["etas"] - eta)))
        Qs = d["P"][:, ie] * kn.EVKB2 * float(a.tau) * 1e-12  # [shift, T, 3, 3]
        src = f"{os.path.basename(pfile)}, 상수 τ={float(a.tau)} ps"
    Q = Qs.mean(0)
    Qse = Qs.std(0, ddof=1) / np.sqrt(len(Qs)) if len(Qs) > 1 else 0 * Q
    j = kn.current(Q, g)
    jse = np.abs(kn.current(Q + Qse, g) - j)  # 대략적 오차 (성분별 SE 동시 이동)

    print(f"[{src}]  η={eta} THz  ∇T={a.grad:.1e} K/m along {np.round(g / a.grad, 3)}  W={a.width * 1e6:.1f} μm")
    print(f"{'T':>5} {'j_x':>10} {'j_y':>10} {'j_z':>10} {'κ_H[W/mK]':>11} {'±':>8} {'τ_eff[ps]':>9}"
          f" {'κ⊥[W/mK]':>9} {'θ':>9} {'ΔT⊥[K]':>9}")
    for t in T:
        it = int(np.argmin(np.abs(TS - t)))
        jj = j[it]
        jn = np.linalg.norm(jj)
        kH = jn / a.grad
        teff = kp = th = dT = ""
        if a.tau == "omega":
            n1 = np.linalg.norm(j1[it])
            teff = f"{jn / n1 * 1e12:9.2f}" if n1 else ""
        if tp is not None and jn > 0:
            e = jj / jn
            kt = tp["kappa"][int(np.argmin(np.abs(tp["TS"] - t)))]
            K = np.zeros((3, 3))
            for (m, n), v in kn.VOIGT.items():
                K[m, n] = K[n, m] = kt[v]
            kperp = e @ K @ e
            kp, th, dT = f"{kperp:9.2f}", f"{kH / kperp:9.2e}", f"{jn * a.width / kperp:9.2e}"
        print(f"{TS[it]:>5.0f} {jj[0]:>10.3e} {jj[1]:>10.3e} {jj[2]:>10.3e} {kH:>11.3e} "
              f"{np.linalg.norm(jse[it]) / a.grad:>8.1e} {teff:>9} {kp:>9} {th:>9} {dT:>9}")
    print("주의: j ∝ (∇T)² -> κ_H ∝ |∇T|. 부호는 Luo 식8 표기 기준 (SM 미대조). 성분·방향은 POSCAR 의 Cartesian x y z.")
    if a.tau == "omega":
        print("주의: τ(ω) 는 bin 평균 (밴드·q 차이 무시). 표 밖 (최저 bin 아래 음향) 은 끝값 고정 -> 30K 이하 신뢰 낮음.")


def run(a):
    if not a.linear:
        return nphe(a)
    from . import aphe

    aphe.load_phonon = lambda d, sc, nac=True: load(a, nac=nac)  # 폴더 대신 pt 공통 입력
    argv = ["--dir", os.path.dirname(os.path.abspath(a.cell)), "--B", a.B, "--axis", a.axis, "--nq", a.nq,
            "--eta", 0.0 if a.eta is None else a.eta]
    argv += ["--nqz", a.nqz] if a.nqz else []
    argv += ["--T", *a.T] if a.T else []
    argv += [f"--{k}" for k in ("bands", "verify") if getattr(a, k)] + (["--no-nac"] if a.no_nac else [])
    argv += ["--save", a.save] if a.save else []
    argv += ["--plot", a.plot] if a.plot else []
    _call(aphe.main, "pt kappa --linear", argv)
