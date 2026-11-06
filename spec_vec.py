#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""46 — 누적 P(ν) 주파수 분해, **nphe_vec.py 커널 + IBZ 대칭 축소**.

29_spectral_P.py 는 27→26→aphe.py 체인을 물고 전체 BZ 를 돌아 nq 가 커지면 못 쓴다.
이 스크립트는 클러스터에 이미 있는 nphe_vec.py 하나만 있으면 돌고, --ibz 로 점을 1/8 로 줄인다.
출력 형식은 29 와 같아서 41/42/44 플롯 스크립트가 그대로 동작한다.

사용 (클러스터, nphe_vec.py 와 같은 폴더에서):
  python3 46_spec_vec.py . --nq 200 --seed 0 --ibz --out hfo2_spec_nq200_s0.npz
검산: bin 을 다 더하면 nphe_vec 의 P_ab 와 같아야 한다 (--check 가 자동 출력).
"""
import os, sys, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util
sp = importlib.util.spec_from_file_location("nphe_vec", os.path.join(HERE, "nphe_vec.py"))
V = importlib.util.module_from_spec(sp); sp.loader.exec_module(V)

AXI = {"x": 0, "y": 1, "z": 2}
EDGES = np.arange(0.0, 30.01, 0.1)          # THz, 29 와 동일


def spec_oneshift(L, nq, delta, etas, TS, block, ibz):
    """-> H[neta, nT, nbin, 3, 3] eV*kB^2, 대표점 수. bin 합 = P_ab."""
    nb = len(EDGES) - 1
    S = np.array(L.ksigns) if ibz else None
    k1 = np.linspace(-0.5, 0.5, nq, endpoint=False) + 0.5 / nq
    N, M = nq ** 3, 2 * nq
    H = np.zeros((len(etas), len(TS), nb, 3, 3))
    nkept = 0
    if ibz:
        off = np.rint(2 * nq * (k1[0] + np.asarray(delta))).astype(np.int64)
        wrap = lambda x: (x + nq - 1) % M - nq + 1
        eb = block * len(S)
    else:
        eb = block
    for i in range(0, N, eb):
        if ibz:
            J = np.stack(np.unravel_index(np.arange(i, min(i + eb, N)), (nq,) * 3), -1)
            K2 = wrap(2 * J + off)
            img = wrap(K2[:, None, :] * S[None])
            code = ((img[..., 0] + nq) * (M + 1) + (img[..., 1] + nq)) * (M + 1) + (img[..., 2] + nq)
            own = ((K2[:, 0] + nq) * (M + 1) + (K2[:, 1] + nq)) * (M + 1) + (K2[:, 2] + nq)
            keep = own == code.max(1)
            if not keep.any():
                continue
            cs = np.sort(code[keep], 1)
            mult = 1 + (np.diff(cs, axis=1) != 0).sum(1)       # 궤도 크기
            K = K2[keep] / M
        else:
            K = k1[np.stack(np.unravel_index(np.arange(i, min(i + eb, N)), (nq,) * 3), -1)] + delta
            K = (K + 0.5) % 1.0 - 0.5
            mult = np.ones(len(K))
        nkept += len(K)
        w, v, Oms = V.modes(L, K, etas)
        pref = V.thermal_pref(w, TS) * mult[None, :, None]      # [nT, nk, n]
        ib = np.clip(np.digitize(w, EDGES) - 1, 0, nb - 1).ravel()
        flat = (np.arange(len(TS))[:, None] * nb + ib[None]).ravel()
        for ie, Om in enumerate(Oms):
            for a in range(3):
                for b in range(3):
                    wt = (pref * (v[:, :, a] * Om[:, :, b])[None]).ravel()
                    H[ie, :, :, a, b] += np.bincount(flat, weights=wt,
                                                     minlength=len(TS) * nb).reshape(len(TS), nb)
    if ibz:
        H = V.sym_project(L.Rc, H)
    norm = 1.0 / (L.Vcell * 1e-30 * np.asarray(TS) ** 2 * N) / (V.E_CHG * V.KB ** 2)
    return H * norm[None, :, None, None, None], nkept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="POSCAR + FORCE_SETS + BORN 폴더")
    ap.add_argument("--nq", type=int, required=True)
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.08, 0.04])
    ap.add_argument("--tmin", type=float, default=10.0)
    ap.add_argument("--tmax", type=float, default=300.0)
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--comps", nargs=2, default=["xz", "zx"])
    ap.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ibz", action="store_true")
    ap.add_argument("--no-nac", action="store_true")
    ap.add_argument("--block", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    L = V.Lattice(V.load(os.path.abspath(a.dir), a.supercell, not a.no_nac))
    TS = np.arange(a.tmin, a.tmax + 1e-9, a.dt)
    etas = list(a.etas)
    print(f"{a.dir}\n  {L.sg}, 밴드 {L.n}, V={L.Vcell:.3f} Ang^3, NAC={'off' if a.no_nac else 'on'}, "
          f"nq={a.nq}, η={etas}, T={TS[0]:.0f}-{TS[-1]:.0f}K, IBZ={'on' if a.ibz else 'off'}", flush=True)
    if a.ibz:
        if L.ksigns is None:
            raise SystemExit("--ibz 미지원 구조")
        safe = V.safe_half_shifts(L, a.nq)
        if not safe:
            raise SystemExit(f"--ibz: nq={a.nq} 에 안전한 대칭 shift 없음 (짝수 nq 사용)")
        delta = np.array(safe[a.seed % len(safe)]) / (2 * a.nq)
    else:
        delta = np.random.default_rng(a.seed).random(3) / a.nq

    t0 = time.time()
    H5, nk = spec_oneshift(L, a.nq, delta, etas, TS, a.block, a.ibz)
    print(f"  대표점 {nk:.3e} / {a.nq**3:.3e} ({100*nk/a.nq**3:.1f}%), {time.time()-t0:.0f}s", flush=True)

    ij = [(AXI[c[0]], AXI[c[1]]) for c in a.comps]
    H = np.stack([H5[:, :, :, i, j] for i, j in ij]) * 1000          # [ncomp, neta, nT, nbin] meV*kB^2
    np.savez(a.out, comps=np.array(a.comps), H=H, Hs=H[None].transpose(1, 0, 2, 3, 4),
             EDGES=EDGES, TS=TS, ETAS=np.array(etas), nq=a.nq, delta=delta)
    ie = int(np.argmin([abs(e - 0.08) for e in etas]))
    for ic, c in enumerate(a.comps):
        for T in (80.0, 300.0):
            if not (TS[0] <= T <= TS[-1]):
                continue
            it = int(np.argmin(np.abs(TS - T)))
            h = H[ic, ie, it]; tot = h.sum()
            cum = np.cumsum(h) / tot if tot else np.zeros_like(h)
            print(f"  P_{c} η={etas[ie]} {TS[it]:.0f}K: 합 {tot:+.4f} meV·kB²  "
                  f"누적50%@{EDGES[1:][np.argmax(cum >= 0.5)]:.1f}THz  90%@{EDGES[1:][np.argmax(cum >= 0.9)]:.1f}THz",
                  flush=True)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
