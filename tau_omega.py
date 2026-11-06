#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""39 — parent(+P) 포논 수명 τ(ω,T) 표 만들기 (phono3py RTA). 40_kappa_nphe.py 의 입력.

DFT fc2+BORN(isochoric/0) + Orb fc3(results/isochoric_fc3/0_fc3.hdf5), 22_pristine_hall_ratio 와 같은 설정.
온도는 spectral_P_nq60.npz 의 TS(10~300K 30점)와 같게 맞춤.
τ = 1/(4π·Γ[THz]) ps  (Γ = phono3py gamma, HWHM). 주파수 bin 별 Γ 의 가중평균(산란율 평균) -> τ.
출력: results/tau_parent.npz  (freq, gamma, weights, TS, kappa[T,6], ctr, tau_bin[T,nbin], nmode_bin)
사용: python 39_tau_omega.py [--mesh 11]   (11^3 기준 수십 분)"""
import os, argparse, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UC = os.path.join(PROJ, "calc", "phonon", "DFT_calc", "isochoric", "0")
FC3 = os.path.join(PROJ, "results", "isochoric_fc3", "0_fc3.hdf5")
BIN = 0.25   # THz, 거친 bin (11^3 메쉬에서 bin 당 모드 수 확보)


def run(mesh, TS):
    import phonopy, phono3py
    from phonopy.interface.calculator import read_crystal_structure
    from phono3py.file_IO import read_fc3_from_hdf5
    cell, _ = read_crystal_structure(os.path.join(UC, "POSCAR"), interface_mode="vasp")
    ph = phono3py.Phono3py(cell, supercell_matrix=[2, 2, 2], primitive_matrix=np.eye(3), log_level=0)
    phon = phonopy.load(unitcell_filename=os.path.join(UC, "POSCAR"),
                        force_sets_filename=os.path.join(UC, "FORCE_SETS"),
                        born_filename=os.path.join(UC, "BORN"),
                        supercell_matrix=[2, 2, 2], primitive_matrix=np.eye(3), is_nac=True, log_level=0)
    ph.fc2 = phon.force_constants; ph.nac_params = phon.nac_params
    ph.fc3 = read_fc3_from_hdf5(FC3)
    ph.mesh_numbers = [mesh] * 3
    ph.init_phph_interaction()
    ph.run_thermal_conductivity(temperatures=TS, write_kappa=False)
    tc = ph.thermal_conductivity
    return (np.array(tc.frequencies), np.array(tc.gamma)[0], np.array(tc.grid_weights),
            np.array(tc.kappa)[0])


def bin_tau(freq, gamma, weights):
    """-> ctr[nbin], tau_bin[T,nbin] ps (빈 bin 은 nan), nmode[nbin]."""
    edges = np.arange(0.0, freq.max() + BIN, BIN)
    f = freq.ravel(); w = np.repeat(weights, freq.shape[1]).astype(float)
    ib = np.digitize(f, edges) - 1
    ok = f > 0.05                                  # Γ 음향 0 모드 제외
    nb = len(edges) - 1
    nmode = np.bincount(ib[ok], minlength=nb)[:nb]
    W = np.bincount(ib[ok], weights=w[ok], minlength=nb)[:nb]
    tau = np.full((len(gamma), nb), np.nan)
    for it, g in enumerate(gamma):
        G = np.bincount(ib[ok], weights=(w * g.ravel())[ok], minlength=nb)[:nb]
        with np.errstate(divide="ignore", invalid="ignore"):
            tau[it] = 1.0 / (4 * np.pi * G / W)    # 평균 Γ -> τ
    tau[:, nmode == 0] = np.nan
    return 0.5 * (edges[:-1] + edges[1:]), tau, nmode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, default=11)
    a = ap.parse_args()
    TS = np.load(os.path.join(PROJ, "results", "spectral_P_nq60.npz"))["TS"]
    t0 = time.time()
    freq, gamma, weights, kappa = run(a.mesh, TS)
    ctr, tau, nmode = bin_tau(freq, gamma, weights)
    out = os.path.join(PROJ, "results", "tau_parent.npz")
    np.savez(out, freq=freq, gamma=gamma, weights=weights, TS=TS, kappa=kappa,
             ctr=ctr, tau_bin=tau, nmode_bin=nmode, mesh=a.mesh)
    print(f"[parent DFT fc2 + Orb fc3, mesh {a.mesh}^3] ({time.time()-t0:.0f}s)")
    for T in (30.0, 80.0, 150.0, 300.0):
        it = list(TS).index(T)
        sel = (ctr > 4) & (ctr < 6)
        print(f"T={T:>4.0f}K  κ_xx,yy,zz={kappa[it,0]:.1f},{kappa[it,1]:.1f},{kappa[it,2]:.1f} W/mK  "
              f"τ(4-6THz)={np.nanmedian(tau[it, sel]):.2f} ps")
    print("저장:", out)


if __name__ == "__main__":
    main()
