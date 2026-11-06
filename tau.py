"""포논 수명 τ(ω,T) 표 (phono3py RTA) → tau_parent.npz — pt kappa --tau omega 의 입력.

fc2 + BORN 은 pt 공통 입력, fc3 는 --fc3 (phono3py hdf5, 예: MACE/Orb 로 만든 것).
τ = 1/(4π·Γ[THz]) ps (Γ = phono3py gamma, HWHM). 주파수 bin (0.25 THz) 별 Γ 가중평균 → τ. κ (RTA) 도 같이 저장.
온도는 --spec (pt spectral 의 npz) 의 T 와 맞춤 (kappa --tau omega 가 같은 T 를 요구). 없으면 -T.
본체 phtopo/tau_omega.py = C:\\Users\\techw\\Desktop\\cjs\\1_Projects\\NPHE_HfO2\\nphe\\scripts\\39_tau_omega.py 사본 (수정 없음).
fc2 는 pt 공통 입력 (--d 로 슈퍼셀), fc3 슈퍼셀은 --fc3-dim (기본 = --d). 본체의 run() 은 2×2×2·파일 이름 고정이라
같은 일을 여기서 다시 함 (phono3py 호출은 본체와 같음, 2×2×2 에서 본체와 결과 같음 확인). bin_tau 는 본체 것.
"""
import time

import numpy as np

from . import tau_omega as to
from .core import load
from .nlhall import t_arg

NAME, HELP = "tau", "phono3py τ(ω,T) 표 → tau_parent.npz (kappa --tau omega 입력)"


def add_args(p):
    p.add_argument("--fc3", required=True, help="fc3 hdf5 (phono3py)")
    p.add_argument("--fc3-dim", nargs="+", default=None, help="fc3 를 만든 슈퍼셀 (예 2 2 2). 기본 = --d")
    p.add_argument("--mesh", type=int, default=11, help="phono3py mesh n³ (11³ 에 수십 분). 기본 11")
    p.add_argument("--spec", help="pt spectral 의 npz — 그 온도를 씀")
    p.add_argument("-T", dest="T", type=t_arg, default=t_arg("10:300:10"), help="--spec 없을 때 온도 시작:끝:간격. 기본 10:300:10")
    p.add_argument("--isotope", action="store_true", help="동위원소 산란 추가 (phono3py 자연 존재비). gamma = 3-포논 + 동위원소")
    p.add_argument("--boundary", type=float, default=None, help="경계 산란 평균자유행로 L [μm] — κ 에만. P′ 는 pt pprime --boundary 로 모드마다")
    p.add_argument("-o", "--out", default="tau_parent.npz", help="기본 tau_parent.npz (kappa --data 폴더에 두면 됨)")


def rta(a, mesh, TS):
    """-> (freq, gamma[T], weights, kappa[T,6]) — tau_omega.run 과 같은 phono3py 호출, 슈퍼셀만 자유."""
    import phono3py
    from phono3py.file_IO import read_fc3_from_hdf5

    from .cli import dim_arg

    phon = load(a)  # fc2 + BORN (NAC), 슈퍼셀 --d
    d3 = dim_arg(" ".join(a.fc3_dim)) if a.fc3_dim else a.dim
    ph = phono3py.Phono3py(phon.unitcell, supercell_matrix=d3, phonon_supercell_matrix=a.dim,
                           primitive_matrix=phon.primitive_matrix, log_level=0)
    ph.fc2 = phon.force_constants
    ph.nac_params = phon.nac_params
    ph.fc3 = read_fc3_from_hdf5(a.fc3)
    ph.mesh_numbers = [mesh] * 3
    ph.init_phph_interaction()
    ph.run_thermal_conductivity(temperatures=TS, write_kappa=False, is_isotope=a.isotope, boundary_mfp=a.boundary)
    tc = ph.thermal_conductivity
    g = np.array(tc.gamma)[0]  # [T, gp, band] 3-포논
    giso = np.array(tc.gamma_isotope)[0] if a.isotope else np.zeros(g.shape[1:])  # [gp, band] 온도 무관
    return (np.array(tc.frequencies), g + giso[None], np.array(tc.grid_weights), np.array(tc.kappa)[0],
            np.array(tc.qpoints), giso)


def run(a):
    TS = np.load(a.spec)["TS"] if a.spec else a.T
    t0 = time.time()
    freq, gamma, weights, kappa, qpts, giso = rta(a, a.mesh, TS)
    ctr, tau, nmode = to.bin_tau(freq, gamma, weights)
    np.savez(a.out, freq=freq, gamma=gamma, weights=weights, TS=TS, kappa=kappa, qpoints=qpts, gamma_iso=giso, boundary=a.boundary or 0.0,  # qpoints: pprime 이 Γ_n(q) 를 펼칠 때
             ctr=ctr, tau_bin=tau, nmode_bin=nmode, mesh=a.mesh)
    extra = (", 동위원소" if a.isotope else "") + (f", 경계 {a.boundary} μm (κ)" if a.boundary else "")
    print(f"[fc2 {a.fc or a.fs} (슈퍼셀 {a.dim}) + fc3 {a.fc3}, mesh {a.mesh}^3{extra}] ({time.time() - t0:.0f}s)")
    for T in (30.0, 80.0, 150.0, 300.0):
        it = int(np.argmin(np.abs(TS - T)))
        sel = (ctr > 4) & (ctr < 6)
        print(f"T={TS[it]:>4.0f}K  κ_xx,yy,zz={kappa[it, 0]:.1f},{kappa[it, 1]:.1f},{kappa[it, 2]:.1f} W/mK  "
              f"τ(4-6THz)={np.nanmedian(tau[it, sel]):.2f} ps")
    print("->", a.out)
