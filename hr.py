"""SR 포논 TB 해밀토니안 (Simphony LOTO_correction = T 용).

Gonze 방식: 슈퍼셀 commensurate q 에서 D_sr = D_full − D_dd (D_dd = 역공간 Ewald 쌍극자 합) → Φ_sr.
H_{ia,jb}(R) = Φ_sr(0,R) / (m_R √(m_i m_j)) × VaspToTHz²   → H(k) 고유값 = ω² [THz²], ndegen 전부 1.
--full 이면 NAC 없는 full hr (phonon_hr.dat, LOTO_correction = F 대조용) 도 쓴다.
마지막에 phonopy 와 주파수 비교 (Max Frequency Error ~1e-13 THz 면 정상).
"""
from datetime import datetime
from pathlib import Path

import numpy as np

from .core import load

NAME, HELP = "hr", "SR TB 해밀토니안 phonon_hr_SR.dat 만들기"


def add_args(p):
    p.add_argument("--full", action="store_true", help="NAC 없는 full hr (phonon_hr.dat) 도 쓰기")
    p.add_argument("-o", "--outdir", default=".", help="출력 폴더. 기본 현재 폴더")


def build_hr_matrix(ph):
    """phonopy 객체 -> (R벡터, H(R)). 고유값이 곧 ω² [THz²]."""
    prim, sup = ph.primitive, ph.supercell
    fc = ph.force_constants * (ph.unit_conversion_factor**2)
    svecs, multi = prim.get_smallest_vectors()
    masses, p2s, s2p = prim.masses, prim.p2s_map, prim.s2p_map
    n_p, n_s, norb = len(prim), len(sup), 3 * len(prim)
    super_pos = sup.scaled_positions @ sup.cell @ np.linalg.inv(prim.cell)

    hr = {}
    for i in range(n_p):
        for j in range(n_p):
            msq = np.sqrt(masses[i] * masses[j])
            for k in range(n_s):
                if s2p[k] != p2s[j]:
                    continue
                m, off = multi[k, i]
                for l in range(m):
                    rvec = svecs[off + l] + super_pos[p2s[i]] - super_pos[s2p[k]]
                    key = tuple(int(v) for v in np.rint(rvec))
                    if key not in hr:
                        hr[key] = np.zeros((norb, norb), dtype=complex)
                    # 다중도 m 을 여기서 접어 넣으므로 ndegen 은 전부 1
                    hr[key][3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = fc[p2s[i], k] / (m * msq)

    keys = [k for k in sorted(hr) if np.abs(hr[k]).sum() > 1e-9]
    return np.array(keys, dtype=int), np.array([hr[k] for k in keys])


def write_hr(rpts, mat, path, tag):
    nrpt, norb, _ = mat.shape
    lines = [f" {tag} generated {datetime.now():%Y-%m-%d %H:%M:%S} by phtopo", f"{norb:12d}", f"{nrpt:12d}"]
    deg = np.ones(nrpt, dtype=int)
    for s in range(0, nrpt, 15):
        lines.append("".join(f"{d:5d}" for d in deg[s : s + 15]))
    for ir in range(nrpt):
        rx, ry, rz = rpts[ir]
        for jorb in range(norb):
            for iorb in range(norb):
                v = mat[ir, iorb, jorb]
                lines.append(f"{rx:8d}{ry:8d}{rz:8d}{iorb + 1:8d}{jorb + 1:8d}{v.real:20.10f}{v.imag:20.10f}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as fh:  # Simphony 용 — CRLF 금지
        fh.write("\n".join(lines) + "\n")


def freqs_from_hr(rpts, mat, q):
    """phonopy 와 같은 부호 규약: 불안정 모드는 음의 주파수 (0 으로 자르면 검증이 어긋나 보임)."""
    hq = np.einsum("r,rij->ij", np.exp(-2j * np.pi * (rpts @ q)), mat)
    vals = np.linalg.eigvalsh((hq + hq.conj().T) / 2.0)
    return np.sign(vals) * np.sqrt(np.abs(vals))


def make_sr(a):
    """-> (rpts, mat, ph_sr, ph_nac). ph_nac 은 nac_params(born 원자당 12개 등) 꺼낼 때 쓴다."""
    if a.born is None or getattr(a, "loto", None) == "F":
        raise SystemExit("BORN 없음 또는 --loto F — SR 해밀토니안은 장거리 (dipole) 항을 빼는 것이라 BORN 이 필요함 (--born 또는 --dir)")
    ph = load(a, nac=True)
    dm = ph.dynamical_matrix
    dm.make_Gonze_nac_dataset()
    ph_sr = load(a, nac=False)
    ph_sr.force_constants = dm.short_range_force_constants
    rpts, mat = build_hr_matrix(ph_sr)
    return rpts, mat, ph_sr, ph


def max_error(rpts, mat, ph_ref, nrand=12):
    rng = np.random.default_rng(42)
    qs = np.vstack([np.zeros((1, 3)), np.eye(3) * 0.5, rng.random((nrand, 3)) - 0.5])
    return max(float(np.abs(np.sort(freqs_from_hr(rpts, mat, q)) - np.sort(ph_ref.get_frequencies(q))).max()) for q in qs)


def run(a):
    out = Path(a.outdir)
    rpts, mat, ph_sr, _ = make_sr(a)
    write_hr(rpts, mat, out / "phonon_hr_SR.dat", "phonon_hr_SR.dat")
    print(f"-> {out / 'phonon_hr_SR.dat'}  ({mat.shape[1]} orbitals x {len(rpts)} R)")
    print(f"   Max Frequency Error (SR): {max_error(rpts, mat, ph_sr):.3e} THz")
    if a.full:
        ph_full = load(a, nac=False)
        r_f, m_f = build_hr_matrix(ph_full)
        write_hr(r_f, m_f, out / "phonon_hr.dat", "phonon_hr.dat")
        print(f"-> {out / 'phonon_hr.dat'}")
        print(f"   Max Frequency Error (full): {max_error(r_f, m_f, ph_full):.3e} THz")
