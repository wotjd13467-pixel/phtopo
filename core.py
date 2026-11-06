"""모든 하위명령이 같은 방식으로 phonopy 를 읽는다."""
import sys

import numpy as np
import phonopy

THZ = 15.633302  # VaspToTHz


def load(a, nac=True):
    """a = argparse 결과 (cell, dim, fs, born). primitive = "P" 고정 — "auto" 면 극성축이 재배열된다."""
    check_dim(a)
    if getattr(a, "loto", None) == "F":
        nac = False
    if nac and a.born is None:
        print("[경고] BORN 없음 → NAC 끔 (극성 물질이면 LO-TO 가 빠져 결과가 달라짐)", file=sys.stderr)
        nac = False
    ph = phonopy.load(
        unitcell_filename=a.cell,
        supercell_matrix=a.dim,
        primitive_matrix="P",
        **({"force_constants_filename": a.fc} if getattr(a, "fc", None) else {"force_sets_filename": a.fs}),
        born_filename=a.born if nac else None,
        is_nac=nac,  # 명시해야 함 — 기본값이 BORN 을 몰래 읽는다
        calculator="vasp",
        symmetrize_fc=True,
        is_compact_fc=False,  # hr 만들 때 fc[p2s[i], k] 인덱싱이 full fc 를 가정
        log_level=0,
    )
    if nac:
        ph.nac_params = dict(ph.nac_params, method="gonze")
    return ph


def guess_dim(a):
    """--d 를 안 줬을 때 슈퍼셀: phonopy_disp.yaml / phonopy.yaml 의 supercell_matrix → 힘 파일 원자 수 = 단위격자 × n³ → 2 2 2."""
    import os

    import yaml
    from phonopy.interface.calculator import read_crystal_structure

    d = os.path.dirname(os.path.abspath(a.cell))
    for y in ("phonopy_disp.yaml", "phonopy.yaml"):
        f = os.path.join(d, y)
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                M = (yaml.safe_load(fh) or {}).get("supercell_matrix")
            if M is not None:
                M = np.array(M, int)
                return (np.diag(M).tolist() if np.count_nonzero(M - np.diag(np.diag(M))) == 0 else M.tolist()), y
    if not os.path.exists(a.cell):  # kappa --P 처럼 POSCAR 가 필요 없는 명령. 필요한 명령은 check_dim 이 알림
        return [2, 2, 2], "기본값"
    natom, src = None, None
    try:
        if getattr(a, "fc", None):
            src = a.fc
            if a.fc.endswith((".hdf5", ".h5")):
                import h5py

                with h5py.File(a.fc, "r") as h:
                    natom = h["force_constants"].shape[1]
            else:
                with open(a.fc) as fh:
                    natom = int(fh.readline().split()[-1])
        else:
            src = a.fs
            with open(a.fs) as fh:
                natom = int(fh.readline().split()[0])
        nuc = len(read_crystal_structure(a.cell, interface_mode="vasp")[0])
    except (OSError, ValueError, IndexError, KeyError):
        return [2, 2, 2], "기본값"  # 파일이 없거나 다른 형식 — phonopy 가 알아서 알림
    n = round((natom / nuc) ** (1 / 3))
    if n**3 * nuc != natom:
        raise SystemExit(f"{src}: 슈퍼셀 원자 {natom} 개 = 단위격자 {nuc} × {natom / nuc:g} — 정육면체가 아니라 추측 불가 → --d 로 지정 (예 --d 3 3 2)")
    return [n] * 3, f"{os.path.basename(src)} {natom} 원자"


def check_dim(a):
    """FORCE_SETS 원자 수 ≠ 단위격자 원자 수 × det(슈퍼셀) 이면 phonopy 가 IndexError 로 죽음 → 미리 알아듣게."""
    import os

    from phonopy.interface.calculator import read_crystal_structure

    if not os.path.exists(a.cell):
        raise SystemExit(f"{a.cell} 없음 — 입력 폴더에서 실행하거나 --dir 폴더")
    if getattr(a, "fc", None):
        return  # 힘상수 파일은 phonopy 가 슈퍼셀 크기를 검사함
    try:
        with open(a.fs) as f:
            nfs = int(f.readline().split()[0])
    except (OSError, ValueError, IndexError):
        return  # 파일 없음·다른 형식은 phonopy 가 알아서 알림
    nuc = len(read_crystal_structure(a.cell, interface_mode="vasp")[0])
    M = np.diag(a.dim) if np.size(a.dim) == 3 else np.array(a.dim)
    nsc = nuc * round(abs(np.linalg.det(M)))
    if nfs != nsc:
        guess = round((nfs / nuc) ** (1 / 3))
        hint = f" — 아마 --d {guess} {guess} {guess}" if guess**3 * nuc == nfs else ""
        raise SystemExit(f"{a.fs}: 슈퍼셀 원자 {nfs} 개인데 --d {np.ravel(a.dim).tolist()} 는 {nuc}×{nsc // nuc} = {nsc} 개{hint}")


def rec_rows(ph):
    """행 = b1, b2, b3 (Cartesian, 1/Å)."""
    return 2 * np.pi * np.linalg.inv(ph.primitive.cell).T


def frame(ph):
    """노드 파일 머리줄: 분수좌표 (rlu) ↔ Cartesian x y z (nlhall·kappa 의 성분 축) 대응."""
    b = rec_rows(ph)
    return "# q = q1 b1 + q2 b2 + q3 b3 (rlu). Cartesian [1/Å]: " + "  ".join(
        f"b{i + 1}=({v[0]:+.4f},{v[1]:+.4f},{v[2]:+.4f})" for i, v in enumerate(b)) + "\n"


def all_bands(a, ph):
    """--band 또는 전체 밴드쌍 1..3N−1 (scan, line 기본)."""
    nb = 3 * len(ph.primitive)
    bands = a.band or list(range(1, nb))
    if max(bands) >= nb:
        raise SystemExit(f"--band {max(bands)}: 이 구조의 밴드쌍은 1..{nb - 1}")
    return bands


def need_band(a):
    """--band 가 꼭 있어야 하는 명령."""
    if not a.band:
        raise SystemExit(f"pt {a.cmd}: --band 필요 (예 --band 17, --band 11-18)")
    return a.band


def parse_bands(s):
    """"17" / "11-18" / "11,13,17-18" -> [아래 밴드 번호...] (1부터, 밴드쌍 = (b, b+1))."""
    out = []
    for part in s.split(","):
        lo, _, hi = part.strip().partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    if not out or min(out) < 1:
        raise ValueError(f"--band 형식: 17 / 11-18 / 11,13 (받은 값 {s!r})")
    return out


def read_nodes(path):
    """-> (q (N,3), band (N,) 또는 None).
    nodes.txt 6열 (q1 q2 q3 f chi band) / 옛 5열 (band 없음) / Simphony Nodes.dat 8열 (끝 3열 = 분수좌표, band 없음)."""
    d = np.atleast_2d(np.loadtxt(path, encoding="utf-8"))  # 머리줄이 한글 — Windows 기본 cp949 면 죽음
    if d.shape[1] >= 8:
        return d[:, 5:8], None
    return d[:, :3], (d[:, 5].astype(int) if d.shape[1] == 6 else None)


def overlap_det(x, y):
    """det <x_i|y_j> — 묶음(누적) FHS 의 한 변. x, y: (..., 3N, n)."""
    return np.linalg.det(np.einsum("...ai,...aj->...ij", x.conj(), y))
