"""Simphony (pn.x) 입력 일괄 생성.

    hr/phonon_hr_SR.dat, hr/phonon_hr.dat   SR hr (LOTO=T 용) / full hr (LOTO=F 대조용)
    band_loto/pn.in        LO–TO 포함 밴드 — 검증된 유일한 조합: SR hr + LOTO_correction = T
    band_nonac/pn.in       NAC 없는 밴드 (대조군, phonopy no-NAC 과 일치해야 함)
    findnodes_<band>/pn.in 밴드쌍 축퇴점 탐색 → Nodes.dat
    chirality_<band>/pn.in WeylChirality. --nodes Nodes.dat 를 주면 좌표를 채움, 없으면 pn.in.template
    --band 11-18 이면 findnodes_11 ... findnodes_18, chirality_11 ... 각각 (NumOccupied = band)
    run_pn.sh              실행 드라이버 (WSL): bash run_pn.sh band_loto findnodes_17
함정: full hr + LOTO=T 는 이중계산, added_LR_in_Real_Space=T 금지, LOTO_BC 는 원자당 하나씩.
Simphony 노드로 χ 를 계산할 것 (phonopy 노드를 넣으면 0.003~0.02 어긋나 χ=0 이 나온 적 있음).
Simphony χ 전역 부호는 phtopo(Kubo) 규약과 반대.
KPATH = seekpath 자동 경로 (pt line 과 같음 — 좌표는 이 POSCAR 역격자 rlu 로 변환).
"""
import shutil
from pathlib import Path

import numpy as np

from .core import read_nodes
from .core import load, need_band
from .hr import build_hr_matrix, make_sr, max_error, write_hr

NAME, HELP = "simphony", "Simphony pn.in + hr 일괄 생성"

def kpath(ph):
    from .line import seekpath_segments

    nm = lambda x: "G" if x == "GAMMA" else x
    return "\n".join(f"{nm(a)} {qa[0]:.6f} {qa[1]:.6f} {qa[2]:.6f}  {nm(b)} {qb[0]:.6f} {qb[1]:.6f} {qb[2]:.6f}"
                     for a, qa, b, qb in seekpath_segments(ph))


def add_args(p):
    p.add_argument("--nodes", help="Simphony Nodes.dat (또는 nodes.txt) — 있으면 chirality/pn.in 좌표를 채움")
    p.add_argument("--radius", type=float, default=0.002, help="WeylChirality 구 반지름. 기본 0.002")
    p.add_argument("--nk", type=int, default=61, help="WeylChirality Nk1=Nk2 (41 은 틀린 적 있음, 121 권장). 기본 61")
    p.add_argument("-o", "--outdir", default=".", help="출력 폴더. 기본 현재 폴더")


def geom_block(prim, nac):
    """LATTICE / ATOMS / LOTO_DT / LOTO_BC — phonopy primitive 에서 직접 (질량도)."""
    out = ["LATTICE", "Angstrom"] + ["".join(f"{v:24.16f}" for v in row) for row in prim.cell]
    out += ["", "ATOMS", str(len(prim)), "Direct"]
    for s, m, p in zip(prim.symbols, prim.masses, prim.scaled_positions):
        out.append(f"{s:<3s} {m:14.6f}  " + "".join(f"{v:24.16f}" for v in p))
    out += ["", "LOTO_DT"] + ["".join(f"{v:22.10f}" for v in row) for row in np.array(nac["dielectric"])]
    born = np.array(nac["born"])
    assert born.shape == (len(prim), 3, 3), f"born shape {born.shape}: 원자당 하나씩이어야 함"
    out += ["", "LOTO_BC"]
    for z in born:
        out += ["".join(f"{v:22.10f}" for v in row) for row in z]
    return "\n".join(out)


def pn_in(hrfile, loto, calc, nocc, params, geo, tail):
    return f"""&TB_FILE
Hrfile = '{hrfile}'
/

&CONTROL
LOTO_correction = {loto}
{calc} = T
/

&SYSTEM
NSLAB       = 1
NumOccupied = {nocc}
SOC         = 0
E_FERMI     = 0.0
/

&PARAMETERS
{params}
/

{geo}

{tail}
"""


def wtext(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:  # CRLF 금지
        fh.write(text)


def run(a):
    need_band(a)
    out = Path(a.outdir)
    rpts, mat, ph_sr, ph = make_sr(a)
    write_hr(rpts, mat, out / "hr" / "phonon_hr_SR.dat", "phonon_hr_SR.dat")
    ph_full = load(a, nac=False)
    r_f, m_f = build_hr_matrix(ph_full)
    write_hr(r_f, m_f, out / "hr" / "phonon_hr.dat", "phonon_hr.dat")
    print(f"-> hr/ ({mat.shape[1]} orbitals x {len(rpts)} R)  roundtrip SR {max_error(rpts, mat, ph_sr):.2e} THz")

    geo = geom_block(ph.primitive, ph.nac_params)
    nb, bands = 3 * len(ph.primitive), a.band
    if a.nodes and len(bands) > 1:
        raise SystemExit("--nodes (Simphony Nodes.dat 는 band 정보 없음) 와 함께면 --band 는 하나만")
    kp = kpath(ph)
    kpath_card = f"KPATH_BULK\n{len(kp.splitlines())}\n{kp}"
    wtext(out / "band_loto" / "pn.in", pn_in("phonon_hr_SR.dat", "T", "BulkBand_calc", nb, "Nk1 = 101", geo, kpath_card))
    wtext(out / "band_nonac" / "pn.in", pn_in("phonon_hr.dat", "F", "BulkBand_calc", nb, "Nk1 = 101", geo, kpath_card))
    for b in bands:
        wtext(out / f"findnodes_{b}" / "pn.in",
              pn_in("phonon_hr_SR.dat", "T", "FindNodes_calc", b, "Nk1 = 13\nNk2 = 13\nNk3 = 13\nGap_threshold = 0.0001", geo,
                    "KCUBE_BULK\n 0.0  0.0  0.0\n 1.0  0.0  0.0\n 0.0  1.0  0.0\n 0.0  0.0  1.0"))
        if a.nodes:
            q = read_nodes(a.nodes)[0]
            card = f"WEYL_CHIRALITY\n{len(q)}\nDirect\n{a.radius}\n" + "\n".join("".join(f"{v:14.7f}" for v in r) for r in q)
            name = "pn.in"
        else:
            card = f"WEYL_CHIRALITY\n<노드개수>\nDirect\n{a.radius}\n<k1>  <k2>  <k3>"
            name = "pn.in.template"
        wtext(out / f"chirality_{b}" / name, pn_in("phonon_hr_SR.dat", "T", "WeylChirality_calc", b,
                                                 f"Nk1 = {a.nk}\nNk2 = {a.nk}", geo, card))
    fn = " ".join(f"findnodes_{b}" for b in bands)
    print(f"-> band_loto/ band_nonac/ {fn}/ chirality_<band>/" + ("pn.in (글라이드면·k=½ 위 노드는 직접 지울 것)" if a.nodes
          else "pn.in.template (findnodes 후: pt simphony --band <b> --nodes findnodes_<b>/Nodes.dat)"))
    shutil.copy(Path(__file__).with_name("run_pn.sh"), out / "run_pn.sh")
    print(f"   실행 (WSL): bash run_pn.sh band_loto {fn}")
