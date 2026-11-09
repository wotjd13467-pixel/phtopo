"""pt 2nd — 2차 힘상수 (FC2) 계산 (phonopy + MLIP).

FC2 는 조화근사 포논 분산의 기반. phonopy 가 FORCE_SETS 로부터 만드는 것을
MLIP 로 직접 계산 (DFT 없이). FC3 보다 변위 수가 훨씬 적어 빠름.

레퍼런스:
  HfO2_accidental_Weyl_phonon/scripts/fc2_333.py
  HfO2_highsym_Weyl_phonon/scripts/tc/11.mlff_fc2_dft_nac.py

입력: POSCAR 만. FORCE_SETS 는 쓰지 않음.
출력 (--format 에 따라):
  fc2.hdf5           phonopy 호환 바이너리 (기본)
  FORCE_CONSTANTS    phonopy 호환 텍스트
  둘 다 phonopy.load(force_constants_filename=...) 또는 pt 의 --fc 옵션으로 바로 재사용 가능.
  phonopy_disp.yaml, forces_fc2.npy (캐시), fc2_meta.json 함께 생성.

사용법:
  pt 2nd                                   MACE 기본 (슈퍼셀은 공통 --d / --dim)
  pt 2nd --model sevennet --sc-fc2 3 3 3   SevenNet, FC2 전용 슈퍼셀 지정
  pt 2nd --disp 0.01                       변위 크기
  pt 2nd --format FORCE_CONSTANTS          텍스트 출력
  pt 2nd --fc2 fc2_mace.hdf5 --outdir mace_run

재시작:
  같은 --outdir 로 다시 돌리면 forces_fc2.npy + forces_fc2.done.json 에서 재개.
  출력 파일이 이미 있으면 skip.
"""
import json
import os
import sys
import time

import numpy as np

from ._mlip import ase_of, load_calc, to_phonopy_atoms

NAME = "2nd"
HELP = "2차 힘상수 (FC2) — phonopy + MLIP (조화 포논 분산 입력)"


def add_args(sp):
    g = sp.add_argument_group("2차 힘상수 (FC2)")
    g.add_argument("--fc2", default=None,
                   help="FC2 출력 파일 (기본: fc2.hdf5 또는 FORCE_CONSTANTS, --format 에 따름). "
                        "이미 있으면 skip")
    g.add_argument("--format", choices=("hdf5", "FORCE_CONSTANTS"), default="hdf5",
                   help="출력 포맷 (기본 hdf5). hdf5 → fc2.hdf5, FORCE_CONSTANTS → 텍스트")
    g.add_argument("--model", choices=("mace", "orb", "sevennet"), default="mace",
                   help="힘 계산 MLIP (기본 mace)")
    g.add_argument("--sc-fc2", dest="sc_fc2", nargs=3, type=int, default=None,
                   metavar=("SX", "SY", "SZ"),
                   help="FC2 전용 슈퍼셀. 기본: 공통 --d/--dim 와 같음")
    g.add_argument("--disp", type=float, default=0.01,
                   help="변위 크기 (Å, 기본 0.01)")
    g.add_argument("--symprec", type=float, default=5e-3,
                   help="대칭 허용오차 (기본 5e-3)")
    g.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                   help="MLIP 디바이스 (기본 cpu)")
    g.add_argument("--no-symmetrize", action="store_true",
                   help="fc2 대칭화 끄기 (기본 켬)")
    g.add_argument("--primitive", choices=("auto", "P", "eye"), default="eye",
                   help="primitive_matrix 선택: eye=np.eye(3) (기본, phtopo 다른 모듈과 일관), "
                        "P=phtopo core.load 와 같음, auto=phonopy 자동")
    g.add_argument("--outdir", default=".",
                   help="체크포인트/부산물 저장 폴더 (기본 현재)")
    g.add_argument("--checkpoint-every", dest="ckpt_every", type=int, default=20,
                   help="변위 N 개마다 캐시 저장 (기본 20 — FC2 는 변위 수가 적음)")


def _supercell_matrix(a):
    sc = a.sc_fc2 if a.sc_fc2 else a.dim
    arr = np.array(sc)
    if arr.ndim == 1 and arr.size == 3:
        return np.diag(arr)
    if arr.shape == (3, 3):
        return arr
    raise ValueError(f"잘못된 supercell 형태: {sc!r}")


def _primitive_matrix(choice):
    if choice == "eye":
        return np.eye(3)
    if choice == "P":
        return "P"
    if choice == "auto":
        return "auto"
    raise ValueError(choice)


def run(a):
    try:
        import phonopy
        from phonopy.file_IO import write_FORCE_CONSTANTS
    except ImportError as e:
        print(f"[2nd] phonopy 필요: pip install phonopy  ({e})", file=sys.stderr)
        sys.exit(2)
    try:
        from ase.io import read as ase_read
    except ImportError as e:
        print(f"[2nd] ase 필요: pip install ase  ({e})", file=sys.stderr)
        sys.exit(2)

    os.makedirs(a.outdir, exist_ok=True)

    def _resolve(name):
        if name is None:
            return None
        return name if os.path.isabs(name) else os.path.join(a.outdir, name)

    # 기본 출력 파일명 결정
    fc2_name = a.fc2 if a.fc2 else ("fc2.hdf5" if a.format == "hdf5" else "FORCE_CONSTANTS")
    fc2_path = _resolve(fc2_name)
    if os.path.exists(fc2_path):
        print(f"[2nd] 이미 존재 — skip: {fc2_path}  (지우거나 --fc2 다른 이름)")
        return

    if not os.path.exists(a.cell):
        print(f"[2nd] POSCAR 없음: {a.cell}", file=sys.stderr)
        sys.exit(1)
    atoms = ase_read(a.cell, format="vasp")
    cell_ph = to_phonopy_atoms(atoms)
    sc_matrix = _supercell_matrix(a)
    pm = _primitive_matrix(a.primitive)

    print(f"[2nd] 모델={a.model}  슈퍼셀={np.ravel(sc_matrix).tolist()}  "
          f"disp={a.disp}Å  symprec={a.symprec}  primitive={a.primitive}  "
          f"format={a.format}", flush=True)

    ph = phonopy.Phonopy(
        cell_ph,
        supercell_matrix=sc_matrix,
        primitive_matrix=pm,
        symprec=a.symprec,
        log_level=1,
    )
    ph.generate_displacements(distance=a.disp)
    scs = ph.supercells_with_displacements
    natoms_sc = len(ph.supercell)
    ntot = len(scs)

    disp_yaml = os.path.join(a.outdir, "phonopy_disp.yaml")
    ph.save(disp_yaml)
    print(f"[2nd] 변위 {ntot} 개  natoms_sc={natoms_sc}  disp_yaml={disp_yaml}", flush=True)

    # 체크포인트
    cache = os.path.join(a.outdir, "forces_fc2.npy")
    prog = os.path.join(a.outdir, "forces_fc2.done.json")
    if os.path.exists(cache) and os.path.exists(prog):
        forces = np.load(cache)
        with open(prog) as fh:
            done = set(json.load(fh))
        if forces.shape != (ntot, natoms_sc, 3):
            print("[2nd] 캐시 shape 불일치 → 처음부터", flush=True)
            forces = np.zeros((ntot, natoms_sc, 3))
            done = set()
    else:
        forces = np.zeros((ntot, natoms_sc, 3))
        done = set()

    todo = [i for i in range(ntot) if i not in done]
    if todo:
        print(f"[2nd] 계산할 변위 {len(todo)} (재개 {len(done)})", flush=True)
        calc = load_calc(a.model, a.device)
        t0 = time.time()
        for k, i in enumerate(todo):
            at = ase_of(scs[i])
            at.calc = calc
            forces[i] = at.get_forces()
            done.add(i)
            if (k + 1) % a.ckpt_every == 0 or k + 1 == len(todo):
                el = time.time() - t0
                rate = el / (k + 1)
                eta = rate * (len(todo) - k - 1)
                print(f"[2nd] {len(done)}/{ntot}  {rate:.2f}s/f  "
                      f"ETA {eta/60:.1f} min", flush=True)
                np.save(cache, forces)
                with open(prog, "w") as fh:
                    json.dump(sorted(done), fh)
    else:
        print("[2nd] 모든 변위 force 캐시됨 — produce_force_constants 바로 진행",
              flush=True)

    ph.forces = forces
    ph.produce_force_constants()
    if not a.no_symmetrize:
        ph.symmetrize_force_constants()

    if a.format == "hdf5":
        from phonopy.file_IO import write_force_constants_to_hdf5
        write_force_constants_to_hdf5(ph.force_constants, filename=fc2_path)
    else:
        write_FORCE_CONSTANTS(ph.force_constants, filename=fc2_path)

    size_kb = os.path.getsize(fc2_path) / 1024
    print(f"[2nd] saved {fc2_path}  shape={list(ph.force_constants.shape)}  "
          f"({size_kb:.1f} KB)", flush=True)

    meta = dict(
        model=a.model,
        supercell=np.ravel(sc_matrix).tolist(),
        primitive=a.primitive,
        disp=a.disp,
        symprec=a.symprec,
        symmetrized=not a.no_symmetrize,
        n_disp=ntot,
        natoms_sc=natoms_sc,
        fc2_shape=list(ph.force_constants.shape),
        fc2_file=os.path.basename(fc2_path),
        format=a.format,
    )
    meta_path = os.path.join(a.outdir, "fc2_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"[2nd] meta: {meta_path}", flush=True)
