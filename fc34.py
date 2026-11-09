"""pt 3rd — 3차 힘상수 (FC3) 계산 (phono3py + MLIP).

FC3 는 포논-포논 산란을 설명하며, 수명 τ(ω,T) → 열전도도/NPHE 의 입력이 된다.

과거 레퍼런스 통합:
  - NPHE_HfO2/nphe/scripts/68_mace_fc3_sc333.py        (MACE)
  - NPHE_HfO2/nphe/scripts/18_isochoric_orb_fc3.py     (ORB, 체크포인트)
  - HfO2_highsym_Weyl_phonon/scripts/tc/2.fc3_phono3py.py  (3모델 통합 + resume)

입력: POSCAR 만 필요 (MLIP 로 힘을 직접 계산). FORCE_SETS/BORN 은 쓰지 않음.
출력: fc3.hdf5, phono3py_disp.yaml, forces_fc3.npy (캐시), fc3_meta.json
      --fc2-out 주면 같은 변위 데이터로 fc2.hdf5 도 함께 저장.

사용법:
  pt 3rd                                  MACE 기본 (슈퍼셀은 공통 --d / --dim)
  pt 3rd --model orb                      ORB (v3 conservative)
  pt 3rd --model sevennet --sc-fc3 2 2 2  SevenNet, FC3 전용 슈퍼셀 지정
  pt 3rd --cutoff 4.0 --disp 0.03         변위/절단 반경 조정
  pt 3rd --fc2-out fc2.hdf5               같은 데이터셋으로 fc2 도 저장

재시작:
  같은 --outdir 로 다시 돌리면 forces_fc3.npy + forces_fc3.done.json 에서 재개.
  fc3.hdf5 가 이미 있으면 바로 skip.

FC4 (pt 4th) 는 phono3py 미지원 → 추후 symfc/hiphive 기반 별도 모듈로 추가 예정.
"""
import json
import os
import sys
import time

import numpy as np

from ._mlip import ase_of as _ase_of, load_calc as _load_calc, to_phonopy_atoms as _to_phonopy_atoms

NAME = "3rd"
HELP = "3차 힘상수 (FC3) — phono3py + MLIP (수명·열전도도·NPHE 입력)"


def add_args(sp):
    g = sp.add_argument_group("3차 힘상수 (FC3)")
    g.add_argument("--fc3", default="fc3.hdf5",
                   help="FC3 출력 파일 (기본 fc3.hdf5). 이미 있으면 skip")
    g.add_argument("--model", choices=("mace", "orb", "sevennet"), default="mace",
                   help="힘 계산 MLIP. mace=MACE-MP medium, "
                        "orb=orb_v3_conservative_inf_omat, sevennet=7net-0 (기본 mace)")
    g.add_argument("--sc-fc3", dest="sc_fc3", nargs=3, type=int, default=None,
                   metavar=("SX", "SY", "SZ"),
                   help="FC3 전용 슈퍼셀. 기본: 공통 --d/--dim 와 같음")
    g.add_argument("--cutoff", type=float, default=4.0,
                   help="cutoff_pair_distance (Å, 기본 4.0)")
    g.add_argument("--disp", type=float, default=0.03,
                   help="변위 크기 (Å, 기본 0.03)")
    g.add_argument("--symprec", type=float, default=5e-3,
                   help="대칭 허용오차 (기본 5e-3)")
    g.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                   help="MLIP 디바이스 (기본 cpu)")
    g.add_argument("--no-symmetrize", action="store_true",
                   help="fc3 대칭화 끄기 (기본 켬)")
    g.add_argument("--fc2-out", dest="fc2_out", default=None,
                   help="같은 fc3 변위 데이터셋에서 fc2 도 저장 (예: fc2.hdf5)")
    g.add_argument("--outdir", default=".",
                   help="체크포인트/부산물 저장 폴더 (기본 현재)")
    g.add_argument("--checkpoint-every", dest="ckpt_every", type=int, default=200,
                   help="변위 N 개마다 캐시 저장 (기본 200)")


# -------- helpers --------

def _supercell_matrix(a):
    sc = a.sc_fc3 if a.sc_fc3 else a.dim
    arr = np.array(sc)
    if arr.ndim == 1 and arr.size == 3:
        return np.diag(arr)
    if arr.shape == (3, 3):
        return arr
    raise ValueError(f"잘못된 supercell 형태: {sc!r}")


# -------- main --------

def run(a):
    try:
        import phono3py
        from phono3py.file_IO import write_fc3_to_hdf5
    except ImportError as e:
        print(f"[3rd] phono3py 필요: pip install phono3py  ({e})", file=sys.stderr)
        sys.exit(2)
    try:
        from ase.io import read as ase_read
    except ImportError as e:
        print(f"[3rd] ase 필요: pip install ase  ({e})", file=sys.stderr)
        sys.exit(2)

    os.makedirs(a.outdir, exist_ok=True)

    def _resolve(name):
        if name is None:
            return None
        return name if os.path.isabs(name) else os.path.join(a.outdir, name)

    fc3_path = _resolve(a.fc3)
    if os.path.exists(fc3_path):
        print(f"[3rd] 이미 존재 — skip: {fc3_path}  (지우거나 --fc3 다른 이름)")
        return

    # 구조
    if not os.path.exists(a.cell):
        print(f"[3rd] POSCAR 없음: {a.cell}", file=sys.stderr)
        sys.exit(1)
    atoms = ase_read(a.cell, format="vasp")
    cell_ph = _to_phonopy_atoms(atoms)
    sc_matrix = _supercell_matrix(a)

    print(f"[3rd] 모델={a.model}  슈퍼셀={np.ravel(sc_matrix).tolist()}  "
          f"disp={a.disp}Å  cutoff={a.cutoff}Å  symprec={a.symprec}", flush=True)

    ph3 = phono3py.Phono3py(
        cell_ph,
        supercell_matrix=sc_matrix,
        primitive_matrix=np.eye(3),
        symprec=a.symprec,
        log_level=1,
    )
    ph3.generate_displacements(distance=a.disp, cutoff_pair_distance=a.cutoff)
    scs = ph3.supercells_with_displacements  # None 은 skip 된 pair
    natoms_sc = len(ph3.supercell)
    ntot = len(scs)
    idx = [i for i, s in enumerate(scs) if s is not None]
    ndo = len(idx)

    disp_yaml = os.path.join(a.outdir, "phono3py_disp.yaml")
    ph3.save(disp_yaml)
    print(f"[3rd] 변위 {ndo}/{ntot} (유효/전체)  natoms_sc={natoms_sc}  "
          f"disp_yaml={disp_yaml}", flush=True)

    # 체크포인트
    cache = os.path.join(a.outdir, "forces_fc3.npy")
    prog = os.path.join(a.outdir, "forces_fc3.done.json")
    if os.path.exists(cache) and os.path.exists(prog):
        forces = np.load(cache)
        with open(prog) as fh:
            done = set(json.load(fh))
        if forces.shape != (ntot, natoms_sc, 3):
            print("[3rd] 캐시 shape 불일치 → 처음부터", flush=True)
            forces = np.zeros((ntot, natoms_sc, 3))
            done = set()
    else:
        forces = np.zeros((ntot, natoms_sc, 3))
        done = set()

    todo = [i for i in idx if i not in done]
    if todo:
        print(f"[3rd] 계산할 변위 {len(todo)} (재개 {len(done)})", flush=True)
        calc = _load_calc(a.model, a.device)
        t0 = time.time()
        for k, i in enumerate(todo):
            at = _ase_of(scs[i])
            at.calc = calc
            forces[i] = at.get_forces()
            done.add(i)
            if (k + 1) % a.ckpt_every == 0 or k + 1 == len(todo):
                el = time.time() - t0
                rate = el / (k + 1)
                eta = rate * (len(todo) - k - 1)
                print(f"[3rd] {len(done)}/{ndo}  {rate:.2f}s/f  "
                      f"ETA {eta/60:.1f} min", flush=True)
                np.save(cache, forces)
                with open(prog, "w") as fh:
                    json.dump(sorted(done), fh)
    else:
        print("[3rd] 모든 변위 force 캐시됨 — produce_fc3 바로 진행", flush=True)

    ph3.forces = forces
    ph3.produce_fc3(symmetrize_fc3r=not a.no_symmetrize)
    write_fc3_to_hdf5(ph3.fc3, filename=fc3_path)
    size_mb = os.path.getsize(fc3_path) / (1024 * 1024)
    print(f"[3rd] saved {fc3_path}  shape={list(ph3.fc3.shape)}  "
          f"({size_mb:.1f} MB)", flush=True)

    fc2_path = None
    if a.fc2_out:
        from phono3py.file_IO import write_fc2_to_hdf5
        ph3.produce_fc2(symmetrize_fc2=True)
        fc2_path = _resolve(a.fc2_out)
        write_fc2_to_hdf5(ph3.fc2, filename=fc2_path)
        print(f"[3rd] saved {fc2_path}  shape={list(ph3.fc2.shape)}", flush=True)

    meta = dict(
        model=a.model,
        supercell=np.ravel(sc_matrix).tolist(),
        disp=a.disp,
        cutoff_pair=a.cutoff,
        symprec=a.symprec,
        symmetrized=not a.no_symmetrize,
        n_disp=ndo,
        n_total=ntot,
        natoms_sc=natoms_sc,
        fc3_shape=list(ph3.fc3.shape),
        fc3_file=os.path.basename(fc3_path),
        fc2_file=(os.path.basename(fc2_path) if fc2_path else None),
    )
    meta_path = os.path.join(a.outdir, "fc3_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"[3rd] meta: {meta_path}", flush=True)
