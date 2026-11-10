"""pt relax — MLIP 로 구조 완화 (lattice + ions).

phonon 계산 전에 힘을 0 근처로 떨어뜨리기 위한 pre-relax.
FC2/FC3 전 상태 (POSCAR_relaxed) 를 만들어 두면 음의 주파수를 줄일 수 있다.

입력: POSCAR 만.
출력 (--outdir, 기본 현재 폴더):
  POSCAR_relaxed          VASP 포맷 완화 구조 (기본 상위 폴더 아닌 outdir)
  relax.traj              ASE trajectory (optimizer 전체)
  relax_meta.json         모델/수렴/힘 요약

사용법:
  pt relax                                MACE-MP-0, cell + ions, fmax=0.01
  pt relax --model sevennet --fix-cell    SevenNet, ions 만 (고정 cell)
  pt relax --fmax 0.005 --steps 500       더 빡세게
  pt relax --out POSCAR_relaxed -o run1   outdir=run1, 결과 run1/POSCAR_relaxed

이후 사용:
  pt 2nd --cell POSCAR_relaxed ...
  pt 3rd --cell POSCAR_relaxed ...
"""
import json
import os
import sys
import time

import numpy as np

from ._mlip import load_calc

NAME = "relax"
HELP = "구조 완화 (lattice + ions) — MLIP (phonon 전 pre-relax)"


def add_args(sp):
    g = sp.add_argument_group("구조 완화 (relax)")
    g.add_argument("--model", choices=("mace", "orb", "sevennet"), default="mace",
                   help="MLIP (기본 mace)")
    g.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                   help="디바이스 (기본 cpu)")
    g.add_argument("--fmax", type=float, default=1e-2,
                   help="수렴 힘 임계 (eV/Å, 기본 0.01)")
    g.add_argument("--steps", type=int, default=300,
                   help="최대 optimizer step (기본 300)")
    g.add_argument("--fix-cell", action="store_true",
                   help="격자 고정, ions 만 완화 (기본: cell + ions 둘 다)")
    g.add_argument("--fix-symmetry", action="store_true",
                   help="공간군 유지 (ase.spacegroup.symmetrize.FixSymmetry)")
    g.add_argument("--pressure", type=float, default=0.0,
                   help="외부 압력 GPa (기본 0, cell 완화 때만 의미)")
    g.add_argument("--optimizer", choices=("fire", "bfgs", "lbfgs"), default="fire",
                   help="ASE optimizer (기본 fire)")
    g.add_argument("--out", default="POSCAR_relaxed",
                   help="완화된 POSCAR 파일명 (기본 POSCAR_relaxed)")
    g.add_argument("--outdir", default=".",
                   help="부산물 저장 폴더 (기본 현재)")


def run(a):
    try:
        from ase.io import read as ase_read
        from ase.io import write as ase_write
        from ase.filters import FrechetCellFilter
        from ase.optimize import BFGS, FIRE, LBFGS
    except ImportError as e:
        print(f"[relax] ase 필요: pip install ase  ({e})", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(a.cell):
        print(f"[relax] POSCAR 없음: {a.cell}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(a.outdir, exist_ok=True)
    out_path = a.out if os.path.isabs(a.out) else os.path.join(a.outdir, a.out)
    traj_path = os.path.join(a.outdir, "relax.traj")
    meta_path = os.path.join(a.outdir, "relax_meta.json")

    atoms = ase_read(a.cell, format="vasp")
    n = len(atoms)
    print(f"[relax] 모델={a.model}  natoms={n}  fix_cell={a.fix_cell}  "
          f"fmax={a.fmax} eV/Å  steps≤{a.steps}  optimizer={a.optimizer}", flush=True)

    calc = load_calc(a.model, a.device)
    atoms.calc = calc

    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    s0 = atoms.get_stress(voigt=False)
    print(f"[relax] 초기 E={e0:.6f} eV  |F|_max={np.linalg.norm(f0, axis=1).max():.4f} eV/Å  "
          f"|σ|_max={np.abs(s0).max()*160.21766208:.3f} GPa", flush=True)

    if a.fix_symmetry:
        try:
            from ase.spacegroup.symmetrize import FixSymmetry
            atoms.set_constraint(FixSymmetry(atoms))
            print("[relax] 공간군 대칭 유지 (FixSymmetry)", flush=True)
        except ImportError:
            print("[relax] FixSymmetry 없음 — ase 버전 낮음, --fix-symmetry 무시", file=sys.stderr)

    # Optimizer 대상: cell 완화면 FrechetCellFilter 로 감싸기
    if a.fix_cell:
        target = atoms
    else:
        # ASE 압력: 0.0 외에는 eV/Å³ 단위 — GPa → eV/Å³
        p_ev_a3 = a.pressure / 160.21766208
        target = FrechetCellFilter(atoms, scalar_pressure=p_ev_a3)

    OPT = {"fire": FIRE, "bfgs": BFGS, "lbfgs": LBFGS}[a.optimizer]
    opt = OPT(target, trajectory=traj_path, logfile="-")

    t0 = time.time()
    converged = opt.run(fmax=a.fmax, steps=a.steps)
    dt = time.time() - t0

    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()
    s1 = atoms.get_stress(voigt=False)
    fmax_final = float(np.linalg.norm(f1, axis=1).max())
    smax_final_gpa = float(np.abs(s1).max() * 160.21766208)

    ase_write(out_path, atoms, format="vasp", direct=True, sort=False, vasp5=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[relax] {'수렴' if converged else '미수렴'}  nsteps={opt.nsteps}  "
          f"ΔE={e1-e0:+.6f} eV  |F|_max={fmax_final:.4f} eV/Å  "
          f"|σ|_max={smax_final_gpa:.3f} GPa  ({dt:.1f}s)", flush=True)
    print(f"[relax] saved {out_path}  ({size_kb:.1f} KB)", flush=True)
    print(f"[relax] traj  {traj_path}", flush=True)

    meta = dict(
        model=a.model, device=a.device,
        optimizer=a.optimizer, fmax=a.fmax, steps_max=a.steps, steps_taken=opt.nsteps,
        fix_cell=a.fix_cell, fix_symmetry=a.fix_symmetry, pressure_GPa=a.pressure,
        converged=bool(converged),
        energy_initial_eV=float(e0), energy_final_eV=float(e1),
        delta_energy_eV=float(e1 - e0),
        fmax_initial_eV_per_A=float(np.linalg.norm(f0, axis=1).max()),
        fmax_final_eV_per_A=fmax_final,
        stress_max_initial_GPa=float(np.abs(s0).max() * 160.21766208),
        stress_max_final_GPa=smax_final_gpa,
        lattice_initial=atoms.cell.array.tolist(),  # final (same object, after relax)
        wall_time_sec=dt,
        out_file=os.path.basename(out_path),
    )
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"[relax] meta  {meta_path}", flush=True)
    if not converged:
        print("[relax] 경고: 미수렴 — --steps 늘리거나 --fmax 완화 / 다른 --optimizer 시도", file=sys.stderr)
