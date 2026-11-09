"""MLIP (MACE/ORB/SevenNet) + phonopy/phono3py 공통 헬퍼.

내부용 (언더스코어 모듈). fc2.py, fc34.py 가 import 해서 재사용.
"""
import sys


def load_calc(model, device):
    """MLIP 계산기 로드. 패키지 없으면 친절한 에러로 exit(2)."""
    if model == "mace":
        try:
            from mace.calculators import mace_mp
        except ImportError as e:
            print(f"[mlip] mace 패키지 없음: pip install mace-torch  ({e})", file=sys.stderr)
            sys.exit(2)
        return mace_mp(model="medium", device=device, default_dtype="float64")
    if model == "orb":
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
        except ImportError as e:
            print(f"[mlip] orb-models 패키지 없음: pip install orb-models  ({e})", file=sys.stderr)
            sys.exit(2)
        ff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
        return ORBCalculator(ff, device=device)
    if model == "sevennet":
        try:
            from sevenn.calculator import SevenNetCalculator
        except ImportError as e:
            print(f"[mlip] sevenn 패키지 없음: pip install sevenn  ({e})", file=sys.stderr)
            sys.exit(2)
        return SevenNetCalculator("7net-0", device=device)
    raise ValueError(f"unknown model {model!r}")


def to_phonopy_atoms(ase_atoms):
    from phonopy.structure.atoms import PhonopyAtoms
    return PhonopyAtoms(
        symbols=ase_atoms.get_chemical_symbols(),
        cell=ase_atoms.get_cell(),
        scaled_positions=ase_atoms.get_scaled_positions(),
    )


def ase_of(phonopy_sc):
    from ase import Atoms
    return Atoms(
        symbols=phonopy_sc.symbols,
        cell=phonopy_sc.cell,
        scaled_positions=phonopy_sc.scaled_positions,
        pbc=True,
    )
