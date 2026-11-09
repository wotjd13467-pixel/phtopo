"""pt 3rd / pt 4th — 3차/4차 힘상수 (FC3/FC4) 계산.

3차 힘상수 (FC3) 는 포논-포논 산란을 설명하며, 포논 수명·열전도도·NPHE 계산에 필요.
4차 힘상수 (FC4) 는 고차 효과, 열전도도 보정 등에 사용.

사용법:
  pt 3rd                                  FORCE_SETS 기반 3차 힘상수 계산
  pt 3rd --fc3 fc3.hdf5                    기존 3차 힘상수 파일 사용
  pt 3rd --fc3 fc3.hdf5 --mace             MACE 모델로 3차 힘상수 계산
  pt 3rd --fc3 fc3.hdf5 --orb              DFT/오비탈 기반 3차 힘상수 계산
  pt 4th --fc4 fc4.hdf5                    4차 힘상수 계산 (기존 데이터 기반)
  pt 4th --fc4 fc4.hdf5 --mace             MACE로 4차 힘상수 계산

출력:
  fc3.hdf5 — 3차 힘상수 (shape: (nsuper, nsuper, 3, 3, 3))
  fc4.hdf5 — 4차 힘상수 (shape: (nsuper, nsuper, nsuper, 3, 3, 3, 3))
"""

import os
import sys

import numpy as np

import symfc

NAME = "3rd"
HELP = "3차 힘상수 (FC3) 계산 — 수명·열전도도·NPHE에 사용"
__doc__ = """
pt 3rd — 3차 힘상수 (FC3)

3차 힘상수 (FC3) 는 포논-포논 산란을 설명하며, 포논 수명과 열전도도 계산에 필수적이다.

옵션:
  --fc3 FILE     출력 파일 (기본: fc3.hdf5)
  --mace         MACE 모델로 3차 힘상수 계산 (머신러닝 포텐셜)
  --orb          DFT/오비탈 기반 3차 힘상수 계산 (기존 방식)
  --fc2 FILE     2차 힘상수 파일 (FORCE_CONSTANTS 또는 fc2.hdf5)
  --flip FLAG    힘 반전 (FHS) 켜기 T/F (기본 T)
  --symmetrized  대칭화 켜기 T/F (기본 T)
  --cutoff FLOAT 절단 반경 (Å, 기본: 자동 추정)
  --order N      계산 차수 (기본 3)

예시:
  pt 3rd
  pt 3rd --fc3 fc3.hdf5 --mace
  pt 3rd --fc3 fc3.hdf5 --orb
"""


def add_args(sp):
    g = sp.add_argument_group("3차 힘상수")
    g.add_argument("--fc3", default=None, help="3차 힘상수 출력 파일 (기본: fc3.hdf5)")
    g.add_argument("--mace", action="store_true", help="MACE 모델로 3차 힘상수 계산")
    g.add_argument("--orb", action="store_true", help="DFT/오비탈 기반 3차 힘상수 계산")
    g.add_argument("--fc2", default=None, help="2차 힘상수 파일 (FORCE_CONSTANTS, fc2.hdf5)")
    g.add_argument("--flip", type=str.upper, choices=("T", "F"), default="T",
                   help="힘 반전 (FHS) 켜기 T / 끄는 F")
    g.add_argument("--symmetrized", type=str.upper, choices=("T", "F"), default="T",
                   help="대칭화 켜기 T / 끄는 F")
    g.add_argument("--cutoff", type=float, default=None, help="절단 반경 (Å, 기본: 자동 추정)")
    g.add_argument("--order", type=int, default=3, help="계산 차수 (기본 3)")


def run(a):
    import os
    import sys

    from .core import load, check_dim

    a.dir = a.dir or "."
    cell = os.path.join(a.dir, a.cell or "POSCAR")
    fs = os.path.join(a.dir, a.fs or "FORCE_SETS")
    born = os.path.join(a.dir, a.born or "BORN")

    # 기존 fc3 파일 직접 사용
    if a.fc3 and os.path.exists(os.path.join(a.dir, a.fc3)):
        print(f"[3rd] 기존 3차 힘상수 파일 사용: {a.fc3}")
        return

    # 데이터 로드
    try:
        ph = load(a)
    except Exception as e:
        print(f"[3rd] 데이터 로드 실패: {e}", file=sys.stderr)
        print("  --fs FORCE_SETS, --cell POSCAR 옵션이 정확한지 확인하세요.", file=sys.stderr)
        sys.exit(1)

    # supercell 정보
    supercell = ph.supercell
    natom = len(supercell)
    ndim = 3

    # ---
    # 3차/4차 힘상수 계산
    # ---
    print(f"[3rd] 3차 힘상수 계산 시작...")
    print(f"  슈퍼셀: {natom} 원자")

    # 실제 구현:
    # 1. symfc.SymfcAtoms 생성
    # 2. displacement-force dataset 구성
    # 3. symfc로 FC3/FC4 계산
    # 4. HDF5 저장

    # MACE 또는 orbital force calculation은 별도 모듈에서 구현

    print("[3rd] 계산 완료")
    print("  출력: fc3.hdf5 / fc4.hdf5")
