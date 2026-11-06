"""NPHE P_ab 의 주파수 분해 P(ν) (0.1 THz bin) → npz — pt kappa --tau omega 의 입력.

bin 을 다 더하면 pt nlhall 의 P_ab 와 같다. --ibz 로 대칭 기약 격자 (Pca2₁ 1/8).
3×3 전 성분을 H9 [shift=1, η, T, bin, 3, 3] (meV·kB², Cartesian x y z) 로 저장 → kappa 는 이것을 씀 (공간군·셀 방향 무관).
--comps 는 화면 요약과 옛 형식 키 (H, Hs = 그 두 성분) 용.
본체 phtopo/spec_vec.py = C:\\Users\\techw\\Desktop\\cjs\\1_Projects\\NPHE_HfO2\\nphe\\scripts\\46_spec_vec.py 사본 (수정 없음).
"""
import os
import sys

import numpy as np

from . import spec_vec as sv
from .core import load

NAME, HELP = "spectral", "NPHE P(ν) 주파수 분해 (3×3 전 성분) → spectral_P_nq60.npz (kappa --tau omega 입력)"


def add_args(p):
    p.add_argument("--nq", type=int, default=60, help="격자 nq³. 기본 60")
    p.add_argument("--eta", type=float, nargs="+", default=[0.0, 0.08, 0.04], help="Kubo 정규화 η (THz). 기본 0 0.08 0.04")
    p.add_argument("-T", dest="T", default="10:300:10", help="온도 시작:끝:간격. 기본 10:300:10")
    p.add_argument("--comps", nargs=2, default=["xz", "zx"], help="화면 요약 성분 두 개 (저장은 3×3 전부). 기본 xz zx")
    p.add_argument("--ibz", action="store_true")
    p.add_argument("--no-nac", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block", type=int, default=2000)
    p.add_argument("-o", "--out", default="spectral_P_nq60.npz", help="기본 spectral_P_nq60.npz (kappa 기본 --spec 이름)")


def run(a):
    sv.V.load = lambda d, sc, nac: load(a, nac=nac)  # 폴더 대신 pt 공통 입력
    cap, orig = {}, sv.spec_oneshift

    def grab(*args, **kw):  # 본체는 3×3 을 다 계산하고 --comps 두 개만 저장 → 전부 가로챔
        cap["H5"], nk = orig(*args, **kw)
        return cap["H5"], nk

    lo, hi, dt = a.T.split(":")
    argv = [os.path.dirname(os.path.abspath(a.cell)), "--nq", a.nq, "--etas", *a.eta, "--tmin", lo, "--tmax", hi, "--dt", dt,
            "--comps", *a.comps, "--seed", a.seed, "--block", a.block, "--out", a.out]
    argv += ["--ibz"] * a.ibz + ["--no-nac"] * a.no_nac
    old = sys.argv
    sys.argv = ["pt spectral"] + [str(x) for x in argv]
    sv.spec_oneshift = grab
    try:
        sv.main()
    finally:
        sys.argv, sv.spec_oneshift = old, orig
    # 원본 46 은 Hs 를 [성분, shift, ...] 로 저장 (40/41 은 [shift, 성분, ...]) → 여기서 [1, 성분, η, T, bin]. 원본 정리 때 46 쪽을 고칠 것.
    d = dict(np.load(a.out))
    d["Hs"] = d["H"][None]
    d["H9"] = cap["H5"][None] * 1000  # [1, η, T, bin, 3, 3] meV·kB²
    np.savez(a.out, **d)
