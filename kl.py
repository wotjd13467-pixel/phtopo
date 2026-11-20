"""격자 열전도도 κ_L(T) — phono3py RTA (계산은 phono3py, pt tau 와 같은 호출 tau.rta).

fc2: FORCE_SETS 또는 --fc fc2.hdf5 (FORCE_SETS 없고 fc2.hdf5 있으면 자동), BORN 있으면 자동. fc3: --fc3 (기본 --dir 의 fc3.hdf5).
동위원소 산란은 기본으로 켬 (phono3py 자연 존재비). 끄려면 --isotope F.
출력: 화면 표 + kl.csv (T_K, κ_xx yy zz yz xz xy [W/mK]).
"""
import os
import time

import numpy as np

from .nlhall import t_arg
from .tau import rta

NAME, HELP = "kl", "격자 열전도도 κ_L(T) — phono3py RTA, 동위원소 기본 켬"
COMP = ["xx", "yy", "zz", "yz", "xz", "xy"]  # phono3py kappa 순서


def add_args(p):
    p.add_argument("--fc3", default="fc3.hdf5", help="fc3 hdf5 (phono3py, --dir 기준). 기본 fc3.hdf5")
    p.add_argument("--fc3-dim", nargs="+", default=None, help="fc3 를 만든 슈퍼셀 (예 2 2 2). 기본 = --d")
    p.add_argument("--mesh", type=int, default=11, help="phono3py mesh n³ — κ 수렴 확인 필수 (11, 13, 15 …). 기본 11")
    p.add_argument("-T", dest="T", type=t_arg, default=t_arg("100:800:100"), help="온도 시작:끝:간격. 기본 100:800:100")
    p.add_argument("--isotope", type=str.upper, choices=("T", "F"), nargs="?", const="T", default="T",
                   help="동위원소 산란 T 켬 / F 끔. 기본 T (실제 시료는 동위원소가 섞여 있으니 켜는 게 맞음)")
    p.add_argument("--boundary", type=float, default=None, help="경계 산란 평균자유행로 L [μm]. 기본 없음")
    p.add_argument("-o", "--out", default="kl.csv", help="기본 kl.csv")


def run(a):
    a.isotope = a.isotope == "T"  # rta 는 bool 로 봄
    a.fc3 = os.path.join(a.dir, a.fc3)
    if not os.path.exists(a.fc3):
        raise SystemExit(f"{a.fc3} 없음 — --fc3 로 fc3 hdf5 를 주세요")
    t0 = time.time()
    kappa = rta(a, a.mesh, a.T)[3]  # [T, 6] W/mK
    np.savetxt(a.out, np.column_stack([a.T, kappa]), delimiter=",", fmt="%.6g", header=",".join(["T_K"] + ["k_" + c for c in COMP]), comments="")
    extra = ("동위원소 켬" if a.isotope else "동위원소 끔") + (f", 경계 {a.boundary} μm" if a.boundary else "")
    print(f"[fc2 {a.fc or a.fs} (슈퍼셀 {a.dim}) + fc3 {a.fc3}, mesh {a.mesh}^3, RTA, {extra}] ({time.time() - t0:.0f}s)")
    print(f"{'T(K)':>6} " + " ".join(f"{'κ_' + c:>9}" for c in COMP) + "   W/mK")
    for T, k in zip(a.T, kappa):
        print(f"{T:>6.0f} " + " ".join(f"{x:>9.3f}" for x in k))
    print("->", a.out)
