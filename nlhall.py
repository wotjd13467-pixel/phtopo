"""비선형 포논 홀 응답 P_ab(T) [meV·k_B²] — NPHE (Luo, PRB 107, L241107 (2023)).

    P_ab = 1/(V T²) Σ_{n,q} E³ (∂ρ₀/∂q_a) Ω_{n,b}
계산 본체는 phtopo/nphe_vec.py = C:\\Users\\techw\\Desktop\\cjs\\1_Projects\\NPHE_HfO2\\nphe\\scripts\\nphe_vec.py 사본 (2026-11-03, 수정 없음).
검증 (원본): PbTiO₃ nq100 → P_xy peak 0.476 (η 0.08), Luo 0.47–0.5.  HfO₂ 수렴은 nq200–300 (η 0.08).
η = Kubo 분모 정규화 (THz). η=0 은 NAC 때문에 폭주 — 숫자로 쓰지 말 것. shift 평균 → mean ± SE.
--ibz: 대칭 기약 격자 (Pca2₁ 1/8). 같은 출력 파일이 있으면 shift 체크포인트를 이어받는다.
출력: nphe_<폴더>_nq<nq>_sc<abc>.npz  (P[shift, η, T, a, b] eV·k_B²)
"""
import os

import numpy as np

from . import nphe_vec as nv
from .core import load

NAME, HELP = "nlhall", "비선형 포논 홀 P_ab(T) (NPHE) → nphe_*.npz"


def t_arg(s):
    lo, hi, dt = (float(x) for x in s.split(":"))
    return np.arange(lo, hi + 1e-9, dt)


def add_args(p):
    p.add_argument("--nq", type=int, nargs="+", default=[100], help="격자 nq³ (여러 개면 수렴표). 기본 100")
    p.add_argument("--shift", type=int, default=2, help="격자 shift 개수 (SE 추정). 기본 2")
    p.add_argument("--eta", type=float, nargs="+", default=[0.0, 0.08, 0.04], help="Kubo 정규화 η (THz), 0 = bare. 기본 0 0.08 0.04")
    p.add_argument("-T", dest="T", type=t_arg, default=t_arg("10:300:10"), help="온도 시작:끝:간격 (K). 기본 10:300:10")
    p.add_argument("--ibz", action="store_true", help="대칭 기약 격자 (Pca2₁ 1/8)")
    p.add_argument("--no-nac", action="store_true", help="BORN 없이")
    p.add_argument("--block", type=int, default=2000, help="한 번에 처리할 k 점 수 (메모리). 기본 2000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", help="출력 npz (기본 nphe_<폴더>_nq<nq>_sc<abc>.npz)")


def run(a):
    ph = load(a, nac=not a.no_nac)
    L = nv.Lattice(ph)
    a.supercell = list(np.ravel(a.dim)) if np.size(a.dim) == 3 else a.dim  # run_nq 가 파일에 기록
    d = os.path.dirname(os.path.abspath(a.cell))
    etas, TS = list(a.eta), a.T
    print(f"{L.sg}, 원자 {L.n // 3}, 밴드 {L.n}, V={L.Vcell:.3f} Å³, NAC={'on' if L.nac else 'off'}, "
          f"nq={a.nq}, shift {a.shift}, η={etas}, T={TS[0]:.0f}-{TS[-1]:.0f} K, D(q) {nv.BACKEND}", flush=True)
    res = {}
    for nq in a.nq:
        tag = f"{os.path.basename(d)}_nq{nq}_sc{''.join(map(str, np.ravel(a.dim)))}"
        out = (a.out if len(a.nq) == 1 else a.out.replace(".npz", "") + f"_nq{nq}.npz") if a.out else f"nphe_{tag}.npz"
        res[nq] = nv.run_nq(L, nq, a, etas, TS, out, d)
    if len(a.nq) > 1:  # 원본 main 과 같은 수렴표
        allow, _ = nv.allowed_components(L.Rc)
        print("\n[nq 수렴표] 허용성분 대칭평균 peak [meV*kB^2] (@T)")
        for i, j in allow:
            for ie, e in enumerate(etas):
                row = []
                for nq, P in res.items():
                    ms = nv.sym_project(L.Rc, P.mean(0) * 1000)[ie, :, i, j]
                    ip = int(np.argmax(np.abs(ms)))
                    row.append(f"nq{nq}: {ms[ip]:+.4f}@{TS[ip]:.0f}K")
                print(f"  P_{nv.AX[i]}{nv.AX[j]} η={e:<5} " + "   ".join(row))
