"""닫힌 상자 곡면 위 FHS 로 정확한 정수 chirality (노드가 몰려 있거나 납작할 때).

묶음(누적) FHS: det <u_i|u_j> (i,j = 밴드 1..n) — 아래 밴드끼리 교차해도 불변, 곡면 위 n/n+1 갭만 열려 있으면 정수.
상자 안 n/n+1 노드들의 χ 합이 나온다 → 상자 크기·위치를 바꿔 노드별로 분리.
상자 = 중심 + [lo, hi] — 분수좌표 (rlu, b1 b2 b3 방향) 평행육면체. 비직교 격자에서도 면이 닫힌다. 부호는 Kubo 규약.
예 (example 데이터, 노드 3개가 0.006 Å⁻¹ ≈ 0.005 rlu 안에 몰림):
  pt box                                                       # ±0.016 큰 상자: 세 노드 합
  pt box --lo -0.0033 -0.0032 -0.0024 --hi 0.0033 0.0032 0.0024   # 면 위 노드만
"""
import numpy as np

from .core import THZ, load, need_band, overlap_det, read_nodes, rec_rows

NAME, HELP = "box", "FHS 상자 곡면 χ (노드가 몰려 있을 때)"


def add_args(p):
    p.add_argument("--center", type=float, nargs=3, help="상자 중심 q1 q2 q3 (분수좌표). 없으면 --nodes 사용")
    p.add_argument("--nodes", default="nodes.txt", help="노드 파일. 기본 nodes.txt")
    p.add_argument("--node", type=int, default=1, help="노드 파일의 몇 번째 노드 (1부터). 밴드는 그 줄의 band 열. 기본 1")
    p.add_argument("--lo", type=float, nargs=3, default=(-0.016, -0.016, -0.016), help="상자 아래 끝 (중심 기준 분수좌표 rlu). 기본 -0.016")
    p.add_argument("--hi", type=float, nargs=3, default=(0.016, 0.016, 0.016), help="상자 위 끝 (rlu). 기본 0.016")
    p.add_argument("--n", type=int, nargs="+", default=(40, 80), help="면당 격자 수 (여러 개면 수렴 확인). 기본 40 80")


def box_chi(ph, q0, lo, hi, n=60, NB=16):
    """q0, lo, hi: 분수좌표 (rlu). NB: 아래 밴드 (0부터). 묶음 = 밴드 0..NB."""
    dm = ph.dynamical_matrix
    rec = rec_rows(ph)
    q0, lo, hi = (np.asarray(x, float) for x in (q0, lo, hi))
    gmin = np.inf

    def vec(q):
        nonlocal gmin
        dm.run(q)
        M = np.array(dm.dynamical_matrix)
        w, V = np.linalg.eigh((M + M.conj().T) / 2)
        f = np.sign(w) * np.sqrt(np.abs(w)) * THZ
        gmin = min(gmin, f[NB + 1] - f[NB])
        return V[:, : NB + 1]

    tot = 0.0
    for ax in range(3):
        u_, v_ = [i for i in range(3) if i != ax]
        for side, val in ((+1, hi[ax]), (-1, lo[ax])):
            s = np.linspace(lo[u_], hi[u_], n + 1)
            t = np.linspace(lo[v_], hi[v_], n + 1)
            flip = np.dot(np.cross(rec[u_], rec[v_]), side * rec[ax]) < 0  # 바깥 방향 (Cartesian b 로 판정 — 비직교·왼손 격자도)
            U = np.empty((n + 1, n + 1), dtype=object)
            for i, si in enumerate(s):
                for j, tj in enumerate(t):
                    q = q0.copy()
                    q[ax], q[u_], q[v_] = q0[ax] + val, q0[u_] + si, q0[v_] + tj
                    U[i, j] = vec(q)
            ph_sum = 0.0
            for i in range(n):
                for j in range(n):
                    p1, p2, p3, p4 = U[i, j], U[i + 1, j], U[i + 1, j + 1], U[i, j + 1]
                    ph_sum += np.angle(overlap_det(p1, p2) * overlap_det(p2, p3) * overlap_det(p3, p4) * overlap_det(p4, p1))
            tot += -ph_sum if flip else ph_sum
    # 사각형 위상 합 = −(Berry flux) → Kubo 부호 (c01 노드 Kubo χ=+1 로 보정)
    return -tot / (2 * np.pi), gmin


def run(a):
    if a.center:
        q0, b = a.center, need_band(a)[0]
    else:
        qs, fb = read_nodes(a.nodes)
        q0 = qs[a.node - 1]
        b = need_band(a)[0] if fb is None else int(fb[a.node - 1])
    ph = load(a)
    print(f"중심 q = ({q0[0]:.5f}, {q0[1]:.5f}, {q0[2]:.5f}), 상자 lo {tuple(a.lo)} hi {tuple(a.hi)} rlu, 밴드 {b}/{b + 1} (묶음 1..{b})")
    for n in a.n:
        c, g = box_chi(ph, q0, a.lo, a.hi, n, b - 1)
        print(f"n={n:3d}  chi = {c:+.4f}   곡면 위 최소 갭 {g:.1e} THz", flush=True)
