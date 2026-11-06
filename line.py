"""고대칭 경로 위 n/n+1 교차 찾기 (고유벡터 추적).

  경로: 기본 seekpath 자동 (좌표를 이 POSCAR 의 역격자로 변환·검산), 또는 --path 로 직접.
  각 구간을 --npts 점으로 → 이웃 점 고유벡터 겹침 |<u_i(k)|u_j(k+dk)>|² 로 밴드를 이음 (헝가리안 매칭).
  추적한 밴드가 정렬 순서 n, n+1 을 맞바꾸는 구간 = 교차 후보 → 1D 최소화로 갭 조임.
    갭 < 1e-5 THz 이면 교차 — scan 과 같은 점 판정: 점이면 Weyl (묶음 FHS χ), 아니면 nodal line 이 선을 지나는 점.
    갭이 남으면 회피 교차 (최소 갭 출력).
  선 전체에서 붙어 있는 구간 (강제 축퇴, 예 Pca2₁ k3=½ 위 홀수 밴드) 은 따로 표시.
출력: line.txt (표), line_nodes.txt (점 노드만, nodes.txt 형식 — chi / box 가 읽음)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar

from .core import all_bands, load
from .scan import TOL, Model, fold

NAME, HELP = "line", "고대칭 경로 위 교차 (고유벡터 추적) → line.txt, line_nodes.txt"


def add_args(p):
    p.add_argument("--path", help='직접 경로: "G X S | Z U" (이름만, | 에서 끊김) 또는 "G 0 0 0 Q 0.1 0.2 0" (좌표 직접), 섞어도 됨. 없으면 seekpath 자동')
    p.add_argument("--npts", type=int, default=201, help="구간당 점 수. 기본 201")
    p.add_argument("--radius", type=float, default=0.002, help="χ 구 반지름 (1/Å). 기본 0.002")


def seekpath_points(ph):
    """-> ({이름: 이 POSCAR 역격자 rlu 좌표}, [(시작, 끝) 이름]) — 이름은 이 구조의 Bravais 격자 기준 (seekpath)."""
    import seekpath

    p = ph.primitive
    r = seekpath.get_path((p.cell, p.scaled_positions, p.numbers), with_time_reversal=True)
    A_rot = np.asarray(p.cell) @ np.asarray(r["rotation_matrix"]).T  # seekpath 의 Cartesian 틀로 돌린 이 격자
    T = np.asarray(r["primitive_lattice"]) @ np.linalg.inv(A_rot)  # L_sp = T @ L
    assert np.abs(T - np.round(T)).max() < 1e-4 and abs(abs(np.linalg.det(T)) - 1) < 1e-6, f"seekpath 격자 변환이 정수가 아님:\n{T}"
    conv = np.linalg.inv(np.round(T)).T  # q_sp = T q → q = q_sp @ inv(T)^T
    return {k: np.asarray(v) @ conv for k, v in r["point_coords"].items()}, r["path"]


def seekpath_segments(ph):
    pc, path = seekpath_points(ph)
    return [(s, pc[s], e, pc[e]) for s, e in path]


def _is_num(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def parse_path(s, ph):
    """"G X S | Z U" (이름만 — seekpath 좌표) / "G 0 0 0 Q 0.1 0.2 0" (좌표 직접) / 섞어도 됨."""
    pc, segs = None, []
    for part in s.split("|"):
        t, pts, i = part.split(), [], 0
        while i < len(t):
            name = t[i]
            if _is_num(name):
                raise SystemExit(f"--path: 이름 자리에 숫자 {name!r} (받은 조각: {part!r}). 형식 '이름' 또는 '이름 q1 q2 q3'")
            xyz = t[i + 1 : i + 4]
            if len(xyz) == 3 and all(map(_is_num, xyz)):
                pts.append((name, np.array(xyz, float)))
                i += 4
                continue
            if pc is None:
                pc = seekpath_points(ph)[0]
            key = {k.upper(): k for k in pc}.get("GAMMA" if name.upper() in ("G", "Γ") else name.upper())
            if key is None:
                raise SystemExit(f"--path: 이 구조에는 점 {name!r} 이 없음. 쓸 수 있는 이름: {' '.join(sorted(pc))} (또는 '{name} q1 q2 q3' 로 좌표 직접)")
            print(f"  {name} = ({pc[key][0]:.4f}, {pc[key][1]:.4f}, {pc[key][2]:.4f})")
            pts.append((name, pc[key]))
            i += 1
        if len(pts) < 2:
            raise SystemExit(f"--path: 구간마다 점이 2개 이상 필요 (받은 조각: {part!r})")
        segs += [(a, qa, b, qb) for (a, qa), (b, qb) in zip(pts, pts[1:])]
    return segs


def track_swaps(ph, Q, b):
    """경로 Q (N,3) 에서 추적 밴드가 정렬 칸 b-1, b (0부터) 를 맞바꾸는 구간 인덱스 k (Q[k]–Q[k+1])."""
    ph.run_qpoints(Q, with_eigenvectors=True)
    V, F = ph.qpoints.eigenvectors, ph.qpoints.frequencies
    out = []
    for k in range(len(Q) - 1):
        O = np.abs(np.einsum("ai,aj->ij", V[k].conj(), V[k + 1])) ** 2
        _, perm = linear_sum_assignment(-O)  # 칸 i (k) → 칸 perm[i] (k+1)
        if perm[b - 1] == b and perm[b] == b - 1:
            out.append(k)
    return out, F[:, b] - F[:, b - 1]


def run(a):
    ph = load(a)
    m = Model(ph)
    segs = parse_path(a.path, ph) if a.path else seekpath_segments(ph)
    bands = all_bands(a, ph)
    L = [f"경로 {'직접' if a.path else 'seekpath'}: " + "  ".join(f"{s}-{e}" for s, _, e, _ in segs), "",
         f"{'band':>4s} {'구간':>8s}{'q1':>10s}{'q2':>10s}{'q3':>10s}{'gap':>10s}  판정"]
    nodes = []
    for b in bands:
        for s, qs, e, qe in segs:
            t = np.linspace(0, 1, a.npts)
            Q = qs + t[:, None] * (qe - qs)
            swaps, gap = track_swaps(ph, Q, b)
            if gap.max() < 1e-4:
                L.append(f"{b:4d} {s + '-' + e:>8s}  선 전체 축퇴 (강제)")
                continue
            for k in swaps:
                r = minimize_scalar(lambda x: m.gap(qs + x * (qe - qs), b), bounds=(t[k], t[k + 1]), method="bounded",
                                    options=dict(xatol=1e-12))
                q = fold(qs + r.x * (qe - qs))
                if r.fun > TOL:
                    kind = f"회피 교차 (최소 갭 {r.fun:.1e} THz)"
                elif not m.is_point(q, b):
                    kind = "nodal line 이 지나감 (점 아님)"
                else:
                    c, _ = m.chi(q, b, [], a.radius)
                    kind = f"Weyl χ = {c:+.3f}"
                    nodes.append((*q, m.freqs(q)[b - 1], c, b))
                end = f"  ({s if r.x < 1e-6 else e} 점 위)" if min(r.x, 1 - r.x) < 1e-6 else ""
                L.append(f"{b:4d} {s + '-' + e:>8s}{q[0]:10.5f}{q[1]:10.5f}{q[2]:10.5f}{r.fun:10.1e}  {kind}{end}")
    txt = "\n".join(L) + "\n"
    with open("line.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    with open("line_nodes.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("# pt line: 고대칭선 위 점 노드\n#       q1          q2          q3     f(THz)     chi  band\n")
        for r in nodes:
            f.write("%12.6f%12.6f%12.6f%10.4f%8.3f%6d\n" % r)
    print(txt + f"-> line.txt, line_nodes.txt (점 노드 {len(nodes)} 개)")
