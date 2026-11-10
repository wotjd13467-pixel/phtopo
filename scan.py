"""BZ 전체 노드 탐색 — 대칭으로 줄인 균일 격자 → 좁히기 → 궤도로 중복 제거 → 묶음 FHS χ.

  1. spglib 기약 격자 (점군 + 시간반전, 반칸 이동이라 Γ·k=0,½ 면을 안 밟음). Pca2₁ 이면 1/8 만 계산.
     한 번 계산한 고유값으로 --band 의 모든 밴드쌍을 같이 본다.
  2. 강제 축퇴 면 감지: k_a = 0, ½ 면 전체에서 n/n+1 이 붙으면 (Pca2₁: k3=½ 의 홀수 밴드) 그 근처 후보 버림.
  3. 후보 = 갭의 국소 최소 (26 이웃), 갭 < --gmax, 기약 대표만 → 3D Nelder–Mead 로 갭 < 1e-5 THz 까지.
     대칭이 고정하는 면 (k_a=0,½) 은 2D 격자 4n² 로 따로 + 면 안 좁히기 → 3D 로는 옆 nodal line 에 묻히는 면 위 노드.
     찾은 노드 둘레 ±0.006 (6방향) 에서 다시 → 한 격자 칸 안에 몰린 노드.
  4. 점 판정: 작은 구 (r, r/2) 위 최소 갭이 ~0 (선·면이 구를 뚫음, 글라이드 면 교차 등) 이거나 r 에 비례하지 않으면 버림.
  5. 대칭 궤도 (R, −R) 로 중복 제거. 대표 = 궤도에서 (q1, q2, q3) 가 가장 큰 것 (Pca2₁: |q1|, |q2|, |q3|).
  6. χ = 묶음 FHS 구 (pt chi 와 같음). 반지름 = min(--radius, 가장 가까운 노드 거리/3).
  7. --verify: 검사 상자 (rlu) 의 FHS flux (pt box) 와 그 안의 찾은 노드 χ 합 (궤도 펼침) 을 비교.
     상자는 대칭에서 자동: 거울·글라이드 면 k_a=0 (nodal line) 이나 강제 축퇴 면 k_a=0 이 있으면 k_a ∈ [ε, ½−ε],
     없으면 [−½+ε, ½−ε] (Pca2₁ → [ε,½−ε]² × [−½+ε,½−ε]). 면이 상자 면을 비스듬히 지나면 (육방 등) 면 위 갭이 닫혀 '검사 불가'.
     다르면 놓친 노드 → 격자를 2배로 다시 (--mesh-max 까지). 상자 안 +1/−1 짝은 합이 0 이라 이 검사로 못 잡음.
     (슬라이스 Chern 은 쓰지 않음: k_a 에 수직한 거울/글라이드가 있으면 C(k_a) ≡ 0 이라 정보가 없음.)
출력: nodes.txt (q1 q2 q3 f chi band) — 궤도 대표만. chi / box / berry / simphony 가 그대로 읽는다.
원본 생각: C:\\Users\\techw\\Desktop\\cjs\\4_Archives\\KPS\\scripts\\core\\full_node_census.py (균일 격자는 노드를 놓친다 → 검사 필요)
"""
import os
import time

import numpy as np
import spglib
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize

from .box import box_chi
from .chi import chirality
from .core import THZ, all_bands, frame, load, rec_rows

NAME, HELP = "scan", "BZ 전체 노드 탐색 (대칭 기약 격자 → Nelder–Mead → 궤도 중복 제거 → FHS χ) → nodes.txt"
TOL = 1e-5  # 노드 판정 갭 (THz)


def add_args(p):
    p.add_argument("--mesh", type=int, default=24, help="균일 격자 n x n x n (짝수로 맞춤). 기본 24")
    p.add_argument("--gmax", type=float, default=0.3, help="후보 갭 상한 (THz). 기본 0.3")
    p.add_argument("--ncand", type=int, default=60, help="밴드쌍당 최대 후보 수 (갭 작은 순). 기본 60")
    p.add_argument("--radius", type=float, default=0.002, help="χ 구 반지름 상한 (1/Å). 기본 0.002")
    p.add_argument("--verify", action="store_true", help="대칭으로 정한 상자의 flux 로 놓친 노드 검사 (밴드당 ~1분)")
    p.add_argument("--mesh-max", type=int, default=48, help="--verify 불일치 시 격자를 2배씩 키우는 상한. 기본 48")
    p.add_argument("-o", "--out", default="nodes.txt", help="출력 파일 (밴드마다 저장, 검사 문제는 # 경고 줄). 기본 nodes.txt")


def fold(q):
    return q - np.round(q)


class Model:
    """phonopy 하나를 감싸 갭·대칭·χ 도구를 모은 것 (line 도 씀)."""

    def __init__(self, ph):
        self.ph, self.dm = ph, ph.dynamical_matrix
        self.rec = rec_rows(ph)
        self.Kinv = np.linalg.inv(self.rec)
        # primitive 격자 기준 회전 (ph.symmetry 는 슈퍼셀 기준 — 대각 아닌 슈퍼셀이면 기저가 다름). q(행) → q @ R
        R = np.array(ph.primitive_symmetry.pointgroup_operations)
        self.ops = np.concatenate([R, -R])  # 시간반전: q → −q (포논은 항상)
        self.opdet = np.concatenate([np.linalg.det(R), np.linalg.det(R)])  # χ 부호: 진회전 +, 거울 −, T 는 보존
        # 대칭이 제자리로 보내는 면 k_a = 0, ½ (연산이 k_a 만 뒤집음). 그 면 위 노드는 면 안에서 안정 (Pca2₁: q3=0 의 2₁T)
        # ponytail: 축에 나란한 면만 (분수좌표 연산이 k_a 하나만 뒤집을 때). 육방 등 비스듬한 거울면은 2D 격자 안 깜 — 3D 격자만.
        flip = lambda r, ax: np.allclose(r, np.diag(np.where(np.arange(3) == ax, -1, 1)))
        self.planes = sorted({(ax, c) for r in self.ops for ax in range(3) if flip(r, ax) for c in (0.0, 0.5)})
        self.mirror_axes = sorted({ax for r in R for ax in range(3) if flip(r, ax)})  # 공간 거울·글라이드 (T 없이) = nodal line 면

    def freqs(self, q):
        self.dm.run(np.asarray(q, float))
        e = np.linalg.eigvalsh(self.dm.dynamical_matrix)
        return np.sign(e) * np.sqrt(np.abs(e)) * THZ

    def gap(self, q, b):
        w = self.freqs(q)
        return w[b] - w[b - 1]

    def orbit(self, q):
        """-> (궤도 점들 (M,3) 접은 분수좌표, χ 부호 (M,)) — 같은 점은 하나만."""
        pts, sg = [], []
        for R, d in zip(self.ops, self.opdet):
            x = fold(np.asarray(q) @ R)
            if all(np.abs(fold(x - y)).max() > 1e-6 for y in pts):
                pts.append(x)
                sg.append(round(d))
        return np.array(pts), np.array(sg)

    def rep(self, q):
        pts, sg = self.orbit(q)
        i = max(range(len(pts)), key=lambda k: tuple(np.round(pts[k], 9)))
        return pts[i]

    def forced_planes(self, bands, n=20, seed=0):
        """{band: [(axis, c)]} — 면 전체에서 b/b+1 이 붙는 k_axis = c (0, ½)."""
        rng = np.random.default_rng(seed)
        out = {b: [] for b in bands}
        for ax in range(3):
            for c in (0.0, 0.5):
                Q = rng.random((n, 3)) - 0.5
                Q[:, ax] = c
                G = np.array([np.diff(self.freqs(q)) for q in Q]).max(0)
                for b in bands:
                    if G[b - 1] < 1e-4:
                        out[b].append((ax, c))
        return out

    def refine_in(self, q0, b, h, ax, c):
        """면 k_ax = c 안에서만 좁히기."""
        keep = [i for i in range(3) if i != ax]

        def full(y):
            q = np.empty(3)
            q[keep], q[ax] = y, c
            return q

        y = np.asarray(q0, float)[keep]
        for sc in (h, h / 5, h / 25, h / 125, h / 625):
            r = minimize(lambda y: self.gap(full(y), b), y, method="Nelder-Mead",
                         options=dict(xatol=1e-12, fatol=1e-17, maxiter=800, initial_simplex=y + np.vstack([0 * y, sc * np.eye(2)])))
            if r.fun < self.gap(full(y), b):
                y = np.array(r.x)
            if r.fun < 1e-9:
                break
        return full(y), self.gap(full(y), b)

    def refine(self, q0, b, h):
        x = np.asarray(q0, float)
        for sc in (h, h / 5, h / 25, h / 125, h / 625):
            r = minimize(lambda q: self.gap(q, b), x, method="Nelder-Mead",
                         options=dict(xatol=1e-12, fatol=1e-17, maxiter=800, initial_simplex=x + np.vstack([0 * x, sc * np.eye(3)])))
            if r.fun < self.gap(x, b):
                x = np.array(r.x)
            if r.fun < 1e-9:
                break
        return x, self.gap(x, b)

    def sphere_min_gap(self, qc, b, r, n=60):
        i = np.arange(n) + 0.5
        A, B = np.arccos(1 - 2 * i / n), np.pi * (1 + 5**0.5) * i
        U = np.stack([np.sin(A) * np.cos(B), np.sin(A) * np.sin(B), np.cos(A)], 1)

        def g(ang):
            d = np.array([np.sin(ang[0]) * np.cos(ang[1]), np.sin(ang[0]) * np.sin(ang[1]), np.cos(ang[0])])
            return self.gap(qc + r * d @ self.Kinv, b)

        best = min(range(n), key=lambda k: g([A[k], B[k]]))
        return minimize(g, [A[best], B[best]], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=300)).fun

    def is_point(self, qc, b, r=0.002):
        """구 (r, r/2) 위 최소 갭. 선·면이 구를 뚫으면 ≤ 5e-8 (Nelder–Mead 정밀도) 이고 r 에 무관.
        점이면 r 과 함께 자람 (선형 2배, 이웃 노드가 가까우면 덜 — example 17/18 덩어리 1.45배).
        크기 기준은 낮게: 17/18 노드는 한 방향이 매우 납작해 g(0.001) = 5e-6 THz."""
        g1, g2 = self.sphere_min_gap(qc, b, r), self.sphere_min_gap(qc, b, r / 2)
        return g2 > 1e-6 and g1 / g2 > 1.2

    def chi(self, qc, b, others, rmax):
        """others: 같은 밴드의 다른 노드 (궤도 펼친 것) — 반지름을 그 거리의 1/3 이하로."""
        d = [np.linalg.norm(fold(o - qc) @ self.rec) for o in others]
        d = [x for x in d if x > 1e-6]
        r = min([rmax] + [x / 3 for x in d])
        return chirality(self.ph, qc, b, r, 48, 96)[0], r


def ir_grid(ph, n):
    """-> (전체 격자 분수좌표 (n³,3), 기약 대표 인덱스 (k,), 전체→기약 위치 (n³,), 주소 → 인덱스 (n,n,n))."""
    p = ph.primitive
    mp, addr = spglib.get_ir_reciprocal_mesh([n] * 3, (p.cell, p.scaled_positions, p.numbers), is_shift=[1, 1, 1],
                                             is_time_reversal=True)
    q = fold((addr + 0.5) / n)
    uniq, inv = np.unique(mp, return_inverse=True)
    idx = np.empty((n, n, n), int)
    idx[tuple((addr % n).T)] = np.arange(len(addr))
    return q, uniq, inv, idx


def search(m, bands, n, a, log=print):
    """균일 격자 n³ → ({band: [대표 q]}, {band: 강제 축퇴 면})."""
    found, forced = {}, {}
    for b, reps, fp, _ in search_iter(m, bands, n, a, log):
        found[b], forced[b] = reps, fp
    return found, forced


def search_iter(m, bands, n, a, log=print):
    """밴드 하나 끝날 때마다 (band, [대표 q], 강제 축퇴 면, [선 위 점 대표 q]) — 전체 밴드는 몇 시간이라 중간 저장용.
    선 위 점 = 갭은 닫혔는데 점 판정에서 떨어진 것 (nodal line·면 위의 표본 점, 곡선 전체가 아님)."""
    n += n % 2
    q, uniq, inv, idx = ir_grid(m.ph, n)
    t0 = time.time()
    W = np.array([m.freqs(x) for x in q[uniq]])
    log(f"격자 {n}^3 = {len(q)} 점, 기약 {len(uniq)} 점 계산 {time.time() - t0:.0f}s")
    forced = m.forced_planes(bands)
    # 대칭 면은 2D 로 4배 촘촘히 — 면 위 노드가 옆 nodal line 과 3D 격자 한 칸 안에 같이 들면 후보가 안 생김
    N = 4 * n
    f = (np.arange(N) + 0.5) / N - 0.5
    pgrid = []
    for ax, c in m.planes:
        P = np.zeros((N, N, 3))
        P[..., [i for i in range(3) if i != ax]] = np.stack(np.meshgrid(f, f, indexing="ij"), -1)
        P[..., ax] = c
        R = np.round([m.rep(x) for x in P.reshape(-1, 3)], 9)
        ur, ui = np.unique(R, axis=0, return_index=True)
        _, inv_ = np.unique(R, axis=0, return_inverse=True)
        Wp = np.array([m.freqs(x) for x in P.reshape(-1, 3)[ui]])
        pgrid.append(((ax, c), P, Wp, inv_.ravel(), ui))
    log(f"대칭 면 {len(m.planes)} 개 x {N}^2 (기약 {sum(len(g[4]) for g in pgrid)} 점) {time.time() - t0:.0f}s")
    for b in bands:
        gfull = (W[:, b] - W[:, b - 1])[inv][idx]  # (n,n,n) 격자 위 갭
        isrep = np.zeros(len(q), bool)
        isrep[uniq] = True
        loc = (gfull <= minimum_filter(gfull, size=3, mode="wrap")) & (gfull < a.gmax)
        cand = sorted((g, i) for g, i in zip(gfull[loc], idx[loc]) if isrep[i]
                      and all(abs(fold(q[i][ax] - c)) > 1.5 / n for ax, c in forced[b]))
        cand = [i for _, i in cand[: a.ncand]]
        reps, lines, linepts = [], 0, []

        def take(x0, h, plane=None):
            nonlocal lines
            x, g = m.refine(x0, b, h) if plane is None else m.refine_in(x0, b, h, *plane)
            if g > TOL or any(abs(fold(x[ax] - c)) < 1e-3 for ax, c in forced[b]):
                return
            x = m.rep(fold(x))
            if np.abs(x).max() < 1e-3:  # Γ: 음향이 붙는 점 (χ=0). NAC 도 방향 의존이라 노드가 아님
                return
            if any(np.abs(fold(x - y)).max() < 1e-4 for y in reps):
                return
            if not m.is_point(x, b):
                lines += 1
                if all(np.abs(fold(x - y)).max() > 1e-3 for y in linepts):
                    linepts.append(x)
                return
            reps.append(x)

        for i in cand:
            take(q[i], 0.5 / n)
            # 3D 로 좁히면 근처 글라이드 면 nodal line 으로 빠질 수 있음 (example 11/12 (0.040, 0.425, 0)) → 가까운 대칭 면 안에서도
            for ax, c in m.planes:
                if abs(fold(q[i][ax] - c)) < 1.5 / n and (ax, c) not in forced[b]:
                    take(q[i], 0.5 / n, (ax, c))
        for pl, P, Wp, inv_, ui in pgrid:
            if pl in forced[b]:
                continue
            gp = (Wp[:, b] - Wp[:, b - 1])[inv_].reshape(N, N)
            first = np.zeros(N * N, bool)
            first[ui] = True  # 기약 대표만
            lp = ((gp <= minimum_filter(gp, size=3, mode="wrap")) & (gp < a.gmax)).ravel() & first
            l0, r0 = lines, len(reps)
            for j in np.flatnonzero(lp)[np.argsort(gp.ravel()[lp])][: a.ncand]:
                x0 = P.reshape(-1, 3)[j]
                if all(abs(fold(x0[ax] - c)) > 1.5 / N for ax, c in forced[b]):
                    take(x0, 0.5 / N, pl)
                # ponytail: 처음 3개가 다 선이면 nodal line 면 (거울·글라이드 — 그 안의 고립 Weyl 은 불안정) → 건너뜀.
                # 선과 점이 한 면에 같이 있는 공간군이면 이 줄을 지울 것 (느려질 뿐 결과는 같음)
                if lines - l0 >= 3 and len(reps) == r0:
                    break
        # 노드는 짝으로 생겨 몰려 있기 쉬움 (example 17/18: 0.005 안에 3개) — 격자 한 칸 안이라 후보가 하나뿐 → 둘레를 다시
        k = 0
        while k < len(reps):
            for d in np.vstack([np.eye(3), -np.eye(3)]) * 0.006:
                take(reps[k] + d, 0.002)
            k += 1
        fp = " ".join(f"k{ax + 1}={c}" for ax, c in forced[b])
        log(f"  {b}/{b + 1}: 후보 {len(cand)} → 점 노드 {len(reps)} (선/면 {lines} 버림{', 강제 축퇴 ' + fp if fp else ''})")
        yield b, reps, forced[b], linepts


def verify_box(m, forced, eps=0.01):
    """-> (lo, hi) 분수좌표. 거울·글라이드 면 k_a=0 (nodal line) 과 강제 축퇴 면은 상자 면에서 ε 이상 떨어뜨림."""
    lo = np.array([eps if ax in m.mirror_axes or (ax, 0.0) in forced else -0.5 + eps for ax in range(3)])
    return lo, np.full(3, 0.5 - eps)


def box_flux(m, b, lo, hi, step=0.02):
    """상자 묶음 FHS flux. 면 격자 간격 ≈ step Å⁻¹."""
    n = int(np.ceil(((hi - lo) * np.linalg.norm(m.rec, axis=1)).max() / step))
    return box_chi(m.ph, np.zeros(3), lo, hi, n, b - 1)


def run(a):
    ph = load(a)
    m = Model(ph)
    bands = all_bands(a, ph)
    n, todo, rows, lrows, bad = a.mesh, list(bands), [], [], {}
    lout = os.path.splitext(a.out)[0] + "_lines.txt"

    def plane_of(x):  # 그 점이 놓인 대칭 불변면 (거울·글라이드면 k_a=0,½ 이면 nodal line 이 흔함)
        return ",".join(f"k{ax + 1}={c:g}" for ax, c in m.planes if abs(fold(x[ax] - c)) < 1e-4) or "-"

    def save():
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# pt scan: BZ 전체, 궤도 대표만, 묶음 FHS chi, bands {bands}\n{frame(ph)}")
            fh.writelines(f"# 경고 밴드 {b}/{b + 1}: {msg}\n" for b, msg in sorted(bad.items()))
            fh.write("#       q1          q2          q3     f(THz)     chi  band\n")
            for r in sorted(rows, key=lambda r: (r[5], -r[0], -r[1])):
                fh.write("%12.6f%12.6f%12.6f%10.4f%8.3f%6d\n" % r)
        with open(lout, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# pt scan: nodal line/면 위 표본 점 — 갭 < 1e-5 THz 인데 고립점이 아님 (작은 구가 선·면에 뚫림).\n"
                     "# 궤도 대표만, 같은 선 위 여러 점 = 그 선의 표본 (곡선 전체 아님). plane = 놓인 대칭면 (- 면 일반 위치)\n"
                     f"{frame(ph)}#       q1          q2          q3     f(THz)  band  plane\n")
            for r in sorted(lrows, key=lambda r: (r[4], r[5], -r[0], -r[1])):
                fh.write("%12.6f%12.6f%12.6f%10.4f%6d  %s\n" % r)

    while todo:
        retry = []
        for b, reps, fp, linepts in search_iter(m, todo, n, a):
            lrows = [r for r in lrows if r[4] != b] + [(*x, m.freqs(x)[b - 1], b, plane_of(x)) for x in linepts]
            allpts = [p for x in reps for p in m.orbit(x)[0]]
            chis = []
            for x in reps:
                c, r = m.chi(x, b, allpts, a.radius)
                chis.append(c)
                rows.append((*x, m.freqs(x)[b - 1], c, b))
                print(f"     ({x[0]:.6f}, {x[1]:.6f}, {x[2]:.6f})  χ = {c:+.3f} (r={r:.4f})  궤도 {len(m.orbit(x)[0])}", flush=True)
            bad.pop(b, None)
            if a.verify:
                lo, hi = verify_box(m, fp)
                expect = sum(c * s for x, c in zip(reps, chis) for p, s in zip(*m.orbit(x)) if np.all((p > lo) & (p < hi)))
                C, g = box_flux(m, b, lo, hi)
                ok = abs(C - expect) < 0.3 and abs(C - round(C)) < 0.1
                box_s = " × ".join(f"[{l:+.2f},{h:+.2f}]" for l, h in zip(lo, hi))
                if g < 1e-3:  # 상자 면이 노드·선을 밟음 → flux 가 정수가 아님
                    bad[b] = f"검사 불가 (상자 면 위 갭 {g:.1e} THz)"
                    print(f"  {b}/{b + 1} 검사 불가: 상자 {box_s} 면 위 갭 {g:.1e} THz (면을 지나는 nodal line·축퇴 면)", flush=True)
                else:
                    print(f"  {b}/{b + 1} 검사: 상자 {box_s} flux {C:+.3f} (면 위 최소 갭 {g:.1e}) vs 찾은 노드 합 {expect:+.0f}"
                          f"  {'OK' if ok else '<- 불일치'}", flush=True)
                    if not ok:
                        bad[b] = f"불일치 (상자 flux {C:+.3f}, 찾은 합 {expect:+.0f}, 격자 {n}³) — 놓친 노드 가능"
                        if 2 * n <= a.mesh_max:
                            rows = [r for r in rows if r[5] != b]
                            retry.append(b)
            save()
        todo = retry
        if todo:
            n *= 2
            print(f"불일치 밴드 {todo} → 격자 {n}^3 로 다시", flush=True)
    print(f"-> {a.out}  (노드 {len(rows)} 개, 궤도 대표), {lout}  (nodal line·면 표본 점 {len(lrows)} 개)")
    for b, msg in sorted(bad.items()):
        print(f"   ⚠ {b}/{b + 1}: {msg}")
