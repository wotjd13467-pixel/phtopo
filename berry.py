"""Kubo Berry curvature 2D 지도 (그림용).

    Ω_n^γ(q) = −2 Im Σ_{m≠n} <n|∂D/∂k_α|m><m|∂D/∂k_β|n> / (W_n − W_m)²     (γ,α,β 순환, 단위 Å²)
노드 면 q[axis] = 0 과 옆 면 q[axis] = delta 를 훑는다.
  - q[axis] = 0    : 면 안 성분 — 노드에서 뻗어 나가면 source (χ=+1), 들어오면 sink (χ=−1)
  - q[axis] = delta: 면에 수직한 성분 — 노드마다 부호 있는 덩어리, 부호 = χ
--band 11-18 이면 밴드마다 따로. 그리는 것은 묶음 Ω = Σ_{n'≤n} Ω_n' — 단일 Ω_n 은 아래 밴드 (n−1/n) 근접 축퇴에서도
터져서 n/n+1 노드와 헷갈림. 묶음이면 그 항이 상쇄되고 n/n+1 monopole 만 남는다 (chi/box 와 같은 생각).
저장·그림 성분은 rlu flux 밀도 F^i = Ω·(b_j × b_k) (i,j,k 순환, 무차원) — 면 q_i = c 를 지나는 Berry flux 가
Σ F^i Δq_j Δq_k 라 비직교 격자에서도 그대로 Chern 수 (2π C) 가 되고, 화살표가 분수좌표 축 위에서 바른 방향.
(직교 격자면 F^i = Ω_i |b_j||b_k|.) 노드 짝은 대칭 궤도 중 이 면 위에 있는 것을 χ 부호 (det R) 와 같이 그림.
범위는 노드가 다 들어오게 자동 확장.
출력: berry_band<band>.npz, berry_band<band>.png
"""
import os
import time

import numpy as np

from .core import load, need_band
from .scan import Model

NAME, HELP = "berry", "Kubo Berry curvature 지도 → berry_band<band>.npz/.png"


def add_args(p):
    p.add_argument("--axis", type=int, default=3, choices=(1, 2, 3), help="노드 면 q[axis]=0. 기본 3")
    p.add_argument("--n", type=int, default=121, help="면 격자 n x n (홀수면 +1 — Γ 피함). 기본 121 (시간 ∝ n²)")
    p.add_argument("--qmax", type=float, default=None,
                   help="훑는 범위 −qmax..qmax (분수좌표). 기본: 0.25, 노드가 밖이면 노드+0.05 까지 (최대 0.5)")
    p.add_argument("--delta", type=float, default=0.01, help="옆 면 offset. 기본 0.01")
    p.add_argument("--nodes", default="nodes.txt", help="노드 파일 (있으면 표시). 기본 nodes.txt")


class Berry:
    def __init__(self, ph, h=1e-4):
        self.dm = ph.dynamical_matrix
        self.B = 2 * np.pi * np.linalg.inv(ph.primitive.cell)  # 열 = b1, b2, b3 -> q_cart = B @ q_frac
        self.Binv = np.linalg.inv(self.B)
        self.nb = 3 * len(ph.primitive)
        self.factor = ph.unit_conversion_factor
        self.h = h

    def D(self, k_cart):
        self.dm.run(self.Binv @ k_cart)
        return np.array(self.dm.dynamical_matrix)

    def omega(self, k_cart):
        """-> (freqs THz [nb], Omega [3, nb])"""
        W, V = np.linalg.eigh(self.D(k_cart))
        dD = []
        for a in range(3):
            e = np.zeros(3)
            e[a] = self.h
            dD.append((self.D(k_cart + e) - self.D(k_cart - e)) / (2 * self.h))
        M = [V.conj().T @ d @ V for d in dD]
        dW = W[:, None] - W[None, :]
        np.fill_diagonal(dW, np.inf)
        den = dW**2
        om = np.zeros((3, self.nb))
        for g, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)]):
            om[g] = -2.0 * np.sum(np.imag(M[a] * M[b].T) / den, axis=1)
        return np.sign(W) * np.sqrt(np.abs(W)) * self.factor, om


def scan_plane(bz, axis, c, us, vs, band):
    iu, iv = [i for i in range(3) if i != axis]
    Om = np.zeros((3, len(vs), len(us)))
    gap = np.zeros((len(vs), len(us)))
    frq = np.zeros_like(gap)
    t0 = time.time()
    for j, v in enumerate(vs):
        for i, u in enumerate(us):
            q = np.zeros(3)
            q[axis], q[iu], q[iv] = c, u, v
            f, om = bz.omega(bz.B @ q)
            Om[:, j, i] = om[:, : band + 1].sum(axis=1)  # 묶음 1..n: 아래 밴드끼리의 근접 축퇴 항은 상쇄
            gap[j, i] = f[band + 1] - f[band]
            frq[j, i] = f[band]
        if j % 20 == 0:
            el = time.time() - t0
            print(f"  row {j + 1}/{len(vs)}  {el:6.1f}s  eta {el / (j + 1) * (len(vs) - j - 1):6.1f}s", flush=True)
    return Om, gap, frq


def run(a):
    bands = need_band(a)
    ph = load(a)
    bz, m = Berry(ph), Model(ph)
    for b in bands:
        one(a, bz, m, b)


def one(a, bz, m, bnum):
    band, ax = bnum - 1, a.axis - 1
    nodes = np.atleast_2d(np.loadtxt(a.nodes, encoding="utf-8")) if os.path.exists(a.nodes) else np.zeros((0, 6))
    if nodes.shape[1] == 6:
        nodes = nodes[nodes[:, 5] == bnum]  # 이 밴드의 노드만 표시
    nodes = nodes[np.abs(nodes[:, a.axis - 1]) < 1e-4]  # 이 면 위 노드만 (scan 은 면 밖 노드도 줌)
    iu, iv = [i for i in range(3) if i != ax]
    marks = [(p, r[4] * sg if len(r) > 4 else 0) for r in nodes for p, sg in zip(*m.orbit(r[:3])) if abs(p[ax]) < 1e-4]
    qmax = a.qmax
    if qmax is None:  # 노드가 다 들어오게 (BZ 경계 0.5 까지)
        far = max((max(abs(p[iu]), abs(p[iv])) for p, _ in marks), default=0.0)
        qmax = min(0.5, max(0.25, far + 0.05))
    n = a.n + a.n % 2  # 짝수 → 대칭 격자가 Γ (q=0) 를 안 밟음. 밟으면 NAC·음향모드로 Ω 가 가짜로 터짐
    us = vs = np.linspace(-qmax, qmax, n)
    print(f"[1/2] q{a.axis} = 0 면 ({n}x{n}, ±{qmax:.3f})")
    Om0, gap0, frq0 = scan_plane(bz, ax, 0.0, us, vs, band)
    print(f"[2/2] q{a.axis} = {a.delta} 면")
    Omd, gapd, frqd = scan_plane(bz, ax, a.delta, us, vs, band)
    F = np.linalg.det(bz.B) * bz.Binv  # F^i = Ω·(b_j × b_k) = det(B) (B⁻¹ Ω)^i — Cartesian Ω → rlu flux 밀도
    Om0, Omd = np.einsum("ic,cyx->iyx", F, Om0), np.einsum("ic,cyx->iyx", F, Omd)

    f = f"berry_band{bnum}.npz"
    np.savez_compressed(f, us=us, vs=vs, axis=a.axis, delta=a.delta, band=bnum, Om0=Om0, gap0=gap0, frq0=frq0,
                        Omd=Omd, gapd=gapd, frqd=frqd, nodes=nodes, Bmat=bz.B)
    print(f"-> {f}")
    print(f"  q{a.axis}=0 면, 면에 수직한 성분 / 면 안 성분 (중앙값) = "
          f"{np.median(np.abs(Om0[ax])) / np.median(np.hypot(*np.delete(Om0, ax, 0))):.1e}   (0 근처여야 정상)")
    print(f"  q{a.axis}=0 면 위 최소 갭 = {gap0.min():.5f} THz")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ou, Ov = Om0[iu], Om0[iv]
    fig, axp = plt.subplots(figsize=(5.5, 5))
    im = axp.pcolormesh(us, vs, np.log10(np.hypot(Ou, Ov) + 1e-3), shading="auto", cmap="viridis")
    axp.streamplot(us, vs, Ou, Ov, color="w", density=1.2, linewidth=0.5, arrowsize=0.6)
    for p, c in marks:  # 궤도 중 이 면 위 짝 (χ 부호 = det R)
        col = "r" if c > 0.5 else "b" if c < -0.5 else "0.6"  # χ≈0 (글라이드 면 교차 등) 은 회색
        axp.plot(p[iu], p[iv], "o", ms=6, mec="k", color=col)
    fig.colorbar(im, ax=axp, label=r"log$_{10}$ |F| (Berry flux per rlu area)")
    axp.set_xlabel(f"q{iu + 1} (rlu)")
    axp.set_ylabel(f"q{iv + 1} (rlu)")
    axp.set_aspect("equal")
    axp.set_title(f"band {bnum}, q{a.axis} = 0  (red χ=+1, blue χ=−1)")
    fig.savefig(f"berry_band{bnum}.png", dpi=200, bbox_inches="tight")
    print(f"-> berry_band{bnum}.png")
