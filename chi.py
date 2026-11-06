"""묶음(누적) FHS 구면 Chern number (phonopy 고유벡터만, Simphony 없이).

  C = (1/2π) Σ_plaquette Im log Π_4변 det M,   M_ij = <u_i(k)|u_j(k')>,  i, j ∈ 아래 밴드 1..n   (부호는 Kubo 규약)
det 는 묶음 1..n 내부의 교차·혼합에 불변 → 아래 밴드끼리 붙어도 상관없고, 구 위에서 n/n+1 갭만 열려 있으면 정수.
그래서 고유벡터 추적이 필요 없다 (추적은 라벨링 도구지 topology 도구가 아님 — HfO2_weyl_phonon.wiki eigenvector-tracking).
구 안의 n/n+1 노드 χ 합이 나온다. 고립 판별: gap(2r)/gap(r) ≈ 2 (선형) 이면 고립. 아니면 반지름 바꾸거나 pt box.
밴드: nodes.txt 의 band 열 (--band 주면 그 밴드 노드만). band 열 없는 파일이면 --band 필수.
출력: chirality.txt
"""
import numpy as np

from .core import load, need_band, overlap_det, read_nodes, rec_rows

NAME, HELP = "chi", "묶음 FHS 구면 Chern number → chirality.txt"


def add_args(p):
    p.add_argument("--radius", type=float, default=0.004, help="구 반지름 (1/Å). 기본 0.004")
    p.add_argument("--ntheta", type=int, default=48)
    p.add_argument("--nphi", type=int, default=96)
    p.add_argument("--nodes", default="nodes.txt", help="노드 파일 (nodes.txt 또는 Simphony Nodes.dat). 기본 nodes.txt")


def sphere(center, radius, nt, npv, rec):
    th = np.linspace(0, np.pi, nt + 1)
    phv = np.linspace(0, 2 * np.pi, npv, endpoint=False)
    T, P = np.meshgrid(th, phv, indexing="ij")
    cart = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1) * radius
    return np.asarray(center) + cart @ np.linalg.inv(rec)


def chirality(ph, center, band, radius, nt, npv):
    """band = 1-based 아래 밴드 n. 밴드 1..n 묶음 FHS. -> (C, 구 위 최소 n/n+1 갭 THz)."""
    Q = sphere(center, radius, nt, npv, rec_rows(ph))
    shp = Q.shape[:2]
    ph.run_qpoints(Q.reshape(-1, 3), with_eigenvectors=True)
    ev = ph.qpoints.eigenvectors.reshape(*shp, *ph.qpoints.eigenvectors.shape[-2:])
    fr = ph.qpoints.frequencies.reshape(*shp, -1)
    u = ev[..., :band]  # (nt+1, nphi, 3N, n) — 열 = 밴드
    u1, u2 = u[:-1], u[1:]
    u3, u4 = np.roll(u[1:], -1, axis=1), np.roll(u[:-1], -1, axis=1)
    prod = overlap_det(u1, u2) * overlap_det(u2, u3) * overlap_det(u3, u4) * overlap_det(u4, u1)
    C = -np.sum(np.angle(prod)) / (2 * np.pi)  # 사각형 위상 합 = −(Berry flux) → Kubo 규약
    return C, (fr[..., band] - fr[..., band - 1]).min()


def run(a):
    ph = load(a)
    qs, fb = read_nodes(a.nodes)
    if fb is None:
        fb = np.full(len(qs), need_band(a)[0])
        if a.band and len(a.band) > 1:
            raise SystemExit(f"{a.nodes} 에 band 열이 없음 — --band 는 하나만 (예 --band 17)")
    elif a.band:
        keep = np.isin(fb, a.band)
        qs, fb = qs[keep], fb[keep]
    L = [f"묶음 FHS 구면 Chern number (밴드 1..n), 격자 {a.ntheta}x{a.nphi}, 반지름 {a.radius} 1/Ang", "",
         f"{'k1':>11s}{'k2':>11s}{'k3':>11s}{'band':>6s}{'chi':>8s}{'gap n/n+1':>11s}"]
    tot = {}
    for q, b in zip(qs, fb):
        C, g = chirality(ph, q, b, a.radius, a.ntheta, a.nphi)
        _, g2 = chirality(ph, q, b, 2 * a.radius, 16, 32)
        ratio = g2 / g if g > 0 else np.inf
        isolated = 1.5 < ratio < 2.6
        if isolated:
            tot[b] = tot.get(b, 0.0) + C
        warn = "" if isolated else f"  <- 고립 아님 의심 (gap(2r)/gap(r)={ratio:.2f}, 고립이면 ~2). 반지름 바꿔 확인, 안 되면 pt box"
        L.append(f"{q[0]:11.6f}{q[1]:11.6f}{q[2]:11.6f}{b:6d}{C:8.3f}{g:11.5f}{warn}")
    L += ["", "밴드별 χ 합 (고립 노드만, 파일의 노드만 — 대칭 짝 포함 BZ 전체 합은 항상 0):"]
    L += [f"  {b}/{b + 1}: {t:+.3f}" for b, t in sorted(tot.items())]
    txt = "\n".join(L) + "\n"
    with open("chirality.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    print(txt + "-> chirality.txt")
