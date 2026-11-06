"""면 하나 위 Weyl 노드 찾기 + 묶음 FHS 구로 chirality (빠른 면 탐색. BZ 전체는 pt scan).

  - 면 q[axis] = 0 (기본 axis=3. Pca2₁ 이면 2₁T 가 Weyl 을 이 면에 고정) 전체 [−½, ½)² 를 반칸 이동 격자로.
  - 격자 국소 최소에서 면 안 Nelder–Mead 로 갭 < 1e-5 THz 까지 조임.
  - 대칭 궤도로 중복 제거 (이 면 위 짝 중 (q1, q2, q3) 가 가장 큰 것), 점 판정은 pt scan 과 같음
    (작은 구 위 최소 갭 — 거울·글라이드 면의 nodal line, 강제 축퇴 면을 버림). 공간군 가정 없음.
  - χ = 노드 둘레 Cartesian 구 위 묶음 FHS (pt chi 와 같음 — 단일 밴드 Kubo 는 납작한 노드에서 정수가 안 나옴, example 17: −0.19).
    반지름 = min(0.002 Å⁻¹, 가장 가까운 짝 노드 거리/3, 가장 가까운 거울·글라이드 면 거리/3).
--band 11-18 이면 밴드쌍 11/12 ... 18/19 각각. 밴드 번호 = 주파수 오름차순 (노드 = 정렬된 n, n+1 이 닿는 점이 정의라 추적 불필요).
출력: nodes.txt (q1 q2 q3 f(THz) chi band) — chi / box / berry 가 읽는다.
노드가 몰려 있으면 구 flux 가 정수가 안 나온다 → pt chi (작은 구), pt box (상자).
"""
import numpy as np
from scipy.ndimage import minimum_filter

from .chi import chirality
from .core import frame, load, need_band, rec_rows
from .scan import TOL, Model, fold

NAME, HELP = "nodes", "면 위 노드 탐색 (갭 격자 → Nelder–Mead) + 묶음 FHS χ → nodes.txt"


def add_args(p):
    p.add_argument("--axis", type=int, default=3, choices=(1, 2, 3), help="노드가 놓인 면 q[axis]=0. 기본 3")
    p.add_argument("--no-nac", action="store_true", help="BORN 없이 (NAC 끔)")
    p.add_argument("-o", "--out", default="nodes.txt", help="출력 파일. 기본 nodes.txt")


def run(a):
    bands = need_band(a)
    ph = load(a, nac=not a.no_nac)
    rows, m = [], Model(ph)
    for b in bands:
        rows += find(ph, b, a.axis, m)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# plane q{a.axis}=0, FHS sphere chi, bands {bands}\n{frame(ph)}"
                 "#       q1          q2          q3     f(THz)     chi  band\n")
        for r in rows:
            fh.write("%12.6f%12.6f%12.6f%10.4f%8.3f%6d\n" % r)
    print(open(a.out, encoding="utf-8").read() if rows else "노드 없음")
    print(f"-> {a.out}")


def find(ph, band, axis, m, n=90):
    """밴드쌍 (band, band+1) 의 면 q[axis]=0 위 노드 -> [(q1, q2, q3, f, chi, band)]."""
    nb = band - 1
    ax = axis - 1
    iu, iv = [i for i in range(3) if i != ax]
    rec = rec_rows(ph)
    Kinv = np.linalg.inv(rec)  # Cartesian -> 분수

    def inplane_rep(q):  # 궤도 중 이 면 위에 있는 짝의 대표
        return max((p for p in m.orbit(q)[0] if abs(p[ax]) < 1e-9), key=lambda p: tuple(np.round(p, 9)))

    print(f"밴드 {band}/{band + 1}, 면 q{axis}=0, 갭 격자 {n}x{n} ...", flush=True)
    f = (np.arange(n) + 0.5) / n - 0.5  # 반칸 이동 — Γ·대칭 면을 안 밟음
    P = np.zeros((n, n, 3))
    P[..., iu], P[..., iv] = np.meshgrid(f, f, indexing="ij")
    Gr = np.array([[m.gap(q, band) for q in row] for row in P])
    loc = (Gr <= minimum_filter(Gr, size=3, mode="wrap")) & (Gr < 0.15)
    seeds = sorted((Gr[i, j], i, j) for i, j in zip(*np.nonzero(loc))
                   if np.allclose(inplane_rep(P[i, j]), fold(P[i, j])))  # 대표 칸만
    print(f"후보 (국소 최소, 갭 < 0.15 THz, 대칭 대표) {len(seeds)} 개 -> Nelder-Mead", flush=True)
    found = []
    for _, i, j in seeds[:25]:
        x, g = m.refine_in(P[i, j], band, 0.5 / n, ax, 0.0)
        if g > TOL:
            continue
        x = inplane_rep(fold(x))
        if np.abs(x).max() < 1e-3:  # Γ: 음향이 붙는 점, 노드 아님
            continue
        if any(np.abs(fold(x - y)).max() < 1e-4 for y in found) or not m.is_point(x, band):
            continue
        found.append(x)

    rows = []
    allpts = [p for x in found for p in m.orbit(x)[0]]
    for x in found:
        d = [np.linalg.norm(fold(o - x) @ rec) for o in allpts]
        d += [abs(fold(x[k] - c)) / np.linalg.norm(Kinv[:, k]) for k in m.mirror_axes for c in (0.0, 0.5)]  # 면까지 Å⁻¹
        r = min([2e-3] + [v / 3 for v in d if v > 1e-6])
        rows.append((*x, m.freqs(x)[nb], chirality(ph, x, band, r, 48, 96)[0], band))
    print(f"  밴드 {band}/{band + 1}: 노드 {len(rows)} 개", flush=True)
    return rows
