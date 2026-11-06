"""포논 분산 — k 경로 자동 (seekpath, 좌표는 이 POSCAR 역격자 rlu 로 변환) 또는 --path.

  pt band                                  자동 경로, BORN 있으면 NAC 포함 → band.dat, band.png
  pt band --path "G X S | Z U"             점 이름만 (이 구조의 seekpath 좌표, | 에서 끊김). 이름은 격자마다 다름
  pt band --path "G 0 0 0  Q 0.1 0.2 0"    좌표 직접 (이름과 섞어도 됨)
  pt band --unit cm                        cm⁻¹
  pt band --loto F                         LO-TO 끔 (기본: BORN 있으면 켬)
  pt band --nodes nodes.txt                노드 파일 중 경로 위에 있는 것 (대칭 궤도 포함) 을 그림에 표시
계산은 phonopy run_band_structure — Γ 의 LO-TO 는 경로 방향으로 처리 (Gonze NAC).
가로축 = 경로 길이 [1/Å, 2π 포함]. 밴드는 주파수 오름차순 (--band 번호와 같음).
출력: <-o>.dat (x q1 q2 q3 f1..f3N, 구간 사이 빈 줄), <-o>.png
"""
import numpy as np

from .core import load, need_band, read_nodes, rec_rows
from .line import parse_path, seekpath_segments
from .scan import Model, fold

NAME, HELP = "band", "포논 분산 (seekpath 자동 경로 또는 --path) → band.dat, band.png"
THZ_TO_CM = 33.35641


def add_args(p):
    p.add_argument("--path", help='직접 경로: "G X S | Z U" (이름만) 또는 "G 0 0 0 Q 0.1 0.2 0" (좌표), 섞어도 됨. 없으면 seekpath 자동')
    p.add_argument("--npts", type=int, default=101, help="구간당 점 수. 기본 101")
    p.add_argument("--unit", choices=("thz", "cm"), default="thz", help="thz 또는 cm (cm⁻¹). 기본 thz")
    p.add_argument("--nodes", help="노드 파일 — 경로 위 노드를 표시")
    p.add_argument("-o", "--out", default="band", help="출력 이름 (band → band.dat, band.png). 기본 band")


def compute(ph, segs, npts):
    """-> (x 구간별, Q 구간별, freq 구간별 [THz], 구간 경계 x)."""
    paths = [qs + np.linspace(0, 1, npts)[:, None] * (qe - qs) for _, qs, _, qe in segs]
    ph.run_band_structure(paths)  # Γ 의 NAC 방향을 경로에서 잡아 줌 (run_qpoints 는 Γ 에서 LO-TO 가 빠짐)
    freqs = [np.array(f) for f in ph.band_structure.frequencies]
    rec = rec_rows(ph)
    xs, x0, ticks = [], 0.0, [0.0]
    for Q in paths:
        d = np.linalg.norm(np.diff(Q, axis=0) @ rec, axis=1)
        xs.append(x0 + np.concatenate([[0.0], np.cumsum(d)]))
        x0 = xs[-1][-1]
        ticks.append(x0)
    return xs, paths, freqs, ticks


def tick_labels(segs):
    nm = lambda s: "Γ" if s.upper() in ("G", "GAMMA") else s
    out = [nm(segs[0][0])]
    for (_, _, e, _), (s, _, _, _) in zip(segs, segs[1:]):
        out.append(nm(e) if e == s else f"{nm(e)}|{nm(s)}")
    return out + [nm(segs[-1][2])]


def on_path(m, nodes, bands, segs, xs, tol=1e-3):
    """노드 (궤도 포함) 가 구간 위면 [(x, f THz, band)]."""
    out = []
    for q, b in zip(nodes, bands):
        for p in m.orbit(q)[0]:
            for (_, qs, _, qe), x in zip(segs, xs):
                d = qe - qs
                t = np.dot(fold(p - qs), d) / np.dot(d, d)
                if 0 <= t <= 1 and np.abs(fold(p - qs - t * d)).max() < tol:
                    out.append((x[0] + t * (x[-1] - x[0]), m.freqs(q)[b - 1], int(b)))
    return out


def run(a):
    ph = load(a)
    segs = parse_path(a.path, ph) if a.path else seekpath_segments(ph)
    xs, paths, freqs, ticks = compute(ph, segs, a.npts)
    k = THZ_TO_CM if a.unit == "cm" else 1.0
    unit = "cm$^{-1}$" if a.unit == "cm" else "THz"
    labels = tick_labels(segs)
    with open(a.out + ".dat", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# pt band: {' '.join(labels)}  f [{a.unit}], x [1/Å]\n# 구간 경계 x: {' '.join(f'{t:.6f}' for t in ticks)}\n")
        for x, Q, f in zip(xs, paths, freqs):
            for i in range(len(x)):
                fh.write(f"{x[i]:10.6f} {Q[i, 0]:9.5f} {Q[i, 1]:9.5f} {Q[i, 2]:9.5f} " + " ".join(f"{v:11.5f}" for v in f[i] * k) + "\n")
            fh.write("\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for x, f in zip(xs, freqs):
        ax.plot(x, f * k, color="#1f4e8c", lw=0.9)
    for t in ticks:
        ax.axvline(t, color="0.6", lw=0.6)
    ax.axhline(0, color="0.6", lw=0.6)
    msg = ""
    if a.nodes:
        qs, fb = read_nodes(a.nodes)
        fb = np.full(len(qs), need_band(a)[0]) if fb is None else fb
        hits = on_path(Model(ph), qs, fb, segs, xs)
        for x, f, _ in hits:
            ax.plot(x, f * k, "o", ms=5, mfc="none", mec="r")
        msg = f", 경로 위 노드 {len(hits)} 개 (파일의 {len(qs)} 개 궤도 중)"
    ax.set_xticks(ticks, labels)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylabel(f"Frequency ({unit})")
    fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight")
    fall = np.concatenate(freqs)
    print(f"경로 {' '.join(labels)}  ({len(segs)} 구간), NAC {'on' if ph.nac_params else 'off'}")
    print(f"f {fall.min() * k:.3f} .. {fall.max() * k:.3f} {a.unit}{msg}")
    if fall.min() < -0.1:
        print(f"[경고] 허수 진동수 {fall.min():.3f} THz — 구조 불안정 또는 슈퍼셀·ASR 확인")
    print(f"-> {a.out}.dat, {a.out}.png")
