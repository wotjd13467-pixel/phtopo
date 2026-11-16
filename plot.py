"""pt nlhall / pt pprime 결과 npz → 그림 png 또는 csv (계산 없음, 파일만 읽음).

  pt plot --nlhall                  현재 폴더 (또는 --dir 폴더) 의 가장 최근 nphe_*.npz
  pt plot --nlhall nphe_x.npz       파일 지정 (--dir 기준 상대경로 가능)
  pt plot --pprime --temp 300       pprime*.npz — 왼쪽 P′ vs T, 오른쪽 그 온도의 P′ 주파수 분해
  pt plot --nlhall --comp xz zx     성분 고정 (기본: 0 아닌 성분 전부, |P| > 최대의 1e-6)
  pt plot --pprime --csv            그림 대신 csv: <이름>.csv (T 열), pprime 은 <이름>_spec<T>K.csv (주파수 열) 도
출력은 현재 폴더, 기본 이름 <npz 이름>.png/.csv. 단위 P: meV·kB², P′: meV·kB²·ps. η 가 여럿이면 열·선을 η 별로.
"""
import glob
import os
import sys

import numpy as np

NAME, HELP = "plot", "nlhall / pprime 결과 npz → 그림 (png) 또는 표 (--csv)"
AX = "xyz"


def comp_arg(s):
    s = s.lower()
    if len(s) != 2 or any(c not in AX for c in s):
        raise ValueError
    return AX.index(s[0]), AX.index(s[1])


def add_args(p):
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--nlhall", nargs="?", const="", metavar="NPZ", help="pt nlhall 의 npz. 이름 없으면 --dir 의 최근 nphe_*.npz")
    g.add_argument("--pprime", nargs="?", const="", metavar="NPZ", help="pt pprime 의 npz. 이름 없으면 --dir 의 최근 pprime*.npz")
    p.add_argument("--comp", nargs="+", type=comp_arg, metavar="ab", help="성분만: --comp xz zx. 기본 0 아닌 성분 전부")
    p.add_argument("--csv", action="store_true", help="그림 대신 csv 로 저장")
    p.add_argument("--temp", type=float, default=80.0, help="pprime 주파수 분해의 온도 [K]. 기본 80")
    p.add_argument("-o", "--out", default=None, help="출력 이름 (확장자 빼고). 기본 <npz 이름>")


def find(a, name, pattern):
    if name:
        f = os.path.join(a.dir, name)
        if not os.path.exists(f):
            sys.exit(f"[plot] {f} 없음")
        return f
    fs = sorted(glob.glob(os.path.join(a.dir, pattern)), key=os.path.getmtime)
    if not fs:
        sys.exit(f"[plot] {os.path.abspath(a.dir or '.')} 에 {pattern} 없음 — 파일 이름을 주세요")
    return fs[-1]


def allowed(P):
    """P[..., 3, 3] → 0 아닌 성분 [(i, j)]"""
    m = np.abs(P).reshape(-1, 3, 3).max(0)
    return [(i, j) for i in range(3) for j in range(3) if m[i, j] > 1e-6 * m.max()]


def save_csv(out, x, xname, cols):
    """cols = [(이름, 값 배열)] → x 열 + 값 열들"""
    np.savetxt(out, np.column_stack([x] + [c for _, c in cols]), delimiter=",", fmt="%.8g",
               header=",".join([xname] + [n for n, _ in cols]), comments="")
    print("->", out)


def run(a):
    f = find(a, a.nlhall, "nphe_*.npz") if a.nlhall is not None else find(a, a.pprime, "pprime*.npz")
    d = np.load(f)
    TS, etas = d["TS"], d["etas"]
    base = os.path.splitext(a.out or os.path.basename(f))[0]
    et = (lambda e: f"_eta{e:g}") if len(etas) > 1 else (lambda e: "")
    nm = lambda i, j: AX[i] + AX[j]
    if a.nlhall is not None:
        P = d["P"] * 1e3  # [shift, η, T, 3, 3] → meV·kB²
        comps = a.comp or allowed(P)
        # (이름, 값, ±std, 성분 번호, η 번호)
        rows = [(f"P_{nm(i, j)}{et(e)}", P[:, ie, :, i, j].mean(0), P[:, ie, :, i, j].std(0), k, ie)
                for k, (i, j) in enumerate(comps) for ie, e in enumerate(etas)]
        if a.csv:
            save_csv(base + ".csv", TS, "T_K", [c for n, y, s, _, _ in rows
                                                 for c in ([(n, y), (n + "_std", s)] if len(P) > 1 else [(n, y)])])
            return
        fig, ax = _plt().subplots(figsize=(6, 4.2))
        for n, y, s, k, ie in rows:
            l, = ax.plot(TS, y, LS[ie % 4], color=f"C{k}", label=n.replace("_eta", "  η="))
            if len(P) > 1:
                ax.fill_between(TS, y - s, y + s, color=l.get_color(), alpha=0.2, lw=0)
        ax.set(xlabel="T (K)", ylabel=r"$P_{ab}$ (meV·$k_B^2$)",
               title=f"{os.path.basename(f)}  nq={int(d['nq'])}, shift {len(P)}" + (" (band = ±std)" if len(P) > 1 else ""))
    else:
        P0, Pm, Pb = (d[k][0] * 1e3 for k in ("P_notau", "P", "P_bin"))  # [η, T, 3, 3]
        spec, E = d["spec"] * 1e3, d["EDGES"]  # [P/P′/P′bin, η, T, bin, 3, 3]
        comps = a.comp or allowed(Pm)
        it = int(np.argmin(np.abs(TS - a.temp)))
        cx = E[:-1] + np.diff(E) / 2
        print(f"T={TS[it]:.0f}K: " + "  ".join(f"P_{nm(i, j)}={P0[0, it, i, j]:+.4f} P′={Pm[0, it, i, j]:+.4f}" for i, j in comps))
        if a.csv:
            save_csv(base + ".csv", TS, "T_K", [(f"{q}_{nm(i, j)}{et(e)}", X[ie, :, i, j]) for i, j in comps
                                                for ie, e in enumerate(etas) for q, X in (("P", P0), ("Pp", Pm), ("Ppbin", Pb))])
            save_csv(f"{base}_spec{TS[it]:.0f}K.csv", cx, "nu_THz", [(f"{q}_{nm(i, j)}{et(e)}", spec[s, ie, it, :, i, j]) for i, j in comps
                                                                     for ie, e in enumerate(etas) for s, q in ((0, "P"), (1, "Pp"), (2, "Ppbin"))])
            return
        fig, (ax, bx) = _plt().subplots(1, 2, figsize=(11, 4.2))
        sp = spec[1, 0, it]  # 첫 η
        top = np.nonzero(np.abs(sp).reshape(len(cx), -1).max(1))[0]
        for k, (i, j) in enumerate(comps):
            c = f"C{k}"
            for ie, e in enumerate(etas):
                ax.plot(TS, Pm[ie, :, i, j], LS[ie % 4], color=c, label=f"P′_{nm(i, j)} (mode τ)" + (f" η={e:g}" if len(etas) > 1 else ""))
            ax.plot(TS, Pb[0, :, i, j], LS[1], color=c, alpha=0.6, label=f"P′bin_{nm(i, j)} (bin τ)")
            bx.step(cx, sp[:, i, j], where="mid", color=c, label=f"P′_{nm(i, j)}")
            bx.plot(cx, np.cumsum(sp[:, i, j]), ":", color=c, label=f"cumul. P′_{nm(i, j)}")
        ax.set(xlabel="T (K)", ylabel=r"$P'_{ab}$ (meV·$k_B^2$·ps)", title=f"{os.path.basename(f)}  nq={int(d['nq'])}")
        bx.set(xlabel="ν (THz)", ylabel=r"$P'_{ab}$ per bin (meV·$k_B^2$·ps)", title=f"T = {TS[it]:.0f} K, bin {E[1] - E[0]:g} THz")
        if len(top):
            bx.set_xlim(0, E[top[-1] + 1] + 0.5)
        bx.axhline(0, color="k", lw=0.5)
        bx.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=200)
    print(f"{f} -> {base}.png")


LS = ["-", "--", ":", "-."]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
