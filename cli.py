"""pt — phonopy 결과로 포논 토폴로지 계산 (파이썬 패키지 이름은 phtopo).

POSCAR, FORCE_SETS, BORN 이 있는 폴더에서:
  pt band                                포논 분산 (seekpath 자동 경로) → band.dat, band.png
  pt hr                                  SR TB 해밀토니안 → phonon_hr_SR.dat (슈퍼셀은 힘 파일에서 자동, --d 로 지정)
  pt nodes    --band 11-18 --axis 3      노드 탐색 + FHS χ → nodes.txt (band 열 포함)
  pt scan     --verify                   BZ 전체 노드 탐색 (대칭 기약 격자, 기본 전체 밴드쌍) → nodes.txt
  pt line                                고대칭 경로 위 교차 (seekpath 자동) → line.txt
  pt chi      --radius 0.002             묶음 FHS 구면 χ (nodes.txt 의 band 마다) → chirality.txt
  pt box      --node 1                   묶음 FHS 상자 χ (노드가 몰려 있을 때)
  pt berry    --band 17                  Berry curvature 지도
  pt simphony --band 17                  Simphony pn.in + hr
  pt nlhall   --nq 100 --ibz             비선형 포논 홀 P_ab(T) (NPHE)
  pt spectral --nq 60 --ibz              NPHE P(ν) 주파수 분해 (kappa --tau omega 입력)
  pt tau      --fc3 fc3.hdf5             phono3py τ(ω,T) (kappa --tau omega, pprime 입력)
  pt pprime   --tau-file tau_parent.npz  P′ = Σ τ_nk P_nk (모드별 수명을 응답 안에)
  pt kappa    --P nphe_*.npz --tau 5     열홀 κ_H (NPHE). --linear 면 자기장 선형 κ_xy(B)
각 명령 -h 로 자세한 설명.
"""
import argparse
import os
import sys

import numpy as np

from . import band, berry, box, chi, fc2, fc34, hr, kappa, line, nlhall, nodes, pprime, relax, scan, simphony, spectral, tau
from .core import parse_bands


def dim_arg(s):
    v = [int(x) for x in s.replace(",", " ").split()]
    if len(v) == 3:
        return v
    if len(v) == 9:
        return np.reshape(v, (3, 3)).tolist()
    raise argparse.ArgumentTypeError(f'--dim 은 정수 3개 또는 9개 ("2 2 2"), 받은 값: {s!r}')


def band_arg(s):
    try:
        return parse_bands(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


# pt / pt -h 첫 화면: 분야별 번호 목록 → 번호 Enter 로 그 명령 자세히 (vaspkit 식)
MENU = [
    ("포논 위상 (분산·Weyl 노드·χ·Berry)", [
        (relax, "pt relax --model mace", "구조 완화 (lattice+ions) — MLIP, phonon 전 pre-relax"),
        (band, "pt band", "포논 분산 (k 경로 자동 또는 --path) → band.dat, band.png"),
        (hr, "pt hr", "phonopy 힘상수 → 단거리 TB 해밀토니안 (Simphony·WannierTools 입력)"),
        (scan, "pt scan --verify", "BZ 전체에서 Weyl 노드 찾기 (대칭으로 줄인 격자 → 좁히기 → χ)"),
        (nodes, "pt nodes --band 17 --axis 3", "면 하나 (q3=0 등) 위 노드만 빠르게"),
        (line, "pt line", "고대칭 경로 위 밴드 교차 (seekpath 자동)"),
        (chi, "pt chi --nodes nodes.txt", "노드 둘레 구의 Chern 수 = chirality χ"),
        (box, "pt box --node 1", "몰린 노드들을 상자로 감싸 χ 합"),
        (berry, "pt berry --band 17", "Berry 곡률 지도 (그림)"),
        (simphony, "pt simphony --band 17", "Simphony 입력 일괄 생성 (교차검증)"),
    ]),
    ("포논 홀 (NPHE·열홀)", [
        (nlhall, "pt nlhall --nq 100 --ibz", "비선형 포논 홀 응답 P_ab(T)"),
        (spectral, "pt spectral --nq 60 --ibz", "P_ab 주파수 분해 P(ν)"),
        (fc34, "pt 3rd --model mace", "3차 힘상수 (FC3) — phono3py + MLIP (τ/κ/NPHE 입력)"),
        (fc2, "pt 2nd --model mace", "2차 힘상수 (FC2) — phonopy + MLIP (조화 포논 분산 입력)"),
        (tau, "pt tau --fc3 fc3.hdf5 --isotope", "phono3py 포논 수명 τ(ω,T)"),
        (pprime, "pt pprime --tau-file tau_parent.npz", "모드별 수명을 넣은 P′ = Σ τ_nk P_nk"),
        (kappa, "pt kappa --P nphe.npz --tau 5", "열홀 κ_H·ΔT⊥ (기본 NPHE, --linear 면 자기장 κ_xy)"),
    ]),
]
MODS = [m for _, items in MENU for m, _, _ in items]


def overview():
    L = ["pt — phonopy 결과 (POSCAR, FORCE_SETS 또는 --fc, BORN) 로 포논 위상과 포논 홀. 그 폴더에서 바로 실행.",
         "공통: --dir 폴더 (입력 폴더, 기본 현재)  --d 3 3 3 (기본 자동)  --band 11-18  -c POSCAR  --fs FORCE_SETS  --fc fc2.hdf5  --born BORN (있으면 자동)  --loto T/F", ""]
    k = 0
    for title, items in MENU:
        L.append(f"[{title}]")
        for m, ex, what in items:
            k += 1
            L.append(f"  {k:>2}. {m.NAME:<9} {what}")
            L.append(f"      {ex}")
        L.append("")
    return "\n".join(L)


def menu(subs):
    print(overview())
    if not sys.stdin.isatty():  # 파이프·스크립트: 목록만
        print("자세히: pt <명령> -h")
        return
    while True:
        try:
            c = input("번호 또는 명령 이름 (자세히 보기, Enter = 끝): ").strip()
        except EOFError:
            return
        if c in ("", "q", "0"):
            return
        m = MODS[int(c) - 1] if c.isdigit() and 1 <= int(c) <= len(MODS) else next((x for x in MODS if x.NAME == c), None)
        if m is None:
            print(f"  1–{len(MODS)} 또는 명령 이름")
            continue
        print("\n" + "=" * 100)
        subs[m.NAME].print_help()
        print("=" * 100 + "\n")
        input("Enter = 목록으로 ")
        print(overview())


def main(argv=None):
    for s in (sys.stdout, sys.stderr):  # Windows cp949 콘솔에서 —, ², Å 출력하다 죽지 않게
        s.reconfigure(errors="replace")
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("입력 (phonopy 와 같은 의미)")
    g.add_argument("--dir", default="", help="입력 파일이 있는 폴더 (-c, --fs, --fc, --born 의 상대 경로 기준). 출력은 현재 폴더. 기본 현재 폴더")
    g.add_argument("-c", "--cell", default="POSCAR", help="단위격자. 기본 POSCAR")
    g.add_argument("-d", "--d", "--dim", dest="dim", nargs="+", default=None,
                   help='슈퍼셀 (FORCE_SETS 만들 때와 같게): --d 3 3 3, --dim "3 3 3", 9개면 행렬. '
                        "기본 자동: phonopy_disp.yaml/phonopy.yaml → 힘 파일 원자 수 (정육면체) → 2 2 2")
    g.add_argument("--fs", default="FORCE_SETS", help="기본 FORCE_SETS")
    g.add_argument("--fc", default=None, help="힘상수 파일 (FORCE_CONSTANTS 또는 fc2.hdf5, 예 TDEP 유효 fc2(T)) — 주면 --fs 대신")
    g.add_argument("--born", default=None, help="기본 BORN — 있으면 자동으로 읽어 NAC 켬, 없으면 경고 후 NAC 끔")
    g.add_argument("--loto", type=str.upper, choices=("T", "F"), default=None,
                   help="LO-TO (NAC) 켬 T / 끔 F. 기본: BORN 있으면 켬. T 인데 BORN 없으면 에러")
    g.add_argument("--band", type=band_arg, default=None,
                   help='아래 밴드 번호 (1부터), 밴드쌍 (b, b+1). "17", "11-18" (= 11/12 ... 18/19), "11,13". '
                        "기본: scan, line 은 전체 밴드쌍, chi, box 는 nodes 파일의 band 열. nodes, berry, simphony 는 필수")

    p = argparse.ArgumentParser(prog="pt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="{" + ",".join(m.NAME for m in MODS) + "}")
    subs = {}
    for mod in MODS:
        sp = sub.add_parser(mod.NAME, parents=[common], help=mod.HELP, description=mod.__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_args(sp)
        sp.set_defaults(func=mod.run)
        subs[mod.NAME] = sp
    args = sys.argv[1:] if argv is None else list(argv)
    if args in ([], ["-h"], ["--help"]):
        return menu(subs)
    a = p.parse_args(args)
    if a.dir and not os.path.isdir(a.dir):  # kappa 의 옛 --dir x (이제 --direction) 도 여기서 걸림
        p.error(f"--dir: 폴더 {a.dir!r} 없음 (kappa 의 ∇T 방향은 --direction)")
    a.cell, a.fs = os.path.join(a.dir, a.cell), os.path.join(a.dir, a.fs)
    a.fc = a.fc and os.path.join(a.dir, a.fc)
    born = os.path.join(a.dir, a.born or "BORN")
    if a.born and not os.path.exists(born):
        p.error(f"--born: {born} 없음")
    a.born = born if os.path.exists(born) else None  # None = 파일 없음 → core.load 가 경고하고 NAC 끔
    if a.loto == "T" and a.born is None:
        p.error(f"--loto T: {born} 없음")
    if a.loto == "F" and hasattr(a, "no_nac"):
        a.no_nac = True  # 본체 사본 (nphe_vec, spec_vec, aphe) 이 보는 플래그
    if a.dim is None:
        from .core import guess_dim

        a.dim, why = guess_dim(a)
        print(f"[슈퍼셀 {' '.join(map(str, np.ravel(a.dim)))} — {why}]", file=sys.stderr)
    else:
        try:
            a.dim = dim_arg(" ".join(a.dim))  # --d 3 3 3 (따옴표 없이) 과 --dim "3 3 3" 둘 다
        except argparse.ArgumentTypeError as e:
            p.error(str(e))
    a.func(a)


if __name__ == "__main__":
    main()
