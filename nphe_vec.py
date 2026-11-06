#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nphe_vec.py — 비선형 포논 홀 응답텐서 P_ab(T) [meV*kB^2], 단독 실행 벡터화판.

  P_ab = 1/(V T^2) sum_{n,q} E^3 (d rho0/d q_a) Omega_{n,b}      (Luo 2023, docs/THEORY_Pab.md)

입력 폴더: POSCAR + FORCE_SETS(또는 FORCE_CONSTANTS) [+ BORN 있으면 NAC on]
           또는 phonopy_params.yaml (이 경우 슈퍼셀은 파일 값 사용)
3x3 전 성분을 한 번에 계산 -> 점군 대칭(의사텐서 det(R) R P R^T)으로 허용/금지 성분·관계를
정확히 판정해 보고 (+ 대칭평균값 병기, 금지성분 잔차 = 격자 잡음 지표).
한 패스에서 D(q)·eigh 공유, Kubo 분모만 η 별로 (η=0 bare + η-정규화
 den=(w_n^2-w_m^2)^2+(2*wbar*η)^2). D(q)는 phonopy 배치 C 루틴(OpenMP, NAC 포함).
shift(격자 평행이동) 평균 -> mean±SE. shift 마다 체크포인트(끊겨도 같은 명령으로 이어감).
Berry 부호: 수정 규약 Om = -2 Im sum_m dD_a[n,m] dD_b[m,n] / den.
검증: PbTiO3 nq100 -> P_xy peak 0.476(η.08)/0.486(η.04), Luo 0.47-0.5.

사용 예:
  python nphe_vec.py <폴더>                                  # nq=100, shift 2, supercell 2 2 2
  python nphe_vec.py <폴더> --nq 64 --shift 4 --supercell 3 3 3
  python nphe_vec.py <폴더> --nq 50 100 200              # 여러 nq 순차 실행 + 마지막에 nq 수렴표
  python nphe_vec.py <폴더> --etas 0 0.08 0.04 --tmin 10 --tmax 300 --dt 10 --no-nac
주의: --supercell 은 FORCE_SETS 를 만든 슈퍼셀과 같아야 함(수렴 파라미터 아님).
"""
import os, time, argparse
import numpy as np
import phonopy
import phonopy.harmonic.dynamical_matrix as _pdm

# D(q) 배치 백엔드: phonopy 버전 따라 이름이 다름 -> 있는 것 사용, 없으면 q별 루프(느리지만 결과 동일)
if hasattr(_pdm, "get_dynamical_matrices_at_qpoints"):            # phonopy >= 4.x (검증 버전 4.1.0)
    _batch, BACKEND = _pdm.get_dynamical_matrices_at_qpoints, "batch-C (get_dynamical_matrices_at_qpoints)"
elif hasattr(_pdm, "run_dynamical_matrix_solver_c"):              # 구버전 이름
    _batch, BACKEND = _pdm.run_dynamical_matrix_solver_c, "batch-C (run_dynamical_matrix_solver_c)"
else:
    _batch, BACKEND = None, "q별 루프 (느림: phonopy 업데이트 권장 -> pip install -U phonopy==4.1.0)"

HBAR, KB, E_CHG = 1.054571817e-34, 1.380649e-23, 1.602176634e-19
wA = 2 * np.pi * 1e12          # THz -> rad/s
vS = 2 * np.pi * 1e2           # THz*Ang -> m/s
AX = "xyz"


def load(d, supercell, nac):
    yml = os.path.join(d, "phonopy_params.yaml")
    if not any(os.path.exists(os.path.join(d, f)) for f in ("FORCE_SETS", "FORCE_CONSTANTS")) \
            and os.path.exists(yml):
        print(f"phonopy_params.yaml 사용 (슈퍼셀은 파일 값, --supercell 무시)", flush=True)
        return phonopy.load(yml, primitive_matrix="P", is_nac=nac, log_level=0)
    fc = {}
    if os.path.exists(os.path.join(d, "FORCE_SETS")):
        fc["force_sets_filename"] = os.path.join(d, "FORCE_SETS")
    elif os.path.exists(os.path.join(d, "FORCE_CONSTANTS")):
        fc["force_constants_filename"] = os.path.join(d, "FORCE_CONSTANTS")
    else:
        raise FileNotFoundError(f"{d}: FORCE_SETS / FORCE_CONSTANTS / phonopy_params.yaml 없음")
    born = os.path.join(d, "BORN") if (nac and os.path.exists(os.path.join(d, "BORN"))) else None
    return phonopy.load(unitcell_filename=os.path.join(d, "POSCAR"), born_filename=born,
                        supercell_matrix=list(supercell), primitive_matrix=np.eye(3),
                        calculator="vasp", is_nac=born is not None, log_level=0, **fc)


class Lattice:
    def __init__(self, ph):
        A = np.array(ph.primitive.cell)                # 행 = 격자벡터 [Ang]
        self.Brec = 2 * np.pi * np.linalg.inv(A).T     # 행 = 역격자벡터
        self.Binv = np.linalg.inv(self.Brec)
        self.Vcell = abs(np.linalg.det(A))             # Ang^3
        self.dm, self.factor = ph.dynamical_matrix, ph.unit_conversion_factor
        self.n = 3 * len(ph.primitive.masses)
        self.nac = ph.nac_params is not None
        sym = ph.primitive_symmetry
        self.sg = sym.get_international_table()
        # 점군 연산 -> 데카르트 R = A^T r (A^T)^-1 ; 의사텐서 투영 Pi(P) = <det(R) R P R^T>
        At = A.T
        self.Rc = [At @ r @ np.linalg.inv(At) for r in sym.pointgroup_operations]
        # k 공간(분수좌표) 작용 M = A R A^-1, 시간반전 -> {±M}. IBZ 는 모두 대각 ±1 일 때만 지원
        Ms = [np.rint(A @ R @ np.linalg.inv(A)).astype(int) for R in self.Rc]
        diag = all(np.count_nonzero(M - np.diag(np.diag(M))) == 0 and set(np.abs(np.diag(M))) == {1} for M in Ms)
        self.ksigns = sorted({tuple(sg * np.diag(M)) for M in Ms for sg in (1, -1)}) if diag else None
        # 2_1 나선축 방향 i: k_i=1/2 면 전체가 축퇴(노달평면) -> 그 면을 지나는 IBZ 격자는 피함
        ops = sym.symmetry_operations if hasattr(sym, "symmetry_operations") else sym.get_symmetry_operations(); self.screw = set()
        for r, t in zip(ops["rotations"], ops["translations"]):
            for i in range(3):
                e = np.zeros(3, int); e[i] = 1
                if np.trace(r) == -1 and (r @ e == e).all() and abs((t[i] % 1) - 0.5) < 1e-3:
                    self.screw.add(i)


def sym_project(Rc, P):
    """P[...,3,3] -> 점군 평균 det(R) R P R^T (대칭 허용 부분공간 투영)."""
    return sum(np.linalg.det(R) * np.einsum("ia,...ab,jb->...ij", R, P, R) for R in Rc) / len(Rc)


def allowed_components(Rc):
    """-> 허용 성분 [(a,b)], 관계 문자열 목록 (2성분 묶임만 표기)."""
    Pi = np.array([sym_project(Rc, np.eye(9)[k].reshape(3, 3)).ravel() for k in range(9)]).T  # Pi[ab,cd]
    Pi[np.abs(Pi) < 1e-4] = 0.0   # 투영계수는 0,±1/2,±1 꼴; POSCAR 반올림 잔차(~1e-8) 제거
    allow = [(k // 3, k % 3) for k in range(9) if np.abs(Pi[k]).max() > 0]
    rel = []
    for k in range(9):
        nz = [j for j in np.nonzero(Pi[k])[0] if j != k]
        if len(nz) == 1 and Pi[k, k] != 0 and k < nz[0]:
            c = (1 - Pi[k, k]) / Pi[k, nz[0]]          # P_j = c * P_k
            rel.append(f"P_{AX[nz[0]//3]}{AX[nz[0]%3]} = {c:+.2f} P_{AX[k//3]}{AX[k%3]}")
    return allow, rel


def bz_grid(nq):
    k = np.linspace(-0.5, 0.5, nq, endpoint=False) + 0.5 / nq
    return np.stack(np.meshgrid(k, k, k, indexing="ij"), -1).reshape(-1, 3)


def Dbatch(L, Kc):
    """Kc[nk,3] cart(1/Ang) -> D[nk,n,n] THz^2."""
    Q = np.ascontiguousarray(Kc @ L.Binv)
    if _batch is not None:
        return np.asarray(_batch(L.dm, Q)) * L.factor ** 2
    out = np.empty((len(Q), L.n, L.n), dtype=complex)
    for i, q in enumerate(Q):
        L.dm.run(q); out[i] = L.dm.dynamical_matrix
    return out * L.factor ** 2


def modes(L, Kf, etas, h=1e-4):
    """Kf[nk,3] frac -> w[nk,n] THz, v[nk,n,3] m/s, Oms[len(etas)] 각 [nk,n,3] m^2."""
    Kc = Kf @ L.Brec
    D0 = Dbatch(L, Kc); D0 = (D0 + D0.conj().transpose(0, 2, 1)) / 2
    w2, E = np.linalg.eigh(D0)
    w = np.sign(w2) * np.sqrt(np.abs(w2))
    EH = E.conj().transpose(0, 2, 1)
    dD = []
    for a in range(3):                                  # 중앙차분 (원본과 동일 h)
        e = np.zeros(3); e[a] = h
        dD.append(EH @ ((Dbatch(L, Kc + e) - Dbatch(L, Kc - e)) / (2 * h)) @ E)
    wsafe = 2 * np.where(np.abs(w) > 1e-6, w, np.inf)
    v = np.stack([np.real(np.diagonal(dD[a], axis1=1, axis2=2)) / wsafe for a in range(3)], -1) * vS
    num = [dD[a] * dD[b].transpose(0, 2, 1) for a, b in ((1, 2), (2, 0), (0, 1))]   # Ω_x,Ω_y,Ω_z
    gap2 = (w2[:, :, None] - w2[:, None, :]) ** 2
    wbar = 0.5 * (np.abs(w)[:, :, None] + np.abs(w)[:, None, :])
    idx = np.arange(w.shape[1]); Oms = []
    for eta in etas:
        den = gap2 + (2 * wbar * eta) ** 2
        den[:, idx, idx] = np.inf
        Oms.append(np.stack([-2 * np.imag((num[g] / den).sum(axis=2)) for g in range(3)], -1) * 1e-20)
    return w, v, Oms


def thermal_pref(w, TS):
    """-> [nT,nk,n] = -(hbar/kT) E^3 rho(1+rho)  (w>0.1 THz 만)."""
    mask = w > 0.1; E = HBAR * w * wA
    out = np.empty((len(TS),) + w.shape)
    for it, T in enumerate(TS):
        x = np.where(mask, E / (KB * T), 1.0)
        rho = 1.0 / np.expm1(np.clip(x, 1e-12, 700))
        out[it] = -(HBAR / (KB * T)) * E ** 3 * np.where(mask, rho * (1 + rho), 0.0)
    return out


def P_oneshift(L, nq, delta, etas, TS, block):
    """-> P[neta,nT,3,3]  eV*kB^2."""
    k1 = np.linspace(-0.5, 0.5, nq, endpoint=False) + 0.5 / nq
    N = nq ** 3   # 격자를 통째로 만들지 않음 (nq=600 이면 5GB+) -> 블록마다 생성, 순서는 bz_grid 와 동일
    acc = np.zeros((len(etas), len(TS), 3, 3))
    for i in range(0, N, block):
        K = k1[np.stack(np.unravel_index(np.arange(i, min(i + block, N)), (nq,) * 3), -1)] + delta
        K = (K + 0.5) % 1.0 - 0.5
        w, v, Oms = modes(L, K, etas)
        pref = thermal_pref(w, TS)
        for ie, Om in enumerate(Oms):
            vO = (v[:, :, :, None] * Om[:, :, None, :]).reshape(-1, 9)   # einsum 보다 온도 많을 때 훨씬 빠름
            acc[ie] += (pref.reshape(len(TS), -1) @ vO).reshape(len(TS), 3, 3)
    norm = 1.0 / (L.Vcell * 1e-30 * TS ** 2 * N) / (E_CHG * KB ** 2)
    return acc * norm[None, :, None, None]


HALF = [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]  # ×1/(2nq): 대칭 보존 shift


def safe_half_shifts(L, nq):
    """대칭 보존 shift 중 안전한 것만: 특수면(k_i=0,1/2)을 지나는 축 ≤1개, 2_1 축 방향은 0개.
    (대칭 강제 축퇴점·선·면 위에서는 축퇴 밴드 분해가 임의라 Kubo 값이 정의되지 않음)
    짝수 nq 면 (0,0,0)+반칸 1축 -> 최대 3개; 홀수 nq 는 모든 축이 특수면을 지나 0개."""
    out = []
    for h in HALF:
        K2 = [(2 * np.arange(nq) + (-nq + 1 + h[i])) for i in range(3)]          # 축별 2nq·k
        has0 = [bool((k == 0).any()) for k in K2]
        hashalf = [bool((np.abs(k) == nq).any()) for k in K2]
        special = [has0[i] or hashalf[i] for i in range(3)]
        # 특수면(k_i=0 또는 1/2)을 지나는 축이 2개 이상이면 그 교선(대칭 강제 축퇴선)에 격자점이 놓임 -> 제외
        if sum(special) > 1 or any(special[i] for i in L.screw):
            continue
        out.append(h)
    return out


def P_oneshift_ibz(L, nq, delta, etas, TS, block):
    """IBZ 판: 격자(대칭 보존 shift)에서 대칭 궤도마다 대표점 1개만 계산, 가중치 = 궤도 크기.
    Σ_전체 f = Π( Σ_대표 w f ),  Π = sym_project (시간반전 상은 v⊗Ω 가 같아 그대로).
    정수 좌표 K2 = 2nq·k 로 궤도 판정 (부동소수 오차 없음). -> P[neta,nT,3,3] eV*kB^2."""
    S = np.array(L.ksigns)
    k1 = np.linspace(-0.5, 0.5, nq, endpoint=False) + 0.5 / nq
    off = np.rint(2 * nq * (k1[0] + np.asarray(delta))).astype(np.int64)   # K2 = 2nq·k = 2j + off_i
    N, M = nq ** 3, 2 * nq
    wrap = lambda x: (x + nq - 1) % M - nq + 1                                  # (-nq, nq]
    acc = np.zeros((len(etas), len(TS), 3, 3)); nkept = 0
    eb = block * len(S)
    for i in range(0, N, eb):
        J = np.stack(np.unravel_index(np.arange(i, min(i + eb, N)), (nq,) * 3), -1)
        K2 = wrap(2 * J + off)
        img = wrap(K2[:, None, :] * S[None])                                     # [nk, |S|, 3]
        code = ((img[..., 0] + nq) * (M + 1) + (img[..., 1] + nq)) * (M + 1) + (img[..., 2] + nq)
        own = ((K2[:, 0] + nq) * (M + 1) + (K2[:, 1] + nq)) * (M + 1) + (K2[:, 2] + nq)
        keep = own == code.max(1)
        if not keep.any():
            continue
        cs = np.sort(code[keep], 1)
        w = 1 + (np.diff(cs, axis=1) != 0).sum(1)                                # 서로 다른 상의 수 = 궤도 크기
        K = K2[keep] / M
        nkept += keep.sum()
        ww, v, Oms = modes(L, K, etas)
        pref = thermal_pref(ww, TS) * w[None, :, None]
        for ie, Om in enumerate(Oms):
            vO = (v[:, :, :, None] * Om[:, :, None, :]).reshape(-1, 9)
            acc[ie] += (pref.reshape(len(TS), -1) @ vO).reshape(len(TS), 3, 3)
    acc = sym_project(L.Rc, acc)
    norm = 1.0 / (L.Vcell * 1e-30 * TS ** 2 * N) / (E_CHG * KB ** 2)
    return acc * norm[None, :, None, None], nkept


def report(P, TS, etas, nq, label, L):
    """P[ns,neta,nT,3,3] eV*kB^2 -> 요약. 성분 허용/금지는 점군 대칭으로 정확히 판정."""
    R = P * 1000
    m = R.mean(0); se = R.std(0) / np.sqrt(len(R)) if len(R) > 1 else np.full_like(m, np.nan)
    ms = sym_project(L.Rc, m)                       # 대칭평균(묶인 성분 평균 -> 잡음↓)
    allow, rel = allowed_components(L.Rc)
    forb = [(a, b) for a in range(3) for b in range(3) if (a, b) not in allow]
    print(f"\n[{label}, {L.sg}] nq={nq}^3, {len(R)} shifts, 단위 meV*kB^2 (SE=shift 간 표준오차)")
    print(f"  대칭 허용: {', '.join(f'P_{AX[a]}{AX[b]}' for a, b in allow) or '없음(중심대칭 -> P≡0)'}"
          + (f"   관계: {'; '.join(rel)}" if rel else ""))
    for a, b in allow:
        for ie, e in enumerate(etas):
            y, s, ys = m[ie, :, a, b], se[ie, :, a, b], ms[ie, :, a, b]
            ip = int(np.argmax(np.abs(ys)))
            print(f"  P_{AX[a]}{AX[b]}  η={e:<5}: peak {y[ip]:+.4f}±{s[ip]:.4f} @ {TS[ip]:.0f}K "
                  f"[대칭평균 {ys[ip]:+.4f}]   P({TS[-1]:.0f}K)={y[-1]:+.4f}±{s[-1]:.4f}")
    if forb:
        nz = max(np.abs(m[:, :, a, b]).max() for a, b in forb)
        print(f"  금지(대칭상 0): {', '.join(f'P_{AX[a]}{AX[b]}' for a, b in forb)}"
              f"  -> 계산 잔차 최대 |{nz:.4f}| (격자 잡음 크기 지표, nq↑면 0으로)")


def main():
    ap = argparse.ArgumentParser(description="NPHE P_ab(T) 벡터화 단독 실행")
    ap.add_argument("dir", help="POSCAR+FORCE_SETS(+BORN) 또는 phonopy_params.yaml 폴더")
    ap.add_argument("--nq", type=int, nargs="+", default=[100], help="여러 개 가능: --nq 50 100 200")
    ap.add_argument("--shift", type=int, default=2, help="격자 shift 개수 (SE 추정)")
    ap.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0, 0.08, 0.04], help="THz, 0=bare Kubo")
    ap.add_argument("--tmin", type=float, default=10.0)
    ap.add_argument("--tmax", type=float, default=300.0)
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--no-nac", action="store_true")
    ap.add_argument("--block", type=int, default=2000, help="한 번에 처리할 k점 수(메모리)")
    ap.add_argument("--seed", type=int, default=0, help="무작위 shift 시드 (--ibz 면 대칭 shift 목록의 시작 번호)")
    ap.add_argument("--ibz", action="store_true", help="대칭으로 계산점 축소(Pca2_1 1/8, P2_1 1/4). shift 는 대칭 보존 반칸 shift 사용")
    ap.add_argument("--out", default=None, help="결과 npz (기본: 현재폴더/nphe_<폴더명>_nq<nq>_sc<abc>.npz)")
    a = ap.parse_args()

    d = os.path.abspath(a.dir)
    TS = np.arange(a.tmin, a.tmax + 1e-9, a.dt)
    etas = list(a.etas)

    try:
        ph = load(d, a.supercell, not a.no_nac)
    except RuntimeError as e:
        if "not consistent with supercell" in str(e):
            raise SystemExit(f"--supercell {a.supercell} 가 FORCE_SETS 의 슈퍼셀과 다름 "
                             f"(FORCE_SETS 를 만든 슈퍼셀 크기로 지정하세요)") from e
        raise
    L = Lattice(ph)
    print(f"phonopy {phonopy.__version__}, D(q) 백엔드: {BACKEND}", flush=True)
    print(f"{d}\n  {L.sg}, 원자 {L.n//3}, 밴드 {L.n}, V={L.Vcell:.3f} Ang^3, NAC={'on' if L.nac else 'off'}, "
          f"nq={a.nq}, shift {a.shift}, η={etas}, T={TS[0]:.0f}-{TS[-1]:.0f}K", flush=True)

    allres = {}
    for nq in a.nq:      # 작은 nq 부터 순서대로; nq 별 파일·체크포인트
        tag = f"{os.path.basename(d.rstrip(os.sep))}_nq{nq}_sc{''.join(map(str, a.supercell))}"
        if a.out:
            out = a.out if len(a.nq) == 1 else a.out.replace(".npz", "") + f"_nq{nq}.npz"
        else:
            out = os.path.join(os.getcwd(), f"nphe_{tag}.npz")
        allres[nq] = run_nq(L, nq, a, etas, TS, out, d)

    if len(a.nq) > 1:    # nq 수렴표: 허용성분 대칭평균 peak
        allow, _ = allowed_components(L.Rc)
        print(f"\n[nq 수렴표] 허용성분 대칭평균 peak [meV*kB^2] (@T)")
        for a_, b_ in allow:
            for ie, e in enumerate(etas):
                row = []
                for nq, P in allres.items():
                    ms = sym_project(L.Rc, P.mean(0) * 1000)[ie, :, a_, b_]
                    ip = int(np.argmax(np.abs(ms)))
                    row.append(f"nq{nq}: {ms[ip]:+.4f}@{TS[ip]:.0f}K")
                print(f"  P_{AX[a_]}{AX[b_]} η={e:<5} " + "   ".join(row))


def run_nq(L, nq, a, etas, TS, out, d):
    """nq 하나: shift 루프 + 체크포인트 + 보고. -> P[ns,neta,nT,3,3]."""
    print(f"\n=== nq={nq}^3 ({nq**3:.2e} k/shift) -> {out}", flush=True)
    res, deltas = [], []
    if os.path.exists(out):
        old = np.load(out)
        if list(old["etas"]) == etas and np.array_equal(old["TS"], TS) and int(old["nq"]) == nq:
            res, deltas = list(old["P"]), list(old["deltas"])
            print(f"  체크포인트 {len(res)} shift 이어받음", flush=True)
        else:
            raise SystemExit(f"{out} 이 다른 설정(η/T/nq)으로 존재 -> --out 으로 다른 파일 지정")
    rng = np.random.default_rng(a.seed)
    if a.ibz and L.ksigns is None:
        raise SystemExit("--ibz: 이 구조의 k 공간 대칭이 축별 부호반전 꼴이 아니라 미지원 -> --ibz 없이 실행")
    if a.ibz and not safe_half_shifts(L, nq):
        raise SystemExit(f"--ibz: nq={nq} 에서 안전한 대칭 격자가 없음 -> 짝수 nq 사용")
    for s in range(a.shift):
        if a.ibz:
            safe = safe_half_shifts(L, nq)
            if s >= len(safe):
                print(f"  [IBZ] nq={nq} 에서 안전한 대칭 shift 는 {len(safe)}개뿐 -> shift {s+1} 생략 (다른 nq 로 SE 추정 권장)", flush=True); break
            delta = np.array(safe[(a.seed + s) % len(safe)]) / (2 * nq)
        else:
            delta = rng.random(3) / nq
        if s < len(res):
            continue
        t0 = time.time()
        if a.ibz:
            P, nk = P_oneshift_ibz(L, nq, delta, etas, TS, a.block)
            print(f"  [IBZ] 대칭 {len(L.ksigns)}개, 계산점 {nk:.3e} / {nq**3:.3e} ({nk/nq**3*100:.1f}%)", flush=True)
            res.append(P)
        else:
            res.append(P_oneshift(L, nq, delta, etas, TS, a.block))
        deltas.append(delta)
        np.savez(out, P=np.array(res), TS=TS, etas=np.array(etas), nq=nq,
                 supercell=np.array(a.supercell), deltas=np.array(deltas), dir=d)
        print(f"  shift {s+1}/{a.shift} 완료 ({time.time()-t0:.0f}s)", flush=True)
    report(np.array(res), TS, etas, nq, os.path.basename(d.rstrip(os.sep)), L)
    print(f"저장: {out}  (P[shift,η,T,a,b] eV*kB^2)")
    return np.array(res)


if __name__ == "__main__":
    main()
