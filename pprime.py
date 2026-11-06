"""P′ — 수명을 응답 안에 넣은 NPHE 텐서.   P′_ab(T) = 1/(V T²) Σ_{n,k} τ_nk(T) E³ (∂ρ₀/∂k_a) Ω_{n,b}   [meV·kB²·ps]

상수 τ 이면 P′ = τ P. 모드마다 τ 가 다르면 j = (1/(Vħ²)) ∇T × (P′ᵀ ∇T) 가 맞는 식 (τ 를 밖으로 못 뺌).
τ_nk: pt tau (phono3py RTA) 의 Γ_n(q) — 기약점 → 대칭 궤도로 거친 격자 전체 → 밀집 격자 k 로 주기 삼선형 보간 (정렬 밴드 번호).
  Γ 점에 닿는 칸 (|q_i| < 1/m) 의 음향 3개만 Γ/ω² 를 보간해 ω² 를 곱함 (Γ 에서 Γ_ac = 0, Γ ∝ ω²; Γ 점 값은 이웃 평균).
  나머지는 Γ 그대로 — 상수 Γ 를 그대로 보존 (tests: τ 고정이면 P′ = τP). 밴드 교차에서 정렬 번호가 바뀌는 곳은 보간이 섞임.
같은 패스에서 셋 다:  P (τ 없음) / P′ 모드 τ / P′_bin (40_kappa_nphe 의 bin 평균 τ(ω)) → bin 평균 근사가 얼마나 틀리는지.
출력 npz: P = P′ (모드 τ, eV·kB²·ps — pt kappa --P <npz> --tau 1 이 그대로 κ_H), P_notau, P_bin, TS, etas,
  spec [P/P′/P′bin, η, T, bin, 3, 3] (0.25 THz bin, EDGES) — bin 합 = 위 셋.
"""
import time

import numpy as np
import spglib

from . import nphe_vec as V
from .core import load
from .scan import Model, fold

NAME, HELP = "pprime", "P′ = Σ τ_nk P_nk (모드별 수명을 응답 안에) → pprime.npz (kappa --P 로 κ_H)"


def add_args(p):
    p.add_argument("--tau-file", default="tau_parent.npz", help="pt tau (또는 39) 의 npz — freq, gamma, mesh. 기본 tau_parent.npz")
    p.add_argument("--nq", type=int, default=60, help="밀집 격자 nq³ (짝수, IBZ). 기본 60")
    p.add_argument("--eta", type=float, nargs="+", default=[0.08], help="Kubo η (THz). 기본 0.08")
    p.add_argument("--boundary", type=float, default=None, help="경계 산란 L [μm]: 1/τ += |v_nk|/L (모드마다, 밀집 격자 v). 기본 없음")
    p.add_argument("--seed", type=int, default=0, help="대칭 보존 반칸 shift 번호")
    p.add_argument("--block", type=int, default=2000)
    p.add_argument("-o", "--out", default="pprime.npz")


def gamma_grid(ph, tp):
    """tau npz 의 기약점 Γ[T, nir, nb] → 거친 격자 전체 G[m, m, m, T, nb], 음향 Γ/ω² Gac[m, m, m, T, 3]."""
    m = int(tp["mesh"])
    if "qpoints" in tp:  # pt tau 가 phono3py 의 기약점 좌표를 저장
        qir = np.asarray(tp["qpoints"])
    else:  # 옛 파일 (39): spglib 로 같은 기약점 — 아래 진동수 대조로 순서 확인
        p = ph.primitive
        mp, addr = spglib.get_ir_reciprocal_mesh([m] * 3, (p.cell, p.scaled_positions, p.numbers), is_shift=[0, 0, 0],
                                                 is_time_reversal=True)
        ir = np.unique(mp)
        if len(ir) != len(tp["freq"]):
            raise SystemExit(f"tau 파일 기약점 {len(tp['freq'])} != 이 구조의 {m}³ 기약점 {len(ir)} — 다른 구조·대칭?")
        qir = addr[ir] / m
    md = Model(ph)
    W = np.array([md.freqs(q) for q in qir])
    ok = np.linalg.norm(qir, axis=1) > 1e-9  # Γ 는 NAC 방향 의존이라 비교 제외
    df = np.abs(W[ok] - tp["freq"][ok]).max()
    if df > 1e-2:
        raise SystemExit(f"tau 파일 진동수와 이 fc2 의 진동수가 다름 (최대 {df:.3f} THz) — 같은 fc2·셀로 만든 tau 파일이어야 함")
    g = np.array(tp["gamma"])  # [T, nir, nb]
    G = np.full((m, m, m) + g.shape[::2], np.nan)
    for i, q in enumerate(qir):
        for x in md.orbit(q)[0]:
            G[tuple(np.rint(x * m).astype(int) % m)] = g[:, i]
    assert not np.isnan(G).any()  # 궤도로 전 격자가 채워져야 함
    W = np.full((m, m, m, 3), np.nan)
    for i, q in enumerate(qir):
        for x in md.orbit(q)[0]:
            W[tuple(np.rint(x * m).astype(int) % m)] = np.maximum(tp["freq"][i, :3], 1e-3)
    Gac = G[..., :3] / W[:, :, :, None, :] ** 2
    nb6 = np.vstack([np.eye(3, dtype=int), -np.eye(3, dtype=int)]) % m
    Gac[0, 0, 0] = np.mean([Gac[tuple(e)] for e in nb6], 0)  # Γ 점: 0/0 → 이웃 평균
    return G, Gac, df


def interp(G, K):
    """주기 삼선형: K[nk,3] 분수좌표 → [T, nk, nb]."""
    m = G.shape[0]
    x = (K % 1.0) * m
    i0 = np.floor(x).astype(int)
    f = x - i0
    out = 0
    for c in np.ndindex(2, 2, 2):
        wgt = np.prod(np.where(np.array(c), f, 1 - f), 1)
        idx = (i0 + c) % m
        out = out + wgt[:, None, None] * G[idx[:, 0], idx[:, 1], idx[:, 2]]
    return out.transpose(1, 0, 2)


def run(a):
    ph = load(a)
    L = V.Lattice(ph)
    tp = np.load(a.tau_file)
    TS = np.asarray(tp["TS"], float)
    G, Gac, df = gamma_grid(ph, tp)
    m = G.shape[0]
    ok = np.isfinite(tp["tau_bin"])
    nq = a.nq + a.nq % 2
    if L.ksigns is None:
        raise SystemExit("IBZ 미지원 구조 (축별 부호반전 대칭 아님)")  # ponytail: 비대각 대칭이면 전체 격자판을 만들 것
    safe = V.safe_half_shifts(L, nq)
    delta = np.array(safe[a.seed % len(safe)]) / (2 * nq)
    S = np.array(L.ksigns)
    k1 = np.linspace(-0.5, 0.5, nq, endpoint=False) + 0.5 / nq
    off = np.rint(2 * nq * (k1[0] + delta)).astype(np.int64)
    N, M = nq**3, 2 * nq
    wrap = lambda x: (x + nq - 1) % M - nq + 1
    acc = np.zeros((3, len(a.eta), len(TS), 9))  # P, P′ 모드, P′ bin
    edges = np.arange(0.0, 30.01, 0.25)  # THz — 주파수 분해 (어느 모드가 P′ 를 만드나)
    spec = np.zeros((3, len(a.eta), len(TS), len(edges) - 1, 9))
    t0, nk = time.time(), 0
    eb = a.block * len(S)
    for i in range(0, N, eb):  # 기약 대표만 (nphe_vec.P_oneshift_ibz 와 같은 판정)
        J = np.stack(np.unravel_index(np.arange(i, min(i + eb, N)), (nq,) * 3), -1)
        K2 = wrap(2 * J + off)
        img = wrap(K2[:, None, :] * S[None])
        code = ((img[..., 0] + nq) * (M + 1) + (img[..., 1] + nq)) * (M + 1) + (img[..., 2] + nq)
        own = ((K2[:, 0] + nq) * (M + 1) + (K2[:, 1] + nq)) * (M + 1) + (K2[:, 2] + nq)
        keep = own == code.max(1)
        if not keep.any():
            continue
        mult = 1 + (np.diff(np.sort(code[keep], 1), axis=1) != 0).sum(1)
        K = K2[keep] / M
        nk += len(K)
        w, v, Oms = V.modes(L, K, a.eta)
        g = interp(G, fold(K))  # [T, nk, nb]
        near = np.abs(fold(K)).max(1) * m < 1  # Γ 점에 닿는 칸
        if near.any():
            g[:, near, :3] = interp(Gac, fold(K[near])) * np.maximum(w[near, :3], 1e-3)[None] ** 2
        if a.boundary:  # Γ_b = |v|/(4πL) [THz] (phono3py 와 같은 식), v m/s, L μm
            g = g + (np.linalg.norm(v, axis=-1) / (a.boundary * 1e-6) * 1e-12 / (4 * np.pi))[None]
        tau = 1.0 / (4 * np.pi * np.maximum(g, 1e-8))  # ps
        taub = np.array([np.interp(w, tp["ctr"][ok[it]], tp["tau_bin"][it][ok[it]]) for it in range(len(TS))])
        pref = V.thermal_pref(w, TS) * mult[None, :, None]
        for ie, Om in enumerate(Oms):
            vO = (v[:, :, :, None] * Om[:, :, None, :]).reshape(len(K), -1, 9)
            nf = len(edges) - 1
            ib = np.clip(np.digitize(w, edges) - 1, 0, nf - 1).ravel()
            flat = (np.arange(len(TS))[:, None] * nf + ib[None]).ravel()  # (T, bin) 한 줄 번호 — spec_vec 과 같은 bincount
            vO = vO.reshape(-1, 9)
            for j, t in enumerate((1.0, tau, taub)):
                pt = (pref * t).reshape(len(TS), -1)
                for c in range(9):
                    spec[j, ie, :, :, c] += np.bincount(flat, (pt * vO[None, :, c]).ravel(), minlength=len(TS) * nf).reshape(len(TS), nf)
        print(f"  {min(i + eb, N) / N:6.1%}  {time.time() - t0:.0f}s", flush=True)
    acc = spec.sum(3)
    norm = 1.0 / (L.Vcell * 1e-30 * TS**2 * N) / (V.E_CHG * V.KB**2)  # → eV·kB² (·ps)
    acc = V.sym_project(L.Rc, acc.reshape(*acc.shape[:3], 3, 3)) * norm[None, None, :, None, None]
    spec = V.sym_project(L.Rc, spec.reshape(*spec.shape[:4], 3, 3)) * norm[None, None, :, None, None, None]
    P0, Pm, Pb = acc[:, None]  # [shift=1, η, T, 3, 3]
    np.savez(a.out, P=Pm, P_notau=P0, P_bin=Pb, spec=spec, EDGES=edges, TS=TS, etas=np.array(a.eta), nq=nq, delta=delta, tau_mesh=int(tp["mesh"]))
    print(f"[{L.sg}, nq {nq}³ IBZ {nk} 점, τ from {a.tau_file} mesh {int(tp['mesh'])}³ (진동수 일치 {df:.1e} THz), {time.time() - t0:.0f}s]")
    allow = [(x, y) for x in range(3) for y in range(3) if np.abs(Pm[0, :, :, x, y]).max() > 1e-6 * np.abs(Pm).max()]
    for ie, e in enumerate(a.eta):
        print(f"η={e}  단위 P: meV·kB²,  P′: meV·kB²·ps,  τ_eff = P′/P [ps]")
        for x, y in allow:
            c = "xyz"[x] + "xyz"[y]
            print(f"  {'T':>5} {'P_' + c:>9} {'P′_' + c:>10} {'P′bin_' + c:>10} {'τ_eff':>7} {'τbin_eff':>8}")
            for T in (30, 80, 150, 300):
                it = int(np.argmin(np.abs(TS - T)))
                p0, pm, pb = (1e3 * X[0, ie, it, x, y] for X in (P0, Pm, Pb))
                print(f"  {TS[it]:>5.0f} {p0:>+9.4f} {pm:>+10.4f} {pb:>+10.4f} {pm / p0:>7.2f} {pb / p0:>8.2f}")
    print("->", a.out, "  (κ_H: pt kappa --P", a.out, "--tau 1 --tau-file", a.tau_file, ")")
