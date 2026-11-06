#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aphe.py — FIELD-INDUCED phonon thermal Hall effect (PHE).

    kappa_ab(T,B) = -(kB^2 T)/(hbar V) * (1/Nk) * sum_q sum_{n=1..3N} c2(x_nq) Omega^g_n(q)

NAMING WARNING — the file name is wrong and is kept only for compatibility.
    What this computes is the ORDINARY (field-induced) phonon thermal Hall
    effect, not an anomalous one.  The only time-reversal-breaking term is the
    external field, entering through

        G = -2A,   A_k = (e/4 M_k)(Z_k^T [B]x + [B]x Z_k)      <- linear in B

    so B = 0 gives G = 0 and kappa == 0 identically.  The CLI takes the odd-in-B
    part, which is the linear-response (ordinary Hall) regime by construction.

    "Anomalous" would require TRS to be broken WITHOUT an applied field
    (magnetic order, or an intrinsic mechanism).  Shear lowers Pca2_1 -> P2_1,
    breaking inversion and the glides, but NOT time reversal: at B = 0 one still
    has Omega_n(-k) = -Omega_n(k), hence kappa_xy == 0 no matter how large the
    shear.  So no amount of strain alone makes this anomalous.

    Use "field-induced phonon thermal Hall effect" in text.  Renaming the module
    would break `import aphe` in ~40 scripts, so the name stays.

Formalism (Qin-Zhou-Shi PRB 86, 104305; Sun-Gao-Wang PRB 103, 214301)
    gyroscopic EOM  w.. + 2A w. + D w = 0,   A_k = (e/4 M_k)(Z_k^T [B]x + [B]x Z_k)
    6N linearisation  H = [[0, i I], [-i D, i G]],  G = -2A      (Sun-Wang Eq.8)
    velocity          V_a = dH/dq_a = [[0,0],[-i dD/dq_a, 0]]
    Berry curvature   Omega^g_n = -Im sum_{m!=n} (Va_nm Vb_mn - Vb_nm Va_mn) / Den_nm
                      with L^dag = R^-1 (biorthogonal), Den_nm = (w_n-w_m)^2 + eta^2

VERIFIED (see `--verify`): at B=0 the 6N expression above, summed over ALL 6N
intermediate states, reproduces the standard 3N phonon Berry curvature
    Omega_n = -2 Im sum_{m!=n} <n|dD_a|m><m|dD_b|n> / (w_n^2 - w_m^2)^2
to machine precision.  The 1/(4 w_n w_m) factors carried by the +/- branch pairs
cancel exactly.  Therefore do NOT replace dD by dD^(1/2): the D^(1/2) Hamiltonian
[[0, i D^1/2],[-i D^1/2, i G]] is the SAME operator conjugated by the
k-DEPENDENT similarity S = diag(D^1/2, 1); using dD^(1/2) together with the 6N
state sum double-counts that transformation.

Units: Omega in Ang^2, V in Ang^3, omega in THz (ordinary frequency, not angular).

CLI
    python aphe.py --dir HfO2/DFT/shear --supercell 2 2 2 --axis z --B 10 \
                   --T 5 10 20 50 100 200 300 --nq 16 --save out.json --bands
    # acoustic branches only (post-processing selection, same cost):
    python aphe.py --dir HfO2/DFT/shear --supercell 2 2 2 --nq 16 --band-select 1 2 3
    # or by frequency window instead of band index (survives band crossings):
    python aphe.py --dir HfO2/DFT/shear --supercell 2 2 2 --nq 16 --wmax 2.0
    python aphe.py --dir HfO2/DFT/shear --supercell 2 2 2 --verify
"""
import argparse, os, time, json
import numpy as np
from scipy.special import spence
import phonopy

# ---------------- SI constants ----------------
E_CHG      = 1.602176634e-19
AMU        = 1.66053906660e-27
KB         = 1.380649e-23
HBAR       = 1.054571817e-34
TWO_PI_THZ = 2*np.pi*1e12          # THz(ordinary) -> rad/s

AXIS_CROSS = {'z': np.array([[0.,-1,0],[1,0,0],[0,0,0]]),
              'x': np.array([[0.,0,0],[0,0,-1],[0,1,0]]),
              'y': np.array([[0.,0,1],[0,0,0],[-1,0,0]])}
AXIS_PLANE = {'z': (0,1), 'x': (1,2), 'y': (2,0)}
KNAME      = {'z':'kappa_xy', 'x':'kappa_yz', 'y':'kappa_zx'}

# --axis accepts either the field/curvature axis ('z','x','y') or the tensor
# component itself.  kappa_xz = -kappa_zx, so it reuses axis 'y' with sign -1.
COMPONENT  = {'z':  ('z', +1, 'kappa_xy'), 'xy': ('z', +1, 'kappa_xy'),
              'x':  ('x', +1, 'kappa_yz'), 'yz': ('x', +1, 'kappa_yz'),
              'y':  ('y', +1, 'kappa_zx'), 'zx': ('y', +1, 'kappa_zx'),
              'xz': ('y', -1, 'kappa_xz')}


# ================= phonon input =================
def load_phonon(d, supercell, nac=True):
    fc_kw = {}
    if os.path.exists(f"{d}/FORCE_SETS"):
        fc_kw['force_sets_filename'] = f"{d}/FORCE_SETS"
    elif os.path.exists(f"{d}/FORCE_CONSTANTS"):
        fc_kw['force_constants_filename'] = f"{d}/FORCE_CONSTANTS"
    else:
        raise FileNotFoundError(f"{d}: need FORCE_SETS or FORCE_CONSTANTS")
    born = f"{d}/BORN" if (nac and os.path.exists(f"{d}/BORN")) else None
    return phonopy.load(unitcell_filename=f"{d}/POSCAR",
                        born_filename=born,
                        supercell_matrix=list(supercell),
                        primitive_matrix=np.eye(3),
                        calculator="vasp", is_nac=born is not None,
                        log_level=0, **fc_kw)


class MagPhonon:
    """Dynamical matrix + magnetic (gyroscopic) coupling.

    zstar : optional (natom,3,3) Born effective charges used ONLY for the
        magnetic coupling G.  Z* plays two independent roles here:

          (1) the non-analytic correction inside D(q)  -> LO-TO splitting
          (2) the field coupling A = (e/4M)(Z^T [B]x + [B]x Z)  -> the whole Hall response

        They are separable.  A short-range MLFF (MACE etc.) has no dipole-dipole
        tail in its force constants, so adding a DFT NAC to D double-counts and
        drives the acoustic branch imaginary -- but dropping Z* entirely sets
        G = 0 and kappa vanishes identically.  The fix is to load the phonons
        with nac=False and pass the DFT Born charges here: no LO-TO splitting
        (a stated approximation), correct charge-field coupling.
    """

    def __init__(self, ph, zstar=None):
        self.dm     = ph.dynamical_matrix
        self.A      = np.array(ph.primitive.cell)          # rows = lattice vectors [Ang]
        self.Brec   = 2*np.pi*np.linalg.inv(self.A).T      # rows = reciprocal vectors
        self.Binv   = np.linalg.inv(self.Brec)
        self.masses = np.array(ph.primitive.masses)
        self.factor = ph.unit_conversion_factor
        self.n      = 3*len(self.masses)
        self.Vcell  = abs(np.linalg.det(self.A))           # Ang^3
        if zstar is not None:
            self.Zstar = np.asarray(zstar, float)
            if self.Zstar.shape != (len(self.masses), 3, 3):
                raise ValueError(f"zstar must be ({len(self.masses)},3,3), "
                                 f"got {self.Zstar.shape}")
        elif ph.nac_params is not None:
            self.Zstar = np.array(ph.nac_params['born'])
        else:
            raise ValueError(
                "no Born charges: the magnetic coupling A = (e/4M)(Z^T[B]x + "
                "[B]xZ) would be zero and kappa identically 0. Load with "
                "nac=True, or pass zstar= explicitly (see class docstring).")
        self.set_field(0.0)

    # k_frac (reduced) -> k_cart [1/Ang]:  k_cart = Brec^T @ k_frac
    def cart(self, kf):   return self.Brec.T @ np.asarray(kf, float)
    def frac(self, kc):   return self.Binv.T @ np.asarray(kc, float)

    def set_field(self, Bmag, axis='z'):
        """G = -2A,  A_k = (e/4M_k)(Z^T [B]x + [B]x Z).  Result in THz."""
        self.axis, self.Bmag = axis, Bmag
        Jx = AXIS_CROSS[axis]
        G = np.zeros((self.n, self.n))
        for k in range(len(self.masses)):
            Z, Mk = self.Zstar[k], self.masses[k]*AMU
            G[3*k:3*k+3, 3*k:3*k+3] = -(E_CHG/(2*Mk))*(Z.T@Jx + Jx@Z)*Bmag/TWO_PI_THZ
        self.G = G

    def Dphys(self, k_cart):
        """Dynamical matrix in THz^2 (eigenvalues = omega^2)."""
        self.dm.run(self.frac(k_cart))
        return np.array(self.dm.dynamical_matrix) * self.factor**2


# ================= Berry curvature =================
def berry(mb, kf, axis='z', h=1e-4, eta=0.0):
    """Return (omega[3N] THz, Omega[3N] Ang^2) for the positive branches at k_frac=kf.

    eta: real Lorentzian regulariser, Den = (w_n-w_m)^2 + eta^2  [THz^2].
         eta=0 gives the bare Kubo expression.
    """
    n  = mb.n
    kc = mb.cart(kf)
    a, b = AXIS_PLANE[axis]
    ea = np.zeros(3); ea[a] = h
    eb = np.zeros(3); eb[b] = h

    D0  = mb.Dphys(kc)
    dDa = (mb.Dphys(kc + ea) - mb.Dphys(kc - ea))/(2*h)
    dDb = (mb.Dphys(kc + eb) - mb.Dphys(kc - eb))/(2*h)

    I, Z = np.eye(n), np.zeros((n, n))
    H  = np.block([[Z, 1j*I], [-1j*D0, 1j*mb.G]])
    VA = np.block([[Z, Z], [-1j*dDa, Z]])
    VB = np.block([[Z, Z], [-1j*dDb, Z]])

    w, R = np.linalg.eig(H)
    try:
        Ldag = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Ldag = np.linalg.pinv(R)

    Ma = Ldag @ VA @ R
    Mb = Ldag @ VB @ R

    dw  = w[:, None] - w[None, :]
    Den = dw*dw + eta*eta                    # symmetric; real-positive when eta>0
    np.fill_diagonal(Den, np.inf)
    # fully antisymmetrised Kubo form (correct also for non-Hermitian H)
    Om = -((Ma*Mb.T - Mb*Ma.T)/Den).sum(axis=1).imag

    pos = _positive_branches(w, n, kf)
    return w[pos].real, Om[pos]


def _positive_branches(w, n, kf=None, tol=1e-8, warn=True):
    """Pick the 3N physical (positive-frequency) branches of the 6N spectrum.

    Sorting by Re(w) and taking the upper half silently assumes the spectrum
    splits exactly 3N/3N.  That fails where an acoustic branch has gone
    imaginary: the +-|w| pair collapses onto Re(w) = 0 and the split becomes
    e.g. 35/35 with 2 near-zero states (measured at k_frac = (0.005,0.005,0.005)
    with the 2x2x2 force constants, where max|Im w| = 6.5e-2 THz).  The upper
    half is then not the physical set and band n no longer means the same mode
    at +B and -B.

    Production grids (shifted Monkhorst, nq <= 20, min |k| = 0.025) never reach
    that region, but the failure must be loud rather than silent.
    """
    npos = int((w.real > tol).sum())
    if npos != n and warn:
        nneg = int((w.real < -tol).sum())
        nzero = int(len(w) - npos - nneg)
        print(f"  [aphe] WARNING 6N branch split is {npos}/{nneg} (+{nzero} near zero) "
              f"instead of {n}/{n} at k_frac={np.round(kf, 6) if kf is not None else '?'}"
              f"; max|Im w| = {np.abs(w.imag).max():.2e} THz. "
              f"An acoustic branch is imaginary here - force constants, not the code. "
              f"Band assignment at this k is not trustworthy.")
    idx = np.argsort(w.real)
    pos = idx[n:]
    return pos[np.argsort(w.real[pos])]


def berry_vector(mb, kf, h=1e-4, eta=0.0):
    """All three Berry-curvature components at once: (omega[3N], Omega[3N,3]).

    One eigendecomposition and six dynamical-matrix evaluations serve all three
    components, so this is ~3x cheaper than calling `berry` per axis.  The field
    direction is whatever `mb.set_field` last set -- the curvature VECTOR under
    one fixed B, which is what a texture plot needs.

    Omega[:,0] = Omega^x (plane y,z),  [:,1] = Omega^y (z,x),  [:,2] = Omega^z (x,y)
    """
    n  = mb.n
    kc = mb.cart(kf)
    D0 = mb.Dphys(kc)
    dD = []
    for a in range(3):
        ea = np.zeros(3); ea[a] = h
        dD.append((mb.Dphys(kc + ea) - mb.Dphys(kc - ea))/(2*h))

    I, Z = np.eye(n), np.zeros((n, n))
    H = np.block([[Z, 1j*I], [-1j*D0, 1j*mb.G]])
    w, R = np.linalg.eig(H)
    try:
        Ldag = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        Ldag = np.linalg.pinv(R)
    M = [Ldag @ np.block([[Z, Z], [-1j*d, Z]]) @ R for d in dD]

    dw  = w[:, None] - w[None, :]
    Den = dw*dw + eta*eta
    np.fill_diagonal(Den, np.inf)

    pos = _positive_branches(w, n, kf)

    out = np.empty((n, 3))
    for g, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
        Om = -((M[a]*M[b].T - M[b]*M[a].T)/Den).sum(axis=1).imag
        out[:, g] = Om[pos]
    return w[pos].real, out


def angular_momentum(mb, kf):
    """Phonon angular momentum of every branch at k_frac=kf, in units of hbar.

        l^a_n(k) = sum_kappa  eps_n(kappa)^dag  M_a  eps_n(kappa),
        M_a = i * [e_a]_x      (so l^z = 2 Im[eps*_x eps_y], etc.)

    Uses the B=0 Hermitian eigenvectors of D, i.e. the standard definition
    (Zhang & Niu).  Returns (omega[3N] THz, L[3N,3] in hbar).

    NOTE: l is a property of the eigenvector AT ONE k; the Berry curvature is a
    property of how that eigenvector CHANGES with k.  A perfectly circularly
    polarised mode (maximal l) can have zero Berry curvature.  Large l does not
    imply large Omega, and only Omega enters kappa.
    """
    D = mb.Dphys(mb.cart(kf))
    D = (D + D.conj().T)/2
    w2, E = np.linalg.eigh(D)                      # E[:, n] = eps_n (3N,)
    nat = len(mb.masses)
    L = np.zeros((mb.n, 3))
    for ai, ax in enumerate(('x', 'y', 'z')):
        M = 1j*AXIS_CROSS[ax]
        for k in range(nat):
            e = E[3*k:3*k+3, :]                    # (3, 3N)
            L[:, ai] += np.einsum('in,ij,jn->n', e.conj(), M, e).real
    return np.sqrt(np.clip(w2, 0, None)), L


def berry_std(mb, kf, axis='z', h=1e-4):
    """Reference: standard 3N phonon Berry curvature at B=0 (validation only)."""
    kc = mb.cart(kf)
    a, b = AXIS_PLANE[axis]
    ea = np.zeros(3); ea[a] = h
    eb = np.zeros(3); eb[b] = h
    D0  = mb.Dphys(kc); D0 = (D0 + D0.conj().T)/2
    dDa = (mb.Dphys(kc + ea) - mb.Dphys(kc - ea))/(2*h)
    dDb = (mb.Dphys(kc + eb) - mb.Dphys(kc - eb))/(2*h)
    w2, E = np.linalg.eigh(D0)
    Xa, Xb = E.conj().T @ dDa @ E, E.conj().T @ dDb @ E
    den = (w2[:, None] - w2[None, :])**2
    np.fill_diagonal(den, np.inf)
    return np.sqrt(np.clip(w2, 0, None)), -2*((Xa*Xb.T/den).sum(axis=1)).imag


# ================= thermal weight =================
def c2_weight(x):
    """c2(rho) with rho = 1/(exp(x)-1), x = hbar*omega/(kB T).

    Uses the exact identity ln((1+rho)/rho) == x, plus an asymptotic branch for
    x > 30 where the literal formula loses every significant digit to
    cancellation between (1+rho)x^2 and ln^2(rho).
    """
    x   = np.atleast_1d(np.asarray(x, float))
    out = np.empty_like(x)
    big = x > 30.0
    xb  = x[big]
    out[big] = np.exp(-xb)*(xb*xb + 2*xb + 2)
    xs = x[~big]
    if xs.size:
        rho = 1.0/np.expm1(np.clip(xs, 1e-12, None))
        out[~big] = (1 + rho)*xs**2 - np.log(rho)**2 - 2*spence(1.0 + rho).real
    return out


# ================= BZ scan =================
def bz_grid(nq, nqz):
    k1 = np.linspace(-0.5, 0.5, nq, endpoint=False) + 0.5/nq
    k3 = (np.linspace(-0.5, 0.5, nqz, endpoint=False) + 0.5/nqz) if nqz > 1 else np.array([0.0])
    return np.array([[a, b, c] for a in k1 for b in k1 for c in k3])


def scan(mb, Bmag, nq, nqz, axis='z', eta=0.0, h=1e-4, verbose=True):
    """Return K, OM(Nk,3N) [THz], OZ(Nk,3N) [Ang^2] on the shifted Monkhorst grid."""
    mb.set_field(Bmag, axis=axis)
    K  = bz_grid(nq, nqz)
    OM = np.empty((len(K), mb.n)); OZ = np.empty_like(OM)
    t0 = time.time()
    for i, kf in enumerate(K):
        OM[i], OZ[i] = berry(mb, kf, axis=axis, h=h, eta=eta)
        if verbose and (i + 1) % max(1, len(K)//10) == 0:
            print(f"      {i+1}/{len(K)}  ({time.time()-t0:.0f}s)", flush=True)
    return K, OM, OZ


def kappa_from_scan(OM, OZ, Ts, Vcell_A3, wmin=0.0, wmax=None, bands=None):
    """kappa per T. Returns dict T -> (total, per_band[3N]) in W/m/K.

    The per-band array is ALWAYS the full 3N decomposition; `bands`/`wmin`/`wmax`
    only restrict what goes into `total`.

    bands : 0-based band indices to keep in the total (None = all).
            Band n is the n-th lowest branch AT EACH k, so a selection follows
            the branch through the BZ but swaps partners at band crossings --
            use `wmin`/`wmax` instead when crossings matter.
    wmin, wmax : keep only modes with wmin < omega <= wmax  [THz].

    NOTE: Omega_n itself needs the full sum over the other 6N states, so
    selecting bands saves no computation -- it is a post-processing choice.
    """
    Nk, nb = OM.shape
    V  = Vcell_A3 * 1e-30                      # Ang^3 -> m^3
    mask = OM > wmin
    if wmax is not None:
        mask &= OM <= wmax
    keep = np.zeros(nb, bool)
    keep[list(bands) if bands is not None else slice(None)] = True
    res = {}
    for T in Ts:
        x  = np.where(mask, HBAR*np.abs(OM)*TWO_PI_THZ/(KB*T), 1.0)
        c2 = c2_weight(x.ravel()).reshape(x.shape)
        term = np.where(mask, c2*OZ, 0.0)      # (Nk, 3N)
        pref = -(KB**2 * T)/(HBAR * V) * 1e-20 / Nk
        per_band = pref*term.sum(axis=0)
        res[T] = (per_band[keep].sum(), per_band)
    return res


def grid_report(OM, soft=0.2):
    lo = OM.min(axis=1)
    return dict(n_k=int(OM.shape[0]),
                n_imag=int((lo <= 0).sum()),
                n_soft=int((lo < soft).sum()),
                omega_min=float(lo.min()),
                omega_max=float(OM.max()))


# ================= verification =================
def verify(mb, axis='z', h=1e-4):
    ok = True
    print("--- 1) B=0: 6N Kubo vs standard 3N Berry curvature ---")
    mb.set_field(0.0, axis)
    for kf in ([0.137, 0.221, 0.083], [0.31, -0.17, 0.42]):
        _,  O6 = berry(mb, kf, axis=axis, h=h, eta=0.0)
        _,  O3 = berry_std(mb, kf, axis=axis, h=h)
        d = np.max(np.abs(O6 - O3))/max(1e-30, np.max(np.abs(O3)))
        print(f"    k={kf}  max rel.dev = {d:.3e}   sum(Om)_6N = {O6.sum():+.2e}")
        ok &= d < 1e-6

    print("--- 2) Omega odd in B (Onsager) ---")
    kf = [0.137, 0.221, 0.083]
    mb.set_field(+10.0, axis); _, Op  = berry(mb, kf, axis=axis, h=h)
    mb.set_field(-10.0, axis); _, Omn = berry(mb, kf, axis=axis, h=h)
    mb.set_field(0.0, axis);   _, O0  = berry(mb, kf, axis=axis, h=h)
    odd = np.max(np.abs(0.5*(Op + Omn) - O0))/max(1e-30, np.max(np.abs(Op - O0)))
    print(f"    |even part - Omega(B=0)| / |odd part| = {odd:.3e}")
    ok &= odd < 1e-2

    print("--- 3) c2 limits ---")
    print(f"    c2(x->0) = {c2_weight([1e-9])[0]:.6f}   (pi^2/3 = {np.pi**2/3:.6f})")
    print(f"    c2(x=40) = {c2_weight([40.0])[0]:.6e}  (asymptotic branch)")
    ok &= abs(c2_weight([1e-9])[0] - np.pi**2/3) < 1e-6

    print("--- 4) finite-difference step h ---")
    mb.set_field(10.0, axis)
    for hh in (1e-3, 3e-4, 1e-4, 3e-5):
        _, O = berry(mb, kf, axis=axis, h=hh)
        print(f"    h={hh:8.1e}  Omega[0:3] = {np.array2string(O[:3], precision=5)}")

    print("--- 5) units: phonopy frequencies vs Dphys eigenvalues ---")
    for kf in ([0.137, 0.221, 0.083], [0.02, 0.05, 0.01]):
        Dm = mb.Dphys(mb.cart(kf)); Dm = (Dm + Dm.conj().T)/2
        w2 = np.linalg.eigvalsh(Dm)
        wme = np.sign(w2)*np.sqrt(np.abs(w2))
        print(f"    k={kf}  omega range [{wme.min():.4f}, {wme.max():.3f}] THz "
              f"(compare with `phonopy --qpoints`)")

    print("--- 6) 6N branch split and conditioning ---")
    ok6 = True
    for kf in ([0.137, 0.221, 0.083], [0.02, 0.05, 0.01], [0.005, 0.005, 0.005]):
        n = mb.n
        D0 = mb.Dphys(mb.cart(kf))
        I, Z = np.eye(n), np.zeros((n, n))
        H = np.block([[Z, 1j*I], [-1j*D0, 1j*mb.G]])
        w, R = np.linalg.eig(H)
        npos = int((w.real > 1e-8).sum()); nneg = int((w.real < -1e-8).sum())
        cond = np.linalg.cond(R)
        bi = np.linalg.norm(np.linalg.inv(R) @ R - np.eye(2*n))
        flag = "" if npos == n and nneg == n else "  <-- SPLIT BROKEN"
        print(f"    k={str(kf):22s} {npos}/{nneg}  max|Im w|={np.abs(w.imag).max():.2e}"
              f"  cond(R)={cond:.2e}  ||L R - I||={bi:.1e}{flag}")
        ok6 &= (cond < 1e8 and bi < 1e-8)
    ok &= ok6

    print("--- 7) TRS at B=0:  w(q)=w(-q),  Omega(q)=-Omega(-q) ---")
    mb.set_field(0.0, axis)
    for kf in ([0.137, 0.221, 0.083], [0.31, -0.17, 0.42]):
        q = np.array(kf)
        wp, Op = berry(mb, q, axis=axis, h=h)
        wm, Omg = berry(mb, -q, axis=axis, h=h)
        dwq = np.max(np.abs(wp - wm))
        dOm = np.max(np.abs(Op + Omg))/max(np.max(np.abs(Op)), 1e-30)
        print(f"    k={kf}  max|dw|={dwq:.2e} THz   |Om(q)+Om(-q)|/max|Om|={dOm:.2e}")
        ok &= dOm < 1e-6

    print("--- 8) dispersion health near Gamma ---")
    for s in (0.30, 0.10, 0.05, 0.02, 0.01):
        Dm = mb.Dphys(mb.cart([s, 0.3*s, 0.0])); Dm = (Dm + Dm.conj().T)/2
        w2 = np.linalg.eigvalsh(Dm)[:3]
        flag = "  <-- IMAGINARY" if w2.min() < 0 else ""
        print(f"    |k_frac|~{s:5.3f}  omega[0:3] = "
              f"{np.array2string(np.sign(w2)*np.sqrt(np.abs(w2)), precision=4)}{flag}")

    print(f"\n  VERIFY: {'PASS' if ok else 'FAIL'}")
    return ok


# ================= CLI =================
def main():
    p = argparse.ArgumentParser(description="field-induced phonon thermal Hall conductivity (PHE)")
    p.add_argument("--dir", required=True)
    p.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2])
    p.add_argument("--axis", default="z", choices=list(COMPONENT),
                   help="field/curvature axis or tensor component: "
                        "z|xy -> kappa_xy, x|yz -> kappa_yz, y|zx -> kappa_zx, "
                        "xz -> kappa_xz (= -kappa_zx, same scan as 'y')")
    p.add_argument("--B", type=float, default=10.0, help="field [T]")
    p.add_argument("--T", type=float, nargs="+",
                   default=[5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300])
    p.add_argument("--nq",  type=int, default=16)
    p.add_argument("--nqz", type=int, default=None,
                   help="k3 grid; default = nq (3D). Use 1 only for a genuine 2D slab.")
    p.add_argument("--eta", type=float, default=0.0,
                   help="Lorentzian regulariser [THz], Den=(dw)^2+eta^2. Default 0.")
    p.add_argument("--h", type=float, default=1e-4, help="finite-difference step [1/Ang]")
    p.add_argument("--wmin", type=float, default=0.0,
                   help="drop modes with omega<=wmin [THz] from the total")
    p.add_argument("--wmax", type=float, default=None,
                   help="drop modes with omega>wmax [THz] from the total")
    p.add_argument("--band-select", type=int, nargs="+", default=None, metavar="N",
                   help="1-based band indices to include in the total "
                        "(e.g. --band-select 1 2 3 for the acoustic branches). "
                        "Band n = n-th lowest branch at each k. Does not speed "
                        "anything up: Omega_n still needs every other band.")
    p.add_argument("--no-nac", action="store_true")
    p.add_argument("--verify", action="store_true", help="internal consistency checks")
    p.add_argument("--bands", action="store_true", help="band-resolved kappa_n(T)")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--plot", type=str, default=None)
    a = p.parse_args()
    requested = a.axis
    a.axis, csign, cname = COMPONENT[a.axis]   # canonical axis, sign, label

    ph = load_phonon(a.dir, a.supercell, nac=not a.no_nac)
    mb = MagPhonon(ph)
    nqz = a.nq if a.nqz is None else a.nqz

    print(f"[PHE] {a.dir}")
    print(f"       {len(mb.masses)} atoms, 3N={mb.n}, V={mb.Vcell:.3f} Ang^3, "
          f"NAC={'off' if a.no_nac else 'on'}")

    if a.verify:
        verify(mb, axis=a.axis, h=a.h)
        return

    print(f"       B={a.B} T, component={cname} "
          f"(--axis {requested} -> scan axis '{a.axis}' = {KNAME[a.axis]}, "
          f"sign {csign:+d}), grid={a.nq}x{a.nq}x{nqz}, "
          f"eta={a.eta} THz, h={a.h}")
    if nqz == 1:
        print("       WARNING: nqz=1 samples only the k3=0 plane. For a 3D crystal "
              "this is NOT the BZ integral.")

    t0 = time.time()
    print(f"  scan B=+{a.B} T", flush=True)
    _, OMp, OZp = scan(mb, +a.B, a.nq, nqz, a.axis, a.eta, a.h)
    print(f"  scan B=-{a.B} T", flush=True)
    _, OMm, OZm = scan(mb, -a.B, a.nq, nqz, a.axis, a.eta, a.h)

    rep = grid_report(OMp)
    print(f"\n  grid: {rep['n_k']} pts, omega in [{rep['omega_min']:.4f}, "
          f"{rep['omega_max']:.3f}] THz")
    if rep['n_imag']:
        print(f"  WARNING: {rep['n_imag']}/{rep['n_k']} k-points have a non-positive "
              f"lowest branch (FC interpolation / supercell too small).")
    if rep['n_soft']:
        print(f"  note: {rep['n_soft']}/{rep['n_k']} k-points have omega_min < 0.2 THz; "
              f"these dominate the low-T result.")

    sel = None if a.band_select is None else [n-1 for n in a.band_select]
    if sel is not None:
        if min(sel) < 0 or max(sel) >= mb.n:
            p.error(f"--band-select must be within 1..{mb.n}")
        print(f"\n  TOTAL RESTRICTED TO BANDS {a.band_select} "
              f"(of {mb.n}); band n = n-th lowest branch at each k")
    if a.wmax is not None or a.wmin > 0:
        print(f"  frequency window: {a.wmin} < omega <= "
              f"{a.wmax if a.wmax is not None else 'inf'} THz")

    # Hall response = odd-in-B part. Removes every even-in-B numerical residue.
    kp = kappa_from_scan(OMp, OZp, a.T, mb.Vcell, a.wmin, a.wmax, sel)
    km = kappa_from_scan(OMm, OZm, a.T, mb.Vcell, a.wmin, a.wmax, sel)

    print(f"\n[{cname}]  B={a.B} T   [W m^-1 K^-1]")
    print(f"   {'T [K]':>7}  {'kappa (odd in B)':>18}  {'even residue':>14}")
    ktot, kband = {}, {}
    for T in a.T:
        odd  = csign*0.5*(kp[T][0] - km[T][0])
        even = csign*0.5*(kp[T][0] + km[T][0])
        ktot[T]  = odd
        kband[T] = csign*0.5*(kp[T][1] - km[T][1])
        print(f"   {T:7.1f}  {odd:+18.5e}  {even:+14.2e}")

    if a.bands:
        show = list(a.T)
        hdr = "  band  " + "".join(f"{T:>13.0f}K" for T in show)
        ref = max(max(abs(v) for v in kband[T]) for T in show)
        print("\n  band-resolved kappa_n [W/m/K]  (band 1 = lowest branch at each k)")
        print(hdr)
        for n in range(mb.n):
            row = [kband[T][n] for T in show]
            if max(abs(v) for v in row) < 1e-4*ref:
                continue
            print(f"  {n+1:4d}  " + "".join(f"{v:+14.3e}" for v in row))
        print("\n  cumulative over bands 1..n  [W/m/K]")
        print(hdr)
        for n in range(mb.n):
            row = [kband[T][:n+1].sum() for T in show]
            print(f"  1-{n+1:<3d} " + "".join(f"{v:+14.3e}" for v in row))

    print(f"\n   [{time.time()-t0:.0f}s]")

    out = dict(dir=a.dir, B=a.B, axis=a.axis, component=cname, sign=csign,
               nq=a.nq, nqz=nqz, eta=a.eta, h=a.h,
               wmin=a.wmin, wmax=a.wmax, band_select=a.band_select,
               Vcell_A3=mb.Vcell, grid=rep,
               kappa={str(T): ktot[T] for T in a.T},
               kappa_band={str(T): kband[T].tolist() for T in a.T})
    if a.save:
        json.dump(out, open(a.save, 'w'), indent=2)
        print(f"   saved {a.save}")

    path = a.plot or (a.save.replace('.json', '_kappa.png') if a.save else None)
    if path:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        Ts = sorted(a.T)
        fig, ax = plt.subplots(1, 2 if a.bands else 1,
                               figsize=(11 if a.bands else 6, 4), squeeze=False)
        ax0 = ax[0][0]
        ax0.plot(Ts, [ktot[T] for T in Ts], 'o-', color='C0')
        ax0.axhline(0, color='gray', lw=.5)
        ax0.set_xlabel('T [K]'); ax0.set_ylabel(f'{cname} [W/m/K]')
        ax0.set_title(f'PHE {cname}  B={a.B} T  grid {a.nq}x{a.nq}x{nqz}')
        ax0.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if a.bands:
            ax1 = ax[0][1]
            kb = np.array([kband[T] for T in Ts])          # (nT, 3N)
            for n in range(min(9, mb.n)):
                ax1.plot(Ts, kb[:, n], label=f'band {n+1}')
            ax1.axhline(0, color='gray', lw=.5)
            ax1.set_xlabel('T [K]'); ax1.set_ylabel('band contribution [W/m/K]')
            ax1.set_title('band-resolved')
            ax1.legend(fontsize=7, ncol=2)
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
        print(f"   plot  {path}")


if __name__ == "__main__":
    main()
