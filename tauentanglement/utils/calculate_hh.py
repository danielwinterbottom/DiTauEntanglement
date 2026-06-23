"""
Python implementation of calculateHH from TauSpinner/tau_reweight_lib.cxx.

All multi-body decay channels are computed using pure Python reimplementations
of the TAUOLA Fortran subroutines in tauola_fortran.py.

Optionally, a compiled Fortran shared library can still be loaded via
TauolaFortran.load("libtauola.so") as a fall-back / cross-check.

Usage
-----
from tauentanglement.utils.calculate_hh import Particle, calculateHH

daughters = [Particle(px, py, pz, e, pdgid), ...]
HH, WTamplit = calculateHH(tau_pdgid, daughters, phi, theta)
"""

import math
import ctypes
import copy
from typing import List, Optional, Tuple

from tauentanglement.utils import tauola_fortran as _tf


# ---------------------------------------------------------------------------
# Particle
# ---------------------------------------------------------------------------

class Particle:
    """Minimal 4-vector + PDG-id, mirroring TauSpinner::Particle."""

    def __init__(self, px: float, py: float, pz: float, e: float, pdgid: int = 0):
        self._px = float(px)
        self._py = float(py)
        self._pz = float(pz)
        self._e  = float(e)
        self._pdgid = int(pdgid)

    def px(self) -> float: return self._px
    def py(self) -> float: return self._py
    def pz(self) -> float: return self._pz
    def e (self) -> float: return self._e
    def pdgid(self) -> int: return self._pdgid

    def rotateXZ(self, theta: float) -> None:
        """Rotation in the X-Z plane (as PHORO2 in photos.f)."""
        c, s = math.cos(theta), math.sin(theta)
        px, pz = self._px, self._pz
        self._px =  c * px + s * pz
        self._pz = -s * px + c * pz

    def rotateXY(self, theta: float) -> None:
        """Rotation in the X-Y plane (as PHORO3 in photos.f)."""
        c, s = math.cos(theta), math.sin(theta)
        px, py = self._px, self._py
        self._px = c * px - s * py
        self._py = s * px + c * py

    def getAnglePhi(self) -> float:
        """Azimuthal angle of (px, py), matching ANGFI/PHOAN1 conventions."""
        px, py = self._px, self._py
        if abs(py) < abs(px):
            buf = math.atan(abs(py / px)) if px != 0. else 0.
            if px < 0.: buf = math.pi - buf
        else:
            r = math.sqrt(px*px + py*py)
            buf = math.acos(px / r) if r > 0. else 0.
        if py < 0.: buf = 2.*math.pi - buf
        return buf

    def getAngleTheta(self) -> float:
        """Polar angle of (pz, px), matching ANGXY/PHOAN2 conventions."""
        px, pz = self._px, self._pz
        if abs(px) < abs(pz):
            buf = math.atan(abs(px / pz)) if pz != 0. else 0.
            if pz < 0.: buf = math.pi - buf
        else:
            r = math.sqrt(pz*pz + px*px)
            buf = math.acos(pz / r) if r > 0. else 0.
        return buf

    def boostAlongZ(self, p_pz: float, p_e: float) -> None:
        """Boost along Z by a particle with (pz, e), matching PHOBO3."""
        m = math.sqrt(p_e*p_e - p_pz*p_pz)
        buf_pz, buf_e = self._pz, self._e
        self._pz = (p_e  * buf_pz + p_pz * buf_e) / m
        self._e  = (p_pz * buf_pz + p_e  * buf_e) / m

    def boostToRestFrame(self, p: 'Particle') -> None:
        """Boost self into the rest frame of particle p (modifies self only)."""
        p_len = math.sqrt(p._px**2 + p._py**2 + p._pz**2)
        phi   = p.getAnglePhi()
        p.rotateXY(-phi)
        theta = p.getAngleTheta()
        p.rotateXY( phi)
        self.rotateXY(-phi)
        self.rotateXZ(-theta)
        self.boostAlongZ(-p_len, p._e)
        self.rotateXZ( theta)
        self.rotateXY( phi)

    def copy(self) -> 'Particle':
        return Particle(self._px, self._py, self._pz, self._e, self._pdgid)


def _step12_reference(
    tau: Particle,
    companion: Particle,
    reference: Particle,
) -> Particle:
    """
    Apply the same step-1 (boost to tau-tau COM) and step-2 (rotate tau to -Z)
    transformations used in _prepare_kinematic_for_hh to a reference particle.

    Used to determine the orientation of the production-plane axes in each tau's
    step-2 frame, without running the full kinematic preparation.

    Parameters
    ----------
    tau        : the tau whose phi/theta step-2 rotations to follow
    companion  : the other tau (needed to compute P_QQ)
    reference  : the reference direction particle (modified in place is NOT done;
                 a copy is made and returned)

    Returns
    -------
    The reference particle after step-1 boost and step-2 rotations.
    """
    P_QQ = Particle(tau.px() + companion.px(),
                    tau.py() + companion.py(),
                    tau.pz() + companion.pz(),
                    tau.e()  + companion.e(), 0)

    tau_c = tau.copy()
    ref_c = reference.copy()

    # Step 1: boost to COM
    tau_c.boostToRestFrame(P_QQ.copy())
    ref_c.boostToRestFrame(P_QQ.copy())

    # Step 2: same phi/theta rotation used on tau daughters
    phi   = tau_c.getAnglePhi()
    tau_c.rotateXY(-phi)
    theta = tau_c.getAngleTheta()
    tau_c.rotateXZ(math.pi - theta)

    ref_c.rotateXY(-phi)
    ref_c.rotateXZ(math.pi - theta)

    return ref_c


def _prepare_kinematic_for_hh(
    tau: Particle,
    nu_tau: Particle,
    tau_daughters: List[Particle],
) -> Tuple[float, float, float, float]:
    """
    Transform tau daughters into the frame needed by calculateHH.

    Mirrors TauSpinner::prepareKinematicForHH exactly:
      1. Boost tau + daughters to rest frame of (tau + nu_tau)
      2. Rotate so tau is along +Z
      3. Boost daughters along Z into tau rest frame
      4. Rotate so the tau-neutrino daughter is along +Z

    tau, nu_tau, and tau_daughters are modified in place.
    Returns (phi2, theta2, tau_com_pz, tau_com_e) where:
      - phi2, theta2  are the neutrino-alignment angles needed by calculateHH
      - tau_com_pz, tau_com_e  are the tau's momentum in the COM frame after
        step 2 (tau is along +Z at this point), needed to boost HH back to COM.
    """
    P_QQ = Particle(tau.px() + nu_tau.px(),
                    tau.py() + nu_tau.py(),
                    tau.pz() + nu_tau.pz(),
                    tau.e()  + nu_tau.e(), 0)

    # Step 1: boost everything to tau-tau COM rest frame
    tau.boostToRestFrame(P_QQ)
    nu_tau.boostToRestFrame(P_QQ)
    for d in tau_daughters:
        d.boostToRestFrame(P_QQ)

    # Step 2: rotate so tau is along +Z
    phi   = tau.getAnglePhi()
    tau.rotateXY(-phi)
    theta = tau.getAngleTheta()
    tau.rotateXZ(math.pi - theta)

    nu_tau.rotateXY(-phi)
    nu_tau.rotateXZ(math.pi - theta)
    for d in tau_daughters:
        d.rotateXY(-phi)
        d.rotateXZ(math.pi - theta)

    # Save tau's COM-frame momentum now (px=py=0, tau along +Z)
    tau_com_pz = tau.pz()
    tau_com_e  = tau.e()

    # Step 3: boost daughters along Z into tau rest frame
    for d in tau_daughters:
        d.boostAlongZ(-tau.pz(), tau.e())

    # Step 4: rotate so the tau-neutrino daughter is along +Z
    phi2 = theta2 = 0.0
    nu_idx = None
    for i, d in enumerate(tau_daughters):
        if abs(d.pdgid()) == 16:
            phi2  = d.getAnglePhi()
            d.rotateXY(-phi2)
            theta2 = d.getAngleTheta()
            d.rotateXZ(-theta2)
            nu_idx = i
            break

    for i, d in enumerate(tau_daughters):
        if i != nu_idx:
            d.rotateXY(-phi2)
            d.rotateXZ(-theta2)

    return phi2, theta2, tau_com_pz, tau_com_e, phi, theta


# ---------------------------------------------------------------------------
# Public: get polarimetric vectors for a ditau event
# ---------------------------------------------------------------------------

def getHHVectors(
    boson:           Particle,
    tau_plus:        Particle,
    tau_minus:       Particle,
    tau_plus_daughters:  List[Particle],
    tau_minus_daughters: List[Particle],
    frame: str = 'tau_rest',
) -> Tuple[List[float], float, List[float], float]:
    """
    Compute polarimetric vectors HH+ and HH- for a ditau event.

    Parameters
    ----------
    boson            : parent boson (Z/H/...); PDG ID used downstream.
    tau_plus         : tau+ (pdgid = -15)
    tau_minus        : tau- (pdgid =  15)
    tau_plus_daughters  : decay products of tau+
    tau_minus_daughters : decay products of tau-
    frame            : coordinate frame for the returned HH vectors.
        'tau_rest'  (default) — each HH is in the respective tau rest frame
                    with Z along the tau flight direction in the tau-tau COM
                    frame.  Use this for spin reweighting.
        'com'       — each HH is boosted back to the tau-tau COM frame after
                    being computed in the tau rest frame.  Use this when you
                    need both vectors in a single shared frame, e.g. for
                    computing CP-sensitive acoplanarity angles.

    Returns
    -------
    HHp      : polarimetric 4-vector [hx, hy, hz, ht] for tau+
    WTamplP  : matrix-element amplitude weight for tau+
    HHm      : polarimetric 4-vector [hx, hy, hz, ht] for tau-
    WTamplM  : matrix-element amplitude weight for tau-
    """
    if frame not in ('tau_rest', 'com'):
        raise ValueError(f"frame must be 'tau_rest' or 'com', got '{frame}'")

    results = []
    for tau, companion, daughters in [
        (tau_plus,  tau_minus, tau_plus_daughters),
        (tau_minus, tau_plus,  tau_minus_daughters),
    ]:
        tau_c       = tau.copy()
        companion_c = companion.copy()
        daughters_c = [d.copy() for d in daughters]

        phi2, theta2, tau_com_pz, tau_com_e, _phi_s2, _theta_s2 = _prepare_kinematic_for_hh(
            tau_c, companion_c, daughters_c
        )
        HH, wt = calculateHH(tau.pdgid(), daughters_c, phi2, theta2)

        if frame == 'com':
            # Undo step-3 boost (tau rest → step-2 COM frame).
            hh_p = Particle(HH[0], HH[1], HH[2], HH[3], 0)
            hh_p.boostAlongZ(tau_com_pz, tau_com_e)
            HH = [hh_p.px(), hh_p.py(), hh_p.pz(), hh_p.e()]

        results.append((HH, wt))

    (HHp, WTamplP), (HHm, WTamplM) = results
    return HHp, WTamplP, HHm, WTamplM


# ---------------------------------------------------------------------------
# Spin-weight pieces for arbitrary B/C tensor reweighting
# ---------------------------------------------------------------------------

def getSpinWeightPieces(boson, tau_plus, tau_minus, tau_plus_daughters, tau_minus_daughters):
    """
    Return the individual pieces of the spin-correlation weight for H→ττ,
    following the density-matrix decomposition (arXiv:2211.10513 Eq. 19):

        WT = 1 + Σᵢ Bᵢ⁺·hp_i + Σⱼ Bⱼ⁻·hm_j + Σᵢⱼ Cᵢⱼ·hp_i·hm_j

    Returns
    -------
    dict with keys:
        'hp_n', 'hp_r', 'hp_k'         — tau+ polarimetric projections
        'hm_n', 'hm_r', 'hm_k'         — tau- polarimetric projections
        'hpX_hmY' for X,Y in {n,r,k}   — 9 cross products hp_X * hm_Y
        'wt_amp_p', 'wt_amp_m'          — spin-averaged decay amplitudes
    """
    _AXES = ('n', 'r', 'k')

    HHp, WTamplP, HHm, WTamplM = getHHVectors(
        boson, tau_plus, tau_minus, tau_plus_daughters, tau_minus_daughters,
        frame='tau_rest'
    )

    hp = {'r':  HHp[0], 'n':  HHp[1], 'k':  HHp[2]}
    hm = {'r': -HHm[0], 'n':  HHm[1], 'k':  HHm[2]}

    pieces = {'wt_amp_p': WTamplP, 'wt_amp_m': WTamplM}
    for a in _AXES:
        pieces[f'hp_{a}'] = hp[a]
        pieces[f'hm_{a}'] = hm[a]
    for a in _AXES:
        for b in _AXES:
            pieces[f'hp{a}_hm{b}'] = hp[a] * hm[b]

    return pieces


def computeSpinWeight(pieces, C=None, Bp=None, Bm=None):
    """
    Compute WT = 1 + Σᵢ Bᵢ⁺·hp_i + Σⱼ Bⱼ⁻·hm_j + Σᵢⱼ Cᵢⱼ·hp_i·hm_j
    from the pieces dict returned by getSpinWeightPieces.

    Parameters
    ----------
    pieces : dict from getSpinWeightPieces
    C  : dict keyed by (axis_i, axis_j) e.g. {('k','k'): -1, ('r','r'): -1, ('n','n'): 1}
    Bp : dict keyed by axis e.g. {'k': 0.5}  (tau+ individual polarization)
    Bm : dict keyed by axis                   (tau- individual polarization)
    """

    _AXES = ('n', 'r', 'k')
    wt = 1.0
    if Bp:
        for a in _AXES:
            wt += Bp.get(a, 0.0) * pieces[f'hp_{a}']
    if Bm:
        for a in _AXES:
            wt += Bm.get(a, 0.0) * pieces[f'hm_{a}']
    if C:
        for a in _AXES:
            for b in _AXES:
                wt += C.get((a, b), 0.0) * pieces[f'hp{a}_hm{b}']
    return wt


def higgsCpMatrix(alpha):
    """
    Return the spin-correlation C dict for H→ττ at CP mixing angle alpha (degrees),
    suitable for use with computeSpinWeight.

    Values are in the shared (r, n, k) basis used by getSpinWeightPieces, following
    the paper (arXiv:2211.10513 Eq. 19).  The mapping between TauSpinner's internal
    parameters and the paper's C matrix (with our hp=τ⁺, hm=τ⁻ transposition) is:

        C_kk = sign = −1
        C_rr = +cos(angle)
        C_nn = +cos(angle)
        C_rn = −sin(angle)
        C_nr = +sin(angle)

    where angle = alpha * pi/90 (alpha=0 → CP-even, alpha=90 → CP-odd).

    alpha = 0   → {('k','k'): -1, ('r','r'): +1, ('n','n'): +1}  (symmetric CP-even)
    alpha = 90  → {('k','k'): -1, ('r','r'): -1, ('n','n'): -1}  (symmetric CP-odd)
    alpha = 45  → {('k','k'): -1, ('r','n'): -1, ('n','r'): +1}
    """
    import math
    angle = alpha * math.pi / 90.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return {
        ('k', 'k'): -1.0,
        ('r', 'r'):  cos_a,
        ('n', 'n'):  cos_a,
        ('r', 'n'): -sin_a,
        ('n', 'r'):  sin_a,
    }


def computeHiggsCPWeight(boson, tau_plus, tau_minus,
                                  tau_plus_daughters, tau_minus_daughters,
                                  alpha):
    """
    Reweighting weight for H→ττ at CP mixing angle alpha (degrees).

    Mirrors the TauSpinner C++ "case of Higgs" formula directly:

        WT = 1 + sign*HHp[2]*HHm[2]
               + Rxx*HHp[0]*HHm[0] + Ryy*HHp[1]*HHm[1]
               + Rxy*HHp[0]*HHm[1] + Ryx*HHp[1]*HHm[0]

    where sign = -1 (Higgs) and the R matrix is set from alpha:
        Rxx = -cos(alpha * pi/90)
        Ryy =  cos(alpha * pi/90)
        Rxy = -sin(alpha * pi/90)
        Ryx = -sin(alpha * pi/90)

    alpha = 0  → CP-even scalar
    alpha = 90 → CP-odd pseudoscalar

    HH vectors are the raw calculateHH output (tau rest frame, step-2 rotated),
    exactly as consumed by TauSpinner — no sign corrections, no frame conversion.

    Parameters
    ----------
    boson, tau_plus, tau_minus          : Particle
    tau_plus_daughters, tau_minus_daughters : list[Particle]
    alpha                               : float, mixing angle in degrees

    Returns
    -------
    float : event weight WT
    """
    import math
    HHp, _, HHm, _ = getHHVectors(
        boson, tau_plus, tau_minus,
        tau_plus_daughters, tau_minus_daughters,
        frame='tau_rest'
    )

    angle = alpha * math.pi / 90.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    Rxx = -cos_a
    Ryy =  cos_a
    Rxy = -sin_a
    Ryx = -sin_a
    sign = -1.0  # Higgs

    return (1.0
            + sign * (HHp[2]) * (HHm[2])
            + Rxx  * (HHp[0]) * (HHm[0])
            + Ryy  * (HHp[1]) * (HHm[1])
            + Rxy  * (HHp[0]) * (HHm[1])
            + Ryx  * (HHp[1]) * (HHm[0]))


# ---------------------------------------------------------------------------
# Fortran library wrapper (optional)
# ---------------------------------------------------------------------------

class _TauolaFortran:
    """Lazy loader for the compiled Tauola Fortran shared library."""

    def __init__(self):
        self._lib: Optional[ctypes.CDLL] = None

    def load(self, path: str) -> None:
        self._lib = ctypes.CDLL(path)

    @property
    def available(self) -> bool:
        return self._lib is not None

    # --- dam2pi_ (2-pi / K-K channel) ---
    def dam2pi(self, MNUM: int,
               PT: List[float], PN: List[float],
               PIM1: List[float], PIM2: List[float]) -> Tuple[float, List[float]]:
        lib = self._lib
        assert lib is not None
        FloatArr4 = ctypes.c_float * 4
        mnum  = ctypes.c_int(MNUM)
        pt    = FloatArr4(*[float(x) for x in PT])
        pn    = FloatArr4(*[float(x) for x in PN])
        pim1  = FloatArr4(*[float(x) for x in PIM1])
        pim2  = FloatArr4(*[float(x) for x in PIM2])
        amplit = ctypes.c_float(0.0)
        hv     = FloatArr4(0.0, 0.0, 0.0, 0.0)
        lib.dam2pi_(ctypes.byref(mnum), pt, pn, pim1, pim2,
                    ctypes.byref(amplit), hv)
        return float(amplit.value), list(hv)

    # --- dampry_ (leptonic channel) ---
    def dampry(self, ITDKRC: int, XK0DEC: float,
               XK: List[float], XA: List[float],
               QP: List[float], XN: List[float]) -> Tuple[float, List[float]]:
        lib = self._lib
        assert lib is not None
        FloatArr4 = ctypes.c_double * 4
        itdkrc = ctypes.c_int(ITDKRC)
        xk0dec = ctypes.c_double(XK0DEC)
        xk_arr = FloatArr4(*[float(x) for x in XK])
        xa_arr = FloatArr4(*[float(x) for x in XA])
        qp_arr = FloatArr4(*[float(x) for x in QP])
        xn_arr = FloatArr4(*[float(x) for x in XN])
        amplit  = ctypes.c_double(0.0)
        hv      = FloatArr4(0.0, 0.0, 0.0, 0.0)
        lib.dampry_(ctypes.byref(itdkrc), ctypes.byref(xk0dec),
                    xk_arr, xa_arr, qp_arr, xn_arr,
                    ctypes.byref(amplit), hv)
        return float(amplit.value), list(hv)

    # --- damppk_ (3-body with kaons / 3-pi) ---
    def damppk(self, MNUM: int,
               PT: List[float], PN: List[float],
               PIM1: List[float], PIM2: List[float],
               PIPL: List[float]) -> Tuple[float, List[float]]:
        lib = self._lib
        assert lib is not None
        FloatArr4 = ctypes.c_float * 4
        mnum  = ctypes.c_int(MNUM)
        pt    = FloatArr4(*[float(x) for x in PT])
        pn    = FloatArr4(*[float(x) for x in PN])
        pim1  = FloatArr4(*[float(x) for x in PIM1])
        pim2  = FloatArr4(*[float(x) for x in PIM2])
        pipl  = FloatArr4(*[float(x) for x in PIPL])
        amplit = ctypes.c_float(0.0)
        hv     = FloatArr4(0.0, 0.0, 0.0, 0.0)
        lib.damppk_(ctypes.byref(mnum), pt, pn, pim1, pim2, pipl,
                    ctypes.byref(amplit), hv)
        return float(amplit.value), list(hv)

    # --- dam4pi_ (4-pi channel) ---
    def dam4pi(self, MNUM: int,
               PT: List[float], PN: List[float],
               PIM1: List[float], PIM2: List[float],
               PIZ:  List[float], PIPL: List[float]) -> Tuple[float, List[float]]:
        lib = self._lib
        assert lib is not None
        FloatArr4 = ctypes.c_float * 4
        mnum  = ctypes.c_int(MNUM)
        pt    = FloatArr4(*[float(x) for x in PT])
        pn    = FloatArr4(*[float(x) for x in PN])
        pim1  = FloatArr4(*[float(x) for x in PIM1])
        pim2  = FloatArr4(*[float(x) for x in PIM2])
        piz   = FloatArr4(*[float(x) for x in PIZ])
        pipl  = FloatArr4(*[float(x) for x in PIPL])
        amplit = ctypes.c_float(0.0)
        hv     = FloatArr4(0.0, 0.0, 0.0, 0.0)
        lib.dam4pi_(ctypes.byref(mnum), pt, pn, pim1, pim2, piz, pipl,
                    ctypes.byref(amplit), hv)
        return float(amplit.value), list(hv)


TauolaFortran = _TauolaFortran()


# ---------------------------------------------------------------------------
# channelMatch
# ---------------------------------------------------------------------------

def channelMatch(particles: List[Particle], *pdgids: int) -> bool:
    """
    Check whether `particles` contains exactly the given pdgids (any order).
    If matched, rearranges `particles` in-place to match the given order.
    Mirrors TauSpinner::channelMatch in tau_reweight_lib.cxx.
    """
    target = [p for p in pdgids if p != 0]
    if len(particles) != len(target):
        return False

    remaining = list(particles)

    for pid in target:
        found = False
        for i, p in enumerate(remaining):
            if p.pdgid() == pid:
                remaining.pop(i)
                found = True
                break
        if not found:
            return False

    # Rearrange particles in-place to match target order
    new_order: List[Particle] = []
    tmp = list(particles)
    for pid in target:
        for i, p in enumerate(tmp):
            if p.pdgid() == pid:
                new_order.append(tmp.pop(i))
                break

    particles[:] = new_order
    return True


# ---------------------------------------------------------------------------
# calculateHH
# ---------------------------------------------------------------------------

AMTAU = 1.777
GFERMI = 1.16637e-5


def calculateHH(tau_pdgid: int,
                tau_daughters: List[Particle],
                phi: float,
                theta: float) -> Tuple[List[float], float]:
    """
    Compute the hadronic/leptonic spin analyser vector HH for a tau decay.

    Parameters
    ----------
    tau_pdgid    : PDG id of the tau (15 for tau^-, -15 for tau^+)
    tau_daughters: list of Particle objects (daughters of the tau)
    phi, theta   : rotation angles to the lab frame

    Returns
    -------
    HH       : list of 4 floats [Hx, Hy, Hz, H0]
    WTamplit : matrix-element amplitude (for normalisation)

    Notes
    -----
    Channels that call Tauola Fortran routines (leptonic, 2-pi, 3-pi, 4-pi)
    require TauolaFortran.load("<path/to/libtauola.so>") to be called first.
    If the library is not loaded those channels return HH=[0,0,0,0], WTamplit=0.
    """
    tau_daughters = list(tau_daughters)   # local copy so we can rearrange
    HH = [0.0, 0.0, 0.0, 0.0]
    WTamplit = 0.0

    # idff = PDG id of the tau: 15 for tau-, -15 for tau+.
    # Fortran CLAXI uses SIGN = IDFF/|IDFF|: +1 for tau-, -1 for tau+.
    idff = tau_pdgid

    def _p4(d):
        return [d.px(), d.py(), d.pz(), d.e()]

    n = len(tau_daughters)

    # ------------------------------------------------------------------
    # Channel 3 / 6 : tau -> pi^- nu  or  tau -> K^- nu  (2-body)
    # ------------------------------------------------------------------
    if n == 2 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -211)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  211)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -321)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  321))
    ):
        # daughters[0] = nu, daughters[1] = pi/K
        nu  = tau_daughters[0]
        mes = tau_daughters[1]
        AMPI2 = (mes.e()**2 - mes.px()**2 - mes.py()**2 - mes.pz()**2)
        AMPI  = math.sqrt(max(AMPI2, 0.0))

        PXQ = AMTAU * mes.e()
        PXN = AMTAU * nu.e()
        QXN = (mes.e() * nu.e()
               - mes.px() * nu.px()
               - mes.py() * nu.py()
               - mes.pz() * nu.pz())
        BRAK = 2.0 * PXQ * QXN - AMPI**2 * PXN

        WTamplit = GFERMI**2 * BRAK / 2.0
        HH[0] = AMTAU * (2.0 * mes.px() * QXN - nu.px() * AMPI**2) / BRAK
        HH[1] = AMTAU * (2.0 * mes.py() * QXN - nu.py() * AMPI**2) / BRAK
        HH[2] = AMTAU * (2.0 * mes.pz() * QXN - nu.pz() * AMPI**2) / BRAK
        HH[3] = 1.0

    # ------------------------------------------------------------------
    # Channel 4 / 22 : tau -> pi^- pi^0 nu  or  tau -> K^- K^0 nu
    # ------------------------------------------------------------------
    elif n == 3 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -211,  111)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  211,  111)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -321,  311)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  321,  311)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -321,  310)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  321,  310)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -321,  130)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  321,  130))
    ):
        MNUM = 0 if (tau_daughters[2].pdgid() == 111) else 3
        PT   = [0.0, 0.0, 0.0, AMTAU]
        WTamplit, hv = _tf.dam2pi(MNUM, PT, _p4(tau_daughters[0]),
                                  _p4(tau_daughters[1]), _p4(tau_daughters[2]), idff)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 7 : tau -> K^- pi^0 nu  or  tau -> pi^- K^0 nu  (K-pi)
    # ------------------------------------------------------------------
    elif n == 3 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -211,  130)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  211,  130)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -211,  310)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  211,  310)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -211,  311)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  211,  311)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -321,  111)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  321,  111))
    ):
        nu  = tau_daughters[0]
        d1  = tau_daughters[1]
        d2  = tau_daughters[2]

        QQ  = [d1.e()  - d2.e(),
               d1.px() - d2.px(),
               d1.py() - d2.py(),
               d1.pz() - d2.pz()]
        PKS = [d1.e()  + d2.e(),
               d1.px() + d2.px(),
               d1.py() + d2.py(),
               d1.pz() + d2.pz()]

        PKSD  = PKS[0]**2 - PKS[1]**2 - PKS[2]**2 - PKS[3]**2
        QQPKS = QQ[0]*PKS[0] - QQ[1]*PKS[1] - QQ[2]*PKS[2] - QQ[3]*PKS[3]

        QQ[0] -= PKS[0] * QQPKS / PKSD
        QQ[1] -= PKS[1] * QQPKS / PKSD
        QQ[2] -= PKS[2] * QQPKS / PKSD
        QQ[3] -= PKS[3] * QQPKS / PKSD

        PRODPQ = AMTAU * QQ[0]
        PRODNQ = (nu.e()  * QQ[0]
                  - nu.px() * QQ[1]
                  - nu.py() * QQ[2]
                  - nu.pz() * QQ[3])
        PRODPN = AMTAU * nu.e()
        QQ2    = QQ[0]**2 - QQ[1]**2 - QQ[2]**2 - QQ[3]**2

        BRAK = 2.0 * PRODPQ * PRODNQ - PRODPN * QQ2

        WTamplit = GFERMI**2 * BRAK / 2.0
        HH[0] = AMTAU * (2.0 * PRODNQ * QQ[1] - QQ2 * nu.px()) / BRAK
        HH[1] = AMTAU * (2.0 * PRODNQ * QQ[2] - QQ2 * nu.py()) / BRAK
        HH[2] = AMTAU * (2.0 * PRODNQ * QQ[3] - QQ2 * nu.pz()) / BRAK
        HH[3] = 1.0

    # ------------------------------------------------------------------
    # Channel 1 : tau -> e nu_tau anti_nu_e  (no photon)
    # ------------------------------------------------------------------
    elif n == 3 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  11, -12)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16, -11,  12))
    ):
        nu  = tau_daughters[0]; lep = tau_daughters[1]; nul = tau_daughters[2]
        me  = 0.511e-3
        QP  = [lep.px(), lep.py(), lep.pz(), math.sqrt(lep.px()**2+lep.py()**2+lep.pz()**2+me**2)]
        XA  = [nul.px(), nul.py(), nul.pz(), math.sqrt(nul.px()**2+nul.py()**2+nul.pz()**2)]
        XN  = _p4(nu); XK = [0.,0.,0.,0.]
        WTamplit, hv = _tf.dampry(0, 0.01, XK, XA, QP, XN)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 1b : tau -> e nu_tau anti_nu_e gamma
    # ------------------------------------------------------------------
    elif n == 4 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  11, -12, 22)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16, -11,  12, 22))
    ):
        nu = tau_daughters[0]; lep = tau_daughters[1]
        nul = tau_daughters[2]; gam = tau_daughters[3]
        me = 0.511e-3
        QP = [lep.px(), lep.py(), lep.pz(), math.sqrt(lep.px()**2+lep.py()**2+lep.pz()**2+me**2)]
        XA = [nul.px(), nul.py(), nul.pz(), math.sqrt(nul.px()**2+nul.py()**2+nul.pz()**2)]
        XK = [gam.px(), gam.py(), gam.pz(), math.sqrt(gam.px()**2+gam.py()**2+gam.pz()**2)]
        XN = _p4(nu)
        XK0DEC = 0.01
        total_e = XK[3]+XA[3]+QP[3]+XN[3]
        if total_e > 0 and XK0DEC > XK[3]/total_e:
            XK0DEC = 0.5*XK[3]/total_e
        WTamplit, hv = _tf.dampry(1, XK0DEC, XK, XA, QP, XN)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 2 : tau -> mu nu_tau anti_nu_mu  (no photon)
    # ------------------------------------------------------------------
    elif n == 3 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  13, -14)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16, -13,  14))
    ):
        nu = tau_daughters[0]; lep = tau_daughters[1]; nul = tau_daughters[2]
        mmu = 0.105659
        QP = [lep.px(), lep.py(), lep.pz(), math.sqrt(lep.px()**2+lep.py()**2+lep.pz()**2+mmu**2)]
        XA = [nul.px(), nul.py(), nul.pz(), math.sqrt(nul.px()**2+nul.py()**2+nul.pz()**2)]
        XN = _p4(nu); XK = [0.,0.,0.,0.]
        WTamplit, hv = _tf.dampry(0, 0.01, XK, XA, QP, XN)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 2b : tau -> mu nu_tau anti_nu_mu gamma
    # ------------------------------------------------------------------
    elif n == 4 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  13, -14, 22)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16, -13,  14, 22))
    ):
        nu = tau_daughters[0]; lep = tau_daughters[1]
        nul = tau_daughters[2]; gam = tau_daughters[3]
        mmu = 0.105659
        QP = [lep.px(), lep.py(), lep.pz(), math.sqrt(lep.px()**2+lep.py()**2+lep.pz()**2+mmu**2)]
        XA = [nul.px(), nul.py(), nul.pz(), math.sqrt(nul.px()**2+nul.py()**2+nul.pz()**2)]
        XK = [gam.px(), gam.py(), gam.pz(), math.sqrt(gam.px()**2+gam.py()**2+gam.pz()**2)]
        XN = _p4(nu)
        XK0DEC = 0.01
        total_e = XK[3]+XA[3]+QP[3]+XN[3]
        if total_e > 0 and XK0DEC > XK[3]/total_e:
            XK0DEC = 0.5*XK[3]/total_e
        WTamplit, hv = _tf.dampry(1, XK0DEC, XK, XA, QP, XN)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 5 : tau -> pi^- pi^0 pi^0 nu  (MNUM=0, chanopt JJ=2)
    # ------------------------------------------------------------------
    elif n == 4 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  111,  111, -211)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  111,  111,  211))
    ):
        PT = [0.,0.,0.,AMTAU]
        WTamplit, hv = _tf.damppk(0, PT, _p4(tau_daughters[0]),
                                  _p4(tau_daughters[1]), _p4(tau_daughters[2]),
                                  _p4(tau_daughters[3]), idff, idk=1)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 5 : tau -> pi^+ pi^- pi^- nu  (MNUM=0, chanopt JJ=1)
    # ------------------------------------------------------------------
    elif n == 4 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -211, -211,  211)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  211,  211, -211))
    ):
        PT = [0.,0.,0.,AMTAU]
        WTamplit, hv = _tf.damppk(0, PT, _p4(tau_daughters[0]),
                                  _p4(tau_daughters[1]), _p4(tau_daughters[2]),
                                  _p4(tau_daughters[3]), idff, idk=2)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 14 : tau -> K^- pi^- K^+  nu  (MNUM=1)
    # ------------------------------------------------------------------
    elif n == 4 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -321, -211,  321)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  321,  211, -321))
    ):
        PT = [0.,0.,0.,AMTAU]
        WTamplit, hv = _tf.damppk(1, PT, _p4(tau_daughters[0]),
                                  _p4(tau_daughters[1]), _p4(tau_daughters[2]),
                                  _p4(tau_daughters[3]), idff)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 15 : tau -> K^0 pi^- K^0(bar) nu  (MNUM=2) — multiple K^0 variants
    # ------------------------------------------------------------------
    elif n == 4 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  311, -211,  311)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  311,  211,  311)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  311, -211,  310)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  311,  211,  310)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  311, -211,  130)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  311,  211,  130)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  310, -211,  311)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  310,  211,  311)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  310, -211,  310)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  310,  211,  310)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  310, -211,  130)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  310,  211,  130)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  130, -211,  311)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  130,  211,  311)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  130, -211,  310)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  130,  211,  310)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  130, -211,  130)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  130,  211,  130))
    ):
        PT = [0.,0.,0.,AMTAU]
        WTamplit, hv = _tf.damppk(2, PT, _p4(tau_daughters[0]),
                                  _p4(tau_daughters[1]), _p4(tau_daughters[2]),
                                  _p4(tau_daughters[3]), idff)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 16 : tau -> K^- K^0 pi^0 nu  (MNUM=3)
    # ------------------------------------------------------------------
    elif n == 4 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -321,  311,  111)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  321,  311,  111)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -321,  310,  111)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  321,  310,  111)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -321,  130,  111)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  321,  130,  111))
    ):
        PT = [0.,0.,0.,AMTAU]
        WTamplit, hv = _tf.damppk(3, PT, _p4(tau_daughters[0]),
                                  _p4(tau_daughters[1]), _p4(tau_daughters[2]),
                                  _p4(tau_daughters[3]), idff)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 17 : tau -> pi^0 pi^0 K^- nu  (MNUM=4)
    # ------------------------------------------------------------------
    elif n == 4 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  111,  111, -321)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  111,  111,  321))
    ):
        PT = [0.,0.,0.,AMTAU]
        WTamplit, hv = _tf.damppk(4, PT, _p4(tau_daughters[0]),
                                  _p4(tau_daughters[1]), _p4(tau_daughters[2]),
                                  _p4(tau_daughters[3]), idff)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 18 : tau -> K^- pi^- pi^+ nu  (MNUM=5)
    # ------------------------------------------------------------------
    elif n == 4 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -321, -211,  211)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  321,  211, -211))
    ):
        PT = [0.,0.,0.,AMTAU]
        WTamplit, hv = _tf.damppk(5, PT, _p4(tau_daughters[0]),
                                  _p4(tau_daughters[1]), _p4(tau_daughters[2]),
                                  _p4(tau_daughters[3]), idff)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 19 : tau -> pi^- K^0(bar) pi^0 nu  (MNUM=6)
    # ------------------------------------------------------------------
    elif n == 4 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -211,  311,  111)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  211,  311,  111)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -211,  310,  111)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  211,  310,  111)) or
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -211,  130,  111)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  211,  130,  111))
    ):
        PT = [0.,0.,0.,AMTAU]
        WTamplit, hv = _tf.damppk(6, PT, _p4(tau_daughters[0]),
                                  _p4(tau_daughters[1]), _p4(tau_daughters[2]),
                                  _p4(tau_daughters[3]), idff)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 8 : tau -> pi^+ pi^+ pi^0 pi^- nu  (MNUM=1, dam4pi)
    # ------------------------------------------------------------------
    elif n == 5 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16, -211, -211,  211,  111)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  211,  211, -211,  111))
    ):
        PT = [0.,0.,0.,AMTAU]
        WTamplit, hv = _tf.dam4pi(1, PT, _p4(tau_daughters[0]),
                                  _p4(tau_daughters[1]), _p4(tau_daughters[2]),
                                  _p4(tau_daughters[4]), _p4(tau_daughters[3]), idff)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # ------------------------------------------------------------------
    # Channel 9 : tau -> pi^0 pi^0 pi^0 pi^- nu  (MNUM=2, dam4pi)
    # ------------------------------------------------------------------
    elif n == 5 and (
        (tau_pdgid ==  15 and channelMatch(tau_daughters,  16,  111,  111,  111, -211)) or
        (tau_pdgid == -15 and channelMatch(tau_daughters, -16,  111,  111,  111,  211))
    ):
        PT = [0.,0.,0.,AMTAU]
        WTamplit, hv = _tf.dam4pi(2, PT, _p4(tau_daughters[0]),
                                  _p4(tau_daughters[1]), _p4(tau_daughters[2]),
                                  _p4(tau_daughters[3]), _p4(tau_daughters[4]), idff)
        HH[0] = -hv[0]; HH[1] = -hv[1]; HH[2] = -hv[2]; HH[3] = hv[3]

    # Unrecognised channel: HH stays [0,0,0,0], WTamplit stays 0

    # ------------------------------------------------------------------
    # Rotate HH by (theta, phi) — mirrors the rotateXZ/rotateXY at end of C++ function
    # ------------------------------------------------------------------
    buf = Particle(HH[0], HH[1], HH[2], HH[3], 0)
    buf.rotateXZ(theta)
    buf.rotateXY(phi)
    HH[0] = buf.px()
    HH[1] = buf.py()
    HH[2] = buf.pz()
    HH[3] = buf.e()

    return HH, WTamplit
