import numpy as np
import ROOT
 
def ldot(a, b):
    """
    Minkowski dot product (+,-,-,-)
    """
    return a[..., 0]*b[..., 0] - np.sum(a[..., 1:]*b[..., 1:], axis=-1)

def mass(a):
    mass2 = ldot(a, a)
    mass = np.sqrt(np.maximum(mass2, 0.0))
    return mass

def mag2(v):
    return np.sum(v[..., 1:]**2, axis=-1)

def spatial(v, unit=False):
    if unit:
        return v[..., 1:] / np.linalg.norm(v[..., 1:], axis=-1, keepdims=True)
    return v[..., 1:]

def boost_vector(p):
    """
    ROOT-like BoostVector(): returns p⃗ / E
    p shape (...,4)
    """
    return spatial(p) / p[..., 0:1]

def add_energy(pvec3):
    """Prepend E=|p| to a (N,3) array, returning (N,4) [E,px,py,pz]."""
    return np.column_stack((np.sqrt((pvec3**2).sum(axis=1)), pvec3))


def add_energies_pair(arr):
    """
    Depending on shape of input:
    Convert (N,6) [nu1_px,nu1_py,nu1_pz, nu2_px,nu2_py,nu2_pz] to (N,8) with E=|p| prepended to each triplet.
    Convert (N,7) [nu1_m,nu1_px,nu1_py,nu1_pz, nu2_px,nu2_py,nu2_pz] to (N,8) with E1=sqrt(p1^2+m1^2) and E2=|p2| prepended to each triplet.
    Convert (N,8) [nu1_m,nu1_px,nu1_py,nu1_pz, nu2_m,nu2_px,nu2_py,nu2_pz] to (N,8) with E1=sqrt(p1^2+m1^2) and E2=sqrt(p2^2+m2^2) prepended to each triplet.
    """
    if arr.shape[1] == 6:
        return np.column_stack((add_energy(arr[:, 0:3]), add_energy(arr[:, 3:6])))
    elif arr.shape[1] == 7:
        mass1 = arr[:, 0]
        obj1_nomass = add_energy(arr[:, 1:4])
        obj1 = np.column_stack((np.sqrt(obj1_nomass[:, 0]**2 + mass1**2), obj1_nomass[:, 1:]))
        obj2 = add_energy(arr[:, 4:7])
        return np.column_stack((obj1, obj2))
    elif arr.shape[1] == 8:
        mass1 = arr[:, 0]
        mass2 = arr[:, 4]
        obj1_nomass = add_energy(arr[:, 1:4])
        obj1 = np.column_stack((np.sqrt(obj1_nomass[:, 0]**2 + mass1**2), obj1_nomass[:, 1:]))
        obj2_nomass = add_energy(arr[:, 5:8])
        obj2 = np.column_stack((np.sqrt(obj2_nomass[:, 0]**2 + mass2**2), obj2_nomass[:, 1:] ))
        return np.column_stack((obj1, obj2))
    else:
        raise ValueError(f"Unexpected input shape {arr.shape}, expected (N,6), (N,7) or (N,8)")


def inv_mass(taus, offset=0):
    t = taus[:, offset:offset+4]
    return np.sqrt(np.maximum(t[:,0]**2 - t[:,1]**2 - t[:,2]**2 - t[:,3]**2, 0))



def boost(v, beta):
    """
    ROOT-style Lorentz boost.
    boost(v, beta)  <->  TLorentzVector::Boost(beta)
    """
    beta2 = np.sum(beta*beta, axis=-1, keepdims=True)
    gamma = 1.0 / np.sqrt(1.0 - beta2)

    bp = np.sum(beta * v[...,1:], axis=-1, keepdims=True)

    v0 = gamma * (v[...,0:1] + bp)

    gamma2 = np.zeros_like(beta2)
    mask = beta2 > 0.0
    gamma2[mask] = (gamma[mask] - 1.0) / beta2[mask]

    vvec = (
        v[...,1:]
        + gamma2 * bp * beta
        + gamma * beta * v[...,0:1]
    )

    return np.concatenate([v0, vvec], axis=-1)

class _P4Wrap:
    """Minimal (E,px,py,pz) wrapper exposing .E/.px/.py/.pz attributes, since
    PolarimetricA1_vectorised was written against awkward Momentum4D records
    but its math only ever touches these four attributes -- a plain (N,4)
    numpy array works identically."""
    def __init__(self, arr):
        self.E, self.px, self.py, self.pz = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def polarimetric_vector_a1(tau, os_pi, ss1_pi, ss2_pi, taucharge):
    """
    a1 (3-prong, tau -> 3pi) polarimetric vector, using the same
    PolarimetricA1_vectorised model as acoplanarity_tools.polarimetric_vec_dm10.

    Inputs (all already boosted into the SAME frame, e.g. the ditau/Higgs
    rest frame -- this function does the final boost into the tau's own rest
    frame internally, matching polarimetric_vector_tau's mask1/mask2 boost
    convention):
      tau, os_pi, ss1_pi, ss2_pi : (N,4) arrays. os_pi is the pion of opposite
                                   charge to the tau, ss1_pi/ss2_pi are the two
                                   same-charge pions (order between ss1/ss2
                                   doesn't matter -- see SortPions convention
                                   in run_delphes.py).
      taucharge                 : (N,) array of +1 (tau+) / -1 (tau-).

    Returns:
      s : (N,3) unit polarimetric direction vector
    """
    from taupolaris.utils.PolarimetricA1 import PolarimetricA1_vectorised

    boost_vec = boost_vector(tau)
    tau_trf  = boost(tau,    -boost_vec)
    os_trf   = boost(os_pi,  -boost_vec)
    ss1_trf  = boost(ss1_pi, -boost_vec)
    ss2_trf  = boost(ss2_pi, -boost_vec)

    pv = PolarimetricA1_vectorised(
        _P4Wrap(tau_trf), _P4Wrap(os_trf), _P4Wrap(ss1_trf), _P4Wrap(ss2_trf), taucharge
    ).PVC()
    vec = -np.stack([pv.x, pv.y, pv.z], axis=-1)  # sign convention matches polarimetric_vec_dm10
    return vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-12)


def polarimetric_vector_tau(
    tau, pi1, piz1,
    tau_npi, tau_npizero,
    lep=None, is_leptonic=None,
    pi2=None, pi3=None, taucharge=None,
):
    """
    Generic tau polarimetric vector.

    Inputs:
      tau, pi1, piz1 : (N,4) arrays. For a 3-prong (tau_npi==3) row, pi1 is
                       the opposite-charge pion (see pi2/pi3/taucharge below);
                       piz1 is unused for those rows.
      tau_npi        : (N,) number of charged pions (1 or 3)
      tau_npizero    : (N,) number of pi0 (only meaningful for tau_npi==1 rows)
      lep            : (N,4) optional lepton (mu/e) momentum, in the same frame
                       as tau/pi1/piz1. Only used for rows where is_leptonic is
                       True.
      is_leptonic    : (N,) optional boolean mask. Rows with True use the
                       lepton-direction polarimetric approximation (lepton
                       direction in the tau rest frame, sign-flipped -- same
                       as polarimetric_vec_leptonic in acoplanarity_tools)
                       instead of pi1/piz1, since a leptonic decay has no
                       hadronic decay products to build the exact polarimetric
                       vector from. Rows with True are excluded from mask1/mask2/
                       mask3 below regardless of tau_npi/tau_npizero.
      pi2, pi3       : (N,4) optional same-charge pions, required together with
                       taucharge for tau_npi==3 (a1) rows. pi1 must be the
                       opposite-charge pion for those rows (SortPions
                       convention in run_delphes.py already orders pi1/pi2/pi3
                       this way at gen level).
      taucharge      : (N,) optional array of +1 (tau+) / -1 (tau-), required
                       together with pi2/pi3 for tau_npi==3 rows.

    Returns:
      s : (N,3) polarimetric direction vector
    """

    boost_vec = boost_vector(tau)
    tau_s = np.zeros_like(tau[...,1:])

    lep_mask = is_leptonic if is_leptonic is not None else np.zeros(len(tau), dtype=bool)

    mask1 = (tau_npi == 1) & (tau_npizero == 0) & ~lep_mask

    pi1_b = boost(pi1[mask1], -boost_vec[mask1])
    tau_s[mask1] = spatial(pi1_b, unit=True)

    mask2 = (tau_npi == 1) & (tau_npizero >= 1) & ~lep_mask

    q = pi1[mask2] - piz1[mask2]
    P = tau[mask2]
    N = tau[mask2] - pi1[mask2] - piz1[mask2]

    qN = ldot(q, N)
    qP = ldot(q, P)
    NP = ldot(N, P)
    q2 = ldot(q, q)

    coeff = mass(P) / (2*qN*qP - q2*NP)

    pv = coeff[:, None] * (2*qN[:, None]*q - q2[:, None]*N)

    pv_b = boost(pv, -boost_vec[mask2])
    tau_s[mask2] = spatial(pv_b, unit=True)

    mask3 = (tau_npi == 3) & ~lep_mask
    if mask3.any():
        if pi2 is None or pi3 is None or taucharge is None:
            raise ValueError("pi2, pi3, and taucharge must be provided for tau_npi==3 (a1) rows")
        tau_s[mask3] = polarimetric_vector_a1(
            tau[mask3], pi1[mask3], pi2[mask3], pi3[mask3], taucharge[mask3]
        )

    if is_leptonic is not None:
        if lep is None:
            raise ValueError("lep must be provided when is_leptonic is provided")
        lep_b = boost(lep[lep_mask], -boost_vec[lep_mask])
        tau_s[lep_mask] = -spatial(lep_b, unit=True)

    return tau_s

def compute_spin_angles(
    taup, taun,
    taup_s, taun_s,
    p_axis=None
):
    """
    Compute cosn/cosr/cosk for tau+ and tau- given tau momenta and
    polarimetric vectors in the SAME frame.

    Inputs:
      taup, taun : (N,4) four-vectors (already boosted to your chosen frame)
      p_axis     : (3,) optional fixed axis. Default is (0,0,-1)

    Returns:
      dict of arrays (N,)
    """

    N = taup.shape[0]

    # fixed axis p-hat (ROOT code effectively uses (0,0,-1))
    if p_axis is None:
        p = np.zeros((N, 3))
        p[:, 2] = -1.0
    else:
        p_axis = np.asarray(p_axis, dtype=float)
        p_axis = p_axis / np.linalg.norm(p_axis)
        p = np.broadcast_to(p_axis, (N, 3)).copy()

    # k-hat = tau+ direction
    k = spatial(taup, unit=True)

    # n-hat = (p x k) direction (normal to plane)
    n = np.cross(p, k)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(n_norm > 0, n / n_norm, 0.0)

    # cosTheta = p · k
    cosTheta = np.sum(p * k, axis=1)

    # r-hat = (p - k cosTheta) direction (in-plane perpendicular)
    r = p - k * cosTheta[:, None]
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    r = np.where(r_norm > 0, r / r_norm, 0.0)

    return {
        "cosn_plus":  np.sum(taup_s * n, axis=1),
        "cosr_plus":  np.sum(taup_s * r, axis=1),
        "cosk_plus":  np.sum(taup_s * k, axis=1),
        "cosn_minus": np.sum(taun_s * n, axis=1),
        "cosr_minus": np.sum(taun_s * r, axis=1),
        "cosk_minus": np.sum(taun_s * k, axis=1),
        "cosTheta":   cosTheta,
    } 

def polarimetric_vector_tau_root(tau, pi1, piz1, tau_npi, tau_npizero):

    if tau_npi == 1 and tau_npizero == 0:   
        pi1.Boost(-tau.BoostVector())
        tau_s = pi1.Vect().Unit()

    elif tau_npi == 1 and tau_npizero >= 1:
        q = pi1 - piz1
        P = tau
        N = tau - pi1 - piz1
        pv = P.M()*(2*(q*N)*q - q.Mag2()*N) * (1/ (2*(q*N)*(q*P) - q.Mag2()*(N*P)))
        pv.Boost(-tau.BoostVector())
        tau_s = pv.Vect().Unit()

    return tau_s   

def EntanglementVariables(C, Bplus=np.array([[0],[0],[0]]), Bminus=np.array([[0],[0],[0]])):
    '''
    Compute concurrence and m12 variables
    Note Bplus and Bminus are not expected to affect these variables at all but they are used as optional inputs in case some intermediate matrices are to be returned as well 
    '''

    # Pauli matrices
    sig1 = np.array([[0, 1],
                     [1, 0]])
    
    sig2 = np.array([[0, -1j],
                     [1j, 0]], dtype=complex)
    
    sig3 = np.array([[1, 0],
                     [0, -1]])
    
    pauli_matrices = [sig1, sig2, sig3]
    
    # identity matrix
    I    = np.array([[1, 0],
                     [0, 1]])
    
    rho1 = np.kron(I, I)
    rho2 = sum(Bplus[i, 0] * np.kron(pauli_matrices[i], I) for i in range(3))
    rho3 = sum(Bminus[j, 0] * np.kron(I, pauli_matrices[j]) for j in range(3))
    
    rho4 = np.zeros((4, 4), dtype=complex)  # Initialize a 4x4 complex matrix
    for i in range(3):
        for j in range(3):
            rho4 += C[i, j] * np.kron(pauli_matrices[i], pauli_matrices[j])
    
    
    rho = 1./4*(rho1+rho2+rho3+rho4) 

    # check if rho is valid density matrix (Hermitian, positive semi-definite, trace=1)
    #if not np.all(np.linalg.eigvals(rho) >= -1e-10):
    #    print("Warning: rho is unphysical - not positive semi-definite!")

   
    trace = np.trace(rho)

    rhostar = np.conj(rho)
    
    z = np.kron(pauli_matrices[1], pauli_matrices[1])
   
    # using formulas from https://journals.aps.org/prd/pdf/10.1103/PhysRevD.107.093002 
    # seems to be different to https://arxiv.org/pdf/2405.09201 but gives the expected answer
    #rho_tilde = z*rhostar*z
    #R = np.sqrt(np.sqrt(rho)*rho_tilde*np.sqrt(rho))
    #R_EVs = sorted(np.linalg.eigvals(R),reverse=True)
    #con = max(0, R_EVs[0]-sum(R_EVs[1:]))

    #using formulas from https://arxiv.org/pdf/2405.09201
    y = np.kron(sig2, sig2)
    R = rho @ y @ rhostar @ y
    R_EVs = np.linalg.eigvals(R)
    lambdas = np.sort(np.sqrt(np.clip(np.real_if_close(R_EVs).real, 0, None)))[::-1]
    con = max(0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3])

    M = C @ C.T
    M_EVs = sorted(np.linalg.eigvals(M),reverse=True)
    m12 = M_EVs[0]+M_EVs[1]
  
    # out also try alternative formulation from https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.80.2245

    return (con.real, m12)

def compute_spin_density_vars(df, prefix='true_'):

    # note currently not sure where the minus signs come from below but they are needed to get the correct matrix, although it doesn't change the entanglement variables at all anyway...
    C11 = -(df[f"{prefix}cosn_plus"]*df[f"{prefix}cosn_minus"]).mean()*9
    C22 = -(df[f"{prefix}cosr_plus"]*df[f"{prefix}cosr_minus"]).mean()*9
    C33 = -(df[f"{prefix}cosk_plus"]*df[f"{prefix}cosk_minus"]).mean()*9
    C12 = -(df[f"{prefix}cosn_plus"]*df[f"{prefix}cosr_minus"]).mean()*9
    C13 = -(df[f"{prefix}cosn_plus"]*df[f"{prefix}cosk_minus"]).mean()*9
    C23 = -(df[f"{prefix}cosr_plus"]*df[f"{prefix}cosk_minus"]).mean()*9
    C21 = -(df[f"{prefix}cosr_plus"]*df[f"{prefix}cosn_minus"]).mean()*9
    C31 = -(df[f"{prefix}cosk_plus"]*df[f"{prefix}cosn_minus"]).mean()*9
    C32 = -(df[f"{prefix}cosk_plus"]*df[f"{prefix}cosr_minus"]).mean()*9


    Bplus1 = -df[f"{prefix}cosn_plus"].mean() * 3
    Bplus2 = -df[f"{prefix}cosr_plus"].mean() * 3
    Bplus3 = -df[f"{prefix}cosk_plus"].mean() * 3
    Bminus1 = df[f"{prefix}cosn_minus"].mean() * 3
    Bminus2 = df[f"{prefix}cosr_minus"].mean() * 3
    Bminus3 = df[f"{prefix}cosk_minus"].mean() * 3

    Bplus = np.array([Bplus1, Bplus2, Bplus3])
    Bminus = np.array([Bminus1, Bminus2, Bminus3])

    C = np.array([[C11, C12, C13],
                  [C21, C22, C23],
                  [C31, C32, C33]])

    con, m12 = EntanglementVariables(C)
   
    return(Bplus, Bminus, C, con, m12)

if __name__ == "__main__":

    # Test vectors
    a = np.array([[10., 1., 2., 3.]])
    b = np.array([[5., -1., 0., 2.]])

    # Manual calculation
    manual = 10*5 - (1*(-1) + 2*0 + 3*2)

    print("ldot:", ldot(a, b)[0])
    print("manual:", manual)
    print()

    v = np.array([[0., 3., 4., 12.]])
    print("mag2:", mag2(v)[0])
    print("manual mag2:", 3**2 + 4**2 + 12**2)
    print()

    print("spatial:", spatial(v)[0])
    print("manual spatial:", [3., 4., 12.])
    print()

    print('spatial unit:', spatial(v, unit=True)[0])
    norm = np.sqrt(3**2 + 4**2 + 12**2)
    print("manual spatial unit:", [3./norm, 4./norm, 12./norm])
    print()
    

    p = np.array([[10., 3., 4., 0.]])
    mass_ = ldot(p, p)**0.5
    print("mass:", mass_[0])

    beta = spatial(p) / p[:,0:1]
    boostvec = boost_vector(p)

    p_rest = boost(p, -boostvec)

    print("boosted:", p_rest)

    import ROOT

    p_root = ROOT.TLorentzVector()
    p_root.SetPxPyPzE(3., 4., 0., 10.)
    boostvec_root = p_root.BoostVector()
    p_root.Boost(-boostvec_root)
    print("ROOT boosted:", p_root.Px(), p_root.Py(), p_root.Pz(), p_root.E())
    print()

    # test the polarimetric vector
    tau = np.array([[20., 5., 10., 15.]])
    pi1 = np.array([[10., 2., 4., 8.]])
    piz1 = np.array([[5., 1., 2., 4.]])
    tau_npi = np.array([1])
    tau_npizero = np.array([1])

    s = polarimetric_vector_tau(tau, pi1, piz1, tau_npi, tau_npizero)
    print("polarimetric vector:", s[0])

    # ROOT version
    tau_root = ROOT.TLorentzVector()
    tau_root.SetPxPyPzE(5., 10., 15., 20.)
    pi1_root = ROOT.TLorentzVector()
    pi1_root.SetPxPyPzE(2., 4., 8., 10.)
    piz1_root = ROOT.TLorentzVector()
    piz1_root.SetPxPyPzE(1., 2., 4., 5.)

    s_root = polarimetric_vector_tau_root(tau_root, pi1_root, piz1_root, 1, 1)
    print("ROOT polarimetric vector:", s_root.X(), s_root.Y(), s_root.Z())
    print()

    # now the same for a 0 pizero case
    tau_npizero = np.array([0])
    piz1 = np.array([[0., 0., 0., 0.]])  # not used
    s = polarimetric_vector_tau(tau, pi1, piz1, tau_npi, tau_npizero)
    print("polarimetric vector (0 pizero):", s[0])
    # ROOT version
    piz1_root.SetPxPyPzE(0., 0., 0., 0.)
    s_root = polarimetric_vector_tau_root(tau_root, pi1_root, piz1_root, 1, 0)
    print("ROOT polarimetric vector (0 pizero):", s_root.X(), s_root.Y(), s_root.Z())
    print()

