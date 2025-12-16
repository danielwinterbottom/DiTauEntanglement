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


def boost(v, beta):
    """
    ROOT-style Lorentz boost.
    boost(v, beta)  <->  TLorentzVector::Boost(beta)
    """
    beta2 = np.sum(beta*beta, axis=-1, keepdims=True)
    gamma = 1.0 / np.sqrt(1.0 - beta2)

    bp = np.sum(beta * v[...,1:], axis=-1, keepdims=True)

    v0 = gamma * (v[...,0:1] + bp)
    vvec = (
        v[...,1:]
        + (gamma - 1.0) * bp * beta / beta2
        + gamma * beta * v[...,0:1]
    )

    return np.concatenate([v0, vvec], axis=-1)

def polarimetric_vector_tau(
    tau, pi1, piz1,
    tau_npi, tau_npizero
):
    """
    Generic tau polarimetric vector.

    Inputs:
      tau, pi1, piz1 : (N,4) arrays
      tau_npi        : (N,) number of charged pions
      tau_npizero    : (N,) number of pi0

    Returns:
      s : (N,3) polarimetric direction vector
    """

    boost_vec = boost_vector(tau)
    tau_s = np.zeros_like(tau[...,1:])

    mask1 = (tau_npi == 1) & (tau_npizero == 0)

    pi1_b = boost(pi1[mask1], -boost_vec[mask1])
    tau_s[mask1] = spatial(pi1_b, unit=True)

    mask2 = (tau_npi == 1) & (tau_npizero == 1)

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