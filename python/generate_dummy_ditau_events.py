import numpy as np
import pandas as pd

def generate_ditau_pair(fixed_orientation=True):
    """
    Generate 4-vectors for Z -> tau+ tau- decay in Z rest frame.

    Parameters
    ----------
    fixed_orientation : bool
        If True, taus go along ±z axis.
        If False, taus are emitted back-to-back with random orientation.

    Returns
    -------
    tau_minus, tau_plus : np.ndarray
        4-vectors [E, px, py, pz] for tau- and tau+ respectively.
    """

    mZ = 91.0     # GeV
    mTau = 1.777  # GeV
    E = mZ / 2.0
    p = np.sqrt(E**2 - mTau**2)

    if fixed_orientation:
        # Along ±z axis
        p_vec_tau_minus = np.array([0.0, 0.0, +p])
        p_vec_tau_plus  = np.array([0.0, 0.0, -p])
    else:
        # Random isotropic direction
        costheta = np.random.uniform(-1, 1)
        sintheta = np.sqrt(1 - costheta**2)
        phi = np.random.uniform(0, 2 * np.pi)

        # Direction unit vector
        ux = sintheta * np.cos(phi)
        uy = sintheta * np.sin(phi)
        uz = costheta

        p_vec_tau_minus = p * np.array([ux, uy, uz])
        p_vec_tau_plus  = -p_vec_tau_minus  # back-to-back

    # 4-vectors [E, px, py, pz]
    tau_minus = np.array([E, *p_vec_tau_minus])
    tau_plus  = np.array([E, *p_vec_tau_plus])

    return tau_minus, tau_plus

def lorentz_boost(four_vector, beta_vec):
    """
    Boost a four-vector by velocity vector beta_vec (v/c).
    This applies the forward Lorentz boost:
      - transforms a four-vector from a frame where the particle has components (E, p)
        to a frame moving with velocity +beta_vec (i.e. E' = gamma(E + beta·p)).
    four_vector: np.array([E, px, py, pz])
    beta_vec: np.array([bx, by, bz]) with |beta| < 1
    returns boosted np.array([E', px', py', pz'])
    """
    bx, by, bz = beta_vec
    b2 = bx*bx + by*by + bz*bz
    if b2 == 0.0:
        return four_vector.copy()
    if b2 >= 1.0:
        raise ValueError("beta >= 1 in lorentz_boost")

    gamma = 1.0 / np.sqrt(1.0 - b2)
    E = four_vector[0]
    p = four_vector[1:]
    bp = bx*p[0] + by*p[1] + bz*p[2]  # beta dot p

    # Boosted energy
    E_prime = gamma * (E + bp)

    # Spatial part
    coeff = (gamma - 1.0) * bp / b2 + gamma * E
    p_prime = p + coeff * beta_vec

    return np.array([E_prime, p_prime[0], p_prime[1], p_prime[2]])


def decay_tau_to_pion_neutrino(tau_4vec):
    """
    Decay a tau into pion + neutrino, assuming isotropy in tau rest frame.

    Parameters
    ----------
    tau_4vec : np.ndarray
        Tau 4-vector in lab (Z rest) frame.

    Returns
    -------
    pion_4vec, neutrino_4vec : np.ndarray
        4-vectors of decay products in the lab frame.
    """

    mTau = 1.777
    mPi = 0.1396
    mNu = 0.0

    # Compute momentum magnitude in tau rest frame
    p_star = (mTau**2 - mPi**2) / (2 * mTau)
    E_pi_star = np.sqrt(p_star**2 + mPi**2)
    E_nu_star = p_star  # neutrino is massless

    # Random isotropic direction
    costheta = np.random.uniform(-1, 1)
    sintheta = np.sqrt(1 - costheta**2)
    phi = np.random.uniform(0, 2 * np.pi)
    ux, uy, uz = sintheta * np.cos(phi), sintheta * np.sin(phi), costheta

    # pion and neutrino 4-vectors in tau rest frame
    pion_rf = np.array([E_pi_star, p_star * ux, p_star * uy, p_star * uz])
    neutrino_rf = np.array([E_nu_star, -p_star * ux, -p_star * uy, -p_star * uz])

    # tau velocity in lab frame
    E_tau = tau_4vec[0]
    p_tau = tau_4vec[1:]
    beta_vec = p_tau / E_tau

    # Boost both to lab frame
    pion_lab = lorentz_boost(pion_rf, beta_vec)
    neutrino_lab = lorentz_boost(neutrino_rf, beta_vec)

    ## Adjust pion charge labeling
    #pion_lab[0] = abs(pion_lab[0])  # ensure positive energy

    return pion_lab, neutrino_lab

def generate_events(n_events=10000, fixed_orientation=False, seed=None):
    """
    Generate Z -> tau+ tau- -> pi+ pi- nu nubar events.

    Parameters
    ----------
    n_events : int
        Number of events to generate.
    fixed_orientation : bool
        If True, taus along z-axis; if False, random isotropic.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns for pion and neutrino four-vector components.
    """
    if seed is not None:
        np.random.seed(seed)

    records = []

    for _ in range(n_events):
        # Step 1: produce taus
        tau_minus, tau_plus = generate_ditau_pair(fixed_orientation=fixed_orientation)

        # Step 2: decay each tau
        pi_minus, nu_tau = decay_tau_to_pion_neutrino(tau_minus)
        pi_plus, nubar_tau = decay_tau_to_pion_neutrino(tau_plus)

        # Step 3: store event data
        records.append({
            # π−
            "pi_minus_E":  pi_minus[0],
            "pi_minus_px": pi_minus[1],
            "pi_minus_py": pi_minus[2],
            "pi_minus_pz": pi_minus[3],
            # π+
            "pi_plus_E":  pi_plus[0],
            "pi_plus_px": pi_plus[1],
            "pi_plus_py": pi_plus[2],
            "pi_plus_pz": pi_plus[3],
            # nu (tau-)
            "nu_E":  nu_tau[0],
            "nu_px": nu_tau[1],
            "nu_py": nu_tau[2],
            "nu_pz": nu_tau[3],
            # nubar (tau+)
            "nubar_E":  nubar_tau[0],
            "nubar_px": nubar_tau[1],
            "nubar_py": nubar_tau[2],
            "nubar_pz": nubar_tau[3],
        })

    df = pd.DataFrame.from_records(records)
    return df

df = generate_events(n_events=1000000, fixed_orientation=False, seed=42)
print(df.head())

output_file = "dummy_z_ditau_events.pkl"
df.to_pickle(output_file)

# below is for testing using analytical solutions
test = True
if test == True:

    import ROOT
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
    from ReconstructTaus import ReconstructTauAnalytically

    # loop over data frame

    for index, row in df.iterrows():

        if index > 10:
            break

        print('\nEvent %i' % index)

        P_Z = ROOT.TLorentzVector(0., 0., 0., 91.)

        taup_pi = ROOT.TLorentzVector(row["pi_plus_px"], row["pi_plus_py"], row["pi_plus_pz"], row["pi_plus_E"])
        taun_pi = ROOT.TLorentzVector(row["pi_minus_px"], row["pi_minus_py"], row["pi_minus_pz"], row["pi_minus_E"])

        # get 4-vectors of true taus by summing pis and nus

        taup_nu = ROOT.TLorentzVector(row["nubar_px"], row["nubar_py"], row["nubar_pz"], row["nubar_E"])
        taun_nu = ROOT.TLorentzVector(row["nu_px"], row["nu_py"], row["nu_pz"], row["nu_E"])

        taup_true = taup_pi + taup_nu
        taun_true = taun_pi + taun_nu

        print("True tau+:", (taup_true.E(), taup_true.Px(), taup_true.Py(), taup_true.Pz()))
        print("True tau-:", (taun_true.E(), taun_true.Px(), taun_true.Py(), taun_true.Pz()))

        solutions = ReconstructTauAnalytically(P_Z, taup_pi, taun_pi, taup_pi, taun_pi, return_values=True)

        for i, solution in enumerate(solutions):

            print(f"Solution {i}:")

            taup = solution[0]
            taun = solution[1]

            print("tau+:", (taup.E(), taup.Px(), taup.Py(), taup.Pz()))
            print("tau-:", (taun.E(), taun.Px(), taun.Py(), taun.Pz()))
    

