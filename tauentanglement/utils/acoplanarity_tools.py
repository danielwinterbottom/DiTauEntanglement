import awkward as ak
import vector
import numpy as np
vector.register_awkward()
from tauentanglement.utils.PolarimetricA1 import PolarimetricA1_vectorised

def spatial(v, return_E=False):
    p3 = ak.zip({"x": v.px, "y": v.py, "z": v.pz}, with_name="Vector3D")
    if return_E:
        return p3, v.E
    else:
        return p3

def boost_vec(v):
    return ak.zip({"x": v.px / v.E, "y": v.py / v.E, "z": v.pz / v.E}, with_name="Vector3D")

def boost4(v, bv):
    r = v.boost(-bv)
    return ak.zip({"px": r.x, "py": r.y, "pz": r.z, "E": r.t}, with_name="Momentum4D")

def polarimetric_vec_dm0(H_pi, boost_vec_tau):
    return spatial(boost4(H_pi, boost_vec_tau)).unit()

def polarimetric_vec_dm1(H_pi, H_pizero, H_tau, boost_vec_tau):
    """Compute the DM1/2 (rho/a11pr) polarimetric vector in the tau rest frame."""
    q = H_pi - H_pizero
    P = H_tau
    N = H_tau - H_pi - H_pizero
    qN = q.dot(N)
    qP = q.dot(P)
    NP = N.dot(P)
    q2 = q.dot(q)
    massP = np.sqrt(np.maximum(P.dot(P), 0.0))
    coeff = massP / (2 * qN * qP - q2 * NP)
    pv = coeff * (2 * qN * q - q2 * N)
    return spatial(boost4(pv, boost_vec_tau)).unit()

def polarimetric_vec_dm10(tau_rf, os_pi_rf, ss1_pi_rf, ss2_pi_rf, taucharge):
    """Compute the DM10 (a1) polarimetric vector in the tau rest frame."""

    # print(">> Using DM10 polarimetric vector calculation.")
    # print('Inputs below')
    # print('Tau RF:', tau_rf)
    # print('OS Pi RF:', os_pi_rf)
    # print('SS1 Pi RF:', ss1_pi_rf)
    # print('SS2 Pi RF:', ss2_pi_rf)
    # print('Tau Charge:', taucharge)
    # print('='*60)
    pv = PolarimetricA1_vectorised(tau_rf, os_pi_rf, ss1_pi_rf, ss2_pi_rf, taucharge).PVC()

    pv_ak = ak.zip({"x": pv.x, "y": pv.y, "z": pv.z}, with_name="Vector3D")
    return pv_ak.unit()

def get_R_P_vectors_all(df, tau_prefix='tau'):
    """Compute R and P vectors for all events, selecting IP (DM0) or pizero (DM1) for R."""
    P = ak.zip({
        "px": df[f"{tau_prefix}_pi1_px"],
        "py": df[f"{tau_prefix}_pi1_py"],
        "pz": df[f"{tau_prefix}_pi1_pz"],
        "E":  df[f"{tau_prefix}_pi1_E"],
    }, with_name="Momentum4D")

    R_ip = ak.zip({
        "px": df[f"{tau_prefix}_pi1_ipx"],
        "py": df[f"{tau_prefix}_pi1_ipy"],
        "pz": df[f"{tau_prefix}_pi1_ipz"],
        "E":  ak.zeros_like(df[f"{tau_prefix}_pi1_ipx"]),
    }, with_name="Momentum4D")

    R_pizero = ak.zip({
        "px": df[f"{tau_prefix}_pizero1_px"],
        "py": df[f"{tau_prefix}_pizero1_py"],
        "pz": df[f"{tau_prefix}_pizero1_pz"],
        "E":  df[f"{tau_prefix}_pizero1_E"],
    }, with_name="Momentum4D")

    is_dm1 = df[f"{tau_prefix}_haspizero"].values == 1
    return ak.where(is_dm1, R_pizero, R_ip), P, is_dm1

def compute_aco_polarimetric(R1, P1, R2, P2):
    R1perp = R1 - ((R1.dot(P1)) / (P1.dot(P1))) * P1
    R2perp = R2 - ((R2.dot(P2)) / (P2.dot(P2))) * P2
    angle = np.arccos((R1perp.unit()).dot(R2perp.unit()))
    sign = (P2.unit()).dot((R1perp.unit()).cross(R2perp.unit()))
    angle = ak.where(sign >= 0, angle, 2 * np.pi - angle)
    return angle

def compute_aco_classic(R1, P1, R2, P2, leg1_is_dp, leg2_is_dp):
    """Compute the acoplanarity angle using IP (DM0) or DP (DM1) method per event."""
    # Boost to visible charged decay product frame
    bv = boost_vec(P1 + P2)

    R1_p3, R1_E = spatial(boost4(R1, bv), return_E=True)
    R2_p3, R2_E = spatial(boost4(R2, bv), return_E=True)
    P1_p3, P1_E = spatial(boost4(P1, bv), return_E=True)
    P2_p3, P2_E = spatial(boost4(P2, bv), return_E=True)

    # Get perpendicular components
    R1perp = R1_p3 - ((R1_p3.dot(P1_p3)) / (P1_p3.dot(P1_p3))) * P1_p3
    R2perp = R2_p3 - ((R2_p3.dot(P2_p3)) / (P2_p3.dot(P2_p3))) * P2_p3

    # Define angle between planes and O*, then apply appropriate shift
    angle = np.arccos((R1perp.unit()).dot(R2perp.unit()))
    sign  = (P2_p3.unit()).dot((R1perp.unit()).cross(R2perp.unit()))
    angle = ak.where(sign >= 0, angle, 2 * np.pi - angle)

    # Apply shift for polarisation of mesons
    Y1 = (R1_E - P1_E) / (P1_E + R1_E)
    Y2 = (R2_E - P2_E) / (P2_E + R2_E)
    meson_sign = ak.where(leg1_is_dp, np.sign(Y1), ak.ones_like(angle))
    meson_sign = meson_sign * ak.where(leg2_is_dp, np.sign(Y2), ak.ones_like(angle))
    needs_shift = (leg1_is_dp | leg2_is_dp) & (meson_sign < 0)
    angle = ak.where(needs_shift, ak.where(angle < np.pi, angle + np.pi, angle - np.pi), angle)
    return angle

def get_ditau_polarimetric(df, tau_prefix='true', reco_pions=True, dm_prefix='reco'):
    """
    tau_prefix: sets whether reco or true tau is used ("true" or "map_pred")
    reco_pions: sets whether to use reco pions or true pions (some storage issues in gen)
    dm_prefix: sets whether the reco or true number of pions is used to classify DMs
    """

    # Get R and P (polarimetric vector edition)
    if reco_pions:
        print(">> Using reconstructed pions for polarimetric vector calculation.")
        pion_prefix = 'reco'
    else:
        print('>> Using true pions for polarimetric vector calculation.')
        pion_prefix = 'true'

    # Tau plus decay mode
    taup_is_dm0 = (df[f"{dm_prefix}_taup_npizero"].values == 0) & (df[f'{dm_prefix}_taup_is3prong'] == 0)
    taup_is_dm1or2 = ((df[f"{dm_prefix}_taup_npizero"].values == 1) | (df[f"{dm_prefix}_taup_npizero"].values == 2)) & (df[f'{dm_prefix}_taup_is3prong'] == 0)
    taup_is_dm10 = (df[f"{dm_prefix}_taup_npizero"].values == 0) & (df[f'{dm_prefix}_taup_is3prong'] == 1)

    # Tau plus decay products
    piOS_p = ak.zip({"px": df[f"{pion_prefix}_taup_pi1_px"], "py": df[f"{pion_prefix}_taup_pi1_py"], "pz": df[f"{pion_prefix}_taup_pi1_pz"], "E": df[f"{pion_prefix}_taup_pi1_E"]}, with_name="Momentum4D")  # only actually OS in 3 prong case
    piSS1_p = ak.zip({"px": df[f"{pion_prefix}_taup_pi2_px"], "py": df[f"{pion_prefix}_taup_pi2_py"], "pz": df[f"{pion_prefix}_taup_pi2_pz"], "E": df[f"{pion_prefix}_taup_pi2_E"]}, with_name="Momentum4D")
    piSS2_p = ak.zip({"px": df[f"{pion_prefix}_taup_pi3_px"], "py": df[f"{pion_prefix}_taup_pi3_py"], "pz": df[f"{pion_prefix}_taup_pi3_pz"], "E": df[f"{pion_prefix}_taup_pi3_E"]}, with_name="Momentum4D")
    pizero_p = ak.zip({"px": df[f"{pion_prefix}_taup_pizero1_px"], "py": df[f"{pion_prefix}_taup_pizero1_py"], "pz": df[f"{pion_prefix}_taup_pizero1_pz"], "E": df[f"{pion_prefix}_taup_pizero1_E"]}, with_name="Momentum4D")

    # Tau plus
    tau_p = ak.zip({"px": df[f"{tau_prefix}_tau_plus_px"], "py": df[f"{tau_prefix}_tau_plus_py"], "pz": df[f"{tau_prefix}_tau_plus_pz"], "E": df[f"{tau_prefix}_tau_plus_E"]}, with_name="Momentum4D")

    # Tau minus decay mode
    taun_is_dm0 = (df[f"{dm_prefix}_taun_npizero"].values == 0) & (df[f'{dm_prefix}_taun_is3prong'] == 0)
    taun_is_dm1or2 = ((df[f"{dm_prefix}_taun_npizero"].values == 1) | (df[f"{dm_prefix}_taun_npizero"].values == 2)) & (df[f'{dm_prefix}_taun_is3prong'] == 0)
    taun_is_dm10 = (df[f"{dm_prefix}_taun_npizero"].values == 0) & (df[f'{dm_prefix}_taun_is3prong'] == 1)

    # Tau minus decay products
    piOS_n = ak.zip({"px": df[f"{pion_prefix}_taun_pi1_px"], "py": df[f"{pion_prefix}_taun_pi1_py"], "pz": df[f"{pion_prefix}_taun_pi1_pz"], "E": df[f"{pion_prefix}_taun_pi1_E"]}, with_name="Momentum4D")
    piSS1_n = ak.zip({"px": df[f"{pion_prefix}_taun_pi2_px"], "py": df[f"{pion_prefix}_taun_pi2_py"], "pz": df[f"{pion_prefix}_taun_pi2_pz"], "E": df[f"{pion_prefix}_taun_pi2_E"]}, with_name="Momentum4D")
    piSS2_n = ak.zip({"px": df[f"{pion_prefix}_taun_pi3_px"], "py": df[f"{pion_prefix}_taun_pi3_py"], "pz": df[f"{pion_prefix}_taun_pi3_pz"], "E": df[f"{pion_prefix}_taun_pi3_E"]}, with_name="Momentum4D")
    pizero_n = ak.zip({"px": df[f"{pion_prefix}_taun_pizero1_px"], "py": df[f"{pion_prefix}_taun_pizero1_py"], "pz": df[f"{pion_prefix}_taun_pizero1_pz"], "E": df[f"{pion_prefix}_taun_pizero1_E"]}, with_name="Momentum4D")

    # Tau minus
    tau_n = ak.zip({"px": df[f"{tau_prefix}_tau_minus_px"], "py": df[f"{tau_prefix}_tau_minus_py"], "pz": df[f"{tau_prefix}_tau_minus_pz"], "E": df[f"{tau_prefix}_tau_minus_E"]}, with_name="Momentum4D")

    # Higgs rest frame boost vectors
    higgs_bv = boost_vec(tau_p + tau_n)

    # Boost the decay products to the Higgs rest frame
    tau_p_rf = boost4(tau_p, higgs_bv)
    tau_n_rf = boost4(tau_n, higgs_bv)
    piOS_p_rf  = boost4(piOS_p, higgs_bv)
    piSS1_p_rf  = boost4(piSS1_p, higgs_bv)
    piSS2_p_rf = boost4(piSS2_p, higgs_bv)
    pizero_p_rf = boost4(pizero_p, higgs_bv)
    piOS_n_rf  = boost4(piOS_n, higgs_bv)
    piSS1_n_rf  = boost4(piSS1_n, higgs_bv)
    piSS2_n_rf = boost4(piSS2_n, higgs_bv)
    pizero_n_rf = boost4(pizero_n, higgs_bv)

    # Tau rest frame boost vectors (from Higgs rest frame)
    bv_taup = boost_vec(tau_p_rf)
    bv_taun = boost_vec(tau_n_rf)

    # Get the polarimetric vectors
    default = ak.zip({"x": ak.full_like(piOS_p_rf.px, -9999.0),
                        "y": ak.full_like(piOS_p_rf.py, -9999.0),
                        "z": ak.full_like(piOS_p_rf.pz, -9999.0)}, with_name="Vector3D")

    taup_s = ak.where(taup_is_dm0, polarimetric_vec_dm0(piOS_p_rf, bv_taup),
                        ak.where(taup_is_dm1or2, polarimetric_vec_dm1(piOS_p_rf, pizero_p_rf, tau_p_rf, bv_taup),
                                 ak.where(taup_is_dm10, polarimetric_vec_dm10(tau_p_rf, piOS_p_rf, piSS1_p_rf, piSS2_p_rf, +1),
                                          default)))

    taun_s = ak.where(taun_is_dm0, polarimetric_vec_dm0(piOS_n_rf, bv_taun),
                        ak.where(taun_is_dm1or2, polarimetric_vec_dm1(piOS_n_rf, pizero_n_rf, tau_n_rf, bv_taun),
                                 ak.where(taun_is_dm10, polarimetric_vec_dm10(tau_n_rf, piOS_n_rf, piSS1_n_rf, piSS2_n_rf, -1),
                                          default)))

    return taup_s, spatial(tau_p_rf).unit(), taun_s, spatial(tau_n_rf).unit()

def get_ditau_polarimetric_A1A1(df, neutrino = 'true'):
    # Build tau plus (always use reco pions due to storage issue)
    pi_p_OS     = ak.zip({"px": df["reco_taup_pi1_px"], "py": df["reco_taup_pi1_py"], "pz": df["reco_taup_pi1_pz"], "E": df["reco_taup_pi1_E"]}, with_name="Momentum4D")
    pi_p_SS1     = ak.zip({"px": df["reco_taup_pi2_px"], "py": df["reco_taup_pi2_py"], "pz": df["reco_taup_pi2_pz"], "E": df["reco_taup_pi2_E"]}, with_name="Momentum4D")
    pi_p_SS2     = ak.zip({"px": df["reco_taup_pi3_px"], "py": df["reco_taup_pi3_py"], "pz": df["reco_taup_pi3_pz"], "E": df["reco_taup_pi3_E"]}, with_name="Momentum4D")
    tau_p = ak.zip({"px": df[f"{neutrino}_tau_plus_px"], "py": df[f"{neutrino}_tau_plus_py"], "pz": df[f"{neutrino}_tau_plus_pz"], "E": df[f"{neutrino}_tau_plus_E"]}, with_name="Momentum4D")

    # Build tau minus (always use reco pions due to storage issue)
    pi_n_OS     = ak.zip({"px": df["reco_taun_pi1_px"], "py": df["reco_taun_pi1_py"], "pz": df["reco_taun_pi1_pz"], "E": df["reco_taun_pi1_E"]}, with_name="Momentum4D")
    pi_n_SS1     = ak.zip({"px": df["reco_taun_pi2_px"], "py": df["reco_taun_pi2_py"], "pz": df["reco_taun_pi2_pz"], "E": df["reco_taun_pi2_E"]}, with_name="Momentum4D")
    pi_n_SS2     = ak.zip({"px": df["reco_taun_pi3_px"], "py": df["reco_taun_pi3_py"], "pz": df["reco_taun_pi3_pz"], "E": df["reco_taun_pi3_E"]}, with_name="Momentum4D")
    tau_n = ak.zip({"px": df[f"{neutrino}_tau_minus_px"], "py": df[f"{neutrino}_tau_minus_py"], "pz": df[f"{neutrino}_tau_minus_pz"], "E": df[f"{neutrino}_tau_minus_E"]}, with_name="Momentum4D")

    # Boost to Higgs rest frame
    higgs_bv = boost_vec(tau_p + tau_n)

    H_tau_p = boost4(tau_p, higgs_bv)
    H_tau_n = boost4(tau_n, higgs_bv)

    tau_p_s = polarimetric_vec_dm10(H_tau_p, pi_p_OS, pi_p_SS2, pi_p_SS1, +1, higgs_bv)
    tau_n_s = polarimetric_vec_dm10(H_tau_n, pi_n_OS, pi_n_SS2, pi_n_SS1, -1, higgs_bv)

    return tau_p_s, spatial(H_tau_p).unit(), tau_n_s, spatial(H_tau_n).unit()

