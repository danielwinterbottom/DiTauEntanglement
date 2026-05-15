import awkward as ak
import vector
import numpy as np
vector.register_awkward()

def spatial(v, return_E=False):
    p3 = ak.zip({"x": v.px, "y": v.py, "z": v.pz}, with_name="Vector3D")
    if return_E:
        return p3, v.E
    else:
        return p3

def boost_vec(v):
    return ak.zip({"x": v.px / v.E, "y": v.py / v.E, "z": v.pz / v.E}, with_name="Vector3D")

def polarimetric_vec_dm0(H_pi, boost_vec_tau):
    return spatial(H_pi.boost(-boost_vec_tau)).unit()

def polarimetric_vec_dm1(H_pi, H_pizero, H_tau, boost_vec_tau):
    """Compute the DM1 (rho) polarimetric vector in the tau rest frame."""
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
    return spatial(pv.boost(-boost_vec_tau)).unit()

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

def compute_aco_polarimetric(R1, P1, R2, P2, dm_taup, dm_taun):
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

    R1_p3, R1_E = spatial(R1.boost(-bv), return_E=True)
    R2_p3, R2_E = spatial(R2.boost(-bv), return_E=True)
    P1_p3, P1_E = spatial(P1.boost(-bv), return_E=True)
    P2_p3, P2_E = spatial(P2.boost(-bv), return_E=True)

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


def get_ditau_polarimetric_gen(df):
    """Compute the R and P vectors in the gen case (polarimetric vector and full tau)"""
    # Build tau plus from decay products
    pi_p     = ak.zip({"px": df["true_taup_pi1_px"], "py": df["true_taup_pi1_py"], "pz": df["true_taup_pi1_pz"], "E": df["true_taup_pi1_E"]}, with_name="Momentum4D")
    # nu_p     = ak.zip({"px": df["true_taup_nu_px"], "py": df["true_taup_nu_py"], "pz": df["true_taup_nu_pz"], "E": np.sqrt(df["true_taup_nu_px"]**2 + df["true_taup_nu_py"]**2 + df["true_taup_nu_pz"]**2)}, with_name="Momentum4D")
    pizero_p = ak.zip({"px": df["true_taup_pizero1_px"], "py": df["true_taup_pizero1_py"], "pz": df["true_taup_pizero1_pz"], "E": df["true_taup_pizero1_E"]}, with_name="Momentum4D")
    true_taup_is_dm1 = df["true_taup_haspizero"].values == 1
    # tau_p = ak.where(true_taup_is_dm1, pi_p + nu_p + pizero_p, pi_p + nu_p) # if want to build from nu
    tau_p = ak.zip({"px": df["true_tau_plus_px"], "py": df["true_tau_plus_py"], "pz": df["true_tau_plus_pz"], "E": df["true_tau_plus_E"]}, with_name="Momentum4D")

    # Build tau minus from decay products
    pi_n     = ak.zip({"px": df["true_taun_pi1_px"], "py": df["true_taun_pi1_py"], "pz": df["true_taun_pi1_pz"], "E": df["true_taun_pi1_E"]}, with_name="Momentum4D")
    # nu_n     = ak.zip({"px": df["true_taun_nu_px"], "py": df["true_taun_nu_py"], "pz": df["true_taun_nu_pz"], "E": np.sqrt(df["true_taun_nu_px"]**2 + df["true_taun_nu_py"]**2 + df["true_taun_nu_pz"]**2)}, with_name="Momentum4D")
    pizero_n = ak.zip({"px": df["true_taun_pizero1_px"], "py": df["true_taun_pizero1_py"], "pz": df["true_taun_pizero1_pz"], "E": df["true_taun_pizero1_E"]}, with_name="Momentum4D")
    true_taun_is_dm1 = df["true_taun_haspizero"].values == 1
    # tau_n = ak.where(true_taun_is_dm1, pi_n + nu_n + pizero_n, pi_n + nu_n) # if want to build from nu
    tau_n = ak.zip({"px": df["true_tau_minus_px"], "py": df["true_tau_minus_py"], "pz": df["true_tau_minus_pz"], "E": df["true_tau_minus_E"]}, with_name="Momentum4D")


    # Boost to Higgs rest frame
    higgs_bv = boost_vec(tau_p + tau_n)

    H_tau_p = tau_p.boost(-higgs_bv)
    H_tau_n = tau_n.boost(-higgs_bv)
    H_pi_p  = pi_p.boost(-higgs_bv)
    H_pi_n  = pi_n.boost(-higgs_bv)
    H_piz_p = pizero_p.boost(-higgs_bv)
    H_piz_n = pizero_n.boost(-higgs_bv)

    # Compute polarimetric vectors in the respective tau rest frames
    bv_taup = boost_vec(H_tau_p)
    bv_taun = boost_vec(H_tau_n)

    taup_s = ak.where(true_taup_is_dm1,
        polarimetric_vec_dm1(H_pi_p, H_piz_p, H_tau_p, bv_taup),
        polarimetric_vec_dm0(H_pi_p, bv_taup),
    )
    taun_s = ak.where(true_taun_is_dm1,
        polarimetric_vec_dm1(H_pi_n, H_piz_n, H_tau_n, bv_taun),
        polarimetric_vec_dm0(H_pi_n, bv_taun),
    )

    return taup_s, spatial(H_tau_p).unit(), taun_s, spatial(H_tau_n).unit()


def get_ditau_polarimetric_reco(df, smeared=True, useMAP=True):
    """Compute the R and P vectors in the reco case using regressed neutrinos."""
    if smeared:
        prefix = 'reco_tau'
    else:
        prefix = 'tau'
    if useMAP:
        pred_prefix = 'map_pred'
    else:
        pred_prefix = 'pred'
    # Define positive tau vectors
    pi_p     = ak.zip({"px": df[f"{prefix}p_pi1_px"], "py": df[f"{prefix}p_pi1_py"], "pz": df[f"{prefix}p_pi1_pz"], "E": df[f"{prefix}p_pi1_E"]}, with_name="Momentum4D")
    tau_p    = ak.zip({"px": df[f"{pred_prefix}_tau_plus_px"], "py": df[f"{pred_prefix}_tau_plus_py"], "pz": df[f"{pred_prefix}_tau_plus_pz"], "E": df[f"{pred_prefix}_tau_plus_E"]}, with_name="Momentum4D")
    pizero_p = ak.zip({"px": df[f"{prefix}p_pizero1_px"], "py": df[f"{prefix}p_pizero1_py"], "pz": df[f"{prefix}p_pizero1_pz"], "E": df[f"{prefix}p_pizero1_E"]}, with_name="Momentum4D")
    taup_is_dm1 = df[f"{prefix}p_haspizero"].values == 1

    # Define negative tau vectors
    pi_n     = ak.zip({"px": df[f"{prefix}n_pi1_px"], "py": df[f"{prefix}n_pi1_py"], "pz": df[f"{prefix}n_pi1_pz"], "E": df[f"{prefix}n_pi1_E"]}, with_name="Momentum4D")
    tau_n    = ak.zip({"px": df[f"{pred_prefix}_tau_minus_px"], "py": df[f"{pred_prefix}_tau_minus_py"], "pz": df[f"{pred_prefix}_tau_minus_pz"], "E": df[f"{pred_prefix}_tau_minus_E"]}, with_name="Momentum4D")
    pizero_n = ak.zip({"px": df[f"{prefix}n_pizero1_px"], "py": df[f"{prefix}n_pizero1_py"], "pz": df[f"{prefix}n_pizero1_pz"], "E": df[f"{prefix}n_pizero1_E"]}, with_name="Momentum4D")
    taun_is_dm1 = df[f"{prefix}n_haspizero"].values == 1

    # Boost to Higgs rest frame
    higgs_bv = boost_vec(tau_p + tau_n)

    H_tau_p = tau_p.boost(-higgs_bv)
    H_tau_n = tau_n.boost(-higgs_bv)
    H_pi_p  = pi_p.boost(-higgs_bv)
    H_pi_n  = pi_n.boost(-higgs_bv)
    H_piz_p = pizero_p.boost(-higgs_bv)
    H_piz_n = pizero_n.boost(-higgs_bv)

    # Compute polarimetric vectors in the respective tau rest frames
    bv_taup = boost_vec(H_tau_p)
    bv_taun = boost_vec(H_tau_n)

    taup_s = ak.where(taup_is_dm1,
        polarimetric_vec_dm1(H_pi_p, H_piz_p, H_tau_p, bv_taup),
        polarimetric_vec_dm0(H_pi_p, bv_taup),
    )
    taun_s = ak.where(taun_is_dm1,
        polarimetric_vec_dm1(H_pi_n, H_piz_n, H_tau_n, bv_taun),
        polarimetric_vec_dm0(H_pi_n, bv_taun),
    )

    return taup_s, spatial(H_tau_p).unit(), taun_s, spatial(H_tau_n).unit()
