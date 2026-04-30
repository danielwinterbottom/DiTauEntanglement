import awkward as ak
import vector
import numpy as np
vector.register_awkward()

def spatial(v):
    p3 = ak.zip({"x": v.px, "y": v.py, "z": v.pz}, with_name="Vector3D")
    return p3, v.E

def get_R_P_vectors(df, dm, tau_prefix = 'tau'):
    if dm in [0, 1]:
        # P is the charged pion momentum
        P = ak.zip({"px": df[f"{tau_prefix}_pi1_px"], "py": df[f"{tau_prefix}_pi1_py"], "pz": df[f"{tau_prefix}_pi1_pz"], "E": df[f"{tau_prefix}_pi1_e"]}, with_name="Momentum4D")
    if dm == 0:
        # R is the impact parameter vector
        R = ak.zip({"px": df[f"{tau_prefix}_pi1_ipx"], "py": df[f"{tau_prefix}_pi1_ipy"], "pz": df[f"{tau_prefix}_pi1_ipz"], "E": ak.zeros_like(df[f"{tau_prefix}_pi1_ipx"])}, with_name="Momentum4D")
    elif dm == 1:
        # R is the neutral pion
        R = ak.zip({"px": df[f"{tau_prefix}_pizero1_px"], "py": df[f"{tau_prefix}_pizero1_py"], "pz": df[f"{tau_prefix}_pizero1_pz"], "E": df[f"{tau_prefix}_pizero1_e"]}, with_name="Momentum4D")

    print("Successfully computed R and P vectors")
    return R, P

# first test for
def compute_acoplanarity_angle(R1, P1, R2, P2, method_leg1, method_leg2, debug=False):

    print(">>> Computing acoplanarity angle with method:", method_leg1, method_leg2)
    sum_visible = P1 + P2 # old code used .boostvec, hopefully works
    boost_vec  = ak.zip({"x": sum_visible.px / sum_visible.E, "y": sum_visible.py / sum_visible.E, "z": sum_visible.pz / sum_visible.E}, with_name="Vector3D")

    if debug:
        print("Before boost:")
        print("P1:", P1)
        print("P2:", P2)
        print("R1:", R1)
        print("R2:", R2)
        print('*'*60, '\n')
        print("Boost characteristics:")
        print("sum_visible:", sum_visible)
        print("boost_vec:",  boost_vec)
        print('*'*60, '\n')


    # get spatial components of R and P vectors in the rest frame of the di-tau system
    R1_boosted_p3, R1_boosted_E  = spatial(R1.boost(-boost_vec))
    R2_boosted_p3, R2_boosted_E  = spatial(R2.boost(-boost_vec))
    P1_boosted_p3, P1_boosted_E = spatial(P1.boost(-boost_vec))
    P2_boosted_p3, P2_boosted_E = spatial(P2.boost(-boost_vec))

    if debug:
        print("After boost:")
        print("P1_boosted_p3, E", P1_boosted_p3, P1_boosted_E)
        print("P2_boosted_p3, E", P2_boosted_p3, P2_boosted_E)
        print("R1_boosted_p3: E", R1_boosted_p3, R1_boosted_E)
        print("R2_boosted_p3: E", R2_boosted_p3, R2_boosted_E)
        print('*'*60, '\n')

    # compute the components of R that are perpendicular to P
    R1perp = R1_boosted_p3 - ((R1_boosted_p3.dot(P1_boosted_p3))/(P1_boosted_p3.dot(P1_boosted_p3))) * P1_boosted_p3
    R2perp = R2_boosted_p3 - ((R2_boosted_p3.dot(P2_boosted_p3))/(P2_boosted_p3.dot(P2_boosted_p3))) * P2_boosted_p3

    # compute phi*
    angle = np.arccos((R1perp.unit()).dot(R2perp.unit()))
    # compute O*
    sign = (P2_boosted_p3.unit()).dot((R1perp.unit()).cross(R2perp.unit()))

    # compute phiCP
    angle = ak.where(sign >= 0, angle, 2 * np.pi - angle)

    # additional shifts for meson polarisation
    meson_sign = ak.ones_like(angle)
    if method_leg1 == 'DP':
        Y1 = (R1_boosted_E - P1_boosted_E) / (P1_boosted_E + R1_boosted_E)
        meson_sign = meson_sign * np.sign(Y1)
    if method_leg2 == 'DP':
        Y2 = (R2_boosted_E - P2_boosted_E) / (P2_boosted_E + R2_boosted_E)
        meson_sign = meson_sign * np.sign(Y2)
    if method_leg1 == 'DP' or method_leg2 == 'DP':
        angle = ak.where(meson_sign < 0, ak.where(angle < np.pi, angle + np.pi, angle - np.pi), angle)

    return angle