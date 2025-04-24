import ROOT
import numpy as np
import math
from iminuit import Minuit
import gc
import random
import copy

import numpy as np

def solve_abcd_values(P_taup_true, P_taun_true, P_Z, P_taupvis, P_taunvis):
    """
    Solve for a,b,c,d given the true 4-vectors and other inputs.

    Parameters:
        P_taup_true (np.array): True 4-vector of the positive tau (shape: (4,)).
        P_taun_true (np.array): True 4-vector of the negative tau (shape: (4,)).
        P_Z (np.array): 4-vector of Z (shape: (4,)).
        P_taupvis (np.array): Visible 4-vector of the positive tau (shape: (4,)).
        P_taunvis (np.array): Visible 4-vector of the negative tau (shape: (4,)).
        q (np.array): 4-vector for the `q` term (shape: (4,)).

    Returns:
        np.array: The solved a,b,c,d values [a, b, c, d].
    """
    # Number of components in 4-vectors (Energy, px, py, pz)
    num_components = 4

    q = compute_q(P_Z*P_Z,P_Z, P_taupvis, P_taunvis)

    # Initialize matrix A and vector b
    A = np.zeros((2 * num_components, 4))
    b = np.zeros(2 * num_components)

    # Fill matrix A and vector b for P_tau+ equation
    for i in range(num_components):
        A[i, 0] = -P_Z[i] / 2  # Coefficient of c_0
        A[i, 1] = P_taupvis[i] / 2  # Coefficient of c_1
        A[i, 2] = -P_taunvis[i] / 2  # Coefficient of c_2
        A[i, 3] = q[i]  # Coefficient of c_3

        b[i] = P_taup_true[i] - P_Z[i] / 2  # Right-hand side

    # Fill matrix A and vector b for P_tau- equation
    for i in range(num_components):
        A[num_components + i, 0] = P_Z[i] / 2  # Coefficient of c_0
        A[num_components + i, 1] = -P_taupvis[i] / 2  # Coefficient of c_1
        A[num_components + i, 2] = P_taunvis[i] / 2  # Coefficient of c_2
        A[num_components + i, 3] = -q[i]  # Coefficient of c_3

        b[num_components + i] = P_taun_true[i] - P_Z[i] / 2  # Right-hand side

    # Solve the system A * [c_0, c_1, c_2, c_3]^T = b
    abcd_values, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return abcd_values

class TauReconstructor:
    def __init__(self, mtau=1.777):
        """
        Initialize the TauReconstructor class.

        Args:
            mtau (float): Mass of the tau particle in GeV.
        """
        self.mtau = mtau
        self.d_min = 0.


    def sort_solutions(self, solutions, mode=1, d_min_reco=None):#, O_y=None, np_point=None, nn_point=None):
        # sort tau solutions depending on the selected mode
        # mode=0 is a random sorting
        # mode=1 is sorting based on the predicted d_min direction
        # mode=2 is sorting based on the tau decay length probabilities
        # mode=3 is sorting based on the sv_delta constraint (tau momenta's allign with SVs)

        if mode == 0:
            np.random.shuffle(solutions)

        elif mode == 1:
            if d_min_reco is not None:
                solutions = sorted(solutions, key=lambda x: x[2].Unit().Dot(d_min_reco.Unit()), reverse=True)
            if d_min_reco is None:
                print("Warning: selected mode=1 for tau sorting but d_min_reco is None. Randomly shuffling solutions instead (mode=0).")
                np.random.shuffle(solutions)
        
        elif mode == 2:
            solutions = sorted(solutions, key=lambda x: x[3], reverse=True)

        elif mode == 3:
            solutions = sorted(solutions, key=lambda x: x[4], reverse=True)

        return solutions

    def reconstruct_tau_alt(self, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, O_y=None, np_point=ROOT.TVector3(), nn_point=ROOT.TVector3(), d_min_reco=None, sv_delta=None, mode=1, verbose=False,no_minimisation=False):
        """
        """

        # use analytic solutions as initial guesses
        analytic_solutions = ReconstructTauAnalytically(P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, return_values=True)
        sorted_solutions = self.sort_solutions(analytic_solutions, mode=mode, d_min_reco=d_min_reco)

        _, _, _, a1, b1, c1, d1 = sorted_solutions[0] 
        _, _, _, a2, b2, c2, d2 = sorted_solutions[1]

        def objective(a, b, c, d):
            vars = [a, b, c, d]
            return self._objective_alt2(vars, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, d_min_reco, np_point, nn_point, O_y)

        initial_guess_1= [a1, b1, c1, d1]
        initial_guess_2= [a2, b2, c2, d2]

        # perform minimisation for each solution
        minuit_1 = Minuit(
            objective,
            a=initial_guess_1[0],
            b=initial_guess_1[1],
            c=initial_guess_1[2],
            d=initial_guess_1[3]
        )

        minuit_2 = Minuit(
            objective,
            a=initial_guess_2[0],
            b=initial_guess_2[1],
            c=initial_guess_2[2],
            d=initial_guess_2[3]
        )

        # Perform the minimization
        if not no_minimisation:
            minuit_1.migrad()
            minuit_2.migrad()
            if verbose: 
                if minuit_1.fmin.is_valid:
                    print("Optimization 1 succeeded!")
                    print(f"Objective value at minimum: {minuit.fval}")
                else:
                    print("Optimization 1 failed.")

                if minuit_2.fmin.is_valid:
                    print("Optimization 2 succeeded!")
                    print(f"Objective value at minimum: {minuit.fval}")
                else:
                    print("Optimization 2 failed.")

            # pick the best solution based on the objective value
            if minuit_1.fmin.is_valid and minuit_2.fmin.is_valid:
                if minuit_1.fval < minuit_2.fval:
                    minuit = minuit_1
                else:
                    minuit = minuit_2
            elif minuit_1.fmin.is_valid:
                minuit = minuit_1
            elif minuit_2.fmin.is_valid:
                minuit = minuit_2

        else:
            minuit = minuit_1

        q = compute_q(P_Z*P_Z,P_Z, P_taupvis, P_taunvis)

        solutions = []

        for minuit in [minuit_1, minuit_2]:

            P_taup_reco = (1.-minuit.values[0])/2*P_Z + minuit.values[1]/2*P_taupvis - minuit.values[2]/2*P_taunvis + minuit.values[3]*q
            P_taun_reco = (1.+minuit.values[0])/2*P_Z - minuit.values[1]/2*P_taupvis + minuit.values[2]/2*P_taunvis - minuit.values[3]*q     

            # set tau masses equal to the expected value since minimization does not strictly enforce this 
            P_taup_reco.SetE((P_taup_reco.P()**2+self.mtau**2)**.5)
            P_taun_reco.SetE((P_taun_reco.P()**2+self.mtau**2)**.5)

            decay_length_prob = 1.

            if O_y is not None and nn_point is not None and np_point is not None:
                P_intersection, N_intersection, _, O  = find_intersections_fixed_Oy(P_taup_reco, P_taun_reco, np_point, P_taup_pi1.Vect().Unit(), nn_point, P_taun_pi1.Vect().Unit(), O_y)
                prob_taup = tau_decay_probability(P_taup_reco.P(), P_taup_reco.E(), (P_intersection-O).Mag())
                prob_taun = tau_decay_probability(P_taun_reco.P(), P_taun_reco.E(), (N_intersection-O).Mag())
                decay_length_prob = prob_taup*prob_taun

                # if either tau is determined to be going in the negative direction then set probabilities to negative numbers so they are unfavoured
                taup_sign = (P_intersection-O).Unit().Dot(P_taup_reco.Vect().Unit())
                taun_sign = (N_intersection-O).Unit().Dot(P_taun_reco.Vect().Unit())
                if taup_sign<0 or taun_sign<0: decay_length_prob = decay_length_prob-1. # instead of setting it to 0 we set it to decay_length_prob-1 so that we can still sort the solutions based on the decay length probability
                #TODO: store this instead just in case we end up with cases where both solutions give negative tau directions when we have smearing
            
            if sv_delta is not None:
                sv_delta_constraint = (sv_delta.Unit().Dot(P_taup_reco.Vect().Unit()) - sv_delta.Unit().Dot(P_taun_reco.Vect().Unit()))/2
            else: sv_delta_constraint = 0.

            d_min, _ = GetDMins(P_taup_reco, P_taun_reco, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1)

            solutions.append((P_taup_reco, P_taun_reco, d_min, decay_length_prob, sv_delta_constraint))

        sorted_solutions = self.sort_solutions(solutions, mode=mode, d_min_reco=d_min_reco)

        # add a solution where minuit.values[3] = 0 (i.e d^2 =0)
        P_taup_reco = (1.-minuit_1.values[0])/2*P_Z + minuit_1.values[1]/2*P_taupvis - minuit_1.values[2]/2*P_taunvis
        P_taun_reco = (1.+minuit_1.values[0])/2*P_Z - minuit_1.values[1]/2*P_taupvis + minuit_1.values[2]/2*P_taunvis
        sorted_solutions.append((P_taup_reco, P_taun_reco, None, None, None))
          
        if verbose:

            print('P_Z:', P_Z.X(), P_Z.Y(), P_Z.Z(), P_Z.M())
            print('P_taupvis:', P_taupvis.X(), P_taupvis.Y(), P_taupvis.Z(), P_taupvis.M())
            print('P_taunvis:', P_taunvis.X(), P_taunvis.Y(), P_taunvis.Z(), P_taunvis.M())
            print('P_taup_pi1:', P_taup_pi1.X(), P_taup_pi1.Y(), P_taup_pi1.Z(), P_taup_pi1.M())
            print('P_taun_pi1:', P_taun_pi1.X(), P_taun_pi1.Y(), P_taun_pi1.Z(), P_taun_pi1.M())
            print('O_y:', O_y)
            print('np_point:', np_point.X(), np_point.Y(), np_point.Z())
            print('nn_point:', nn_point.X(), nn_point.Y(), nn_point.Z())
            print('d_min_reco:', d_min_reco.X(), d_min_reco.Y(), d_min_reco.Z())
            print('sv_delta:', sv_delta.X(), sv_delta.Y(), sv_delta.Z())
            print('mode:', mode)

            print('numerical solution 1:')
            print('tau+:', sorted_solutions[0][0].X(), sorted_solutions[0][0].Y(), sorted_solutions[0][0].Z(), sorted_solutions[0][0].M())
            print('tau-:', sorted_solutions[0][1].X(), sorted_solutions[0][1].Y(), sorted_solutions[0][1].Z(), sorted_solutions[0][1].M())
            print('d_min:', sorted_solutions[0][2].X(), sorted_solutions[0][2].Y(), sorted_solutions[0][2].Z())
            print('d_min constraint:', sorted_solutions[0][2].Unit().Dot(d_min_reco.Unit()))
            print('decay length probability:', sorted_solutions[0][3])

            print('numerical solution 2:')
            print('tau+:', sorted_solutions[1][0].X(), sorted_solutions[1][0].Y(), sorted_solutions[1][0].Z(), sorted_solutions[1][0].M())
            print('tau-:', sorted_solutions[1][1].X(), sorted_solutions[1][1].Y(), sorted_solutions[1][1].Z(), sorted_solutions[1][1].M())
            print('d_min:', sorted_solutions[1][2].X(), sorted_solutions[1][2].Y(), sorted_solutions[1][2].Z())
            print('d_min constraint:', sorted_solutions[1][2].Unit().Dot(d_min_reco.Unit()))
            print('decay length probability:', sorted_solutions[1][3])

        return sorted_solutions

    def _objective_alt(self, vars, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, d_min_reco=None, np_point=None, nn_point=None, O_y=None):

        a, b, c, d = vars

        q = compute_q(P_Z*P_Z,P_Z, P_taupvis, P_taunvis)

        x = P_Z * P_taupvis
        y = P_Z * P_taunvis
        z = P_taupvis * P_taunvis

        m_Z_sq = P_Z.Mag2()
        m_pvis_sq = P_taupvis.Mag2()
        m_nvis_sq = P_taunvis.Mag2()
        q_sq = q*q

        eq1 = self.mtau**2 + m_pvis_sq -x +a*x -b*m_pvis_sq +c*z
        eq2 = self.mtau**2 + m_nvis_sq -y -a*y + b*z -c*m_nvis_sq
        eq3 = a*m_Z_sq -b*x + c*y + (b**2+c**2)/4*(m_nvis_sq - m_pvis_sq)
        eq4 = (1.+a**2)/2*m_Z_sq + b**2*m_pvis_sq/2 +c**2*m_nvis_sq/2 +2*d**2*q_sq -a*b*x + a*c*y - b*c*z - 2*self.mtau**2

        P_taup = (1.-a)/2*P_Z + b/2*P_taupvis - c/2*P_taunvis + d*q
        P_taun = (1.+a)/2*P_Z - b/2*P_taupvis + c/2*P_taunvis - d*q

        if d_min_reco is not None and np_point is not None and nn_point is not None and O_y is not None and False: # not implemented for now, need to think how to do this properly
            P_intersection, N_intersection, _, O  = find_intersections_fixed_Oy(P_taup, P_taun, np_point, P_taup_pi1.Vect().Unit(), nn_point, P_taun_pi1.Vect().Unit(), O_y)

            if isinstance(P_taup, ROOT.TLorentzVector):
                prob_taup = tau_decay_probability(P_taup.P(), P_taup.E(), (P_intersection-O).Mag())
            else: prob_taup=1. 
            if isinstance(P_taun, ROOT.TLorentzVector):
                prob_taun = tau_decay_probability(P_taun.P(), P_taun.E(), (N_intersection-O).Mag())  
            else: prob_taun=1.
            # check for nans and infinities and replace with 0
            if math.isnan(prob_taup) or math.isinf(prob_taup): prob_taup=0.
            if math.isnan(prob_taun) or math.isinf(prob_taun): prob_taun=0.

            eq5 = 1.-prob_taup*prob_taun # I would not use this as it biases towards small tau displacements
            #eq5 = 1-min(prob_taup,0.33)*min(prob_taun,0.33) # this effectivly doesn't penalise taus provided they are within 1 sigma of the expected decay length, otherwise we penalise them

            P_taun_pi1_new = ChangeFrame(P_taun, P_taun_pi1, P_taun_pi1)
            P_taup_pi1_new = ChangeFrame(P_taun, P_taun_pi1, P_taup_pi1)
            d_min = PredictDmin(
                P_taun_pi1_new, P_taup_pi1_new, ROOT.TVector3(0.0, 0.0, -1.0)
            ).Unit()
            d_min = ChangeFrame(P_taun, P_taun_pi1, d_min, reverse=True)
            #eq5 = (1.0 - d_min.Unit().Dot(d_min_reco.Unit()))**2
            return (eq1**2 + eq2**2 + eq3**2 + eq4**2) #+ eq5*1000.
        else: 
            return eq1**2 + eq2**2 + eq3**2 + eq4**2

    def _objective_alt2(self, vars, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, d_min_reco=None, np_point=None, nn_point=None, O_y=None):

        a, b, c, d = vars

        q = compute_q(P_Z*P_Z,P_Z, P_taupvis, P_taunvis)

        x = P_Z * P_taupvis
        y = P_Z * P_taunvis
        z = P_taupvis * P_taunvis

        m_Z_sq = P_Z.Mag2()
        m_pvis_sq = P_taupvis.Mag2()
        m_nvis_sq = P_taunvis.Mag2()
        q_sq = q*q

        eq1 = self.mtau**2 + m_pvis_sq -x +a*x -b*m_pvis_sq +c*z
        eq2 = self.mtau**2 + m_nvis_sq -y -a*y + b*z -c*m_nvis_sq
        eq3 = a*m_Z_sq -b*x + c*y + (b**2+c**2)/4*(m_nvis_sq - m_pvis_sq)
        eq4 = (1.+a**2)/2*m_Z_sq + b**2*m_pvis_sq/2 +c**2*m_nvis_sq/2 +2*d**2*q_sq -a*b*x + a*c*y - b*c*z - 2*self.mtau**2

        P_taup = (1.-a)/2*P_Z + b/2*P_taupvis - c/2*P_taunvis + d*q
        P_taun = (1.+a)/2*P_Z - b/2*P_taupvis + c/2*P_taunvis - d*q

        if d_min_reco is not None and np_point is not None and nn_point is not None and O_y is not None: # not implemented for now, need to think how to do this properly
            P_intersection, N_intersection, _, O  = find_intersections_fixed_Oy(P_taup, P_taun, np_point, P_taup_pi1.Vect().Unit(), nn_point, P_taun_pi1.Vect().Unit(), O_y)

            #if isinstance(P_taup, ROOT.TLorentzVector):
            #    prob_taup = tau_decay_probability(P_taup.P(), P_taup.E(), (P_intersection-O).Mag())
            #else: prob_taup=1. 
            #if isinstance(P_taun, ROOT.TLorentzVector):
            #    prob_taun = tau_decay_probability(P_taun.P(), P_taun.E(), (N_intersection-O).Mag())  
            #else: prob_taun=1.
            ## check for nans and infinities and replace with 0
            #if math.isnan(prob_taup) or math.isinf(prob_taup): prob_taup=0.
            #if math.isnan(prob_taun) or math.isinf(prob_taun): prob_taun=0.

            dist1 = (P_intersection-O).Mag()
            dist2 = (N_intersection-O).Mag()

            dist1_sign = np.sign((P_intersection-O).Unit().Dot(P_taup.Vect().Unit()))
            dist2_sign = np.sign((N_intersection-O).Unit().Dot(P_taun.Vect().Unit()))

            #eq5 = dist1+dist2 +1000*(dist1_sign<0) + 1000*(dist2_sign<0) # penalise solutions where the tau is going in the wrong direction
            eq5 = 1000*(dist1_sign<0) + 1000*(dist2_sign<0)

            #eq5 = 1.-prob_taup*prob_taun # I would not use this as it biases towards small tau displacements
            #eq5 = 1-min(prob_taup,0.33)*min(prob_taun,0.33) # this effectivly doesn't penalise taus provided they are within 1 sigma of the expected decay length, otherwise we penalise them

            #P_taun_pi1_new = ChangeFrame(P_taun, P_taun_pi1, P_taun_pi1)
            #P_taup_pi1_new = ChangeFrame(P_taun, P_taun_pi1, P_taup_pi1)
            #d_min = PredictDmin(
            #    P_taun_pi1_new, P_taup_pi1_new, ROOT.TVector3(0.0, 0.0, -1.0)
            #).Unit()
            #d_min = ChangeFrame(P_taun, P_taun_pi1, d_min, reverse=True)
            #eq5 = (1.0 - d_min.Unit().Dot(d_min_reco.Unit()))**2
            #print('eq5:', eq5)
            return (eq1**2 + eq2**2 + eq3**2 + eq4**2) + eq5
        else: 
            return eq1**2 + eq2**2 + eq3**2 + eq4**2

def ChangeFrame(taun, taunvis, vec, reverse=False):
    '''
    Rotate the coordinate axis as follows:
    z-direction should be direction of the tau-
    xz-axis should adjust the direction so that the h- lies on the x-z plane, in the +ve x-direction.

    Args:
        taun (TVector3): The tau- vector.
        taunvis (TVector3): The visible tau- vector (h-).
        vec (TVector3): The vector to rotate.
        reverse (bool): If True, applies the inverse transformation to return to the original frame.

    Returns:
        TVector3: The rotated vector.
    '''
    vec_new = vec.Clone()
    taunvis_copy = taunvis.Clone()
    
    # Define the rotation angles to allign with tau- direction
    theta = taun.Theta()
    phi = taun.Phi()

    # Rotate taunvis to get second phi angle for rotation
    taunvis_copy.RotateZ(-phi)
    taunvis_copy.RotateY(-theta)
    phi2 = taunvis_copy.Phi()  # This is the phi angle of the rotated taunvis vector 

    if reverse:
        # Reverse transformation: Undo the rotations in reverse order

        vec_new.RotateZ(phi2)
        vec_new.RotateY(theta)
        vec_new.RotateZ(phi)
    else:
        # Forward transformation: Apply the rotations

        vec_new.RotateZ(-phi)
        vec_new.RotateY(-theta)
        vec_new.RotateZ(-phi2)

        # Check that h- is in the +ve x-direction
        if taunvis == vec and vec_new.X() < 0:
            raise ValueError("h- not pointing in the +ve x direction")

    return vec_new

def compare_lorentz_pairs(pair1, pair2):
    """
    Compare two pairs of TLorentzVectors by calculating the sum of squares of differences
    between their x, y, and z components.
    
    Parameters:
    - pair1: tuple of two TLorentzVectors (vector1, vector2)
    - pair2: tuple of two TLorentzVectors (vector3, vector4)
    
    Returns:
    - float: Sum of the squared differences for the x, y, and z components.
    """
    # Extract TLorentzVectors from pairs
    vec1, vec2    = pair1
    vec3, vec4, _, _, _   = pair2

    # Compute the squared differences for each component
    dx = (vec1.X() - vec3.X())**2 + (vec2.X() - vec4.X())**2
    dy = (vec1.Y() - vec3.Y())**2 + (vec2.Y() - vec4.Y())**2
    dz = (vec1.Z() - vec3.Z())**2 + (vec2.Z() - vec4.Z())**2

    # Return the sum of squared differences
    return (dx + dy + dz)**.5

def GetOpeningAngle(tau, h):
    beta = tau.P()/tau.E()
    if beta >= 1:
        #print('Warning: Beta is >= 1, invalid for physical particles (Beta = %g). Recomputing using the expected tau mass.' % beta)
        beta = tau.P()/(tau.P()**2 + 1.777**2)**0.5 # if beta is unphysical then recalculate it using the expected mass of the tau lepton
        #print('recalculated beta:', beta)
    gamma = 1. / (1. - beta**2)**0.5
    x = h.E()/tau.E()
    r = h.M()/tau.M()

    costheta = min((gamma*x - (1.+r**2)/(2*gamma))/(beta*(gamma**2*x**2-r**2)**.5),1.)
    sintheta = (((1.-r**2)**2/4 - (x-(1.+r**2)/2)**2/beta**2)/(gamma**2*x**2-r**2))**.5
    # if sintheta is complex then set to 0
    if sintheta.imag != 0:
        sintheta = 0.
    if round(math.acos(costheta), 3) != round(math.asin(sintheta), 3):    
        raise ValueError("theta angles do not match", math.acos(costheta), math.asin(sintheta))

    return math.acos(costheta)


# analytical reconstruction functions 

# Define the Levi-Civita symbol in 4D
def levi_civita_4d():
    """Returns the 4D Levi-Civita tensor."""
    epsilon = np.zeros((4, 4, 4, 4), dtype=int)
    indices = [0, 1, 2, 3]

    for perm in np.array(np.meshgrid(*[indices]*4)).T.reshape(-1, 4):
        i, j, k, l = perm
        sign = np.sign(np.linalg.det([[1 if x == y else 0 for x in indices] for y in perm]))
        epsilon[i, j, k, l] = sign

    return epsilon

def compute_q(M2, p_a, p_b, p_c):
    """
    Computes p^mu = ε_{μνρσ} p_a^ν p_b^ρ p_c^σ / M2
    """
    epsilon = levi_civita_4d()
    p_mu = np.zeros(4)

    # Loop over μ, ν, r, s
    for m in range(4):
        for n in range(4):
            for r in range(4):
                for s in range(4):
                    # Contribution from ε_{μνρσ} * p_a^ν * p_b^ρ * p_c^ρσ
                    p_mu[m] += epsilon[m, n, r, s] * p_a[n] * p_b[r] * p_c[s]

    return ROOT.TLorentzVector(p_mu[0]/M2, p_mu[1]/M2, p_mu[2]/M2, p_mu[3]/M2)

def ReconstructTauAnalytically(P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, O_y=None, np_point=ROOT.TVector3(), nn_point=ROOT.TVector3(), verbose=False, return_values=False):
    
    '''
    Reconstuct tau lepton 4-momenta with 2-fold degeneracy
    following formulation in Appendix C of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.107.093002
    '''

    m_tau = 1.777

    m_taupvis = P_taupvis * P_taupvis
    m_taunvis = P_taunvis * P_taunvis

    q = compute_q(P_Z*P_Z,P_Z, P_taupvis, P_taunvis)

    x = P_Z * P_taupvis
    y = P_Z * P_taunvis
    z = P_taupvis * P_taunvis

    m_Z_sq = P_Z.M2()

    M = np.array([[-x,     P_taupvis.M2(), -z     ],
                  [y,      -z,      P_taunvis.M2()],
                  [m_Z_sq, -x,      y]])

    lamx = m_tau**2 + P_taupvis.M2() -x
    lamy = m_tau**2 + P_taunvis.M2() -y

    L = np.array([[lamx],
                  [lamy],
                  [0.]])

    M_inv = np.linalg.inv(M)

    v = np.dot(M_inv, L)

    a = v[0][0]
    b = v[1][0]
    c = v[2][0]

    dsq = 1./(-4*q*q) * ( (1+a**2)*m_Z_sq + b**2*P_taupvis.M2() + c**2*P_taunvis.M2() -4*m_tau**2 + 2*(a*c*y - a*b*x - b*c*z)) # this version gives the correct result, but need to work out why...
    d = dsq**.5 if dsq > 0 else 0.

    solutions = []

    for i, d in enumerate([d,-d]):
        taup = (1.-a)/2*P_Z + b/2*P_taupvis - c/2*P_taunvis + d*q
        taun = (1.+a)/2*P_Z - b/2*P_taupvis + c/2*P_taunvis - d*q

        P_taunvis_new = ChangeFrame(taun, P_taunvis, P_taunvis)
        P_taupvis_new = ChangeFrame(taun, P_taunvis, P_taupvis)

        d_min = PredictDmin(
            P_taunvis_new, P_taupvis_new, ROOT.TVector3(0.0, 0.0, -1.0)
        ).Unit()
        d_min = ChangeFrame(taun, P_taunvis, d_min, reverse=True)    

        if return_values: solutions.append((taup, taun, d_min, a, b, c, d))
        else: solutions.append((taup, taun, d_min))
    return tuple(solutions)

def GetDMins(P_taup, P_taun, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1):
    P_taunvis_new = ChangeFrame(P_taun, P_taunvis, P_taunvis)
    P_taupvis_new = ChangeFrame(P_taun, P_taunvis, P_taupvis)

    d_min = PredictDmin(
        P_taunvis_new, P_taupvis_new, ROOT.TVector3(0.0, 0.0, -1.0)
    ).Unit()
    d_min = ChangeFrame(P_taun, P_taunvis, d_min, reverse=True)  

    P_taun_pi1_new = ChangeFrame(P_taun, P_taun_pi1, P_taun_pi1)
    P_taup_pi1_new = ChangeFrame(P_taun, P_taun_pi1, P_taup_pi1)

    d_min_tracks = PredictDmin(
        P_taun_pi1_new, P_taup_pi1_new, ROOT.TVector3(0.0, 0.0, -1.0)
    ).Unit()
    d_min_tracks = ChangeFrame(P_taun, P_taun_pi1, d_min_tracks, reverse=True) 

    return d_min, d_min_tracks

def GetGenImpactParam(primary_vtx, secondary_vtx, part_vec):
    sv = secondary_vtx - primary_vtx # secondary vertex wrt primary vertex    
    unit_vec = part_vec.Unit() # direction of particle / track
    ip = (sv - sv.Dot(unit_vec)*unit_vec)
    return ip  

def FindDMin(p1, d1, p2, d2, return_points=False):
    """
    Find the vector pointing from the closest point on Line 1 to the closest point on Line 2 using ROOT classes.

    Args:
        p1 (TVector3): A point on Line 1.
        d1 (TVector3): Direction vector of Line 1.
        p2 (TVector3): A point on Line 2.
        d2 (TVector3): Direction vector of Line 2.

    Returns:
        TVector3: Vector pointing from the closest point on Line 1 to the closest point on Line 2.
    """
    # Normalize direction vectors
    d1 = d1.Unit()
    d2 = d2.Unit()

    # Compute cross product and its magnitude squared
    cross_d1_d2 = d1.Cross(d2)
    denom = cross_d1_d2.Mag2()

    # Handle parallel lines (cross product is zero)
    if denom == 0:
        raise ValueError("The lines are parallel or nearly parallel.")

    # Compute t1 and t2 using determinant approach
    dp = p2 - p1
    t1 = (dp.Dot(d2.Cross(cross_d1_d2))) / denom
    t2 = (dp.Dot(d1.Cross(cross_d1_d2))) / denom

    # Closest points on each line
    pca1 = p1 + d1 * t1
    pca2 = p2 + d2 * t2

    if return_points:
        return pca2 - pca1, pca1, pca2

    else: 
        return pca2 - pca1

def GetPointShifts(p1, d1, p2, d2, delta):

    # d2,p2 is the line that the shifts in the point are computed for

    # Normalize direction vectors
    d1 = d1.Unit()
    d2 = d2.Unit()

    # Compute cross product and its magnitude squared
    cross_d1_d2 = d1.Cross(d2)
    denom = cross_d1_d2.Mag2()

    # Handle parallel lines (cross product is zero)
    if denom == 0:
        raise ValueError("The lines are parallel or nearly parallel.")

    C2 = d1.Cross(cross_d1_d2)* (1./denom)   

    delta_arr = np.array([delta.X(), delta.Y(), delta.Z()])

    M_2 = np.array([[1+d2.X()*C2.X(), d2.X()*C2.Y(), d2.X()*C2.Z()],
                    [d2.Y()*C2.X(), 1+d2.Y()*C2.Y(), d2.Y()*C2.Z()],
                    [d2.Z()*C2.X(), d2.Z()*C2.Y(), 1+d2.Z()*C2.Z()]])

    M_2_inv = np.linalg.pinv(M_2)

    sigma_2 = np.dot(M_2_inv, delta_arr)

    return sigma_2                    

def PredictD(P_taun, P_taup, P_taunvis, P_taupvis, d_min):
  d = ROOT.TVector3()

  n_n = P_taunvis.Vect().Unit()
  n_p = P_taupvis.Vect().Unit()

  theta_n = GetOpeningAngle(P_taun, P_taunvis)
  theta_p = GetOpeningAngle(P_taup, P_taupvis)

  sin_n = math.sin(theta_n)
  sin_p = math.sin(theta_p)
  cos_n = math.cos(theta_n)
  cos_p = math.cos(theta_p)


  proj = d_min.Dot(n_p.Cross(n_n))

  if sin_n*sin_p==0.: cosphi=1.
  else: cosphi = (n_n.Dot(n_p) + cos_n*cos_p)/(sin_n*sin_p)
  cosphi = max(-1.0, min(1.0, cosphi))
  sinphi = math.sin(math.acos(cosphi))

  if sin_n*sin_p*sinphi == 0: l = 1 # TODO: this might need a better fix
  else: l = abs(proj/(sin_n*sin_p*sinphi)) # this l seems to be close but slightly different to the true l even when inputting gen quantities, need to figure out why


  d_min_over_l = d_min * (1./l)

  fact1 = (cos_p*n_p.Dot(n_n) + cos_n) / (1.-(n_n.Dot(n_p))**2)
  fact2 = (-cos_n*n_p.Dot(n_n) - cos_p) / (1.-(n_n.Dot(n_p))**2)

  term1 = n_n*fact1
  term2 = n_p*fact2
  d_over_l = d_min_over_l - term1 - term2

  #print('d_unit_reco:', d_over_l.X(), d_over_l.Y(), d_over_l.Z())
  #print('d_reco:', d_over_l.X()*l_true, d_over_l.Y()*l_true, d_over_l.Z()*l_true)

  d.SetX(d_over_l.X()*l)
  d.SetY(d_over_l.Y()*l)
  d.SetZ(d_over_l.Z()*l)

  return d

def PredictDmin(P_taunvis, P_taupvis, d=None):
    if d is None: d_over_l = ROOT.TVector3(0.,0.,-1.) # by definition in the rotated coordinate frame
    else: d_over_l = d
    n_n = P_taunvis.Vect().Unit()
    n_p = P_taupvis.Vect().Unit()
    fact1 = (d_over_l.Dot(n_p)*n_n.Dot(n_p) - d_over_l.Dot(n_n))/(1.-(n_n.Dot(n_p))**2)
    fact2 = (d_over_l.Dot(n_n)*n_n.Dot(n_p) - d_over_l.Dot(n_p))/(1.-(n_n.Dot(n_p))**2)
    term1 = n_n*fact1
    term2 = n_p*fact2

    d_min = d_over_l + (term1 + term2)
    return d_min

def find_intersections(taup, taun, np_point, np_dir, nn_point, nn_dir):
    # assumes PV is known (hard-coded as 0,0,0)
    # Convert TVector3 inputs to NumPy arrays for computation
    taup_arr = np.array([taup.X(), taup.Y(), taup.Z()])
    taun_arr = np.array([taun.X(), taun.Y(), taun.Z()])
    np_point_arr = np.array([np_point.X(), np_point.Y(), np_point.Z()])
    np_dir_arr = np.array([np_dir.X(), np_dir.Y(), np_dir.Z()])
    nn_point_arr = np.array([nn_point.X(), nn_point.Y(), nn_point.Z()])
    nn_dir_arr = np.array([nn_dir.X(), nn_dir.Y(), nn_dir.Z()])

    # Solve for lambda_p, lambda_n, t_p, t_n
    A = np.array([
        [taup_arr[0], -np_dir_arr[0], 0, 0],
        [taup_arr[1], -np_dir_arr[1], 0, 0],
        [taup_arr[2], -np_dir_arr[2], 0, 0],
        [0, 0, taun_arr[0], -nn_dir_arr[0]],
        [0, 0, taun_arr[1], -nn_dir_arr[1]],
        [0, 0, taun_arr[2], -nn_dir_arr[2]]
    ])  

    b = np.array([
        np_point_arr[0], np_point_arr[1], np_point_arr[2],
        nn_point_arr[0], nn_point_arr[1], nn_point_arr[2]
    ])
    
    try:
        solution = np.linalg.lstsq(A, b, rcond=None)[0]
        lambda_p, tp, lambda_n, tn = solution
    except np.linalg.LinAlgError:
        print("No unique solution exists.")
        return None
  
    
    # Compute intersection points
    P_intersection_arr = taup_arr * lambda_p
    N_intersection_arr = taun_arr * lambda_n
    
    # Convert back to TVector3
    P_intersection = ROOT.TVector3(*P_intersection_arr)
    N_intersection = ROOT.TVector3(*N_intersection_arr)
    
    d_min_pred = FindDMin(N_intersection, nn_dir.Unit(), P_intersection, np_dir.Unit())

    return P_intersection, N_intersection, d_min_pred

def find_intersections_fixed_OxOy(taup, taun, np_point, np_dir, nn_point, nn_dir, O_x, O_y):
    # assumes only Ox and Oy are known (excatly)
    # Convert TVector3 inputs to NumPy arrays
    taup_arr = np.array([taup.X(), taup.Y(), taup.Z()])
    taun_arr = np.array([taun.X(), taun.Y(), taun.Z()])
    np_point_arr = np.array([np_point.X(), np_point.Y(), np_point.Z()])
    np_dir_arr = np.array([np_dir.X(), np_dir.Y(), np_dir.Z()])
    nn_point_arr = np.array([nn_point.X(), nn_point.Y(), nn_point.Z()])
    nn_dir_arr = np.array([nn_dir.X(), nn_dir.Y(), nn_dir.Z()])

    A = np.array([
        [taup_arr[0], -np_dir_arr[0], 0, 0, 0],
        [taup_arr[1], -np_dir_arr[1], 0, 0, 0],
        [taup_arr[2], -np_dir_arr[2], 0, 0, 1],
        [0, 0. , taun_arr[0], -nn_dir_arr[0], 0],
        [0, 0. , taun_arr[1], -nn_dir_arr[1], 0],
        [0, 0. , taun_arr[2], -nn_dir_arr[2], 1],
    ])    

    b = np.array([
        np_point_arr[0] - O_x,
        np_point_arr[1] - O_y,
        np_point_arr[2], 
        nn_point_arr[0] - O_x,
        nn_point_arr[1] - O_y,
        nn_point_arr[2],
    ])
    
    # Solve for lambda_p, tp, lambda_n, tn, O_z
    solution = np.linalg.lstsq(A, b, rcond=None)[0]
    lambda_p, tp, lambda_n, tn, O_z = solution
     
    O_arr = np.array([O_x, O_y, O_z])

    # Compute intersection points
    P_intersection_arr = O_arr + taup_arr * lambda_p
    N_intersection_arr = O_arr + taun_arr * lambda_n
    
    # Convert back to TVector3
    P_intersection = ROOT.TVector3(*P_intersection_arr)
    N_intersection = ROOT.TVector3(*N_intersection_arr)

    d_min_pred = FindDMin(N_intersection, nn_dir.Unit(), P_intersection, np_dir.Unit())

    return P_intersection, N_intersection, d_min_pred, ROOT.TVector3(O_x, O_y, O_z)  

def find_intersections_fixed_Oy(taup, taun, np_point, np_dir, nn_point, nn_dir, O_y):
    # assumes only Oy is known (excatly)
    # Convert TVector3 inputs to NumPy arrays
    taup_arr = np.array([taup.X(), taup.Y(), taup.Z()])
    taun_arr = np.array([taun.X(), taun.Y(), taun.Z()])
    np_point_arr = np.array([np_point.X(), np_point.Y(), np_point.Z()])
    np_dir_arr = np.array([np_dir.X(), np_dir.Y(), np_dir.Z()])
    nn_point_arr = np.array([nn_point.X(), nn_point.Y(), nn_point.Z()])
    nn_dir_arr = np.array([nn_dir.X(), nn_dir.Y(), nn_dir.Z()])

    A = np.array([
        [taup_arr[0], -np_dir_arr[0], 0, 0, 0, 1],
        [taup_arr[1], -np_dir_arr[1], 0, 0, 0, 0],
        [taup_arr[2], -np_dir_arr[2], 0, 0, 1, 0],
        [0, 0. , taun_arr[0], -nn_dir_arr[0], 0, 1],
        [0, 0. , taun_arr[1], -nn_dir_arr[1], 0, 0],
        [0, 0. , taun_arr[2], -nn_dir_arr[2], 1, 0],
    ])

    b = np.array([
        np_point_arr[0],
        np_point_arr[1] - O_y,
        np_point_arr[2], 
        nn_point_arr[0],
        nn_point_arr[1] - O_y,
        nn_point_arr[2],
    ])
    
    # Solve for lambda_p, tp, lambda_n, tn, O_z, O_x
    # check if matrix is singlular and if so use least squares
    try:
        # Attempt to solve the system using np.linalg.solve
        solution = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to least squares if solve fails
        #print("Warning: Matrix equation could not be solved exactly. Using least squares instead.")
        solution = np.linalg.lstsq(A, b, rcond=None)[0]
    lambda_p, tp, lambda_n, tn, O_z, O_x = solution
     
    O_arr = np.array([O_x, O_y, O_z])

    # Compute intersection points
    P_intersection_arr = O_arr + taup_arr * lambda_p
    N_intersection_arr = O_arr + taun_arr * lambda_n
    
    # Convert back to TVector3
    P_intersection = ROOT.TVector3(*P_intersection_arr)
    N_intersection = ROOT.TVector3(*N_intersection_arr)

    d_min_pred = FindDMin(N_intersection, nn_dir.Unit(), P_intersection, np_dir.Unit())

    return P_intersection, N_intersection, d_min_pred, ROOT.TVector3(O_x, O_y, O_z)
  
def find_O_and_lambdas_from_intersections(taup, taun, P_intersection, N_intersection, O_y=None):
    """
    Given two intersection points, computes the origin (O) and the lambda scalars (lambda_p and lambda_n).

    Parameters:
        taup (TVector3): Direction vector for the first trajectory.
        taun (TVector3): Direction vector for the second trajectory.
        P_intersection (TVector3): Known intersection point for the first trajectory.
        N_intersection (TVector3): Known intersection point for the second trajectory.

    Returns:
        tuple: Origin (O as TVector3), lambda_p, and lambda_n.
    """
    # Convert TVector3 inputs to NumPy arrays
    taup_arr = np.array([taup.X(), taup.Y(), taup.Z()])
    taun_arr = np.array([taun.X(), taun.Y(), taun.Z()])
    P_intersection_arr = np.array([P_intersection.X(), P_intersection.Y(), P_intersection.Z()])
    N_intersection_arr = np.array([N_intersection.X(), N_intersection.Y(), N_intersection.Z()])
    
    # System of equations Ax = b
    # O_x + lambda_p * taup_x = P_intersection_x
    # O_y + lambda_p * taup_y = P_intersection_y
    # O_z + lambda_p * taup_z = P_intersection_z
    # O_x + lambda_n * taun_x = N_intersection_x
    # O_y + lambda_n * taun_y = N_intersection_y
    # O_z + lambda_n * taun_z = N_intersection_z
    
    if O_y is None:
        # O_y determined as well
        # this won't work if taus are perfectly back-to-back
        A = np.array([
            [1, 0, 0, taup_arr[0], 0],
            [0, 1, 0, taup_arr[1], 0],
            [0, 0, 1, taup_arr[2], 0],
            [1, 0, 0, 0, taun_arr[0]],
            [0, 1, 0, 0, taun_arr[1]],
            [0, 0, 1, 0, taun_arr[2]],
        ])

        b = np.concatenate((P_intersection_arr, N_intersection_arr))

        solution = np.linalg.lstsq(A, b, rcond=None)[0]
        O_x, O_y, O_z, lambda_p, lambda_n = solution

    else:
        # O_y is not determined
        A = np.array([
            [1, 0, taup_arr[0], 0],
            [0, 0, taup_arr[1], 0],
            [0, 1, taup_arr[2], 0],
            [1, 0, 0, taun_arr[0]],
            [0, 0, 0, taun_arr[1]],
            [0, 1, 0, taun_arr[2]],
        ])

        b = np.array([P_intersection_arr[0],
                     P_intersection_arr[1] - O_y,
                     P_intersection_arr[2],
                     N_intersection_arr[0],
                     N_intersection_arr[1] - O_y,
                     N_intersection_arr[2]])

        solution = np.linalg.lstsq(A, b, rcond=None)[0]
        O_x, O_z, lambda_p, lambda_n = solution
    
    # Create the origin as a TVector3
    O = ROOT.TVector3(O_x, O_y, O_z)
    
    return O, lambda_p, lambda_n

def tau_decay_probability(p, E, L):
    c = 3.0e8  # Speed of light in m/s
    tau = 2.9e-13  # Tau lifetime in seconds

    beta = p/E
    if beta >= 1:
        #print('Warning: Beta is >= 1, invalid for physical particles (Beta = %g). Recomputing using the expected tau mass.' % beta)
        beta = p/(p**2 + 1.777**2)**0.5 # if beta is unphysical then recalculate it using the expected mass of the tau lepton
        #print('recalculated beta:', beta)
    gamma = 1. / (1. - beta**2)**0.5
    
    lambda_decay = gamma * beta * c * tau  # Mean decay length in meters
    #lambda_decay = 45./1.777*c* tau # assume tau energy ~ 1/2 mZ for computing gamma - need to do this otherwise algo prefers to raise tau energy to make any decay length consistent with the tau lifetime
    L_m = L/1000. # convert from mm to meters
    
    return np.exp(-L_m / lambda_decay)  # Survival probability

def closest_distance(P1, P2, P):
    v = P2 - P1  # Direction vector of the line
    w = P - P1   # Vector from P1 to the point P
    
    t = w.Dot(v) / v.Mag2()  # Projection factor

    P_closest = P1 + t * v  # Closest point on the line

    return (P - P_closest).Mag()  # Distance between P and P_closest  

class Smearing():
    """
    Class to smear the energy and angular resolution of tracks.
    """

    def __init__(self):
        self.E_smearing = ROOT.TF1("E_smearing","TMath::Gaus(x,1,[0])",0,2)
        self.Angular_smearing = ROOT.TF1("Angular_smearing","TMath::Gaus(x,0,0.001)",-1,1) # approximate guess for now (this number was quoted for electromagnetic objects but is probably better for tracks)
        self.IP_z_smearing = ROOT.TF1("IP_z_smearing","TMath::Gaus(x,0,0.042)",-1,1) # number from David's thesis, note unit is mm
        self.IP_xy_smearing = ROOT.TF1("IP_xy_smearing","TMath::Gaus(x,0,0.023)",-1,1) # number from David's thesis (number for r * sqrt(1/2)), note unit is mm

        #assume same numbers as for IPs for now - need to find proper values
        #https://arxiv.org/pdf/hep-ex/0009058 states that the resolution for a B0 decay length is 200microns which is the diff between 2 vertices
        #so we can assume that the SV resolution is 200/sqrt(2)microns ~ 140 micros
        # this seems quite large so wpould be good to check it...
        self.SV_xyz_smearing = ROOT.TF1("SV_xyz_smearing","TMath::Gaus(x,0,0.14)",-1,1)

        self.POINT_z_smearing = ROOT.TF1("POINT_z_smearing","TMath::Gaus(x,0,0.0381)",-1,1)
        self.POINT_xy_smearing = ROOT.TF1("POINT_xy_smearing","TMath::Gaus(x,0,0.0381)",-1,1) 
        #self.POINT_z_smearing = ROOT.TF1("POINT_z_smearing","TMath::Gaus(x,0,0.0042/1.41)",-1,1)
        #self.POINT_xy_smearing = ROOT.TF1("POINT_xy_smearing","TMath::Gaus(x,0,0.0023/1.41)",-1,1) 
        #TODO: need to understand how to translate IP smearing into these numbers properly...

        # pi0 numbers taken from here: https://cds.cern.ch/record/272484/files/ppe-94-170.pdf
        self.Pi0_Angular_smearing = ROOT.TF1("Angular_smearing","TMath::Gaus(x,0,4*(2.5/sqrt([0])+0.25)/1000.)",-1,1) # approximate guess for now, took the numbers from the paper for ECAL showers and conservatively scaled by 10
        self.Pi0_E_smearing = ROOT.TF1("Pi0_E_smearing","TMath::Gaus(x,1,0.065)",0,2) # approximate guess for now

        self.Ox_smearing = ROOT.TF1("Ox_smearing","TMath::Gaus(x,0,0.15)",-1.5,1.5) # from https://cds.cern.ch/record/317914/files/sl-96-074.pdf
        self.Oy_smearing = ROOT.TF1("Oy_smearing","TMath::Gaus(x,0,0.011)",-0.11,0.11) # from https://cds.cern.ch/record/317914/files/sl-96-074.pdf
        self.Oz_smearing = ROOT.TF1("Oz_smearing","TMath::Gaus(x,0,7)",-70,70) # from https://cds.cern.ch/record/317914/files/sl-96-074.pdf

        self.Q_E_smearing = ROOT.TF1("Q_E_smearing","TMath::Gaus(x,1,0.18/sqrt([0])+0.009)",0,2) # energy resolution of detected photons
        self.Q_Angular_smearing = ROOT.TF1("Q_Angular_smearing","TMath::Gaus(x,0,(2.5/sqrt([0])+0.25)/1000.)",-1,1) # approximate guess for now (this number was quoted for electromagnetic objects but is probably better for tracks)

    def SmearPi0(self,pi0):
        if pi0 is None: return None
        E = pi0.E()
        self.Pi0_Angular_smearing.SetParameter(0,E)
        rand_E = self.Pi0_E_smearing.GetRandom()
        rand_dphi = self.Pi0_Angular_smearing.GetRandom()
        rand_dtheta = self.Pi0_Angular_smearing.GetRandom()

        pi0_smeared = ROOT.TLorentzVector(pi0)
        phi = pi0_smeared.Phi()
        new_phi = ROOT.TVector2.Phi_mpi_pi(phi + rand_dphi)

        theta = pi0_smeared.Theta()
        new_theta = theta + rand_dtheta

        pi0_smeared.SetPhi(new_phi)
        pi0_smeared.SetTheta(new_theta)
        pi0_smeared *= rand_E

        return pi0_smeared


    def SmearTrack(self,track):
        if track is None: return None
        E_res = 0.0006*track.P() # use p dependent resolution from David's thesis
        self.E_smearing.SetParameter(0,E_res)
        rand_E = self.E_smearing.GetRandom()
        rand_dphi = self.Angular_smearing.GetRandom()
        rand_dtheta = self.Angular_smearing.GetRandom()

        track_smeared = ROOT.TLorentzVector(track)

        phi = track_smeared.Phi()
        new_phi = ROOT.TVector2.Phi_mpi_pi(phi + rand_dphi)

        theta = track_smeared.Theta()
        new_theta = theta + rand_dtheta

        track_smeared.SetPhi(new_phi)
        track_smeared.SetTheta(new_theta)
        track_smeared *= rand_E

        return track_smeared

    def SmearDmin(self, dmin):
        if dmin is None: return None
        dmin_smeared = ROOT.TVector3(dmin)

        rand_z = self.IP_z_smearing.GetRandom()
        rand_x = self.IP_xy_smearing.GetRandom()
        rand_y = self.IP_xy_smearing.GetRandom()

        dmin_smeared.SetX(dmin_smeared.X() + rand_x)
        dmin_smeared.SetY(dmin_smeared.Y() + rand_y)
        dmin_smeared.SetZ(dmin_smeared.Z() + rand_z)

        return dmin_smeared
    
    def SmearPoint(self, point):
        if point is None: return None
        point_smeared = ROOT.TVector3(point)
        rand_x = self.POINT_xy_smearing.GetRandom()
        rand_y = self.POINT_xy_smearing.GetRandom()
        rand_z = self.POINT_z_smearing.GetRandom()

        point_smeared.SetX(point_smeared.X() + rand_x)
        point_smeared.SetY(point_smeared.Y() + rand_y)
        point_smeared.SetZ(point_smeared.Z() + rand_z)

        return point_smeared

    def SmearSV(self, SV):
        if SV is None: return None
        SV_smeared = ROOT.TVector3(SV)
        rand_x = self.SV_xyz_smearing.GetRandom()
        rand_y = self.SV_xyz_smearing.GetRandom()
        rand_z = self.SV_xyz_smearing.GetRandom()

        SV_smeared.SetX(SV_smeared.X() + rand_x)
        SV_smeared.SetY(SV_smeared.Y() + rand_y)
        SV_smeared.SetZ(SV_smeared.Z() + rand_z)

        return SV_smeared

    def SmearQ(self, Q):

        if Q is None: return None

        beam = ROOT.TLorentzVector(0.,0.,0.,91.18800354003906)
        P_photons = beam - Q

        E = P_photons.E()

        self.Q_Angular_smearing.SetParameter(0,E)
        self.Q_E_smearing.SetParameter(0,E)
        if E>0: 
            rand_E = self.Q_E_smearing.GetRandom()
            rand_dphi = self.Q_Angular_smearing.GetRandom()
            rand_dtheta = self.Q_Angular_smearing.GetRandom()
        else: 
            rand_E = 0
            rand_dphi = 0
            rand_dtheta = 0

        P_photons_smeared = ROOT.TLorentzVector(P_photons)
        phi = P_photons.Phi()
        new_phi = ROOT.TVector2.Phi_mpi_pi(phi + rand_dphi)

        theta = P_photons.Theta()
        new_theta = theta + rand_dtheta

        P_photons_smeared.SetPhi(new_phi)
        P_photons_smeared.SetTheta(new_theta)
        P_photons_smeared *= rand_E

        # to account for undetected photons we set the photon energy to 0 if it is below 0.5 GeV, the 0.5 GeV is a hopefully conservative guess for now
        if E < 0.5: P_photons_smeared*=0

        Q_smeared = beam - P_photons_smeared

        return Q_smeared

    def SmearBS(self, BS):
        if BS is None: return None
        # smear the beamspot position
        BS_smeared = ROOT.TVector3(BS)
        rand_x = self.Ox_smearing.GetRandom()
        rand_y = self.Oy_smearing.GetRandom()
        rand_z = self.Oz_smearing.GetRandom()
        BS_smeared.SetX(BS.X() + rand_x)
        BS_smeared.SetY(BS.Y() + rand_y)
        BS_smeared.SetZ(BS.Z() + rand_z)

        return BS_smeared        

if __name__ == '__main__':

    smearing = Smearing()
    apply_smearing = True
    #f = ROOT.TFile('pythia_output_ee_To_pipinunu_no_entanglementMG.root')
    #f = ROOT.TFile('pythia_output_rhorhoMG.root')
    #f = ROOT.TFile('pythia_output_pipiMG.root')
    f = ROOT.TFile('pythia_output_a1a1MG.root')
    tree = f.Get('tree')
    count_total = 0
    count_correct = 0
    d_min_reco_ave = 0.
    mean_dsol = 0.

    d_min_tracks_constraint_correct = 0
    d_min_vis_constraint_correct = 0
    d_min_solution_reco_constraint_correct_1 = 0
    d_min_solution_reco_constraint_correct_2 = 0
    d_min_solution_reco_constraint_correct = 0 
    signs_correct = 0
     

    print('starting...')
    for i in range(1,1000):
        count_total+=1
        tree.GetEntry(i)
        P_taup_true = ROOT.TLorentzVector(tree.taup_px, tree.taup_py, tree.taup_pz, tree.taup_e) 
        P_taun_true = ROOT.TLorentzVector(tree.taun_px, tree.taun_py, tree.taun_pz, tree.taun_e) 

        P_taup_pi1 = ROOT.TLorentzVector(tree.taup_pi1_px, tree.taup_pi1_py, tree.taup_pi1_pz, tree.taup_pi1_e)
        P_taun_pi1 = ROOT.TLorentzVector(tree.taun_pi1_px, tree.taun_pi1_py, tree.taun_pi1_pz, tree.taun_pi1_e)

        P_taup_pi2 = ROOT.TLorentzVector(tree.taup_pi2_px, tree.taup_pi2_py, tree.taup_pi2_pz, tree.taup_pi2_e)
        P_taun_pi2 = ROOT.TLorentzVector(tree.taun_pi2_px, tree.taun_pi2_py, tree.taun_pi2_pz, tree.taun_pi2_e)

        P_taup_pi3 = ROOT.TLorentzVector(tree.taup_pi3_px, tree.taup_pi3_py, tree.taup_pi3_pz, tree.taup_pi3_e)
        P_taun_pi3 = ROOT.TLorentzVector(tree.taun_pi3_px, tree.taun_pi3_py, tree.taun_pi3_pz, tree.taun_pi3_e)

        P_taup_pizero1 = ROOT.TLorentzVector(tree.taup_pizero1_px, tree.taup_pizero1_py, tree.taup_pizero1_pz, tree.taup_pizero1_e)
        P_taun_pizero1 = ROOT.TLorentzVector(tree.taun_pizero1_px, tree.taun_pizero1_py, tree.taun_pizero1_pz, tree.taun_pizero1_e)

        #P_taup_nu = ROOT.TLorentzVector(tree.taup_nu_px, tree.taup_nu_py, tree.taup_nu_pz, tree.taup_nu_e)
        #P_taun_nu = ROOT.TLorentzVector(tree.taun_nu_px, tree.taun_nu_py, tree.taun_nu_pz, tree.taun_nu_e)

        P_taupvis = P_taup_pi1.Clone()
        P_tauptracks = P_taup_pi1.Clone()
        if tree.taup_pizero1_e > 0:
            P_taupvis += P_taup_pizero1
        if tree.taup_pi2_e > 0:
            P_taupvis += P_taup_pi2
            P_tauptracks += P_taup_pi2
        if tree.taup_pi3_e > 0:
            P_taupvis += P_taup_pi3
            P_tauptracks += P_taup_pi3

        P_taunvis = P_taun_pi1.Clone()
        P_tauntracks = P_taun_pi1.Clone()
        if tree.taun_pizero1_e > 0:
            P_taunvis += P_taun_pizero1
        if tree.taun_pi2_e > 0:
            P_taunvis += P_taun_pi2
            P_tauntracks += P_taun_pi2
        if tree.taun_pi3_e > 0:
            P_taunvis += P_taun_pi3
            P_tauntracks += P_taun_pi3
    
        P_Z = P_taup_true+P_taun_true
        E_Z = P_Z.E()
        #P_Z = ROOT.TLorentzVector(0.,0.,0.,91.188) # assuming we don't know ISR and have to assume momentum is balanced
  
        # compute IPs from SVs

        # note that the below assuems that taus are produced at 0,0,0 which might not be true for some MC samples! 
        VERTEX_taup = ROOT.TVector3(tree.taup_pi1_vx, tree.taup_pi1_vy, tree.taup_pi1_vz) # in mm
        VERTEX_taun = ROOT.TVector3(tree.taun_pi1_vx, tree.taun_pi1_vy, tree.taun_pi1_vz) # in mm

        #VERTEX_taup += ROOT.TVector3(10.,5.,-7)
        #VERTEX_taun += ROOT.TVector3(10.,5,-7)

        l_true = abs((VERTEX_taup-VERTEX_taun).Mag())
        d_true = VERTEX_taup-VERTEX_taun


        rand1 = random.uniform(-100,100)
        rand2 = random.uniform(-100,100)
        VERTEX_taun_rand = VERTEX_taun + P_taun_pi1.Vect().Unit()*rand1
        VERTEX_taup_rand = VERTEX_taup + P_taup_pi1.Vect().Unit()*rand2
        d_min_tracks_reco, taun_pca, taup_pca = FindDMin(VERTEX_taun_rand, P_tauntracks.Vect().Unit(), VERTEX_taup_rand, P_tauptracks.Vect().Unit(), return_points=True)
        d_min_vis_reco = FindDMin(taun_pca, P_taunvis.Vect().Unit(), taup_pca, P_taupvis.Vect().Unit())
        d_min_gen_reco = FindDMin(VERTEX_taun, P_taunvis.Vect().Unit(), VERTEX_taup, P_taupvis.Vect().Unit())


        BS = ROOT.TVector3(0.,0.,0.)
        VERTEX_taun_reco = VERTEX_taun.Clone()
        VERTEX_taup_reco = VERTEX_taup.Clone()
        if apply_smearing: 
            BS = smearing.SmearBS(BS)
            #TODO apply propper SV vertex smearing here instead of point based numbers!
            VERTEX_taun_reco = smearing.SmearSV(VERTEX_taun_reco)
            VERTEX_taup_reco = smearing.SmearSV(VERTEX_taup_reco)
            P_Z = smearing.SmearQ(P_Z)

        d_min_sv_reco = FindDMin(VERTEX_taun_reco, P_tauntracks.Vect().Unit(), VERTEX_taup_reco, P_tauptracks.Vect().Unit())

        # if 3-prongs then define sv_delta, otherwise set it to None
        sv_delta = None
        if tree.taup_pi2_e > 0 and tree.taup_pi3_e > 0 and tree.taun_pi2_e > 0 and tree.taun_pi3_e > 0:
            sv_delta = VERTEX_taup_reco - VERTEX_taun_reco

        if apply_smearing:
            # TODO: update smearing for case of pi + pi0 decays!
            P_taupvis = smearing.SmearTrack(P_taupvis)
            P_taunvis = smearing.SmearTrack(P_taunvis)

            P_taup_pi1 = smearing.SmearTrack(P_taup_pi1)
            P_taun_pi1 = smearing.SmearTrack(P_taun_pi1)

            P_taup_pi2 = smearing.SmearTrack(P_taup_pi2)
            P_taun_pi2 = smearing.SmearTrack(P_taun_pi2)

            P_taup_pi3 = smearing.SmearTrack(P_taup_pi3)
            P_taun_pi3 = smearing.SmearTrack(P_taun_pi3)            

            d_min_tracks_reco = smearing.SmearDmin(d_min_tracks_reco)

        d_min_reco = d_min_tracks_reco
        # if 3-prongs then take the sv version 
        if tree.taup_pi2_e > 0 and tree.taup_pi3_e > 0 and tree.taun_pi2_e > 0 and tree.taun_pi3_e > 0:  
            d_min_reco = d_min_sv_reco 
            mode = 3
        elif tree.taup_pizero1_e > 0 and tree.taun_pizero1_e > 0:
            mode = 2
        else:
            mode =1


        reconstructor = TauReconstructor()
        solutions = reconstructor.reconstruct_tau_alt(P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, O_y=BS.Y(), np_point=VERTEX_taup_rand, nn_point=VERTEX_taun_rand, d_min_reco=d_min_reco, sv_delta=sv_delta, mode=mode, no_minimisation=True)#d_min_vis_reco
        #numerical_solutions = reconstructor.reconstruct_tau_alt(P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, O_y=BS.Y(), np_point=VERTEX_taup_rand, nn_point=VERTEX_taun_rand, d_min_reco=d_min_vis_reco, no_minimisation=False)
        #P_taup_reco, P_taun_reco, d_min_numeric, decay_length_prob = numerical_solutions[0]
        #P_taup_reco_alt, P_taun_reco_alt, d_min_numeric_alt, decay_length_prob_alt = numerical_solutions[1]

        sv_delta_1 = (sv_delta.Unit().Dot(solutions[0][0].Vect().Unit()) - sv_delta.Unit().Dot(solutions[0][1].Vect().Unit()))/2 if sv_delta is not None else 0
        sv_delta_2 = (sv_delta.Unit().Dot(solutions[1][0].Vect().Unit()) - sv_delta.Unit().Dot(solutions[1][1].Vect().Unit()))/2 if sv_delta is not None else 0

        #temp: sort taus here based on sv_delta_i, preferring larger values
        #if sv_delta_1 < sv_delta_2:
        #    solutions = [solutions[1],solutions[0]]
        #    sv_delta_1, sv_delta_2 = sv_delta_2, sv_delta_1
        #np.random.shuffle(solutions)

        taup_sv_check_1 = VERTEX_taup.Unit().Dot(solutions[0][0].Vect().Unit())
        taun_sv_check_1 = VERTEX_taun.Unit().Dot(solutions[0][1].Vect().Unit())
        taup_sv_check_2 = VERTEX_taup.Unit().Dot(solutions[1][0].Vect().Unit())
        taun_sv_check_2 = VERTEX_taun.Unit().Dot(solutions[1][1].Vect().Unit())

        ##sort solutions based on the dot product of the SV direction with the true SV direction
        #if taup_sv_check_1*taun_sv_check_1 < taup_sv_check_2*taun_sv_check_2:
        #    solutions = [solutions[1],solutions[0]] # TODO put this in function earlier on but note can;t do this exactly as don't know the PV yet - need to solve for it!!!


        O_1, lambda_p_1, lambda_n_1 = find_O_and_lambdas_from_intersections(solutions[0][0], solutions[0][1], VERTEX_taup_reco, VERTEX_taun_reco)
        O_2, lambda_p_2, lambda_n_2 = find_O_and_lambdas_from_intersections(solutions[1][0], solutions[1][1], VERTEX_taup_reco, VERTEX_taun_reco)

        O_1_alt, lambda_p_1_alt, lambda_n_1_alt = find_O_and_lambdas_from_intersections(solutions[0][0], solutions[0][1], VERTEX_taup_reco, VERTEX_taun_reco, BS.Y())
        O_2_alt, lambda_p_2_alt, lambda_n_2_alt = find_O_and_lambdas_from_intersections(solutions[1][0], solutions[1][1], VERTEX_taup_reco, VERTEX_taun_reco, BS.Y())

        #print('!!!!!!!')
        #print('O_1:', O_1.X(), O_1.Y(), O_1.Z())
        #print('O_2:', O_2.X(), O_2.Y(), O_2.Z())
        #print('lambda_p_1:', lambda_p_1)
        #print('lambda_n_1:', lambda_n_1)
        #print('lambda_p_2:', lambda_p_2)
        #print('lambda_n_2:', lambda_n_2)
        #print('O_1_alt:', O_1_alt.X(), O_1_alt.Y(), O_1_alt.Z())
        #print('O_2_alt:', O_2_alt.X(), O_2_alt.Y(), O_2_alt.Z())
        #print('lambda_p_1_alt:', lambda_p_1_alt)
        #print('lambda_n_1_alt:', lambda_n_1_alt)
        #print('lambda_p_2_alt:', lambda_p_2_alt)
        #print('lambda_n_2_alt:', lambda_n_2_alt)


        #print(solutions[0][0].Vect().Unit().Cross(solutions[0][1].Vect().Unit()).Mag())
        #print(solutions[1][0].Vect().Unit().Cross(solutions[1][1].Vect().Unit()).Mag())
        #use alternative O and lambdas in cases where taus are parallel
        if solutions[0][0].Vect().Unit().Cross(solutions[0][1].Vect().Unit()).Mag() == 0:
            O_1 = O_1_alt
            lambda_p_1 = lambda_p_1_alt
            lambda_n_1 = lambda_n_1_alt
        if solutions[1][0].Vect().Unit().Cross(solutions[1][1].Vect().Unit()).Mag() == 0:
            O_2 = O_2_alt
            lambda_p_2 = lambda_p_2_alt
            lambda_n_2 = lambda_n_2_alt

        #print('----')
        #print('O_1:', O_1.X(), O_1.Y(), O_1.Z())
        #print('O_2:', O_2.X(), O_2.Y(), O_2.Z())
        #print('lambda_p_1:', lambda_p_1)
        #print('lambda_n_1:', lambda_n_1)
        #print('lambda_p_2:', lambda_p_2)
        #print('lambda_n_2:', lambda_n_2)
        #print('O_1_alt:', O_1_alt.X(), O_1_alt.Y(), O_1_alt.Z())
        #print('O_2_alt:', O_2_alt.X(), O_2_alt.Y(), O_2_alt.Z())
        #print('lambda_p_1_alt:', lambda_p_1_alt)
        #print('lambda_n_1_alt:', lambda_n_1_alt)
        #print('lambda_p_2_alt:', lambda_p_2_alt)
        #print('lambda_n_2_alt:', lambda_n_2_alt)
        
        d_sol1 = compare_lorentz_pairs((P_taup_true,P_taun_true),solutions[0])
        d_sol2 = compare_lorentz_pairs((P_taup_true,P_taun_true),solutions[1])

        FoundCorrectSolution = (d_sol1<=d_sol2)
        if FoundCorrectSolution: count_correct+=1   

        mean_dsol += d_sol1 

        P_intersection_1, N_intersection_1, _, O_1  = find_intersections_fixed_Oy(solutions[0][0].Vect().Unit(), solutions[0][1].Vect().Unit(), VERTEX_taup_rand, P_taup_pi1.Vect().Unit(), VERTEX_taun_rand, P_taun_pi1.Vect().Unit(),BS.Y())
        P_intersection_2, N_intersection_2, _, O_2  = find_intersections_fixed_Oy(solutions[1][0].Vect().Unit(), solutions[1][1].Vect().Unit(), VERTEX_taup_rand, P_taup_pi1.Vect().Unit(), VERTEX_taun_rand, P_taun_pi1.Vect().Unit(),BS.Y())
        taup_prob_1 = tau_decay_probability(solutions[0][0].P(), solutions[0][0].E(), (P_intersection_1-O_1).Mag())
        taun_prob_1 = tau_decay_probability(solutions[0][1].P(), solutions[0][1].E(), (N_intersection_1-O_1).Mag())
        flight_length_prob_1 = taup_prob_1*taun_prob_1
        taup_prob_2 = tau_decay_probability(solutions[1][0].P(), solutions[1][0].E(), (P_intersection_2-O_2).Mag())
        taun_prob_2 = tau_decay_probability(solutions[1][1].P(), solutions[1][1].E(), (N_intersection_2-O_2).Mag())
        flight_length_prob_2 = taup_prob_2*taun_prob_2

        dOx_1 = abs(O_1.X()-BS.X())
        dOx_2 = abs(O_2.Y()-BS.Y())

        d_min_solution_reco_1 = FindDMin(N_intersection_1, P_taunvis.Vect().Unit(), P_intersection_1, P_taupvis.Vect().Unit())
        d_min_tracks_solution_reco_1 = FindDMin(N_intersection_1, P_taun_pi1.Vect().Unit(), P_intersection_1, P_taup_pi1.Vect().Unit())
        d_min_pred_1, d_min_tracks_pred_1 = GetDMins(solutions[0][0], solutions[0][1], P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1)

        taup_pi0 = P_taup_pizero1.Clone()
        taun_pi0 = P_taun_pizero1.Clone()

        _, d_min_tracks_pred_1_new =  GetDMins(solutions[0][0]-taup_pi0, solutions[0][1]+taun_pi0, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1)
        _, d_min_tracks_pred_2_new =  GetDMins(solutions[1][0]-taup_pi0, solutions[1][1]+taun_pi0, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1)

        d_min_gen_constraint_1 = d_min_pred_1.Unit().Dot(d_min_gen_reco.Unit())
        d_min_tracks_constraint_1 = d_min_tracks_pred_1.Unit().Dot(d_min_tracks_reco.Unit())
        d_min_tracks_vs_vis_constraint_1 = np.sign(d_min_pred_1.Unit().Dot(d_min_tracks_reco.Unit()))
        d_min_vis_constraint_1 = d_min_pred_1.Unit().Dot(d_min_vis_reco.Unit())
        d_min_solution_reco_constraint_1 = d_min_pred_1.Unit().Dot(d_min_solution_reco_1.Unit())
        d_min_sv_reco_constraint_1 = d_min_pred_1.Unit().Dot(d_min_sv_reco.Unit())

        #d_d_min = abs(d_min_solution_reco_constraint_1.Mag()-d_min_pred_1.Mag())

        d_min_solution_reco_2 = FindDMin(N_intersection_2, P_taunvis.Vect().Unit(), P_intersection_2, P_taupvis.Vect().Unit())
        d_min_tracks_solution_reco_2 = FindDMin(N_intersection_2, P_taun_pi1.Vect().Unit(), P_intersection_2, P_taup_pi1.Vect().Unit())
        d_min_pred_2, d_min_tracks_pred_2 = GetDMins(solutions[1][0], solutions[1][1], P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1)
        d_min_gen_constraint_2 = d_min_pred_2.Unit().Dot(d_min_gen_reco.Unit())
        d_min_tracks_constraint_2 = d_min_tracks_pred_2.Unit().Dot(d_min_tracks_reco.Unit())
        d_min_tracks_vs_vis_constraint_2 = np.sign(d_min_pred_2.Unit().Dot(d_min_tracks_reco.Unit()))
        d_min_vis_constraint_2 = d_min_pred_2.Unit().Dot(d_min_vis_reco.Unit())
        d_min_solution_reco_constraint_2 = d_min_pred_2.Unit().Dot(d_min_solution_reco_2.Unit())
        d_min_sv_reco_constraint_2 = d_min_pred_2.Unit().Dot(d_min_sv_reco.Unit())


        # check if tau direction is consistent with SV-PV direction
        taup_sign_1 = (P_intersection_1-O_1).Unit().Dot(solutions[0][0].Vect().Unit())
        taun_sign_1 = (N_intersection_1-O_1).Unit().Dot(solutions[0][1].Vect().Unit())
        taup_sign_2 = (P_intersection_2-O_2).Unit().Dot(solutions[1][0].Vect().Unit())
        taun_sign_2 = (N_intersection_2-O_2).Unit().Dot(solutions[1][1].Vect().Unit())


        d_1 = PredictD(solutions[0][1], solutions[0][0], P_taunvis, P_taupvis, d_min_tracks_reco)
        d_2 = PredictD(solutions[1][1], solutions[1][0], P_taunvis, P_taupvis, d_min_tracks_reco)

        d_gen = VERTEX_taup-VERTEX_taun

        correct_signp = taup_sign_1
        correct_signn = taun_sign_1
        incorrect_signp = taup_sign_2
        incorrect_signn = taun_sign_2
        if not FoundCorrectSolution:
            correct_signp = taup_sign_2
            correct_signn = taun_sign_2
            incorrect_signp = taup_sign_1
            incorrect_signn = taun_sign_1

        if ((correct_signp>0) and (correct_signn>0)) and not (incorrect_signp>0 and incorrect_signn>0): signs_correct+=1 


        #bool_d_min_solution_reco_constraint_correct_1 = (np.sign(d_min_gen_constraint*d_min_solution_reco_constraint)>0)
        #if np.sign(d_min_gen_constraint*d_min_tracks_constraint)>0:  d_min_tracks_constraint_correct += 1
        #if np.sign(d_min_gen_constraint*d_min_vis_constraint)>0:  d_min_vis_constraint_correct += 1
        #if bool_d_min_solution_reco_constraint_correct_1:  d_min_solution_reco_constraint_correct_1 += 1

        if i < 100: # only print a few events

            print('\n---------------------------------------')
            print('Event %i' %i)
            print('\nTrue taus:\n')
            print('tau+:', P_taup_true.X(), P_taup_true.Y(), P_taup_true.Z(), P_taup_true.T(), P_taup_true.M())
            print('tau-:', P_taun_true.X(), P_taun_true.Y(), P_taun_true.Z(), P_taun_true.T(), P_taun_true.M())
    
            #print('\nReco taus (numerically):')
            #print('\nsolution 1:')
            #print('tau+:', numerical_solutions[0][0].X(), numerical_solutions[0][0].Y(), numerical_solutions[0][0].Z(), numerical_solutions[0][0].T(), numerical_solutions[0][0].M())
            #print('tau-:', numerical_solutions[0][1].X(), numerical_solutions[0][1].Y(), numerical_solutions[0][1].Z(), numerical_solutions[0][1].T(), numerical_solutions[0][1].M())

            #print('\nsolution 2:')
            #print('tau+:', numerical_solutions[1][0].X(), numerical_solutions[1][0].Y(), numerical_solutions[1][0].Z(), numerical_solutions[1][0].T(), numerical_solutions[1][0].M())
            #print('tau-:', numerical_solutions[1][1].X(), numerical_solutions[1][1].Y(), numerical_solutions[1][1].Z(), numerical_solutions[1][1].T(), numerical_solutions[1][1].M())

            print('\nReco taus (analytically):')
            print('\nsolution 1:')
            print('tau+:', solutions[0][0].X(), solutions[0][0].Y(), solutions[0][0].Z(), solutions[0][0].T(), solutions[0][0].M())
            print('tau-:', solutions[0][1].X(), solutions[0][1].Y(), solutions[0][1].Z(), solutions[0][1].T(), solutions[0][1].M())
            #print('flight length prob:', flight_length_prob_1)
            #print('O:', O_1.X(), O_1.Y(), O_1.Z())

            #print('d_min_gen_reco:', d_min_gen_reco.Unit().X(), d_min_gen_reco.Unit().Y(), d_min_gen_reco.Unit().Z(), d_min_gen_reco.Mag())
            print('d_min_sv_reco:', d_min_sv_reco.Unit().X(), d_min_sv_reco.Unit().Y(), d_min_sv_reco.Unit().Z(), d_min_sv_reco.Mag())
            print('d_min_tracks_reco:',d_min_tracks_reco.Unit().X(), d_min_tracks_reco.Unit().Y(), d_min_tracks_reco.Unit().Z(), d_min_tracks_reco.Mag())
            #print('d_min_vis_reco:',  d_min_vis_reco.Unit().X(), d_min_vis_reco.Unit().Y(), d_min_vis_reco.Unit().Z(), d_min_vis_reco.Mag())
            #print('d_min_solutions_reco:', d_min_solution_reco_1.Unit().X(), d_min_solution_reco_1.Unit().Y(), d_min_solution_reco_1.Unit().Z(), d_min_solution_reco_1.Mag())
            #print('d_min_tracks_solution_reco:', d_min_tracks_solution_reco_1.Unit().X(), d_min_tracks_solution_reco_1.Unit().Y(), d_min_tracks_solution_reco_1.Unit().Z(), d_min_tracks_solution_reco_1.Mag())
            print('d_min_tracks_pred_new:',d_min_tracks_pred_1_new.Unit().X(), d_min_tracks_pred_1_new.Unit().Y(), d_min_tracks_pred_1_new.Unit().Z(), d_min_tracks_pred_1_new.Mag())

            print('d_min_pred:',d_min_pred_1.Unit().X(), d_min_pred_1.Unit().Y(), d_min_pred_1.Unit().Z(), d_min_pred_1.Mag())
            print('d_min_tracks_pred:',d_min_tracks_pred_1.Unit().X(), d_min_tracks_pred_1.Unit().Y(), d_min_tracks_pred_1.Unit().Z(), d_min_tracks_pred_1.Mag())

            #print('d_min_gen constraint:', d_min_gen_constraint_1)
            print('d_min_tracks constraint:', d_min_tracks_constraint_1)
            #print('d_min_tracks_vs_vis constraint:', d_min_tracks_vs_vis_constraint_1)
            #print('d_min_vis constraint:', d_min_vis_constraint_1)
            #print('d_min_solutions_reco constraint:', d_min_solution_reco_constraint_1)
            print('d_min_sv_reco_constraint:', d_min_sv_reco_constraint_1)
            #print('taup_sign:',taup_sign_1)
            #print('taun_sign:',taun_sign_1)
            #print('taup_sv_check:', taup_sv_check_1)
            #print('taun_sv_check:', taun_sv_check_1)

            print('sv_delta:', sv_delta_1)

            #print('d_1:',d_1.X(),d_1.Y(),d_1.Z(), d_1.Mag())
            #print('d_gen:',d_gen.X(),d_gen.Y(),d_gen.Z(), d_gen.Mag())
            #print('P-N', (P_intersection_1-N_intersection_1).Mag())



            print('\nsolution 2:')
            print('tau+:', solutions[1][0].X(), solutions[1][0].Y(), solutions[1][0].Z(), solutions[1][0].T(), solutions[1][0].M())
            print('tau-:', solutions[1][1].X(), solutions[1][1].Y(), solutions[1][1].Z(), solutions[1][1].T(), solutions[1][1].M())
            #print('flight length prob:', flight_length_prob_2)
            #print('O:', O_2.X(), O_2.Y(), O_2.Z())

            #print('d_min_gen_reco:', d_min_gen_reco.Unit().X(), d_min_gen_reco.Unit().Y(), d_min_gen_reco.Unit().Z(), d_min_gen_reco.Mag())
            print('d_min_sv_reco:', d_min_sv_reco.Unit().X(), d_min_sv_reco.Unit().Y(), d_min_sv_reco.Unit().Z(), d_min_sv_reco.Mag())
            print('d_min_tracks_reco:',d_min_tracks_reco.Unit().X(), d_min_tracks_reco.Unit().Y(), d_min_tracks_reco.Unit().Z(), d_min_tracks_reco.Mag())
            #print('d_min_vis_reco:',  d_min_vis_reco.Unit().X(), d_min_vis_reco.Unit().Y(), d_min_vis_reco.Unit().Z(), d_min_vis_reco.Mag())
            #print('d_min_solutions_reco:', d_min_solution_reco_2.Unit().X(), d_min_solution_reco_2.Unit().Y(), d_min_solution_reco_2.Unit().Z(), d_min_solution_reco_2.Mag()) 
            #print('d_min_tracks_solution_reco:', d_min_tracks_solution_reco_2.Unit().X(), d_min_tracks_solution_reco_2.Unit().Y(), d_min_tracks_solution_reco_2.Unit().Z(), d_min_tracks_solution_reco_2.Mag()) 

            print('d_min_pred:',d_min_pred_2.Unit().X(), d_min_pred_2.Unit().Y(), d_min_pred_2.Unit().Z(), d_min_pred_2.Mag())
            print('d_min_tracks_pred:',d_min_tracks_pred_2.Unit().X(), d_min_tracks_pred_2.Unit().Y(), d_min_tracks_pred_2.Unit().Z(), d_min_tracks_pred_2.Mag())  
            print('d_min_tracks_pred_new:',d_min_tracks_pred_2_new.Unit().X(), d_min_tracks_pred_2_new.Unit().Y(), d_min_tracks_pred_2_new.Unit().Z(), d_min_tracks_pred_2_new.Mag())      
            print('d_min_sv_reco_constraint:', d_min_sv_reco_constraint_2)

            #print('d_min_gen constraint:', d_min_gen_constraint_2)
            print('d_min_tracks constraint:', d_min_tracks_constraint_2)
            #print('d_min_tracks_vs_vis constraint:', d_min_tracks_vs_vis_constraint_2)
            #print('d_min_vis constraint:', d_min_vis_constraint_2)
            #print('d_min_solutions_reco constraint:', d_min_solution_reco_constraint_2)
            #print('taup_sign:',taup_sign_2)
            #print('taun_sign:',taun_sign_2)
            #print('taup_sv_check:', taup_sv_check_2)
            #print('taun_sv_check:', taun_sv_check_2)

            print('sv_delta:', sv_delta_2)

            #print('d_2:',d_2.X(),d_2.Y(),d_2.Z(), d_2.Mag())
            #print('d_gen:',d_gen.X(),d_gen.Y(),d_gen.Z(), d_gen.Mag())
            #print('P-N', (P_intersection_2-N_intersection_2).Mag())

            P_taup_pi1_new_1 = ChangeFrame(solutions[0][1]-P_taun_pizero1, P_taun_pi1, P_taup_pi1)
            P_taun_pi1_new_1 = ChangeFrame(solutions[0][1]-P_taun_pizero1, P_taun_pi1, P_taun_pi1)
            P_taup_pi1_new_2 = ChangeFrame(solutions[1][1]-P_taun_pizero1, P_taun_pi1, P_taup_pi1)
            P_taun_pi1_new_2 = ChangeFrame(solutions[1][1]-P_taun_pizero1, P_taun_pi1, P_taun_pi1)

            P_taupvis_new_1 = ChangeFrame(solutions[0][1], P_taunvis, P_taupvis)
            P_taunvis_new_1 = ChangeFrame(solutions[0][1], P_taunvis, P_taunvis)
            P_taupvis_new_2 = ChangeFrame(solutions[1][1], P_taunvis, P_taupvis)
            P_taunvis_new_2 = ChangeFrame(solutions[1][1], P_taunvis, P_taunvis)            


            #P_taup_pi1_new_1 = ChangeFrame(solutions[0][1], P_taun_pi1, P_taup_pi1)
            #P_taun_pi1_new_1 = ChangeFrame(solutions[0][1], P_taun_pi1, P_taun_pi1)
            #P_taup_pi1_new_2 = ChangeFrame(solutions[1][1], P_taun_pi1, P_taup_pi1)
            #P_taun_pi1_new_2 = ChangeFrame(solutions[1][1], P_taun_pi1, P_taun_pi1)

            print('P_taup_pi1_new_1:', P_taup_pi1_new_1.X(), P_taup_pi1_new_1.Y(), P_taup_pi1_new_1.Z(), P_taup_pi1_new_1.T(), P_taup_pi1_new_1.M())
            print('P_taun_pi1_new_1:', P_taun_pi1_new_1.X(), P_taun_pi1_new_1.Y(), P_taun_pi1_new_1.Z(), P_taun_pi1_new_1.T(), P_taun_pi1_new_1.M())
            print('P_taup_pi1_new_2:', P_taup_pi1_new_2.X(), P_taup_pi1_new_2.Y(), P_taup_pi1_new_2.Z(), P_taup_pi1_new_2.T(), P_taup_pi1_new_2.M())
            print('P_taun_pi1_new_2:', P_taun_pi1_new_2.X(), P_taun_pi1_new_2.Y(), P_taun_pi1_new_2.Z(), P_taun_pi1_new_2.T(), P_taun_pi1_new_2.M())

            print('P_taupvis_new_1:', P_taupvis_new_1.X(), P_taupvis_new_1.Y(), P_taupvis_new_1.Z(), P_taupvis_new_1.T(), P_taupvis_new_1.M())
            print('P_taunvis_new_1:', P_taunvis_new_1.X(), P_taunvis_new_1.Y(), P_taunvis_new_1.Z(), P_taunvis_new_1.T(), P_taunvis_new_1.M())
            print('P_taupvis_new_2:', P_taupvis_new_2.X(), P_taupvis_new_2.Y(), P_taupvis_new_2.Z(), P_taupvis_new_2.T(), P_taupvis_new_2.M())
            print('P_taunvis_new_2:', P_taunvis_new_2.X(), P_taunvis_new_2.Y(), P_taunvis_new_2.Z(), P_taunvis_new_2.T(), P_taunvis_new_2.M())



            sign_1 = np.sign(P_taupvis_new_1.Y())
            sign_2 = np.sign(P_taupvis_new_2.Y())

            print('sign_1:', sign_1)
            print('sign_2:', sign_2)

            # if signs are equal then we take largest value as the +ve sign:
            if sign_1 == sign_2:
                if P_taup_pi1_new_1.Y() > P_taup_pi1_new_2.Y():
                    sign_1 = +1
                    sign_2 = -1
                else:
                    sign_1 = -1
                    sign_2 = +1

            print('sign_alt_1:', sign_1)
            print('sign_alt_2:', sign_2)

            #sign_pred = np.sign(d_min_tracks_reco.Dot(P_taup_pi1.Vect().Unit().Cross(P_taun_pi1.Vect().Unit())))
            sign_pred = np.sign(d_min_tracks_reco.Dot(P_taupvis.Vect().Unit().Cross(P_taunvis.Vect().Unit())))
            print('sign_pred:', sign_pred)

            print('correct solution found?', FoundCorrectSolution)
            print('\n')

    mean_dsol/=count_total
    #mean_dsol = mean_dsol**.5

    d_min_tracks_constraint_correct = float(d_min_tracks_constraint_correct)/count_total
    d_min_vis_constraint_correct= float(d_min_vis_constraint_correct)/count_total
    d_min_solution_reco_constraint_correct_1 = float(d_min_solution_reco_constraint_correct_1)/count_total
    d_min_solution_reco_constraint_correct_2 = float(d_min_solution_reco_constraint_correct_2)/count_total
    d_min_solution_reco_constraint_correct = float(d_min_solution_reco_constraint_correct)/count_total

    signs_correct = float(signs_correct)/count_total

    print('d_min_tracks_constraint_correct:', d_min_tracks_constraint_correct)
    print('d_min_vis_constraint_correct:', d_min_vis_constraint_correct)
    print('d_min_solution_reco_constraint_correct_1:', d_min_solution_reco_constraint_correct_1)
    print('d_min_solution_reco_constraint_correct_2:', d_min_solution_reco_constraint_correct_2)
    print('d_min_solution_reco_constraint_correct:', d_min_solution_reco_constraint_correct)
    print('signs_correct:', signs_correct)


    print('Found correct solution for %i / %i events = %.1f%%' % (count_correct, count_total, count_correct/count_total*100))
    print('mean d_sol = %g' % mean_dsol)
