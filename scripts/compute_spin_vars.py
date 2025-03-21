import ROOT
import argparse
import math
from array import array
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
from PolarimetricA1 import PolarimetricA1
from ReconstructTaus import *
import random
import numpy as np

# each LEP detector should have about 140000 Z->tautau events (not accounting for acceptance effects)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Input ROOT file.")
parser.add_argument("-o", "--output", required=True, help="Output ROOT file.")
parser.add_argument('--n_events', '-n', help='Maximum number of events to process', default=-1, type=int)
parser.add_argument('--n_skip', '-s', help='skip n_events*n_skip', default=0, type=int)
parser.add_argument('--smear_mode', help='if mode=1 then smear the objects and use reco-like variables to reconstruct the taus, if mode =0 then don\'t smear the objects but still use the reco-like variables to obtain the tau', default=1, type=int)

args = parser.parse_args()

# Open the input ROOT file
input_root = ROOT.TFile(args.input, "READ")
if input_root.IsZombie():
    raise Exception(f"Unable to open file {args.input}")

# Access the tree in the input ROOT file
tree = input_root.Get("tree")  # Replace 'tree' with your tree name
if not tree:
    raise Exception("Tree not found in the input file.")
    input_root.Close()

# Create an output ROOT file
output_root = ROOT.TFile(args.output, "RECREATE")

# Create a new tree to store the output
new_tree = ROOT.TTree("new_tree","Event Tree")

branches = [
        'cosn_plus',
        'cosr_plus',
        'cosk_plus',
        'cosn_minus',
        'cosr_minus',
        'cosk_minus',
        'cosTheta',

        'cosn_plus_reco',
        'cosr_plus_reco',
        'cosk_plus_reco',
        'cosn_minus_reco',
        'cosr_minus_reco',
        'cosk_minus_reco',
        'cosTheta_reco',    

        'd_min_constraint',
        'mass',
        'reco_mass',

        'taun_npi',
        'taun_npizero',
        'taup_npi',
        'taup_npizero',

        'taup_vz',

        'reco_taup_vx',    
        'reco_taup_vy',    
        'reco_taup_vz', 
        'reco_taup_pi1_py',
        'reco_taup_pi1_px',    
        'reco_taup_pi1_pz',    
        'reco_taup_pi1_e',     
        'reco_taup_pi1_ipx',    
        'reco_taup_pi1_ipy',    
        'reco_taup_pi1_ipz',    
        'reco_taup_pi2_px',    
        'reco_taup_pi2_py',    
        'reco_taup_pi2_pz',    
        'reco_taup_pi2_e',        
        'reco_taup_pi3_px',    
        'reco_taup_pi3_py',    
        'reco_taup_pi3_pz',    
        'reco_taup_pi3_e',        
        'reco_taup_pizero1_px',
        'reco_taup_pizero1_py',
        'reco_taup_pizero1_pz',
        'reco_taup_pizero1_e', 
        'reco_taup_pizero2_px',
        'reco_taup_pizero2_py',
        'reco_taup_pizero2_pz',
        'reco_taup_pizero2_e',      
        'reco_taun_vx',    
        'reco_taun_vy',    
        'reco_taun_vz', 
        'reco_taun_pi1_px',    
        'reco_taun_pi1_py',    
        'reco_taun_pi1_pz',    
        'reco_taun_pi1_e',     
        'reco_taun_pi1_ipx',    
        'reco_taun_pi1_ipy',    
        'reco_taun_pi1_ipz',    
        'reco_taun_pi2_px',    
        'reco_taun_pi2_py',    
        'reco_taun_pi2_pz',    
        'reco_taun_pi2_e',        
        'reco_taun_pi3_px',    
        'reco_taun_pi3_py',    
        'reco_taun_pi3_pz',    
        'reco_taun_pi3_e',        
        'reco_taun_pizero1_px',
        'reco_taun_pizero1_py',
        'reco_taun_pizero1_pz',
        'reco_taun_pizero1_e', 
        'reco_taun_pizero2_px',
        'reco_taun_pizero2_py',
        'reco_taun_pizero2_pz',
        'reco_taun_pizero2_e', 
        'reco_Z_px',
        'reco_Z_py',
        'reco_Z_pz',
        'reco_Z_e',
        'BS_x',
        'BS_y',
        'BS_z',

]
branch_vals = {}
for b in branches:
    branch_vals[b] = array('f',[0])
    new_tree.Branch(b,  branch_vals[b],  '%s/F' % b)

# Determine the range of entries to process
n_entries = tree.GetEntries()
start_entry = args.n_skip * args.n_events if args.n_events > 0 else 0
end_entry = start_entry + args.n_events if args.n_events > 0 else n_entries

if start_entry >= n_entries:
    raise Exception("Error: Start entry exceeds total number of entries in the tree.")
    input_root.Close()
    output_root.Close()

end_entry = min(end_entry, n_entries)

# Loop over the entries in the input tree
count = 0
mean_dsol=0.

# Note reco reconstruction only implemented for pipi channel so far...

smearing = Smearing()
reconstructor = TauReconstructor()


count_correct = 0

for i in range(start_entry, end_entry):

    # initialize the branch values to zero
    for b in branches:
        branch_vals[b][0] = 0

    tree.GetEntry(i)

    branch_vals['taup_npi'][0] = tree.taup_npi
    branch_vals['taup_npizero'][0] = tree.taup_npizero
    branch_vals['taun_npi'][0] = tree.taun_npi
    branch_vals['taun_npizero'][0] = tree.taun_npizero

    # get tau 4-vectors
    taup = ROOT.TLorentzVector(tree.taup_px, tree.taup_py, tree.taup_pz, tree.taup_e)
    taun = ROOT.TLorentzVector(tree.taun_px, tree.taun_py, tree.taun_pz, tree.taun_e)

    taup_pi1 = ROOT.TLorentzVector(tree.taup_pi1_px, tree.taup_pi1_py, tree.taup_pi1_pz, tree.taup_pi1_e)
    taun_pi1 = ROOT.TLorentzVector(tree.taun_pi1_px, tree.taun_pi1_py, tree.taun_pi1_pz, tree.taun_pi1_e)

    if tree.taup_npi == 3:
        taup_pi3 = ROOT.TLorentzVector(tree.taup_pi3_px, tree.taup_pi3_py, tree.taup_pi3_pz, tree.taup_pi3_e)
        taup_pi2 = ROOT.TLorentzVector(tree.taup_pi2_px, tree.taup_pi2_py, tree.taup_pi2_pz, tree.taup_pi2_e)
        taup_SV = ROOT.TVector3(tree.taup_pi1_vx, tree.taup_pi1_vy, tree.taup_pi1_vz)
    else: 
        taup_pi2 = None
        taup_pi3 = None
        taup_SV = None
    if tree.taun_npi == 3:
        taun_pi2 = ROOT.TLorentzVector(tree.taun_pi2_px, tree.taun_pi2_py, tree.taun_pi2_pz, tree.taun_pi2_e)
        taun_pi3 = ROOT.TLorentzVector(tree.taun_pi3_px, tree.taun_pi3_py, tree.taun_pi3_pz, tree.taun_pi3_e)
        taun_SV = ROOT.TVector3(tree.taun_pi1_vx, tree.taun_pi1_vy, tree.taun_pi1_vz)
    else:
        taun_pi2 = None
        taun_pi3 = None
        taun_SV = None

    if args.smear_mode in [1,2]: 
        taup_pi1_reco = smearing.SmearTrack(taup_pi1)
        taun_pi1_reco = smearing.SmearTrack(taun_pi1)
        taup_pi2_reco = smearing.SmearTrack(taup_pi2)
        taun_pi2_reco = smearing.SmearTrack(taun_pi2)
        taup_pi3_reco = smearing.SmearTrack(taup_pi3)
        taun_pi3_reco = smearing.SmearTrack(taun_pi3)
    else: 
        if taup_pi1 is not None: taup_pi1_reco = taup_pi1.Clone()
        else: taup_pi1_reco = None
        if taun_pi1 is not None: taun_pi1_reco = taun_pi1.Clone()
        else: taun_pi1_reco = None
        if taup_pi2 is not None: taup_pi2_reco = taup_pi2.Clone()
        else: taup_pi2_reco = None
        if taun_pi2 is not None: taun_pi2_reco = taun_pi2.Clone()
        else: taun_pi2_reco = None
        if taup_pi3 is not None: taup_pi3_reco = taup_pi3.Clone()
        else: taup_pi3_reco = None
        if taun_pi3 is not None: taun_pi3_reco = taun_pi3.Clone()
        else: taun_pi3_reco = None

    if taup_pi1 is not None:
        branch_vals['reco_taup_pi1_px'][0] = taup_pi1_reco.Px()
        branch_vals['reco_taup_pi1_py'][0] = taup_pi1_reco.Py()
        branch_vals['reco_taup_pi1_pz'][0] = taup_pi1_reco.Pz()
        branch_vals['reco_taup_pi1_e'][0] = taup_pi1_reco.E()
    if taup_pi2 is not None:
        branch_vals['reco_taup_pi2_px'][0] = taup_pi2_reco.Px()
        branch_vals['reco_taup_pi2_py'][0] = taup_pi2_reco.Py()
        branch_vals['reco_taup_pi2_pz'][0] = taup_pi2_reco.Pz()
        branch_vals['reco_taup_pi2_e'][0] = taup_pi2_reco.E()
    if taup_pi3 is not None:
        branch_vals['reco_taup_pi3_px'][0] = taup_pi3_reco.Px()
        branch_vals['reco_taup_pi3_py'][0] = taup_pi3_reco.Py()
        branch_vals['reco_taup_pi3_pz'][0] = taup_pi3_reco.Pz()
        branch_vals['reco_taup_pi3_e'][0] = taup_pi3_reco.E()
    if taun_pi1 is not None:
        branch_vals['reco_taun_pi1_px'][0] = taun_pi1_reco.Px()
        branch_vals['reco_taun_pi1_py'][0] = taun_pi1_reco.Py()
        branch_vals['reco_taun_pi1_pz'][0] = taun_pi1_reco.Pz()
        branch_vals['reco_taun_pi1_e'][0] = taun_pi1_reco.E()
    if taun_pi2 is not None:
        branch_vals['reco_taun_pi2_px'][0] = taun_pi2_reco.Px()
        branch_vals['reco_taun_pi2_py'][0] = taun_pi2_reco.Py()
        branch_vals['reco_taun_pi2_pz'][0] = taun_pi2_reco.Pz()
        branch_vals['reco_taun_pi2_e'][0] = taun_pi2_reco.E()
    if taun_pi3 is not None:
        branch_vals['reco_taun_pi3_px'][0] = taun_pi3_reco.Px()
        branch_vals['reco_taun_pi3_py'][0] = taun_pi3_reco.Py()
        branch_vals['reco_taun_pi3_pz'][0] = taun_pi3_reco.Pz()
        branch_vals['reco_taun_pi3_e'][0] = taun_pi3_reco.E()
        

    taup_pizero1 = ROOT.TLorentzVector(tree.taup_pizero1_px, tree.taup_pizero1_py, tree.taup_pizero1_pz, tree.taup_pizero1_e)
    taun_pizero1 = ROOT.TLorentzVector(tree.taun_pizero1_px, tree.taun_pizero1_py, tree.taun_pizero1_pz, tree.taun_pizero1_e)

    taup_pizero2 = ROOT.TLorentzVector(tree.taup_pizero2_px, tree.taup_pizero2_py, tree.taup_pizero2_pz, tree.taup_pizero2_e)
    taun_pizero2 = ROOT.TLorentzVector(tree.taun_pizero2_px, tree.taun_pizero2_py, tree.taun_pizero2_pz, tree.taun_pizero2_e)

    if args.smear_mode in [1,3]: 
        taup_pizero1_reco = smearing.SmearPi0(taup_pizero1)
        taup_pizero2_reco = smearing.SmearPi0(taup_pizero2)
    else: 
        taup_pizero1_reco = taup_pizero1.Clone()
        taup_pizero2_reco = taup_pizero2.Clone()
    if args.smear_mode in [1,3]: 
        taun_pizero1_reco = smearing.SmearPi0(taun_pizero1)
        taun_pizero2_reco = smearing.SmearPi0(taun_pizero2)
    else: 
        taun_pizero1_reco = taun_pizero1.Clone()
        taun_pizero2_reco = taun_pizero2.Clone()

    branch_vals['reco_taup_pizero1_px'][0] = taup_pizero1_reco.Px()
    branch_vals['reco_taup_pizero1_py'][0] = taup_pizero1_reco.Py()
    branch_vals['reco_taup_pizero1_pz'][0] = taup_pizero1_reco.Pz()
    branch_vals['reco_taup_pizero1_e'][0] = taup_pizero1_reco.E()
    branch_vals['reco_taup_pizero2_px'][0] = taup_pizero2_reco.Px()
    branch_vals['reco_taup_pizero2_py'][0] = taup_pizero2_reco.Py()
    branch_vals['reco_taup_pizero2_pz'][0] = taup_pizero2_reco.Pz()
    branch_vals['reco_taup_pizero2_e'][0] = taup_pizero2_reco.E()
    branch_vals['reco_taun_pizero1_px'][0] = taun_pizero1_reco.Px()
    branch_vals['reco_taun_pizero1_py'][0] = taun_pizero1_reco.Py()
    branch_vals['reco_taun_pizero1_pz'][0] = taun_pizero1_reco.Pz()
    branch_vals['reco_taun_pizero1_e'][0] = taun_pizero1_reco.E()
    branch_vals['reco_taun_pizero2_px'][0] = taun_pizero2_reco.Px()
    branch_vals['reco_taun_pizero2_py'][0] = taun_pizero2_reco.Py()
    branch_vals['reco_taun_pizero2_pz'][0] = taun_pizero2_reco.Pz()
    branch_vals['reco_taun_pizero2_e'][0] = taun_pizero2_reco.E()


    VERTEX_taup = ROOT.TVector3(tree.taup_pi1_vx, tree.taup_pi1_vy, tree.taup_pi1_vz) # in mm
    VERTEX_taun = ROOT.TVector3(tree.taun_pi1_vx, tree.taun_pi1_vy, tree.taun_pi1_vz) # in mm

    #old method for d_min determination and smearing
    d_min_old = FindDMin(VERTEX_taun, taun_pi1.Vect().Unit(), VERTEX_taup, taup_pi1.Vect().Unit())
    #if args.smear_mode in [1,4]: d_min_reco = smearing.SmearDmin(d_min)
    #else: d_min_reco = d_min.Clone()

    # in new method we smear the points on the tracks seperatly and then recomput d_min
    # This shouldn't be needed but we shift the points on the tracks to approximatly the distance between the ALEPH beam and the first pixel layer (~6cm)
    POINT_taun = VERTEX_taun + taun_pi1.Vect().Unit()*60
    POINT_taup = VERTEX_taup + taup_pi1.Vect().Unit()*60


    d_min, d_min_point_n, d_min_point_p = FindDMin(POINT_taun, taun_pi1.Vect().Unit(), POINT_taup, taup_pi1.Vect().Unit(), return_points=True)

    if args.smear_mode in [1,4]: 
        d_min_point_n_reco = smearing.SmearPoint(d_min_point_n)
        d_min_point_p_reco = smearing.SmearPoint(d_min_point_p)
    else:
        d_min_point_n_reco = d_min_point_n.Clone()
        d_min_point_p_reco = d_min_point_p.Clone()
   

    #d_min_reco = d_min_point_p_reco-d_min_point_n_reco
    d_min_reco = FindDMin(d_min_point_n_reco, taun_pi1.Vect().Unit(), d_min_point_p_reco, taup_pi1.Vect().Unit())

    # old d_min smearing smears d_min directly
    if args.smear_mode in [1,4]: 
        d_min_reco_1 = smearing.SmearDmin(d_min)
    else: 
        d_min_reco_1 = d_min.Clone()

    # in gen level samples the PV is always at the origin
    BS = ROOT.TVector3(0.,0.,0.)
    if args.smear_mode in [1,5]:
        BS_reco = smearing.SmearBS(BS)
    else:
        BS_reco = BS.Clone()

    branch_vals['BS_x'][0] = BS_reco.X()
    branch_vals['BS_y'][0] = BS_reco.Y()
    branch_vals['BS_z'][0] = BS_reco.Z()

    # define ips wrt to the beam spot
    d_min_point_p_reco_wrt_bs = d_min_point_p_reco - BS_reco
    d_min_point_n_reco_wrt_bs = d_min_point_n_reco - BS_reco

    branch_vals['reco_taup_pi1_ipx'][0] = d_min_point_p_reco_wrt_bs.X()
    branch_vals['reco_taup_pi1_ipy'][0] = d_min_point_p_reco_wrt_bs.Y()
    branch_vals['reco_taup_pi1_ipz'][0] = d_min_point_p_reco_wrt_bs.Z()
    branch_vals['reco_taun_pi1_ipx'][0] = d_min_point_n_reco_wrt_bs.X()
    branch_vals['reco_taun_pi1_ipy'][0] = d_min_point_n_reco_wrt_bs.Y()
    branch_vals['reco_taun_pi1_ipz'][0] = d_min_point_n_reco_wrt_bs.Z()

    if tree.taup_npizero == 0 and tree.taup_npi == 1:
        taupvis_reco = taup_pi1_reco.Clone()
    elif tree.taup_npizero == 1 and tree.taup_npi == 1:   
        taupvis_reco = taup_pi1_reco + taup_pizero1_reco
    elif tree.taup_npizero == 2 and tree.taup_npi == 1:   
        taupvis_reco = taup_pi1_reco + taup_pizero1_reco + taup_pizero2_reco
    elif tree.taup_npizero == 0 and tree.taup_npi == 3:
        taupvis_reco = taup_pi1_reco + taup_pi2_reco + taup_pi3_reco
    else: 
        taupvis_reco = taup_pi1_reco.Clone()

    if tree.taun_npizero == 0 and tree.taun_npi == 1:
        taunvis_reco = taun_pi1_reco.Clone()
    elif tree.taun_npizero == 1 and tree.taun_npi == 1:
        taunvis_reco = taun_pi1_reco + taun_pizero1_reco 
    elif tree.taun_npizero == 2 and tree.taun_npi == 1:
        taunvis_reco = taun_pi1_reco + taun_pizero1_reco + taun_pizero2_reco
    elif tree.taun_npizero == 0 and tree.taun_npi == 3:
        taunvis_reco = taun_pi1_reco + taun_pi2_reco + taun_pi3_reco
    else:
        taunvis_reco = taun_pi1_reco.Clone()

    mode=1
    if (tree.taup_npi == 1 and tree.taup_npizero == 1) or (tree.taun_npi == 1 and tree.taun_npizero == 1):
        mode = 2
    elif tree.taup_npi == 3 and tree.taun_npi == 3:
        mode = 3


    #P_Z = ROOT.TLorentzVector(0.,0.,0.,91.188) # assuming we don't know ISR and have to assume momentum is balanced
    P_Z = taup + taun # assuming we know the P_Z because of detecting the other photons etc
    if args.smear_mode in [1,6] : P_Z_reco = smearing.SmearQ(P_Z) 
    else: P_Z_reco = P_Z.Clone()

    branch_vals['reco_Z_px'][0] = P_Z_reco.Px()
    branch_vals['reco_Z_py'][0] = P_Z_reco.Py()
    branch_vals['reco_Z_pz'][0] = P_Z_reco.Pz()
    branch_vals['reco_Z_e'][0] = P_Z_reco.E()

    if args.smear_mode in [1,7]:
        taup_SV_reco = smearing.SmearSV(taup_SV)
        taun_SV_reco = smearing.SmearSV(taun_SV)
    elif taup_SV is not None:
        taup_SV_reco = taup_SV.Clone()
        taun_SV_reco = taun_SV.Clone()
    else: 
        taup_SV_reco = None
        taun_SV_reco = None

    if taup_SV_reco is not None and taun_SV_reco is not None:
        sv_delta = taup_SV_reco - taun_SV_reco
    else: sv_delta = None

    if taup_SV_reco is not None:

        # define the sv wrt the beamspot
        taup_SV_reco_wrt_bs = taup_SV_reco - BS_reco

        branch_vals['reco_taup_vx'][0] = taup_SV_reco_wrt_bs.X()
        branch_vals['reco_taup_vy'][0] = taup_SV_reco_wrt_bs.Y()
        branch_vals['reco_taup_vz'][0] = taup_SV_reco_wrt_bs.Z()
    if taun_SV_reco is not None:

        # define the sv wrt the beamspot
        taun_SV_reco_wrt_bs = taun_SV_reco - BS_reco

        branch_vals['reco_taun_vx'][0] = taun_SV_reco_wrt_bs.X()
        branch_vals['reco_taun_vy'][0] = taun_SV_reco_wrt_bs.Y()
        branch_vals['reco_taun_vz'][0] = taun_SV_reco_wrt_bs.Z()

    branch_vals['taup_vz'][0] = tree.taup_pi1_vz # use this to check the events match the friend tree

    branch_vals['mass'][0] = P_Z.M()
    branch_vals['reco_mass'][0] = P_Z_reco.M()

    solutions = reconstructor.reconstruct_tau_alt(P_Z_reco, taupvis_reco, taunvis_reco, taup_pi1_reco, taun_pi1_reco, np_point=d_min_point_p_reco, nn_point=d_min_point_n_reco, O_y=BS_reco.Y(), d_min_reco=d_min_reco, sv_delta=sv_delta, mode=mode, no_minimisation=True)

    ###np.random.shuffle(solutions) #shuffle solutions randomly for checks

    taup_reco, taun_reco, d_min_pred, decay_length_prob, sv_delta_constraint = solutions[0]

    # determine if correct solution was found
    d_sol1 = compare_lorentz_pairs((taup,taun),solutions[0])
    d_sol2 = compare_lorentz_pairs((taup,taun),solutions[1])

    mean_dsol += d_sol1 

    # assume we got the correct solutions if the 4-vectors of the taus are closer to the gen-level taus - not completly correct in general but should be a good approximation
    correct_solutions = (d_sol1 <= d_sol2)
    
    if correct_solutions: count_correct+=1

    #TODO: could try to use a random number sampling of the solutions to avoid a bias towards solutions with smaller decay lengths
    d_min_constraint = d_min_pred.Dot(d_min_reco)

    if tree.taup_npi == 1 and tree.taup_npizero == 0:
        taup_pi1.Boost(-taup.BoostVector())
        taup_s = taup_pi1.Vect().Unit()

        taup_pi1_reco.Boost(-taup_reco.BoostVector())
        taup_s_reco = taup_pi1_reco.Vect().Unit()
    elif tree.taup_npi == 1 and tree.taup_npizero >= 1:
        q = taup_pi1  - taup_pizero1
        P = taup
        N = taup - taup_pi1 - taup_pizero1
        pv = P.M()*(2*(q*N)*q - q.Mag2()*N) * (1/ (2*(q*N)*(q*P) - q.Mag2()*(N*P)))
        pv.Boost(-taup.BoostVector())
        taup_s = pv.Vect().Unit()

        q = taup_pi1_reco  - taup_pizero1_reco
        P = taup_reco
        N = taup_reco - taup_pi1_reco - taup_pizero1_reco
        pv = P.M()*(2*(q*N)*q - q.Mag2()*N) * (1/ (2*(q*N)*(q*P) - q.Mag2()*(N*P)))
        pv.Boost(-taup_reco.BoostVector())
        taup_s_reco = pv.Vect().Unit()
    elif tree.taup_npi == 3:
        pv =  -PolarimetricA1(taup, taup_pi1, taup_pi2, taup_pi3, +1).PVC()
        pv.Boost(-taup.BoostVector())
        taup_s = pv.Vect().Unit()

        pv = -PolarimetricA1(taup_reco, taup_pi1_reco, taup_pi2_reco, taup_pi3_reco, +1).PVC()
        pv.Boost(-taup_reco.BoostVector())
        taup_s_reco = pv.Vect().Unit()
    else: 
        print("WARNING: Number of pions not equal to 1 or 3")
        new_tree.Fill() # any missing variables will be filled with 0
        continue

    if tree.taun_npi == 1 and tree.taun_npizero == 0:   
        taun_pi1.Boost(-taun.BoostVector())
        taun_s = taun_pi1.Vect().Unit()

        taun_pi1_reco.Boost(-taun_reco.BoostVector())
        taun_s_reco = taun_pi1_reco.Vect().Unit()
    elif tree.taun_npi == 1 and tree.taun_npizero >= 1:
        q = taun_pi1  - taun_pizero1
        P = taun
        N = taun - taun_pi1 - taun_pizero1
        pv = P.M()*(2*(q*N)*q - q.Mag2()*N) * (1/ (2*(q*N)*(q*P) - q.Mag2()*(N*P)))
        pv.Boost(-taun.BoostVector())
        taun_s = pv.Vect().Unit()

        q = taun_pi1_reco  - taun_pizero1_reco
        P = taun_reco
        N = taun_reco - taun_pi1_reco - taun_pizero1_reco
        pv = P.M()*(2*(q*N)*q - q.Mag2()*N) * (1/ (2*(q*N)*(q*P) - q.Mag2()*(N*P)))
        pv.Boost(-taun_reco.BoostVector())
        taun_s_reco = pv.Vect().Unit()                  
    elif tree.taun_npi == 3:
        pv =  -PolarimetricA1(taun, taun_pi1, taun_pi2, taun_pi3, +1).PVC()
        pv.Boost(-taun.BoostVector())
        taun_s = pv.Vect().Unit()

        pv = -PolarimetricA1(taun_reco, taun_pi1_reco, taun_pi2_reco, taun_pi3_reco, +1).PVC()
        pv.Boost(-taun_reco.BoostVector())
        taun_s_reco = pv.Vect().Unit()
    else: 
        print("WARNING: Number of pions not equal to 1 or 3")
        new_tree.Fill() # any missing variables will be filled with 0
        continue

    # compute coordinate vectors here (n,r,k)
    # p is direction of e+ beam - this will be known also for reco variables!
    p = ROOT.TVector3(tree.z_x, tree.z_y, tree.z_z).Unit()

    # k is direction of tau+
    k = taup.Vect().Unit()
    n = (p.Cross(k)).Unit()
    cosTheta = p.Dot(k)
    r = (p - (k*cosTheta)).Unit() 

    # k is direction of tau+
    k_reco = taup_reco.Vect().Unit()
    n_reco = (p.Cross(k_reco)).Unit()
    cosTheta_reco = p.Dot(k_reco)
    r_reco = (p - (k_reco*cosTheta_reco)).Unit()    

    branch_vals['cosn_plus'][0] = taup_s.Dot(n)
    branch_vals['cosr_plus'][0] = taup_s.Dot(r)
    branch_vals['cosk_plus'][0] = taup_s.Dot(k)
    branch_vals['cosn_minus'][0] = taun_s.Dot(n)
    branch_vals['cosr_minus'][0] = taun_s.Dot(r)
    branch_vals['cosk_minus'][0] = taun_s.Dot(k)
    branch_vals['cosTheta'][0] = cosTheta

    branch_vals['cosn_plus_reco'][0] = taup_s_reco.Dot(n_reco)
    branch_vals['cosr_plus_reco'][0] = taup_s_reco.Dot(r_reco)
    branch_vals['cosk_plus_reco'][0] = taup_s_reco.Dot(k_reco)
    branch_vals['cosn_minus_reco'][0] = taun_s_reco.Dot(n_reco)
    branch_vals['cosr_minus_reco'][0] = taun_s_reco.Dot(r_reco)
    branch_vals['cosk_minus_reco'][0] = taun_s_reco.Dot(k_reco)
    branch_vals['cosTheta_reco'][0] = cosTheta_reco 


    ## Fill the new tree
    new_tree.Fill()
    count+=1
    if count % 1000 == 0:
        print('Processed %i events' % count)
        new_tree.AutoSave("SaveSelf")

print('count:', count)
print('Correct solutions found in %i events out of %i = %.1f%%' % (count_correct, count, float(count_correct)/count*100))
mean_dsol/=count
print('mean d_sol = %g' % mean_dsol)

# Write the new tree to the output file
new_tree.Write()

if new_tree.GetEntries() != tree.GetEntries():
    print("Warning: The number of entries in the new tree does not match the original tree.")
    print(f"Original tree entries: {tree.GetEntries()}, New tree entries: {new_tree.GetEntries()}")

# Close the files
input_root.Close()
output_root.Close()
print('Finished running smearing and reconstruction')
