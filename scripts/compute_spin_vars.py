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
        'reco_taup_nu_px',
        'reco_taup_nu_py',
        'reco_taup_nu_pz',
        'reco_taun_nu_px',
        'reco_taun_nu_py',
        'reco_taun_nu_pz',
        'reco_alt_taup_nu_px',
        'reco_alt_taup_nu_py',
        'reco_alt_taup_nu_pz',
        'reco_alt_taun_nu_px',
        'reco_alt_taun_nu_py',
        'reco_alt_taun_nu_pz',
        'reco_d0_taup_nu_px',
        'reco_d0_taup_nu_py',
        'reco_d0_taup_nu_pz',
        'reco_d0_taun_nu_px',
        'reco_d0_taun_nu_py',
        'reco_d0_taun_nu_pz',

        'reco_dplus_taup_nu_px',
        'reco_dplus_taup_nu_py',
        'reco_dplus_taup_nu_pz',
        'reco_dplus_taun_nu_px',
        'reco_dplus_taun_nu_py',
        'reco_dplus_taun_nu_pz',
        'reco_dminus_taup_nu_px',
        'reco_dminus_taup_nu_py',
        'reco_dminus_taup_nu_pz',
        'reco_dminus_taun_nu_px',
        'reco_dminus_taun_nu_py',
        'reco_dminus_taun_nu_pz',


        'dplus_taup_nu_px',
        'dplus_taup_nu_py',
        'dplus_taup_nu_pz',
        'dplus_taun_nu_px',
        'dplus_taun_nu_py',
        'dplus_taun_nu_pz',
        'dminus_taup_nu_px',
        'dminus_taup_nu_py',
        'dminus_taup_nu_pz',
        'dminus_taun_nu_px',
        'dminus_taun_nu_py',
        'dminus_taun_nu_pz',
        'dsign',
        'dsign_alt',
        'taup_nu_px',
        'taup_nu_py',
        'taup_nu_pz',
        'taun_nu_px',
        'taun_nu_py',
        'taun_nu_pz',

        'dplus_taup_l',
        'dplus_taun_l',
        'dplus_taup_l_sv',
        'dplus_taun_l_sv',
        'dplus_dmin_constraint',
        'dplus_sv_delta_constraint',
        'dminus_taup_l',
        'dminus_taun_l',
        'dminus_taup_l_sv',
        'dminus_taun_l_sv',
        'dminus_dmin_constraint',
        'dminus_sv_delta_constraint',

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
    d_min_reco, pca_n_reco, pca_p_reco = FindDMin(d_min_point_n_reco, taun_pi1.Vect().Unit(), d_min_point_p_reco, taup_pi1.Vect().Unit(), return_points=True)

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
    pca_p_reco_wrt_bs = pca_p_reco - BS_reco
    pca_n_reco_wrt_bs = pca_n_reco - BS_reco

    branch_vals['reco_taup_pi1_ipx'][0] = pca_p_reco_wrt_bs.X()
    branch_vals['reco_taup_pi1_ipy'][0] = pca_p_reco_wrt_bs.Y()
    branch_vals['reco_taup_pi1_ipz'][0] = pca_p_reco_wrt_bs.Z()
    branch_vals['reco_taun_pi1_ipx'][0] = pca_n_reco_wrt_bs.X()
    branch_vals['reco_taun_pi1_ipy'][0] = pca_n_reco_wrt_bs.Y()
    branch_vals['reco_taun_pi1_ipz'][0] = pca_n_reco_wrt_bs.Z()


    if tree.taup_npizero == 0 and tree.taup_npi == 1:
        taupvis_reco = taup_pi1_reco.Clone()
        taupvis = taup_pi1.Clone()
    elif tree.taup_npizero == 1 and tree.taup_npi == 1:   
        taupvis_reco = taup_pi1_reco + taup_pizero1_reco
        taupvis = taup_pi1 + taup_pizero1
    elif tree.taup_npizero == 2 and tree.taup_npi == 1:   
        taupvis_reco = taup_pi1_reco + taup_pizero1_reco + taup_pizero2_reco
        taupvis = taup_pi1 + taup_pizero1 + taup_pizero2
    elif tree.taup_npizero == 0 and tree.taup_npi == 3:
        taupvis_reco = taup_pi1_reco + taup_pi2_reco + taup_pi3_reco
        taupvis = taup_pi1 + taup_pi2 + taup_pi3
    else: 
        taupvis_reco = taup_pi1_reco.Clone()
        taupvis = taup_pi1.Clone()

    if tree.taun_npizero == 0 and tree.taun_npi == 1:
        taunvis_reco = taun_pi1_reco.Clone()
        taunvis = taun_pi1.Clone()
    elif tree.taun_npizero == 1 and tree.taun_npi == 1:
        taunvis_reco = taun_pi1_reco + taun_pizero1_reco 
        taunvis = taun_pi1 + taun_pizero1
    elif tree.taun_npizero == 2 and tree.taun_npi == 1:
        taunvis_reco = taun_pi1_reco + taun_pizero1_reco + taun_pizero2_reco
        taunvis = taun_pi1 + taun_pizero1 + taun_pizero2
    elif tree.taun_npizero == 0 and tree.taun_npi == 3:
        taunvis_reco = taun_pi1_reco + taun_pi2_reco + taun_pi3_reco
        taunvis = taun_pi1 + taun_pi2 + taun_pi3
    else:
        taunvis_reco = taun_pi1_reco.Clone()
        taunvis = taun_pi1.Clone()

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
    else:
        if taup_SV is not None: taup_SV_reco = taup_SV.Clone()
        else: taup_SV_reco = None
        if taun_SV is not None: taun_SV_reco = taun_SV.Clone()
        else: taun_SV_reco = None

    if taup_SV_reco is not None and taun_SV_reco is not None:
        sv_delta_reco = taup_SV_reco - taun_SV_reco
        sv_delta = taup_SV - taun_SV
    else: 
        sv_delta_reco = None
        sv_delta = None

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

    solutions = reconstructor.reconstruct_tau_alt(P_Z_reco, taupvis_reco, taunvis_reco, taup_pi1_reco, taun_pi1_reco, np_point=d_min_point_p_reco, nn_point=d_min_point_n_reco, O_y=BS_reco.Y(), d_min_reco=d_min_reco, sv_delta=sv_delta_reco, mode=mode, no_minimisation=True)
    solutions_gen = ReconstructTauAnalytically(P_Z, taupvis, taunvis, taup_pi1, taun_pi1, return_values=True)

    # if first element has -ve d and second has -ve, then switch the solutions (they should already be arranged this way though, but just incase...)
    if solutions_gen[0][6] < 0 and solutions_gen[1][6] > 0:
        solutions_gen[0], solutions_gen[1] = solutions_gen[1], solutions_gen[0]
    dplus_taun_nu = (solutions_gen[0][1]-taunvis)
    dplus_taup_nu = (solutions_gen[0][0]-taupvis)
    dminus_taun_nu = (solutions_gen[1][1]-taunvis)
    dminus_taup_nu = (solutions_gen[1][0]-taupvis)
    branch_vals['dplus_taup_nu_px'][0] = dplus_taup_nu.Px()
    branch_vals['dplus_taup_nu_py'][0] = dplus_taup_nu.Py()
    branch_vals['dplus_taup_nu_pz'][0] = dplus_taup_nu.Pz()
    branch_vals['dplus_taun_nu_px'][0] = dplus_taun_nu.Px()
    branch_vals['dplus_taun_nu_py'][0] = dplus_taun_nu.Py()
    branch_vals['dplus_taun_nu_pz'][0] = dplus_taun_nu.Pz()
    branch_vals['dminus_taup_nu_px'][0] = dminus_taup_nu.Px()
    branch_vals['dminus_taup_nu_py'][0] = dminus_taup_nu.Py()
    branch_vals['dminus_taup_nu_pz'][0] = dminus_taup_nu.Pz()
    branch_vals['dminus_taun_nu_px'][0] = dminus_taun_nu.Px()
    branch_vals['dminus_taun_nu_py'][0] = dminus_taun_nu.Py()
    branch_vals['dminus_taun_nu_pz'][0] = dminus_taun_nu.Pz()
        

    # get the correct solution for the gen tau
    taun = ROOT.TLorentzVector(tree.taun_px, tree.taun_py, tree.taun_pz, tree.taun_e)
    taup = ROOT.TLorentzVector(tree.taup_px, tree.taup_py, tree.taup_pz, tree.taup_e)

    taup_nu = ROOT.TLorentzVector(tree.taup_nu_px, tree.taup_nu_py, tree.taup_nu_pz, tree.taup_nu_e)
    taun_nu = ROOT.TLorentzVector(tree.taun_nu_px, tree.taun_nu_py, tree.taun_nu_pz, tree.taun_nu_e)
    branch_vals['taup_nu_px'][0] = taup_nu.Px()
    branch_vals['taup_nu_py'][0] = taup_nu.Py()
    branch_vals['taup_nu_pz'][0] = taup_nu.Pz()
    branch_vals['taun_nu_px'][0] = taun_nu.Px()
    branch_vals['taun_nu_py'][0] = taun_nu.Py()
    branch_vals['taun_nu_pz'][0] = taun_nu.Pz()

    _, _, _, dsign = solve_abcd_values(taup, taun, P_Z, taupvis, taunvis)
    branch_vals['dsign'][0] = np.sign(dsign)

    # determine the correct sign for dsign by checking which os the two solutions match most clostly to the true value
    d_sol1 = compare_lorentz_pairs((taup,taun),solutions_gen[0][:5])
    d_sol2 = compare_lorentz_pairs((taup,taun),solutions_gen[1][:5])

    if d_sol1<d_sol2:
        dsign_alt = 1
    elif d_sol1>d_sol2:
        dsign_alt = -1
    else: 
        dsign_alt = 0.
    branch_vals['dsign_alt'][0] = dsign_alt


        

    ###np.random.shuffle(solutions) #shuffle solutions randomly for checks

    taup_reco, taun_reco, d_min_pred, decay_length_prob, sv_delta_constraint = solutions[0]
    taup_nu_reco = taup_reco - taupvis_reco
    taun_nu_reco = taun_reco - taunvis_reco
    branch_vals['reco_taup_nu_px'][0] = taup_nu_reco.Px()
    branch_vals['reco_taup_nu_py'][0] = taup_nu_reco.Py()
    branch_vals['reco_taup_nu_pz'][0] = taup_nu_reco.Pz()
    branch_vals['reco_taun_nu_px'][0] = taun_nu_reco.Px()
    branch_vals['reco_taun_nu_py'][0] = taun_nu_reco.Py()
    branch_vals['reco_taun_nu_pz'][0] = taun_nu_reco.Pz()

    taup_alt_reco, taun_alt_reco, _, _, _ = solutions[1]
    taup_nu_alt_reco = taup_alt_reco - taupvis_reco
    taun_nu_alt_reco = taun_alt_reco - taunvis_reco
    branch_vals['reco_alt_taup_nu_px'][0] = taup_nu_alt_reco.Px()
    branch_vals['reco_alt_taup_nu_py'][0] = taup_nu_alt_reco.Py()
    branch_vals['reco_alt_taup_nu_pz'][0] = taup_nu_alt_reco.Pz()
    branch_vals['reco_alt_taun_nu_px'][0] = taun_nu_alt_reco.Px()
    branch_vals['reco_alt_taun_nu_py'][0] = taun_nu_alt_reco.Py()
    branch_vals['reco_alt_taun_nu_pz'][0] = taun_nu_alt_reco.Pz()

    taup_d0_reco, taun_d0_reco, _, _, _ = solutions[2]
    taup_nu_d0_reco = taup_d0_reco - taupvis_reco
    taun_nu_d0_reco = taun_d0_reco - taunvis_reco
    branch_vals['reco_d0_taup_nu_px'][0] = taup_nu_d0_reco.Px()
    branch_vals['reco_d0_taup_nu_py'][0] = taup_nu_d0_reco.Py()
    branch_vals['reco_d0_taup_nu_pz'][0] = taup_nu_d0_reco.Pz()
    branch_vals['reco_d0_taun_nu_px'][0] = taun_nu_d0_reco.Px()
    branch_vals['reco_d0_taun_nu_py'][0] = taun_nu_d0_reco.Py()
    branch_vals['reco_d0_taun_nu_pz'][0] = taun_nu_d0_reco.Pz()

    taup_dplus_reco, taun_dplus_reco, _, _, _ = solutions[3]
    taup_nu_dplus_reco = taup_dplus_reco - taupvis_reco
    taun_nu_dplus_reco = taun_dplus_reco - taunvis_reco
    branch_vals['reco_dplus_taup_nu_px'][0] = taup_nu_dplus_reco.Px()
    branch_vals['reco_dplus_taup_nu_py'][0] = taup_nu_dplus_reco.Py()
    branch_vals['reco_dplus_taup_nu_pz'][0] = taup_nu_dplus_reco.Pz()
    branch_vals['reco_dplus_taun_nu_px'][0] = taun_nu_dplus_reco.Px()
    branch_vals['reco_dplus_taun_nu_py'][0] = taun_nu_dplus_reco.Py()
    branch_vals['reco_dplus_taun_nu_pz'][0] = taun_nu_dplus_reco.Pz()
    
    if taup_SV_reco is not None:
        np_point = taup_SV_reco 
        dir_p = taupvis_reco.Vect().Unit()
    else:
        np_point = d_min_point_p_reco
        dir_p = taup_pi1.Vect().Unit()
   
    if taun_SV_reco is not None:
        nn_point= taun_SV_reco 
        dir_n = taunvis_reco.Vect().Unit()
    else:
        nn_point = d_min_point_n_reco
        dir_n = taun_pi1.Vect().Unit()


    d_min_reco_forvars, _, _ = FindDMin(nn_point, dir_n, np_point, dir_p, return_points=True)
    dplus_taup_l, dplus_taun_l, dplus_taup_l_sv, dplus_taun_l_sv, dplus_dmin1_constraint, dplus_dmin2_constraint, dplus_sv_delta_constraint = GetDsignVars(taup=taup_dplus_reco, taun=taun_dplus_reco, taupvis=taupvis_reco, taunvis=taunvis_reco, taup_pi1=taup_pi1_reco, taun_pi1=taun_pi1_reco, O_y=BS_reco.Y(), np_point=d_min_point_p_reco, nn_point=d_min_point_n_reco, d_min_reco=d_min_reco_forvars, taup_sv=taup_SV_reco, taun_sv=taun_SV_reco)

    taup_dminus_reco, taun_dminus_reco, _, _, _ = solutions[4]
    taup_nu_dminus_reco = taup_dminus_reco - taupvis_reco
    taun_nu_dminus_reco = taun_dminus_reco - taunvis_reco
    branch_vals['reco_dminus_taup_nu_px'][0] = taup_nu_dminus_reco.Px()
    branch_vals['reco_dminus_taup_nu_py'][0] = taup_nu_dminus_reco.Py()
    branch_vals['reco_dminus_taup_nu_pz'][0] = taup_nu_dminus_reco.Pz()
    branch_vals['reco_dminus_taun_nu_px'][0] = taun_nu_dminus_reco.Px()
    branch_vals['reco_dminus_taun_nu_py'][0] = taun_nu_dminus_reco.Py()
    branch_vals['reco_dminus_taun_nu_pz'][0] = taun_nu_dminus_reco.Pz()
    dminus_taup_l, dminus_taun_l, dminus_taup_l_sv, dminus_taun_l_sv, dminus_dmin1_constraint, dminus_dmin2_constraint, dminus_sv_delta_constraint = GetDsignVars(taup=taup_dminus_reco, taun=taun_dminus_reco, taupvis=taupvis_reco, taunvis=taunvis_reco, taup_pi1=taup_pi1_reco, taun_pi1=taun_pi1_reco, O_y=BS_reco.Y(), np_point=d_min_point_p_reco, nn_point=d_min_point_n_reco, d_min_reco=d_min_reco_forvars, taup_sv=taup_SV_reco, taun_sv=taun_SV_reco)
    

    #print('!!!!!!!')
    #print(taup_dplus_reco.X(), taup_dminus_reco.X(), taup_reco.X())
    #print('gen tau:', taup.X())
    #print(tree.taup_npi, tree.taup_npizero, tree.taun_npi, tree.taun_npizero)
    #print('dsign:', branch_vals['dsign'][0], dsign_alt) #TODO - check this
    #print('plus: ', dplus_taup_l, dplus_taun_l, dplus_taup_l_sv, dplus_taun_l_sv, dplus_dmin1_constraint, dplus_dmin2_constraint, dplus_sv_delta_constraint)
    #print('minus:', dminus_taup_l, dminus_taun_l, dminus_taup_l_sv, dminus_taun_l_sv, dminus_dmin1_constraint, dminus_dmin2_constraint, dminus_sv_delta_constraint)
  #
    #if taup_SV_reco is not None and taun_SV_reco is not None:
    #    print('SV reco vs gen:', np_point.X(), taup_SV.X(), nn_point.X(), taun_SV.X())
    #    print('dirs reco vs gen:', dir_p.X(), taupvis.Vect().Unit().X(), dir_n.X(), taunvis.Vect().Unit().X())
    #    d_min_reco_forvars_gen, _, _ = FindDMin(taun_SV, taunvis.Vect().Unit(), taup_SV, taupvis.Vect().Unit(), return_points=True)
    #    print('d_min gen vs reco (x):', d_min_reco_forvars_gen.Unit().X(), d_min_reco_forvars.Unit().X())
    #    print('d_min gen vs reco (y):', d_min_reco_forvars_gen.Unit().Y(), d_min_reco_forvars.Unit().Y())
    #    print('d_min gen vs reco (z):', d_min_reco_forvars_gen.Unit().Z(), d_min_reco_forvars.Unit().Z())
  
    branch_vals['dplus_taup_l'][0] = dplus_taup_l
    branch_vals['dplus_taun_l'][0] = dplus_taun_l
    branch_vals['dplus_taup_l_sv'][0] = dplus_taup_l_sv
    branch_vals['dplus_taun_l_sv'][0] = dplus_taun_l_sv
    branch_vals['dplus_dmin_constraint'][0] = dplus_dmin1_constraint
    branch_vals['dplus_sv_delta_constraint'][0] = dplus_sv_delta_constraint
    branch_vals['dminus_taup_l'][0] = dminus_taup_l
    branch_vals['dminus_taun_l'][0] = dminus_taun_l
    branch_vals['dminus_taup_l_sv'][0] = dminus_taup_l_sv
    branch_vals['dminus_taun_l_sv'][0] = dminus_taun_l_sv
    branch_vals['dminus_dmin_constraint'][0] = dminus_dmin1_constraint
    branch_vals['dminus_sv_delta_constraint'][0] = dminus_sv_delta_constraint



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
