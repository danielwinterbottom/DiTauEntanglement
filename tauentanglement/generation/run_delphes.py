#!/usr/bin/env python3

import sys
import ROOT
import math
import argparse
from array import array
from tauentanglement.utils.ReconstructTaus import FindDMin_Point

def TraceTauMother(part, particles, verbose=False):
    """
    Recursively trace tau's ancestry until the first non-tau mother is found.
    """
    mother_indices = []
    if part.M1 >= 0:
        mother_indices.append(part.M1)
    if part.M2 >= 0:
        mother_indices.append(part.M2)

    # Take the first mother (can expand to loop over all if needed)
    mother = particles.At(mother_indices[0])
    mother_id = mother_indices[0]

    mother_pdgid = abs(mother.PID)

    if abs(mother.PID) == 15:
        # Keep going up the ancestry
        mother_id, mother_pdgid = TraceTauMother(mother, particles, verbose)
    else:
        # Print non-tau mother info unless it is in the excluded list
        if verbose:
            print('First non-tau mother found for tau:')
            print('id = %i, status = %i, pT = %.4f, eta = %.4f, phi = %.4f' %
                  (mother.PID, mother.Status, mother.PT, mother.Eta, mother.Phi))
        # check if there are any other mothers and if so print these as well
        if len(mother_indices) > 1:
            print("Other mothers found:")
            for idx in mother_indices[1:]:
                other_mother = particles.At(idx)
                if not other_mother.PID in mother_ids_to_exclude:
                    if verbose:
                        print('id = %i, status = %i, pT = %.4f, eta = %.4f, phi = %.4f' %
                              (other_mother.PID, other_mother.Status, other_mother.PT, other_mother.Eta, other_mother.Phi))

    return mother_id, mother_pdgid

def GetStableDaughters(part, particles):
    """
    Recursively get all stable daughters of a particle.
    """
    stable_daughters = []

    for i in range(particles.GetEntries()):
        p = particles.At(i)
        if (p.M1 >=0 and particles.At(p.M1).GetUniqueID() == part.GetUniqueID()) or (p.M2 >=0 and particles.At(p.M2).GetUniqueID() == part.GetUniqueID()):
            if p.Status == 1:  # Stable particle
                stable_daughters.append(p)
            else:
                stable_daughters.extend(GetStableDaughters(p, particles))
    return stable_daughters

def get_impact_parameter(p):

    #FindDMin_Point(taup_vtx_vec3, taup_pi1_dir_vec3, pv_vec3)
    p_vtx_vec3 = ROOT.TVector3(p.X, p.Y, p.Z)
    # set up direction using PT, Eta, Phi
    px = p.PT * math.cos(p.Phi)
    py = p.PT * math.sin(p.Phi)
    pz = p.PT * math.sinh(p.Eta)
    p_dir_vec3 = ROOT.TVector3(px, py, pz).Unit()

    pv_vec3 = ROOT.TVector3(0, 0, 0)
    impact_point_vec3 = FindDMin_Point(p_vtx_vec3, p_dir_vec3, pv_vec3)

    # also compute d0 and dz

    d0 = (impact_point_vec3 - pv_vec3).Perp()
    dz = impact_point_vec3.Z()
    return impact_point_vec3, d0, dz

parser = argparse.ArgumentParser()
parser.add_argument("--output" ,'-o', help="Name of output file")
parser.add_argument("--input" ,'-i', help="Name of input file")
args = parser.parse_args()

# Adjust this path if needed:
ROOT.gSystem.Load("libDelphes")

chain = ROOT.TChain("Delphes")
chain.Add(args.input)

reader = ROOT.ExRootTreeReader(chain)

particles = reader.UseBranch("Particle") # gen particles 
MET = reader.UseBranch("MissingET")
genMET = reader.UseBranch("GenMissingET")
photon = reader.UseBranch("EFlowPhoton")
eflowtrack = reader.UseBranch("EFlowTrack")

#setup output tree

branches = [
  'taup_npi', 'taup_npizero',
  'taun_npi', 'taun_npizero',
  'met_px', 'met_py',
  'taup_pi1_px', 'taup_pi1_py', 'taup_pi1_pz', 'taup_pi1_e',
  'taup_pi1_ipx', 'taup_pi1_ipy', 'taup_pi1_ipz',
  'taup_pi2_px', 'taup_pi2_py', 'taup_pi2_pz', 'taup_pi2_e',
  'taup_pi3_px', 'taup_pi3_py', 'taup_pi3_pz', 'taup_pi3_e',
  'taun_pi1_px', 'taun_pi1_py', 'taun_pi1_pz', 'taun_pi1_e',
  'taun_pi1_ipx', 'taun_pi1_ipy', 'taun_pi1_ipz',
  'taun_pi2_px', 'taun_pi2_py', 'taun_pi2_pz', 'taun_pi2_e',
  'taun_pi3_px', 'taun_pi3_py', 'taun_pi3_pz', 'taun_pi3_e',
  'taup_pizero1_px', 'taup_pizero1_py', 'taup_pizero1_pz', 'taup_pizero1_e',
  'taun_pizero1_px', 'taun_pizero1_py', 'taun_pizero1_pz', 'taun_pizero1_e',
  'taup_lep_px', 'taup_lep_py', 'taup_lep_pz', 'taup_lep_e',
  'taun_lep_px', 'taun_lep_py', 'taun_lep_pz', 'taun_lep_e',
  'taup_sv_x', 'taup_sv_y', 'taup_sv_z',
  'taun_sv_x', 'taun_sv_y', 'taun_sv_z',
  'taup_nu_px', 'taup_nu_py', 'taup_nu_pz',
  'taun_nu_px', 'taun_nu_py', 'taun_nu_pz'
]

fout = ROOT.TFile(args.output,'RECREATE')
tree = ROOT.TTree('tree','')

branch_vals = {}
for b in branches:
    branch_vals[b] = array('f',[0])
    tree.Branch(b,  branch_vals[b],  '%s/F' % b)


for iev in range(reader.GetEntries()):

    # initialize the branch values to zero
    for b in branches:
        branch_vals[b][0] = 0

    reader.ReadEntry(iev)

    gen_taus = []

    for i in range(particles.GetEntries()):
        p = particles.At(i)
        if abs(p.PID) == 15 and p.Status == 2: gen_taus.append(p)

    # for selecting the tau pair we first want to make all combinations of oppositly charged taus (opp PID) and same mother
    # if multiple pairs are found then the order of preference is given by mother PDGID: 25, 23, 22
    # if multiple pairs are found with the same mother PDGID then we select the pair with the highest sum of pT
    tau_pairs = []
    for i in range(len(gen_taus)):
        for j in range(i+1, len(gen_taus)):
            tau1 = gen_taus[i]
            tau2 = gen_taus[j]
            if tau1.PID * tau2.PID < 0: # opposite charge and same mother

                tau1_mother, tau1_mother_pdgid = TraceTauMother(tau1, particles, verbose=False)
                tau2_mother, tau2_mother_pdgid = TraceTauMother(tau2, particles, verbose=False)
                if tau1_mother == tau2_mother:
                    tau_pairs.append((tau1, tau2, tau1_mother_pdgid))


    if len(tau_pairs) == 0:
        print("  No tau pairs found, skipping event")
        continue
    if len(tau_pairs) > 1:
        print(f"  Found {len(tau_pairs)} tau pairs, applying selection criteria:")
        # apply selection criteria to select one pair
        tau_pairs.sort(key=lambda x: (x[2] != 25, x[2] != 23, x[2] != 22, -(x[0].PT + x[1].PT))) # sort by mother PDGID preference and then sum of pT
    best_pair = tau_pairs[0]

    # get taup and taun 
    taup = best_pair[0] if best_pair[0].Charge > 0 else best_pair[1]
    taun = best_pair[1] if best_pair[0].Charge > 0 else best_pair[0]

    taup_daughter = GetStableDaughters(taup, particles)
    taun_daughter = GetStableDaughters(taun, particles)

    # get charged particles from taus, gammas from taus, and neutrinos from taus
    taup_pis = []
    taun_pis = []
    taup_gammas = []
    taun_gammas = []
    taup_neutrinos = []
    taun_neutrinos = []
    taup_leptons = []
    taun_leptons = []
    for d in taup_daughter:
        if d.PID == 22:
            taup_gammas.append(d)
        elif abs(d.PID) in [11, 13]:
            taup_leptons.append(d)
        elif d.Charge != 0:
            taup_pis.append(d)
        elif abs(d.PID) in [12, 14, 16]:
            taup_neutrinos.append(d)

    for d in taun_daughter:
        if d.PID == 22:
            taun_gammas.append(d)
        elif abs(d.PID) in [11, 13]:
            taun_leptons.append(d)
        elif d.Charge != 0:
            taun_pis.append(d)
        elif abs(d.PID) in [12, 14, 16]:
            taun_neutrinos.append(d)

    # now we get the pi0 by summing together the gammas - note for dm=2 this will really be the 4-vector of 2 pi0s but this is anyway more similar to what we have for reco taus
    taup_pi0 = ROOT.TLorentzVector()
    for g in taup_gammas:
        taup_pi0 += g.P4()
    taun_pi0 = ROOT.TLorentzVector()
    for g in taun_gammas:
        taun_pi0 += g.P4()

    # now we get the sum of the neutrios as well (for hadronic taus this will just be a singlet neutrino but for leptonic decays it is two neutrinos)
    taup_neutrinos_sum = ROOT.TLorentzVector()
    for n in taup_neutrinos:
        taup_neutrinos_sum += n.P4()
    taun_neutrinos_sum = ROOT.TLorentzVector()
    for n in taun_neutrinos:
        taun_neutrinos_sum += n.P4()

    # define visible taus for use later on
    taup_vis = taup.P4() - taup_neutrinos_sum
    taun_vis = taun.P4() - taun_neutrinos_sum

#branches = [
#  'taup_npi', 'taup_npizero',
#  'taun_npi', 'taun_npizero',
#  'met_px', 'met_py',
#  'taup_pi1_px', 'taup_pi1_py', 'taup_pi1_pz', 'taup_pi1_e',
#  'taup_pi1_ipx', 'taup_pi1_ipy', 'taup_pi1_ipz',
#  'taup_pi2_px', 'taup_pi2_py', 'taup_pi2_pz', 'taup_pi2_e',
#  'taup_pi3_px', 'taup_pi3_py', 'taup_pi3_pz', 'taup_pi3_e',
#  'taun_pi1_px', 'taun_pi1_py', 'taun_pi1_pz', 'taun_pi1_e',
#  'taun_pi1_ipx', 'taun_pi1_ipy', 'taun_pi1_ipz',
#  'taun_pi2_px', 'taun_pi2_py', 'taun_pi2_pz', 'taun_pi2_e',
#  'taun_pi3_px', 'taun_pi3_py', 'taun_pi3_pz', 'taun_pi3_e',
#  'taup_pizero1_px', 'taup_pizero1_py', 'taup_pizero1_pz', 'taup_pizero1_e',
#  'taun_pizero1_px', 'taun_pizero1_py', 'taun_pizero1_pz', 'taun_pizero1_e',
#  'taup_lep_px', 'taup_lep_py', 'taup_lep_pz', 'taup_lep_e',
#  'taun_lep_px', 'taun_lep_py', 'taun_lep_pz', 'taun_lep_e',
#  'taup_sv_x', 'taup_sv_y', 'taup_sv_z',
#  'taun_sv_x', 'taun_sv_y', 'taun_sv_z',
#  'taup_nu_px', 'taup_nu_py', 'taup_nu_pz',
#  'taun_nu_px', 'taun_nu_py', 'taun_nu_pz'
#]

    branch_vals['taup_npi'][0] = len(taup_pis)
    branch_vals['taup_npizero'][0] = len(taup_gammas)/2 # 2 gammas per pi0
    branch_vals['taun_npi'][0] = len(taun_pis)
    branch_vals['taun_npizero'][0] = len(taun_gammas)/2 # 2 gammas per pi0
    branch_vals['met_px'][0] = genMET.At(0).MET * math.cos(genMET.At(0).Phi)
    branch_vals['met_py'][0] = genMET.At(0).MET * math.sin(genMET.At(0).Phi)
    branch_vals['taup_pizero1_px'][0] = taup_pi0.Px()
    branch_vals['taup_pizero1_py'][0] = taup_pi0.Py()
    branch_vals['taup_pizero1_pz'][0] = taup_pi0.Pz()
    branch_vals['taup_pizero1_e'][0] = taup_pi0.E()
    branch_vals['taun_pizero1_px'][0] = taun_pi0.Px()
    branch_vals['taun_pizero1_py'][0] = taun_pi0.Py()
    branch_vals['taun_pizero1_pz'][0] = taun_pi0.Pz()
    branch_vals['taun_pizero1_e'][0] = taun_pi0.E()
    branch_vals['taup_nu_px'][0] = taup_neutrinos_sum.Px()
    branch_vals['taup_nu_py'][0] = taup_neutrinos_sum.Py()
    branch_vals['taup_nu_pz'][0] = taup_neutrinos_sum.Pz()
    branch_vals['taun_nu_px'][0] = taun_neutrinos_sum.Px()
    branch_vals['taun_nu_py'][0] = taun_neutrinos_sum.Py()
    branch_vals['taun_nu_pz'][0] = taun_neutrinos_sum.Pz()
    branch_vals['taup_lep_px'][0] = taup_leptons[0].P4().Px() if len(taup_leptons) > 0 else 0
    branch_vals['taup_lep_py'][0] = taup_leptons[0].P4().Py() if len(taup_leptons) > 0 else 0
    branch_vals['taup_lep_pz'][0] = taup_leptons[0].P4().Pz() if len(taup_leptons) > 0 else 0
    branch_vals['taup_lep_e'][0] = taup_leptons[0].P4().E() if len(taup_leptons) > 0 else 0
    branch_vals['taun_lep_px'][0] = taun_leptons[0].P4().Px() if len(taun_leptons) > 0 else 0
    branch_vals['taun_lep_py'][0] = taun_leptons[0].P4().Py() if len(taun_leptons) > 0 else 0
    branch_vals['taun_lep_pz'][0] = taun_leptons[0].P4().Pz() if len(taun_leptons) > 0 else 0
    branch_vals['taun_lep_e'][0] = taun_leptons[0].P4().E() if len(taun_leptons) > 0 else 0

    branch_vals['taup_pi1_px'][0] = taup_pis[0].P4().Px() if len(taup_pis) > 0 else 0
    branch_vals['taup_pi1_py'][0] = taup_pis[0].P4().Py() if len(taup_pis) > 0 else 0
    branch_vals['taup_pi1_pz'][0] = taup_pis[0].P4().Pz() if len(taup_pis) > 0 else 0
    branch_vals['taup_pi1_e'][0] = taup_pis[0].P4().E() if len(taup_pis) > 0 else 0



    tree.Fill()
    
