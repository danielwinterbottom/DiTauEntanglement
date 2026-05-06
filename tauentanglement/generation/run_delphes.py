#!/usr/bin/env python3
# IP and SV etc units are in mm

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


def GetMatchingRecoParticle(gen_vis, reco_particles, threshold=0.4):
    matched_particles = []
    for i in range(reco_particles.GetEntries()):
        reco = reco_particles.At(i)
        dR = gen_vis.DeltaR(reco.P4())
        if dR < threshold:
            matched_particles.append(reco)
    return matched_particles

def GetPi0(gammas):
    # get pi0 energy by summing energies of gammas
    # get direction by taking energy weighted eta and phi of gammas
    # set mass to mass of pi0
    E = 0 
    eta = 0
    phi = 0
    for g in gammas:
        E += g.E
        eta += g.E * g.Eta
        phi += g.E * g.Phi
    if E > 0:
        eta /= E
        phi /= E
        pi0 = ROOT.TLorentzVector()
        pi0.SetPtEtaPhiM(E / math.cosh(eta), eta, phi, 0.1349)
        return pi0
    else:
        return ROOT.TLorentzVector()

def RecoStrips(cands_, stripPtThreshold=2.5):

    #sort cands by pT (highest pT first)
    cands_.sort(key=lambda x: x.ET, reverse=True)

    # remove candidates that dont pass pT-eta cuts
    cands_ = [c for c in cands_ if c.ET > 1 and abs(c.Eta) < 2.5]

    strips = []

    while len(cands_) > 0:
        associated = []
        notAssociated = []

        stripVector = cands_[0].P4()
        associated.append(cands_[0])
        cands_ = cands_[1:]

        # loop and associate other candidates
        repeat = True
        while repeat:
            repeat = False
            for c in cands_:
                etaAssociationDistance = min(max(0.2*pow(c.ET, -0.66) + 0.2*pow(stripVector.Pt(), -0.66), 0.05), 0.15)
                phiAssociationDistance = min(max(0.35*pow(c.ET, -0.71) + 0.35*pow(stripVector.Pt(), -0.71), 0.05), 0.30)
                if abs(c.Eta - stripVector.Eta()) < etaAssociationDistance and abs(ROOT.Math.VectorUtil.DeltaPhi(c.P4(), stripVector)) < phiAssociationDistance:
                    stripVector += c.P4()
                    associated.append(c)
                    repeat = True
                else:
                    notAssociated.append(c)

            # swap the candidate vector with the non associated vector
            cands_ = notAssociated
            notAssociated = []

        strip = GetPi0(associated)
        if strip.Pt() >= stripPtThreshold:
            strips.append((strip, associated))

    return strips

def CheckWithinSigCone(tau_cand):
    tau_vis = tau_cand[0]
    cands = tau_cand[1]+tau_cand[2]
    Rsig = max(min(3.0/tau_vis.Pt(),0.1),0.05)

    for c in cands:
        if isinstance(c, ROOT.TLorentzVector):
            c_p4 = c
        else:
            c_p4 = c.P4()
        if tau_vis.DeltaR(c_p4) > Rsig:
            return False
    return True

def GetTauCands(tracks, strips, incDM2=True):
    #apply pT and eta cuts to tracks
    tracks_ = [t for t in tracks if t.PT > 0.5 and abs(t.Eta) < 2.5]

    # first make all combination of 1 track and 0,1,2 strips
    tau_cands = []
    for t in tracks_:
        tau_vis = t.P4()
        tau_cands.append([tau_vis, [t], []]) # 1 track, 0 strips
        #TODO: might need to add the additional delta_mass part to the min and max masses below
        for s in strips:
            mass_2body = (t.P4() + s[0]).M()
            pT_2body = (t.P4() + s[0]).Pt()
            mass_2body_min = 0.3
            mass_2body_max = max(min(1.3 * math.sqrt(pT_2body/100), 4.2), 1.3)
            if mass_2body > mass_2body_min and mass_2body < mass_2body_max:
                tau_vis = t.P4() + s[0]
                tau_cands.append([tau_vis, [t], [s[0]]]) # 1 track, 1 strip
            if incDM2:
                for s2 in strips:
                    if s2 != s:
                        mass_3body = (t.P4() + s[0] + s2[0]).M()
                        pT_3body = (t.P4() + s[0] + s2[0]).Pt()
                        mass_3body_min = 0.4
                        mass_3body_max = max(min(1.2 * math.sqrt(pT_3body/100), 4.0), 1.2)
                        if mass_3body > mass_3body_min and mass_3body < mass_3body_max:
                            tau_vis = t.P4() + s[0] + s2[0]
                            tau_cands.append([tau_vis, [t], [s[0], s2[0]]]) # 1 track, 2 strips

    # filter any tau_cands not passing signal cone requirement
    tau_cands = [c for c in tau_cands if CheckWithinSigCone(c)]

    #sort tau_cands by pT
    tau_cands.sort(key=lambda x: x[0].Pt(), reverse=True)

    return tau_cands


    

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
photons = reader.UseBranch("EFlowPhoton")
eflowtracks = reader.UseBranch("EFlowTrack")
tracks = reader.UseBranch("Track")

#setup output tree

branches = [
  'taup_npi', 'taup_npizero',
  'taun_npi', 'taun_npizero',
  'met_px', 'met_py',
  'taup_vis_pT', 'taup_vis_eta',
  'taun_vis_pT', 'taun_vis_eta',
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
  'taun_nu_px', 'taun_nu_py', 'taun_nu_pz',
]

fout = ROOT.TFile(args.output,'RECREATE')
tree = ROOT.TTree('tree','')

branch_vals = {}
for b in branches:
    branch_vals[b] = array('f',[0])
    tree.Branch(b,  branch_vals[b],  '%s/F' % b)
    # add a reco branch for everything except the nus
    if not b.startswith('taup_nu') and not b.startswith('taun_nu'):
        branch_vals['reco_' + b] = array('f',[0])
        tree.Branch('reco_' + b,  branch_vals['reco_' + b],  'reco_%s/F' % b)


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

    branch_vals['taup_npi'][0] = len(taup_pis)
    branch_vals['taup_npizero'][0] = len(taup_gammas)//2 # 2 gammas per pi0 - round down if odd number of gammas
    branch_vals['taun_npi'][0] = len(taun_pis)
    branch_vals['taun_npizero'][0] = len(taun_gammas)//2 # 2 gammas per pi0

    branch_vals['taup_vis_pT'][0] = taup_vis.Pt()
    branch_vals['taup_vis_eta'][0] = taup_vis.Eta()
    branch_vals['taun_vis_pT'][0] = taun_vis.Pt()
    branch_vals['taun_vis_eta'][0] = taun_vis.Eta()

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
    branch_vals['taun_pi1_px'][0] = taun_pis[0].P4().Px() if len(taun_pis) > 0 else 0
    branch_vals['taun_pi1_py'][0] = taun_pis[0].P4().Py() if len(taun_pis) > 0 else 0
    branch_vals['taun_pi1_pz'][0] = taun_pis[0].P4().Pz() if len(taun_pis) > 0 else 0
    branch_vals['taun_pi1_e'][0] = taun_pis[0].P4().E() if len(taun_pis) > 0 else 0

    taup_ip = get_impact_parameter(taup_pis[0]) if len(taup_pis) > 0 else (ROOT.TVector3(0,0,0), 0, 0)
    branch_vals['taup_pi1_ipx'][0] = taup_ip[0].X()
    branch_vals['taup_pi1_ipy'][0] = taup_ip[0].Y()
    branch_vals['taup_pi1_ipz'][0] = taup_ip[0].Z()

    taun_ip = get_impact_parameter(taun_pis[0]) if len(taun_pis) > 0 else (ROOT.TVector3(0,0,0), 0, 0)
    branch_vals['taun_pi1_ipx'][0] = taun_ip[0].X()
    branch_vals['taun_pi1_ipy'][0] = taun_ip[0].Y()
    branch_vals['taun_pi1_ipz'][0] = taun_ip[0].Z()

    # get SVs but only if 3-prongs 
    taup_sv = ROOT.TVector3(taup.X, taup.Y, taup.Z) if len(taup_pis) > 2 else ROOT.TVector3(0,0,0)
    branch_vals['taup_sv_x'][0] = taup_sv.X()
    branch_vals['taup_sv_y'][0] = taup_sv.Y()
    branch_vals['taup_sv_z'][0] = taup_sv.Z()

    taun_sv = ROOT.TVector3(taun.X, taun.Y, taun.Z) if len(taun_pis) > 2 else ROOT.TVector3(0,0,0)
    branch_vals['taun_sv_x'][0] = taun_sv.X()
    branch_vals['taun_sv_y'][0] = taun_sv.Y()
    branch_vals['taun_sv_z'][0] = taun_sv.Z()

    # get smeared "reco" quantities
    # first we dR match reco particles to gen visible taus
    taup_reco_track_matches = GetMatchingRecoParticle(taup_vis, eflowtracks, threshold=0.5)
    taup_reco_photon_matches = GetMatchingRecoParticle(taup_vis, photons, threshold=0.5)
    taun_reco_track_matches = GetMatchingRecoParticle(taun_vis, eflowtracks, threshold=0.5) # add in the track matches as well (these will have some smearing but also have charge and impact parameter info which is useful for the study)
    taun_reco_photon_matches = GetMatchingRecoParticle(taun_vis, photons, threshold=0.5)

    taup_strips = RecoStrips(taup_reco_photon_matches)
    # sort by pT of the strip
    taup_strips.sort(key=lambda x: x[0].Pt(), reverse=True)

    taun_strips = RecoStrips(taun_reco_photon_matches)
    # sort by pT of the strip
    taun_strips.sort(key=lambda x: x[0].Pt(), reverse=True)

    taup_cands = GetTauCands(taup_reco_track_matches, taup_strips, incDM2=False) # Note: not allowing DM2 taus for now to match CMS reco where DM2 is very rare 
    reco_taup_vis = taup_cands[0][0] if len(taup_cands) > 0 else ROOT.TLorentzVector()
    taun_cands = GetTauCands(taun_reco_track_matches, taun_strips, incDM2=False)
    reco_taun_vis = taun_cands[0][0] if len(taun_cands) > 0 else ROOT.TLorentzVector()

    branch_vals['reco_taup_vis_pT'][0] = reco_taup_vis.Pt()
    branch_vals['reco_taup_vis_eta'][0] = reco_taup_vis.Eta()
    branch_vals['reco_taun_vis_pT'][0] = reco_taun_vis.Pt()
    branch_vals['reco_taun_vis_eta'][0] = reco_taun_vis.Eta()

    # TODO: implement 3-prongs at some point - the below assumes only dm=0 and dm=1 are present
    if len(taup_cands) > 0:
        branch_vals['reco_taup_npi'][0] = len(taup_cands[0][1])
        branch_vals['reco_taup_npizero'][0] = len(taup_cands[0][2])
        branch_vals['reco_taup_pi1_px'][0] = taup_cands[0][1][0].P4().Px()
        branch_vals['reco_taup_pi1_py'][0] = taup_cands[0][1][0].P4().Py()
        branch_vals['reco_taup_pi1_pz'][0] = taup_cands[0][1][0].P4().Pz()
        branch_vals['reco_taup_pi1_e'][0] = taup_cands[0][1][0].P4().E()

        branch_vals['reco_taup_pi1_ipx'][0] = taup_cands[0][1][0].Xd
        branch_vals['reco_taup_pi1_ipy'][0] = taup_cands[0][1][0].Yd
        branch_vals['reco_taup_pi1_ipz'][0] = taup_cands[0][1][0].Zd

        if len(taup_cands[0][2]) > 0:
            branch_vals['reco_taup_pizero1_px'][0] = taup_cands[0][2][0].Px()
            branch_vals['reco_taup_pizero1_py'][0] = taup_cands[0][2][0].Py()
            branch_vals['reco_taup_pizero1_pz'][0] = taup_cands[0][2][0].Pz()
            branch_vals['reco_taup_pizero1_e'][0] = taup_cands[0][2][0].E()

    if len(taun_cands) > 0:
        branch_vals['reco_taun_npi'][0] = len(taun_cands[0][1])
        branch_vals['reco_taun_npizero'][0] = len(taun_cands[0][2])
        branch_vals['reco_taun_pi1_px'][0] = taun_cands[0][1][0].P4().Px()
        branch_vals['reco_taun_pi1_py'][0] = taun_cands[0][1][0].P4().Py()
        branch_vals['reco_taun_pi1_pz'][0] = taun_cands[0][1][0].P4().Pz()
        branch_vals['reco_taun_pi1_e'][0] = taun_cands[0][1][0].P4().E()

        branch_vals['reco_taun_pi1_ipx'][0] = taun_cands[0][1][0].Xd
        branch_vals['reco_taun_pi1_ipy'][0] = taun_cands[0][1][0].Yd
        branch_vals['reco_taun_pi1_ipz'][0] = taun_cands[0][1][0].Zd

        if len(taun_cands[0][2]) > 0:
            branch_vals['reco_taun_pizero1_px'][0] = taun_cands[0][2][0].Px()
            branch_vals['reco_taun_pizero1_py'][0] = taun_cands[0][2][0].Py()
            branch_vals['reco_taun_pizero1_pz'][0] = taun_cands[0][2][0].Pz()
            branch_vals['reco_taun_pizero1_e'][0] = taun_cands[0][2][0].E()

    #TODO: add SVs for other pions for 3-prongs, and leptons for leptonic modes 

    #store reco MET
    branch_vals['reco_met_px'][0] = MET.At(0).MET * math.cos(MET.At(0).Phi)
    branch_vals['reco_met_py'][0] = MET.At(0).MET * math.sin(MET.At(0).Phi)

    tree.Fill()

fout.Write()
fout.Close()
    
