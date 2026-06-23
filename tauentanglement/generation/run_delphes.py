#!/usr/bin/env python3
# IP and SV etc units are in mm

import sys
import ROOT
import math
import argparse
from array import array
import numpy as np
from tauentanglement.utils.ReconstructTaus import FindDMin_Point, FindVertexLSQ

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
                #if not other_mother.PID in mother_ids_to_exclude:
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

def get_impact_parameter(p, pv_3vec=ROOT.TVector3(0, 0, 0), reco_track=False):

    if reco_track:
        p_vtx_vec3 = ROOT.TVector3(p.Xd, p.Yd, p.Zd) # note this isn't the vertex but another point on the track but it should work the same
    else: 
        p_vtx_vec3 = ROOT.TVector3(p.X, p.Y, p.Z)
    # set up direction using PT, Eta, Phi
    px = p.PT * math.cos(p.Phi)
    py = p.PT * math.sin(p.Phi)
    pz = p.PT * math.sinh(p.Eta)
    p_dir_vec3 = ROOT.TVector3(px, py, pz).Unit()

    impact_point_vec3 = FindDMin_Point(p_vtx_vec3, p_dir_vec3, pv_3vec)

    return impact_point_vec3

def get_pseudo_impact_parameter(p_dir_vec3, sv_vec3, pv_3vec=ROOT.TVector3(0, 0, 0)):
    impact_point_vec3 = FindDMin_Point(sv_vec3, p_dir_vec3.Unit(), pv_3vec)
    return impact_point_vec3


def get_3d_point_from_phi_d0_dz(phi, d0, dz):
    #d0 = x0*sinphi -y0*cosphi
    dx = d0 * math.sin(phi)
    dy = -d0 * math.cos(phi)

    return ROOT.TVector3(dx, dy, dz)

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
    if len(tau_cand) > 3:
        cands += tau_cand[3] # add in the leptons if they exist

    Rsig = max(min(3.0/tau_vis.Pt(),0.1),0.05)

    for c in cands:
        if isinstance(c, ROOT.TLorentzVector):
            c_p4 = c
        else:
            c_p4 = c.P4()
        if tau_vis.DeltaR(c_p4) > Rsig:
            return False
    return True

def GetTauCands(tracks, strips, incDM2=True, match_charge=None):

    tau_cands = []

    #apply pT and eta cuts to tracks
    tracks_ = [] # non muon / electron tracks
    muon_tracks = [] # tracks identified as electrons or muons
    electron_tracks = []
    for t in tracks:
        if t.PT > 0.5 and abs(t.Eta) < 2.5:
            if abs(t.PID) == 11: # remove tracks identified as electrons
                electron_tracks.append(t)
            elif abs(t.PID) == 13: # remove tracks identified as muons
                muon_tracks.append(t)
            elif abs(t.PID) not in [11, 13]: # remove tracks identified as electrons or muons
                tracks_.append(t)

    # all leptons become their own tau candidate
    for p in electron_tracks+muon_tracks:
        tau_vis = p.P4()
        tau_cands.append([tau_vis, [], [], [p] ]) # 1 track, 0 strips

    # first make all combination of 1 track and 0,1,2 strips
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
    
    # make all 2 track combinations
    for i in range(len(tracks_)):
        for j in range(i+1, len(tracks_)):
            t1 = tracks_[i]
            t2 = tracks_[j]
            tau_vis = t1.P4() + t2.P4()
            mass_2body = tau_vis.M()
            mass_2body_max = 1.2
            if mass_2body < mass_2body_max:
                tau_cands.append([tau_vis, [t1, t2], []]) # 2 tracks, 0 strips


    # make all combinations of 3 tracks and 0, 1 strips
    # first find all combination of 3 tracks with total charge = +/- 1
    for i in range(len(tracks_)):
        for j in range(i+1, len(tracks_)):
            for k in range(j+1, len(tracks_)):
                t1 = tracks_[i]
                t2 = tracks_[j]
                t3 = tracks_[k]
                if abs(t1.Charge + t2.Charge + t3.Charge) != 1: continue
                tau_vis = t1.P4() + t2.P4() + t3.P4()
                mass_3body_min =  0.8
                mass_3body_max =  1.5
                mass_3body = tau_vis.M()
                if mass_3body > mass_3body_min and mass_3body < mass_3body_max:
                    tau_cands.append([tau_vis, [t1, t2, t3], []]) # 3 tracks, 0 strips
                # now find combination with 1 strip
                for s in strips:
                    mass_4body = (t1.P4() + t2.P4() + t3.P4() + s[0]).M()
                    pT_4body = (t1.P4() + t2.P4() + t3.P4() + s[0]).Pt()
                    mass_4body_min = 0.9
                    mass_4body_max = 1.6
                    if mass_4body > mass_4body_min and mass_4body < mass_4body_max:
                        tau_vis = t1.P4() + t2.P4() + t3.P4() + s[0]
                        tau_cands.append([tau_vis, [t1, t2, t3], [s[0]]]) # 3 tracks, 1 strip

    # filter any tau_cands not passing signal cone requirement
    tau_cands = [c for c in tau_cands if CheckWithinSigCone(c)]

    #sort tau_cands by pT
    tau_cands.sort(key=lambda x: x[0].Pt(), reverse=True)
    # if a muon or electron candidate exists, put these at the front of the list (they will be used preferentially in the analysis)
    tau_cands.sort(key=lambda x: len(x) > 3, reverse=True)

    # if best tau has only 2 prongs then we remove the event
    if len(tau_cands) > 0:
        best_cand = tau_cands[0]
        if len(best_cand[1]) == 2:
            tau_cands = [] # if the best candidate has only 2 prongs, then reject all candidates

    # if match_charge then require the best candidate to match the charge of the tau
    if match_charge is not None and len(tau_cands) > 0:
        best_cand = tau_cands[0]
        charge = 0
        for t in best_cand[1]:
            charge += t.Charge
        if len(best_cand) > 3: # add in the leptons if they exist
            for l in best_cand[3]:
                charge += l.Charge
        if charge != match_charge:
            tau_cands = [] # if the best candidate doesn't match the charge, then reject all candidates

    return tau_cands

def GetTauCands(tracks, strips, incDM2=True, match_charge=None):

    tau_cands = []

    #apply pT and eta cuts to tracks
    tracks_ = [] # non muon / electron tracks
    muon_tracks = [] # tracks identified as electrons or muons
    electron_tracks = []
    for t in tracks:
        if t.PT > 0.5 and abs(t.Eta) < 2.5:
            if abs(t.PID) == 11: # remove tracks identified as electrons
                electron_tracks.append(t)
            elif abs(t.PID) == 13: # remove tracks identified as muons
                muon_tracks.append(t)
            elif abs(t.PID) not in [11, 13]: # remove tracks identified as electrons or muons
                tracks_.append(t)

    # all leptons become their own tau candidate
    for p in electron_tracks+muon_tracks:
        tau_vis = p.P4()
        tau_cands.append([tau_vis, [], [], [p] ]) # 1 track, 0 strips

    # first make all combination of 1 track and 0,1,2 strips
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
    
    # make all 2 track combinations
    for i in range(len(tracks_)):
        for j in range(i+1, len(tracks_)):
            t1 = tracks_[i]
            t2 = tracks_[j]
            tau_vis = t1.P4() + t2.P4()
            mass_2body = tau_vis.M()
            mass_2body_max = 1.2
            if mass_2body < mass_2body_max:
                tau_cands.append([tau_vis, [t1, t2], []]) # 2 tracks, 0 strips


    # make all combinations of 3 tracks and 0, 1 strips
    # first find all combination of 3 tracks with total charge = +/- 1
    for i in range(len(tracks_)):
        for j in range(i+1, len(tracks_)):
            for k in range(j+1, len(tracks_)):
                t1 = tracks_[i]
                t2 = tracks_[j]
                t3 = tracks_[k]
                if abs(t1.Charge + t2.Charge + t3.Charge) != 1: continue
                tau_vis = t1.P4() + t2.P4() + t3.P4()
                mass_3body_min =  0.8
                mass_3body_max =  1.5
                mass_3body = tau_vis.M()
                if mass_3body > mass_3body_min and mass_3body < mass_3body_max:
                    tau_cands.append([tau_vis, [t1, t2, t3], []]) # 3 tracks, 0 strips
                # now find combination with 1 strip
                for s in strips:
                    mass_4body = (t1.P4() + t2.P4() + t3.P4() + s[0]).M()
                    pT_4body = (t1.P4() + t2.P4() + t3.P4() + s[0]).Pt()
                    mass_4body_min = 0.9
                    mass_4body_max = 1.6
                    if mass_4body > mass_4body_min and mass_4body < mass_4body_max:
                        tau_vis = t1.P4() + t2.P4() + t3.P4() + s[0]
                        tau_cands.append([tau_vis, [t1, t2, t3], [s[0]]]) # 3 tracks, 1 strip

    # filter any tau_cands not passing signal cone requirement
    tau_cands = [c for c in tau_cands if CheckWithinSigCone(c)]

    #sort tau_cands by pT
    tau_cands.sort(key=lambda x: x[0].Pt(), reverse=True)
    # if a muon or electron candidate exists, put these at the front of the list (they will be used preferentially in the analysis)
    tau_cands.sort(key=lambda x: len(x) > 3, reverse=True)

    # if best tau has only 2 prongs then we remove the event
    if len(tau_cands) > 0:
        best_cand = tau_cands[0]
        if len(best_cand[1]) == 2:
            tau_cands = [] # if the best candidate has only 2 prongs, then reject all candidates

    # if match_charge then require the best candidate to match the charge of the tau
    if match_charge is not None and len(tau_cands) > 0:
        best_cand = tau_cands[0]
        charge = 0
        for t in best_cand[1]:
            charge += t.Charge
        if len(best_cand) > 3: # add in the leptons if they exist
            for l in best_cand[3]:
                charge += l.Charge
        if charge != match_charge:
            tau_cands = [] # if the best candidate doesn't match the charge, then reject all candidates

    return tau_cands

class ResolutionGraph:
    def __init__(self, path, unit_conversion=1.0):
        pts = []
        ress = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # skip if line contains any text
                if any(c.isalpha() for c in line):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    pt = float(parts[0])
                    res = float(parts[1])
                except ValueError:
                    continue

                pts.append(pt)
                ress.append(res * unit_conversion)

        self.log_pt = np.log(np.asarray(pts))
        self.res = np.asarray(ress)

    def eval(self, pt):
        log_pt = np.log(pt)

        # clamp to avoid extrapolation
        log_pt = np.clip(log_pt, self.log_pt[0], self.log_pt[-1])

        return np.interp(log_pt, self.log_pt, self.res)

class TrackAngularSmearer:
    def __init__(
        self,
        base="tauentanglement/generation/configs/track_resolutions/",
        seed=None,
    ):
        self.rng = np.random.default_rng(seed)

        self.phi_res = {
            1: ResolutionGraph(base + "phi_resolution_eta_0_to_0p9.txt", unit_conversion=1e-3), # convert from mrad to rad
            2: ResolutionGraph(base + "phi_resolution_eta_0p9_to_1p4.txt", unit_conversion=1e-3),
            3: ResolutionGraph(base + "phi_resolution_eta_1p4_to_2p5.txt", unit_conversion=1e-3),
        }

        self.cottheta_res = {
            1: ResolutionGraph(base + "cottheta_resolution_eta_0_to_0p9.txt", unit_conversion=1e-3),
            2: ResolutionGraph(base + "cottheta_resolution_eta_0p9_to_1p4.txt", unit_conversion=1e-3),
            3: ResolutionGraph(base + "cottheta_resolution_eta_1p4_to_2p5.txt", unit_conversion=1e-3),
        }

        self.d0_res = {
            1: ResolutionGraph(base + "d0_resolution_eta_0_to_0p9.txt", unit_conversion=1e-3), # convert from microns to mm
            2: ResolutionGraph(base + "d0_resolution_eta_0p9_to_1p4.txt", unit_conversion=1e-3),
            3: ResolutionGraph(base + "d0_resolution_eta_1p4_to_2p5.txt", unit_conversion=1e-3),
        }

        self.dz_res = {
            1: ResolutionGraph(base + "z0_resolution_eta_0_to_0p9.txt", unit_conversion=1e-3),
            2: ResolutionGraph(base + "z0_resolution_eta_0p9_to_1p4.txt", unit_conversion=1e-3),
            3: ResolutionGraph(base + "z0_resolution_eta_1p4_to_2p5.txt", unit_conversion=1e-3),
        }

    def get_eta_bin(self, eta):
        abs_eta = abs(eta)

        if abs_eta < 0.9:
            return 1
        elif abs_eta < 1.4:
            return 2
        else:
            return 3

    def wrap_phi(self, phi):
        return np.arctan2(np.sin(phi), np.cos(phi))

    def get_phi_resolution(self, eta, pt):
        eta_bin = self.get_eta_bin(eta)
        return self.phi_res[eta_bin].eval(pt)

    def get_cottheta_resolution(self, eta, pt):
        eta_bin = self.get_eta_bin(eta)
        return self.cottheta_res[eta_bin].eval(pt)

    def get_d0_resolution(self, eta, pt):
        eta_bin = self.get_eta_bin(eta)
        return self.d0_res[eta_bin].eval(pt)

    def get_dz_resolution(self, eta, pt):
        eta_bin = self.get_eta_bin(eta)
        return self.dz_res[eta_bin].eval(pt)

    def smear_eta_phi(self, eta, phi, pt):
        sigma_phi = self.get_phi_resolution(eta, pt)
        sigma_cottheta = self.get_cottheta_resolution(eta, pt)

        phi_smeared = phi + self.rng.normal(0.0, sigma_phi)
        phi_smeared = self.wrap_phi(phi_smeared)

        cottheta = np.sinh(eta)
        cottheta_smeared = cottheta + self.rng.normal(0.0, sigma_cottheta)
        eta_smeared = np.arcsinh(cottheta_smeared)

        return eta_smeared, phi_smeared

    def smear_tlorentzvector_angles(self, vec):
        pt = vec.Pt()
        eta = vec.Eta()
        phi = vec.Phi()

        p = vec.P()
        energy = vec.E()

        eta_smeared, phi_smeared = self.smear_eta_phi(eta, phi, pt)

        pt_smeared = p / np.cosh(eta_smeared)

        px = pt_smeared * np.cos(phi_smeared)
        py = pt_smeared * np.sin(phi_smeared)
        pz = p * np.tanh(eta_smeared)

        vec_smeared = ROOT.TLorentzVector()
        vec_smeared.SetPxPyPzE(px, py, pz, energy)

        return vec_smeared

    def smear_d0_dz(self, d0, dz, eta, pt):
        sigma_d0 = self.get_d0_resolution(eta, pt)
        sigma_dz = self.get_dz_resolution(eta, pt)

        d0_smeared = d0 + self.rng.normal(0.0, sigma_d0)
        dz_smeared = dz + self.rng.normal(0.0, sigma_dz)

        return d0_smeared, dz_smeared

def smear_PV(pv_3vec):
    # CMS PV resolutions from here: https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-20-006/CMS-HIG-20-006_Figure-aux_026.png
    sigma_x = 0.005
    sigma_y = 0.005
    sigma_z = 0.029

    smeared_pv_3vec = ROOT.TVector3(
        pv_3vec.X() + np.random.normal(0.0, sigma_x),
        pv_3vec.Y() + np.random.normal(0.0, sigma_y),
        pv_3vec.Z() + np.random.normal(0.0, sigma_z)
    )

    return smeared_pv_3vec

def SortPions(pions, tau_charge):
    # for 3 prong taus sort the pions based on charge and pT
    # the first pion is the highest pT pion of opposite charge to the tau, the second pion is the highest pT pion of same charge as the tau, and the third pion is the lowest pT pion of same charge as the tau
    
    # first split by charge
    same_charge = [pion for pion in pions if pion.Charge == tau_charge]
    opposite_charge = [pion for pion in pions if pion.Charge != tau_charge]

    # sort each group by pT
    same_charge.sort(key=lambda x: x.PT, reverse=True)
    opposite_charge.sort(key=lambda x: x.PT, reverse=True)

    # return the sorted pions
    return opposite_charge + same_charge


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

# for track angular smearing:
smearer = TrackAngularSmearer(seed=12345)

#setup output tree

branches = [
  'taup_npi', 'taup_npizero',
  'taun_npi', 'taun_npizero',
  'taup_nele', 'taun_nele',
  'taup_nmu', 'taun_nmu',
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
  'taup_lep_ipx', 'taup_lep_ipy', 'taup_lep_ipz',
  'taun_lep_ipx', 'taun_lep_ipy', 'taun_lep_ipz',
  'taup_charged_px', 'taup_charged_py', 'taup_charged_pz', 'taup_charged_e',
  'taun_charged_px', 'taun_charged_py', 'taun_charged_pz', 'taun_charged_e',
  'taup_charged_ipx', 'taup_charged_ipy', 'taup_charged_ipz',
  'taun_charged_ipx', 'taun_charged_ipy', 'taun_charged_ipz',
  'taup_sv_x', 'taup_sv_y', 'taup_sv_z',
  'taun_sv_x', 'taun_sv_y', 'taun_sv_z',
  'taup_nu_px', 'taup_nu_py', 'taup_nu_pz', 'taup_nu_m',
  'taun_nu_px', 'taun_nu_py', 'taun_nu_pz', 'taun_nu_m',
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
    for b in branch_vals:
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

    #sort the pions if more than 1
    if len(taup_pis) > 1:
        taup_pis = SortPions(taup_pis, tau_charge=1)
    if len(taun_pis) > 1:
        taun_pis = SortPions(taun_pis, tau_charge=-1)

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

    taup_charged_sum = ROOT.TLorentzVector()
    for c in taup_pis+taup_leptons:
        taup_charged_sum += c.P4()
    taun_charged_sum = ROOT.TLorentzVector()
    for c in taun_pis+taun_leptons:
        taun_charged_sum += c.P4()

    # define visible taus for use later on
    taup_vis = taup.P4() - taup_neutrinos_sum
    taun_vis = taun.P4() - taun_neutrinos_sum

    branch_vals['taup_npi'][0] = len(taup_pis)
    branch_vals['taup_npizero'][0] = len(taup_gammas)//2 # 2 gammas per pi0 - round down if odd number of gammas
    branch_vals['taun_npi'][0] = len(taun_pis)
    branch_vals['taun_npizero'][0] = len(taun_gammas)//2 # 2 gammas per pi0

    branch_vals['taup_nele'][0] = len([l for l in taup_leptons if abs(l.PID) == 11])
    branch_vals['taun_nele'][0] = len([l for l in taun_leptons if abs(l.PID) == 11])
    branch_vals['taup_nmu'][0] = len([l for l in taup_leptons if abs(l.PID) == 13])
    branch_vals['taun_nmu'][0] = len([l for l in taun_leptons if abs(l.PID) == 13])

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
    branch_vals['taup_nu_m'][0] = max(taup_neutrinos_sum.M(),0) if len(taup_neutrinos) > 1 else 0 # set mass to zero if no neutrinos to avoid weird edge cases where we get a small non zero mass from numerical precision
    branch_vals['taun_nu_px'][0] = taun_neutrinos_sum.Px()
    branch_vals['taun_nu_py'][0] = taun_neutrinos_sum.Py()
    branch_vals['taun_nu_pz'][0] = taun_neutrinos_sum.Pz()
    branch_vals['taun_nu_m'][0] = max(taun_neutrinos_sum.M(), 0) if len(taun_neutrinos) > 1 else 0 # set mass to zero if no neutrinos to avoid weird edge cases where we get a small non zero mass from numerical precision

    branch_vals['taup_lep_px'][0] = taup_leptons[0].P4().Px() if len(taup_leptons) > 0 else 0
    branch_vals['taup_lep_py'][0] = taup_leptons[0].P4().Py() if len(taup_leptons) > 0 else 0
    branch_vals['taup_lep_pz'][0] = taup_leptons[0].P4().Pz() if len(taup_leptons) > 0 else 0
    branch_vals['taup_lep_e'][0] = taup_leptons[0].P4().E() if len(taup_leptons) > 0 else 0
    branch_vals['taun_lep_px'][0] = taun_leptons[0].P4().Px() if len(taun_leptons) > 0 else 0
    branch_vals['taun_lep_py'][0] = taun_leptons[0].P4().Py() if len(taun_leptons) > 0 else 0
    branch_vals['taun_lep_pz'][0] = taun_leptons[0].P4().Pz() if len(taun_leptons) > 0 else 0
    branch_vals['taun_lep_e'][0] = taun_leptons[0].P4().E() if len(taun_leptons) > 0 else 0

    # store ips for leptons
    taup_lep_ip = get_impact_parameter(taup_leptons[0]) if len(taup_leptons) > 0 else ROOT.TVector3(0,0,0)
    branch_vals['taup_lep_ipx'][0] = taup_lep_ip.X()
    branch_vals['taup_lep_ipy'][0] = taup_lep_ip.Y()
    branch_vals['taup_lep_ipz'][0] = taup_lep_ip.Z()
    taun_lep_ip = get_impact_parameter(taun_leptons[0]) if len(taun_leptons) > 0 else ROOT.TVector3(0,0,0)
    branch_vals['taun_lep_ipx'][0] = taun_lep_ip.X()
    branch_vals['taun_lep_ipy'][0] = taun_lep_ip.Y()
    branch_vals['taun_lep_ipz'][0] = taun_lep_ip.Z()

    branch_vals['taup_pi1_px'][0] = taup_pis[0].P4().Px() if len(taup_pis) > 0 else 0
    branch_vals['taup_pi1_py'][0] = taup_pis[0].P4().Py() if len(taup_pis) > 0 else 0
    branch_vals['taup_pi1_pz'][0] = taup_pis[0].P4().Pz() if len(taup_pis) > 0 else 0
    branch_vals['taup_pi1_e'][0] = taup_pis[0].P4().E() if len(taup_pis) > 0 else 0
    branch_vals['taun_pi1_px'][0] = taun_pis[0].P4().Px() if len(taun_pis) > 0 else 0
    branch_vals['taun_pi1_py'][0] = taun_pis[0].P4().Py() if len(taun_pis) > 0 else 0
    branch_vals['taun_pi1_pz'][0] = taun_pis[0].P4().Pz() if len(taun_pis) > 0 else 0
    branch_vals['taun_pi1_e'][0] = taun_pis[0].P4().E() if len(taun_pis) > 0 else 0

    taup_ip = get_impact_parameter(taup_pis[0]) if len(taup_pis) > 0 else ROOT.TVector3(0,0,0)
    branch_vals['taup_pi1_ipx'][0] = taup_ip.X()
    branch_vals['taup_pi1_ipy'][0] = taup_ip.Y()
    branch_vals['taup_pi1_ipz'][0] = taup_ip.Z()

    taun_ip = get_impact_parameter(taun_pis[0]) if len(taun_pis) > 0 else ROOT.TVector3(0,0,0)
    branch_vals['taun_pi1_ipx'][0] = taun_ip.X()
    branch_vals['taun_pi1_ipy'][0] = taun_ip.Y()
    branch_vals['taun_pi1_ipz'][0] = taun_ip.Z()

    if len(taup_pis) > 1:
        branch_vals['taup_pi2_px'][0] = taup_pis[1].P4().Px()
        branch_vals['taup_pi2_py'][0] = taup_pis[1].P4().Py()
        branch_vals['taup_pi2_pz'][0] = taup_pis[1].P4().Pz()
        branch_vals['taup_pi2_e'][0] = taup_pis[1].P4().E()
    if len(taup_pis) > 2:
        branch_vals['taup_pi3_px'][0] = taup_pis[2].P4().Px()
        branch_vals['taup_pi3_py'][0] = taup_pis[2].P4().Py()
        branch_vals['taup_pi3_pz'][0] = taup_pis[2].P4().Pz()
        branch_vals['taup_pi3_e'][0] = taup_pis[2].P4().E()

    if len(taun_pis) > 1:
        branch_vals['taun_pi2_px'][0] = taun_pis[1].P4().Px()
        branch_vals['taun_pi2_py'][0] = taun_pis[1].P4().Py()
        branch_vals['taun_pi2_pz'][0] = taun_pis[1].P4().Pz()
        branch_vals['taun_pi2_e'][0] = taun_pis[1].P4().E()
    if len(taun_pis) > 2:
        branch_vals['taun_pi3_px'][0] = taun_pis[2].P4().Px()
        branch_vals['taun_pi3_py'][0] = taun_pis[2].P4().Py()
        branch_vals['taun_pi3_pz'][0] = taun_pis[2].P4().Pz()
        branch_vals['taun_pi3_e'][0] = taun_pis[2].P4().E()

    branch_vals['taup_charged_px'][0] = taup_charged_sum.Px()
    branch_vals['taup_charged_py'][0] = taup_charged_sum.Py()
    branch_vals['taup_charged_pz'][0] = taup_charged_sum.Pz()
    branch_vals['taup_charged_e'][0] = taup_charged_sum.E()
    branch_vals['taun_charged_px'][0] = taun_charged_sum.Px()
    branch_vals['taun_charged_py'][0] = taun_charged_sum.Py()
    branch_vals['taun_charged_pz'][0] = taun_charged_sum.Pz()
    branch_vals['taun_charged_e'][0] = taun_charged_sum.E()

    # get SVs but only if 3-prongs 
    taup_sv = ROOT.TVector3(taup_pis[0].X, taup_pis[0].Y, taup_pis[0].Z) if len(taup_pis) > 2 else ROOT.TVector3(0,0,0)
    branch_vals['taup_sv_x'][0] = taup_sv.X()
    branch_vals['taup_sv_y'][0] = taup_sv.Y()
    branch_vals['taup_sv_z'][0] = taup_sv.Z()

    taun_sv = ROOT.TVector3(taun_pis[0].X, taun_pis[0].Y, taun_pis[0].Z) if len(taun_pis) > 2 else ROOT.TVector3(0,0,0)
    branch_vals['taun_sv_x'][0] = taun_sv.X()
    branch_vals['taun_sv_y'][0] = taun_sv.Y()
    branch_vals['taun_sv_z'][0] = taun_sv.Z()

    # for charged ips. if tau is a 1-prong pion decay than set to pi1 ip
    # if tau is a lepton set to lepton ip
    # if tau is a 3-prong decay then estimate the ip using the sv and the pseudo-track from summing the charged decay products
    if len(taup_pis) == 1:
        branch_vals['taup_charged_ipx'][0] = taup_ip.X()
        branch_vals['taup_charged_ipy'][0] = taup_ip.Y()
        branch_vals['taup_charged_ipz'][0] = taup_ip.Z()
    elif len(taup_leptons) == 1:
        branch_vals['taup_charged_ipx'][0] = taup_lep_ip.X()
        branch_vals['taup_charged_ipy'][0] = taup_lep_ip.Y()
        branch_vals['taup_charged_ipz'][0] = taup_lep_ip.Z()
    elif len(taup_pis) > 1:
        taup_charged_ip = get_pseudo_impact_parameter(taup_charged_sum.Vect(), taup_sv)
        branch_vals['taup_charged_ipx'][0] = taup_charged_ip.X()
        branch_vals['taup_charged_ipy'][0] = taup_charged_ip.Y()
        branch_vals['taup_charged_ipz'][0] = taup_charged_ip.Z()

    if len(taun_pis) == 1:
        branch_vals['taun_charged_ipx'][0] = taun_ip.X()
        branch_vals['taun_charged_ipy'][0] = taun_ip.Y()
        branch_vals['taun_charged_ipz'][0] = taun_ip.Z()
    elif len(taun_leptons) == 1:
        branch_vals['taun_charged_ipx'][0] = taun_lep_ip.X()
        branch_vals['taun_charged_ipy'][0] = taun_lep_ip.Y()
        branch_vals['taun_charged_ipz'][0] = taun_lep_ip.Z()
    elif len(taun_pis) > 1:
        taun_charged_ip = get_pseudo_impact_parameter(taun_charged_sum.Vect().Unit(), taun_sv)
        branch_vals['taun_charged_ipx'][0] = taun_charged_ip.X()
        branch_vals['taun_charged_ipy'][0] = taun_charged_ip.Y()
        branch_vals['taun_charged_ipz'][0] = taun_charged_ip.Z()

    # get smeared "reco" quantities
    # first we dR match reco particles to gen visible taus
    taup_reco_track_matches = GetMatchingRecoParticle(taup_vis, eflowtracks, threshold=0.5)
    taup_reco_photon_matches = GetMatchingRecoParticle(taup_vis, photons, threshold=0.5)
    taun_reco_track_matches = GetMatchingRecoParticle(taun_vis, eflowtracks, threshold=0.5) # add in the track matches as well (these will have some smearing but also have charge and impact parameter info which is useful for the study)
    taun_reco_photon_matches = GetMatchingRecoParticle(taun_vis, photons, threshold=0.5)

    do_track_angular_smearing = True

    if do_track_angular_smearing:
        # apply eta-phi and ip smearing for tracks, since this is not done by delphes
        for t in taup_reco_track_matches:
            new_p4 = smearer.smear_tlorentzvector_angles(t.P4())
            new_d0, new_dz = smearer.smear_d0_dz(t.D0, t.DZ, t.Eta, t.PT)
            t.PT = new_p4.Pt()
            t.Eta = new_p4.Eta()
            t.Phi = new_p4.Phi()
            t.D0 = new_d0
            t.DZ = new_dz
            t.CtgTheta = np.sinh(t.Eta) # recalculate CtgTheta based on new Eta
        for t in taun_reco_track_matches:
            new_p4 = smearer.smear_tlorentzvector_angles(t.P4())
            new_d0, new_dz = smearer.smear_d0_dz(t.D0, t.DZ, t.Eta, t.PT)
            t.PT = new_p4.Pt()
            t.Eta = new_p4.Eta()
            t.Phi = new_p4.Phi()
            t.D0 = new_d0
            t.DZ = new_dz
            t.CtgTheta = np.sinh(t.Eta) # recalculate CtgTheta based on new Eta
    taup_strips = RecoStrips(taup_reco_photon_matches)
    # sort by pT of the strip
    taup_strips.sort(key=lambda x: x[0].Pt(), reverse=True)

    taun_strips = RecoStrips(taun_reco_photon_matches)
    # sort by pT of the strip
    taun_strips.sort(key=lambda x: x[0].Pt(), reverse=True)

    # get tau candidates, we also match the charges to the tau+ and tau- to avoid using the same track twice

    taup_cands = GetTauCands(taup_reco_track_matches, taup_strips, incDM2=False, match_charge=1) # Note: not allowing DM2 taus for now to match CMS reco where DM2 is very rare 
    taun_cands = GetTauCands(taun_reco_track_matches, taun_strips, incDM2=False, match_charge=-1)

    reco_taup_vis = taup_cands[0][0] if len(taup_cands) > 0 else ROOT.TLorentzVector()
    reco_taun_vis = taun_cands[0][0] if len(taun_cands) > 0 else ROOT.TLorentzVector()

    # we apply a dR cut to make sure the taus are not overlapping mimicking what is done for a real analysis
    dR_taus = reco_taup_vis.DeltaR(reco_taun_vis)
    if dR_taus < 0.5:
        continue

    # as an extra check make sure the taus don't share any tracks or strips
    shared_tracks = set(taup_cands[0][1]) & set(taun_cands[0][1]) if len(taup_cands) > 0 and len(taun_cands) > 0 else set()
    shared_strips = set(taup_cands[0][2]) & set(taun_cands[0][2]) if len(taup_cands) > 0 and len(taun_cands) > 0 else set()
    if len(shared_tracks) > 0 or len(shared_strips) > 0:
        print('WARNING: Found overlapping tau candidates with shared tracks or strips, skipping event')
        continue

    branch_vals['reco_taup_vis_pT'][0] = reco_taup_vis.Pt()
    branch_vals['reco_taup_vis_eta'][0] = reco_taup_vis.Eta()
    branch_vals['reco_taun_vis_pT'][0] = reco_taun_vis.Pt()
    branch_vals['reco_taun_vis_eta'][0] = reco_taun_vis.Eta()

    reco_pv_3vec = ROOT.TVector3(0,0,0) # gen-level PV position is always 0,0,0
    if do_track_angular_smearing:
        reco_pv_3vec = smear_PV(ROOT.TVector3(0,0,0)) # smear the PV position (gen-level is always 0,0,0)

    reco_taup_sv = ROOT.TVector3(0,0,0)
    reco_taun_sv = ROOT.TVector3(0,0,0)

    if len(taup_cands) > 0:

        if len(taup_cands[0][1]) > 0:
            branch_vals['reco_taup_npi'][0] = len(taup_cands[0][1])
            branch_vals['reco_taup_npizero'][0] = len(taup_cands[0][2])            

            branch_vals['reco_taup_pi1_px'][0] = taup_cands[0][1][0].P4().Px()
            branch_vals['reco_taup_pi1_py'][0] = taup_cands[0][1][0].P4().Py()
            branch_vals['reco_taup_pi1_pz'][0] = taup_cands[0][1][0].P4().Pz()
            branch_vals['reco_taup_pi1_e'][0] = taup_cands[0][1][0].P4().E()

            reco_taup_point = get_3d_point_from_phi_d0_dz(taup_cands[0][1][0].Phi, taup_cands[0][1][0].D0, taup_cands[0][1][0].DZ)
            # overwrite Xd, Yd, Zd with the new 3D point
            taup_cands[0][1][0].Xd = reco_taup_point.X()
            taup_cands[0][1][0].Yd = reco_taup_point.Y()
            taup_cands[0][1][0].Zd = reco_taup_point.Z()
            reco_taup_ip = get_impact_parameter(taup_cands[0][1][0], pv_3vec=reco_pv_3vec, reco_track=True)

            branch_vals['reco_taup_pi1_ipx'][0] = reco_taup_ip.X()
            branch_vals['reco_taup_pi1_ipy'][0] = reco_taup_ip.Y()
            branch_vals['reco_taup_pi1_ipz'][0] = reco_taup_ip.Z()

        if len(taup_cands[0][1]) > 1:
            branch_vals['reco_taup_pi2_px'][0] = taup_cands[0][1][1].P4().Px()
            branch_vals['reco_taup_pi2_py'][0] = taup_cands[0][1][1].P4().Py()
            branch_vals['reco_taup_pi2_pz'][0] = taup_cands[0][1][1].P4().Pz()
            branch_vals['reco_taup_pi2_e'][0] = taup_cands[0][1][1].P4().E()
        if len(taup_cands[0][1]) > 2:
            branch_vals['reco_taup_pi3_px'][0] = taup_cands[0][1][2].P4().Px()
            branch_vals['reco_taup_pi3_py'][0] = taup_cands[0][1][2].P4().Py()
            branch_vals['reco_taup_pi3_pz'][0] = taup_cands[0][1][2].P4().Pz()
            branch_vals['reco_taup_pi3_e'][0] = taup_cands[0][1][2].P4().E()

            reco_directions = []
            reco_points = []
            for t in taup_cands[0][1][:3]: # use up to the first 3 tracks for the SV estimation
                reco_directions.append(t.P4().Vect().Unit())
                reco_points.append(ROOT.TVector3(t.Xd, t.Yd, t.Zd))
            reco_taup_sv_lsq = FindVertexLSQ(reco_points[0], reco_directions[0], reco_points[1], reco_directions[1], reco_points[2], reco_directions[2])

            reco_taup_sv = reco_taup_sv_lsq - reco_pv_3vec

            branch_vals['reco_taup_sv_x'][0] = reco_taup_sv.X()
            branch_vals['reco_taup_sv_y'][0] = reco_taup_sv.Y()
            branch_vals['reco_taup_sv_z'][0] = reco_taup_sv.Z()

        if len(taup_cands[0]) > 3 and len(taup_cands[0][3]) > 0:
            branch_vals['reco_taup_lep_px'][0] = taup_cands[0][3][0].P4().Px()
            branch_vals['reco_taup_lep_py'][0] = taup_cands[0][3][0].P4().Py()
            branch_vals['reco_taup_lep_pz'][0] = taup_cands[0][3][0].P4().Pz()
            branch_vals['reco_taup_lep_e'][0] = taup_cands[0][3][0].P4().E()

            # identify if the lepton was electron or muon based on PID
            if abs(taup_cands[0][3][0].PID) == 11:
                branch_vals['reco_taup_nele'][0] = 1
            elif abs(taup_cands[0][3][0].PID) == 13:
                branch_vals['reco_taup_nmu'][0] = 1

            reco_lep_ip = get_impact_parameter(taup_cands[0][3][0], pv_3vec=reco_pv_3vec, reco_track=True)
            branch_vals['reco_taup_lep_ipx'][0] = reco_lep_ip.X()
            branch_vals['reco_taup_lep_ipy'][0] = reco_lep_ip.Y()
            branch_vals['reco_taup_lep_ipz'][0] = reco_lep_ip.Z()

        if len(taup_cands[0][2]) > 0:
            branch_vals['reco_taup_pizero1_px'][0] = taup_cands[0][2][0].Px()
            branch_vals['reco_taup_pizero1_py'][0] = taup_cands[0][2][0].Py()
            branch_vals['reco_taup_pizero1_pz'][0] = taup_cands[0][2][0].Pz()
            branch_vals['reco_taup_pizero1_e'][0] = taup_cands[0][2][0].E()

        # setup the charged sum 4-vec and ip
        # if lepton decay set this to the lepton
        # if 1 prong hadronic decay set this to the pi
        # if 3-prong set the 4-vect to the sum of the 3 charged pions and the ip to the pseudo-ip from the SV and the charged sum momentum
        # store the reco SV for the 3-prongs as well
        if len(taup_cands[0][1]) > 2:
            charged_4vec = ROOT.TLorentzVector()
            for i in range(3):
                charged_4vec += taup_cands[0][1][i].P4()
            branch_vals['reco_taup_charged_px'][0]  = charged_4vec.Px()
            branch_vals['reco_taup_charged_py'][0]  = charged_4vec.Py()
            branch_vals['reco_taup_charged_pz'][0]  = charged_4vec.Pz()
            branch_vals['reco_taup_charged_e'][0]   = charged_4vec.E()
            taup_charged_ip = get_pseudo_impact_parameter(charged_4vec.Vect().Unit(), reco_taup_sv, pv_3vec=reco_pv_3vec)
            branch_vals['reco_taup_charged_ipx'][0] = taup_charged_ip.X()
            branch_vals['reco_taup_charged_ipy'][0] = taup_charged_ip.Y()
            branch_vals['reco_taup_charged_ipz'][0] = taup_charged_ip.Z()

            branch_vals['reco_taup_sv_x'][0] = reco_taup_sv.X()
            branch_vals['reco_taup_sv_y'][0] = reco_taup_sv.Y()
            branch_vals['reco_taup_sv_z'][0] = reco_taup_sv.Z()
        else: 
            if len(taup_cands[0]) > 3 and len(taup_cands[0][3]) > 0:
                charged_name = 'lep'
            elif len(taup_cands[0][1]) == 1:
                charged_name = 'pi1'

            branch_vals['reco_taup_charged_px'][0]  = branch_vals[f'reco_taup_{charged_name}_px'][0]
            branch_vals['reco_taup_charged_py'][0]  = branch_vals[f'reco_taup_{charged_name}_py'][0]
            branch_vals['reco_taup_charged_pz'][0]  = branch_vals[f'reco_taup_{charged_name}_pz'][0]
            branch_vals['reco_taup_charged_e'][0]   = branch_vals[f'reco_taup_{charged_name}_e'][0]
            branch_vals['reco_taup_charged_ipx'][0] = branch_vals[f'reco_taup_{charged_name}_ipx'][0]
            branch_vals['reco_taup_charged_ipy'][0] = branch_vals[f'reco_taup_{charged_name}_ipy'][0]
            branch_vals['reco_taup_charged_ipz'][0] = branch_vals[f'reco_taup_{charged_name}_ipz'][0]
            

    if len(taun_cands) > 0:

        if len(taun_cands[0][1]) > 0:
            branch_vals['reco_taun_npi'][0] = len(taun_cands[0][1])
            branch_vals['reco_taun_npizero'][0] = len(taun_cands[0][2])
            branch_vals['reco_taun_pi1_px'][0] = taun_cands[0][1][0].P4().Px()
            branch_vals['reco_taun_pi1_py'][0] = taun_cands[0][1][0].P4().Py()
            branch_vals['reco_taun_pi1_pz'][0] = taun_cands[0][1][0].P4().Pz()
            branch_vals['reco_taun_pi1_e'][0] = taun_cands[0][1][0].P4().E()

            reco_taun_point = get_3d_point_from_phi_d0_dz(taun_cands[0][1][0].Phi, taun_cands[0][1][0].D0, taun_cands[0][1][0].DZ)
            # overwrite Xd, Yd, Zd with the new 3D point
            taun_cands[0][1][0].Xd = reco_taun_point.X()
            taun_cands[0][1][0].Yd = reco_taun_point.Y()
            taun_cands[0][1][0].Zd = reco_taun_point.Z()

            reco_taun_ip = get_impact_parameter(taun_cands[0][1][0], pv_3vec=reco_pv_3vec, reco_track=True)
            branch_vals['reco_taun_pi1_ipx'][0] = reco_taun_ip.X()
            branch_vals['reco_taun_pi1_ipy'][0] = reco_taun_ip.Y()
            branch_vals['reco_taun_pi1_ipz'][0] = reco_taun_ip.Z()

        if len(taun_cands[0][1]) > 1:
            branch_vals['reco_taun_pi2_px'][0] = taun_cands[0][1][1].P4().Px()
            branch_vals['reco_taun_pi2_py'][0] = taun_cands[0][1][1].P4().Py()
            branch_vals['reco_taun_pi2_pz'][0] = taun_cands[0][1][1].P4().Pz()
            branch_vals['reco_taun_pi2_e'][0] = taun_cands[0][1][1].P4().E()
        if len(taun_cands[0][1]) > 2:
            branch_vals['reco_taun_pi3_px'][0] = taun_cands[0][1][2].P4().Px()
            branch_vals['reco_taun_pi3_py'][0] = taun_cands[0][1][2].P4().Py()
            branch_vals['reco_taun_pi3_pz'][0] = taun_cands[0][1][2].P4().Pz()
            branch_vals['reco_taun_pi3_e'][0] = taun_cands[0][1][2].P4().E()

            reco_directions = []
            reco_points = []
            for t in taun_cands[0][1][:3]: # use up to the first 3 tracks for the SV estimation
                reco_directions.append(t.P4().Vect().Unit())
                reco_points.append(ROOT.TVector3(t.Xd, t.Yd, t.Zd))
            reco_taun_sv_lsq = FindVertexLSQ(reco_points[0], reco_directions[0], reco_points[1], reco_directions[1], reco_points[2], reco_directions[2])

            reco_taun_sv = reco_taun_sv_lsq - reco_pv_3vec

            branch_vals['reco_taun_sv_x'][0] = reco_taun_sv.X()
            branch_vals['reco_taun_sv_y'][0] = reco_taun_sv.Y()
            branch_vals['reco_taun_sv_z'][0] = reco_taun_sv.Z()

        if len(taun_cands[0]) > 3 and len(taun_cands[0][3]) > 0:
            branch_vals['reco_taun_lep_px'][0] = taun_cands[0][3][0].P4().Px()
            branch_vals['reco_taun_lep_py'][0] = taun_cands[0][3][0].P4().Py()
            branch_vals['reco_taun_lep_pz'][0] = taun_cands[0][3][0].P4().Pz()
            branch_vals['reco_taun_lep_e'][0] = taun_cands[0][3][0].P4().E()

            reco_lep_ip = get_impact_parameter(taun_cands[0][3][0], pv_3vec=reco_pv_3vec, reco_track=True)
            branch_vals['reco_taun_lep_ipx'][0] = reco_lep_ip.X()
            branch_vals['reco_taun_lep_ipy'][0] = reco_lep_ip.Y()
            branch_vals['reco_taun_lep_ipz'][0] = reco_lep_ip.Z()

            # identify if the lepton was electron or muon based on PID
            if abs(taun_cands[0][3][0].PID) == 11:
                branch_vals['reco_taun_nele'][0] = 1
            elif abs(taun_cands[0][3][0].PID) == 13:
                branch_vals['reco_taun_nmu'][0] = 1

        if len(taun_cands[0][2]) > 0:
            branch_vals['reco_taun_pizero1_px'][0] = taun_cands[0][2][0].Px()
            branch_vals['reco_taun_pizero1_py'][0] = taun_cands[0][2][0].Py()
            branch_vals['reco_taun_pizero1_pz'][0] = taun_cands[0][2][0].Pz()
            branch_vals['reco_taun_pizero1_e'][0] = taun_cands[0][2][0].E()

        # setup the charged sum 4-vec and ip
        # if lepton decay set this to the lepton
        # if 1 prong hadronic decay set this to the pi
        # if 3-prong set the 4-vect to the sum of the 3 charged pions and the ip to the pseudo-ip from the SV and the charged sum momentum
        # store the reco SV for the 3-prongs as well
        if len(taun_cands[0][1]) > 2:
            charged_4vec = ROOT.TLorentzVector()
            for i in range(3):
                charged_4vec += taun_cands[0][1][i].P4()
            branch_vals['reco_taun_charged_px'][0]  = charged_4vec.Px()
            branch_vals['reco_taun_charged_py'][0]  = charged_4vec.Py()
            branch_vals['reco_taun_charged_pz'][0]  = charged_4vec.Pz()
            branch_vals['reco_taun_charged_e'][0]   = charged_4vec.E()
            taun_charged_ip = get_pseudo_impact_parameter(charged_4vec.Vect().Unit(), reco_taun_sv, pv_3vec=reco_pv_3vec)
            branch_vals['reco_taun_charged_ipx'][0] = taun_charged_ip.X()
            branch_vals['reco_taun_charged_ipy'][0] = taun_charged_ip.Y()
            branch_vals['reco_taun_charged_ipz'][0] = taun_charged_ip.Z()

            branch_vals['reco_taun_sv_x'][0] = reco_taun_sv.X()
            branch_vals['reco_taun_sv_y'][0] = reco_taun_sv.Y()
            branch_vals['reco_taun_sv_z'][0] = reco_taun_sv.Z()
        else: 
            if len(taun_cands[0]) > 3 and len(taun_cands[0][3]) > 0:
                charged_name = 'lep'
            elif len(taun_cands[0][1]) == 1:
                charged_name = 'pi1'

            branch_vals['reco_taun_charged_px'][0]  = branch_vals[f'reco_taun_{charged_name}_px'][0]
            branch_vals['reco_taun_charged_py'][0]  = branch_vals[f'reco_taun_{charged_name}_py'][0]
            branch_vals['reco_taun_charged_pz'][0]  = branch_vals[f'reco_taun_{charged_name}_pz'][0]
            branch_vals['reco_taun_charged_e'][0]   = branch_vals[f'reco_taun_{charged_name}_e'][0]
            branch_vals['reco_taun_charged_ipx'][0] = branch_vals[f'reco_taun_{charged_name}_ipx'][0]
            branch_vals['reco_taun_charged_ipy'][0] = branch_vals[f'reco_taun_{charged_name}_ipy'][0]
            branch_vals['reco_taun_charged_ipz'][0] = branch_vals[f'reco_taun_{charged_name}_ipz'][0]
            

    for tau in ['taup', 'taun']:
        # for  1-prong + 1/2 pi0 decays it is hard to replicate the proper decay mode application 
        # so it will be sampled using the number from here instead: https://cds.cern.ch/record/2727092 
        #first check if gen decay mode is 1-prong + 1 pi0 
        if branch_vals[f'reco_{tau}_npi'][0] == 1 and branch_vals[f'reco_{tau}_npizero'][0] == 1:
            is_true_dm_0 = branch_vals[f'{tau}_npi'][0] == 1 and branch_vals[f'{tau}_npizero'][0] == 0
            is_true_dm_1 = branch_vals[f'{tau}_npi'][0] == 1 and branch_vals[f'{tau}_npizero'][0] == 1
            is_true_dm_2 = branch_vals[f'{tau}_npi'][0] == 1 and branch_vals[f'{tau}_npizero'][0] == 2
            if is_true_dm_0:
                reco_dm_1_frac = 0.058/(0.058 + 0.015)
                reco_dm_2_frac = 0.015/(0.058 + 0.015)
            elif is_true_dm_2:
                reco_dm_1_frac = 0.182/(0.182+0.555)
                reco_dm_2_frac = 0.555/(0.182+0.555)
            elif is_true_dm_1:
                reco_dm_1_frac = 0.677/(0.677+0.246)
                reco_dm_2_frac = 0.246/(0.677+0.246)
            else: 
                # for rarer modes e.g 3-prong mis-IDs just 50-50 sample them 
                reco_dm_1_frac = 0.5
                reco_dm_2_frac = 0.5
            rand = np.random.rand()   
            if rand < reco_dm_1_frac:
                # assign as dm 1
                branch_vals[f'reco_{tau}_npizero'][0] = 1
            else:
                # assign as dm 2
                branch_vals[f'reco_{tau}_npizero'][0] = 2 

    #store reco MET
    branch_vals['reco_met_px'][0] = MET.At(0).MET * math.cos(MET.At(0).Phi)
    branch_vals['reco_met_py'][0] = MET.At(0).MET * math.sin(MET.At(0).Phi)

    tree.Fill()

fout.Write()
fout.Close()
    
