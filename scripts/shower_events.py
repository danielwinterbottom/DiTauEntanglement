import pythia8
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
#from pyHepMC3 import HepMC3
#from Pythia8ToHepMC3 import Pythia8ToHepMC3
import ROOT
from array import array
from ReconstructTaus import FindDMin_Point, FindDMin

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'LHE file to be converted')
parser.add_argument('--output', '-o', help= 'Name of output file',default='pythia_output.hepmc')
parser.add_argument('--cmnd_file', '-c', help= 'Pythia8 command file')
parser.add_argument('--n_events', '-n', help= 'Maximum number of events to process', default=-1, type=int)
parser.add_argument('--n_skip', '-s', help= 'skip n_events*n_skip', default=0, type=int)
parser.add_argument('--seed', help= 'Random seed for Pythia', default=1, type=int)
parser.add_argument('--phi', help= 'pythia definition of CP mixing angle in degrees (only used for ee->H -> tautau sample) CP-even=pi/2, CP-odd=0, max-mix=pi/4', default=1.5708, type=float)
parser.add_argument('--extra_vars', action='store_true', help= 'If set, will also store additional variables in the output root file including the analytically predicted tau and spin sensitive quantities used to measure the rho matrix')


args = parser.parse_args()


# setup output root file
# taup = positive tau, indices will label the pions
# for now only setup for pipi channel so only index 1 is used for the pion from tau->pinu decays

branches = [
'z_x',
'z_y',
'z_z',
'taup_px', 
'taup_py', 
'taup_pz',
'taup_e',
'taup_npi',
'taup_npizero',
'taup_nmu',
'taup_pi1_px',
'taup_pi1_py',
'taup_pi1_pz',
'taup_pi1_e',
'taup_pi1_vx',
'taup_pi1_vy',
'taup_pi1_vz',
'taup_pi1_ipx',
'taup_pi1_ipy',
'taup_pi1_ipz',
'taup_pi2_px',
'taup_pi2_py',
'taup_pi2_pz',
'taup_pi2_e',
'taup_pi2_vx',
'taup_pi2_vy',
'taup_pi2_vz',
'taup_pi3_px',
'taup_pi3_py',
'taup_pi3_pz',
'taup_pi3_e',
'taup_pi3_vx',
'taup_pi3_vy',
'taup_pi3_vz',
'taup_pizero1_px',
'taup_pizero1_py',
'taup_pizero1_pz',
'taup_pizero1_e',
'taup_pizero2_px',
'taup_pizero2_py',
'taup_pizero2_pz',
'taup_pizero2_e',
'taup_nu_px',
'taup_nu_py',
'taup_nu_pz',
'taup_nu_e',
'taun_px',
'taun_py',
'taun_pz',
'taun_e',
'taun_npi',
'taun_npizero',
'taun_nmu',
'taun_pi1_px',
'taun_pi1_py',
'taun_pi1_pz',
'taun_pi1_e',
'taun_pi1_vx',
'taun_pi1_vy',
'taun_pi1_vz',
'taun_pi1_ipx',
'taun_pi1_ipy',
'taun_pi1_ipz',
'taun_pi2_px',
'taun_pi2_py',
'taun_pi2_pz',
'taun_pi2_e',
'taun_pi2_vx',
'taun_pi2_vy',
'taun_pi2_vz',
'taun_pi3_px',
'taun_pi3_py',
'taun_pi3_pz',
'taun_pi3_e',
'taun_pi3_vx',
'taun_pi3_vy',
'taun_pi3_vz',
'taun_pizero1_px',
'taun_pizero1_py',
'taun_pizero1_pz',
'taun_pizero1_e',
'taun_pizero2_px',
'taun_pizero2_py',
'taun_pizero2_pz',
'taun_pizero2_e',
'taun_nu_px',
'taun_nu_py',
'taun_nu_pz',
'taun_nu_e',

'taup_mu_px',
'taup_mu_py',
'taup_mu_pz',
'taup_mu_e',
'taun_mu_px',
'taun_mu_py',
'taun_mu_pz',
'taun_mu_e',

'taup_vis_pt',
'taun_vis_pt',
'm_vis',

'pv_x',
'pv_y',
'pv_z',
'met_px',
'met_py',
'met_pz',

'taup_first_px', 
'taup_first_py', 
'taup_first_pz',
'taup_first_e',
'taun_first_px',
'taun_first_py',
'taun_first_pz',
'taun_first_e',

'boson_mass',
'boson_pt',

]

if args.extra_vars:
    branches += [
        'solution1_taup_px',
        'solution1_taup_py',
        'solution1_taup_pz',
        'solution1_taup_e',
        'solution1_taun_px',
        'solution1_taun_py',
        'solution1_taun_pz',
        'solution1_taun_e',
        'solution2_taup_px',
        'solution2_taup_py',
        'solution2_taup_pz',
        'solution2_taup_e',
        'solution2_taun_px',
        'solution2_taun_py',
        'solution2_taun_pz',
        'solution2_taun_e',
        'cosn_plus',
        'cosr_plus',
        'cosk_plus',
        'cosn_minus',
        'cosr_minus',
        'cosk_minus',
        'cosTheta',

    ]

if '.hepmc' in args.output: root_output = args.output.replace('.hepmc','.root')
else: root_output=args.output+'.root'
fout = ROOT.TFile(root_output,'RECREATE')
tree = ROOT.TTree('tree','')

branch_vals = {}
for b in branches:
    branch_vals[b] = array('f',[0])
    tree.Branch(b,  branch_vals[b],  '%s/F' % b)
    if (b.startswith('taup_') or b.startswith('taun_')) and 'first' not in b:
        # add also branches for first copy
        b_first_name = b.replace('taup','taup_LHE').replace('taun','taun_LHE')
        branch_vals[b_first_name] = array('f',[0])
        tree.Branch(b_first_name,  branch_vals[b_first_name],  '%s/F' % b_first_name)


pythia = pythia8.Pythia("")
pythia.readFile(args.cmnd_file)

#pythia_process = "eeToZtoTauTau"
#pythia_process = "ppToHToTauTau"
#pythia_process = "eeToHtoTauTau"
pythia_process="eeToZHtoTauTau"


if args.input:
    pythia.readString("Beams:frameType = 4")
    pythia.readString("Beams:LHEF = %s" % args.input)
elif pythia_process == "eeToZtoTauTau":
    print('Producing full e+e- -> Z -> tautau event in pythia')
    # if no LHE file given then produce full event setup using pythia for the hard process as well
    pythia.readString("Beams:idA = -11") # Positron
    pythia.readString("Beams:idB = 11")  # Electron
    pythia.readString("Beams:eCM = 91.2")  # Center-of-mass energy = Z resonance
    pythia.readString("TauDecays:externalMode = 1")

    # Enable Z production and decay to taus
    pythia.readString("WeakSingleBoson:ffbar2gmZ = on")
    pythia.readString("23:onMode = off")  # Turn off all Z decays
    pythia.readString("23:onIfAny = 15")  # Enable Z -> tau+ tau-
elif pythia_process == "ppToHToTauTau": # for pp->H->tautau using pythia
    print('Producing full pp -> H -> tautau event in pythia')
    pythia.readString("Beams:idA = 2212") # Proton
    pythia.readString("Beams:idB = 2212")  # Proton
    pythia.readString("Beams:eCM = 13600")  # Center-of-mass energy
    # Enable H production and decay to taus
    pythia.readString("HiggsSM:gg2H = on")

    # Force Higgs to decay only to tau+ tau-
    pythia.readString("25:onMode = off")
    pythia.readString("25:onIfAny = 15")

    pythia.readString("HiggsH1:parity = 4")
    pythia.readString(f"HiggsH1:phiParity = {args.phi}")
elif pythia_process == "eeToHtoTauTau": # for e+e- -> H -> tautau using pythia (not that the is a realistic production mode but it's just to get a sample of Higgs bosons at rest)
    print('Producing full e+e -> H-> tautau event in pythia')
    pythia.readString("Beams:idA = -11") # Positron
    pythia.readString("Beams:idB = 11")  # Electron
    pythia.readString("Beams:eCM = 125")  # Center-of-mass energy
    #pythia.readString("TauDecays:externalMode = 0")
    # Enable H production and decay to taus

    pythia.readString("HiggsSM:ffbar2H = on")

    # Force Higgs to decay only to tau+ tau-
    pythia.readString("25:onMode = off")
    pythia.readString("25:onIfAny = 15")
    pythia.readString("HiggsH1:parity = 4")
    pythia.readString(f"HiggsH1:phiParity = {args.phi}")
elif pythia_process == "eeToZHtoTauTau":
    print("Producing full e+e- -> ZH -> ee/mumu/jj + tautau in pythia")

    pythia.readString("Beams:idA = -11")   # e+
    pythia.readString("Beams:idB = 11")    # e-
    pythia.readString("Beams:eCM = 240.")  # FCC-ee Higgs factory energy

    # Higgsstrahlung: e+e- -> ZH
    pythia.readString("HiggsSM:ffbar2HZ = on")

    # Force H -> tau tau
    pythia.readString("25:onMode = off")
    pythia.readString("25:onIfMatch = 15 -15")

    # choose Z decays, we use either ee, mumu, or jj decays only since we dont want additional MET in the events
    pythia.readString("23:onMode = off")
    pythia.readString("23:onIfAny = 11 13 1 2 3 4 5")

    pythia.readString("HiggsH1:parity = 4")
    pythia.readString(f"HiggsH1:phiParity = {args.phi}")

pythia.readString("Random:setSeed = on")
pythia.readString(f"Random:seed = {args.seed}")  # Set random seed for reproducibility

pythia.init()

pythia.LHAeventSkip(args.n_skip*args.n_events)

#hepmc_converter = Pythia8ToHepMC3()
#hepmc_writer = HepMC3.WriterAscii(args.output)

def IsLastCopy(part, event):
    ''' 
    check if particle is the last copy by checking if it has no daughters of the same pdgid
    check may not work for some particle types - tested only for taus at the moment
    '''
    LastCopy = True
    pdgid = part.id()
    for p in part.daughterList():
        if event[p].id() == pdgid: LastCopy = False

    return LastCopy

def IsFirstCopy(part, event):
    ''' 
    check if particle is the first copy by checking if it has no mothers of the same pdgid
    check may not work for some particle types - tested only for taus at the moment
    '''
    FirstCopy = True
    pdgid = part.id()
    for p in part.motherList():
        if event[p].id() == pdgid: FirstCopy = False

    return FirstCopy

def TraceTauMother(part, event, mother_ids_to_exclude=[]):
    """
    Recursively trace tau's ancestry until the first non-tau mother is found.
    """
    mother_indices = part.motherList()
    
    if not mother_indices:
        print(f"No mother found for particle with id {part.id()}")
        return

    # Take the first mother (can expand to loop over all if needed)
    mother = event[mother_indices[0]]

    if abs(mother.id()) == 15:
        # Keep going up the ancestry
        TraceTauMother(mother, event, mother_ids_to_exclude)
    else:
        # Print non-tau mother info unless it is in the excluded list
        if not mother.id() in mother_ids_to_exclude:
            print("First non-tau mother found for tau with status %i:" % part.status())
            print('id = %i, status = %i, px = %.4f, py = %.4f, pz = %.4f, e = %.4f' %
                  (mother.id(), mother.status(), mother.px(), mother.py(), mother.pz(), mother.e()))
            # check if there are any other mothers and if so print these as well
            if len(mother_indices) > 1:
                print("Other mothers found:")
                for idx in mother_indices[1:]:
                    other_mother = event[idx]
                    if not other_mother.id() in mother_ids_to_exclude:
                        print('id = %i, status = %i, px = %.4f, py = %.4f, pz = %.4f, e = %.4f' %
                              (other_mother.id(), other_mother.status(), other_mother.px(), other_mother.py(), other_mother.pz(), other_mother.e()))


def GetPiDaughters(part, event):
    pis = []
    pi0s = []
    nus = []
    mus = []

    rho0_mass = 0.7755

    # Retrieve the charge of the parent tau
    tau_charge = part.charge()

    daughter_pdgids = []

    for d in part.daughterList():
        daughter = event[d]
        daughter_pdgids.append(daughter.id())
        if abs(daughter.id()) == 211:
            pis.append(daughter)
        if abs(daughter.id()) == 111:
            pi0s.append(daughter)
        if abs(daughter.id()) == 16:
            nus.append(daughter)
        if abs(daughter.id()) == 13:
            mus.append(daughter)

    if len(pis) == 3:
        # Separate the pion with the opposite charge to the parent tau
        first_pi = next(pi for pi in pis if pi.charge() != tau_charge)

        # Remove the first pion from the list
        remaining_pis = [pi for pi in pis if pi != first_pi]

        # Sort the remaining pions based on the mass of the pair with first_pi
        remaining_pis.sort(
            key=lambda pi: abs((first_pi.p() + pi.p()).mCalc() - rho0_mass)
        )

        # Combine the sorted list
        sorted_pis = [first_pi] + remaining_pis

#        print('checking pi sorting:')
#        print('tau charge = %i' % tau_charge)
#        for i, pi in enumerate(sorted_pis):
#            print('pi%i charge = %i' % (i, pi.charge()))
#            if i>0:
#                print('mass = %.4f, mass diff = %.4f' % ((sorted_pis[0].p() + pi.p()).mCalc(), abs((sorted_pis[0].p() + pi.p()).mCalc() - rho0_mass)))

        return (sorted_pis, pi0s, nus, mus, daughter_pdgids)

    return (pis, pi0s, nus, mus, daughter_pdgids)

def TauRotationCorrection(taup_LHE, taun_LHE, taup, taun):
    ditau = taup + taun
    ditau_LHE = taup_LHE + taun_LHE
    # check if masses are consistent, if not print a warning
    if abs(ditau.M() - ditau_LHE.M()) > 1e-6:
        print('Warning: ditau mass mismatch: %.6f vs %.6f' % (ditau.M(), ditau_LHE.M()))
    # boost taup's to ditau rest frames
    taup_LHE.Boost(-ditau_LHE.BoostVector())
    taup.Boost(-ditau.BoostVector())
    v_orig = taup.Vect().Unit()
    v_target = taup_LHE.Vect().Unit()

    axis = v_orig.Cross(v_target)
    if axis.Mag() < 1e-8:
        # Vectors already aligned or exactly opposite
        angle = 0 if v_orig.Dot(v_target) > 0 else math.pi
        axis = ROOT.TVector3(1, 0, 0)  # Arbitrary orthogonal axis
    else:
        axis = axis.Unit()
        angle = math.acos(v_orig.Dot(v_target))

    # Construct rotation
    rotation = ROOT.TRotation()
    rotation.Rotate(angle, axis)

    boost_to_LHE_frame = ROOT.TLorentzRotation(ditau_LHE.BoostVector())
    boost_to_orig_rest = ROOT.TLorentzRotation(-ditau.BoostVector())
    rot = ROOT.TLorentzRotation(rotation)
    transform = boost_to_orig_rest.Inverse() * rot * boost_to_orig_rest

    return transform

stopGenerating = False
count = 0

pythia.next() # generate first event

while not stopGenerating:

    # initialize the branch values to zero
    for b in branches:
        branch_vals[b][0] = 0

    stopGenerating = pythia.infoPython().atEndOfFile()
    if args.n_events>0 and count+1 >= args.n_events: stopGenerating = True


    #print('-------------------------------')
    #print('event %i' % (count+1))

    #print('particles:')


    first_pis = []
    first_pizeros = []
    first_nus = []
    first_mus = []
    first_taus = []
    taus_LHE = []
    for i, part in enumerate(pythia.process):
        pdgid = part.id()
        if abs(pdgid) == 15:

            taus_LHE.append(part)

            tau_name = 'taun_LHE' if pdgid == 15 else 'taup_LHE'
            pis, pi0s, nus, mus, daughter_pdgids = GetPiDaughters(part,pythia.process)
            branch_vals['%(tau_name)s_px' % vars()][0] = part.px()
            branch_vals['%(tau_name)s_py' % vars()][0] = part.py()
            branch_vals['%(tau_name)s_pz' % vars()][0] = part.pz()
            branch_vals['%(tau_name)s_e' % vars()][0]  = part.e()
            branch_vals['%(tau_name)s_npi' % vars()][0]  = len(pis)
            branch_vals['%(tau_name)s_npizero' % vars()][0]  = len(pi0s)
            branch_vals['%(tau_name)s_nmu' % vars()][0]  = len(mus)

            if len(pis) > 0:
                branch_vals['%(tau_name)s_pi1_px' % vars()][0] = pis[0].px()
                branch_vals['%(tau_name)s_pi1_py' % vars()][0] = pis[0].py()
                branch_vals['%(tau_name)s_pi1_pz' % vars()][0] = pis[0].pz()
                branch_vals['%(tau_name)s_pi1_e' % vars()][0]  = pis[0].e()
                branch_vals['%(tau_name)s_pi1_vx' % vars()][0] = pis[0].xProd() # mm units
                branch_vals['%(tau_name)s_pi1_vy' % vars()][0] = pis[0].yProd()
                branch_vals['%(tau_name)s_pi1_vz' % vars()][0] = pis[0].zProd()

            if len(nus) > 0:
                branch_vals['%(tau_name)s_nu_px' % vars()][0] = nus[0].px()
                branch_vals['%(tau_name)s_nu_py' % vars()][0] = nus[0].py()
                branch_vals['%(tau_name)s_nu_pz' % vars()][0] = nus[0].pz()
                branch_vals['%(tau_name)s_nu_e' % vars()][0]  = nus[0].e()

            if len(pis) > 1:
                branch_vals['%(tau_name)s_pi2_px' % vars()][0] = pis[1].px()
                branch_vals['%(tau_name)s_pi2_py' % vars()][0] = pis[1].py()
                branch_vals['%(tau_name)s_pi2_pz' % vars()][0] = pis[1].pz()
                branch_vals['%(tau_name)s_pi2_e' % vars()][0]  = pis[1].e()
                branch_vals['%(tau_name)s_pi2_vx' % vars()][0] = pis[1].xProd()
                branch_vals['%(tau_name)s_pi2_vy' % vars()][0] = pis[1].yProd()
                branch_vals['%(tau_name)s_pi2_vz' % vars()][0] = pis[1].zProd()
                
            if len(pis) > 2:
                branch_vals['%(tau_name)s_pi3_px' % vars()][0] = pis[2].px()
                branch_vals['%(tau_name)s_pi3_py' % vars()][0] = pis[2].py()
                branch_vals['%(tau_name)s_pi3_pz' % vars()][0] = pis[2].pz()
                branch_vals['%(tau_name)s_pi3_e' % vars()][0]  = pis[2].e()
                branch_vals['%(tau_name)s_pi3_vx' % vars()][0] = pis[2].xProd()
                branch_vals['%(tau_name)s_pi3_vy' % vars()][0] = pis[2].yProd()
                branch_vals['%(tau_name)s_pi3_vz' % vars()][0] = pis[2].zProd()

            if len(pi0s) > 0:
                branch_vals['%(tau_name)s_pizero1_px' % vars()][0] = pi0s[0].px()
                branch_vals['%(tau_name)s_pizero1_py' % vars()][0] = pi0s[0].py()
                branch_vals['%(tau_name)s_pizero1_pz' % vars()][0] = pi0s[0].pz()            
                branch_vals['%(tau_name)s_pizero1_e' % vars()][0]  = pi0s[0].e()
            if len(pi0s) > 1:
                branch_vals['%(tau_name)s_pizero2_px' % vars()][0] = pi0s[1].px()
                branch_vals['%(tau_name)s_pizero2_py' % vars()][0] = pi0s[1].py()
                branch_vals['%(tau_name)s_pizero2_pz' % vars()][0] = pi0s[1].pz()            
                branch_vals['%(tau_name)s_pizero2_e' % vars()][0]  = pi0s[1].e()

            if len(mus) > 0:
                branch_vals['%(tau_name)s_mu_px' % vars()][0] = mus[0].px()
                branch_vals['%(tau_name)s_mu_py' % vars()][0] = mus[0].py()
                branch_vals['%(tau_name)s_mu_pz' % vars()][0] = mus[0].pz()            
                branch_vals['%(tau_name)s_mu_e' % vars()][0]  = mus[0].e()

    taup_vis_px = branch_vals['taup_LHE_pi1_px'][0] + branch_vals['taup_LHE_pi2_px'][0] + branch_vals['taup_LHE_pi3_px'][0] + branch_vals['taup_LHE_pizero1_px'][0] + branch_vals['taup_LHE_pizero2_px'][0] + branch_vals['taup_LHE_mu_px'][0]
    taup_vis_py = branch_vals['taup_LHE_pi1_py'][0] + branch_vals['taup_LHE_pi2_py'][0] + branch_vals['taup_LHE_pi3_py'][0] + branch_vals['taup_LHE_pizero1_py'][0] + branch_vals['taup_LHE_pizero2_py'][0] + branch_vals['taup_LHE_mu_py'][0]
    taun_vis_px = branch_vals['taun_LHE_pi1_px'][0] + branch_vals['taun_LHE_pi2_px'][0] + branch_vals['taun_LHE_pi3_px'][0] + branch_vals['taun_LHE_pizero1_px'][0] + branch_vals['taun_LHE_pizero2_px'][0] + branch_vals['taun_LHE_mu_px'][0]
    taun_vis_py = branch_vals['taun_LHE_pi1_py'][0] + branch_vals['taun_LHE_pi2_py'][0] + branch_vals['taun_LHE_pi3_py'][0] + branch_vals['taun_LHE_pizero1_py'][0] + branch_vals['taun_LHE_pizero2_py'][0] + branch_vals['taun_LHE_mu_py'][0]

    branch_vals['taup_LHE_vis_pt'][0] = (taup_vis_px**2 + taup_vis_py**2)**.5
    branch_vals['taun_LHE_vis_pt'][0] = (taun_vis_px**2 + taun_vis_py**2)**.5

    taus_first_copy = []

    for i, part in enumerate(pythia.event):
        pdgid = part.id()
        mother_ids = [pythia.event[x].id() for x in part.motherList()]
        daughter_ids = [pythia.event[x].id() for x in part.daughterList()]
        #print(pdgid, part.e(), part.charge(), part.status(), mother_ids, daughter_ids)
        LastCopy = IsLastCopy(part, pythia.event)
        FirstCopy = IsFirstCopy(part, pythia.event)

        if pdgid == 11 and len(mother_ids) == 0:
            # the e+ directions defines the z direction
            # not really needed to store this since its always in the -z direction due to how the sample is produced..
            z_x = part.px()
            z_y = part.py()
            z_z = part.pz()
            r = (z_x**2 + z_y**2 + z_z**2)**.5
            z_x/=r
            z_y/=r
            z_z/=r
            branch_vals['z_x' % vars()][0] = z_x
            branch_vals['z_y' % vars()][0] = z_y
            branch_vals['z_z' % vars()][0] = z_z

        if abs(pdgid) == 15 and FirstCopy:
            taus_first_copy.append(part)
            tau_name = 'taun_first' if pdgid == 15 else 'taup_first'
            branch_vals['%(tau_name)s_px' % vars()][0] = part.px()
            branch_vals['%(tau_name)s_py' % vars()][0] = part.py()
            branch_vals['%(tau_name)s_pz' % vars()][0] = part.pz()
            branch_vals['%(tau_name)s_e' % vars()][0]  = part.e()


        if abs(pdgid) == 15 and LastCopy and part.status() > -70: # this status cut should remove taus from hadron decays
            #TraceTauMother(part, pythia.event, mother_ids_to_exclude=[25,23])
            #print('Tau with status %i found: id =%i, px = %.4f, py = %.4f, pz = %.4f, e = %.4f' % 
            #      (part.status(), part.id(), part.px(), part.py(), part.pz(), part.e()))
            pis, pi0s, nus, mus, daughter_pdgids = GetPiDaughters(part,pythia.event)
            tau_name = 'taun' if pdgid == 15 else 'taup'
            branch_vals['%(tau_name)s_px' % vars()][0] = part.px()
            branch_vals['%(tau_name)s_py' % vars()][0] = part.py()
            branch_vals['%(tau_name)s_pz' % vars()][0] = part.pz()
            branch_vals['%(tau_name)s_e' % vars()][0]  = part.e()
            branch_vals['%(tau_name)s_npi' % vars()][0]  = len(pis)
            branch_vals['%(tau_name)s_npizero' % vars()][0]  = len(pi0s)
            branch_vals['%(tau_name)s_nmu' % vars()][0]  = len(mus)

            if len(pis) == 0 and len(mus) == 0:
                print('Warning: no pions or muons found for tau %i' % part.id())
                print('daughter pdgids:')
                print(daughter_pdgids)
                # print the mothers of this tau
                print('mothers:')
                for m in part.motherList():
                    print(pythia.event[m].id())

            # store tau vertex as the pv
            branch_vals['pv_x'][0] = part.xProd()
            branch_vals['pv_y'][0] = part.yProd()
            branch_vals['pv_z'][0] = part.zProd()

            if len(pis) > 0:
                branch_vals['%(tau_name)s_pi1_px' % vars()][0] = pis[0].px()
                branch_vals['%(tau_name)s_pi1_py' % vars()][0] = pis[0].py()
                branch_vals['%(tau_name)s_pi1_pz' % vars()][0] = pis[0].pz()
                branch_vals['%(tau_name)s_pi1_e' % vars()][0]  = pis[0].e()
                branch_vals['%(tau_name)s_pi1_vx' % vars()][0] = pis[0].xProd() # mm units
                branch_vals['%(tau_name)s_pi1_vy' % vars()][0] = pis[0].yProd()
                branch_vals['%(tau_name)s_pi1_vz' % vars()][0] = pis[0].zProd()

            if len(nus) > 0:
                branch_vals['%(tau_name)s_nu_px' % vars()][0] = nus[0].px()
                branch_vals['%(tau_name)s_nu_py' % vars()][0] = nus[0].py()
                branch_vals['%(tau_name)s_nu_pz' % vars()][0] = nus[0].pz()
                branch_vals['%(tau_name)s_nu_e' % vars()][0]  = nus[0].e()

            if len(pis) > 1:
                branch_vals['%(tau_name)s_pi2_px' % vars()][0] = pis[1].px()
                branch_vals['%(tau_name)s_pi2_py' % vars()][0] = pis[1].py()
                branch_vals['%(tau_name)s_pi2_pz' % vars()][0] = pis[1].pz()
                branch_vals['%(tau_name)s_pi2_e' % vars()][0]  = pis[1].e()
                branch_vals['%(tau_name)s_pi2_vx' % vars()][0] = pis[1].xProd()
                branch_vals['%(tau_name)s_pi2_vy' % vars()][0] = pis[1].yProd()
                branch_vals['%(tau_name)s_pi2_vz' % vars()][0] = pis[1].zProd()
                
            if len(pis) > 2:
                branch_vals['%(tau_name)s_pi3_px' % vars()][0] = pis[2].px()
                branch_vals['%(tau_name)s_pi3_py' % vars()][0] = pis[2].py()
                branch_vals['%(tau_name)s_pi3_pz' % vars()][0] = pis[2].pz()
                branch_vals['%(tau_name)s_pi3_e' % vars()][0]  = pis[2].e()
                branch_vals['%(tau_name)s_pi3_vx' % vars()][0] = pis[2].xProd()
                branch_vals['%(tau_name)s_pi3_vy' % vars()][0] = pis[2].yProd()
                branch_vals['%(tau_name)s_pi3_vz' % vars()][0] = pis[2].zProd()

            if len(pi0s) > 0:
                branch_vals['%(tau_name)s_pizero1_px' % vars()][0] = pi0s[0].px()
                branch_vals['%(tau_name)s_pizero1_py' % vars()][0] = pi0s[0].py()
                branch_vals['%(tau_name)s_pizero1_pz' % vars()][0] = pi0s[0].pz()            
                branch_vals['%(tau_name)s_pizero1_e' % vars()][0]  = pi0s[0].e()
            if len(pi0s) > 1:
                branch_vals['%(tau_name)s_pizero2_px' % vars()][0] = pi0s[1].px()
                branch_vals['%(tau_name)s_pizero2_py' % vars()][0] = pi0s[1].py()
                branch_vals['%(tau_name)s_pizero2_pz' % vars()][0] = pi0s[1].pz()            
                branch_vals['%(tau_name)s_pizero2_e' % vars()][0]  = pi0s[1].e()

            if len(mus) > 0:
                branch_vals['%(tau_name)s_mu_px' % vars()][0] = mus[0].px()
                branch_vals['%(tau_name)s_mu_py' % vars()][0] = mus[0].py()
                branch_vals['%(tau_name)s_mu_pz' % vars()][0] = mus[0].pz()            
                branch_vals['%(tau_name)s_mu_e' % vars()][0]  = mus[0].e()

    # sum all visible tau momenta
    taup_vis_px = branch_vals['taup_pi1_px'][0] + branch_vals['taup_pi2_px'][0] + branch_vals['taup_pi3_px'][0] + branch_vals['taup_pizero1_px'][0] + branch_vals['taup_pizero2_px'][0] + branch_vals['taup_mu_px'][0]
    taup_vis_py = branch_vals['taup_pi1_py'][0] + branch_vals['taup_pi2_py'][0] + branch_vals['taup_pi3_py'][0] + branch_vals['taup_pizero1_py'][0] + branch_vals['taup_pizero2_py'][0] + branch_vals['taup_mu_py'][0]
    taup_vis_pz = branch_vals['taup_pi1_pz'][0] + branch_vals['taup_pi2_pz'][0] + branch_vals['taup_pi3_pz'][0] + branch_vals['taup_pizero1_pz'][0] + branch_vals['taup_pizero2_pz'][0] + branch_vals['taup_mu_pz'][0]
    taup_vis_E = branch_vals['taup_pi1_e'][0] + branch_vals['taup_pi2_e'][0] + branch_vals['taup_pi3_e'][0] + branch_vals['taup_pizero1_e'][0] + branch_vals['taup_pizero2_e'][0] + branch_vals['taup_mu_e'][0]
    taun_vis_px = branch_vals['taun_pi1_px'][0] + branch_vals['taun_pi2_px'][0] + branch_vals['taun_pi3_px'][0] + branch_vals['taun_pizero1_px'][0] + branch_vals['taun_pizero2_px'][0] + branch_vals['taun_mu_px'][0]
    taun_vis_py = branch_vals['taun_pi1_py'][0] + branch_vals['taun_pi2_py'][0] + branch_vals['taun_pi3_py'][0] + branch_vals['taun_pizero1_py'][0] + branch_vals['taun_pizero2_py'][0] + branch_vals['taun_mu_py'][0]
    taun_vis_pz = branch_vals['taun_pi1_pz'][0] + branch_vals['taun_pi2_pz'][0] + branch_vals['taun_pi3_pz'][0] + branch_vals['taun_pizero1_pz'][0] + branch_vals['taun_pizero2_pz'][0] + branch_vals['taun_mu_pz'][0]
    taun_vis_E = branch_vals['taun_pi1_e'][0] + branch_vals['taun_pi2_e'][0] + branch_vals['taun_pi3_e'][0] + branch_vals['taun_pizero1_e'][0] + branch_vals['taun_pizero2_e'][0] + branch_vals['taun_mu_e'][0]
    branch_vals['taup_vis_pt'][0] = (taup_vis_px**2 + taup_vis_py**2)**.5
    branch_vals['taun_vis_pt'][0] = (taun_vis_px**2 + taun_vis_py**2)**.5
    m_vis_sq = (taup_vis_E + taun_vis_E)**2 - (taup_vis_px + taun_vis_px)**2 - (taup_vis_py + taun_vis_py)**2 - (taup_vis_pz + taun_vis_pz)**2
    if m_vis_sq < 0: 
        branch_vals['m_vis'][0] = -1
    else:
        branch_vals['m_vis'][0] = m_vis_sq**.5

    # compute met_x, met_y, and met_z but summing neutrinos from both tau decays
    met_px = branch_vals['taup_nu_px'][0] + branch_vals['taun_nu_px'][0]
    met_py = branch_vals['taup_nu_py'][0] + branch_vals['taun_nu_py'][0]
    met_pz = branch_vals['taup_nu_pz'][0] + branch_vals['taun_nu_pz'][0]

    # compute gen impact parameters
    # first need to store PV, tau vertex, and tau pion's direction as TVector3
    pv_vec3 = ROOT.TVector3(branch_vals['pv_x'][0], branch_vals['pv_y'][0], branch_vals['pv_z'][0])
    taup_pi1_vec3 = ROOT.TVector3(branch_vals['taup_pi1_px'][0], branch_vals['taup_pi1_py'][0], branch_vals['taup_pi1_pz'][0])
    taun_pi1_vec3 = ROOT.TVector3(branch_vals['taun_pi1_px'][0], branch_vals['taun_pi1_py'][0], branch_vals['taun_pi1_pz'][0])
    taup_pi1_dir_vec3 = taup_pi1_vec3.Unit()
    taun_pi1_dir_vec3 = taun_pi1_vec3.Unit()
    taup_vtx_vec3 = ROOT.TVector3(branch_vals['taup_pi1_vx'][0], branch_vals['taup_pi1_vy'][0], branch_vals['taup_pi1_vz'][0])
    taun_vtx_vec3 = ROOT.TVector3(branch_vals['taun_pi1_vx'][0], branch_vals['taun_pi1_vy'][0], branch_vals['taun_pi1_vz'][0])

    pca_p_wrt_pv = FindDMin_Point(taup_vtx_vec3, taup_pi1_dir_vec3, pv_vec3)
    pca_n_wrt_pv = FindDMin_Point(taun_vtx_vec3, taun_pi1_dir_vec3, pv_vec3)

    branch_vals['taup_pi1_ipx'][0] = pca_p_wrt_pv.X()
    branch_vals['taup_pi1_ipy'][0] = pca_p_wrt_pv.Y()
    branch_vals['taup_pi1_ipz'][0] = pca_p_wrt_pv.Z()
    branch_vals['taun_pi1_ipx'][0] = pca_n_wrt_pv.X()
    branch_vals['taun_pi1_ipy'][0] = pca_n_wrt_pv.Y()
    branch_vals['taun_pi1_ipz'][0] = pca_n_wrt_pv.Z()

    # get tau's 4 memonta as TLorentzVectors
    taup = ROOT.TLorentzVector(branch_vals['taup_px'][0], branch_vals['taup_py'][0], branch_vals['taup_pz'][0], branch_vals['taup_e'][0])
    taun = ROOT.TLorentzVector(branch_vals['taun_px'][0], branch_vals['taun_py'][0], branch_vals['taun_pz'][0], branch_vals['taun_e'][0])

    branch_vals['boson_mass'][0] = (taup + taun).M()
    branch_vals['boson_pt'][0] = (taup + taun).Pt()


    if args.extra_vars:
        from ReconstructTaus import ReconstructTauAnalytically, FindDMin
        from PolarimetricA1 import PolarimetricA1

        P_boson = taup + taun
        taup_vis_px = branch_vals['taup_pi1_px'][0]+branch_vals['taup_pi2_px'][0]+branch_vals['taup_pi3_px'][0]+branch_vals['taup_pizero1_px'][0]+branch_vals['taup_pizero2_px'][0]+branch_vals['taup_mu_px'][0]
        taup_vis_py = branch_vals['taup_pi1_py'][0]+branch_vals['taup_pi2_py'][0]+branch_vals['taup_pi3_py'][0]+branch_vals['taup_pizero1_py'][0]+branch_vals['taup_pizero2_py'][0]+branch_vals['taup_mu_py'][0]
        taup_vis_pz = branch_vals['taup_pi1_pz'][0]+branch_vals['taup_pi2_pz'][0]+branch_vals['taup_pi3_pz'][0]+branch_vals['taup_pizero1_pz'][0]+branch_vals['taup_pizero2_pz'][0]+branch_vals['taup_mu_pz'][0]
        taup_vis_E = branch_vals['taup_pi1_e'][0] + branch_vals['taup_pi2_e'][0] + branch_vals['taup_pi3_e'][0] + branch_vals['taup_pizero1_e'][0] + branch_vals['taup_pizero2_e'][0] + branch_vals['taup_mu_e'][0]
        taun_vis_px = branch_vals['taun_pi1_px'][0]+branch_vals['taun_pi2_px'][0]+branch_vals['taun_pi3_px'][0]+branch_vals['taun_pizero1_px'][0]+branch_vals['taun_pizero2_px'][0]+branch_vals['taun_mu_px'][0]
        taun_vis_py = branch_vals['taun_pi1_py'][0]+branch_vals['taun_pi2_py'][0]+branch_vals['taun_pi3_py'][0]+branch_vals['taun_pizero1_py'][0]+branch_vals['taun_pizero2_py'][0]+branch_vals['taun_mu_py'][0]
        taun_vis_pz = branch_vals['taun_pi1_pz'][0]+branch_vals['taun_pi2_pz'][0]+branch_vals['taun_pi3_pz'][0]+branch_vals['taun_pizero1_pz'][0]+branch_vals['taun_pizero2_pz'][0]+branch_vals['taun_mu_pz'][0]
        taun_vis_E = branch_vals['taun_pi1_e'][0] + branch_vals['taun_pi2_e'][0] + branch_vals['taun_pi3_e'][0] + branch_vals['taun_pizero1_e'][0] + branch_vals['taun_pizero2_e'][0] + branch_vals['taun_mu_e'][0]
        taup_vis = ROOT.TLorentzVector(taup_vis_px, taup_vis_py, taup_vis_pz, taup_vis_E)
        taun_vis = ROOT.TLorentzVector(taun_vis_px, taun_vis_py, taun_vis_pz, taun_vis_E)

        taup_pi1 = ROOT.TLorentzVector(branch_vals['taup_pi1_px'][0], branch_vals['taup_pi1_py'][0], branch_vals['taup_pi1_pz'][0], branch_vals['taup_pi1_e'][0])
        taun_pi1 = ROOT.TLorentzVector(branch_vals['taun_pi1_px'][0], branch_vals['taun_pi1_py'][0], branch_vals['taun_pi1_pz'][0], branch_vals['taun_pi1_e'][0])

        solutions = ReconstructTauAnalytically(P_boson, taup_vis, taun_vis)
        ip1_3vec = ROOT.TVector3(branch_vals['taup_pi1_ipx'][0], branch_vals['taup_pi1_ipy'][0], branch_vals['taup_pi1_ipz'][0])
        ip2_3vec = ROOT.TVector3(branch_vals['taun_pi1_ipx'][0], branch_vals['taun_pi1_ipy'][0], branch_vals['taun_pi1_ipz'][0])

        solution1_dmin_taup  = FindDMin(pv_vec3, solutions[0][0].Vect().Unit(), ip1_3vec, taup_pi1.Vect().Unit()).Mag()
        solution1_dmin_taun = FindDMin(pv_vec3, solutions[0][1].Vect().Unit(), ip2_3vec, taun_pi1.Vect().Unit()).Mag()
        solution2_dmin_taup  = FindDMin(pv_vec3, solutions[1][0].Vect().Unit(), ip1_3vec, taup_pi1.Vect().Unit()).Mag()
        solution2_dmin_taun = FindDMin(pv_vec3, solutions[1][1].Vect().Unit(), ip2_3vec, taun_pi1.Vect().Unit()).Mag()

        solution1_dmin_tot = (solution1_dmin_taup**2 + solution1_dmin_taun**2)**0.5
        solution2_dmin_tot = (solution2_dmin_taup**2 + solution2_dmin_taun**2)**0.5

        # swap solutions if solution 2 has smaller total dmin to the impact parameters than solution 1
        if solution2_dmin_tot < solution1_dmin_tot:
            temp = solutions[0]
            solutions = (solutions[1], solutions[0])

        # store both solutions
        branch_vals['solution1_taup_px'][0] = solutions[0][0].Px()
        branch_vals['solution1_taup_py'][0] = solutions[0][0].Py()
        branch_vals['solution1_taup_pz'][0] = solutions[0][0].Pz()
        branch_vals['solution1_taup_e'][0]  = solutions[0][0].E()
        branch_vals['solution1_taun_px'][0] = solutions[0][1].Px()
        branch_vals['solution1_taun_py'][0] = solutions[0][1].Py()
        branch_vals['solution1_taun_pz'][0] = solutions[0][1].Pz()
        branch_vals['solution1_taun_e'][0]  = solutions[0][1].E()
        branch_vals['solution2_taup_px'][0] = solutions[1][0].Px()
        branch_vals['solution2_taup_py'][0] = solutions[1][0].Py()
        branch_vals['solution2_taup_pz'][0] = solutions[1][0].Pz()
        branch_vals['solution2_taup_e'][0]  = solutions[1][0].E()
        branch_vals['solution2_taun_px'][0] = solutions[1][1].Px()
        branch_vals['solution2_taun_py'][0] = solutions[1][1].Py()
        branch_vals['solution2_taun_pz'][0] = solutions[1][1].Pz()
        branch_vals['solution2_taun_e'][0]  = solutions[1][1].E()

        taun_npi = branch_vals['taun_npi'][0]
        taun_npizero = branch_vals['taun_npizero'][0]
        taup_npi = branch_vals['taup_npi'][0]
        taup_npizero = branch_vals['taup_npizero'][0]

        if taup_npizero > 0:
            taup_pizero1 = ROOT.TLorentzVector(branch_vals['taup_pizero1_px'][0], branch_vals['taup_pizero1_py'][0], branch_vals['taup_pizero1_pz'][0], branch_vals['taup_pizero1_e'][0])
        else:
            taup_pizero1 = None
        if taun_npizero > 0:
            taun_pizero1 = ROOT.TLorentzVector(branch_vals['taun_pizero1_px'][0], branch_vals['taun_pizero1_py'][0], branch_vals['taun_pizero1_pz'][0], branch_vals['taun_pizero1_e'][0])
        else: 
            taun_pizero1 = None

        if taup_npi == 3:
            taup_pi3 = ROOT.TLorentzVector(branch_vals['taup_pi3_px'][0], branch_vals['taup_pi3_py'][0], branch_vals['taup_pi3_pz'][0], branch_vals['taup_pi3_e'][0])
            taup_pi2 = ROOT.TLorentzVector(branch_vals['taup_pi2_px'][0], branch_vals['taup_pi2_py'][0], branch_vals['taup_pi2_pz'][0], branch_vals['taup_pi2_e'][0])
        else: 
            taup_pi2 = None
            taup_pi3 = None
        if taun_npi == 3:
            taun_pi3 = ROOT.TLorentzVector(branch_vals['taun_pi3_px'][0], branch_vals['taun_pi3_py'][0], branch_vals['taun_pi3_pz'][0], branch_vals['taun_pi3_e'][0])
            taun_pi2 = ROOT.TLorentzVector(branch_vals['taun_pi2_px'][0], branch_vals['taun_pi2_py'][0], branch_vals['taun_pi2_pz'][0], branch_vals['taun_pi2_e'][0])
        else:
            taun_pi2 = None
            taun_pi3 = None

        if taup_npizero >1:
            taup_pizero2 = ROOT.TLorentzVector(branch_vals['taup_pizero2_px'][0], branch_vals['taup_pizero2_py'][0], branch_vals['taup_pizero2_pz'][0], branch_vals['taup_pizero2_e'][0])
        else:
            taup_pizero2 = None
        if taun_npizero >1:
            taun_pizero2 = ROOT.TLorentzVector(branch_vals['taun_pizero2_px'][0], branch_vals['taun_pizero2_py'][0], branch_vals['taun_pizero2_pz'][0], branch_vals['taun_pizero2_e'][0])
        else:
            taun_pizero2 = None

        # define ditau rest frames
        ditau = taup + taun
        boost = -ditau.BoostVector()

        # boost taus and decay products to ditau rest frame
        taup.Boost(boost)
        taun.Boost(boost)
        taup_pi1.Boost(boost)
        taun_pi1.Boost(boost)
        if taup_pi2 is not None: taup_pi2.Boost(boost)
        if taup_pi3 is not None: taup_pi3.Boost(boost)
        if taun_pi2 is not None: taun_pi2.Boost(boost)
        if taun_pi3 is not None: taun_pi3.Boost(boost)
        if taup_pizero1 is not None: taup_pizero1.Boost(boost)
        if taup_pizero2 is not None: taup_pizero2.Boost(boost)
        if taun_pizero1 is not None: taun_pizero1.Boost(boost)
        if taun_pizero2 is not None: taun_pizero2.Boost(boost)

        if taun_npi == 1 and taun_npizero == 0:   
            taun_pi1.Boost(-taun.BoostVector())
            taun_s = taun_pi1.Vect().Unit()
        elif taun_npi == 1 and taun_npizero >= 1:
            q = taun_pi1  - taun_pizero1
            P = taun
            N = taun - taun_pi1 - taun_pizero1
            pv = P.M()*(2*(q*N)*q - q.Mag2()*N) * (1/ (2*(q*N)*(q*P) - q.Mag2()*(N*P)))
            pv.Boost(-taun.BoostVector())
            taun_s = pv.Vect().Unit()             
        elif taun_npi == 3:
            pv =  -PolarimetricA1(taun, taun_pi1, taun_pi2, taun_pi3, +1).PVC()
            pv.Boost(-taun.BoostVector())
            taun_s = pv.Vect().Unit()
        else: 
            print("WARNING: Number of pions not equal to 1 or 3")
            new_tree.Fill() # any missing variables will be filled with 0
            continue

        if taup_npi == 1 and taup_npizero == 0:
            taup_pi1.Boost(-taup.BoostVector())
            taup_s = taup_pi1.Vect().Unit()
        elif taup_npi == 1 and taup_npizero >= 1:
            q = taup_pi1  - taup_pizero1
            P = taup
            N = taup - taup_pi1 - taup_pizero1
            pv = P.M()*(2*(q*N)*q - q.Mag2()*N) * (1/ (2*(q*N)*(q*P) - q.Mag2()*(N*P)))
            pv.Boost(-taup.BoostVector())
            taup_s = pv.Vect().Unit()
        elif taup_npi == 3:
            pv =  -PolarimetricA1(taup, taup_pi1, taup_pi2, taup_pi3, +1).PVC()
            pv.Boost(-taup.BoostVector())
            taup_s = pv.Vect().Unit()
        else: 
            print("WARNING: Number of pions not equal to 1 or 3")
            new_tree.Fill() # any missing variables will be filled with 0
            continue

        # get taus in ditau rest frame - note this is already done above
        taup_COM = taup.Clone()
        taun_COM = taun.Clone()

        p = ROOT.TVector3(0, 0, -1)
        # k is direction of tau+
        k = taup_COM.Vect().Unit()
        n = (p.Cross(k)).Unit()
        cosTheta = p.Dot(k)
        r = (p - (k*cosTheta)).Unit() 

        branch_vals['cosn_plus'][0] = taup_s.Dot(n)
        branch_vals['cosr_plus'][0] = taup_s.Dot(r)
        branch_vals['cosk_plus'][0] = taup_s.Dot(k)
        branch_vals['cosn_minus'][0] = taun_s.Dot(n)
        branch_vals['cosr_minus'][0] = taun_s.Dot(r)
        branch_vals['cosk_minus'][0] = taun_s.Dot(k)
        branch_vals['cosTheta'][0] = cosTheta


    #hepmc_event = HepMC3.GenEvent()
    #hepmc_converter.fill_next_event1(pythia, hepmc_event, count+1)
    #hepmc_writer.write_event(hepmc_event)

    if not stopGenerating: 
        # check if both taun and taup were found (LastCopy), if not print a warning and don't fill the tree
        if branch_vals['taup_e'][0] == 0 or branch_vals['taun_e'][0] == 0:
            print('Warning: taup or taun not found in event %i' % (count+1))
            print('taup_e = %.4f, taun_e = %.4f' % (branch_vals['taup_e'][0], branch_vals['taun_e'][0]))
        #else: 
        tree.Fill()

    if count % 1000 == 0:
        print('Processed %i events' % count)
        tree.AutoSave("SaveSelf")

    count+=1
    if not stopGenerating:
        # make sure next event is valid 
        pythia.next()

# Finalize
#pythia.stat()
#hepmc_writer.close()  

tree.Write()
fout.Close()
print('Finished running pythia')
