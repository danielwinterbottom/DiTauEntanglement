import pythia8
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
#from pyHepMC3 import HepMC3
#from Pythia8ToHepMC3 import Pythia8ToHepMC3
import ROOT
from array import array

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'LHE file to be converted')
parser.add_argument('--output', '-o', help= 'Name of output file',default='pythia_output.hepmc')
parser.add_argument('--cmnd_file', '-c', help= 'Pythia8 command file')
parser.add_argument('--n_events', '-n', help= 'Maximum number of events to process', default=-1, type=int)
parser.add_argument('--n_skip', '-s', help= 'skip n_events*n_skip', default=0, type=int)
parser.add_argument('--seed', help= 'Random seed for Pythia', default=1, type=int)

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
]


if '.hepmc' in args.output: root_output = args.output.replace('.hepmc','.root')
else: root_output=args.output+'.root'
fout = ROOT.TFile(root_output,'RECREATE')
tree = ROOT.TTree('tree','')

branch_vals = {}
for b in branches:
    branch_vals[b] = array('f',[0])
    tree.Branch(b,  branch_vals[b],  '%s/F' % b)
    if b.startswith('taup_') or b.startswith('taun_'):
        # add also branches for first copy
        b_first_name = b.replace('taup','taup_LHE').replace('taun','taun_LHE')
        branch_vals[b_first_name] = array('f',[0])
        tree.Branch(b_first_name,  branch_vals[b_first_name],  '%s/F' % b_first_name)


pythia = pythia8.Pythia("")
pythia.readFile(args.cmnd_file)

if args.input:
    pythia.readString("Beams:frameType = 4")
    pythia.readString("Beams:LHEF = %s" % args.input)
elif False:
    print('Producing full event in pythia')
    # if no LHE file given then produce full event setup using pythia for the hard process as well
    pythia.readString("Beams:idA = -11") # Positron
    pythia.readString("Beams:idB = 11")  # Electron
    pythia.readString("Beams:eCM = 91.2")  # Center-of-mass energy = Z resonance
    pythia.readString("TauDecays:externalMode = 1")

    # Enable Z production and decay to taus
    pythia.readString("WeakSingleBoson:ffbar2gmZ = on")
    pythia.readString("23:onMode = off")  # Turn off all Z decays
    pythia.readString("23:onIfAny = 15")  # Enable Z -> tau+ tau-
else: # for pp->H->tautau using pythia
    print('Producing full pp event in pythia')
    pythia.readString("Beams:idA = 2212") # Proton
    pythia.readString("Beams:idB = 2212")  # Proton
    pythia.readString("Beams:eCM = 13600")  # Center-of-mass energy
    pythia.readString("TauDecays:externalMode = 0")
    # Enable H production and decay to taus

    pythia.readString("HiggsSM:gg2H = on")

    # Force Higgs to decay only to tau+ tau-
    pythia.readString("25:onMode = off")
    pythia.readString("25:onIfAny = 15")

    # Force taus to decay only to pi± nu
    pythia.readString("15:onMode = off")               # turn off all τ− decays
    pythia.readString("15:onIfMatch = -211 16")        # τ− → π− ν_τ


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
    for i, part in enumerate(pythia.process):
        pdgid = part.id()
        if abs(pdgid) == 15:

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

    for i, part in enumerate(pythia.event):
        pdgid = part.id()
        mother_ids = [pythia.event[x].id() for x in part.motherList()]
        daughter_ids = [pythia.event[x].id() for x in part.daughterList()]
        #print(pdgid, part.e(), part.charge(), part.status(), mother_ids, daughter_ids)
        LastCopy = IsLastCopy(part, pythia.event)

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


        if abs(pdgid) == 15 and LastCopy:
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


    #hepmc_event = HepMC3.GenEvent()
    #hepmc_converter.fill_next_event1(pythia, hepmc_event, count+1)
    #hepmc_writer.write_event(hepmc_event)

    if not stopGenerating: tree.Fill()

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
