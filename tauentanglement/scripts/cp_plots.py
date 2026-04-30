import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import mplhep as hep
from tauentanglement.utils.acoplanarity_tools import get_R_P_vectors, compute_acoplanarity_angle
import numpy as np

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 16})

CP_even_df = pd.read_parquet("/vols/cms/lcr119/offline/HiggsCP/DiTauEntanglement/prepared_LHC_data/ppToHToTauTau_DM0and1_CPEven/full_onorm_dataframe.parquet")
CP_odd_df = pd.read_parquet("/vols/cms/lcr119/offline/HiggsCP/DiTauEntanglement/prepared_LHC_data/ppToHToTauTau_DM0and1_CPOdd/full_onorm_dataframe.parquet")

def get_phiCP(df, dm_taup, dm_taun):
    # start with only DM0
    df = df[df['taup_haspizero'] == dm_taup]
    df = df[df['taun_haspizero'] == dm_taun]
    print("Number of events with DM tau+ =", dm_taup, "and DM tau- =", dm_taun, "is", len(df))
    R1, P1 = get_R_P_vectors(df, dm=dm_taup, tau_prefix = 'taup')
    R2, P2 = get_R_P_vectors(df, dm=dm_taun, tau_prefix = 'taun')
    if dm_taup == 0:
        method_leg1 = 'IP'
    else:
        method_leg1 = 'DP'
    if dm_taun == 0:
        method_leg2 = 'IP'
    else:
        method_leg2 = 'DP'
    phiCP = compute_acoplanarity_angle(R1, P1, R2, P2, method_leg1, method_leg2, debug=True)
    df['phiCP'] = phiCP
    return df

DM0 DM0
CP_even_df = get_phiCP(CP_even_df, dm_taup=0, dm_taun=0)
print(CP_even_df['phiCP'].describe())
CP_odd_df = get_phiCP(CP_odd_df, dm_taup=0, dm_taun=0)
print(CP_odd_df['phiCP'].describe())
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(CP_even_df['phiCP'], bins=20, histtype='step', label = 'CP-even', density=True, linewidth=2, color='red')
ax.hist(CP_odd_df['phiCP'], bins=20, histtype='step', label = 'CP-odd', density=True, linewidth=2, color='blue')
ax.set_xlabel(r'$\phi_{CP}$')
ax.set_title('DM0 - DM0')
ax.set_xlim(0, 2 * np.pi)
ax.legend()
plt.savefig('DM0DM0.pdf')


# DM1 DM1
CP_even_df = get_phiCP(CP_even_df, dm_taup=1, dm_taun=1)
print(CP_even_df['phiCP'].describe())
CP_odd_df = get_phiCP(CP_odd_df, dm_taup=1, dm_taun=1)
print(CP_odd_df['phiCP'].describe())
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(CP_even_df['phiCP'], bins=20, histtype='step', label = 'CP-even', density=True, linewidth=2, color='red')
ax.hist(CP_odd_df['phiCP'], bins=20, histtype='step', label = 'CP-odd', density=True, linewidth=2, color='blue')
ax.set_xlabel(r'$\phi_{CP}$')
ax.set_title('DM1 - DM1')
ax.set_xlim(0, 2 * np.pi)
ax.legend()
plt.savefig('DM1DM1.pdf')




# DM0 DM1
CP_even_df = get_phiCP(CP_even_df, dm_taup=0, dm_taun=1)
print(CP_even_df['phiCP'].describe())
CP_odd_df = get_phiCP(CP_odd_df, dm_taup=0, dm_taun=1)
print(CP_odd_df['phiCP'].describe())
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(CP_even_df['phiCP'], bins=20, histtype='step', label = 'CP-even', density=True, linewidth=2, color='red')
ax.hist(CP_odd_df['phiCP'], bins=20, histtype='step', label = 'CP-odd', density=True, linewidth=2, color='blue')
ax.set_xlabel(r'$\phi_{CP}$')
ax.set_title('DM0 - DM1')
ax.set_xlim(0, 2 * np.pi)
ax.legend()
plt.savefig('DM0DM1.pdf')




# DM1 DM0
CP_even_df = get_phiCP(CP_even_df, dm_taup=1, dm_taun=0)
print(CP_even_df['phiCP'].describe())
CP_odd_df = get_phiCP(CP_odd_df, dm_taup=1, dm_taun=0)
print(CP_odd_df['phiCP'].describe())
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(CP_even_df['phiCP'], bins=20, histtype='step', label = 'CP-even', density=True, linewidth=2, color='red')
ax.hist(CP_odd_df['phiCP'], bins=20, histtype='step', label = 'CP-odd', density=True, linewidth=2, color='blue')
ax.set_xlabel(r'$\phi_{CP}$')
ax.set_title('DM1 - DM0')
ax.set_xlim(0, 2 * np.pi)
ax.legend()
plt.savefig('DM1DM0.pdf')

