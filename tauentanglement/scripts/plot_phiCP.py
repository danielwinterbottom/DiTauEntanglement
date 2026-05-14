import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import numpy as np
import awkward as ak
import argparse
import os

from tauentanglement.utils.acoplanarity_tools import (
    compute_aco_polarimetric,
    get_R_P_vectors_all,
    compute_aco_classic,
    get_ditau_polarimetric_gen,
    get_ditau_polarimetric_reco,
)

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 16})

options = {
    'files':{  # set files here (ones from eval have all info we need)
            'even':  "/vols/cms/lcr119/offline/HiggsCP/DiTauEntanglement/outputs_model_NFlows_LHC_onnorm_reco_May11/output_results_CPEven_GRADIENT.parquet",
            'odd':   "/vols/cms/lcr119/offline/HiggsCP/DiTauEntanglement/outputs_model_NFlows_LHC_onnorm_reco_May11/output_results_CPOdd_GRADIENT.parquet"
    },
    'gen': {
        'label': 'Generator Neutrino',
        'tag':   'POL_GEN',
    },
    'recoRun3': {
        'label': 'Run 3 Reconstruction',
        'tag':   'RecoRun3',
    },
    'recoNu': {
        'label': 'Regressed Neutrino',
        'tag':   'RecoNu_Smeared',
    },
}


def compute_phicp_all(df, option):
    """Compute phiCP for all events in the df (splitting of methods by DM done automatically, vectorised)"""
    df = df.copy()
    if option == 'gen':
        R1, P1, R2, P2 = get_ditau_polarimetric_gen(df)
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2, 0, 0)
    elif option == 'recoNu':
        R1, P1, R2, P2 = get_ditau_polarimetric_reco(df)
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2, 0, 0)
    elif option == 'recoRun3':
        R1, P1, leg1_is_dp = get_R_P_vectors_all(df, tau_prefix='reco_taup')
        R2, P2, leg2_is_dp = get_R_P_vectors_all(df, tau_prefix='reco_taun')
        phiCP = compute_aco_classic(R1, P1, R2, P2, leg1_is_dp, leg2_is_dp)
    df['phiCP'] = np.array(phiCP)
    return df


def load_data():
    cfg = options['files']
    # read = pd.read_pickle if option == 'recoNu' else pd.read_parquet
    read = pd.read_parquet
    return read(cfg['even']), read(cfg['odd'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--option', choices=['gen', 'recoRun3', 'recoNu'],
                        default='recoRun3', help="Reconstruction method to use.")
    parser.add_argument('--output-dir', default='.', help="Directory for output PDFs.")


    args = parser.parse_args()

    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)

    even_df, odd_df = load_data()
    even_df = compute_phicp_all(even_df, args.option)
    odd_df  = compute_phicp_all(odd_df,  args.option)


    for dm_taup, dm_taun in [[0, 0], [0,1], [1,1]]:

        if dm_taup != dm_taun:
            dm_mask = lambda df, p=dm_taup, n=dm_taun: ((df['reco_taup_haspizero'] == p) & (df['reco_taun_haspizero'] == n)) | ((df['reco_taup_haspizero'] == n) & (df['reco_taun_haspizero'] == p))
        else:
            dm_mask = lambda df, p=dm_taup, n=dm_taun: (df['reco_taup_haspizero'] == p) & (df['reco_taun_haspizero'] == n)
        even = even_df[dm_mask(even_df)]
        odd  = odd_df[dm_mask(odd_df)]

        # kin_mask = lambda df: (df['reco_taup_vis_pT'] > 20) & (df['reco_taun_vis_pT'] > 20)
        # even = even[kin_mask(even)]
        # odd = odd[kin_mask(odd)]

        print(f"DM{dm_taup}-DM{dm_taun}: {len(even)} CP even, {len(odd)} CP odd events")
        print(even['phiCP'].describe())
        print(odd['phiCP'].describe())

        fig, ax = plt.subplots(figsize=(8, 6))
        bin_edges = np.linspace(0, 2 * np.pi, 21)
        even_counts, _ = np.histogram(even['phiCP'], bins=bin_edges, density=True)
        odd_counts, _  = np.histogram(odd['phiCP'],  bins=bin_edges, density=True)
        ax.hist(even['phiCP'], bins=bin_edges, histtype='step', label='CP-even',
                density=True, linewidth=2, color='red')
        ax.hist(odd['phiCP'],  bins=bin_edges, histtype='step', label='CP-odd',
                density=True, linewidth=2, color='blue')
        avg = 0.5 * (even_counts + odd_counts)
        asymmetry = np.mean(np.abs(even_counts - odd_counts) / avg)
        ax.set_xlabel(r'$\phi_{CP}$')
        ax.set_title(f'DM{dm_taup} - DM{dm_taun} - {options[args.option]["label"]}')
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 0.28)
        ax.legend()
        ax.text(0.05, 0.95, f'Asymmetry: {asymmetry:.4f}', transform=ax.transAxes,
                verticalalignment='top', fontweight='bold')
        out = f"{args.output_dir}/DM{dm_taup}DM{dm_taun}_{options[args.option]['tag']}.pdf"
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")


if __name__ == '__main__':
    main()
