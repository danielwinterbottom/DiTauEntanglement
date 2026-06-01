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
'even': '/vols/cms/dw515/DiTauEntanglement_new/DiTauEntanglement/outputs_model_NFlows_LHC_onnorm_reco_mixedCPtraining_HadronicOnly_May27/output_results_CPEven.parquet',
'odd': '/vols/cms/dw515/DiTauEntanglement_new/DiTauEntanglement/outputs_model_NFlows_LHC_onnorm_reco_mixedCPtraining_HadronicOnly_May27/output_results_CPOdd.parquet',
'mix': None
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


def compute_phicp_all(df, option, use_map=True):
    # Compute phiCP for all events in the df (splitting of methods by DM done automatically, vectorised)
    df = df.copy()
    if option == 'gen':
        R1, P1, R2, P2 = get_ditau_polarimetric_gen(df)
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2, 0, 0)
    elif option == 'recoNu':
        R1, P1, R2, P2 = get_ditau_polarimetric_reco(df, useMAP=use_map)
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2, 0, 0)
    elif option == 'recoRun3':
        R1, P1, leg1_is_dp = get_R_P_vectors_all(df, tau_prefix='reco_taup')
        R2, P2, leg2_is_dp = get_R_P_vectors_all(df, tau_prefix='reco_taun')
        phiCP = compute_aco_classic(R1, P1, R2, P2, leg1_is_dp, leg2_is_dp)
    df['phiCP'] = np.array(phiCP)
    return df


def plot_phicp_histogram(ax, data, bin_edges, variable, label, color, hide_errors=False):
    bin_width = bin_edges[1] - bin_edges[0]
    step_x = np.repeat(bin_edges, 2)[1:-1]
    raw, _ = np.histogram(data[variable], bins=bin_edges)
    counts = raw / (raw.sum() * bin_width)
    ax.hist(data[variable], bins=bin_edges, histtype='step', label=label,
            density=True, linewidth=2, color=color)
    if not hide_errors:
        err = np.sqrt(raw) / (raw.sum() * bin_width)
        ax.fill_between(step_x, np.repeat(counts - err, 2),
                        np.repeat(counts + err, 2), alpha=0.25, color=color)
    return counts


def load_data():
    cfg = options['files']
    read = pd.read_parquet
    mix_df = read(cfg['mix']) if cfg.get('mix') is not None else None
    return read(cfg['even']), read(cfg['odd']), mix_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--option', choices=['gen', 'recoRun3', 'recoNu'],
                        default='recoRun3', help="Reconstruction method to use.")
    parser.add_argument('--output-dir', default='.', help="Directory for output PDFs.")
    parser.add_argument('--useMLP', action='store_true')
    parser.add_argument('--GENfilter', action='store_true',
                        help="Use true_ prefix for DM/prong masks instead of reco_.")
    parser.add_argument('--hide-errors', action='store_true',
                        help="Hide Poisson error bands on the bins (shown by default).")

    args = parser.parse_args()

    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.option}", exist_ok=True)

    even_df, odd_df, mix_df = load_data()
    use_map = not args.useMLP
    even_df = compute_phicp_all(even_df, args.option, use_map=use_map)
    odd_df  = compute_phicp_all(odd_df,  args.option, use_map=use_map)
    if mix_df is not None:
        mix_df = compute_phicp_all(mix_df, args.option, use_map=use_map)


    for dm_taup, dm_taun in [[0, 0], [0,1], [1,1], [2,2], [1,2], [0,2]]: # [npi+/-, npi0]

        pfx = 'true_' if args.GENfilter else 'reco_'
        single_prong_mask = lambda df, p=pfx: (df[f'{p}taup_is3prong'] == 0) & (df[f'{p}taun_is3prong'] == 0)
        if dm_taup != dm_taun:
            dm_mask = lambda df, p=dm_taup, n=dm_taun, pfx=pfx: ((df[f'{pfx}taup_npizero'] == p) & (df[f'{pfx}taun_npizero'] == n)) | ((df[f'{pfx}taun_npizero'] == n) & (df[f'{pfx}taup_npizero'] == p))
        else:
            dm_mask = lambda df, p=dm_taup, n=dm_taun, pfx=pfx: (df[f'{pfx}taup_npizero'] == p) & (df[f'{pfx}taun_npizero'] == n)
        even = even_df[dm_mask(even_df)]
        even = even[single_prong_mask(even)]
        odd  = odd_df[dm_mask(odd_df)]
        odd  = odd[single_prong_mask(odd)]

        # kin_mask = lambda df: (df['reco_taup_vis_pT'] > 20) & (df['reco_taun_vis_pT'] > 20)
        # even = even[kin_mask(even)]
        # odd = odd[kin_mask(odd)]

        print(f"DM{dm_taup}-DM{dm_taun}: {len(even)} CP even, {len(odd)} CP odd events")
        print(even['phiCP'].describe())
        print(odd['phiCP'].describe())

        fig, ax = plt.subplots(figsize=(8, 6))
        bin_edges = np.linspace(0, 2 * np.pi, 21)
        hide = args.hide_errors
        even_counts = plot_phicp_histogram(ax, even, bin_edges, 'phiCP', 'CP-even', 'red',   hide)
        odd_counts  = plot_phicp_histogram(ax, odd,  bin_edges, 'phiCP', 'CP-odd',  'blue',  hide)
        if mix_df is not None:
            mix = mix_df[dm_mask(mix_df)]
            mix = mix[single_prong_mask(mix)]
            plot_phicp_histogram(ax, mix, bin_edges, 'phiCP', 'CP-mix', 'green', hide)
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

        # save numpy arrays to remake plots in future
        np.savez(
            f"{args.output_dir}/logs/DM{dm_taup}DM{dm_taun}_{options[args.option]['tag']}.npz",
            even_counts=even_counts,
            odd_counts=odd_counts,
            bin_edges=bin_edges,
        )

if __name__ == '__main__':
    main()
