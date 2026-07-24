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
    compute_aco_classic_a1a1,
    get_ditau_polarimetric
)

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 16})

options = {
    'files':{  # set files here (ones from eval have all info we need)
'even': 'outputs_model_LHC_TransformerFlow_Hadronic_25e_July15_RERUN/output_results_CPEven.parquet',
'odd': 'outputs_model_LHC_TransformerFlow_Hadronic_25e_July15_RERUN/output_results_CPOdd.parquet',
'mix': 'outputs_model_LHC_TransformerFlow_Hadronic_25e_July15_RERUN/output_results_CPMix.parquet',
'Z': 'outputs_model_LHC_TransformerFlow_Hadronic_25e_July15_RERUN/output_results_Z.parquet',
'sl_even': 'outputs_model_LHC_TransformerFlow_SemiLeptonic_25e_June23_TRIAL2/output_results_CPEven.parquet',
'sl_odd': 'outputs_model_LHC_TransformerFlow_SemiLeptonic_25e_June23_TRIAL2/output_results_CPOdd.parquet',
'sl_mix': None,
'sl_Z': None,
# 'even': 'outputs_NoFlows_June/outputs_Run3_withFastMTT_June24/output_results_CPEven.parquet', # has fastmtt added
# 'odd': 'outputs_NoFlows_June/outputs_Run3_withFastMTT_June24/output_results_CPOdd.parquet', # has fastmtt added
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


def replace_failed_map(df, threshold=1.0):
    """Replace map_pred_* values with the sampled pred_* prediction for events where
    the MAP optimiser failed (spike at 0 GeV)."""
    df = df.copy()
    nu_failed = df["map_pred_nu_E"] < threshold if "map_pred_nu_E" in df.columns else pd.Series(False, index=df.index)
    nubar_failed = df["map_pred_nubar_E"] < threshold if "map_pred_nubar_E" in df.columns else pd.Series(False, index=df.index)
    failed = nu_failed | nubar_failed
    frac_failed = failed.sum() / len(df) if len(df) > 0 else 0.0
    print(f">> replaceFailed: {failed.sum()}/{len(df)} ({frac_failed:.2%}) events had failed MAP estimates, replacing with sampled predictions")

    map_cols = [c for c in df.columns if c.startswith("map_pred_")]
    for map_col in map_cols:
        pred_col = "pred_" + map_col[len("map_pred_"):]
        if pred_col in df.columns:
            df[map_col] = np.where(failed, df[pred_col], df[map_col])
    return df


def compute_phicp_all(df, option, use_map=True):
    # Compute phiCP for all events in the df (splitting of methods by DM done automatically, vectorised)
    df = df.copy()
    if option == 'gen':
        R1, P1, R2, P2 = get_ditau_polarimetric(df, tau_prefix='true', reco_pions=True)
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'recoNu':
        tau_prefix = 'map_pred' if use_map else 'pred'
        R1, P1, R2, P2 = get_ditau_polarimetric(df, tau_prefix=tau_prefix, reco_pions=True)
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'recoRun3':
        R1, P1, leg1_is_dp = get_R_P_vectors_all(df, tau_prefix='taup', use_map=use_map)
        R2, P2, leg2_is_dp = get_R_P_vectors_all(df, tau_prefix='taun', use_map=use_map)
        phiCPmain = compute_aco_classic(R1, P1, R2, P2, leg1_is_dp, leg2_is_dp)
        phiCPa1a1 = compute_aco_classic_a1a1(df)
        phiCP = np.where((df['taup_DM'] == 10) & (df['taun_DM'] == 10), phiCPa1a1, phiCPmain)
    df['phiCP'] = np.array(phiCP)
    return df

def add_DM(df, dm_prefix='reco'):
    for tau in ['taup', 'taun']:
        tau_is_lep = df[f'{dm_prefix}_{tau}_ishadronic'].values == 0
        tau_is_dm0 = (df[f"{dm_prefix}_{tau}_npizero"].values == 0) & (df[f'{dm_prefix}_{tau}_is3prong'] == 0) & (~tau_is_lep)
        tau_is_dm1 = (df[f"{dm_prefix}_{tau}_npizero"].values == 1) & (df[f'{dm_prefix}_{tau}_is3prong'] == 0) & (~tau_is_lep)
        tau_is_dm2 = ((df[f"{dm_prefix}_{tau}_npizero"].values == 1) | (df[f"{dm_prefix}_{tau}_npizero"].values == 2)) & (df[f'{dm_prefix}_{tau}_is3prong'] == 0) & (~tau_is_lep)
        tau_is_dm10 = (df[f"{dm_prefix}_{tau}_npizero"].values == 0) & (df[f'{dm_prefix}_{tau}_is3prong'] == 1) & (~tau_is_lep)
        tau_is_dm11 = (df[f"{dm_prefix}_{tau}_npizero"].values == 1) & (df[f'{dm_prefix}_{tau}_is3prong'] == 1) & (~tau_is_lep)
        df[f'{tau}_DM'] = np.where(tau_is_dm0, 0,
                             np.where(tau_is_dm1, 1,
                                      np.where(tau_is_dm2, 2,
                                               np.where(tau_is_dm10, 10, 
                                                    np.where(tau_is_dm11, 11,
                                                        np.where(tau_is_lep, 100, -1))))))
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


def _load_and_process_one(path, name, option, use_map, dm_pfx, replace_failed, extra_pt_cut=-1):
    """Read a single parquet file, run it through the full pipeline (pt cut, MAP
    failure replacement, DM tagging, phiCP), then immediately drop everything except
    the few columns the plotting loop needs - so only one raw dataframe is ever
    resident in memory at a time, instead of all of even/odd/mix/Z at once."""
    if path is None:
        return None
    print(f'{name} File: {path}')
    df = pd.read_parquet(path)

    # estimate visible pT from sum of reco charged/pizero momenta and apply cut if extra_pt_cut>0
    if extra_pt_cut > 0:
        taup_px = df['reco_taup_charged_px'] + df['reco_taup_pizero1_px']
        taun_px = df['reco_taun_charged_px'] + df['reco_taun_pizero1_px']
        taup_py = df['reco_taup_charged_py'] + df['reco_taup_pizero1_py']
        taun_py = df['reco_taun_charged_py'] + df['reco_taun_pizero1_py']
        vis_pt = np.sqrt((taup_px + taun_px)**2 + (taup_py + taun_py)**2)
        df = df[vis_pt > extra_pt_cut]

    if replace_failed:
        if use_map:
            df = replace_failed_map(df)
        else:
            print(">> Warning: --replaceFailed has no effect when not using MAP estimates")

    df = add_DM(df, dm_prefix=dm_pfx)
    df = compute_phicp_all(df, option, use_map=use_map)
    return df[['taup_DM', 'taun_DM', 'phiCP']].copy()


def load_data(prefix, option, use_map, dm_pfx, replace_failed, extra_pt_cut=-1):
    cfg = options['files']
    even_df = _load_and_process_one(cfg[f'{prefix}even'], 'EVEN', option, use_map, dm_pfx, replace_failed, extra_pt_cut)
    odd_df  = _load_and_process_one(cfg[f'{prefix}odd'],  'ODD',  option, use_map, dm_pfx, replace_failed, extra_pt_cut)
    mix_df  = _load_and_process_one(cfg.get(f'{prefix}mix', None), 'MIX', option, use_map, dm_pfx, replace_failed, extra_pt_cut)
    Z_df    = _load_and_process_one(cfg.get(f'{prefix}Z', None),   'Z',   option, use_map, dm_pfx, replace_failed, extra_pt_cut)
    return even_df, odd_df, mix_df, Z_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--option', choices=['gen', 'recoRun3', 'recoNu'],
                        default='recoRun3', help="Reconstruction method to use.")
    parser.add_argument('--output-dir', default='.', help="Directory for output PDFs.")
    parser.add_argument('--useMLP', action='store_true')
    parser.add_argument('--replaceFailed', action='store_true',
                        help="If using MAP estimate, replace failed MAP predictions with the sampled prediction")
    parser.add_argument('--GENfilter', action='store_true',
                        help="Use true_ prefix for DM/prong masks instead of reco_.")
    parser.add_argument('--hide-errors', action='store_true',
                        help="Hide Poisson error bands on the bins (shown by default).")
    parser.add_argument('--leptonic_mode', default=0, type=int, choices=[0,1,2],
                        help="If 0 use hadronic decay modes, for 1 use semileptonic, for 2 use fully leptonic (not currently supported).")

    args = parser.parse_args()

    if args.leptonic_mode == 2:
        raise NotImplementedError("Fully leptonic mode not currently supported.")

    do_DM10= False

    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.option}", exist_ok=True)

    use_map = not args.useMLP
    dm_pfx = 'true' if args.GENfilter else 'reco'

    even_df, odd_df, mix_df, Z_df = load_data(
        prefix='sl_' if args.leptonic_mode == 1 else '',
        option=args.option, use_map=use_map, dm_pfx=dm_pfx,
        replace_failed=args.replaceFailed,
    )

    if args.leptonic_mode == 1:
        dm_combs = [[100, 0], [100,1], [100,2], [100,10]]
    else: 
        dm_combs = [[0, 0], [0,1], [1,1], [2,2], [1,2], [0,2], [10,10], [0,10], [1,10], [2,10]]

    for dm_taup, dm_taun in dm_combs:

        dm_mask = lambda df, p=dm_taup, n=dm_taun: ((df['taup_DM'] == p) & (df['taun_DM'] == n)) | ((df['taun_DM'] == n) & (df['taup_DM'] == p))
        even = even_df[dm_mask(even_df)]
        odd  = odd_df[dm_mask(odd_df)]

        print(f"DM{dm_taup}-DM{dm_taun}: {len(even)} CP even, {len(odd)} CP odd events")

        fig, ax = plt.subplots(figsize=(8, 6))
        bin_edges = np.linspace(0, 2 * np.pi, 21)
        hide = args.hide_errors
        even_counts = plot_phicp_histogram(ax, even, bin_edges, 'phiCP', 'CP-even', 'red',   hide)
        odd_counts  = plot_phicp_histogram(ax, odd,  bin_edges, 'phiCP', 'CP-odd',  'blue',  hide)
        if mix_df is not None:
            mix = mix_df[dm_mask(mix_df)]
            plot_phicp_histogram(ax, mix, bin_edges, 'phiCP', 'CP-mix', 'green', hide)
        if Z_df is not None:
            Z = Z_df[dm_mask(Z_df)]
            plot_phicp_histogram(ax, Z, bin_edges, 'phiCP', 'Z', 'black', hide)
        avg = 0.5 * (even_counts + odd_counts)
        asymmetry = np.mean(np.abs(even_counts - odd_counts) / avg)

        significance = 0 
        for i in range(len(even_counts)):
            b_est = (odd_counts[i] + even_counts[i])*0.5*4
            #temp = odd_counts[i] - even_counts[i] + (even_counts[i]+b_est)*np.log((even_counts[i]+b_est)/(odd_counts[i]+b_est)) if even_counts[i] > 0 and odd_counts[i] > 0 else 0
            temp = (odd_counts[i] - even_counts[i])**2
            significance += temp
        #significance = np.sqrt(2 * significance)
        significance = np.sqrt(significance)

        

        ax.set_xlabel(r'$\phi_{CP}$')
        ax.set_title(f'DM{dm_taup} - DM{dm_taun} - {options[args.option]["label"]}')
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 0.28)
        ax.legend()
        ax.text(0.05, 0.95, f'Asymmetry: {asymmetry:.4f}', transform=ax.transAxes,
                verticalalignment='top', fontweight='bold')
        ax.text(0.05, 0.85, f'Asymmetry (quadrature): {significance:.4f}', transform=ax.transAxes,
                verticalalignment='top', fontweight='bold')
        out = f"{args.output_dir}/{args.option}/DM{dm_taup}DM{dm_taun}_{options[args.option]['tag']}.pdf"
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
