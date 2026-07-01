import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", default="outputs_model_LHC_TransformerFlow_Hadronic_100e_June22_TRIAL2")
parser.add_argument("--add-cp", action="store_true", help="Also plot CPOdd and CPMix samples (reweighted from CPEven)")
parser.add_argument("--add-zprime", action="store_true", help="Also plot ZPrime sample")
parser.add_argument("--n-bins", type=int, default=50)
parser.add_argument("--ratio", action="store_true", help="Add ratio panel below main plot")
parser.add_argument("--var-prefix", default="map_pred")
parser.add_argument("--no-cuts", action="store_true", help="Skip event selection cuts")
args = parser.parse_args()

_axes = ['n', 'r', 'k']
_wt_cols = [f'wt_hp_{a}_hm_{b}' for a in _axes for b in _axes]

_needed_cols = (
    ['map_pred_cosk_plus', 'map_pred_cosk_minus',
     'map_pred_cosn_plus', 'map_pred_cosn_minus',
     'map_pred_cosr_plus', 'map_pred_cosr_minus'] +
    _wt_cols +
    ['true_taup_npizero', 'true_taun_npizero', 'true_taup_is3prong', 'true_taun_is3prong',
     'reco_taup_npizero', 'reco_taun_npizero', 'reco_taup_is3prong', 'reco_taun_is3prong']
)

def apply_cuts(df, skip=False):
    if skip:
        return df.reset_index(drop=True)
    gen_mask = (df['true_taup_npizero'] <= 2) & (df['true_taun_npizero'] <= 2) & \
               (df['true_taup_is3prong'] == 0) & (df['true_taun_is3prong'] == 0)
    reco_mask = (df['reco_taup_npizero'] < 2) & (df['reco_taun_npizero'] < 2) & \
                (df['reco_taup_is3prong'] == 0) & (df['reco_taun_is3prong'] == 0)
    return df[gen_mask & reco_mask].reset_index(drop=True)

def get_spin_matrix(phiCP_deg):
    delta = np.radians(phiCP_deg)
    return np.array([
        [np.cos(2*delta),  np.sin(2*delta), 0],
        [-np.sin(2*delta), np.cos(2*delta), 0],
        [0, 0, -1]
    ])

def compute_weights(df, C):
    wt = np.ones(len(df), dtype=np.float64)
    for i, a in enumerate(_axes):
        for j, b in enumerate(_axes):
            if C[i, j] != 0:
                wt += C[i, j] * df[f'wt_hp_{a}_hm_{b}'].values
    return np.clip(wt, 0, None)

print("Loading samples...")
df_sig = apply_cuts(pd.read_parquet(f"{args.input_dir}/output_results_UnCorr.parquet", columns=_needed_cols), skip=args.no_cuts)
df_z   = apply_cuts(pd.read_parquet(f"{args.input_dir}/output_results_ZToTauTau.parquet",
                                     columns=[c for c in _needed_cols if c not in _wt_cols]), skip=args.no_cuts)

samples = {}

C_cpeven = get_spin_matrix(0)
samples['CP-even ($H$)'] = (df_sig, compute_weights(df_sig, C_cpeven), 'royalblue')

samples['$Z\\to\\tau\\tau$'] = (df_z, np.ones(len(df_z)), 'black')

if args.add_cp:
    C_cpodd = get_spin_matrix(90)
    C_cpmix = get_spin_matrix(45)
    samples['CP-odd ($A$)']  = (df_sig, compute_weights(df_sig, C_cpodd), 'firebrick')
    samples['CP-mix (45°)']  = (df_sig, compute_weights(df_sig, C_cpmix), 'forestgreen')

if args.add_zprime:
    df_zp = apply_cuts(pd.read_parquet(f"{args.input_dir}/output_results_ZPrimeToTauTau.parquet",
                                        columns=[c for c in _needed_cols if c not in _wt_cols]), skip=args.no_cuts)
    samples["$Z'\\to\\tau\\tau$"] = (df_zp, np.ones(len(df_zp)), 'darkorange')

variables = [
    {
        'name': 'kk',
        'expr': lambda df: df['map_pred_cosk_plus'].values * df['map_pred_cosk_minus'].values,
        'xlabel': r'$\cos\theta^+_k \cdot \cos\theta^-_k$',
        'range': (-1, 1),
        'fname': 'cosk_product',
    },
    {
        'name': 'nn-rr',
        'expr': lambda df: (df['map_pred_cosn_plus'].values * df['map_pred_cosn_minus'].values
                          - df['map_pred_cosr_plus'].values * df['map_pred_cosr_minus'].values),
        'xlabel': r'$\cos\theta^+_n \cdot \cos\theta^-_n - \cos\theta^+_r \cdot \cos\theta^-_r$',
        'range': (-1, 1),
        'fname': 'cosnn_minus_cosrr',
    },
]

for var in variables:
    if args.ratio:
        fig, (ax, ax_ratio) = plt.subplots(
            2, 1, figsize=(7, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax_ratio = None

    ref_counts = None
    ref_label = None
    all_counts = {}

    for label, (df, wt, color) in samples.items():
        vals = var['expr'](df)
        counts, edges = np.histogram(vals, bins=args.n_bins, range=var['range'], weights=wt)
        counts = counts / counts.sum()
        all_counts[label] = counts
        if ref_counts is None:
            ref_counts = counts
            ref_label = label
        ax.step(edges, np.append(counts, counts[-1]), where='post',
                label=label, color=color, linewidth=1.5)

    ax.set_ylabel('Normalised', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(var['range'])

    if ax_ratio is not None:
        for label, (df, wt, color) in samples.items():
            counts = all_counts[label]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(ref_counts > 0, counts / ref_counts, np.nan)
            ax_ratio.step(edges, np.append(ratio, ratio[-1]), where='post',
                          color=color, linewidth=1.5)

        ax_ratio.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
        ax_ratio.set_xlabel(var['xlabel'], fontsize=13)
        ax_ratio.set_ylabel(f'Ratio to\n{ref_label}', fontsize=10)
        ax_ratio.set_xlim(var['range'])
        ax_ratio.set_ylim(0, 2)
        fig.subplots_adjust(hspace=0.05)
    else:
        ax.set_xlabel(var['xlabel'], fontsize=13)

    fig.tight_layout()

    suffix = ''
    if args.add_cp:
        suffix += '_withCP'
    if args.add_zprime:
        suffix += '_withZPrime'
    fname = f"{var['fname']}{suffix}.pdf"
    fig.savefig(fname)
    print(f"Saved {fname}")
    plt.close(fig)
