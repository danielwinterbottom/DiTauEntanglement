"""
Plot toy distributions of con and m12 from two NLL-fit output files
(standard entanglement vs no-entanglement) and estimate the p-value
for separating them, assuming data lands at the median of the signal.
"""

import argparse
import numpy as np
import ROOT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--sig-file",
                    default="nll_fits_fast_incdm2/toys_tree.root",
                    help="ROOT file with entanglement toys (signal hypothesis)")
parser.add_argument("--bkg-file",
                    default="nll_fits_fast_incdm2_no_entanglement/toys_tree.root",
                    help="ROOT file with no-entanglement toys (null hypothesis)")
parser.add_argument("--tree", default="toy_tree")
parser.add_argument("--n-bins", type=int, default=50)
parser.add_argument("--outdir", default=".")
args = parser.parse_args()

variables = [
    {"name": "con",  "xlabel": r"$\mathcal{C}$",  "logy": True,  "xrange": (0, 1.0)},
    {"name": "m12",  "xlabel": r"$m_{12}$",        "logy": False, "xrange": (0, None)},
]

def load(path, tree, variables):
    f = ROOT.TFile.Open(path)
    t = f.Get(tree)
    out = {v: np.array([getattr(e, v) for e in t]) for v in variables}
    f.Close()
    return out

print("Loading toys...")
var_names = [v["name"] for v in variables]
sig = load(args.sig_file, args.tree, var_names)
bkg = load(args.bkg_file, args.tree, var_names)

for var in variables:
    sig_vals = sig[var["name"]]
    bkg_vals = bkg[var["name"]]

    all_vals = np.concatenate([sig_vals, bkg_vals])
    lo = var["xrange"][0] if var["xrange"][0] is not None else np.percentile(all_vals, 1)
    hi = var["xrange"][1] if var["xrange"][1] is not None else np.percentile(all_vals, 99)
    bins = np.linspace(lo, hi, args.n_bins + 1)

    sig_counts, _ = np.histogram(sig_vals, bins=bins)
    bkg_counts, _ = np.histogram(bkg_vals, bins=bins)
    sig_norm = sig_counts / sig_counts.sum()
    bkg_norm = bkg_counts / bkg_counts.sum()

    # Median of signal distribution as proxy for "observed data"
    median_sig = np.median(sig_vals)

    # p-value: fraction of null (no-entanglement) toys more extreme than median_sig.
    # "More extreme" means further toward signal median relative to bkg median.
    bkg_median = np.median(bkg_vals)
    if median_sig >= bkg_median:
        p_value = np.mean(bkg_vals >= median_sig)
    else:
        p_value = np.mean(bkg_vals <= median_sig)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 5))

    def filled_step(ax, bins, counts, color, label):
        x = np.repeat(bins, 2)
        y = np.concatenate([[0], np.repeat(counts, 2), [0]])
        ax.fill_between(x, y, color=color, alpha=0.35, step=None)
        ax.step(bins, np.append(counts, 0), where='post',
                color=color, linewidth=1.5, label=label)

    filled_step(ax, bins, sig_norm, 'royalblue', 'With entanglement')
    filled_step(ax, bins, bkg_norm, 'firebrick', 'No entanglement')

    # Arrow at signal median
    ymax = max(sig_norm.max(), bkg_norm.max())
    if var["logy"]:
        ax.set_yscale('log')
        ymin_log = min(sig_norm[sig_norm > 0].min(), bkg_norm[bkg_norm > 0].min())
        yaxis_min = ymin_log * 0.5
        ax.set_ylim(yaxis_min, ymax * 30)
        arrow_bot = yaxis_min
        arrow_top = ymin_log * 50
        text_y = arrow_top
    else:
        arrow_bot = 0
        arrow_top = ymax * 0.28
        text_y = arrow_top
        ax.set_ylim(0, ymax * 1.45)

    ax.annotate('', xy=(median_sig, arrow_bot),
                xytext=(median_sig, arrow_top),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
    ax.text(median_sig, text_y, f'median\n({median_sig:.3f})',
            color='black', fontsize=9, ha='center', va='bottom')

    from matplotlib.patches import Patch
    p_label = f'p(no-entanglement $\\geq$ median) = {p_value:.4f}'
    ax.legend(handles=[
        plt.Line2D([], [], color='royalblue', linewidth=1.5, label='With entanglement'),
        plt.Line2D([], [], color='firebrick',  linewidth=1.5, label='No entanglement'),
        Patch(color='none', label=p_label),
    ], fontsize=10)

    ax.set_xlim(lo, hi)
    ax.set_xlabel(var["xlabel"], fontsize=13)
    ax.set_ylabel('Normalised', fontsize=13)

    fig.tight_layout()
    fname = f"{args.outdir}/toy_separation_{var['name']}.pdf"
    fig.savefig(fname)
    print(f"Saved {fname}  |  {var['name']}: p={p_value:.4f}")
    plt.close(fig)
