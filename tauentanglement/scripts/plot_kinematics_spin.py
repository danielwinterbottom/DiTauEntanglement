import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import numpy as np
import argparse
import os


plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 16})


spin_vars = {
    "cosn_plus":  r"$\cos\theta_n^+$",
    "cosr_plus":  r"$\cos\theta_r^+$",
    "cosk_plus":  r"$\cos\theta_k^+$",
    "cosn_minus": r"$\cos\theta_n^-$",
    "cosr_minus": r"$\cos\theta_r^-$",
    "cosk_minus": r"$\cos\theta_k^-$",
    "cosTheta":   r"$\cos\Theta$",
}


pred_taunu_vars = {
    "nubar_E":        r"$E^\bar{\nu}$",
    "nubar_px":       r"$p_x^\bar{\nu}$",
    "nubar_py":       r"$p_y^\bar{\nu}$",
    "nubar_pz":       r"$p_z^\bar{\nu}$",
    "nu_E":           r"$E^{\nu}$",
    "nu_px":          r"$p_x^\nu$",
    "nu_py":          r"$p_y^\nu$",
    "nu_pz":          r"$p_z^\nu$",
    "tau_plus_E":     r"$E^{\tau^+}$",
    "tau_plus_px":    r"$p_x^\tau$",
    "tau_plus_py":    r"$p_y^\tau$",
    "tau_plus_pz":    r"$p_z^\tau$",
    "tau_minus_E":    r"$E^{\tau^-}$",
    "tau_minus_px":   r"$p_x^\tau$",
    "tau_minus_py":   r"$p_y^\tau$",
    "tau_minus_pz":   r"$p_z^\tau$",
    "tau_plus_mass":  r"$m_{\tau^+}$",
    "tau_minus_mass": r"$m_{\tau^-}$",
    "boson_mass":     r"$m_{\tau^+\tau^-}$",
}


spin_density_vars = {
    "C11": r"$\cos\theta_n^+ \cos\theta_n^-$",
    "C22": r"$\cos\theta_r^+ \cos\theta_r^-$",
    "C33": r"$\cos\theta_k^+ \cos\theta_k^-$",
    "C12": r"$\cos\theta_n^+ \cos\theta_r^-$",
    "C13": r"$\cos\theta_n^+ \cos\theta_k^-$",
    "C23": r"$\cos\theta_r^+ \cos\theta_k^-$",
    "C21": r"$\cos\theta_r^+ \cos\theta_n^-$",
    "C31": r"$\cos\theta_k^+ \cos\theta_n^-$",
    "C32": r"$\cos\theta_k^+ \cos\theta_r^-$",
}


spin_density_products = {
    "C11": ("cosn_plus", "cosn_minus"),
    "C22": ("cosr_plus", "cosr_minus"),
    "C33": ("cosk_plus", "cosk_minus"),
    "C12": ("cosn_plus", "cosr_minus"),
    "C13": ("cosn_plus", "cosk_minus"),
    "C23": ("cosr_plus", "cosk_minus"),
    "C21": ("cosr_plus", "cosn_minus"),
    "C31": ("cosk_plus", "cosn_minus"),
    "C32": ("cosk_plus", "cosr_minus"),
}


m_tau = 1.77686
m_boson = 125.0

truth_color = '#C9B583'
pred_color = '#0072B2'
sampled_color = '#CC3311'


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


def add_spin_density_cols(df):
    """Add cos-product spin density columns for every true_/map_pred_/pred_ prefix."""
    df = df.copy()
    for name, (var1, var2) in spin_density_products.items():
        for prefix in ("true_", "map_pred_", "pred_"):
            col1, col2 = f"{prefix}{var1}", f"{prefix}{var2}"
            if col1 in df.columns and col2 in df.columns:
                df[f"{prefix}{name}"] = df[col1] * df[col2]
    return df


def _resolution_diff(true_vals, other_vals):
    """Per-event (other - true) resolution, aligned on the shared event index."""
    common_idx = true_vals.index.intersection(other_vals.index)
    return other_vals.loc[common_idx] - true_vals.loc[common_idx]


def _draw_resolution(ax_res, true_vals, pred_vals, sampled_vals, show_sampled):
    """Draw the per-event (pred - truth) resolution histogram on the right,
    x-range capped to the 1st-99th percentile so outlier tails don't dominate."""
    diff_pred = _resolution_diff(true_vals, pred_vals)
    diff_parts = [diff_pred]
    if show_sampled:
        diff_sampled = _resolution_diff(true_vals, sampled_vals)
        diff_parts.append(diff_sampled)
    combined_diff = pd.concat(diff_parts)
    bin_range = (combined_diff.quantile(0.01), combined_diff.quantile(0.99))
    diff_bins = np.histogram_bin_edges(combined_diff, bins=50, range=bin_range)

    pred_mu, pred_sigma = diff_pred.mean(), diff_pred.std()
    ax_res.axvline(0.0, color="gray", linestyle="dashed", linewidth=1)
    ax_res.hist(diff_pred, bins=diff_bins, histtype="step", density=True, linewidth=2.2, color=pred_color,
                label=fr"Mode ($\mu$={pred_mu:.3f}, $\sigma$={pred_sigma:.3f})")
    if show_sampled:
        sampled_mu, sampled_sigma = diff_sampled.mean(), diff_sampled.std()
        ax_res.hist(diff_sampled, bins=diff_bins, histtype="step", density=True, linewidth=2.2, color=sampled_color,
                    label=fr"Sampled ($\mu$={sampled_mu:.3f}, $\sigma$={sampled_sigma:.3f})")
    ax_res.set_xlabel("Resolution (Pred. - Truth)")
    ax_res.set_ylabel("a.u.")
    ax_res.set_xlim(diff_bins[0], diff_bins[-1])
    ax_res.legend(loc="upper right", frameon=False, fontsize=11)


def _draw_truth_pred_sampled(ax, ax_ratio, ax_res, true_vals, pred_vals, sampled_vals, show_sampled, bins, ymax_mult):
    """Draw the gray-filled truth / blue predicted / orange sampled overlay,
    its pred-over-truth ratio panel, and the per-event resolution panel,
    shared by all two-panel paper plots."""
    true_counts, _ = np.histogram(true_vals, bins=bins)
    true_hist, _ = np.histogram(true_vals, bins=bins, density=True)
    pred_hist, _ = np.histogram(pred_vals, bins=bins, density=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(true_hist > 0, pred_hist / true_hist, np.nan)
        # Relative statistical (Poisson) uncertainty on the truth histogram itself,
        # shown as a band around 1.0 in the ratio panel.
        true_rel_err = np.where(true_counts > 0, 1.0 / np.sqrt(true_counts), np.nan)
    if show_sampled:
        sampled_hist, _ = np.histogram(sampled_vals, bins=bins, density=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            sampled_ratio = np.where(true_hist > 0, sampled_hist / true_hist, np.nan)

    ax.hist(true_vals, bins=bins, histtype="stepfilled", density=True,
            linewidth=0, color=truth_color, alpha=0.3,
            label="Generator Truth")
    if show_sampled:
        ax.hist(sampled_vals, bins=bins, histtype="step", density=True,
                linewidth=2.2, linestyle="solid", color=sampled_color,
                label="TauPolaris (sampled)")
    ax.hist(pred_vals, bins=bins, histtype="step", density=True,
            linewidth=2.2, linestyle="solid", color=pred_color,
            label="TauPolaris (mode)")
    ax.set_ylabel("a.u.")
    ax.set_xlim(bins[0], bins[-1])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * ymax_mult)
    ax.legend(loc="upper right", frameon=False)
    ax.tick_params(labelbottom=False)

    band_lower = np.where(true_counts > 0, 1.0 - true_rel_err, np.nan)
    band_upper = np.where(true_counts > 0, 1.0 + true_rel_err, np.nan)
    ax_ratio.fill_between(bins, np.append(band_lower, band_lower[-1]), np.append(band_upper, band_upper[-1]),
                           step="post", color="lightgrey", alpha=0.6, linewidth=0, zorder=0)
    ax_ratio.axhline(1.0, color="gray", linestyle="dashed", linewidth=1)
    ax_ratio.stairs(ratio, bins, linewidth=2, color=pred_color)
    if show_sampled:
        ax_ratio.stairs(sampled_ratio, bins, linewidth=2, color=sampled_color)
    ax_ratio.set_ylabel("Pred. / Truth")
    ax_ratio.set_xlim(bins[0], bins[-1])
    ax_ratio.set_ylim(0.5, 1.5)

    _draw_resolution(ax_res, true_vals, pred_vals, sampled_vals, show_sampled)


def _var_bins(var, true_vals, pred_vals, sampled_vals, show_sampled, bounded):
    """Histogram bin edges for a two-panel plot: fixed [-1, 1] for bounded
    (cosine) vars, otherwise a data-driven quantile range."""
    if bounded:
        return np.linspace(-1.0, 1.0, 51)
    combined_parts = [true_vals, pred_vals]
    if show_sampled:
        combined_parts.append(sampled_vals)
    combined = pd.concat(combined_parts)
    if var.endswith('_E') or var.endswith('_mass'):
        bin_range = (combined.min(), combined.quantile(0.98))
    else:
        bin_range = (combined.quantile(0.01), combined.quantile(0.99))
    return np.histogram_bin_edges(combined, bins=50, range=bin_range)


def plot_two_panel_vars(df, var_dict, output_dir, subdir, useMAP=True, bounded=False):
    """Truth/pred/sampled overlay + ratio panel for every var in var_dict.
    bounded=True fixes the histogram range to [-1, 1] (spin_vars, spin_density_vars);
    otherwise the range is data-driven (pred_taunu_vars kinematics)."""
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    for var, label in var_dict.items():
        true_col = f"true_{var}"
        pred_col = f"map_pred_{var}" if useMAP else f"pred_{var}"
        sampled_col = f"pred_{var}"

        if pred_col not in df.columns or true_col not in df.columns:
            print(f"Warning: Missing columns for {var}, skipping")
            continue

        # Only overlay the raw sampled prediction when the main line is the MAP estimate;
        # when --sampled is used the main line already is pred_*, so it would be a duplicate.
        show_sampled = useMAP and sampled_col != pred_col and sampled_col in df.columns

        true_vals = df[true_col].dropna()
        pred_vals = df[pred_col].dropna()
        sampled_vals = df[sampled_col].dropna() if show_sampled else None
        ymax_mult = 1.45 if show_sampled else 1.35
        bins = _var_bins(var, true_vals, pred_vals, sampled_vals, show_sampled, bounded)

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1.5, 1.5], hspace=0.05, wspace=0.3)
        ax = fig.add_subplot(gs[0, 0])
        ax_ratio = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_res = fig.add_subplot(gs[:, 1])
        _draw_truth_pred_sampled(ax, ax_ratio, ax_res, true_vals, pred_vals, sampled_vals, show_sampled, bins, ymax_mult)
        ax_ratio.set_xlabel(label)

        fig.savefig(os.path.join(output_dir, subdir, f"{var}.pdf"))
        plt.close(fig)
        print(f">> Saved paper plot {var}.pdf")


def plot_spin_vars(df, output_dir, useMAP=True, tag=''):
    suffix = f'{tag}_' if tag else ''
    plot_two_panel_vars(df, spin_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=True)


def plot_spin_density_vars(df, output_dir, useMAP=True, tag=''):
    df = add_spin_density_cols(df)
    suffix = f'{tag}_' if tag else ''
    plot_two_panel_vars(df, spin_density_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=True)


mass_vars = {"tau_plus_mass", "tau_minus_mass", "boson_mass"}


def plot_mass_vars(df, output_dir, useMAP=True, tag=''):
    """Single-panel plots for the tau/boson masses: their truth is a known
    fixed value (m_tau or m_boson), so there's no truth distribution to
    overlay — just a dashed truth line against the predicted histogram(s)."""
    suffix = f'{tag}_' if tag else ''
    subdir = f'{suffix}paper_plots'
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    for var in mass_vars:
        label = pred_taunu_vars[var]
        true_col = f"true_{var}"
        pred_col = f"map_pred_{var}" if useMAP else f"pred_{var}"
        sampled_col = f"pred_{var}"

        if pred_col not in df.columns:
            print(f"Warning: Missing columns for {var}, skipping")
            continue

        show_sampled = useMAP and sampled_col != pred_col and sampled_col in df.columns
        has_true = true_col in df.columns
        pred_vals = df[pred_col].dropna()
        sampled_vals = df[sampled_col].dropna() if show_sampled else None
        ymax_mult = 1.45 if show_sampled else 1.35

        is_tau_mass = var in ("tau_plus_mass", "tau_minus_mass")
        truth_val = m_tau if is_tau_mass else m_boson
        bin_range = (1.0, 2.5) if is_tau_mass else (m_boson - 50, m_boson + 50)
        bins = np.linspace(bin_range[0], bin_range[1], 51)

        pred_mu, pred_sigma = pred_vals.mean(), pred_vals.std()

        if has_true:
            fig = plt.figure(figsize=(15, 7))
            gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.5], wspace=0.3)
            ax = fig.add_subplot(gs[0, 0])
        else:
            fig, ax = plt.subplots(figsize=(8, 7))

        ax.axvline(truth_val, color=truth_color, linestyle="dashed", linewidth=2.2,
                   label=fr"Generator Truth ($\mu$={truth_val:.2f} GeV)")
        ax.hist(pred_vals, bins=bins, histtype="step", density=True,
                linewidth=2.2, linestyle="solid", color=pred_color,
                label=fr"TauPolaris Predicted ($\mu$={pred_mu:.2f}, $\sigma$={pred_sigma:.2f} GeV)")
        if show_sampled:
            sampled_mu, sampled_sigma = sampled_vals.mean(), sampled_vals.std()
            ax.hist(sampled_vals, bins=bins, histtype="step", density=True,
                    linewidth=2.2, linestyle="solid", color=sampled_color,
                    label=fr"TauPolaris Sampled ($\mu$={sampled_mu:.2f}, $\sigma$={sampled_sigma:.2f} GeV)")
        ax.set_xlabel(label)
        ax.set_ylabel("a.u.")
        ax.set_xlim(bins[0], bins[-1])
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * ymax_mult)
        ax.legend(loc="upper right", frameon=False)

        if has_true:
            ax_res = fig.add_subplot(gs[0, 1])
            true_vals = df[true_col].dropna()
            _draw_resolution(ax_res, true_vals, pred_vals, sampled_vals, show_sampled)

        fig.savefig(os.path.join(output_dir, subdir, f"{var}.pdf"))
        plt.close(fig)
        print(f">> Saved paper plot {var}.pdf")


def paper_plot(df, output_dir, useMAP=True, tag=''):
    suffix = f'{tag}_' if tag else ''
    kinematic_vars = {v: l for v, l in pred_taunu_vars.items() if v not in mass_vars}
    plot_mass_vars(df, output_dir, useMAP=useMAP, tag=tag)
    plot_two_panel_vars(df, kinematic_vars, output_dir, f'{suffix}paper_plots', useMAP=useMAP, bounded=False)


def main():

    parser = argparse.ArgumentParser(description="Plot kinematics and spin variables")
    parser.add_argument("--input", type=str, help="Path to input parquet file")
    parser.add_argument('--useMLP', action='store_true')
    parser.add_argument('--sampled', action='store_true')
    parser.add_argument('--replaceFailed', action='store_true', help="If using MAP estimate, replace failed MAP predictions with the sampled prediction")
    parser.add_argument('--tag', type=str, default='', help="Tag appended to output subdirectory names")
    args = parser.parse_args()
    use_map = not args.useMLP
    if args.sampled:
        use_map = False
    df = pd.read_parquet(args.input)
    output_dir = args.input.replace('.parquet', '')
    print(f">> Loaded {len(df)} events from {args.input}")

    if args.replaceFailed:
        if use_map:
            df = replace_failed_map(df)
        else:
            print(">> Warning: --replaceFailed has no effect when not using MAP estimates")

    plot_spin_vars(df, output_dir, use_map, tag=args.tag)
    paper_plot(df, output_dir, use_map, tag=args.tag)
    plot_spin_density_vars(df, output_dir, use_map, tag=args.tag)


if __name__ == "__main__":
    main()

