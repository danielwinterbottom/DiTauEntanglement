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
    "nubar_E":        r"$\bar{\nu}$ $E$",
    "nubar_px":       r"$\bar{\nu}$ $p_x$",
    "nubar_py":       r"$\bar{\nu}$ $p_y$",
    "nubar_pz":       r"$\bar{\nu}$ $p_z$",
    "nu_E":           r"$\nu$ $E$",
    "nu_px":          r"$\nu$ $p_x$",
    "nu_py":          r"$\nu$ $p_y$",
    "nu_pz":          r"$\nu$ $p_z$",
    "tau_plus_E":     r"$\tau^+$ $E$",
    "tau_plus_px":    r"$\tau^+$ $p_x$",
    "tau_plus_py":    r"$\tau^+$ $p_y$",
    "tau_plus_pz":    r"$\tau^+$ $p_z$",
    "tau_minus_E":    r"$\tau^-$ $E$",
    "tau_minus_px":   r"$\tau^-$ $p_x$",
    "tau_minus_py":   r"$\tau^-$ $p_y$",
    "tau_minus_pz":   r"$\tau^-$ $p_z$",
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

reco_vars = {
    "reco_taup_haspizero":    r"$\tau^+$ has $\pi^0$",
    "reco_taun_haspizero":    r"$\tau^-$ has $\pi^0$",
    "reco_taup_ishadronic":   r"$\tau^+$ is hadronic",
    "reco_taun_ishadronic":   r"$\tau^-$ is hadronic",
    "reco_taup_npizero":      r"$\tau^+$ $N_{\pi^0}$",
    "reco_taun_npizero":      r"$\tau^-$ $N_{\pi^0}$",
    "reco_taup_is3prong":     r"$\tau^+$ is 3-prong",
    "reco_taun_is3prong":     r"$\tau^-$ is 3-prong",
    "reco_taup_ismuon":       r"$\tau^+$ is $\mu$",
    "reco_taun_ismuon":       r"$\tau^-$ is $\mu$",
    "reco_taup_iselectron":   r"$\tau^+$ is $e$",
    "reco_taun_iselectron":   r"$\tau^-$ is $e$",
    "reco_taup_pi1_E":        r"$\tau^+$ $\pi_1$ $E$",
    "reco_taup_pi1_px":       r"$\tau^+$ $\pi_1$ $p_x$",
    "reco_taup_pi1_py":       r"$\tau^+$ $\pi_1$ $p_y$",
    "reco_taup_pi1_pz":       r"$\tau^+$ $\pi_1$ $p_z$",
    "reco_taup_pizero1_E":    r"$\tau^+$ $\pi^0_1$ $E$",
    "reco_taup_pizero1_px":   r"$\tau^+$ $\pi^0_1$ $p_x$",
    "reco_taup_pizero1_py":   r"$\tau^+$ $\pi^0_1$ $p_y$",
    "reco_taup_pizero1_pz":   r"$\tau^+$ $\pi^0_1$ $p_z$",
    "reco_taun_pi1_E":        r"$\tau^-$ $\pi_1$ $E$",
    "reco_taun_pi1_px":       r"$\tau^-$ $\pi_1$ $p_x$",
    "reco_taun_pi1_py":       r"$\tau^-$ $\pi_1$ $p_y$",
    "reco_taun_pi1_pz":       r"$\tau^-$ $\pi_1$ $p_z$",
    "reco_taun_pizero1_E":    r"$\tau^-$ $\pi^0_1$ $E$",
    "reco_taun_pizero1_px":   r"$\tau^-$ $\pi^0_1$ $p_x$",
    "reco_taun_pizero1_py":   r"$\tau^-$ $\pi^0_1$ $p_y$",
    "reco_taun_pizero1_pz":   r"$\tau^-$ $\pi^0_1$ $p_z$",
    "reco_taup_pi1_ipx":      r"$\tau^+$ $\pi_1$ $ip_x$",
    "reco_taup_pi1_ipy":      r"$\tau^+$ $\pi_1$ $ip_y$",
    "reco_taup_pi1_ipz":      r"$\tau^+$ $\pi_1$ $ip_z$",
    "reco_taun_pi1_ipx":      r"$\tau^-$ $\pi_1$ $ip_x$",
    "reco_taun_pi1_ipy":      r"$\tau^-$ $\pi_1$ $ip_y$",
    "reco_taun_pi1_ipz":      r"$\tau^-$ $\pi_1$ $ip_z$",
    "reco_taup_charged_E":    r"$\tau^+$ charged $E$",
    "reco_taup_charged_px":   r"$\tau^+$ charged $p_x$",
    "reco_taup_charged_py":   r"$\tau^+$ charged $p_y$",
    "reco_taup_charged_pz":   r"$\tau^+$ charged $p_z$",
    "reco_taun_charged_E":    r"$\tau^-$ charged $E$",
    "reco_taun_charged_px":   r"$\tau^-$ charged $p_x$",
    "reco_taun_charged_py":   r"$\tau^-$ charged $p_y$",
    "reco_taun_charged_pz":   r"$\tau^-$ charged $p_z$",
    "reco_taup_charged_ipx":  r"$\tau^+$ charged $ip_x$",
    "reco_taup_charged_ipy":  r"$\tau^+$ charged $ip_y$",
    "reco_taup_charged_ipz":  r"$\tau^+$ charged $ip_z$",
    "reco_taun_charged_ipx":  r"$\tau^-$ charged $ip_x$",
    "reco_taun_charged_ipy":  r"$\tau^-$ charged $ip_y$",
    "reco_taun_charged_ipz":  r"$\tau^-$ charged $ip_z$",
    "reco_taup_sv_x":         r"$\tau^+$ SV $x$",
    "reco_taup_sv_y":         r"$\tau^+$ SV $y$",
    "reco_taup_sv_z":         r"$\tau^+$ SV $z$",
    "reco_taun_sv_x":         r"$\tau^-$ SV $x$",
    "reco_taun_sv_y":         r"$\tau^-$ SV $y$",
    "reco_taun_sv_z":         r"$\tau^-$ SV $z$",
    "reco_taup_pi2_E":        r"$\tau^+$ $\pi_2$ $E$",
    "reco_taup_pi2_px":       r"$\tau^+$ $\pi_2$ $p_x$",
    "reco_taup_pi2_py":       r"$\tau^+$ $\pi_2$ $p_y$",
    "reco_taup_pi2_pz":       r"$\tau^+$ $\pi_2$ $p_z$",
    "reco_taun_pi2_E":        r"$\tau^-$ $\pi_2$ $E$",
    "reco_taun_pi2_px":       r"$\tau^-$ $\pi_2$ $p_x$",
    "reco_taun_pi2_py":       r"$\tau^-$ $\pi_2$ $p_y$",
    "reco_taun_pi2_pz":       r"$\tau^-$ $\pi_2$ $p_z$",
    "reco_taup_pi3_E":        r"$\tau^+$ $\pi_3$ $E$",
    "reco_taup_pi3_px":       r"$\tau^+$ $\pi_3$ $p_x$",
    "reco_taup_pi3_py":       r"$\tau^+$ $\pi_3$ $p_y$",
    "reco_taup_pi3_pz":       r"$\tau^+$ $\pi_3$ $p_z$",
    "reco_taun_pi3_E":        r"$\tau^-$ $\pi_3$ $E$",
    "reco_taun_pi3_px":       r"$\tau^-$ $\pi_3$ $p_x$",
    "reco_taun_pi3_py":       r"$\tau^-$ $\pi_3$ $p_y$",
    "reco_taun_pi3_pz":       r"$\tau^-$ $\pi_3$ $p_z$",
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



def plotter(df, var_dict, output_dir, subdir, clip=False, useMAP=True):

    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    for var, label in var_dict.items():
        true_col = f"true_{var}"
        pred_col = f"map_pred_{var}" if useMAP else f"pred_{var}"

        combined = pd.concat([df[true_col].dropna() if true_col in df.columns else pd.Series(dtype=float), df[pred_col].dropna() if pred_col in df.columns else pd.Series(dtype=float),])
        if len(combined) > 0:
            if clip:
                if var.endswith('_E') or var.endswith('_mass'):
                    bin_range = (combined.min(), combined.quantile(0.98))
                else:
                    bin_range = (combined.quantile(0.01), combined.quantile(0.99))
            else:
                bin_range = (combined.min(), combined.max())
            bins = np.histogram_bin_edges(combined, bins=50, range=bin_range)
        else:
            bins = 50

        fig, (ax_main, ax_diff) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(label)
        for col, color, legend_label in [(true_col, "#00c04b", "Generator"),(pred_col, "#2979ff", "Predicted")]:
            if col in df.columns:
                ax_main.hist(df[col].dropna(), bins=bins, histtype="step", linewidth=1.5, color=color, label=legend_label)
            else:
                print(f"Warning: Column {col} not found")
        ax_main.set_xlabel(label)
        ax_main.set_ylabel("Events")
        ax_main.set_xlim(bins[0], bins[-1])
        ax_main.legend()
        if clip:
            ax_main.text(0.97, 0.97, "(2% clipped)", ha="right", va="top", fontsize=11, transform=ax_main.transAxes, color="grey")

        if true_col in df.columns and pred_col in df.columns:
            mask = df[true_col].notna() & df[pred_col].notna()
            if clip:
                lo, hi = bin_range
                mask &= df[true_col].between(lo, hi) & df[pred_col].between(lo, hi)
            diff = df.loc[mask, col] - df.loc[mask, true_col]
            diff_bins = np.histogram_bin_edges(diff, bins=50)
            ax_diff.hist(diff, bins=diff_bins, histtype="stepfilled", linewidth=1.5, alpha=0.7, color="#C84B2F", edgecolor="#7a2010")
            ax_diff.set_xlim(diff_bins[0], diff_bins[-1])
        else:
            print(f"Warning: Missing columns, so no diff possible")
        ax_diff.set_xlabel("MAP - Generator (per event)")
        ax_diff.set_ylabel("Events")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, subdir, f"{var}.pdf"))
        plt.close(fig)
        print(f">> Saved {var}.pdf")




def add_spin_density_cols(df, useMAP=True):
    pred_prefix = 'map_pred_' if useMAP else 'pred_'
    print('Adding columns for spin density')
    for name, (var1, var2) in spin_density_products.items():
        for prefix in ("true_", pred_prefix):
            col1 = f"{prefix}{var1}"
            if var2 is not None:
                col2 = f"{prefix}{var2}"
                if col1 in df.columns and col2 in df.columns:
                    df[f"{prefix}{name}"] = df[col1] * df[col2]
            else:
                if col1 in df.columns:
                    df[f"{prefix}{name}"] = df[col1]
    return df


def plot_spin_density_vars(df, output_dir, useMAP, tag=''):
    df = add_spin_density_cols(df, useMAP=useMAP)
    suffix = f'{tag}_' if tag else ''
    plotter(df, spin_density_vars, output_dir, f'{suffix}spin_density', useMAP=useMAP)


def plot_spin_vars(df, output_dir, useMAP, tag=''):
    suffix = f'{tag}_' if tag else ''
    plotter(df, spin_vars, output_dir, f'{suffix}spin_vars', useMAP=useMAP)


def plot_pred_taunu(df, output_dir, useMAP, tag=''):
    suffix = f'{tag}_' if tag else ''
    plotter(df, pred_taunu_vars, output_dir, f'{suffix}pred_taunu', clip=True, useMAP=useMAP)


def plot_reco_vars(df, output_dir, tag=''):
    subdir = f'{tag}_reco_vars' if tag else 'reco_vars'
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    for var, label in reco_vars.items():
        reco_col = var
        stem = var[len("reco_"):]
        true_col = f"true_{stem}"

        combined_parts = []
        for col in (true_col, reco_col):
            if col in df.columns:
                combined_parts.append(df[col].dropna())
        combined = pd.concat(combined_parts) if combined_parts else pd.Series(dtype=float)

        if len(combined) > 0:
            if var.endswith('_E') or var.endswith('_mass'):
                bin_range = (combined.min(), combined.quantile(0.98))
            else:
                bin_range = (combined.quantile(0.01), combined.quantile(0.99))
            bins = np.histogram_bin_edges(combined, bins=50, range=bin_range)
        else:
            bins = 50

        fig, (ax_main, ax_diff) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(label)
        for col, color, legend_label in [(true_col, "#00c04b", "Generator"), (reco_col, "#2979ff", "Reconstructed")]:
            if col in df.columns:
                ax_main.hist(df[col].dropna(), bins=bins, histtype="step", linewidth=1.5, color=color, label=legend_label)
            else:
                print(f"Warning: Column {col} not found")
        ax_main.set_xlabel(label)
        ax_main.set_ylabel("Events")
        ax_main.set_xlim(bins[0], bins[-1])
        ax_main.legend()
        ax_main.text(0.97, 0.97, "(2% clipped)", ha="right", va="top", fontsize=11, transform=ax_main.transAxes, color="grey")

        if true_col in df.columns and reco_col in df.columns:
            mask = df[true_col].notna() & df[reco_col].notna()
            lo, hi = bin_range
            mask &= df[true_col].between(lo, hi) & df[reco_col].between(lo, hi)
            diff = df.loc[mask, reco_col] - df.loc[mask, true_col]
            diff_bins = np.histogram_bin_edges(diff, bins=50)
            ax_diff.hist(diff, bins=diff_bins, histtype="stepfilled", linewidth=1.5, alpha=0.7, color="#C84B2F", edgecolor="#7a2010")
            ax_diff.set_xlim(diff_bins[0], diff_bins[-1])
        else:
            print(f"Warning: Missing columns, so no diff possible")
        ax_diff.set_xlabel("Reconstructed - Generator (per event)")
        ax_diff.set_ylabel("Events")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, subdir, f"{var}.pdf"))
        plt.close(fig)
        print(f">> Saved {var}.pdf")


def main():

    parser = argparse.ArgumentParser(description="Plot kinematics and spin variables")
    parser.add_argument("--input", type=str, help="Path to input parquet file")
    parser.add_argument('--useMLP', action='store_true')
    parser.add_argument('--sampled', action='store_true')
    parser.add_argument('--tag', type=str, default='', help="Tag appended to output subdirectory names")
    args = parser.parse_args()
    use_map = not args.useMLP
    if args.sampled:
        use_map = False
    df = pd.read_parquet(args.input)
    output_dir = args.input.replace('.parquet', '')
    print(f">> Loaded {len(df)} events from {args.input}")

    # # neutrino cut
    # pred_col = f"map_pred_nu_E" if use_map else f"pred_{var}"
    # mask = df[pred_col] < 5
    # # mask = (df[pred_col] > 5) & (df['map_pred_nubar_E'] > 5)
    # df = df[mask]

    plot_reco_vars(df, output_dir, tag=args.tag)
    plot_spin_vars(df, output_dir, use_map, tag=args.tag)
    plot_pred_taunu(df, output_dir, use_map, tag=args.tag)
    plot_spin_density_vars(df, output_dir, use_map, tag=args.tag)





if __name__ == "__main__":
    main()

