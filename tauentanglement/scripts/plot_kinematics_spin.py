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


def plot_spin_density_vars(df, output_dir, useMAP):
    df = add_spin_density_cols(df, useMAP=useMAP)
    plotter(df, spin_density_vars, output_dir, 'spin_density', useMAP=useMAP)


def plot_spin_vars(df, output_dir, useMAP):
    plotter(df, spin_vars, output_dir, 'spin_vars', useMAP=useMAP)


def plot_pred_taunu(df, output_dir, useMAP):
    plotter(df, pred_taunu_vars, output_dir, 'pred_taunu', clip=True, useMAP=useMAP)



def main():

    parser = argparse.ArgumentParser(description="Plot kinematics and spin variables")
    parser.add_argument("--input", type=str, help="Path to input parquet file")
    parser.add_argument('--useMLP', action='store_true')
    args = parser.parse_args()
    use_map = not args.useMLP
    df = pd.read_parquet(args.input)
    output_dir = args.input.replace('.parquet', '')
    print(f">> Loaded {len(df)} events from {args.input}")

    plot_spin_vars(df, output_dir, use_map)
    plot_pred_taunu(df, output_dir, use_map)
    plot_spin_density_vars(df, output_dir, use_map)



if __name__ == "__main__":
    main()

