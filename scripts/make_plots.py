import ROOT
import argparse
import uproot3
import matplotlib.pyplot as plt
import numpy as np

def true_vs_reco_plot(df, variable="cosn_plus", n_bins=100, output_dir="./plots/"):
    # extract values
    true_values = df['true_' + variable]
    ana_pred_values = df['ana_pred_' + variable]
    alt_pred_values = df['alt_pred_' + variable]

    # first make plots comparing distributions
    lower_bound = min(true_values.min(), ana_pred_values.min(), alt_pred_values.min())
    upper_bound = max(true_values.max(), ana_pred_values.max(), alt_pred_values.max())
    bins = np.linspace(lower_bound, upper_bound, n_bins)

    plt.figure(figsize=(10, 6))
    plt.hist(
        true_values,
        bins=bins,
        histtype='step',
        linewidth=2,
        density=True,
        color='black',
        label='True'
    )
    plt.hist(
        ana_pred_values,
        bins=bins,
        histtype='step',
        linewidth=2,
        density=True,
        color='blue',
        label='Analytical Reco'
    )
    plt.hist(
        alt_pred_values,
        bins=bins,
        histtype='step',
        linewidth=2,
        density=True,
        color='red',
        label='NFlow Reco'
    )
    plt.xlabel(variable)
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distributions_{variable}.pdf')
    plt.close()

    # differences
    ana_pred_diff = ana_pred_values - true_values
    alt_pred_diff = alt_pred_values - true_values

    # bin range from alt_pred_diff to cover 98% of distribution
    lower_bound = alt_pred_diff.quantile(0.01)
    upper_bound = alt_pred_diff.quantile(0.99)
    bins = np.linspace(lower_bound, upper_bound, n_bins)

    # compute mean and RMS
    ana_mean = ana_pred_diff.mean()
    ana_rms  = ana_pred_diff.std()

    alt_mean = alt_pred_diff.mean()
    alt_rms  = alt_pred_diff.std()

    plt.figure(figsize=(10, 6))

    plt.hist(
        ana_pred_diff,
        bins=bins,
        histtype='step',
        linewidth=2,
        density=True,
        color='blue',
        label=f'analytical - true\n'
              f'μ = {ana_mean:.3e}, RMS = {ana_rms:.3e}'
    )

    plt.hist(
        alt_pred_diff,
        bins=bins,
        histtype='step',
        linewidth=2,
        density=True,
        color='red',
        label=f'NFlow - true\n'
              f'μ = {alt_mean:.3e}, RMS = {alt_rms:.3e}'
    )

    # draw dashed line at 0
    plt.axvline(0, color='black', linestyle='--', linewidth=1)

    plt.xlabel(f'{variable} (reco - true)')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(f'{output_dir}/true_vs_reco_{variable}.pdf')
    plt.close()

def ComputeZMass(df):
    # compute the Z mass (ditau invariant mass) and add to dataframe
    for x in ['true_', 'ana_pred_', 'alt_pred_']:
        E_plus = df[x + 'tau_plus_E']
        E_minus = df[x + 'tau_minus_E']
        px_plus = df[x + 'tau_plus_px']
        px_minus = df[x + 'tau_minus_px']
        py_plus = df[x + 'tau_plus_py']
        py_minus = df[x + 'tau_minus_py']
        pz_plus = df[x + 'tau_plus_pz']
        pz_minus = df[x + 'tau_minus_pz']
        mass_squared = (E_plus + E_minus)**2 - (px_plus + px_minus)**2 - (py_plus + py_minus)**2 - (pz_plus + pz_minus)**2
        mass_squared = mass_squared.clip(lower=0)  # avoid negative values
        df[x + 'Zmass'] = np.sqrt(mass_squared)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input ROOT file.")
    args = parser.parse_args()

    input_file = args.input

    # load root file into df 
    with uproot3.open(input_file) as file:
        tree = file["tree"]
        df = tree.pandas.df()

    print(df[:5])

    # add true tau mass to dataframe
    df['true_tau_plus_mass'] = 1.777
    df['true_tau_minus_mass'] = 1.777

    # define cosncosn variables etc

    for x in ['true_', 'ana_pred_', 'alt_pred_']:
        df[x + "cosncosn"] = df[x + "cosn_plus"] * df[x + "cosn_minus"]
        df[x + "cosrcosr"] = df[x + "cosr_plus"] * df[x + "cosr_minus"]
        df[x + "coskcosk"] = df[x + "cosk_plus"] * df[x + "cosk_minus"]
        df[x + "cosncosr"] = df[x + "cosn_plus"] * df[x + "cosr_minus"]
        df[x + "cosncosk"] = df[x + "cosn_plus"] * df[x + "cosk_minus"]
        df[x + "cosrcosk"] = df[x + "cosr_plus"] * df[x + "cosk_minus"]
        df[x + "cosrcosn"] = df[x + "cosr_plus"] * df[x + "cosn_minus"]
        df[x + "coskcosn"] = df[x + "cosk_plus"] * df[x + "cosn_minus"]
        df[x + "coskcosr"] = df[x + "cosk_plus"] * df[x + "cosr_minus"]

    df = ComputeZMass(df)

    variables = ["cosn_plus", "cosn_minus",
                 "cosr_plus", "cosr_minus",
                 "cosk_plus", "cosk_minus",
                 "nu_px", "nu_py", "nu_pz", "nu_E",
                 "nubar_px", "nubar_py", "nubar_pz", "nubar_E",
                 "tau_plus_px", "tau_plus_py", "tau_plus_pz", "tau_plus_E", "tau_plus_mass",
                 "tau_minus_px", "tau_minus_py", "tau_minus_pz", "tau_minus_E","tau_minus_mass",
                 "Zmass",
                 "cosncosn", "cosrcosr", "coskcosk",
                 "cosncosr", "cosncosk", "cosrcosk",
                 "cosrcosn", "coskcosn", "coskcosr",
                 ]

    for var in variables:
        true_vs_reco_plot(df,variable=var, n_bins=100, output_dir="./plots/")
