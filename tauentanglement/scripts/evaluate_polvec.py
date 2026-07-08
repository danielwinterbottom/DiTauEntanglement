"""
Evaluation script for the direct polarimetric-vector / tau-momentum flow
(config_polvec.yaml style models: 12 outputs = ts_hh_taup/taun (polarimetric
vectors) + undecayed_taup/taun (tau momenta)).

Compares true vs. predicted (MAP estimate):
  - tau 4-vectors (px, py, pz, E)
  - polarimetric vectors (x, y, z + angular separation)
  - phiCP
  - entanglement / spin-correlation "cos" variables (the B+/B-/C matrix)

Deliberately kept separate from evaluate.py (which handles the older
neutrino-regression models and already has a lot of leptonic-mode/coordinate
branching) -- this one only has to handle one fixed 12-value output layout.

Test data is read from data_config['test_dataset']/['test_output_name'] (single
string or parallel lists), the same convention as evaluate.py, via
DataProcessing.get_test_dataset.

Run (from the DiTauEntanglement directory):
    python3 tauentanglement/scripts/evaluate_polvec.py --config config_polvec.yaml --max_events 2000
"""
import argparse
import os
import numpy as np
import pandas as pd
import awkward as ak
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tauentanglement.python.DataProcessing import RegressionDataset, get_test_dataset
from tauentanglement.python.NN_Tools import load_model, get_device
from tauentanglement.python.Evaluation_Tools import flow_map_predict, plot_spin_density_matrix
from tauentanglement.utils.coordinate_conversions import ConvertFromOrthonormalNRK_Predictions_PolVec
from tauentanglement.utils.kinematic_helpers import compute_spin_density_vars, boost, boost_vector
from tauentanglement.utils.acoplanarity_tools import compute_aco_polarimetric

M_TAU = 1.77686
PHICP_BINS = 20  # phiCP always uses 20 bins, independent of --num_bins

CARTESIAN_OUTPUT_ORDER = [
    'ts_hh_taup_x', 'ts_hh_taup_y', 'ts_hh_taup_z',
    'ts_hh_taun_x', 'ts_hh_taun_y', 'ts_hh_taun_z',
    'undecayed_taup_px', 'undecayed_taup_py', 'undecayed_taup_pz',
    'undecayed_taun_px', 'undecayed_taun_py', 'undecayed_taun_pz',
]
ONORM_OUTPUT_ORDER = [
    'ts_hh_taup_n', 'ts_hh_taup_r', 'ts_hh_taup_k',
    'ts_hh_taun_n', 'ts_hh_taun_r', 'ts_hh_taun_k',
    'undecayed_taup_n', 'undecayed_taup_r', 'undecayed_taup_k',
    'undecayed_taun_n', 'undecayed_taun_r', 'undecayed_taun_k',
]


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _to_vec3(arr):
    """(N,3) numpy array -> awkward Vector3D, for acoplanarity_tools.compute_aco_polarimetric."""
    return ak.zip({"x": arr[:, 0], "y": arr[:, 1], "z": arr[:, 2]}, with_name="Vector3D")


def compute_phiCP(h1, n1, h2, n2):
    """
    phiCP via acoplanarity_tools.compute_aco_polarimetric (same arXiv:1811.03969
    eq. 3-4 formula validated against this codebase's stored true_cosn/r/k and
    against TauSpinner CP-mixing weights earlier in this project). R1=h1/P1=n1
    and R2=h2/P2=n2 follow that function's own convention (R1=tau+ polarimetric,
    P1=tau+ direction in compute_aco_classic_a1a1) -- doesn't matter which
    physical tau is "1" vs "2" as long as the same labelling is used for both
    the true and predicted vectors being compared.

    h1, n1, h2, n2 : (N,3) numpy arrays
    """
    phicp = compute_aco_polarimetric(_to_vec3(h1), _to_vec3(n1), _to_vec3(h2), _to_vec3(n2))
    return ak.to_numpy(phicp)


def tau_directions_com(taup_p3, taun_p3, taup_E, taun_E):
    """Boost tau+/tau- to their common (ditau) rest frame; return unit direction vectors."""
    taup4 = np.column_stack([taup_E, taup_p3])
    taun4 = np.column_stack([taun_E, taun_p3])
    com_boost = boost_vector(taup4 + taun4)
    taup4_com = boost(taup4, -com_boost)
    kx = unit(taup4_com[:, 1:])
    return kx, -kx  # taup direction, taun direction


def plot_true_vs_pred_2d(true_vals, pred_vals, labels, suptitle, outpath, num_bins, sym_range=None):
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
    if len(labels) == 1:
        axes = [axes]
    for ax, label, tv, pv in zip(axes, labels, true_vals, pred_vals):
        if sym_range == 'auto':
            lim = np.percentile(np.abs(tv), 99)
            rng = [[-lim, lim], [-lim, lim]]
        elif sym_range is not None:
            rng = [list(sym_range), list(sym_range)]
        else:
            rng = None
        ax.hist2d(tv, pv, bins=num_bins, range=rng, cmin=1)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
        ax.set_xlabel(f'true {label}')
        ax.set_ylabel(f'pred {label}')
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def add_DM(df, dm_prefix='reco'):
    """Per-tau decay-mode code, same categorisation/convention as plot_phiCP.py's
    add_DM: 0=1-prong no pi0, 1=1-prong+1pi0, 2=1-prong+2pi0, 10=3-prong, 11=3-prong+pi0,
    100=leptonic, -1=unmatched. Returns (dm_taup, dm_taun) arrays."""
    dm = {}
    for tau in ['taup', 'taun']:
        is_lep = df[f'{dm_prefix}_{tau}_ishadronic'].values == 0
        is_dm0 = (df[f'{dm_prefix}_{tau}_npizero'].values == 0) & (df[f'{dm_prefix}_{tau}_is3prong'].values == 0) & (~is_lep)
        is_dm1 = (df[f'{dm_prefix}_{tau}_npizero'].values == 1) & (df[f'{dm_prefix}_{tau}_is3prong'].values == 0) & (~is_lep)
        is_dm2 = ((df[f'{dm_prefix}_{tau}_npizero'].values == 1) | (df[f'{dm_prefix}_{tau}_npizero'].values == 2)) & (df[f'{dm_prefix}_{tau}_is3prong'].values == 0) & (~is_lep)
        is_dm10 = (df[f'{dm_prefix}_{tau}_npizero'].values == 0) & (df[f'{dm_prefix}_{tau}_is3prong'].values == 1) & (~is_lep)
        is_dm11 = (df[f'{dm_prefix}_{tau}_npizero'].values == 1) & (df[f'{dm_prefix}_{tau}_is3prong'].values == 1) & (~is_lep)
        dm[tau] = np.where(is_dm0, 0,
                    np.where(is_dm1, 1,
                        np.where(is_dm2, 2,
                            np.where(is_dm10, 10,
                                np.where(is_dm11, 11,
                                    np.where(is_lep, 100, -1))))))
    return dm['taup'], dm['taun']


def asymmetry_quadrature(counts_a, counts_b):
    """Same 'Asymmetry (quadrature)' metric as plot_phiCP.py: sqrt(sum((a-b)^2))
    over the (density-normalized) per-bin counts of two phiCP histograms."""
    return np.sqrt(np.sum((counts_a - counts_b) ** 2))


def plot_phiCP_cp_comparison(true_phiCP, pred_phiCP, w_cpeven, w_cpodd, title, outpath, num_bins):
    """The true/predicted x CP-even/CP-odd 4-curve phiCP comparison, factored out
    so it can be reused both for the full sample and per decay-mode combination."""
    fig, ax = plt.subplots(figsize=(7, 6))
    bins = np.linspace(0, 2 * np.pi, num_bins + 1)
    true_even_counts, _, _ = ax.hist(true_phiCP, bins=bins, weights=w_cpeven, density=True, histtype='step',
            linewidth=1.5, linestyle='--', color='steelblue', label='true, CP-even')
    pred_even_counts, _, _ = ax.hist(pred_phiCP, bins=bins, weights=w_cpeven, density=True, histtype='step',
            linewidth=1.5, color='steelblue', label='predicted, CP-even')
    true_odd_counts, _, _ = ax.hist(true_phiCP, bins=bins, weights=w_cpodd, density=True, histtype='step',
            linewidth=1.5, linestyle='--', color='tomato', label='true, CP-odd')
    pred_odd_counts, _, _ = ax.hist(pred_phiCP, bins=bins, weights=w_cpodd, density=True, histtype='step',
            linewidth=1.5, color='tomato', label='predicted, CP-odd')

    true_asym = asymmetry_quadrature(true_even_counts, true_odd_counts)
    pred_asym = asymmetry_quadrature(pred_even_counts, pred_odd_counts)
    ax.text(0.05, 0.95, f'Asymmetry (quadrature), true: {true_asym:.4f}', transform=ax.transAxes,
            verticalalignment='top', fontweight='bold', fontsize=9)
    ax.text(0.05, 0.89, f'Asymmetry (quadrature), pred: {pred_asym:.4f}', transform=ax.transAxes,
            verticalalignment='top', fontweight='bold', fontsize=9)

    ax.set_xlabel(r'$\phi_{CP}$ [rad]')
    ax.set_ylabel('a.u.')
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', required=True, help='path to the configuration file')
    argparser.add_argument('--useCPU', action='store_true', help='force CPU evaluation')
    argparser.add_argument('--max_events', type=int, default=5000, help='number of test events to evaluate on')
    argparser.add_argument('--num_bins', type=int, default=50)
    argparser.add_argument('--oneprong', action='store_true', help='whether to only evaluate on 1-prong taus only')
    argparser.add_argument('--threeprong', action='store_true', help='whether to only evaluate on events with at least 1 3-prong tau')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']
    nn_config = config['SetupNN']

    device = torch.device('cpu') if args.useCPU else get_device()
    print(f">> Using device: {device}")

    output_dir = f"outputs_{nn_config['model_name']}"

    input_features = data_config['Features']['input_features']
    coordinates = data_config['coordinates']
    output_features = data_config['Features']['output_features'][coordinates]
    leptonic_mode = data_config.get('leptonic_mode', -1)
    norm_data = np.load(f'{output_dir}/normalization_params.npz')

    # --- load model (same for every test_dataset below) ---
    model = load_model(nn_config['hyperparams'], input_features, output_features,
                        useTransformer=nn_config.get('use_transformer', True), leptonic_mode=leptonic_mode)
    # prefer the final saved model (written once training completes); fall back to
    # the best/partial checkpoints under plots/ (same convention as evaluate.py) so
    # this also works against a still-training or interrupted run.
    output_plots_dir = os.path.join(output_dir, 'plots')
    model_path = os.path.join(output_dir, f"{nn_config['model_name']}.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(output_plots_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(output_plots_dir, 'partial_model.pth')
    print(f">> Loading model weights from {model_path}")
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.float().to(device)
    model.eval()

    # --- resolve test_dataset / test_output_name (single string or parallel lists),
    # same convention as evaluate.py ---
    if isinstance(data_config['test_dataset'], list):
        test_datasets = data_config['test_dataset']
        if not isinstance(data_config['test_output_name'], list):
            raise ValueError("If test_dataset is a list, test_output_name must also be a list")
        if len(data_config['test_output_name']) != len(test_datasets):
            raise ValueError("test_output_name must have the same length as test_dataset")
        test_output_names = data_config['test_output_name']
    else:
        test_datasets = [data_config['test_dataset']]
        if isinstance(data_config['test_output_name'], list):
            raise ValueError("If test_dataset is not a list, test_output_name must not be a list")
        test_output_names = [data_config['test_output_name']]

    for test_dataset_path, test_output_name in zip(test_datasets, test_output_names):
        print(f">> Evaluating on test dataset {test_dataset_path} (output name: {test_output_name})")
        data_config['test_dataset'] = test_dataset_path
        dataset, df, _, _ = get_test_dataset(data_config, norm_data, oneprong=args.oneprong, threeprong=args.threeprong)

        if args.max_events is not None and len(df) > args.max_events:
            df = df.iloc[:args.max_events].reset_index(drop=True)
            dataset = RegressionDataset(
                df, input_features, output_features,
                normalize_inputs=True, normalize_outputs=True,
                input_mean=torch.from_numpy(norm_data['input_mean']),
                input_std=torch.from_numpy(norm_data['input_std']),
                output_mean=torch.from_numpy(norm_data['output_mean']),
                output_std=torch.from_numpy(norm_data['output_std']),
            )
        print(f">> Evaluating on {len(df)} events")

        outdir = os.path.join(output_dir, 'plots_eval_polvec', test_output_name)
        os.makedirs(outdir, exist_ok=True)

        # --- MAP prediction, in the model's native (training) coordinate space ---
        X, _ = dataset[:]
        X = X.to(device)
        print(f">> Running MAP prediction (method={nn_config.get('map_method', 'gradient')})...")
        chunk_size = 50000 #nn_config.get('chunk_size', 20000 if device.type == 'cpu' else 10000)
        _, predictions_native = flow_map_predict(
            model, X, test_dataset=dataset,
            method=nn_config.get('map_method', 'gradient'),
            num_draws=nn_config.get('map_num_draws', 100),
            chunk_size=chunk_size
        )
        pred_native_df = pd.DataFrame(predictions_native, columns=output_features)

        # --- convert predictions to Cartesian (true values are already Cartesian in df) ---
        if coordinates == 'onorm':
            pred_cart = ConvertFromOrthonormalNRK_Predictions_PolVec(
                pred_native_df[ONORM_OUTPUT_ORDER].values,
                reco_taup_charged=df[['reco_taup_charged_px', 'reco_taup_charged_py', 'reco_taup_charged_pz']].values,
                reco_taup_pizero=df[['reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz']].values,
                reco_taun_charged=df[['reco_taun_charged_px', 'reco_taun_charged_py', 'reco_taun_charged_pz']].values,
                reco_taun_pizero=df[['reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz']].values,
            )
            pred_cart_df = pd.DataFrame(pred_cart, columns=CARTESIAN_OUTPUT_ORDER)
        elif coordinates == 'standard':
            pred_cart_df = pred_native_df[CARTESIAN_OUTPUT_ORDER].reset_index(drop=True)
        else:
            raise ValueError(f"coordinates='{coordinates}' not supported by this script (only 'standard'/'onorm')")

        true_cart_df = df[CARTESIAN_OUTPUT_ORDER].reset_index(drop=True)

        # === 1. tau 4-vector comparison ===
        print(">> Plotting tau 4-vector comparisons...")
        for tau in ['taup', 'taun']:
            true_p = true_cart_df[[f'undecayed_{tau}_px', f'undecayed_{tau}_py', f'undecayed_{tau}_pz']].values
            pred_p = pred_cart_df[[f'undecayed_{tau}_px', f'undecayed_{tau}_py', f'undecayed_{tau}_pz']].values
            true_E = df[f'undecayed_{tau}_e'].values
            pred_E = np.sqrt(np.sum(pred_p ** 2, axis=1) + M_TAU ** 2)

            true_vals = [true_p[:, 0], true_p[:, 1], true_p[:, 2], true_E]
            pred_vals = [pred_p[:, 0], pred_p[:, 1], pred_p[:, 2], pred_E]
            plot_true_vs_pred_2d(
                true_vals, pred_vals, [f'{tau}_px', f'{tau}_py', f'{tau}_pz', f'{tau}_E'],
                f'Tau 4-vector: {tau}', os.path.join(outdir, f'tau4vec_{tau}.pdf'),
                args.num_bins, sym_range='auto',
            )

        # === 2. polarimetric vector comparison ===
        print(">> Plotting polarimetric vector comparisons...")
        for tau in ['taup', 'taun']:
            true_h = true_cart_df[[f'ts_hh_{tau}_x', f'ts_hh_{tau}_y', f'ts_hh_{tau}_z']].values
            pred_h_unit = unit(pred_cart_df[[f'ts_hh_{tau}_x', f'ts_hh_{tau}_y', f'ts_hh_{tau}_z']].values)

            plot_true_vs_pred_2d(
                [true_h[:, 0], true_h[:, 1], true_h[:, 2]],
                [pred_h_unit[:, 0], pred_h_unit[:, 1], pred_h_unit[:, 2]],
                [f'ts_hh_{tau}_x', f'ts_hh_{tau}_y', f'ts_hh_{tau}_z'],
                f'Polarimetric vector: {tau}', os.path.join(outdir, f'polarimetric_{tau}.pdf'),
                args.num_bins, sym_range=(-1, 1),
            )

            ang_sep = np.degrees(np.arccos(np.clip(np.sum(true_h * pred_h_unit, axis=1), -1, 1)))
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(ang_sep, bins=args.num_bins, histtype='step', linewidth=1.5)
            ax.set_xlabel(f'angle(true, pred) [deg] -- {tau}')
            ax.set_ylabel('events')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f'polarimetric_angle_{tau}.pdf'), dpi=130)
            plt.close(fig)

        # === 3. phiCP ===
        print(">> Computing phiCP (true vs predicted)...")
        def get_phiCP(cart_df, E_taup, E_taun):
            h1 = cart_df[['ts_hh_taun_x', 'ts_hh_taun_y', 'ts_hh_taun_z']].values
            h2 = cart_df[['ts_hh_taup_x', 'ts_hh_taup_y', 'ts_hh_taup_z']].values
            p_taup = cart_df[['undecayed_taup_px', 'undecayed_taup_py', 'undecayed_taup_pz']].values
            p_taun = cart_df[['undecayed_taun_px', 'undecayed_taun_py', 'undecayed_taun_pz']].values
            n2, n1 = tau_directions_com(p_taup, p_taun, E_taup, E_taun)
            return compute_phiCP(h1, n1, h2, n2)

        true_E_taup = df['undecayed_taup_e'].values
        true_E_taun = df['undecayed_taun_e'].values
        pred_p_taup = pred_cart_df[['undecayed_taup_px', 'undecayed_taup_py', 'undecayed_taup_pz']].values
        pred_p_taun = pred_cart_df[['undecayed_taun_px', 'undecayed_taun_py', 'undecayed_taun_pz']].values
        pred_E_taup = np.sqrt(np.sum(pred_p_taup ** 2, axis=1) + M_TAU ** 2)
        pred_E_taun = np.sqrt(np.sum(pred_p_taun ** 2, axis=1) + M_TAU ** 2)

        true_phiCP = get_phiCP(true_cart_df, true_E_taup, true_E_taun)
        pred_phiCP = get_phiCP(pred_cart_df, pred_E_taup, pred_E_taun)

        # Reweight the (single, UnCorr) sample to CP-even/CP-odd hypotheses using
        # TauSpinner's own per-event weights, and compare true vs. predicted phiCP
        # under each hypothesis -- this tests whether the model's *predicted*
        # polarimetric vectors preserve enough information to still separate
        # CP-even from CP-odd once reweighted, not just whether they match truth
        # event-by-event.
        have_weights = 'tauspinner_wt_alpha0' in df.columns and 'tauspinner_wt_alpha90' in df.columns
        if have_weights:
            w_cpeven = df['tauspinner_wt_alpha0'].values
            w_cpodd = df['tauspinner_wt_alpha90'].values
            plot_phiCP_cp_comparison(true_phiCP, pred_phiCP, w_cpeven, w_cpodd,
                                      'All events', os.path.join(outdir, 'phiCP.pdf'), PHICP_BINS)
        else:
            print(">> WARNING: tauspinner_wt_alpha0/90 not found in test dataframe -- "
                  "falling back to unweighted true-vs-predicted phiCP.")
            fig, ax = plt.subplots(figsize=(6, 5))
            bins = np.linspace(0, 2 * np.pi, PHICP_BINS + 1)
            ax.hist(true_phiCP, bins=bins, density=True, histtype='step', linewidth=1.5, label='true')
            ax.hist(pred_phiCP, bins=bins, density=True, histtype='step', linewidth=1.5, label='predicted')
            ax.set_xlabel(r'$\phi_{CP}$ [rad]')
            ax.set_ylabel('a.u.')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, 'phiCP.pdf'), dpi=130)
            plt.close(fig)

        # === 3b. phiCP per decay-mode combination (same style as plot_phiCP.py) ===
        if have_weights:
            print(">> Plotting phiCP per decay-mode combination...")
            dm_taup, dm_taun = add_DM(df, dm_prefix='reco')
            dm_combs = [
                [0, 0], [0, 1], [1, 1], [2, 2], [1, 2], [0, 2], [10, 10], [0, 10], [1, 10], [2, 10],
                [0, 11], [1, 11], [2, 11], [10, 11], [11, 11],
                [100, 0], [100, 1], [100, 2], [100, 10], [100, 11], [100, 100],
            ]
            dm_outdir = os.path.join(outdir, 'phiCP_by_dm')
            os.makedirs(dm_outdir, exist_ok=True)
            min_events = 20
            for dm_p, dm_n in dm_combs:
                mask = ((dm_taup == dm_p) & (dm_taun == dm_n)) | ((dm_taup == dm_n) & (dm_taun == dm_p))
                n_events = mask.sum()
                if n_events < min_events:
                    continue
                plot_phiCP_cp_comparison(
                    true_phiCP[mask], pred_phiCP[mask], w_cpeven[mask], w_cpodd[mask],
                    f'DM{dm_p} - DM{dm_n} ({n_events} events)',
                    os.path.join(dm_outdir, f'phiCP_DM{dm_p}_DM{dm_n}.pdf'), PHICP_BINS,
                )
            print(f">> Saved per-DM phiCP plots to {dm_outdir}")

        # === 3c. ditau (boson candidate) invariant mass ===
        print(">> Plotting ditau invariant mass...")
        true_p_taup = true_cart_df[['undecayed_taup_px', 'undecayed_taup_py', 'undecayed_taup_pz']].values
        true_p_taun = true_cart_df[['undecayed_taun_px', 'undecayed_taun_py', 'undecayed_taun_pz']].values

        def ditau_mass(E_taup, p_taup, E_taun, p_taun):
            Etot = E_taup + E_taun
            ptot = p_taup + p_taun
            m2 = Etot ** 2 - np.sum(ptot ** 2, axis=1)
            return np.sqrt(np.clip(m2, 0, None))

        true_mass = ditau_mass(true_E_taup, true_p_taup, true_E_taun, true_p_taun)
        pred_mass = ditau_mass(pred_E_taup, pred_p_taup, pred_E_taun, pred_p_taun)

        plot_true_vs_pred_2d(
            [true_mass], [pred_mass], ['ditau_mass'],
            'Ditau invariant mass', os.path.join(outdir, 'ditau_mass_2d.pdf'),
            args.num_bins, sym_range=None,
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        lo, hi = np.percentile(true_mass, [0.5, 99.5])
        bins = np.linspace(lo, hi, args.num_bins + 1)
        ax.hist(true_mass, bins=bins, density=True, histtype='step', linewidth=1.5, label='true')
        ax.hist(pred_mass, bins=bins, density=True, histtype='step', linewidth=1.5, label='predicted')
        ax.set_xlabel('ditau mass [GeV]')
        ax.set_ylabel('a.u.')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'ditau_mass_1d.pdf'), dpi=130)
        plt.close(fig)

        # === 4. entanglement / spin-correlation cos variables ===
        # NOTE: these n/r/k projections use the *visible-tau-referenced* basis the
        # model was trained against (see ConvertToOrthonormalNRK), not the
        # true-tau-direction basis of kinematic_helpers.compute_spin_angles -- this
        # compares the model's predictions against its own training-target
        # definition, not the fully "canonical" spin-density-matrix basis.
        if coordinates == 'onorm':
            print(">> Computing entanglement/spin-correlation variables...")
            ent_df = pd.DataFrame({
                'true_cosn_plus': df['ts_hh_taup_n'], 'true_cosr_plus': df['ts_hh_taup_r'], 'true_cosk_plus': df['ts_hh_taup_k'],
                'true_cosn_minus': df['ts_hh_taun_n'], 'true_cosr_minus': df['ts_hh_taun_r'], 'true_cosk_minus': df['ts_hh_taun_k'],
                'pred_cosn_plus': pred_native_df['ts_hh_taup_n'], 'pred_cosr_plus': pred_native_df['ts_hh_taup_r'], 'pred_cosk_plus': pred_native_df['ts_hh_taup_k'],
                'pred_cosn_minus': pred_native_df['ts_hh_taun_n'], 'pred_cosr_minus': pred_native_df['ts_hh_taun_r'], 'pred_cosk_minus': pred_native_df['ts_hh_taun_k'],
            })
            true_vars = compute_spin_density_vars(ent_df, prefix='true_')
            pred_vars = compute_spin_density_vars(ent_df, prefix='pred_')
            print('True  (B+, B-, C, concurrence, m12):', true_vars)
            print('Pred  (B+, B-, C, concurrence, m12):', pred_vars)
            plot_spin_density_matrix({'true': true_vars, 'pred': pred_vars}, dm_category='polvec_quicktest', outdir=outdir)
        else:
            print(">> Skipping entanglement-matrix plot (only implemented for coordinates='onorm').")

        # === 5. save results (incl. TauSpinner weights) to parquet ===
        print(">> Saving results dataframe (with TauSpinner weights passed through)...")
        pred_h_taup_unit = unit(pred_cart_df[['ts_hh_taup_x', 'ts_hh_taup_y', 'ts_hh_taup_z']].values)
        pred_h_taun_unit = unit(pred_cart_df[['ts_hh_taun_x', 'ts_hh_taun_y', 'ts_hh_taun_z']].values)

        results_df = pd.DataFrame({
            'true_tau_plus_E':  true_E_taup,
            'true_tau_plus_px': true_cart_df['undecayed_taup_px'].values,
            'true_tau_plus_py': true_cart_df['undecayed_taup_py'].values,
            'true_tau_plus_pz': true_cart_df['undecayed_taup_pz'].values,
            'true_tau_minus_E':  true_E_taun,
            'true_tau_minus_px': true_cart_df['undecayed_taun_px'].values,
            'true_tau_minus_py': true_cart_df['undecayed_taun_py'].values,
            'true_tau_minus_pz': true_cart_df['undecayed_taun_pz'].values,
            'pred_tau_plus_E':  pred_E_taup,
            'pred_tau_plus_px': pred_cart_df['undecayed_taup_px'].values,
            'pred_tau_plus_py': pred_cart_df['undecayed_taup_py'].values,
            'pred_tau_plus_pz': pred_cart_df['undecayed_taup_pz'].values,
            'pred_tau_minus_E':  pred_E_taun,
            'pred_tau_minus_px': pred_cart_df['undecayed_taun_px'].values,
            'pred_tau_minus_py': pred_cart_df['undecayed_taun_py'].values,
            'pred_tau_minus_pz': pred_cart_df['undecayed_taun_pz'].values,
            'true_ts_hh_taup_x': true_cart_df['ts_hh_taup_x'].values,
            'true_ts_hh_taup_y': true_cart_df['ts_hh_taup_y'].values,
            'true_ts_hh_taup_z': true_cart_df['ts_hh_taup_z'].values,
            'true_ts_hh_taun_x': true_cart_df['ts_hh_taun_x'].values,
            'true_ts_hh_taun_y': true_cart_df['ts_hh_taun_y'].values,
            'true_ts_hh_taun_z': true_cart_df['ts_hh_taun_z'].values,
            'pred_ts_hh_taup_x': pred_h_taup_unit[:, 0],
            'pred_ts_hh_taup_y': pred_h_taup_unit[:, 1],
            'pred_ts_hh_taup_z': pred_h_taup_unit[:, 2],
            'pred_ts_hh_taun_x': pred_h_taun_unit[:, 0],
            'pred_ts_hh_taun_y': pred_h_taun_unit[:, 1],
            'pred_ts_hh_taun_z': pred_h_taun_unit[:, 2],
            'true_phiCP': true_phiCP,
            'pred_phiCP': pred_phiCP,
        })

        # pass through TauSpinner weights (and any spin-correlation weight pieces) if present
        _AXES = ('n', 'r', 'k')
        _passthrough_cols = (
            [f'tauspinner_wt_alpha{a}' for a in [0, 45, 90]] +
            [f'wt_hp_{a}' for a in _AXES] +
            [f'wt_hm_{a}' for a in _AXES] +
            [f'wt_hp_{a}_hm_{b}' for a in _AXES for b in _AXES]
        )
        cols_to_pass = [c for c in _passthrough_cols if c in df.columns]
        missing_cols = [c for c in _passthrough_cols if c not in df.columns]
        if missing_cols:
            print(f">> WARNING: the following TauSpinner weight columns were not found and won't be saved: {missing_cols}")
        if cols_to_pass:
            results_df = pd.concat([results_df, df[cols_to_pass].reset_index(drop=True)], axis=1)

        results_path = os.path.join(output_dir, f'polvec_eval_results_{test_output_name}.parquet')
        results_df.to_parquet(results_path)
        print(f">> Saved results dataframe ({len(results_df)} events, {len(results_df.columns)} columns) to {results_path}")

        print(f">> Done with {test_output_name}. Plots saved to {outdir}")


if __name__ == "__main__":
    main()
