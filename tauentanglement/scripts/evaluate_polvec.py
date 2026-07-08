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

from tauentanglement.python.DataProcessing import RegressionDataset
from tauentanglement.python.NN_Tools import load_model, get_device
from tauentanglement.python.Evaluation_Tools import flow_map_predict, plot_spin_density_matrix
from tauentanglement.utils.coordinate_conversions import ConvertFromOrthonormalNRK_Predictions_PolVec
from tauentanglement.utils.kinematic_helpers import compute_spin_density_vars, boost, boost_vector
from tauentanglement.utils.acoplanarity_tools import compute_aco_polarimetric

M_TAU = 1.77686

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


def load_test_dataframe(data_config, nn_config, leptonic_mode, max_events):
    """Reconstruct the exact filename convention used by get_train_val_test_datasets
    (DataProcessing.py), and load the *unpruned* saved test split -- it retains all
    original columns (true Cartesian + reco visible-tau vectors), unlike the in-memory
    copy used for training which is restricted to input+output features only."""
    dataset_name = data_config['datasets'][0]
    extra_name = ''
    if data_config['coordinates'] == 'standard':
        extra_name += 'cartesian'
    if leptonic_mode >= 0:
        extra_name += f"_leptonic_mode_{leptonic_mode}"
    if data_config.get('inc_three_prongs', False):
        extra_name += "_inc_three_prongs"
    if nn_config.get('use_transformer', False):
        extra_name += "_transformer"
    path = os.path.join(data_config['output_dir'], dataset_name, f'test_dataframe_{extra_name}.parquet')
    print(f">> Loading test dataframe from {path}")
    df = pd.read_parquet(path)
    if max_events is not None:
        df = df.iloc[:max_events].reset_index(drop=True)
    print(f">> Evaluating on {len(df)} events")
    return df


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


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', required=True, help='path to the configuration file')
    argparser.add_argument('--useCPU', action='store_true', help='force CPU evaluation')
    argparser.add_argument('--max_events', type=int, default=5000, help='number of test events to evaluate on')
    argparser.add_argument('--num_bins', type=int, default=50)
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']
    nn_config = config['SetupNN']

    device = torch.device('cpu') if args.useCPU else get_device()
    print(f">> Using device: {device}")

    output_dir = f"outputs_{nn_config['model_name']}"
    output_plots_dir = f"{output_dir}/plots"    
    outdir = os.path.join(output_dir, 'plots_eval_polvec')
    os.makedirs(outdir, exist_ok=True)

    input_features = data_config['Features']['input_features']
    coordinates = data_config['coordinates']
    output_features = data_config['Features']['output_features'][coordinates]
    leptonic_mode = data_config.get('leptonic_mode', -1)

    df = load_test_dataframe(data_config, nn_config, leptonic_mode, args.max_events)

    # --- build dataset using the normalization saved at training time ---
    norm_data = np.load(f'{output_dir}/normalization_params.npz')
    dataset = RegressionDataset(
        df, input_features, output_features,
        normalize_inputs=True, normalize_outputs=True,
        input_mean=torch.from_numpy(norm_data['input_mean']),
        input_std=torch.from_numpy(norm_data['input_std']),
        output_mean=torch.from_numpy(norm_data['output_mean']),
        output_std=torch.from_numpy(norm_data['output_std']),
    )

    # --- load model ---
    model = load_model(nn_config['hyperparams'], input_features, output_features,
                        useTransformer=nn_config.get('use_transformer', True), leptonic_mode=leptonic_mode)

    model_path = f'{output_plots_dir}/best_model.pth'
    print(f"Using model {nn_config['model_name']}")
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):  # check if model exists, if not take partial model
        model_path = f'{output_plots_dir}/partial_model.pth'
    try:  # load model and optimizer
        model.load_state_dict(torch.load(model_path))
    except:
        print(f"Loading model from {model_path} failed. Trying to load from CPU.")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(">> Successfully loaded model")
    model.eval()

    # --- MAP prediction, in the model's native (training) coordinate space ---
    X, _ = dataset[:]
    X = X.to(device)
    print(f">> Running MAP prediction (method={nn_config.get('map_method', 'gradient')})...")
    _, predictions_native = flow_map_predict(
        model, X, test_dataset=dataset,
        method=nn_config.get('map_method', 'gradient'),
        num_draws=nn_config.get('map_num_draws', 100),
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

    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.linspace(0, 2 * np.pi, args.num_bins + 1)
    ax.hist(true_phiCP, bins=bins, density=True, histtype='step', linewidth=1.5, label='true')
    ax.hist(pred_phiCP, bins=bins, density=True, histtype='step', linewidth=1.5, label='predicted')
    ax.set_xlabel(r'$\phi_{CP}$ [rad]')
    ax.set_ylabel('a.u.')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'phiCP.pdf'), dpi=130)
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

    print(f">> Done. Plots saved to {outdir}")


if __name__ == "__main__":
    main()
