import torch
from tqdm import tqdm
import numpy as np
from tauentanglement.utils.kinematic_helpers import polarimetric_vector_tau, compute_spin_angles, boost_vector, boost, compute_spin_density_vars
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import mplhep as hep


def flow_map_predict(
    model,
    X,
    test_dataset=None,
    num_draws=100,
    chunk_size=5000,
    method='stochastic',
    n_steps=200,
    lr=1e-2,
):
    """
    Compute MAP (maximum log-probability) predictions from a normalizing flow.

    Parameters
    ----------
    model : ConditionalFlow
        The trained normalizing flow model.
    X : torch.Tensor
        Conditioning features of shape [B, context_dim].
    test_dataset : object, optional
        Must supply .destandardize_outputs(tensor). If None, no destandardization is performed.
    num_draws : int
        Number of samples per event. Used by method='stochastic' and 'gradient_warmstart'.
    chunk_size : int
        Number of events to process at once (controls memory usage).
    method : str
        'stochastic'          — draw num_draws samples per event and pick the highest log-prob
                                one. Uses sample_and_log_prob() for a single forward pass.
        'latent_zero'         — decode the latent origin z=0 for each event. Deterministic,
                                single forward pass. Best for unimodal posteriors.
        'gradient'            — z-space MAP: optimise z via Adam using the change-of-variables
                                identity log p = log p_Z(z) - log|det J_decode|. Initialises
                                from z=0. Single decode pass per step.
        'gradient_warmstart'  — same as 'gradient' but initialises z from the best of
                                num_draws stochastic samples (encoded back to z-space).
                                Handles multimodal posteriors: sampling finds the right basin,
                                gradient descent finds the exact mode within it.
    n_steps : int
        Number of Adam steps. Only used by gradient methods.
    lr : float
        Adam learning rate. Only used by gradient methods.

    Returns
    -------
    samples_norm_alt : torch.Tensor, shape [B, features]
        MAP-selected samples in normalized (flow) space.
    samples_alt : np.ndarray or None
        Destandardized samples, or None if test_dataset not provided.
    """

    model.eval()

    B = X.shape[0]
    all_best_samples = []

    if method == 'latent_zero':
        # this doesn't work very well, but might as well leave it in
        latent_dim = model.flow._distribution._shape[0]
        for start in tqdm(range(0, B, chunk_size), desc="Processing chunks (latent_zero)"):
            end = min(start + chunk_size, B)
            X_chunk = X[start:end]
            C = X_chunk.shape[0]
            z_zero = torch.zeros(C, latent_dim, device=X_chunk.device)
            with torch.no_grad():
                x_map, _ = model.decode(z_zero, context=X_chunk)
            all_best_samples.append(x_map.cpu())

    elif method == 'stochastic':
        # sample from the pdf
        for start in tqdm(range(0, B, chunk_size), desc="Processing chunks (stochastic)"):
            end = min(start + chunk_size, B)
            X_chunk = X[start:end]
            C = X_chunk.shape[0]
            with torch.no_grad():
                # single pass get both [C, num_draws, F] and [C, num_draws]
                samples_norm_chunk, log_probs = model.sample_and_log_prob(
                    num_samples=num_draws, context=X_chunk
                )
            best_idx = torch.argmax(log_probs, dim=1)              # [C]
            best_samples_chunk = samples_norm_chunk[torch.arange(C), best_idx]  # [C, F]
            all_best_samples.append(best_samples_chunk.cpu())

    elif method == 'gradient':
        # optimisation of the MAP estimate
        latent_dim = model.flow._distribution._shape[0]
        for start in tqdm(range(0, B, chunk_size), desc="Processing chunks (gradient)"):
            end = min(start + chunk_size, B)
            X_chunk = X[start:end]
            C = X_chunk.shape[0]

            # optimise z directly using the change-of-variables log p(x|c) = log p_Z(z) - log|det J_decode(z)|
            # (z-space geometry is isotropic so easier to optimise)
            z = torch.zeros(C, latent_dim, device=X_chunk.device, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=lr)

            for _ in range(n_steps):
                optimizer.zero_grad()
                _, logabsdet = model.decode(z, context=X_chunk)
                log_pz = -0.5 * (z ** 2).sum(dim=-1)
                log_p = log_pz - logabsdet
                (-log_p.sum()).backward()
                optimizer.step()

            with torch.no_grad():
                x_map, _ = model.decode(z, context=X_chunk)
            all_best_samples.append(x_map.cpu())

    elif method == 'gradient_warmstart':
        # mix of stochastic
        latent_dim = model.flow._distribution._shape[0]
        for start in tqdm(range(0, B, chunk_size), desc="Processing chunks (gradient_warmstart)"):
            end = min(start + chunk_size, B)
            X_chunk = X[start:end]
            C = X_chunk.shape[0]

            # sample stochastic to get a good start pointwith torch.no_grad():
            with torch.no_grad():
                # single pass get both [C, num_draws, F] and [C, num_draws]
                samples_norm_chunk, log_probs = model.sample_and_log_prob(
                    num_samples=num_draws, context=X_chunk
                )
            best_idx = torch.argmax(log_probs, dim=1)              # [C]
            x_best = samples_norm_chunk[torch.arange(C), best_idx]  # [C, F]

            # encode this best sampled point to z space
            with torch.no_grad():
                z_init, _ = model.encode(x_best, context=X_chunk)

            # optimise from there
            z = z_init.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([z], lr=lr)

            for _ in range(n_steps):
                optimizer.zero_grad()
                _, logabsdet = model.decode(z, context=X_chunk)
                log_pz = -0.5 * (z ** 2).sum(dim=-1)
                log_p = log_pz - logabsdet
                (-log_p.sum()).backward()
                optimizer.step()

            with torch.no_grad():
                x_map, _ = model.decode(z, context=X_chunk)
            all_best_samples.append(x_map.cpu())

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'stochastic', 'latent_zero', 'gradient', or 'gradient_warmstart'.")

    samples_norm_alt = torch.cat(all_best_samples, dim=0)

    if test_dataset is not None:
        samples_alt = test_dataset.destandardize_outputs(samples_norm_alt).cpu().numpy()
    else:
        samples_alt = None

    return samples_norm_alt, samples_alt



def compute_spin_vars(df, tau_pred_prefix='true_', tau_vis_prefix=''):

    taup = df[[f'{tau_pred_prefix}tau_plus_E', f'{tau_pred_prefix}tau_plus_px', f'{tau_pred_prefix}tau_plus_py', f'{tau_pred_prefix}tau_plus_pz']].values
    taun = df[[f'{tau_pred_prefix}tau_minus_E', f'{tau_pred_prefix}tau_minus_px', f'{tau_pred_prefix}tau_minus_py', f'{tau_pred_prefix}tau_minus_pz']].values
    taup_pi1 = df[[f'{tau_vis_prefix}taup_pi1_E', f'{tau_vis_prefix}taup_pi1_px', f'{tau_vis_prefix}taup_pi1_py', f'{tau_vis_prefix}taup_pi1_pz']].values
    taup_pizero1 = df[[f'{tau_vis_prefix}taup_pizero1_E', f'{tau_vis_prefix}taup_pizero1_px', f'{tau_vis_prefix}taup_pizero1_py', f'{tau_vis_prefix}taup_pizero1_pz']].values
    taun_pi1 = df[[f'{tau_vis_prefix}taun_pi1_E', f'{tau_vis_prefix}taun_pi1_px', f'{tau_vis_prefix}taun_pi1_py', f'{tau_vis_prefix}taun_pi1_pz']].values
    taun_pizero1 = df[[f'{tau_vis_prefix}taun_pizero1_E', f'{tau_vis_prefix}taun_pizero1_px', f'{tau_vis_prefix}taun_pizero1_py', f'{tau_vis_prefix}taun_pizero1_pz']].values

    com_boost_vec = boost_vector(taup + taun)
    taup = boost(taup, -com_boost_vec)
    taun = boost(taun, -com_boost_vec)
    # boost decay products as well
    taup_pi1 = boost(taup_pi1, -com_boost_vec)
    taup_pizero1 = boost(taup_pizero1, -com_boost_vec)
    taun_pi1 = boost(taun_pi1, -com_boost_vec)
    taun_pizero1 = boost(taun_pizero1, -com_boost_vec)

    taup_s = polarimetric_vector_tau(
        taup, taup_pi1, taup_pizero1,
        np.ones_like(df[f'{tau_vis_prefix}taup_haspizero'].values), df[f'{tau_vis_prefix}taup_haspizero'].values
    )
    taun_s = polarimetric_vector_tau(
        taun, taun_pi1, taun_pizero1,
        np.ones_like(df[f'{tau_vis_prefix}taun_haspizero'].values), df[f'{tau_vis_prefix}taun_haspizero'].values
    )

    spin_angles = compute_spin_angles(
        taup, taun,
        taup_s, taun_s,
        p_axis=None
    )

    # now add these to the dataframe
    for key, values in spin_angles.items():
        df[f'{tau_pred_prefix}{key}'] = values

    return df

def save_sampled_pdfs(
    model,
    dataset,
    device,
    df,
    output_features,
    event_number,
    num_samples=50000,
    bins=100,
    outdir="pdf_slices_sampled",
    use_polar=False,
    use_onorm=False,
):
    """
    Estimate 1D marginals p(x_i | context) by directly sampling the conditional flow
    and plotting *normalized histograms* (no KDE).
    """

    os.makedirs(outdir, exist_ok=True)

    model.eval()

    X = dataset.X
    y = dataset.y
    X, y = X.to(device), y.to(device)

    # select just one event from row = event_number

    X = X[event_number].unsqueeze(0)

    with torch.no_grad():
        predictions_norm = model.sample(num_samples=num_samples, context=X).squeeze()

    predictions = dataset.destandardize_outputs(predictions_norm).cpu().numpy()

    if use_polar:
        # convert predictions to cartesian coordinates
        predictions = ConvertPredictionsToCartesian(predictions, output_features)
    if use_onorm:
        predictions = ConvertFromOrthonormalNRK_Predictions(
            predictions,
            taup_pi=df[['reco_taup_pi1_E', 'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz']].values,
            taup_pi0=df[['reco_taup_pizero1_E', 'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz']].values,
            taun_pi=df[['reco_taun_pi1_E', 'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz']].values,
            taun_pi0=df[['reco_taun_pizero1_E', 'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz']].values,
        )


    n_bins = bins   # preserve the integer

    for i, v in enumerate(output_features):

        # find binning to show 99% of the PDF distribution
        v_values = predictions[:, i]
        lower_bound = np.percentile(v_values, 0.5)
        upper_bound = np.percentile(v_values, 99.5)
        bins = np.linspace(lower_bound, upper_bound, n_bins)

        pred_i = predictions[:, i]
        plt.figure(figsize=(6, 4))
        plt.hist(pred_i, bins=bins, density=True, histtype='step', linewidth=2)
        # draw true value as an arrow
        true_value = dataset.destandardize_outputs(y[event_number].unsqueeze(0)).cpu().numpy()[0, i]
        if use_polar:
            # convert true value to cartesian coordinates
            true_value = ConvertPredictionsToCartesian(
                np.array([true_value]),
                output_features
            )[0, i]
        if use_onorm:
            true_value = ConvertFromOrthonormalNRK_Predictions(
                np.array([true_value]),
                taup_pi=df[['reco_taup_pi1_E', 'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz']].values[event_number:event_number+1],
                taup_pi0=df[['reco_taup_pizero1_E', 'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz']].values[event_number:event_number+1],
                taun_pi=df[['reco_taun_pi1_E', 'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz']].values[event_number:event_number+1],
                taun_pi0=df[['reco_taun_pizero1_E', 'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz']].values[event_number:event_number+1],
            )[0, i]
        plt.axvline(true_value, color='r', linestyle='--', linewidth=2, label='True Value')

        # also get analytical solutions if available
        analytical_col_0 = f'ana_pred_{v}'
        analytical_col_1 = f'ana_alt_pred_{v}'

        if analytical_col_0 in df.columns and analytical_col_1 in df.columns:
            analytical_sol_0 = df.iloc[event_number][analytical_col_0]
            analytical_sol_1 = df.iloc[event_number][analytical_col_1]
            plt.axvline(analytical_sol_0, color='g', linestyle='--', linewidth=2, label='Preferred Analytical Solution')
            plt.axvline(analytical_sol_1, color='b', linestyle=':', linewidth=2, label='Alternative Analytical Solution')
            plt.legend()

        plt.xlabel(v)
        plt.ylabel("pdf (sampled)")
        plt.title(f"Sampled p({v} | context of event {event_number})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"event{event_number}_{v}.pdf"))
        plt.close()

def plot_spin_density_matrix(results, dm_category, outdir):
    os.makedirs(outdir, exist_ok=True)

    labels = list(results.keys())
    n = len(labels)
    axes_labels = ['n', 'r', 'k']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def _draw_row_separators(ax, n_rows, n_cols):
        for row in range(1, n_rows):
            ax.hlines(row - 0.5, -0.5, n_cols - 0.5, colors='white', linewidths=2)

    # C matrix
    vmax = max(np.abs(results[l][2]).max() for l in labels)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        C = results[label][2]
        im = ax.imshow(C, cmap='RdBu', norm=norm)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(axes_labels)
        ax.set_yticklabels(axes_labels)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{C[i, j]:.3f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(C[i, j]) > 0.6 * vmax else 'black')
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Spin correlation matrix C  —  {dm_category}', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'C_matrix_{dm_category}.pdf'), bbox_inches='tight')
    plt.close()

    # B+ and B-
    bplus_mat  = np.array([results[l][0] for l in labels])   # [n, 3]
    bminus_mat = np.array([results[l][1] for l in labels])   # [n, 3]
    b_vmax = max(np.abs(bplus_mat).max(), np.abs(bminus_mat).max())
    b_norm = TwoSlopeNorm(vmin=-b_vmax, vcenter=0, vmax=b_vmax)

    fig, axes = plt.subplots(1, 2, figsize=(7, 0.6 * n + 1.5))
    for ax, mat, title in zip(axes, [bplus_mat, bminus_mat], ['$B^+$', '$B^-$']):
        im = ax.imshow(mat, cmap='RdBu', norm=b_norm, aspect='auto')
        ax.set_xticks(range(3))
        ax.set_xticklabels(axes_labels)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(3):
                ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(mat[i, j]) > 0.6 * b_vmax else 'black')
        _draw_row_separators(ax, n, 3)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Polarisation vectors  —  {dm_category}', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'B_vectors_{dm_category}.pdf'), bbox_inches='tight')
    plt.close()

    # Entanglement info
    ent_mat = np.array([[results[l][3].real, results[l][4].real] for l in labels])  # [n, 2]
    e_vmax = np.abs(ent_mat).max()
    e_norm = TwoSlopeNorm(vmin=-e_vmax, vcenter=0, vmax=e_vmax) if e_vmax > 0 else None

    fig, ax = plt.subplots(figsize=(4, 0.6 * n + 1.5))
    im = ax.imshow(ent_mat, cmap='RdBu', norm=e_norm, aspect='auto')
    ax.set_xticks(range(2))
    ax.set_xticklabels(['concurrence', '$m_{12}$'])
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(2):
            ax.text(j, i, f'{ent_mat[i, j]:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if e_vmax > 0 and abs(ent_mat[i, j]) > 0.6 * e_vmax else 'black')
    _draw_row_separators(ax, n, 2)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Entanglement variables  —  {dm_category}', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'entanglement_{dm_category}.pdf'), bbox_inches='tight')
    plt.close()
