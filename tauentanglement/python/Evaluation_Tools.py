import torch
from tqdm import tqdm
import numpy as np
from tauentanglement.utils.kinematic_helpers import polarimetric_vector_tau, compute_spin_angles, boost_vector, boost, compute_spin_density_vars
import os
import matplotlib.pyplot as plt

def flow_map_predict(
    model,
    X,
    test_dataset=None,
    num_draws=100,
    chunk_size=5000,
):
    """
    Compute MAP (maximum log-probability) predictions from a normalizing flow.
    
    Parameters
    ----------
    model : flow model
        The trained normalizing flow model.
    X : torch.Tensor
        Conditioning features of shape [B, context_dim].
    test_dataset : object, optional
        Must supply .destandardize_outputs(tensor). If None, no destandardization is performed.
    num_draws : int
        Number of samples per event to approximate the MAP estimate.
    chunk_size : int
        Number of events to process at once (controls memory usage).

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

    for start in tqdm(range(0, B, chunk_size), desc="Processing chunks"):
        end = min(start + chunk_size, B)
        X_chunk = X[start:end]
        C = X_chunk.shape[0]

        # -----------------------------------------------------------
        # 1. Sample num_draws from the flow for this chunk
        #    samples_norm_chunk: [C, num_draws, features]
        # -----------------------------------------------------------
        with torch.no_grad():
            samples_norm_chunk = model.sample(num_samples=num_draws, context=X_chunk)

        # Flatten for log_prob input
        # [C, D, F] → [C*D, F]
        flat_samples = samples_norm_chunk.reshape(C * num_draws, -1)

        # Repeat context for each sample
        # [C, ctx] → [C*D, ctx]
        flat_context = X_chunk.repeat_interleave(num_draws, dim=0)

        # -----------------------------------------------------------
        # 2. Compute log_prob for all C*D samples
        # -----------------------------------------------------------
        with torch.no_grad():
            flat_log_probs = model.log_prob(flat_samples, context=flat_context)

        # Reshape back to [C, D]
        log_probs = flat_log_probs.view(C, num_draws)

        # -----------------------------------------------------------
        # 3. Select the best (MAP) sample per event
        # -----------------------------------------------------------
        best_idx = torch.argmax(log_probs, dim=1)   # [C]
        batch_idx = torch.arange(C)

        best_samples_chunk = samples_norm_chunk[batch_idx, best_idx]  # [C, F]

        all_best_samples.append(best_samples_chunk.cpu())

    # -----------------------------------------------------------
    # Combine chunks → [B, F]
    # -----------------------------------------------------------
    samples_norm_alt = torch.cat(all_best_samples, dim=0)

    # -----------------------------------------------------------
    # Optional destandardization
    # -----------------------------------------------------------
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
