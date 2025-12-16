import argparse
import numpy as np
import pandas as pd
import os
from NN_tools_new import RegressionDataset, ConditionalFlow
from schedules import CosineAnnealingExpDecayLR
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from helpers import polarimetric_vector_tau, compute_spin_angles, boost_vector, boost

def setup_model_and_training(hp, verbose=True, reload=False, batch_norm=False):
    train_dataloader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False)

    model = ConditionalFlow(input_dim=len(output_features), raw_condition_dim=len(input_features),
                            context_dim=hp['condition_net_output_size'],
                            cond_hidden_dim=hp['condition_net_hidden_size'],
                            cond_num_blocks=hp['condition_net_num_blocks'],
                            num_layers=hp['num_layers'], num_bins=hp['num_bins'], tail_bound=hp['tail_bound'], 
                            hidden_size=hp['hidden_size'], num_blocks=hp['num_blocks'],
                            batch_norm=batch_norm)

    if verbose:
        print(model)
        # print number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of model parameters: {total_params}")
    optimizer = optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

    scheduler = None

    #gamma = -math.log(0.1)/(hp['epochs_to_10perc_lr'] * len(train_dataloader))
    #scheduler = CosineAnnealingExpDecayLR(optimizer, T_max=2 * len(train_dataloader), gamma=gamma)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp['num_epochs'] * len(train_dataloader),eta_min=0.0)    

    # check if reload is true, if so load model from output_dir if it exists
    if reload:
        model_name = args.model_name
        model_path = f'{output_dir}/{model_name}.pth'
        partial_model_path = f'{output_plots_dir}/partial_model.pth'
        if os.path.exists(model_path):
            print(f"Reloading model from {model_path}...")
        elif os.path.exists(partial_model_path):
            model_path = partial_model_path
            print(f"Reloading model from {model_path}...")
        else:
            print(f"Model path {model_path} does not exist. Can't reload. Exiting.")
            exit(1)
        # if loading model we will copy the old model before loading it
        copied_name = model_path.replace('.pth', '_copy.pth')
        os.system(f'cp {model_path} {copied_name}')
        # load model
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print(f"Loading model from {model_path} failed. Trying to load from CPU.")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model, optimizer, train_dataloader, test_dataloader, scheduler

def train_model(model, optimizer, train_dataloader, test_dataloader, num_epochs=10, device="cpu", verbose=True, output_plots_dir=None,
    save_every_N=None, recompute_train_loss=True, scheduler=None):
    model.to(device)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs+1):
        running_loss=0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = -model.log_prob(inputs=y, context=X).mean()
            loss.backward()
            optimizer.step()

            # changing learning rate per batch
            if scheduler:
                lr = scheduler.get_last_lr()
                scheduler.step()
        
            if verbose and epoch<5 and batch % 100 == 0:
                print(f'Batch {batch} | loss {loss.item()} | lr: {lr[0]} ' if scheduler else f'Batch {batch} | loss {loss.item()}')
            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)

        model.eval()
        if recompute_train_loss:
            # recompute train loss for better estimate
            sum_train_loss = 0
            with torch.no_grad():
                for X_train, y_train in train_dataloader:
                    X_train, y_train = X_train.to(device), y_train.to(device)
                    train_loss = -model.log_prob(inputs=y_train, context=X_train).mean()
                    sum_train_loss += train_loss.item()
            train_loss = sum_train_loss / len(train_dataloader)

        history["train_loss"].append(train_loss)

        # validation phase
        val_running_loss = 0
        with torch.no_grad():
            for X_val, y_val in test_dataloader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_loss = -model.log_prob(inputs=y_val, context=X_val).mean()
                val_running_loss += val_loss.item()
        val_loss = val_running_loss / len(test_dataloader)
        history["val_loss"].append(val_loss)

        # save model if its loss is better than the previous best
        if val_loss < best_val_loss and output_plots_dir:
            print(f"New best model found at epoch {epoch} with val_loss {val_loss}. Saving model...")
            torch.save(model.state_dict(), f'{output_plots_dir}/best_model.pth')
        best_val_loss = min(best_val_loss, val_loss)

        if verbose and epoch % 1 == 0:
            LR_string = f" | LR: {lr[0]:.2e}" if scheduler else ""
            print(f'Epoch {epoch} | loss {train_loss} | val_loss {val_loss} | lr {LR_string}')

        if epoch > 0 and output_plots_dir: plot_loss(history["train_loss"], history["val_loss"], output_dir=output_plots_dir)

        if save_every_N and epoch % save_every_N == 0 and output_plots_dir:
            print(f"Saving model checkpoint at epoch {epoch}...")
            torch.save(model.state_dict(), f'{output_plots_dir}/partial_model.pth')

    print("Training Completed. Trained for {} epochs.".format(epoch))

    return best_val_loss, history

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

    for start in range(0, B, chunk_size):
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


def plot_loss(loss_values, val_loss_values, output_dir='nn_plots'):
    plt.figure()
    plt.plot(range(1, len(loss_values)+1), loss_values, label='train loss')
    plt.plot(range(1, len(val_loss_values)+1), val_loss_values, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_vs_epoch_dummy.pdf')
    plt.close()

def save_sampled_pdfs(
    model,
    dataset,
    df,
    output_features,
    event_number,
    num_samples=50000,
    bins=100,
    outdir="pdf_slices_sampled"
):
    """
    Estimate 1D marginals p(x_i | context) by directly sampling the conditional flow
    and plotting *normalized histograms* (no KDE).
    """

    os.makedirs(outdir, exist_ok=True)

    model.eval()

    X = dataset.X
    y = dataset.y

    # select just one event from row = event_number

    X = X[event_number].unsqueeze(0)

    with torch.no_grad():
        predictions_norm = model.sample(num_samples=num_samples, context=X).squeeze()

    predictions = dataset.destandardize_outputs(predictions_norm).cpu().numpy()


    n_bins = bins   # preserve the integer

    for i, v in enumerate(output_features):

        # find binning to show 98% of the PDF distribution
        v_values = predictions[:, i]
        lower_bound = np.percentile(v_values, 0.5)
        upper_bound = np.percentile(v_values, 99.5)
        bins = np.linspace(lower_bound, upper_bound, n_bins)

        pred_i = predictions[:, i]
        plt.figure(figsize=(6, 4))
        plt.hist(pred_i, bins=bins, density=True, histtype='step', linewidth=2)
        # draw true value as an arrow
        true_value = dataset.destandardize_outputs(y[event_number].unsqueeze(0)).cpu().numpy()[0, i]
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

def augment_with_analytical(
    df: pd.DataFrame,
    output_features: list,
    analytical_output_features: list,
    shuffle: bool = True,
) -> pd.DataFrame:

    if len(output_features) != len(analytical_output_features):
        raise ValueError("output_features and analytical_output_features must have the same length")

    # True dataframe
    df_true = df.copy()
    df_true["is_analytical"] = 0.0

    # Analytical dataframe
    df_analytical = df.copy()
    # first drop the truth features
    df_analytical = df_analytical.drop(columns=output_features)
    df_analytical["is_analytical"] = 1.0

    #don't drop the analytical values, but make a copy of these with the output_feature nameings
    for out_feat, ana_feat in zip(output_features, analytical_output_features):
        df_analytical[out_feat] = df_analytical[ana_feat] 

    # Combine dataframes
    df_out = pd.concat([df_true, df_analytical], ignore_index=True)

    if shuffle:
        df_out = df_out.sample(frac=1).reset_index(drop=True)

    return df_out

def ConvertToPolar(df,prefix):
    # convert the px, py, pz columns with given prefix to polar coordinates (pt, eta, phi)
    px = df[f'{prefix}x']
    py = df[f'{prefix}y']
    pz = df[f'{prefix}z']

    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    # to avoid division by zero
    theta = np.arccos(np.clip(pz / p, -1.0, 1.0))
    eta = -np.log(np.tan(theta / 2))
    phi = np.arctan2(py, px)

    df[f'{prefix}_pt'] = pt
    df[f'{prefix}_eta'] = eta
    df[f'{prefix}_phi'] = phi

    # drop the original x, y, z columns
    df = df.drop(columns=[f'{prefix}x', f'{prefix}y', f'{prefix}z'])

    return df

def ConvertToCartesian(df,prefix):
    # convert the pt, eta, phi columns with given prefix to cartesian coordinates (px, py, pz)
    pt = df[f'{prefix}_pt']
    eta = df[f'{prefix}_eta']
    phi = df[f'{prefix}_phi']

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    df[f'{prefix}x'] = px
    df[f'{prefix}y'] = py
    df[f'{prefix}z'] = pz

    # drop the original pt, eta, phi columns
    df = df.drop(columns=[f'{prefix}_pt', f'{prefix}_eta', f'{prefix}_phi'])

    return df

# make a similar function for converting back the predictions which aren't labelled with a prefix
def ConvertPredictionsToCartesian(predictions, output_features):
    # predictions is a numpy array of shape [N_events, N_features]
    predictions_df = pd.DataFrame(predictions, columns=output_features)

    # find all prefixes by removing the _pt, _eta, _phi suffixes
    prefixes = set()
    for col in predictions_df.columns:
        if col.endswith('_pt'):
            prefixes.add(col[:-3])
        elif col.endswith('_eta'):
            prefixes.add(col[:-4])
        elif col.endswith('_phi'):
            prefixes.add(col[:-4])

    for prefix in prefixes:
        predictions_df = ConvertToCartesian(predictions_df, prefix)

    # return as numpy array
    return predictions_df.values

def compute_spin_vars(df, tau_prefix='true_'):

    taup = df[[f'{tau_prefix}tau_plus_E', f'{tau_prefix}tau_plus_px', f'{tau_prefix}tau_plus_py', f'{tau_prefix}tau_plus_pz']].values
    taun = df[[f'{tau_prefix}tau_minus_E', f'{tau_prefix}tau_minus_px', f'{tau_prefix}tau_minus_py', f'{tau_prefix}tau_minus_pz']].values
    taup_pi1 = df[['taup_pi1_E', 'taup_pi1_px', 'taup_pi1_py', 'taup_pi1_pz']].values
    taup_pizero1 = df[['taup_pizero1_E', 'taup_pizero1_px', 'taup_pizero1_py', 'taup_pizero1_pz']].values
    taun_pi1 = df[['taun_pi1_E', 'taun_pi1_px', 'taun_pi1_py', 'taun_pi1_pz']].values
    taun_pizero1 = df[['taun_pizero1_E', 'taun_pizero1_px', 'taun_pizero1_py', 'taun_pizero1_pz']].values

    com_boost_vec = boost_vector(taup + taun)
    taup = boost(taup, -com_boost_vec)
    taun = boost(taun, -com_boost_vec)

    taup_s = polarimetric_vector_tau(
        taup, taup_pi1, taup_pizero1,
        np.ones_like(df['taup_haspizero'].values), df['taup_haspizero'].values
    )
    taun_s = polarimetric_vector_tau(
        taun, taun_pi1, taun_pizero1,
        np.ones_like(df['taun_haspizero'].values), df['taun_haspizero'].values
    )

    spin_angles = compute_spin_angles(
        taup, taun,
        taup_s, taun_s,
        p_axis=None
    )

    # now add these to the dataframe
    for key, values in spin_angles.items():
        df[f'{tau_prefix}{key}'] = values

    return df

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()   

    argparser.add_argument('--stages', '-s', help='a list of stages to run', type=int, nargs="+")
    argparser.add_argument('--model_name', '-m', help='the name of the model output name', type=str, default='LEP_nflow_model')
    argparser.add_argument('--n_epochs', '-n', help='number of training epochs', type=int, default=10)
    argparser.add_argument('--n_trials', '-t', help='number of hyperparameter optimization trials', type=int, default=100)
    argparser.add_argument('--reload', '-r', help='reload from existing model', action='store_true')
    argparser.add_argument('--inc_reco_taus', help='whether to include the taus reconstructed by the analytical model as inputs', action='store_true')
    argparser.add_argument('--mix_true_and_analytical', help='If set then produce mixed dataset using both truth and analytical neutrino solutions and add flag as input variable that determines which one is used', action='store_true')
    argparser.add_argument('--use_polar', help='whether to use polar coordinates for inputs and outputs', action='store_true')
    args = argparser.parse_args()

    # make output directory called outputs_{model_name}, with plots subdirectory
    output_dir = f"outputs_{args.model_name}"
    output_plots_dir = f"{output_dir}/plots"
    os.makedirs(output_plots_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_features = ['dmin_x', 'dmin_y', 'dmin_z',
                       'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz', 'reco_taup_pi1_e',
                       'reco_taup_pi1_ipx', 'reco_taup_pi1_ipy', 'reco_taup_pi1_ipz',
                       'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz', 'reco_taup_pizero1_e',
                       'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz', 'reco_taun_pi1_e',
                       'reco_taun_pi1_ipx', 'reco_taun_pi1_ipy', 'reco_taun_pi1_ipz',
                       'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz', 'reco_taun_pizero1_e',
                       'BS_x', 'BS_y', 'BS_z',
                       'taup_haspizero', 'taun_haspizero']

    if args.inc_reco_taus:
        input_features += [
            'reco_taup_nu_px', 'reco_taup_nu_py', 'reco_taup_nu_pz',
            'reco_taun_nu_px', 'reco_taun_nu_py', 'reco_taun_nu_pz',
            'reco_alt_taup_nu_px', 'reco_alt_taup_nu_py', 'reco_alt_taup_nu_pz',
            'reco_alt_taun_nu_px', 'reco_alt_taun_nu_py', 'reco_alt_taun_nu_pz'
        ]

    if args.use_polar:
        output_features = [
            'taup_nu_p_pt', 'taup_nu_p_eta', 'taup_nu_p_phi',
            'taun_nu_p_pt', 'taun_nu_p_eta', 'taun_nu_p_phi'
        ]
    else:
        output_features = [
            'taup_nu_px', 'taup_nu_py', 'taup_nu_pz',
            'taun_nu_px', 'taun_nu_py', 'taun_nu_pz'
        ]

    # stage one prepares the dataframe
    if 1 in args.stages:

        print("Preparing dataframe...")

        import uproot3
        input_file_name = '/vols/cms/dw515/HH_reweighting/DiTauEntanglement/batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_10000000_Njob_10000/pythia_events_extravars.root'
        tree = uproot3.open(input_file_name)['new_tree']

        variables = [
            'taup_npi', 'taup_npizero',
            'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz', 'reco_taup_pi1_e',
            'reco_taup_pi1_ipx', 'reco_taup_pi1_ipy', 'reco_taup_pi1_ipz',
            'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz', 'reco_taup_pizero1_e',
            'taun_npi', 'taun_npizero',
            'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz', 'reco_taun_pi1_e',
            'reco_taun_pi1_ipx', 'reco_taun_pi1_ipy', 'reco_taun_pi1_ipz',
            'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz', 'reco_taun_pizero1_e',
            'reco_mass',
            'BS_x', 'BS_y', 'BS_z',
            'taup_nu_px', 'taup_nu_py', 'taup_nu_pz',
            'taun_nu_px', 'taun_nu_py', 'taun_nu_pz',
            'reco_taup_nu_px', 'reco_taup_nu_py', 'reco_taup_nu_pz',
            'reco_taun_nu_px', 'reco_taun_nu_py', 'reco_taun_nu_pz',
            'reco_alt_taup_nu_px', 'reco_alt_taup_nu_py', 'reco_alt_taup_nu_pz',
            'reco_alt_taun_nu_px', 'reco_alt_taun_nu_py', 'reco_alt_taun_nu_pz'
        ]

        df = tree.pandas.df(variables)
        # filter the non pi and pipizero decay modes
        df = df[(df['taup_npi'] == 1) & (df['taup_npizero'] < 2)]
        df = df[(df['taun_npi'] == 1) & (df['taun_npizero'] < 2)]

        # now we add a float which is 0 or 1 depending on whether tau is pi or pipizero, then delete the npi and npizero columns
        df['taup_haspizero'] = (df['taup_npizero'] > 0).astype(float)
        df['taun_haspizero'] = (df['taun_npizero'] > 0).astype(float)
        df = df.drop(columns=['taup_npi', 'taup_npizero', 'taun_npi', 'taun_npizero'])

        # also apply a reco_mass cut to select events close to the Z pole with little boost
        df = df[(df['reco_mass'] > 91)]
        # now remove the reco_mass column
        df = df.drop(columns=['reco_mass'])

        # compute the d_min vector by subrtacting the 2 impact parameters
        df['dmin_x'] = df['reco_taup_pi1_ipx'] - df['reco_taun_pi1_ipx']
        df['dmin_y'] = df['reco_taup_pi1_ipy'] - df['reco_taun_pi1_ipy']
        df['dmin_z'] = df['reco_taup_pi1_ipz'] - df['reco_taun_pi1_ipz']  

        if args.use_polar:
            # convert output to polar coordinates
            df = ConvertToPolar(df, 'taup_nu_p')
            df = ConvertToPolar(df, 'taun_nu_p')

            ## uncomment to convert inputs to polar coordinates as well
            #df = ConvertToPolar(df, 'reco_taup_pi1_p')
            #df = ConvertToPolar(df, 'reco_taup_pizero1_p')
            #df = ConvertToPolar(df, 'reco_taun_pi1_p')
            #df = ConvertToPolar(df, 'reco_taun_pizero1_p')
            #df = ConvertToPolar(df, 'reco_taup_nu_p')
            #df = ConvertToPolar(df, 'reco_taun_nu_p ')
            #df = ConvertToPolar(df, 'reco_alt_taup_nu_p')
            #df = ConvertToPolar(df, 'reco_alt_taun_nu_p')
            #df = ConvertToPolar(df, 'reco_taup_pi1_ip')
            #df = ConvertToPolar(df, 'reco_taun_pi1_ip')
            #df = ConvertToPolar(df, 'dmin_')

            df.to_pickle('ditau_nu_regression_ee_to_tauhtauh_polar_dataframe.pkl')

        else:
            df.to_pickle('ditau_nu_regression_ee_to_tauhtauh_dataframe.pkl')

        print("Dataframe prepared and saved.")

    else: # load the dataframe
        if args.use_polar:
            df = pd.read_pickle('ditau_nu_regression_ee_to_tauhtauh_polar_dataframe.pkl')
        else:
            df = pd.read_pickle('ditau_nu_regression_ee_to_tauhtauh_dataframe.pkl')

    # split dataset into train and test

    train_size = int(0.9 * len(df))
    test_size = len(df) - train_size
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # define datasets and normalize inputs and outputs

    if args.mix_true_and_analytical:
        # now produce dataset using analytical solutions as the output variables as well
        analytical_output_features = ['reco_' + x for x in output_features] #TODO: this wont work at the moment for polar coordinates

        # ensure these are dropped from input features if present
        input_features = [f for f in input_features if f not in analytical_output_features]

        # add the is_analytical flag to input features
        input_features.append("is_analytical")

        train_df = augment_with_analytical(
            train_df,
            output_features,
            analytical_output_features,
        )
        test_df_copy = test_df.copy()
        test_df = augment_with_analytical(
            test_df,
            output_features,
            analytical_output_features,
        )

    #print the names of all the columns and information on the number of events in the dataframe
    print('Columns in dataframe:', df.columns.tolist())
    print('Number of events in dataframe:', len(df))
    # print number of input features
    print('Number of input features:', len(input_features))

    train_dataset = RegressionDataset(train_df, input_features, output_features, normalize_inputs=True, normalize_outputs=True)
    in_mean, in_std = train_dataset.input_mean, train_dataset.input_std
    out_mean, out_std = train_dataset.output_mean, train_dataset.output_std
    test_dataset = RegressionDataset(test_df, input_features, output_features, normalize_inputs=True, normalize_outputs=True,
                                      input_mean=in_mean, input_std=in_std, output_mean=out_mean, output_std=out_std)

    # store the means and stds used for normalization
    np.savez(f'{output_dir}/normalization_params.npz',
             input_mean=in_mean, input_std=in_std,
             output_mean=out_mean, output_std=out_std)

    # leave stage 2 for not - this will involve optimising NN using optuna
    if 2 in args.stages:
        print("Starting hyperparameter optimization...")

        import optuna
        import optuna.visualization.matplotlib as optplt

        def live_plot_callback(study, trial):
            plt.figure(figsize=(6, 4))
            optplt.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(os.path.join(output_plots_dir, "optimization_history.pdf"))

            # now make a linear plot with high outliers removed

            losses = np.array([t.value for t in study.trials if t.value is not None])

            # Define cutoff for very large losses - e.g., 95th percentile
            high_cut = np.quantile(losses, 0.95)

            # Keep all trials with reasonable (non-outlier) losses
            trimmed_trials = [t for t in study.trials if t.value is not None and t.value <= high_cut]

            # Create a temporary filtered study for plotting
            filtered_study = optuna.create_study(direction=study.direction)
            filtered_study.add_trials(trimmed_trials)

            plt.figure(figsize=(6, 4))
            optplt.plot_optimization_history(filtered_study)
            plt.tight_layout()
            plt.savefig(os.path.join(output_plots_dir, "optimization_history_filtered.pdf"))
            plt.close()

        def objective(trial):

            print(f"\n🔹 Starting trial {trial.number}...")

            hp = {
                'batch_size': trial.suggest_categorical('batch_size', [4096, 8192, 16384]),
                'num_layers': trial.suggest_int('num_layers', 2, 10),
                'num_bins': trial.suggest_int('num_bins', 8, 20, step=2),
                'tail_bound': trial.suggest_float('tail_bound', 1.0, 5.0, step=1.0),
                'hidden_size': trial.suggest_int('hidden_size', 50, 300, step=50),
                'num_blocks': trial.suggest_int('num_blocks', 1, 4),
                'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
                #'epochs_to_10perc_lr': trial.suggest_int('epochs_to_10perc_lr', 20, 100, step=10),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
                'condition_net_hidden_size': trial.suggest_int('condition_net_hidden_size', 50, 300, step=50),
                'condition_net_num_blocks': trial.suggest_int('condition_net_num_blocks', 0, 5),
                'condition_net_output_size': trial.suggest_int('condition_net_output_size', 6, 30),
                'num_epochs': args.n_epochs,
            }

            model, optimizer, train_loader, test_loader, scheduler = setup_model_and_training(hp, verbose=False)

            # prine if model has too many parameters
            num_params = sum(p.numel() for p in model.parameters())
            trial.set_user_attr("n_params", num_params)
            max_N_params = 2*1e6
            if num_params > max_N_params:
                raise optuna.TrialPruned(f"Too many parameters: {num_params}")

            best_val_loss, history = train_model(model, optimizer, train_loader, test_loader, num_epochs=args.n_epochs, device=device, verbose=False, output_plots_dir=None,
                save_every_N=None, scheduler=scheduler, recompute_train_loss=False,)

            return best_val_loss
        db_path = os.path.join(output_dir, "nn_optuna_study.db")

        study = optuna.create_study(
            study_name="nn_hyperparam_optimization",
            direction="minimize",
            storage=f"sqlite:///{db_path}",  # store the study in the output_dir
            load_if_exists=True  # if file exists, load it instead of creating new
        )

        study.optimize(objective, n_trials=args.n_trials, callbacks=[live_plot_callback])

        best_trial = study.best_trial

        print("Best trial:")
        print(f"  Validation loss: {best_trial.value:.6f}")
        print("  Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        # also print the hyper parameters that were fixed
        fixed_hyperparams = {k: v for k, v in hp.items() if k not in best_trial.params}
        print("  Fixed hyperparameters:")
        for key, value in fixed_hyperparams.items():
            print(f"    {key}: {value}")

        exit()

#Trial 9 finished with value: -19.865106669339266 and parameters: {'batch_s
#ize': 8192, 'num_layers': 4, 'num_bins': 18, 'tail_bound': 5.0, 'hidden_size': 50, 'num_blocks': 1, 'lr': 0.0009381276074910463, 'weight_decay': 1.3195987584367093e-05, 'condition_net_hidden_size': 300, '
#condition_net_num_blocks': 1, 'condition_net_output_size': 17}. Best is trial 9 with value: -19.865106669339266.

    # load the model with the best hyperparameters found in stage 2
    # for now we just hard code the hyperparameters
    #hp = {
    #    'batch_size': 8192,
    #    'num_layers': 10,
    #    'num_bins': 16,
    #    'tail_bound': 3.0,
    #    'hidden_size': 200,
    #    'num_blocks': 2,
    #    'lr': 0.001,
    #    'weight_decay': 1e-4,
    #    #'epochs_to_10perc_lr': 100,
    #    'condition_net_hidden_size': 200,
    #    'condition_net_num_blocks': 0, #4
    #    'condition_net_output_size': 10,
    #    'num_epochs': args.n_epochs,
    #}
    # best hps so far from trial 9
    hp = {
        'batch_size': 8192,
        'num_layers': 4,
        'num_bins': 18,
        'tail_bound': 5.0,
        'hidden_size': 50,
        'num_blocks': 1,
        'lr': 0.00094,
        'weight_decay': 1.32e-05,
        #'epochs_to_10perc_lr': 100,
        'condition_net_hidden_size': 300,
        'condition_net_num_blocks': 1, #4
        'condition_net_output_size': 17,
        'num_epochs': args.n_epochs,
    }
    model, optimizer, train_loader, test_loader, scheduler = setup_model_and_training(hp, reload=args.reload, batch_norm=False)

    if 3 in args.stages:

        print("Starting training...")

        best_val_loss, history = train_model(model, optimizer, train_loader, test_loader, num_epochs=args.n_epochs, device=device, verbose=True, output_plots_dir=output_plots_dir,save_every_N=1, scheduler=scheduler)

        model_name = args.model_name
        torch.save(model.state_dict(), f'{output_dir}/{model_name}.pth')

        print('Finished training.')

    if 4 in args.stages:

        import uproot3

        print('Evaluating final model on test dataset...')

        model_path = f'{output_dir}/best_model.pth'
        # check if model exists, if not take  partial model
        if not os.path.exists(model_path):
            model_path = f'{output_plots_dir}/partial_model.pth'

        try:
            # load model and optimizer
            model.load_state_dict(torch.load(model_path))

        except:
            print(f"Loading model from {model_path} failed. Trying to load from CPU.")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        model.eval()

        if args.mix_true_and_analytical:
            # in this case we have to build the test dataset again starting from test_df, since it is used differently during the training and evaluation
            # select the input, output, and context features from test_df
            #first split test_df into is_analytical = 0 and 1
            test_df = test_df_copy.copy()
            test_df['is_analytical'] = 1.0
            test_dataset = RegressionDataset(test_df, input_features, analytical_output_features, normalize_inputs=True, normalize_outputs=True,
                                              input_mean=in_mean, input_std=in_std, output_mean=out_mean, output_std=out_std)

            #now flip the is_analytical flag to 0 for the true dataset
            test_df['is_analytical'] = 0.0
            test_dataset_true = RegressionDataset(test_df, input_features, output_features, normalize_inputs=True, normalize_outputs=True,
                                              input_mean=in_mean, input_std=in_std, output_mean=out_mean, output_std=out_std)

            X_analytical, y_analytical = test_dataset[:]
            X_true, y_true = test_dataset_true[:]
            # now we map to latent space
            with torch.no_grad():
                X_analytical = X_analytical.to(device)
                y_analytical = y_analytical.to(device)
                X_true = X_true.to(device)
                y_true = y_true.to(device)

                z, _ = model.encode(inputs=y_analytical, context=X_analytical)
                # now we decode with the true context
                predictions_norm, _ = model.decode(z=z, context=X_true)

                predictions = test_dataset.destandardize_outputs(predictions_norm).cpu().numpy()

                predictions_alt = None

            test_df = test_df_copy.copy()

            X_test, _ = test_dataset_true[:]

        else:
            X_test, _ = test_dataset[:]

            # move X_test and model to CPU
            X_test = X_test.cpu()
            model = model.cpu()
            with torch.no_grad():
                predictions_norm = model.sample(num_samples=1, context=X_test).squeeze()     

            # destandardize predictions so that they are in physical units
            predictions = test_dataset.destandardize_outputs(predictions_norm).cpu().numpy()

        if args.use_polar:
            # convert predictions back to cartesian coordinates
            predictions = ConvertPredictionsToCartesian(predictions, output_features)

        # define alternative prediction by taking most probable value from flow 
        # to do this we sample 100 times and take the case with the best log probability
        # estimate most likely solution using flow_map_predict function
        samples_norm_alt, samples_alt = flow_map_predict(
            model,
            X_test,
            test_dataset=test_dataset,
            num_draws=100,
            chunk_size=5000 if device.type == 'cpu' else 50000,
        )

        if args.use_polar:
            # convert predictions_alt back to cartesian coordinates
            samples_alt = ConvertPredictionsToCartesian(samples_alt, output_features)       

        # unpack MAP outputs
        alt_nubar_px = samples_alt[:,0]
        alt_nubar_py = samples_alt[:,1]
        alt_nubar_pz = samples_alt[:,2]
        alt_nu_px    = samples_alt[:,3]
        alt_nu_py    = samples_alt[:,4]
        alt_nu_pz    = samples_alt[:,5]
        
        # energies
        alt_nu_E    = np.sqrt(alt_nu_px**2    + alt_nu_py**2    + alt_nu_pz**2)
        alt_nubar_E = np.sqrt(alt_nubar_px**2 + alt_nubar_py**2 + alt_nubar_pz**2)
        
        # final MAP / "alternative" array
        predictions_alt = np.column_stack(
            (alt_nu_E, alt_nu_px, alt_nu_py, alt_nu_pz,
             alt_nubar_E, alt_nubar_px, alt_nubar_py, alt_nubar_pz)
        )

    
        true_values = test_df[output_features].values

        if args.use_polar:
            # convert true values back to cartesian coordinates
            true_values = ConvertPredictionsToCartesian(true_values, output_features)

        ana_values = test_df[['reco_taup_nu_px', 'reco_taup_nu_py', 'reco_taup_nu_pz',
            'reco_taun_nu_px', 'reco_taun_nu_py', 'reco_taun_nu_pz']].values 

        # predictions dont include E so we need to compute them
        # compute E for nu and nubar
        nubar_px = predictions[:,0]
        nubar_py = predictions[:,1]
        nubar_pz = predictions[:,2]
        nu_px = predictions[:,3]
        nu_py = predictions[:,4]
        nu_pz = predictions[:,5]
        nu_E = np.sqrt(nu_px**2 + nu_py**2 + nu_pz**2)
        nubar_E = np.sqrt(nubar_px**2 + nubar_py**2 + nubar_pz**2)
        predictions = np.column_stack((nu_E, nu_px, nu_py, nu_pz, nubar_E, nubar_px, nubar_py, nubar_pz))

        # get E components for true values as well
        true_nubar_px = true_values[:,0]
        true_nubar_py = true_values[:,1]
        true_nubar_pz = true_values[:,2]
        true_nu_px = true_values[:,3]
        true_nu_py = true_values[:,4]
        true_nu_pz = true_values[:,5]
        true_nu_E = np.sqrt(true_nu_px**2 + true_nu_py**2 + true_nu_pz**2)
        true_nubar_E = np.sqrt(true_nubar_px**2 + true_nubar_py**2 + true_nubar_pz**2)
        true_values = np.column_stack((true_nu_E, true_nu_px, true_nu_py, true_nu_pz,
                                       true_nubar_E, true_nubar_px, true_nubar_py, true_nubar_pz))
    
        # get true taus by summing with pis and pizeros
        taun_pi = test_df[['reco_taun_pi1_e', 'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz']].values
        taup_pi  = test_df[['reco_taup_pi1_e', 'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz']].values
        taun_pizero  = test_df[['reco_taun_pizero1_e', 'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz']].values
        taup_pizero = test_df[['reco_taup_pizero1_e', 'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz']].values

        # tau- = nu + pi + pizero
        taun = true_values[:, 0:4] + taun_pi + taun_pizero
        
        # tau+ = nu + pi + pizero
        taup = true_values[:, 4:8] + taup_pi + taup_pizero
        
        # final true taus (8 columns total)
        true_taus = np.concatenate([taun, taup], axis=1)
        
        # same for predictions
        taun_pred = predictions[:, 0:4] + taun_pi + taun_pizero
        taup_pred = predictions[:, 4:8] + taup_pi + taup_pizero
        
        pred_taus = np.concatenate([taun_pred, taup_pred], axis=1)

        # get alternative predictions
        if predictions_alt is not None:
            alt_taun_pred = predictions_alt[:, 0:4] + taun_pi + taun_pizero
            alt_taup_pred = predictions_alt[:, 4:8] + taup_pi + taup_pizero
            pred_taus_alt = np.concatenate([alt_taun_pred, alt_taup_pred], axis=1)

        # get analytical precitions using reco_taup_nu and reco_taun_nu
        # first get the pis and pizeros again
        reco_taup_nu = test_df[['reco_taup_nu_px', 'reco_taup_nu_py', 'reco_taup_nu_pz']].values
        reco_taun_nu = test_df[['reco_taun_nu_px', 'reco_taun_nu_py', 'reco_taun_nu_pz']].values
        # compute E for these nus
        reco_taup_nu_E = np.sqrt(reco_taup_nu[:,0]**2 + reco_taup_nu[:,1]**2 + reco_taup_nu[:,2]**2)
        reco_taun_nu_E = np.sqrt(reco_taun_nu[:,0]**2 + reco_taun_nu[:,1]**2 + reco_taun_nu[:,2]**2)
        reco_taup_nu = np.column_stack((reco_taup_nu_E, reco_taup_nu))
        reco_taun_nu = np.column_stack((reco_taun_nu_E, reco_taun_nu))

        ana_pred_values = np.concatenate([reco_taun_nu, reco_taup_nu], axis=1)

        # get the predicted taus as well
        ana_pred_taup = reco_taup_nu + taup_pi + taup_pizero
        ana_pred_taun = reco_taun_nu + taun_pi + taun_pizero
        ana_pred_taus =  np.concatenate([ana_pred_taun, ana_pred_taup], axis=1)

        # get the alt ones as well
        reco_alt_taup_nu = test_df[['reco_alt_taup_nu_px', 'reco_alt_taup_nu_py', 'reco_alt_taup_nu_pz']].values
        reco_alt_taun_nu = test_df[['reco_alt_taun_nu_px', 'reco_alt_taun_nu_py', 'reco_alt_taun_nu_pz']].values
        # compute E for these nus
        reco_alt_taup_nu_E = np.sqrt(reco_alt_taup_nu[:,0]**2 + reco_alt_taup_nu[:,1]**2 + reco_alt_taup_nu[:,2]**2)
        reco_alt_taun_nu_E = np.sqrt(reco_alt_taun_nu[:,0]**2 + reco_alt_taun_nu[:,1]**2 + reco_alt_taun_nu[:,2]**2)
        reco_alt_taup_nu = np.column_stack((reco_alt_taup_nu_E, reco_alt_taup_nu))
        reco_alt_taun_nu = np.column_stack((reco_alt_taun_nu_E, reco_alt_taun_nu))

        ana_alt_pred_values = np.concatenate([reco_alt_taun_nu, reco_alt_taup_nu], axis=1)

        # get the predicted taus as well
        ana_alt_pred_taup = reco_alt_taup_nu + taup_pi + taup_pizero
        ana_alt_pred_taun = reco_alt_taun_nu + taun_pi + taun_pizero
        ana_alt_pred_taus =  np.concatenate([ana_alt_pred_taun, ana_alt_pred_taup], axis=1)

        # collect true and predicted nus true and predicted taus AND pi's into pandas dataframe, lable the collumns

        taup_haspizero = test_df['taup_haspizero'].values.reshape(-1,1)
        taun_haspizero = test_df['taun_haspizero'].values.reshape(-1,1)


        results_df = pd.DataFrame(data=np.concatenate([true_values, predictions, ana_pred_values, ana_alt_pred_values, true_taus, pred_taus, ana_pred_taus, ana_alt_pred_taus, taun_haspizero, taup_haspizero,
                                  taup_pi, taup_pizero, taun_pi, taun_pizero], axis=1),
                                  columns=['true_nu_E', 'true_nu_px', 'true_nu_py', 'true_nu_pz',
                                           'true_nubar_E', 'true_nubar_px', 'true_nubar_py', 'true_nubar_pz',
                                           'pred_nu_E', 'pred_nu_px', 'pred_nu_py', 'pred_nu_pz',
                                           'pred_nubar_E', 'pred_nubar_px', 'pred_nubar_py', 'pred_nubar_pz',
                                           'ana_pred_nu_E', 'ana_pred_nu_px', 'ana_pred_nu_py', 'ana_pred_nu_pz',
                                           'ana_pred_nubar_E', 'ana_pred_nubar_px', 'ana_pred_nubar_py', 'ana_pred_nubar_pz',
                                           'ana_alt_pred_nu_E', 'ana_alt_pred_nu_px', 'ana_alt_pred_nu_py', 'ana_alt_pred_nu_pz',
                                           'ana_alt_pred_nubar_E', 'ana_alt_pred_nubar_px', 'ana_alt_pred_nubar_py', 'ana_alt_pred_nubar_pz',
                                           'true_tau_minus_E', 'true_tau_minus_px', 'true_tau_minus_py', 'true_tau_minus_pz',
                                           'true_tau_plus_E',  'true_tau_plus_px',  'true_tau_plus_py',  'true_tau_plus_pz',
                                           'pred_tau_minus_E', 'pred_tau_minus_px', 'pred_tau_minus_py', 'pred_tau_minus_pz',
                                           'pred_tau_plus_E',  'pred_tau_plus_px',  'pred_tau_plus_py',  'pred_tau_plus_pz',
                                           'ana_pred_tau_minus_E', 'ana_pred_tau_minus_px', 'ana_pred_tau_minus_py', 'ana_pred_tau_minus_pz',
                                           'ana_pred_tau_plus_E',  'ana_pred_tau_plus_px',  'ana_pred_tau_plus_py',  'ana_pred_tau_plus_pz',
                                           'ana_alt_pred_tau_minus_E', 'ana_alt_pred_tau_minus_px', 'ana_alt_pred_tau_minus_py', 'ana_alt_pred_tau_minus_pz',
                                           'ana_alt_pred_tau_plus_E',  'ana_alt_pred_tau_plus_px',  'ana_alt_pred_tau_plus_py',  'ana_alt_pred_tau_plus_pz',
                                           'taun_haspizero', 'taup_haspizero',
                                           'taup_pi1_E', 'taup_pi1_px', 'taup_pi1_py', 'taup_pi1_pz',
                                           'taup_pizero1_E', 'taup_pizero1_px', 'taup_pizero1_py', 'taup_pizero1_pz',
                                           'taun_pi1_E', 'taun_pi1_px', 'taun_pi1_py', 'taun_pi1_pz',
                                           'taun_pizero1_E', 'taun_pizero1_px', 'taun_pizero1_py', 'taun_pizero1_pz',
                                           ])    

        if predictions_alt is not None:
            results_sf_extra = pd.DataFrame(data=np.concatenate([predictions_alt, pred_taus_alt], axis=1),
                                  columns=[
                                           'alt_pred_nu_E', 'alt_pred_nu_px', 'alt_pred_nu_py', 'alt_pred_nu_pz',
                                           'alt_pred_nubar_E', 'alt_pred_nubar_px', 'alt_pred_nubar_py', 'alt_pred_nubar_pz',
                                             'alt_pred_tau_minus_E', 'alt_pred_tau_minus_px', 'alt_pred_tau_minus_py', 'alt_pred_tau_minus_pz',
                                             'alt_pred_tau_plus_E',  'alt_pred_tau_plus_px',  'alt_pred_tau_plus_py',  'alt_pred_tau_plus_pz',
                                           ])
            results_df = pd.concat([results_df, results_sf_extra], axis=1)    
    
        # compute predicted mass of taus and Z boson and store on dataframe
        pred_tau_minus_mass = np.sqrt(np.maximum(pred_taus[:,0]**2 - pred_taus[:,1]**2 - pred_taus[:,2]**2 - pred_taus[:,3]**2, 0))
        pred_tau_plus_mass  = np.sqrt(np.maximum(pred_taus[:,4]**2 - pred_taus[:,5]**2 - pred_taus[:,6]**2 - pred_taus[:,7]**2, 0))
        pred_z_mass = np.sqrt(np.maximum((pred_taus[:,0] + pred_taus[:,4])**2 - (pred_taus[:,1] + pred_taus[:,5])**2 - (pred_taus[:,2] + pred_taus[:,6])**2 - (pred_taus[:,3] + pred_taus[:,7])**2, 0))
        results_df['true_tau_minus_mass'] = np.sqrt(np.maximum(true_taus[:,0]**2 - true_taus[:,1]**2 - true_taus[:,2]**2 - true_taus[:,3]**2, 0))
        results_df['true_tau_plus_mass'] = np.sqrt(np.maximum(true_taus[:,4]**2 - true_taus[:,5]**2 - true_taus[:,6]**2 - true_taus[:,7]**2, 0))
        results_df['pred_tau_minus_mass'] = pred_tau_minus_mass
        results_df['pred_tau_plus_mass'] = pred_tau_plus_mass
        if predictions_alt is not None:
            results_df['alt_pred_tau_minus_mass'] = np.sqrt(np.maximum(pred_taus_alt[:,0]**2 - pred_taus_alt[:,1]**2 - pred_taus_alt[:,2]**2 - pred_taus_alt[:,3]**2, 0))
            results_df['alt_pred_tau_plus_mass'] = np.sqrt(np.maximum(pred_taus_alt[:,4]**2 - pred_taus_alt[:,5]**2 - pred_taus_alt[:,6]**2 - pred_taus_alt[:,7]**2, 0))
        results_df['ana_pred_tau_minus_mass'] = np.sqrt(np.maximum(ana_pred_taus[:,0]**2 - ana_pred_taus[:,1]**2 - ana_pred_taus[:,2]**2 - ana_pred_taus[:,3]**2, 0))
        results_df['ana_pred_tau_plus_mass'] = np.sqrt(np.maximum(ana_pred_taus[:,4]**2 - ana_pred_taus[:,5]**2 - ana_pred_taus[:,6]**2 - ana_pred_taus[:,7]**2, 0))
        results_df['pred_z_mass'] = pred_z_mass

        # get spin vars for true_tau
        print("Computing spin variables...")
        results_df = compute_spin_vars(results_df, tau_prefix='true_')
        # get spin vars for ana_pred_tau
        results_df = compute_spin_vars(results_df, tau_prefix='ana_pred_')
        # get spin vars for pred_tau
        results_df = compute_spin_vars(results_df, tau_prefix='pred_')
        # get spin vars for alt_pred_tau if present
        if predictions_alt is not None:
            results_df = compute_spin_vars(results_df, tau_prefix='alt_pred_')


        # write the results dataframe to a pickle file
        results_df.to_pickle(f"{output_dir}/output_results.pkl")
    
        # write root file aswell
        output_root_file = f"{output_dir}/output_results.root"
    
        with uproot3.recreate(output_root_file) as f:
            # Create the tree inside the file (name it "tree")
            f["tree"] = uproot3.newtree({col: np.float32 for col in results_df.columns})
        
            # Fill the tree
            f["tree"].extend(results_df.to_dict(orient='list'))

        # make a few plots of the samples PDFs vs the analytical solutions for some events
        for event_number in [0, 1, 2, 3, 4]:
            save_sampled_pdfs(
                model=model,
                dataset=test_dataset,
                df=results_df,
                output_features=['nubar_px', 'nubar_py', 'nubar_pz',
                                 'nu_px', 'nu_py', 'nu_pz'],
                event_number=event_number,
                num_samples=50000,
                bins=100,
                outdir=f"{output_dir}/pdf_slices_sampled"
            )