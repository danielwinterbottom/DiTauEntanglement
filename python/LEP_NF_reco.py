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

def setup_model_and_training(hp, verbose=True, reload=False, batch_norm=False):
    train_dataloader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False)

    model = ConditionalFlow(input_dim=len(output_features), raw_condition_dim=len(input_features),
                            context_dim=hp['condition_net_output_size'],
                            cond_hidden_dim=hp['condition_net_hidden_size'],
                            num_layers=hp['num_layers'], num_bins=hp['num_bins'], tail_bound=hp['tail_bound'], 
                            hidden_size=hp['hidden_size'], num_blocks=hp['num_blocks'],
                            affine_hidden_size=hp['affine_hidden_size'], affine_num_blocks=hp['affine_num_blocks'],batch_norm=batch_norm)

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

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()   

    argparser.add_argument('--stages', '-s', help='a list of stages to run', type=int, nargs="+")
    argparser.add_argument('--model_name', '-m', help='the name of the model output name', type=str, default='LEP_nflow_model')
    argparser.add_argument('--n_epochs', '-n', help='number of training epochs', type=int, default=10)
    argparser.add_argument('--n_trials', '-t', help='number of hyperparameter optimization trials', type=int, default=100)
    argparser.add_argument('--reload', '-r', help='reload from existing model', action='store_true')
    argparser.add_argument('--inc_reco_taus', help='whether to include the taus reconstructed by the analytical model as inputs', action='store_true')
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

        df.to_pickle('ditau_nu_regression_ee_to_tauhtauh_dataframe.pkl')

        print("Dataframe prepared and saved.")

    else: # load the dataframe
        df = pd.read_pickle('ditau_nu_regression_ee_to_tauhtauh_dataframe.pkl')

    #print the names of all the columns and information on the number of events in the dataframe
    print('Columns in dataframe:', df.columns.tolist())
    print('Number of events in dataframe:', len(df))
    # print number of input features
    print('Number of input features:', len(input_features))

    # split dataset into train and test

    train_size = int(0.9 * len(df))
    test_size = len(df) - train_size
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # define datasets and normalize inputs and outputs
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
            plt.yscale("log")
            plt.savefig(os.path.join(output_plots_dir, "optimization_history_log.pdf"))
            plt.close()

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
                'batch_size': trial.suggest_categorical('batch_size', [2048, 4096, 8192, 16384]),
                'num_layers': trial.suggest_int('num_layers', 2, 10),
                'num_bins': trial.suggest_int('num_bins', 8, 20, step=2),
                'tail_bound': trial.suggest_float('tail_bound', 1.0, 5.0, step=1.0),
                'hidden_size': trial.suggest_int('hidden_size', 50, 300, step=50),
                'num_blocks': trial.suggest_int('num_blocks', 1, 2),
                'affine_hidden_size': trial.suggest_int('affine_hidden_size', 50, 300, step=50),
                'affine_num_blocks': trial.suggest_int('affine_num_blocks', 1, 2),
                'lr': trial.suggest_loguniform('lr', 1e-6, 1e-2),
                #'epochs_to_10perc_lr': trial.suggest_int('epochs_to_10perc_lr', 20, 100, step=10),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
                'condition_net_hidden_size': trial.suggest_int('condition_net_hidden_size', 50, 300, step=50),
                'condition_net_num_blocks': trial.suggest_int('condition_net_num_blocks', 1, 5),
                'condition_net_output_size': trial.suggest_int('condition_net_output_size', 6, 30),
                'num_epochs': args.n_epochs,
            }

            model, optimizer, train_loader, test_loader, scheduler = setup_model_and_training(hp, verbose=False)

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

    # load the model with the best hyperparameters found in stage 2
    # for now we just hard code the hyperparameters
    hp = {
        'batch_size': 8192,
        'num_layers': 7,
        'num_bins': 16,
        'tail_bound': 3.0,
        'hidden_size': 200,
        'num_blocks': 1,
        'affine_hidden_size': 100,
        'affine_num_blocks': 1,
        'lr': 0.001,
        'weight_decay': 1e-4,
        #'epochs_to_10perc_lr': 100,
        'condition_net_hidden_size': 200,
        'condition_net_num_blocks': 4,
        'condition_net_output_size': 20,
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

        X_test, _ = test_dataset[:]
        # move X_test and model to CPU
        X_test = X_test.cpu()
        model = model.cpu()
        with torch.no_grad():
            predictions_norm = model.sample(num_samples=1, context=X_test).squeeze()     

        # destandardize predictions so that they are in physical units
        predictions = test_dataset.destandardize_outputs(predictions_norm).cpu().numpy()

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

        # define alternative prediction by taking most probable value from flow instead of sampling
        # to do this we sample 100 times and take the case with the best log probability

#        num_draws = 100
#        
#        with torch.no_grad():
#            samples_norm = model.sample(num_samples=num_draws, context=X_test)
#
#        B, D, F = samples_norm.shape
#        
#        flat_samples = samples_norm.reshape(B * D, F)
#        flat_context = X_test.repeat_interleave(D, dim=0)
#        
#        with torch.no_grad():
#            flat_log_probs = model.log_prob(flat_samples, context=flat_context)
#        
#        log_probs = flat_log_probs.view(B, D)
#
#        best_idx = torch.argmax(log_probs, dim=1)
#        batch_idx = torch.arange(B, device=samples_norm.device)
#        samples_norm_alt = samples_norm[batch_idx, best_idx]
#        
#        # destandardize
#        samples_alt = test_dataset.destandardize_outputs(samples_norm_alt).cpu().numpy()

        # estimate most likely solution using flow_map_predict function
        samples_norm_alt, samples_alt = flow_map_predict(
            model,
            X_test,
            test_dataset=test_dataset,
            num_draws=100,
            chunk_size=50000
        )

        # unpack MAP outputs → **use alt_* names here only**
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

        results_df = pd.DataFrame(data=np.concatenate([true_values, predictions, predictions_alt, ana_pred_values, ana_alt_pred_values, true_taus, pred_taus, pred_taus_alt, ana_pred_taus, ana_alt_pred_taus, taun_haspizero, taup_haspizero], axis=1),
                                  columns=['true_nu_E', 'true_nu_px', 'true_nu_py', 'true_nu_pz',
                                           'true_nubar_E', 'true_nubar_px', 'true_nubar_py', 'true_nubar_pz',
                                           'pred_nu_E', 'pred_nu_px', 'pred_nu_py', 'pred_nu_pz',
                                           'pred_nubar_E', 'pred_nubar_px', 'pred_nubar_py', 'pred_nubar_pz',
                                           'alt_pred_nu_E', 'alt_pred_nu_px', 'alt_pred_nu_py', 'alt_pred_nu_pz',
                                           'alt_pred_nubar_E', 'alt_pred_nubar_px', 'alt_pred_nubar_py', 'alt_pred_nubar_pz',
                                           'ana_pred_nu_E', 'ana_pred_nu_px', 'ana_pred_nu_py', 'ana_pred_nu_pz',
                                           'ana_pred_nubar_E', 'ana_pred_nubar_px', 'ana_pred_nubar_py', 'ana_pred_nubar_pz',
                                           'ana_alt_pred_nu_E', 'ana_alt_pred_nu_px', 'ana_alt_pred_nu_py', 'ana_alt_pred_nu_pz',
                                           'ana_alt_pred_nubar_E', 'ana_alt_pred_nubar_px', 'ana_alt_pred_nubar_py', 'ana_alt_pred_nubar_pz',
                                           'true_tau_minus_E', 'true_tau_minus_px', 'true_tau_minus_py', 'true_tau_minus_pz',
                                           'true_tau_plus_E',  'true_tau_plus_px',  'true_tau_plus_py',  'true_tau_plus_pz',
                                           'pred_tau_minus_E', 'pred_tau_minus_px', 'pred_tau_minus_py', 'pred_tau_minus_pz',
                                           'pred_tau_plus_E',  'pred_tau_plus_px',  'pred_tau_plus_py',  'pred_tau_plus_pz',
                                           'alt_pred_tau_minus_E', 'alt_pred_tau_minus_px', 'alt_pred_tau_minus_py', 'alt_pred_tau_minus_pz',
                                           'alt_pred_tau_plus_E',  'alt_pred_tau_plus_px',  'alt_pred_tau_plus_py',  'alt_pred_tau_plus_pz',
                                           'ana_pred_tau_minus_E', 'ana_pred_tau_minus_px', 'ana_pred_tau_minus_py', 'ana_pred_tau_minus_pz',
                                           'ana_pred_tau_plus_E',  'ana_pred_tau_plus_px',  'ana_pred_tau_plus_py',  'ana_pred_tau_plus_pz',
                                           'ana_alt_pred_tau_minus_E', 'ana_alt_pred_tau_minus_px', 'ana_alt_pred_tau_minus_py', 'ana_alt_pred_tau_minus_pz',
                                           'ana_alt_pred_tau_plus_E',  'ana_alt_pred_tau_plus_px',  'ana_alt_pred_tau_plus_py',  'ana_alt_pred_tau_plus_pz',
                                           'taun_haspizero', 'taup_haspizero'
                                           ])


    
        # compute predicted mass of taus and Z boson and store on dataframe
        pred_tau_minus_mass = np.sqrt(np.maximum(pred_taus[:,0]**2 - pred_taus[:,1]**2 - pred_taus[:,2]**2 - pred_taus[:,3]**2, 0))
        pred_tau_plus_mass  = np.sqrt(np.maximum(pred_taus[:,4]**2 - pred_taus[:,5]**2 - pred_taus[:,6]**2 - pred_taus[:,7]**2, 0))
        pred_z_mass = np.sqrt(np.maximum((pred_taus[:,0] + pred_taus[:,4])**2 - (pred_taus[:,1] + pred_taus[:,5])**2 - (pred_taus[:,2] + pred_taus[:,6])**2 - (pred_taus[:,3] + pred_taus[:,7])**2, 0))
        results_df['true_tau_minus_mass'] = np.sqrt(np.maximum(true_taus[:,0]**2 - true_taus[:,1]**2 - true_taus[:,2]**2 - true_taus[:,3]**2, 0))
        results_df['true_tau_plus_mass'] = np.sqrt(np.maximum(true_taus[:,4]**2 - true_taus[:,5]**2 - true_taus[:,6]**2 - true_taus[:,7]**2, 0))
        results_df['pred_tau_minus_mass'] = pred_tau_minus_mass
        results_df['pred_tau_plus_mass'] = pred_tau_plus_mass
        results_df['alt_pred_tau_minus_mass'] = np.sqrt(np.maximum(pred_taus_alt[:,0]**2 - pred_taus_alt[:,1]**2 - pred_taus_alt[:,2]**2 - pred_taus_alt[:,3]**2, 0))
        results_df['alt_pred_tau_plus_mass'] = np.sqrt(np.maximum(pred_taus_alt[:,4]**2 - pred_taus_alt[:,5]**2 - pred_taus_alt[:,6]**2 - pred_taus_alt[:,7]**2, 0))
        results_df['ana_pred_tau_minus_mass'] = np.sqrt(np.maximum(ana_pred_taus[:,0]**2 - ana_pred_taus[:,1]**2 - ana_pred_taus[:,2]**2 - ana_pred_taus[:,3]**2, 0))
        results_df['ana_pred_tau_plus_mass'] = np.sqrt(np.maximum(ana_pred_taus[:,4]**2 - ana_pred_taus[:,5]**2 - ana_pred_taus[:,6]**2 - ana_pred_taus[:,7]**2, 0))
        results_df['pred_z_mass'] = pred_z_mass

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