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

    gamma = -math.log(0.1)/(40 * len(train_dataloader))
    scheduler = CosineAnnealingExpDecayLR(optimizer, T_max=2 * len(train_dataloader), gamma=gamma)

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

def plot_loss(loss_values, val_loss_values, output_dir='nn_plots'):
    plt.figure()
    plt.plot(range(1, len(loss_values)+1), loss_values, label='train loss')
    plt.plot(range(1, len(val_loss_values)+1), val_loss_values, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_vs_epoch_dummy.pdf')
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
        'condition_net_hidden_size': 200,
        'condition_net_num_blocks': 4,
        'condition_net_output_size': 20,
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

        model_name = args.model_name
        model_path = f'{output_dir}/{model_name}.pth'
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
        nu_px = predictions[:,0]
        nu_py = predictions[:,1]
        nu_pz = predictions[:,2]
        nubar_px = predictions[:,3]
        nubar_py = predictions[:,4]
        nubar_pz = predictions[:,5]
        nu_E = np.sqrt(nu_px**2 + nu_py**2 + nu_pz**2)
        nubar_E = np.sqrt(nubar_px**2 + nubar_py**2 + nubar_pz**2)
        predictions = np.column_stack((nu_E, nu_px, nu_py, nu_pz, nubar_E, nubar_px, nubar_py, nubar_pz))
        true_values = test_df[output_features].values

        # get E components for true values as well
        true_nu_px = true_values[:,0]
        true_nu_py = true_values[:,1]
        true_nu_pz = true_values[:,2]
        true_nubar_px = true_values[:,3]
        true_nubar_py = true_values[:,4]
        true_nubar_pz = true_values[:,5]
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

        # get analytical precitions using reco_taup_nu and reco_taun_nu
        # first get the pis and pizeros again
        reco_taup_nu = test_df[['reco_taup_nu_px', 'reco_taup_nu_py', 'reco_taup_nu_pz']].values
        reco_taun_nu = test_df[['reco_taun_nu_px', 'reco_taun_nu_py', 'reco_taun_nu_pz']].values
        # compute E for these nus
        reco_taup_nu_E = np.sqrt(reco_taup_nu[:,0]**2 + reco_taup_nu[:,1]**2 + reco_taup_nu[:,2]**2)
        reco_taun_nu_E = np.sqrt(reco_taun_nu[:,0]**2 + reco_taun_nu[:,1]**2 + reco_taun_nu[:,2]**2)
        reco_taup_nu = np.column_stack((reco_taup_nu_E, reco_taup_nu))
        reco_taun_nu = np.column_stack((reco_taun_nu_E, reco_taun_nu))

        ana_pred_values = np.concatenate([reco_taup_nu, reco_taun_nu], axis=1)

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

        ana_alt_pred_values = np.concatenate([reco_alt_taup_nu, reco_alt_taun_nu], axis=1)

        # get the predicted taus as well
        ana_alt_pred_taup = reco_alt_taup_nu + taup_pi + taup_pizero
        ana_alt_pred_taun = reco_alt_taun_nu + taun_pi + taun_pizero
        ana_alt_pred_taus =  np.concatenate([ana_alt_pred_taun, ana_alt_pred_taup], axis=1)

        # collect true and predicted nus true and predicted taus AND pi's into pandas dataframe, lable the collumns

        results_df = pd.DataFrame(data=np.concatenate([true_values, predictions, ana_pred_values, ana_alt_pred_values, true_taus, pred_taus, ana_pred_taus, ana_alt_pred_taus], axis=1),
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
                                           ])


    
        # compute predicted mass of taus and Z boson and store on dataframe
        pred_tau_minus_mass = np.sqrt(np.maximum(pred_taus[:,0]**2 - pred_taus[:,1]**2 - pred_taus[:,2]**2 - pred_taus[:,3]**2, 0))
        pred_tau_plus_mass  = np.sqrt(np.maximum(pred_taus[:,4]**2 - pred_taus[:,5]**2 - pred_taus[:,6]**2 - pred_taus[:,7]**2, 0))
        pred_z_mass = np.sqrt(np.maximum((pred_taus[:,0] + pred_taus[:,4])**2 - (pred_taus[:,1] + pred_taus[:,5])**2 - (pred_taus[:,2] + pred_taus[:,6])**2 - (pred_taus[:,3] + pred_taus[:,7])**2, 0))
        results_df['pred_tau_minus_mass'] = pred_tau_minus_mass
        results_df['pred_tau_plus_mass'] = pred_tau_plus_mass
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