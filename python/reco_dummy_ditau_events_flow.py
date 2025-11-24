import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
import nflows
from nflows.nn import nets
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.lu import LULinear
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform, AffineCouplingTransform
from nflows.flows.base import Flow
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os

from reco_dummy_ditau_events import RegressionDataset, project_4vec_euclidean_df, plot_loss

def NormalizingFlow(input_size=8, 
                    context_features=6, 
                    num_layers=8, 
                    num_bins=8, 
                    tail_bound=2.0, 
                    hidden_size=64, 
                    num_blocks=2,
                    affine_hidden_size=64,
                    affine_num_blocks=2):
    """Creates a normalizing flow model using nflows library."""
    transforms = []

    def create_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features, context_features=context_features,
            hidden_features=hidden_size, num_blocks=num_blocks,
            use_batch_norm=False,
            activation=nn.ReLU())

    def create_affine_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features, context_features=context_features,
            hidden_features=affine_hidden_size, num_blocks=affine_num_blocks,
            use_batch_norm=False,
            activation=nn.ReLU())

    for _ in range(num_layers):
        mask = nflows.utils.torchutils.create_mid_split_binary_mask(input_size) 
        # if features are ordered so that tau- features come first and then tau+ features and they have the same number 
        # then this mask will alternate which use tau is used to learn the spline parameters for the other tau  
        transforms.append(ReversePermutation(features=input_size))
        transforms.append(nflows.transforms.LULinear(input_size))
        transforms.append(AffineCouplingTransform(mask, create_affine_net))
        transforms.append(ReversePermutation(features=input_size))
        transforms.append(nflows.transforms.LULinear(input_size))
        transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask, create_net, tails='linear', num_bins=num_bins, tail_bound=tail_bound))
        # add a LU triangular layer
        #transforms.append(nflows.transforms.LULinear(input_size))

    transform = CompositeTransform(transforms)
    distribution = StandardNormal([input_size])
    flow = Flow(transform, distribution)
    return flow

def train_model(model, optimizer, train_dataloader, test_dataloader, num_epochs=10, verbose=True, output_plots_dir=None,
    save_every_N=None, recompute_train_loss=True, scheduler=None):
    model.to(device)
    #best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}


    for epoch in range(1, num_epochs+1):
        running_loss=0
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = -model.log_prob(inputs=y, context=X).mean()
            loss.backward()
            optimizer.step()
            if verbose and epoch<3 and batch % 100 == 0:
                print(f'Batch {batch} | loss {loss.item()}')
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

        if scheduler:
            lr = scheduler.get_last_lr()
            scheduler.step(val_loss)

        if verbose and epoch % 1 == 0:
            LR_string = f" | LR: {lr[0]:.2e}" if scheduler else ""
            print(f'Epoch {epoch} | loss {train_loss} | val_loss {val_loss} | lr {LR_string}')

        if epoch > 0 and output_plots_dir: plot_loss(history["train_loss"], history["val_loss"], output_dir=output_plots_dir)

        if save_every_N and epoch % save_every_N == 0 and output_plots_dir:
            print(f"Saving model checkpoint at epoch {epoch}...")
            torch.save(model.state_dict(), f'{output_plots_dir}/partial_model.pth')

    print("Training Completed. Trained for {} epochs.".format(epoch))

    return best_val_loss, history

if __name__ == '__main__':


    argparser = argparse.ArgumentParser()
    argparser.add_argument('--stages', '-s', help='a list of stages to run', type=int, nargs="+")
    argparser.add_argument('--model_name', '-m', help='the name of the model output name', type=str, default='dummy_ditau_nu_regression_model_flow')
    argparser.add_argument('--n_epochs', '-n', help='number of training epochs', type=int, default=10)
    argparser.add_argument('--n_trials', '-t', help='number of hyperparameter optimization trials', type=int, default=100)
    argparser.add_argument('--reload', '-r', help='reload from existing model', action='store_true')
    args = argparser.parse_args()

    # make output directory called outputs_{model_name}, with plots subdirectory
    output_dir = f"outputs_{args.model_name}_max_epochs_{args.n_epochs}"
    output_plots_dir = f"{output_dir}/plots"
    os.makedirs(output_plots_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_pickle("dummy_z_ditau_events_10M.pkl")
    #df = pd.read_pickle("dummy_z_ditau_events.pkl")

    # input featurs will be context features
    input_features = [ 'pi_minus_E', 'pi_minus_px', 'pi_minus_py', 'pi_minus_pz',
                       'pi_plus_E',  'pi_plus_px',  'pi_plus_py',  'pi_plus_pz' ]
                       
    
    output_features = [ 'nu_px', 'nu_py', 'nu_pz',
                        'nubar_px', 'nubar_py', 'nubar_pz' ]

    train_size = int(0.8 * len(df))
    test_size = len(df) - train_size
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    # define datasets and normalize inputs (not outputs)
    train_dataset = RegressionDataset(train_df, input_features, output_features, normalize_inputs=True, normalize_outputs=True)
    in_mean, in_std = train_dataset.input_mean, train_dataset.input_std
    out_mean, out_std = train_dataset.output_mean, train_dataset.output_std
    test_dataset = RegressionDataset(test_df, input_features, output_features, normalize_inputs=True, normalize_outputs=True,
                                      input_mean=in_mean, input_std=in_std, output_mean=out_mean, output_std=out_std)

    # store the means and stds used for normalization
    np.savez(f'{output_dir}/normalization_params.npz',
             input_mean=in_mean, input_std=in_std,
             output_mean=out_mean, output_std=out_std)


    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    num_epochs = args.n_epochs

    optimize = 1 in args.stages
    train = 2 in args.stages
    add_analytical_solutions = 3 in args.stages
    test = 4 in args.stages

    def setup_model_and_training(hp, verbose=True, reload=False):

        train_dataloader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False)

        model = NormalizingFlow(input_size=len(output_features), context_features=len(input_features),
                                 num_layers=hp['num_layers'], num_bins=hp['num_bins'], tail_bound=hp['tail_bound'], 
                                 hidden_size=hp['hidden_size'], num_blocks=hp['num_blocks'],
                                 affine_hidden_size=hp['affine_hidden_size'], affine_num_blocks=hp['affine_num_blocks'])

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

        optimizer = optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1**.5, min_lr=1e-8, verbose=True)

        model.to(device)

        return model, optimizer, train_dataloader, test_dataloader, scheduler

    hp = {
        'batch_size': 2048*2,
        'num_layers': 4,
        'num_bins': 16,
        'tail_bound': 3.0,
        'hidden_size': 100,
        'num_blocks': 2,
        'affine_hidden_size': 100,
        'affine_num_blocks': 2,
        'lr': 1e-3,
        'weight_decay': 1e-4
    }

    model, optimizer, train_loader, test_loader, scheduler = setup_model_and_training(hp, reload=args.reload)

    if optimize:
        print("Starting hyperparameter optimization...")

        import torch.optim as optim
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
                'batch_size': trial.suggest_categorical('batch_size', [1024, 2048, 4096, 8192, 16384]),
                'num_layers': trial.suggest_int('num_layers', 2, 10),
                'num_bins': trial.suggest_int('num_bins', 8, 20, step=2),
                'tail_bound': trial.suggest_float('tail_bound', 1.0, 5.0, step=1.0),
                'hidden_size': trial.suggest_int('hidden_size', 50, 300, step=50),
                'num_blocks': trial.suggest_int('num_blocks', 1, 5),
                'affine_hidden_size': trial.suggest_int('affine_hidden_size', 50, 300, step=50),
                'affine_num_blocks': trial.suggest_int('affine_num_blocks', 1, 5),
                'lr': trial.suggest_loguniform('lr', 1e-6, 1e-2),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
            }

            model, optimizer, train_loader, test_loader, scheduler = setup_model_and_training(hp, verbose=False)

    if train:
    
        print("Starting training...")

        best_val_loss, history = train_model(model, optimizer, train_loader, test_loader, num_epochs=num_epochs, verbose=True, output_plots_dir=output_plots_dir,
            save_every_N=1, scheduler=scheduler)

        model_name = args.model_name
        torch.save(model.state_dict(), f'{output_dir}/{model_name}.pth')


    if test:
        print("Starting testing...")

        import matplotlib.pyplot as plt
        import uproot3
    
        test_df = pd.read_pickle("dummy_ditau_events_10M_test_df_with_analytical_solutions.pkl")

        model_name = args.model_name
        model_path = f'{output_dir}/{model_name}.pth'
        # check if model exists, if not take  partial model
        if not os.path.exists(model_path):
            model_path = f'{output_plots_dir}/partial_model.pth'

        try:
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

        print(predictions_norm[:10])

        # destandardize predictions so that they are in physical units
        predictions = test_dataset.destandardize_outputs(predictions_norm).cpu().numpy()


        print(predictions[:10])

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
    
        # get true taus by summing with pis
        pi_minus = test_df[['pi_minus_E', 'pi_minus_px', 'pi_minus_py', 'pi_minus_pz']].values
        pi_plus  = test_df[['pi_plus_E',  'pi_plus_px',  'pi_plus_py',  'pi_plus_pz']].values

        true_taus = true_values + np.concatenate([pi_minus, pi_plus], axis=1)
        pred_taus = predictions + np.concatenate([pi_minus, pi_plus], axis=1)
    
        # collect true and predicted nus true and predicted taus AND pi's into pandas dataframe, lable the collumns
    
        results_df = pd.DataFrame(data=np.concatenate([true_values, predictions, true_taus, pred_taus, pi_minus, pi_plus], axis=1),
                                  columns=['true_nu_E', 'true_nu_px', 'true_nu_py', 'true_nu_pz',
                                           'true_nubar_E', 'true_nubar_px', 'true_nubar_py', 'true_nubar_pz',
                                           'pred_nu_E', 'pred_nu_px', 'pred_nu_py', 'pred_nu_pz',
                                           'pred_nubar_E', 'pred_nubar_px', 'pred_nubar_py', 'pred_nubar_pz',
                                            'true_tau_minus_E', 'true_tau_minus_px', 'true_tau_minus_py', 'true_tau_minus_pz',
                                            'true_tau_plus_E',  'true_tau_plus_px',  'true_tau_plus_py',  'true_tau_plus_pz',
                                            'pred_tau_minus_E', 'pred_tau_minus_px', 'pred_tau_minus_py', 'pred_tau_minus_pz',
                                            'pred_tau_plus_E',  'pred_tau_plus_px',  'pred_tau_plus_py',  'pred_tau_plus_pz',
                                            'pi_minus_E', 'pi_minus_px', 'pi_minus_py', 'pi_minus_pz',
                                            'pi_plus_E',  'pi_plus_px',  'pi_plus_py',  'pi_plus_pz'
                                           ])
    
        # compute predicted mass of taus and Z boson and store on dataframe
        pred_tau_minus_mass = np.sqrt(np.maximum(pred_taus[:,0]**2 - pred_taus[:,1]**2 - pred_taus[:,2]**2 - pred_taus[:,3]**2, 0))
        pred_tau_plus_mass  = np.sqrt(np.maximum(pred_taus[:,4]**2 - pred_taus[:,5]**2 - pred_taus[:,6]**2 - pred_taus[:,7]**2, 0))
        pred_z_mass = np.sqrt(np.maximum((pred_taus[:,0] + pred_taus[:,4])**2 - (pred_taus[:,1] + pred_taus[:,5])**2 - (pred_taus[:,2] + pred_taus[:,6])**2 - (pred_taus[:,3] + pred_taus[:,7])**2, 0))
        results_df['pred_tau_minus_mass'] = pred_tau_minus_mass
        results_df['pred_tau_plus_mass'] = pred_tau_plus_mass
        results_df['pred_z_mass'] = pred_z_mass


        # add analytical results from test_df to results_df, loop obver E, px, py, pz, loop over particle types, loop over sol 0 and 1
        for sol in [0,1]:
            for particle in ['nu', 'nubar', 'tau_plus', 'tau_minus']:
                for comp in ['E', 'px', 'py', 'pz']:
                    results_df[f'analytical_sol_{sol}_{particle}_{comp}'] = test_df[f'analytical_sol_{sol}_{particle}_{comp}'].to_numpy()

        # compute average of 2 analytical solutions and add to results_df
        for particle in ['nu', 'nubar', 'tau_plus', 'tau_minus']:
            for comp in ['E', 'px', 'py', 'pz']:
                results_df[f'analytical_avg_{particle}_{comp}'] = 0.5 * (test_df[f'analytical_sol_0_{particle}_{comp}'].to_numpy() + test_df[f'analytical_sol_1_{particle}_{comp}'].to_numpy())

        # get average analytical tau masses and z masses and add to results_df
        analytical_tau_minus_mass = np.sqrt(np.maximum(results_df['analytical_avg_tau_minus_E']**2 - results_df['analytical_avg_tau_minus_px']**2 - results_df['analytical_avg_tau_minus_py']**2 - results_df['analytical_avg_tau_minus_pz']**2, 0))
        analytical_tau_plus_mass  = np.sqrt(np.maximum(results_df['analytical_avg_tau_plus_E']**2 - results_df['analytical_avg_tau_plus_px']**2 - results_df['analytical_avg_tau_plus_py']**2 - results_df['analytical_avg_tau_plus_pz']**2, 0))
        analytical_z_mass = np.sqrt(np.maximum((results_df['analytical_avg_tau_minus_E'] + results_df['analytical_avg_tau_plus_E'])**2 - (results_df['analytical_avg_tau_minus_px'] + results_df['analytical_avg_tau_plus_px'])**2 - (results_df['analytical_avg_tau_minus_py'] + results_df['analytical_avg_tau_plus_py'])**2 - (results_df['analytical_avg_tau_minus_pz'] + results_df['analytical_avg_tau_plus_pz'])**2, 0))
        results_df['analytical_avg_tau_minus_mass'] = analytical_tau_minus_mass
        results_df['analytical_avg_tau_plus_mass'] = analytical_tau_plus_mass
        results_df['analytical_avg_z_mass'] = analytical_z_mass

        # add analytical x and q as well
        for comp in ['E', 'px', 'py', 'pz']:
            results_df[f'analytical_x_{comp}'] = test_df[f'analytical_x_{comp}'].to_numpy()
            results_df[f'analytical_q_{comp}'] = test_df[f'analytical_q_{comp}'].to_numpy()
    
        # add predicted x by projecting predicted tau plus onto analytical q
        results_df = project_4vec_euclidean_df(results_df)
    
        # store taus and neutrinos with the x component subtracted, for pred and true
    
        for comp in ['E', 'px', 'py', 'pz']:
            results_df[f'pred_nu_no_x_{comp}'] = results_df[f'pred_nu_{comp}'] + results_df[f'pred_x_{comp}']
            results_df[f'pred_nubar_no_x_{comp}'] = results_df[f'pred_nubar_{comp}'] - results_df[f'pred_x_{comp}'] 
            results_df[f'pred_tau_minus_no_x_{comp}'] = results_df[f'pred_tau_minus_{comp}'] + results_df[f'pred_x_{comp}']
            results_df[f'pred_tau_plus_no_x_{comp}'] = results_df[f'pred_tau_plus_{comp}'] - results_df[f'pred_x_{comp}']   
            results_df[f'true_nu_no_x_{comp}'] = results_df[f'true_nu_{comp}'] + results_df[f'analytical_x_{comp}']
            results_df[f'true_nubar_no_x_{comp}'] = results_df[f'true_nubar_{comp}'] - results_df[f'analytical_x_{comp}']
            results_df[f'true_tau_minus_no_x_{comp}'] = results_df[f'true_tau_minus_{comp}'] + results_df[f'analytical_x_{comp}']
            results_df[f'true_tau_plus_no_x_{comp}'] = results_df[f'true_tau_plus_{comp}'] - results_df[f'analytical_x_{comp}']
    
        # write the results dataframe to a pickle file
        results_df.to_pickle(f"{output_dir}/dummy_ditau_nu_regression_results.pkl")
    
        # write root file aswell
        output_root_file = f"{output_dir}/dummy_ditau_nu_regression_results.root"
    
        with uproot3.recreate(output_root_file) as f:
            # Create the tree inside the file (name it "tree")
            f["tree"] = uproot3.newtree({col: np.float32 for col in results_df.columns})
        
            # Fill the tree
            f["tree"].extend(results_df.to_dict(orient='list'))