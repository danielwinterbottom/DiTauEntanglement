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
import math
from schedules import CosineAnnealingExpDecayLR

from reco_dummy_ditau_events import RegressionDataset, project_4vec_euclidean_df, plot_loss

def NormalizingFlow(input_size=8, 
                    context_features=6, 
                    num_layers=8, 
                    num_bins=8, 
                    tail_bound=2.0, 
                    hidden_size=64, 
                    num_blocks=2,
                    affine_hidden_size=64,
                    affine_num_blocks=2,
                    batch_norm=False):
    """Creates a normalizing flow model using nflows library."""
    transforms = []

    def create_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features, context_features=context_features,
            hidden_features=hidden_size, num_blocks=num_blocks,
            use_batch_norm=batch_norm,
            activation=nn.ReLU())

    def create_affine_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features, context_features=context_features,
            hidden_features=affine_hidden_size, num_blocks=affine_num_blocks,
            use_batch_norm=batch_norm,
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

class ConditionalFlow(nn.Module):
    def __init__(self, 
                 input_dim,
                 raw_condition_dim,
                 context_dim,
                 cond_hidden_dim=64,
                 batch_norm=False,
                 **flow_kwargs
    ):
        super().__init__()

        self.condition_net = nets.ResidualNet(
            in_features=raw_condition_dim,
            out_features=context_dim,
            hidden_features=cond_hidden_dim,
            num_blocks=2,
            activation=nn.ReLU(),
            use_batch_norm=batch_norm
        )

        self.flow = NormalizingFlow(
            input_size=input_dim,
            context_features=context_dim,
            **flow_kwargs
        )

    def log_prob(self, inputs, context):
        cond_embed = self.condition_net(context)
        return self.flow.log_prob(inputs=inputs, context=cond_embed)

    def sample(self, num_samples, context):
        cond_embed = self.condition_net(context)
        return self.flow.sample(num_samples=num_samples, context=cond_embed)

# the following functions are to check that tes tand train data are really independent:

def hash_event(tensor_event):
    """Convert an event tensor to a stable hash."""
    # detach to avoid gradients, move to CPU, convert to numpy bytes
    return hash(tensor_event.detach().cpu().numpy().tobytes())

def check_overlap(loader_a, loader_b):
    print('Checking for overlaps between two datasets...')
    hashes_a = set()
    hashes_b = set()

    n_samples_a = 0
    n_samples_b = 0

    # Process loader A
    for X, y in loader_a:
        for i in range(X.shape[0]):               # iterate over actual samples
            hashes_a.add(hash_event(X[i]))       # or hash (X[i], y[i]) if needed
            n_samples_a += 1

    # Process loader B
    for X, y in loader_b:
        for i in range(X.shape[0]):
            hashes_b.add(hash_event(X[i]))
            n_samples_b += 1
    print(f"Loader A: {n_samples_a} samples, Loader B: {n_samples_b} samples.")
    overlap = hashes_a & hashes_b

    if overlap:
        print(f"Overlap detected: {len(overlap)} events.")
    else:
        print("No overlaps — datasets are disjoint.")

    return overlap

def range_test(model, optimizer, train_dataloader, test_dataloader, min_lr=1e-6, max_lr=1.0, num_iters=2000, n_val_batches=10, device="cpu", output_plots_dir='nn_plots'):
    
    print("Starting range test...")
    model.to(device)
    backup = {k: v.clone() for k, v in model.state_dict().items()}
    # set AdamW optimizer lr to min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = min_lr

    # exp scheduler
    gamma = (max_lr / min_lr) ** (1 / num_iters)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    #create a test X and y using n_val_batches from test_dataloader
    val_X_list = []
    val_y_list = []
    for i, (X_val, y_val) in enumerate(test_dataloader):
        if i >= n_val_batches:
            break
        X_val, y_val = X_val.to(device), y_val.to(device)
        val_X_list.append(X_val)
        val_y_list.append(y_val)
    val_X = torch.cat(val_X_list)
    val_y = torch.cat(val_y_list)


    run_training = True
    lr_values = []
    loss_values = []
    i = 0
    best_loss = float('inf')
    while run_training:
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = -model.log_prob(inputs=y, context=X).mean()
            loss.backward()
            optimizer.step()

            lr = scheduler.get_last_lr()
            lr_values.append(lr[0])

            # evaluate the validation loss
            with torch.no_grad():
                val_loss = -model.log_prob(inputs=val_y, context=val_X).mean()
                loss_values.append(val_loss.item())
                best_loss = min(best_loss, val_loss.item())
                if batch % 100 == 0:
                    # print and plot loss every 100 batches
                    print('batch {} lr: {}, val_loss: {}'.format(batch, lr, val_loss.item()))
                    plot_loss_vs_batch(loss_values, lr_values, output_dir=output_plots_dir)
            scheduler.step()
            # switch back to backup model weights - i.e the initial model
            model.load_state_dict(backup)

            i += 1
            if i == num_iters or val_loss.item() > 4 * best_loss:
                run_training = False
                break

    plot_loss_vs_batch(loss_values, lr_values, output_dir=output_plots_dir)
    print("Range test completed.")

def train_model(model, optimizer, train_dataloader, test_dataloader, num_epochs=10, device="cpu", verbose=True, output_plots_dir=None,
    save_every_N=None, recompute_train_loss=True, scheduler=None):
    model.to(device)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    ## temp - get val loss before trainign to check that it is restarting as expected
    #val_running_loss = 0
    #with torch.no_grad():
    #    for X_val, y_val in test_dataloader:
    #        X_val, y_val = X_val.to(device), y_val.to(device)
    #        val_loss = -model.log_prob(inputs=y_val, context=X_val).mean()
    #        val_running_loss += val_loss.item()
    #val_loss = val_running_loss / len(test_dataloader)
    #print(f"Initial validation loss before training: {val_loss}")

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

        #check overlaps
        #check_overlap(train_dataloader, test_dataloader)

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

        #if scheduler: # if you do it here then it only changes the learning rate once per epoch
        #    lr = scheduler.get_last_lr()
        #    #scheduler.step(val_loss) # only for ReduceLROnPlateau
        #    scheduler.step()

        if verbose and epoch % 1 == 0:
            LR_string = f" | LR: {lr[0]:.2e}" if scheduler else ""
            print(f'Epoch {epoch} | loss {train_loss} | val_loss {val_loss} | lr {LR_string}')

        if epoch > 0 and output_plots_dir: plot_loss(history["train_loss"], history["val_loss"], output_dir=output_plots_dir)

        if save_every_N and epoch % save_every_N == 0 and output_plots_dir:
            print(f"Saving model checkpoint at epoch {epoch}...")
            torch.save(model.state_dict(), f'{output_plots_dir}/partial_model.pth')

    print("Training Completed. Trained for {} epochs.".format(epoch))

    return best_val_loss, history

def plot_loss_vs_batch(loss_values, lr_values, output_dir='nn_plots'):
    plt.figure()
    plt.plot(range(1, len(loss_values)+1), loss_values, label='loss')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_vs_batch_dummy.pdf')
    # now plot loss vs lr_values
    plt.figure()
    plt.plot(lr_values, loss_values, label='loss vs lr')
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_vs_lr_dummy.pdf')
    plt.close()
    # make a log plot of loss vs lr as well
    plt.figure()
    plt.plot(lr_values, loss_values, label='loss vs lr')
    plt.xscale('log')
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_vs_lr_log_dummy.pdf')

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

    for i, v in enumerate(output_features):

        pred_i = predictions[:, i]
        plt.figure(figsize=(6, 4))
        plt.hist(pred_i, bins=bins, density=True, histtype='step', linewidth=2)
        # draw true value as an arrow
        true_value = dataset.destandardize_outputs(y[event_number].unsqueeze(0)).cpu().numpy()[0, i]
        plt.axvline(true_value, color='r', linestyle='--', linewidth=2)

        # also get analytical solutions if available
        analytical_col_0 = f'analytical_sol_0_{v}'
        analytical_col_1 = f'analytical_sol_1_{v}'
        if analytical_col_0 in df.columns and analytical_col_1 in df.columns:
            analytical_sol_0 = df.iloc[event_number][analytical_col_0]
            analytical_sol_1 = df.iloc[event_number][analytical_col_1]
            plt.axvline(analytical_sol_0, color='g', linestyle=':', linewidth=2, label='Analytical Sol 0')
            plt.axvline(analytical_sol_1, color='b', linestyle=':', linewidth=2, label='Analytical Sol 1')
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

    def setup_model_and_training(hp, verbose=True, reload=False, batch_norm=False):

        train_dataloader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False)

        if hp.get('use_condition_net', False):
            print("Using Conditional Flow Model")
            model = ConditionalFlow(input_dim=len(output_features), raw_condition_dim=len(input_features),
                                    context_dim=hp['condition_net_output_size'],
                                    cond_hidden_dim=hp['condition_net_hidden_size'],
                                    num_layers=hp['num_layers'], num_bins=hp['num_bins'], tail_bound=hp['tail_bound'], 
                                    hidden_size=hp['hidden_size'], num_blocks=hp['num_blocks'],
                                    affine_hidden_size=hp['affine_hidden_size'], affine_num_blocks=hp['affine_num_blocks'],batch_norm=batch_norm)


        else:
            model = NormalizingFlow(input_size=len(output_features), context_features=len(input_features),
                                     num_layers=hp['num_layers'], num_bins=hp['num_bins'], tail_bound=hp['tail_bound'], 
                                     hidden_size=hp['hidden_size'], num_blocks=hp['num_blocks'],
                                     affine_hidden_size=hp['affine_hidden_size'], affine_num_blocks=hp['affine_num_blocks'],batch_norm=batch_norm)


        if verbose:
            print(model)
            # print number of parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of model parameters: {total_params}")

        optimizer = optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])

        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1**.5, min_lr=1e-8, verbose=True)

        ## use scheduler cosine annealing that will make learning rate go from initial value to 0 over num_epochs
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer,
        #    T_max=num_epochs,
        #    eta_min=0.0
        #)

        # define cyclic learning rate scheduler based on cosineAnnearlinLR that goes between initial value and 0 every 2 epochs
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2 * len(train_dataloader), eta_min=0.0)

        # use the cyclicLR scheduler to go between hp['lr'] and 1e-4 every 2 epochs
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=hp['lr'], step_size_up=2 * len(train_dataloader), mode='triangular')

        # define gamma so that lr decreases by factor of 10 every 40 epochs
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

    if optimize:
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
                'batch_size': trial.suggest_categorical('batch_size', [1024, 2048, 4096, 8192, 16384]),
                'num_layers': trial.suggest_int('num_layers', 2, 10),
                'num_bins': trial.suggest_int('num_bins', 8, 20, step=2),
                'tail_bound': trial.suggest_float('tail_bound', 1.0, 5.0, step=1.0),
                'hidden_size': trial.suggest_int('hidden_size', 50, 300, step=50),
                'num_blocks': trial.suggest_int('num_blocks', 1, 2),
                'affine_hidden_size': trial.suggest_int('affine_hidden_size', 50, 300, step=50),
                'affine_num_blocks': trial.suggest_int('affine_num_blocks', 1, 2),
                'lr': trial.suggest_loguniform('lr', 1e-6, 1e-2),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
                'use_condition_net': True,
                'condition_net_hidden_size': trial.suggest_int('condition_net_hidden_size', 50, 300, step=50),
                'condition_net_num_blocks': trial.suggest_int('condition_net_num_blocks', 1, 5),
                'condition_net_output_size': trial.suggest_int('condition_net_output_size', 6, 30),
            }

            model, optimizer, train_loader, test_loader, scheduler = setup_model_and_training(hp, verbose=False)

            best_val_loss, history = train_model(model, optimizer, train_loader, test_loader, num_epochs=num_epochs, device=device, verbose=False, output_plots_dir=None,
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

    #TODO: load Hps from optuna study best trial if available
    #hp = {
    #    'batch_size': 2048*2,
    #    'num_layers': 4,
    #    'num_bins': 16,
    #    'tail_bound': 3.0,
    #    'hidden_size': 100,
    #    'num_blocks': 2,
    #    'affine_hidden_size': 100,
    #    'affine_num_blocks': 2,
    #    'lr': 1e-3,
    #    'weight_decay': 1e-4,
    #    'use_condition_net': True,
    #    'condition_net_hidden_size': 100,
    #    'condition_net_num_blocks': 2,
    #    'condition_net_output_size': 10,
    #}

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
        #'lr': 1e-5, #1e-3
        'weight_decay': 1e-4,
        'use_condition_net': True,
        'condition_net_hidden_size': 200,
        'condition_net_num_blocks': 4,
        'condition_net_output_size': 20,
    }

    model, optimizer, train_loader, test_loader, scheduler = setup_model_and_training(hp, reload=args.reload, batch_norm=False) # note using batch_norm=True has issues if trying to restart the training again...

    if train:
    
        print("Starting training...")

        #range_test(model, optimizer, train_loader, test_loader, min_lr=0.00000005, max_lr=0.005, num_iters=2000, n_val_batches=10, device=device, output_plots_dir=output_plots_dir)

        best_val_loss, history = train_model(model, optimizer, train_loader, test_loader, num_epochs=num_epochs, device=device, verbose=True, output_plots_dir=output_plots_dir,
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

        for event_number in [0, 1, 2, 3, 4]:
            save_sampled_pdfs(
                model=model,
                dataset=test_dataset,
                df=results_df,
                output_features=output_features,
                event_number=event_number,
                num_samples=50000,
                bins=100,
                outdir="pdf_slices_sampled"
            )
