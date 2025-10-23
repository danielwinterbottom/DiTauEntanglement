import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import argparse

class RegressionDataset(Dataset):
    def __init__(self, dataframe, input_features, output_features, 
                 input_mean=None, input_std=None, 
                 output_mean=None, output_std=None,
                 normalize_inputs=True, normalize_outputs=False, eps=1e-8):
        """
        A regression dataset that can standardize features using provided means/stds.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data.
        input_features : list[str]
            Names of input columns.
        output_features : list[str]
            Names of target columns.
        input_mean, input_std : torch.Tensor or None
            If given, used for input normalization.
            If None, computed from the current dataframe.
        output_mean, output_std : torch.Tensor or None
            Same as above, for outputs.
        normalize_inputs, normalize_outputs : bool
            Whether to apply standardization.
        eps : float
            Small value to prevent division by zero.
        """
        X = torch.tensor(dataframe[input_features].values, dtype=torch.float32)
        y = torch.tensor(dataframe[output_features].values, dtype=torch.float32)

        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.eps = eps

        # ---- Input normalization ----
        if normalize_inputs:
            if input_mean is None or input_std is None:
                self.input_mean = X.mean(dim=0, keepdim=True)
                self.input_std = X.std(dim=0, keepdim=True).clamp_min(eps)
            else:
                self.input_mean = input_mean
                self.input_std = input_std.clamp_min(eps)

            X = (X - self.input_mean) / self.input_std
        else:
            self.input_mean = torch.zeros(X.shape[1])
            self.input_std = torch.ones(X.shape[1])

        # ---- Output normalization ----
        if normalize_outputs:
            if output_mean is None or output_std is None:
                self.output_mean = y.mean(dim=0, keepdim=True)
                self.output_std = y.std(dim=0, keepdim=True).clamp_min(eps)
            else:
                self.output_mean = output_mean
                self.output_std = output_std.clamp_min(eps)

            y = (y - self.output_mean) / self.output_std
        else:
            self.output_mean = torch.zeros(y.shape[1])
            self.output_std = torch.ones(y.shape[1])

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def destandardize_outputs(self, y_norm):
        """
        Convert standardized outputs back to physical units.
        """
        return y_norm * self.output_std + self.output_mean

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def reset(self):
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss)*(1. + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class SimpleNN(nn.Module):

    def __init__(self, input_dim, output_dim, n_hidden_layers, n_nodes, dropout=0):
        super(SimpleNN, self).__init__()

        layers = OrderedDict()
        layers['input'] = nn.Linear(input_dim, n_nodes)
        layers['bn_input'] = nn.BatchNorm1d(n_nodes)
        layers['relu_input'] = nn.ReLU()
        if dropout > 0:
                layers['dropout_input'] = nn.Dropout(dropout)
        for i in range(n_hidden_layers):
            layers[f'hidden_{i+1}'] = nn.Linear(n_nodes, n_nodes)
            layers[f'hidden_bn_{i+1}'] = nn.BatchNorm1d(n_nodes)
            layers[f'hidden_relu_{i+1}'] = nn.ReLU()
            if dropout > 0:
                layers[f'hidden_dropout_{i+1}'] = nn.Dropout(dropout)
        layers['output'] = nn.Linear(n_nodes, output_dim)

        self.layers = nn.Sequential(layers)

        # print a summry of the model
        print('Model summary:')
        print(self.layers)
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Number of parameters: {n_params}')
 

    def forward(self, x):
        x = self.layers(x)
        return x

def plot_loss(loss_values, val_loss_values, running_loss_values=None):
    plt.figure()
    plt.plot(range(1, len(loss_values)+1), loss_values[1:], label='train loss')
    plt.plot(range(1, len(val_loss_values)+1), val_loss_values[1:], label='validation loss')
    if running_loss_values is not None: plt.plot(range(1, len(running_loss_values)+1), running_loss_values[1:], label='running loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'nn_plots/loss_vs_epoch_dummy.pdf')
    plt.close()

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    n_epochs=50,
    device="cpu",
    verbose=True,
    recompute_train_loss=True,
    early_stopper=None
):
    """
    Train a PyTorch model and evaluate performance, with optional recomputation
    of training loss at the end of each epoch (to match validation evaluation).

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : torch.nn.Module
        Loss function (e.g., nn.MSELoss()).
    optimizer : torch.optim.Optimizer
        Optimizer (e.g., torch.optim.Adam).
    n_epochs : int
        Number of training epochs.
    device : str
        "cpu" or "cuda".
    verbose : bool
        Whether to print epoch losses.
    recompute_train_loss : bool
        If True, recompute the training loss at the end of each epoch using
        the final model weights (more accurate but slower).
    early_stopper : EarlyStopper or None
        Early stopping mechanism to halt training when validation loss stops improving.

    Returns
    -------
    best_val_loss : float
        Minimum validation loss achieved.
    history : dict
        Dictionary containing loss histories:
        {
            "train_loss": [...],
            "val_loss": [...]
        }
    """

    model.to(device)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    if early_stopper: early_stopper.reset()

    for epoch in range(1, n_epochs + 1):
        # --- Training phase ---
        model.train()
        running_train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)

        # --- Optional: recompute train loss with final weights ---
        if recompute_train_loss:
            model.eval()
            with torch.no_grad():
                running_train_loss = 0.0
                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    running_train_loss += criterion(y_pred, y).item()
            train_loss = running_train_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                running_val_loss += criterion(y_pred, y).item()
        val_loss = running_val_loss / len(val_loader)
        best_val_loss = min(best_val_loss, val_loss)

        if early_stopper and early_stopper.early_stop(val_loss):
            print(f"Early stopping triggered for epoch {epoch+1}")
            break

        # --- Store results ---
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            width = len(str(n_epochs))
            print(
                f"Epoch {epoch:{width}d}/{n_epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            )    

        if epoch > 1: plot_loss(history["train_loss"], history["val_loss"])

    return best_val_loss, history

def combined_schedule(epoch, peak_lr, low_lr, start_epochs, ramp_epochs, gamma):
    ramp_factor = peak_lr / low_lr
    if epoch < start_epochs:
        return 1.0  # phase 1: constant low_lr
    elif epoch < start_epochs + ramp_epochs:
        # phase 2: linear ramp
        ramp_progress = (epoch - start_epochs) / ramp_epochs
        return 1.0 + (ramp_factor - 1.0) * ramp_progress
    else:
        # phase 3: exponential decay from peak_lr
        decay_steps = epoch - (start_epochs + ramp_epochs)
        return ramp_factor * (gamma ** decay_steps)

def project_4vec_euclidean_df(df, 
                              p_prefix='pred_tau_plus', 
                              q_prefix='analytical_q', 
                              out_prefix='pred_x'):
    """
    Vectorized Euclidean 4-vector projection:
        p_parallel = (p·q / q·q) * q
    Works directly on a pandas DataFrame (no event loop).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns {p_prefix}_{px,py,pz,E} and {q_prefix}_{px,py,pz,E}.
    p_prefix : str
        Prefix for the vector being projected.
    q_prefix : str
        Prefix for the reference vector.
    out_prefix : str
        Prefix for output projected components.

    Returns
    -------
    df : pandas.DataFrame
        Copy of df with new columns {out_prefix}_{px,py,pz,E}.
    """
    # extract components
    px, py, pz, pE = (df[f"{p_prefix}_{c}"] for c in ["px", "py", "pz", "E"])
    qx, qy, qz, qE = (df[f"{q_prefix}_{c}"] for c in ["px", "py", "pz", "E"])

    # compute projection scalar α (vectorized)
    num = px*qx + py*qy + pz*qz + pE*qE
    den = qx**2 + qy**2 + qz**2 + qE**2
    alpha = num / den.replace(0, np.nan)  # avoid div-by-zero

    # projected components
    df[f"{out_prefix}_px"] = alpha * qx
    df[f"{out_prefix}_py"] = alpha * qy
    df[f"{out_prefix}_pz"] = alpha * qz
    df[f"{out_prefix}_E"]  = alpha * qE

    return df

def compare_true_pred_kinematics(
    df,
    particle_prefix,
    bins=60,
    output_dir="nn_plots",
    frac=0.99,
):
    """
    Compare true vs predicted kinematic variables for a given particle type,
    and save the plots as PDF files (no titles, no dashed lines).
    Adds mean and RMS annotations to each plot.
    Shape plots use full range; residuals and ratios use quantile trimming,
    with binning defined over the trimmed range for consistent resolution.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'true_' and 'pred_' prefixed columns.
    particle_prefix : str
        Base name of particle (e.g., 'nu', 'tau_plus', 'tau_minus').
    bins : int
        Number of histogram bins.
    output_dir : str
        Directory to save plots.
    frac : float
        Fraction of events to keep for axis limits on residuals/ratios.
        (e.g. 0.99 keeps central 99%)
    """
    os.makedirs(output_dir, exist_ok=True)
    components = ["px", "py", "pz", "E"]

    for comp in components:
        true_col = f"true_{particle_prefix}_{comp}"
        pred_col = f"pred_{particle_prefix}_{comp}"

        if true_col not in df.columns or pred_col not in df.columns:
            print(f"Skipping {particle_prefix} {comp}: columns not found.")
            continue

        true_vals = df[true_col].to_numpy()
        pred_vals = df[pred_col].to_numpy()

        # --- 1. Shape comparison
        plt.figure(figsize=(5.5, 3.8))
        plt.hist(true_vals, bins=bins, alpha=0.6, label="True", density=True, histtype="stepfilled")
        plt.hist(pred_vals, bins=bins, alpha=0.6, label="Pred", density=True, histtype="step")

        # Add mean & RMS annotation
        txt = (
            f"True: μ={np.mean(true_vals):.2f}, σ={np.std(true_vals):.2f}\n"
            f"Pred: μ={np.mean(pred_vals):.2f}, σ={np.std(pred_vals):.2f}"
        )
        plt.text(
            0.97, 0.95, txt, transform=plt.gca().transAxes,
            fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none")
        )

        plt.xlabel(f"{comp} [GeV]")
        plt.ylabel("Normalized counts")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{particle_prefix}_{comp}_shape.pdf"))
        plt.close()

        # --- 2. Residuals (Δ = pred - true)
        delta = pred_vals - true_vals
        q_low, q_high = np.quantile(delta, [(1 - frac)/2, 1 - (1 - frac)/2])
        hist_range = (q_low, q_high)

        plt.figure(figsize=(5.5, 3.8))
        plt.hist(delta, bins=bins, range=hist_range, density=True, histtype="stepfilled", alpha=0.7)

        txt = f"μ={np.mean(delta):.3f}, σ={np.std(delta):.3f}"
        plt.text(
            0.97, 0.95, txt, transform=plt.gca().transAxes,
            fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none")
        )

        plt.xlabel(f"Δ{comp} [GeV]")
        plt.ylabel("Normalized counts")
        plt.xlim(hist_range)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{particle_prefix}_{comp}_residual.pdf"))
        plt.close()

        # --- 3. Energy ratio (E only)
        if comp == "E":
            ratio = pred_vals / np.where(true_vals != 0, true_vals, np.nan)
            ratio = ratio[np.isfinite(ratio)]
            q_low, q_high = np.quantile(ratio, [(1 - frac)/2, 1 - (1 - frac)/2])
            hist_range = (q_low, q_high)

            plt.figure(figsize=(5.5, 3.8))
            plt.hist(ratio, bins=bins, range=hist_range, density=True, histtype="stepfilled", alpha=0.7)

            txt = f"μ={np.mean(ratio):.3f}, σ={np.std(ratio):.3f}"
            plt.text(
                0.97, 0.95, txt, transform=plt.gca().transAxes,
                fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none")
            )

            plt.xlabel("E_pred / E_true")
            plt.ylabel("Normalized counts")
            plt.xlim(hist_range)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{particle_prefix}_E_ratio.pdf"))
            plt.close()

        print(f"Saved PDF plots with stats for {particle_prefix} {comp}")

def compare_analytical_pred_x(
    df,
    bins=60,
    output_dir="nn_plots",
    frac=0.99,
):
    """
    Compare analytical vs predicted kinematic components (px, py, pz)
    for the intermediate particle 'x' and save plots as PDFs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'analytical_x_' and 'pred_x_' columns.
    bins : int
        Number of histogram bins.
    output_dir : str
        Directory to save plots.
    frac : float
        Fraction of events to keep for x-axis limits (e.g. 0.99 keeps central 99%).
    """
    os.makedirs(output_dir, exist_ok=True)

    components = ["px", "py", "pz"]

    for comp in components:
        analytical_col = f"analytical_x_{comp}"
        pred_col = f"pred_x_{comp}"

        if analytical_col not in df.columns or pred_col not in df.columns:
            print(f"Skipping {comp}: missing columns.")
            continue

        analytical_vals = df[analytical_col].to_numpy()
        pred_vals = df[pred_col].to_numpy()

        # Determine common trimmed range for both plots
        combined_vals = np.concatenate([analytical_vals, pred_vals])
        q_low, q_high = np.quantile(combined_vals, [(1 - frac)/2, 1 - (1 - frac)/2])
        hist_range = (q_low, q_high)

        # --- 1. Shape comparison (trimmed range)
        plt.figure(figsize=(5.5, 3.8))
        plt.hist(analytical_vals, bins=bins, range=hist_range, alpha=0.5,
                 label="Analytical", density=True, histtype="stepfilled")
        plt.hist(pred_vals, bins=bins, range=hist_range, alpha=0.6,
                 label="Predicted", density=True, histtype="step")
        plt.xlabel(f"{comp} [GeV]")
        plt.ylabel("Normalized counts")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"x_{comp}_shape.pdf"))
        plt.close()

        # --- 2. Residuals (Δ = pred - analytical)
        delta = pred_vals - analytical_vals
        q_low_d, q_high_d = np.quantile(delta, [(1 - frac)/2, 1 - (1 - frac)/2])
        hist_range_d = (q_low_d, q_high_d)

        plt.figure(figsize=(5.5, 3.8))
        plt.hist(delta, bins=bins, range=hist_range_d, density=True,
                 histtype="stepfilled", alpha=0.7)
        plt.xlabel(f"Δ{comp} [GeV]")
        plt.ylabel("Normalized counts")
        plt.xlim(hist_range_d)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"x_{comp}_residual.pdf"))
        plt.close()

        print(f"Saved x_{comp} plots to {output_dir}")

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--stages', '-s', help='a list of stages to run', type=int, nargs="+")
    argparser.add_argument('--model_name', '-m', help='the name of the model output name', type=str, default='dummy_ditau_nu_regression_model')
    argparser.add_argument('--n_epochs', help='number of training epochs', type=int, default=10)
    args = argparser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    df = pd.read_pickle("dummy_z_ditau_events.pkl")
    
    input_features = [ 'pi_minus_E', 'pi_minus_px', 'pi_minus_py', 'pi_minus_pz',
                       'pi_plus_E',  'pi_plus_px',  'pi_plus_py',  'pi_plus_pz' ]
    
    output_features = [ 'nu_E', 'nu_px', 'nu_py', 'nu_pz',
                        'nubar_E', 'nubar_px', 'nubar_py', 'nubar_pz' ]
    
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
    
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    num_epochs = args.n_epochs
    # hyperparameters
    hp = {
        'batch_size': 1024, # 1024
        'lr': 0.0001, # 0.001
        'weight_decay': 0.001, # 0.001
        'n_hidden_layers': 4, # 4
        'n_nodes': 100, # 100
        'dropout': 0.0, # 0.0
    }

    def setup_model_and_training(hp):

        train_dataloader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False)
        model = SimpleNN(len(input_features), len(output_features), n_hidden_layers=hp['n_hidden_layers'], n_nodes=hp['n_nodes'], dropout=hp['dropout'])    
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
        
        model.to(device)

        return model, criterion, optimizer, device, train_dataloader, test_dataloader

    model, criterion, optimizer, device, train_dataloader, test_dataloader = setup_model_and_training(hp)
    
    
    loss_values = []
    val_loss_values = []
    running_loss_values = []
    early_stopper = EarlyStopper(patience=10, min_delta=0.)
    
    train = True
    add_analytical_solutions = False
    test = True


    if train:
    
        print("Starting training...")

        best_val_loss, _ = train_model(
            model,
            train_dataloader,
            test_dataloader,
            criterion,
            optimizer,
            n_epochs=num_epochs,
            device=device,
            verbose=True,
            recompute_train_loss=True,
            early_stopper=early_stopper
        )
    
#        for epoch in range(num_epochs):
#            #model.train()
#            running_loss= 0.0
#            for i, (X, y) in enumerate(train_dataloader):
#                # move data to GPU
#                X = X.to(device)
#                y = y.to(device)
#                optimizer.zero_grad()
#                outputs = model(X)
#                loss = criterion(outputs, y)
#                loss.backward()
#                optimizer.step()
#    
#                running_loss += loss.item()
#    
#            running_loss /= len(train_dataloader)
#            running_loss_values.append(running_loss)
#    
#            # get the validation loss
#            #model.eval()
#            model.to(device)
#            with torch.no_grad():
#                val_loss = 0.0
#                train_loss = 0.0
#                for i, (X, y) in enumerate(train_dataloader):
#                    X = X.to(device)
#                    y = y.to(device)
#                    outputs = model(X)
#                    loss = criterion(outputs, y)
#                    train_loss += loss.item()
#                train_loss /= len(train_dataloader)
#                loss_values.append(train_loss)
#                val_loss = 0.0
#                for i, (X, y) in enumerate(test_dataloader):
#                    X = X.to(device)
#                    y = y.to(device)
#                    outputs = model(X)
#                    loss = criterion(outputs, y)
#                    val_loss += loss.item()
#                val_loss /= len(test_dataloader)
#                val_loss_values.append(val_loss)
#    
#                if early_stopper.early_stop(val_loss):
#                    print(f"Early stopping triggered for epoch {epoch+1}")
#                    break
#    
#            print(f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, running_loss: {running_loss:.6f}')
#    
#            if epoch > 1: plot_loss(loss_values, val_loss_values, running_loss_values)
    
        model_name = args.model_name
    
        torch.save(model.state_dict(), f'{model_name}.pth')
    
    
    if add_analytical_solutions:

        print("Adding analytical solutions to test dataset...")
    
        import ROOT
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
        from ReconstructTaus import ReconstructTauAnalytically, solve_abcd_values, compute_q
    
        #print length of test dataset
        print(f"Test dataset size: {len(test_dataset)}")
    
        # loop over dataframe and add analytical solutions for each event
        
        P_Z = ROOT.TLorentzVector(0,0,0,91.0) # Z boson at rest
        for index, row in test_df.iterrows():
    
            P_taun_pi = ROOT.TLorentzVector(row['pi_minus_px'], row['pi_minus_py'], row['pi_minus_pz'], row['pi_minus_E'])
            P_taup_pi  = ROOT.TLorentzVector(row['pi_plus_px'],  row['pi_plus_py'],  row['pi_plus_pz'],  row['pi_plus_E'])
    
            P_taun_nu = ROOT.TLorentzVector(row['nu_px'], row['nu_py'], row['nu_pz'], row['nu_E'])
            P_taup_nubar = ROOT.TLorentzVector(row['nubar_px'], row['nubar_py'], row['nubar_pz'], row['nubar_E'])
    
            P_taup = P_taup_pi + P_taup_nubar
            P_taun = P_taun_pi + P_taun_nu
    
            # get the vector 'x' that is orthogonal to both pi directions and the Z direction, this is the component that the analytical method doersn't know the sign of
            abcd_taup = solve_abcd_values(P_taup, P_taun, P_Z, P_taup_pi, P_taun_pi)
            d = abcd_taup[3]
            q = compute_q(P_Z*P_Z,P_Z, P_taup_pi, P_taun_pi)
            x = d*q
    
            # store x on the dataframe
            test_df.at[index, 'analytical_x_E'] = x.E()
            test_df.at[index, 'analytical_x_px'] = x.Px()
            test_df.at[index, 'analytical_x_py'] = x.Py()
            test_df.at[index, 'analytical_x_pz'] = x.Pz()
    
            # store q as well
            test_df.at[index, 'analytical_q_E'] = q.E()
            test_df.at[index, 'analytical_q_px'] = q.Px()
            test_df.at[index, 'analytical_q_py'] = q.Py()
            test_df.at[index, 'analytical_q_pz'] = q.Pz()
    
            solutions = ReconstructTauAnalytically(P_Z, P_taup_pi, P_taun_pi, P_taup_pi, P_taun_pi, return_values=True)
    
            for i, solution in enumerate(solutions):
                an_sol_taup = solution[0]
                an_sol_taun = solution[1]
    
                an_sol_nu = an_sol_taup - P_taup_pi
                an_sol_nubar = an_sol_taun - P_taun_pi
    
                # store analytical solutions on the dataframe
                test_df.at[index, f'analytical_sol_{i}_nu_E'] = an_sol_nu.E()
                test_df.at[index, f'analytical_sol_{i}_nu_px'] = an_sol_nu.Px()
                test_df.at[index, f'analytical_sol_{i}_nu_py'] = an_sol_nu.Py()
                test_df.at[index, f'analytical_sol_{i}_nu_pz'] = an_sol_nu.Pz()
                test_df.at[index, f'analytical_sol_{i}_nubar_E'] = an_sol_nubar.E()
                test_df.at[index, f'analytical_sol_{i}_nubar_px'] = an_sol_nubar.Px()
                test_df.at[index, f'analytical_sol_{i}_nubar_py'] = an_sol_nubar.Py()
                test_df.at[index, f'analytical_sol_{i}_nubar_pz'] = an_sol_nubar.Pz()
                test_df.at[index, f'analytical_sol_{i}_tau_plus_E'] = an_sol_taup.E()
                test_df.at[index, f'analytical_sol_{i}_tau_plus_px'] = an_sol_taup.Px()
                test_df.at[index, f'analytical_sol_{i}_tau_plus_py'] = an_sol_taup.Py()
                test_df.at[index, f'analytical_sol_{i}_tau_plus_pz'] = an_sol_taup.Pz()
                test_df.at[index, f'analytical_sol_{i}_tau_minus_E'] = an_sol_taun.E()
                test_df.at[index, f'analytical_sol_{i}_tau_minus_px'] = an_sol_taun.Px()
                test_df.at[index, f'analytical_sol_{i}_tau_minus_py'] = an_sol_taun.Py()
                test_df.at[index, f'analytical_sol_{i}_tau_minus_pz'] = an_sol_taun.Pz()
    
        # save test_df with analytical solutions added
        test_df.to_pickle("dummy_ditau_events_test_df_with_analytical_solutions.pkl")
    
    if test:
    
        import matplotlib.pyplot as plt
        import uproot3
    
        # load test dataframe with analytical solutions added
        test_df = pd.read_pickle("dummy_ditau_events_test_df_with_analytical_solutions.pkl")
    
        print("Starting testing...")
    
        model_name = args.model_name
        model_path = f'{model_name}.pth'
    
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print(f"Loading model from {model_path} failed. Trying to load from CPU.")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    
        X_test, _ = test_dataset[:]
        with torch.no_grad():
            predictions_norm = model(X_test)
    
        # destandardize predictions so that they are in physical units
        predictions = test_dataset.destandardize_outputs(predictions_norm).numpy()
    
        true_values = test_df[output_features].values
    
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
    
        # add analytical results from test_df to results_df, loop obver E, px, py, pz, loop over particle types, loop over sol 0 and 1
        for sol in [0,1]:
            for particle in ['nu', 'nubar', 'tau_plus', 'tau_minus']:
                for comp in ['E', 'px', 'py', 'pz']:
                    results_df[f'analytical_sol_{sol}_{particle}_{comp}'] = test_df[f'analytical_sol_{sol}_{particle}_{comp}'].to_numpy()
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
        results_df.to_pickle("dummy_ditau_nu_regression_results.pkl")
    
        # write root file aswell
        output_root_file = "dummy_ditau_nu_regression_results.root"
    
        with uproot3.recreate(output_root_file) as f:
            # Create the tree inside the file (name it "tree")
            f["tree"] = uproot3.newtree({col: np.float32 for col in results_df.columns})
        
            # Fill the tree
            f["tree"].extend(results_df.to_dict(orient='list'))
    
    
        # make plots
    
        compare_true_pred_kinematics(results_df, "nu")
        compare_true_pred_kinematics(results_df, "nubar")
        compare_true_pred_kinematics(results_df, "tau_minus")
        compare_true_pred_kinematics(results_df, "tau_plus")
    
        compare_true_pred_kinematics(results_df, "nu_no_x")
        compare_true_pred_kinematics(results_df, "nubar_no_x")
        compare_true_pred_kinematics(results_df, "tau_minus_no_x")
        compare_true_pred_kinematics(results_df, "tau_plus_no_x")
    
        compare_analytical_pred_x(results_df)
    