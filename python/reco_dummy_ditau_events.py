import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt
import os

class RegressionDataset(Dataset):
    def __init__(self, dataframe, input_features, output_features):
        # Convert entire dataset to tensors at initialization (avoids slow indexing)
        self.X = torch.tensor(dataframe[input_features].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[output_features].values, dtype=torch.float32)

    def get_input_means_stds(self):
        """
        Compute the means and standard deviations of the input features.
        """
        means = self.X.mean(dim=0)
        stds = self.X.std(dim=0)
        return means, stds

    def get_output_means_stds(self):
        """
        Compute the means and standard deviations of the output features.
        """
        means = self.y.mean(dim=0)
        stds = self.y.std(dim=0)
        return means, stds

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
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

    def __init__(self, input_dim, output_dim, n_hidden_layers, n_nodes):
        super(SimpleNN, self).__init__()

        layers = OrderedDict()
        layers['input'] = nn.Linear(input_dim, n_nodes)
        layers['bn_input'] = nn.BatchNorm1d(n_nodes)
        layers['relu_input'] = nn.ReLU()
        for i in range(n_hidden_layers):
            layers[f'hidden_{i+1}'] = nn.Linear(n_nodes, n_nodes)
            layers[f'hidden_bn_{i+1}'] = nn.BatchNorm1d(n_nodes)
            layers[f'hidden_relu_{i+1}'] = nn.ReLU()
        layers['output'] = nn.Linear(n_nodes, output_dim)

        self.layers = nn.Sequential(layers)

        #self.apply(initialize_weights)

        # print a summry of the model
        print('Model summary:')
        print(self.layers)
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Number of parameters: {n_params}')
 

    def forward(self, x):
        x = self.layers(x)
        return x

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

def RotateFrame(rot_vec, vec):
    '''
    Rotate the coordinate axis so that the z-axis is aligned with the direction of the rot_vec.
    

    Args:
        rot_vec (TVector3): The vector to align the z-axis with.
        vec (TVector3): The vector to rotate.

    Returns:
        TVector3: The rotated vector.
    '''
    vec_new = vec.Clone()
    
    # Define the rotation angles to allign with rot_vec direction
    theta = rot_vec.Theta()
    phi = rot_vec.Phi()
    vec_new.RotateZ(-phi)
    vec_new.RotateY(-theta)

    return vec_new

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

df = pd.read_pickle("dummy_z_ditau_events.pkl")
print(df.head())

input_features = [ 'pi_minus_E', 'pi_minus_px', 'pi_minus_py', 'pi_minus_pz',
                   'pi_plus_E',  'pi_plus_px',  'pi_plus_py',  'pi_plus_pz' ]

output_features = [ 'nu_E', 'nu_px', 'nu_py', 'nu_pz',
                    'nubar_E', 'nubar_px', 'nubar_py', 'nubar_pz' ]

train_size = int(0.8 * len(df))
test_size = len(df) - train_size

train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

train_dataset = RegressionDataset(train_df, input_features, output_features)
test_dataset = RegressionDataset(test_df, input_features, output_features)

# compute means and stds for input and output features for scaling
train_in_means, train_in_stds = train_dataset.get_input_means_stds()
train_out_means, train_out_stds = train_dataset.get_output_means_stds()
test_in_means, test_in_stds = test_dataset.get_input_means_stds()
test_out_means, test_out_stds = test_dataset.get_output_means_stds()

batch_size = 1024
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


print(f"Training dataset size: {len(train_dataset)}")
print(f"Testing dataset size: {len(test_dataset)}")

model = SimpleNN(len(input_features), len(output_features), n_hidden_layers=4, n_nodes=100) # 2 hidden before
criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.001) #weight_decay=0.0001
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

def plot_loss(loss_values, val_loss_values, running_loss_values=None):
    plt.figure()
    plt.plot(range(2, len(loss_values)+1), loss_values[1:], label='train loss')
    plt.plot(range(2, len(val_loss_values)+1), val_loss_values[1:], label='validation loss')
    if running_loss_values is not None: plt.plot(range(2, len(running_loss_values)+1), running_loss_values[1:], label='running loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'nn_plots/loss_vs_epoch_dummy.pdf')
    plt.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 50

loss_values = []
val_loss_values = []
running_loss_values = []
early_stopper = EarlyStopper(patience=5, min_delta=0.)

train = False
add_analytical_solutions = True
test = True
plot = False

if train:

    print("Starting training...")

    for epoch in range(num_epochs):
        #model.train()
        running_loss= 0.0
        for i, (X, y) in enumerate(train_dataloader):
            # move data to GPU
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(train_dataloader)
        running_loss_values.append(running_loss)

        # get the validation loss
        #model.eval()
        model.to(device)
        with torch.no_grad():
            val_loss = 0.0
            train_loss = 0.0
            for i, (X, y) in enumerate(train_dataloader):
                X = X.to(device)
                y = y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                train_loss += loss.item()
            train_loss /= len(train_dataloader)
            loss_values.append(train_loss)
            val_loss = 0.0
            for i, (X, y) in enumerate(test_dataloader):
                X = X.to(device)
                y = y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
            val_loss /= len(test_dataloader)
            val_loss_values.append(val_loss)

            if early_stopper.early_stop(val_loss):
                print(f"Early stopping triggered for epoch {epoch+1}")
                break

        print(f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, running_loss: {running_loss:.6f}')

        if epoch > 1: plot_loss(loss_values, val_loss_values, running_loss_values)

    model_name = "dummy_ditau_nu_regression_model"

    torch.save(model.state_dict(), f'{model_name}.pth')


if add_analytical_solutions:

    import ROOT
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
    from ReconstructTaus import ReconstructTauAnalytically, solve_abcd_values, compute_q

    test_df = test_df.iloc[:1].copy() #TODO delete after!

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
    test_df.to_pickle("dummy_ditau_events_test_df_with_analytical_solutions_test.pkl")

if test:

    # load test dataframe with analytical solutions added
    test_df = pd.read_pickle("dummy_ditau_events_test_df_with_analytical_solutions_test.pkl")

    # load only first 1 events
    test_df = test_df.iloc[:1].copy()

    import uproot3

    print("Starting testing...")

    model_name = "dummy_ditau_nu_regression_model"
    model_path = f'{model_name}.pth'

    try:
        model.load_state_dict(torch.load(model_path))
    except:
        print(f"Loading model from {model_path} failed. Trying to load from CPU.")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, (X, y) in enumerate(test_dataloader):
            outputs = model(X)

            predictions.append(outputs.numpy())


    predictions = np.concatenate(predictions, axis=0)

    #take only first events to match test_df length
    predictions = predictions[:len(test_df)]

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

    # temp cross check

    import ROOT
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
    from ReconstructTaus import ReconstructTauAnalytically, solve_abcd_values, compute_q, project_4vec_euclidean

    P_Z = ROOT.TLorentzVector(0,0,0,91.0) # Z boson at rest
    for index, row in results_df.iterrows():

        P_taun_pi = ROOT.TLorentzVector(row['pi_minus_px'], row['pi_minus_py'], row['pi_minus_pz'], row['pi_minus_E'])
        P_taup_pi  = ROOT.TLorentzVector(row['pi_plus_px'],  row['pi_plus_py'],  row['pi_plus_pz'],  row['pi_plus_E'])

        P_pred_taup = ROOT.TLorentzVector(row['pred_tau_plus_px'],  row['pred_tau_plus_py'],  row['pred_tau_plus_pz'],  row['pred_tau_plus_E'])
        P_pred_taun = ROOT.TLorentzVector(row['pred_tau_minus_px'], row['pred_tau_minus_py'], row['pred_tau_minus_pz'], row['pred_tau_minus_E'])

        #P_pred_taup = ROOT.TLorentzVector(row['true_tau_plus_px'],  row['true_tau_plus_py'],  row['true_tau_plus_pz'],  row['true_tau_plus_E'])
        #P_pred_taun = ROOT.TLorentzVector(row['true_tau_minus_px'], row['true_tau_minus_py'], row['true_tau_minus_pz'], row['true_tau_minus_E'])

        abcd_taup_pred = solve_abcd_values(P_pred_taup, P_pred_taun, P_Z, P_taup_pi, P_taun_pi)
        d_pred = abcd_taup_pred[3]
        q = compute_q(P_Z*P_Z,P_Z, P_taup_pi, P_taun_pi)
        x_pred = d_pred*q

        x_pred_alt2 = project_4vec_euclidean(P_pred_taup, q)

        q_unit = q.Vect().Unit()
        #alt3 is dot propduct of 1_unit and taup along q_unit
        dot_product = P_pred_taup.Vect().Dot(q_unit)
        x_pred_alt3 = ROOT.TVector3(q_unit)
        x_pred_alt3 *= dot_product


        results_df.at[index, 'alt_pred_x_E'] = x_pred.E()
        results_df.at[index, 'alt_pred_x_px'] = x_pred.Px()
        results_df.at[index, 'alt_pred_x_py'] = x_pred.Py()
        results_df.at[index, 'alt_pred_x_pz'] = x_pred.Pz()

        # add alt2 as well
        results_df.at[index, 'alt2_pred_x_E'] = x_pred_alt2.E()
        results_df.at[index, 'alt2_pred_x_px'] = x_pred_alt2.Px()
        results_df.at[index, 'alt2_pred_x_py'] = x_pred_alt2.Py()
        results_df.at[index, 'alt2_pred_x_pz'] = x_pred_alt2.Pz()

        results_df.at[index, 'alt3_pred_x_px'] = x_pred_alt3.Px()
        results_df.at[index, 'alt3_pred_x_py'] = x_pred_alt3.Py()
        results_df.at[index, 'alt3_pred_x_pz'] = x_pred_alt3.Pz()

        # store alternative q's as well
        results_df.at[index, 'alt_analytical_q_E'] = q.E()
        results_df.at[index, 'alt_analytical_q_px'] = q.Px()
        results_df.at[index, 'alt_analytical_q_py'] = q.Py()
        results_df.at[index, 'alt_analytical_q_pz'] = q.Pz()

    # print data frame rows comparing x_pred and alt_x_pred
    for index, row in results_df.iterrows():
        print(f"\nEvent {index}:")
        print(f"pred_x:     E={row['pred_x_E']}, px={row['pred_x_px']}, py={row['pred_x_py']}, pz={row['pred_x_pz']}")
        print(f"alt_pred_x: E={row['alt_pred_x_E']}, px={row['alt_pred_x_px']}, py={row['alt_pred_x_py']}, pz={row['alt_pred_x_pz']}")
        print(f"alt2_pred_x:E={row['alt2_pred_x_E']}, px={row['alt2_pred_x_px']}, py={row['alt2_pred_x_py']}, pz={row['alt2_pred_x_pz']}")
        print(f"alt3_pred_x:E=blah, px={row['alt3_pred_x_px']}, py={row['alt3_pred_x_py']}, pz={row['alt3_pred_x_pz']}")
        ## compare q's as well
        #print(f"analytical_q:     E={row['analytical_q_E']}, px={row['analytical_q_px']}, py={row['analytical_q_py']}, pz={row['analytical_q_pz']}")
        #print(f"alt_analytical_q: E={row['alt_analytical_q_E']}, px={row['alt_analytical_q_px']}, py={row['alt_analytical_q_py']}, pz={row['alt_analytical_q_pz']}")
        print("")



if test and False:

    # load test dataframe with analytical solutions added
    test_df = pd.read_pickle("dummy_ditau_events_test_df_with_analytical_solutions.pkl")

    import ROOT
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
    from ReconstructTaus import ReconstructTauAnalytically, solve_abcd_values, compute_q
    import uproot3

    print("Starting testing...")

    model_name = "dummy_ditau_nu_regression_model"
    model_path = f'{model_name}.pth'

    try:
        model.load_state_dict(torch.load(model_path))
    except:
        print(f"Loading model from {model_path} failed. Trying to load from CPU.")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    predictions = []
    with torch.no_grad():
        for i, (X, y) in enumerate(test_dataloader):
            outputs = model(X)

            predictions.append(outputs.numpy())

    predictions = np.concatenate(predictions, axis=0)

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

    # loop over dataframe and add analytical solutions for each event
    
    P_Z = ROOT.TLorentzVector(0,0,0,91.0) # Z boson at rest
    for index, row in results_df.iterrows():

        P_taun_pi = ROOT.TLorentzVector(row['pi_minus_px'], row['pi_minus_py'], row['pi_minus_pz'], row['pi_minus_E'])
        P_taup_pi  = ROOT.TLorentzVector(row['pi_plus_px'],  row['pi_plus_py'],  row['pi_plus_pz'],  row['pi_plus_E'])

        P_taup = ROOT.TLorentzVector(row['true_tau_plus_px'],  row['true_tau_plus_py'],  row['true_tau_plus_pz'],  row['true_tau_plus_E'])
        P_taun = ROOT.TLorentzVector(row['true_tau_minus_px'], row['true_tau_minus_py'], row['true_tau_minus_pz'], row['true_tau_minus_E'])

        # get the vector 'x' that is orthogonal to both pi directions and the Z direction, this is the component that the analytical method doersn't know the sign of
        abcd_taup = solve_abcd_values(P_taup, P_taun, P_Z, P_taup_pi, P_taun_pi)
        d = abcd_taup[3]
        q = compute_q(P_Z*P_Z,P_Z, P_taup_pi, P_taun_pi)
        x = d*q

        P_pred_taup = ROOT.TLorentzVector(row['pred_tau_plus_px'],  row['pred_tau_plus_py'],  row['pred_tau_plus_pz'],  row['pred_tau_plus_E'])
        P_pred_taun = ROOT.TLorentzVector(row['pred_tau_minus_px'], row['pred_tau_minus_py'], row['pred_tau_minus_pz'], row['pred_tau_minus_E'])

        abcd_taup_pred = solve_abcd_values(P_pred_taup, P_pred_taun, P_Z, P_taup_pi, P_taun_pi)
        d_pred = abcd_taup_pred[3]
        x_pred = d_pred*q

        # store x on the dataframe
        results_df.at[index, 'analytical_x_E'] = x.E()
        results_df.at[index, 'analytical_x_px'] = x.Px()
        results_df.at[index, 'analytical_x_py'] = x.Py()
        results_df.at[index, 'analytical_x_pz'] = x.Pz()

        results_df.at[index, 'pred_x_E'] = x_pred.E()
        results_df.at[index, 'pred_x_px'] = x_pred.Px()
        results_df.at[index, 'pred_x_py'] = x_pred.Py()
        results_df.at[index, 'pred_x_pz'] = x_pred.Pz()

        # store taus and neutrinos with the x component subtracted

        P_taup_no_x = P_taup - x
        P_taun_no_x = P_taun + x
        P_nubar_no_x = P_taup_no_x - P_taup_pi
        P_nu_no_x = P_taun_no_x - P_taun_pi

        results_df.at[index, 'true_nu_no_x_E'] = P_nu_no_x.E()
        results_df.at[index, 'true_nu_no_x_px'] = P_nu_no_x.Px()
        results_df.at[index, 'true_nu_no_x_py'] = P_nu_no_x.Py()
        results_df.at[index, 'true_nu_no_x_pz'] = P_nu_no_x.Pz()
        results_df.at[index, 'true_nubar_no_x_E'] = P_nubar_no_x.E()
        results_df.at[index, 'true_nubar_no_x_px'] = P_nubar_no_x.Px()
        results_df.at[index, 'true_nubar_no_x_py'] = P_nubar_no_x.Py()
        results_df.at[index, 'true_nubar_no_x_pz'] = P_nubar_no_x.Pz()

        results_df.at[index, 'true_tau_plus_no_x_E'] = P_taup_no_x.E()
        results_df.at[index, 'true_tau_plus_no_x_px'] = P_taup_no_x.Px()
        results_df.at[index, 'true_tau_plus_no_x_py'] = P_taup_no_x.Py()
        results_df.at[index, 'true_tau_plus_no_x_pz'] = P_taup_no_x.Pz()
        results_df.at[index, 'true_tau_minus_no_x_E'] = P_taun_no_x.E()
        results_df.at[index, 'true_tau_minus_no_x_px'] = P_taun_no_x.Px()
        results_df.at[index, 'true_tau_minus_no_x_py'] = P_taun_no_x.Py()
        results_df.at[index, 'true_tau_minus_no_x_pz'] = P_taun_no_x.Pz()


        solutions = ReconstructTauAnalytically(P_Z, P_taup_pi, P_taun_pi, P_taup_pi, P_taun_pi, return_values=True)

        for i, solution in enumerate(solutions):
            an_sol_taup = solution[0]
            an_sol_taun = solution[1]

            an_sol_nu = an_sol_taup - P_taup_pi
            an_sol_nubar = an_sol_taun - P_taun_pi

            # store analytical solutions on the dataframe
            results_df.at[index, f'analytical_sol_{i}_nu_E'] = an_sol_nu.E()
            results_df.at[index, f'analytical_sol_{i}_nu_px'] = an_sol_nu.Px()
            results_df.at[index, f'analytical_sol_{i}_nu_py'] = an_sol_nu.Py()
            results_df.at[index, f'analytical_sol_{i}_nu_pz'] = an_sol_nu.Pz()
            results_df.at[index, f'analytical_sol_{i}_nubar_E'] = an_sol_nubar.E()
            results_df.at[index, f'analytical_sol_{i}_nubar_px'] = an_sol_nubar.Px()
            results_df.at[index, f'analytical_sol_{i}_nubar_py'] = an_sol_nubar.Py()
            results_df.at[index, f'analytical_sol_{i}_nubar_pz'] = an_sol_nubar.Pz()
            results_df.at[index, f'analytical_sol_{i}_tau_plus_E'] = an_sol_taup.E()
            results_df.at[index, f'analytical_sol_{i}_tau_plus_px'] = an_sol_taup.Px()
            results_df.at[index, f'analytical_sol_{i}_tau_plus_py'] = an_sol_taup.Py()
            results_df.at[index, f'analytical_sol_{i}_tau_plus_pz'] = an_sol_taup.Pz()
            results_df.at[index, f'analytical_sol_{i}_tau_minus_E'] = an_sol_taun.E()
            results_df.at[index, f'analytical_sol_{i}_tau_minus_px'] = an_sol_taun.Px()
            results_df.at[index, f'analytical_sol_{i}_tau_minus_py'] = an_sol_taun.Py()
            results_df.at[index, f'analytical_sol_{i}_tau_minus_pz'] = an_sol_taun.Pz()

    # add pred tau and nu with no x (subtract it from the predicted taus and nus)
    results_df['pred_nu_no_x_E'] = results_df['pred_nu_E'] - results_df['pred_x_E']
    results_df['pred_nu_no_x_px'] = results_df['pred_nu_px'] - results_df['pred_x_px']
    results_df['pred_nu_no_x_py'] = results_df['pred_nu_py'] - results_df['pred_x_py']
    results_df['pred_nu_no_x_pz'] = results_df['pred_nu_pz'] - results_df['pred_x_pz']
    results_df['pred_nubar_no_x_E'] = results_df['pred_nubar_E'] + results_df['pred_x_E']
    results_df['pred_nubar_no_x_px'] = results_df['pred_nubar_px'] + results_df['pred_x_px']
    results_df['pred_nubar_no_x_py'] = results_df['pred_nubar_py'] + results_df['pred_x_py']
    results_df['pred_nubar_no_x_pz'] = results_df['pred_nubar_pz'] + results_df['pred_x_pz']
    results_df['pred_tau_plus_no_x_E'] = results_df['pred_tau_plus_E'] - results_df['pred_x_E']
    results_df['pred_tau_plus_no_x_px'] = results_df['pred_tau_plus_px'] - results_df['pred_x_px']
    results_df['pred_tau_plus_no_x_py'] = results_df['pred_tau_plus_py'] - results_df['pred_x_py']
    results_df['pred_tau_plus_no_x_pz'] = results_df['pred_tau_plus_pz'] - results_df['pred_x_pz']
    results_df['pred_tau_minus_no_x_E'] = results_df['pred_tau_minus_E'] + results_df['pred_x_E']
    results_df['pred_tau_minus_no_x_px'] = results_df['pred_tau_minus_px'] + results_df['pred_x_px']
    results_df['pred_tau_minus_no_x_py'] = results_df['pred_tau_minus_py'] + results_df['pred_x_py']
    results_df['pred_tau_minus_no_x_pz'] = results_df['pred_tau_minus_pz'] + results_df['pred_x_pz']

    print("First 5 true values:")
    print(true_values[:5])
    print("First 5 predicted values:")
    print(predictions[:5])
    print("First 5 true taus:")
    print(true_taus[:5])
    print("First 5 predicted taus:")
    print(pred_taus[:5])

    # print out the first 5 rows of the dataframe
    print("First 5 rows of the results dataframe:")
    print(results_df.head())

    #print all the names of the columns in the dataframe
    print("Columns in the results dataframe:")
    print(results_df.columns.tolist())


    # like the above but print events in sequence

    for j in range(5):
        print(f"Event {j}:")
        for i, col in enumerate(output_features):
            print(f" {col} {results_df.iloc[j]['true_'+col]:.3f} vs {results_df.iloc[j]['pred_'+col]:.3f}")
        # print x values
        print(f" analytical x: E {results_df.iloc[j]['analytical_x_E']:.3f} px {results_df.iloc[j]['analytical_x_px']:.3f} py {results_df.iloc[j]['analytical_x_py']:.3f} pz {results_df.iloc[j]['analytical_x_pz']:.3f}")
        print(f" pred x: E {x_pred.E():.3f} px {x_pred.Px():.3f} py {x_pred.Py():.3f} pz {x_pred.Pz():.3f}")
        # print analytical neutrino solutions
        for i in range(len(solutions)):
            print(f" analytical solution {i} nu: E {results_df.iloc[j][f'analytical_sol_{i}_nu_E']:.3f} px {results_df.iloc[j][f'analytical_sol_{i}_nu_px']:.3f} py {results_df.iloc[j][f'analytical_sol_{i}_nu_py']:.3f} pz {results_df.iloc[j][f'analytical_sol_{i}_nu_pz']:.3f}")
            print(f" analytical solution {i} nubar: E {results_df.iloc[j][f'analytical_sol_{i}_nubar_E']:.3f} px {results_df.iloc[j][f'analytical_sol_{i}_nubar_px']:.3f} py {results_df.iloc[j][f'analytical_sol_{i}_nubar_py']:.3f} pz {results_df.iloc[j][f'analytical_sol_{i}_nubar_pz']:.3f}")
        print("\n")

    # write the results dataframe to a pickle file
    results_df.to_pickle("dummy_ditau_nu_regression_results.pkl")

    output_root_file = "dummy_ditau_nu_regression_results.root"

    with uproot3.recreate(output_root_file) as f:
        # Create the tree inside the file (name it "tree")
        f["tree"] = uproot3.newtree({col: np.float32 for col in results_df.columns})
    
        # Fill the tree
        f["tree"].extend(results_df.to_dict(orient='list'))

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

if plot:

    import matplotlib.pyplot as plt

    results_df = pd.read_pickle("dummy_ditau_nu_regression_results.pkl")

    compare_true_pred_kinematics(results_df, "nu")
    compare_true_pred_kinematics(results_df, "nubar")
    compare_true_pred_kinematics(results_df, "tau_minus")
    compare_true_pred_kinematics(results_df, "tau_plus")

    compare_true_pred_kinematics(results_df, "nu_no_x")
    compare_true_pred_kinematics(results_df, "nubar_no_x")
    compare_true_pred_kinematics(results_df, "tau_minus_no_x")
    compare_true_pred_kinematics(results_df, "tau_plus_no_x")

    compare_analytical_pred_x(results_df)
