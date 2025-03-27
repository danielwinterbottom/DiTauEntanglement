import pandas as pd
import uproot
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
import ROOT
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
from PolarimetricA1 import PolarimetricA1
import matplotlib.pyplot as plt


def read_root_in_chunks(filename, treename, output_name='df.pkl', nchunks=10, variables=None, verbosity=0):

    """
    Read a ROOT file into dataframe in chunks and save each chunk as a pickle file.
    Parameters:
    - filename (str): The name of the ROOT file to read.
    - treename (str): The name of the TTree to read.
    - output_name (str): The name of the output pickle file.
    - nchunks (int): The number of chunks to divide the data into.
    - verbosity (int): The level of verbosity for printing information.
    """

    # Check if the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    if filename.endswith('.pkl'):
        raise ValueError('Filename must end with .pkl')

    if verbosity > 0: print('Preparing dataframes for sample: %s using %i chunks' %(filename,nchunks))
    # Open the root file
    tree = uproot.open(filename)[treename]

    # Get the total number of entries
    num_entries = tree.num_entries

    chunksize = int(num_entries/nchunks)

    if verbosity > 0: print('Total events in sample = %(num_entries)i, number of events per chunk = %(chunksize)g' % vars())

    # Iterate over the chunks
    for i in range(nchunks):
        # Calculate start and stop indices for the current chunk
        start = i * chunksize
        stop = min((i + 1) * chunksize, num_entries)
        if verbosity > 1: print ('Processing chunk %(i)i, start and stop indices = %(start)s and %(stop)s' % vars())

        # Read the current chunk into a dataframe
        if variables is None: variables = tree.keys()
        # filter any variables that are not in the tree
        variables = [x for x in variables if x in tree.keys()]
        if verbosity > 1: print('Reading variables: %s' % variables)
        df = tree.arrays(variables, library="pd", entry_start=start, entry_stop=stop) 

        if verbosity > 1: 
            print('Number of events in chunk = %i' % len(df))
            print("First Entry in chunk:")
            print(df.head(1))
            print("Last Entry in chunk:")
            print(df.tail(1))

        # save dataframe then delete

        output = output_name.replace('.pkl', '_chunk_%i.pkl' % i)

        if verbosity > 1: print('Writing %(output)s\n' % vars())
        
        df.to_pickle(output)
        del df

def PrepareDataframes(filename1, filename2, treename1, treename2, nchunks=10, verbosity=0):

    """
    Prepare dataframes from ROOT files containing different friend trees.
    The Two trees contain different variables for the same events.
    """

    variables = [
        'taup_px','taup_py','taup_pz','taup_e',
        'taun_px','taun_py','taun_pz','taun_e',
        'taup_nu_px','taup_nu_py','taup_nu_pz','taup_nu_e',
        'taun_nu_px','taun_nu_py','taun_nu_pz','taun_nu_e',
        'taup_npi','taup_npizero',
        'taun_npi','taun_npizero',
        'reco_taup_pi1_px','reco_taup_pi1_py','reco_taup_pi1_pz','reco_taup_pi1_e',
        'reco_taup_pi2_px','reco_taup_pi2_py','reco_taup_pi2_pz','reco_taup_pi2_e',
        'reco_taup_pi3_px','reco_taup_pi3_py','reco_taup_pi3_pz','reco_taup_pi3_e',
        'reco_taup_pizero1_px','reco_taup_pizero1_py','reco_taup_pizero1_pz','reco_taup_pizero1_e',
        'reco_taup_pizero2_px','reco_taup_pizero2_py','reco_taup_pizero2_pz','reco_taup_pizero2_e',
        'reco_taun_pi1_px','reco_taun_pi1_py','reco_taun_pi1_pz','reco_taun_pi1_e',
        'reco_taun_pi2_px','reco_taun_pi2_py','reco_taun_pi2_pz','reco_taun_pi2_e',
        'reco_taun_pi3_px','reco_taun_pi3_py','reco_taun_pi3_pz','reco_taun_pi3_e',
        'reco_taun_pizero1_px','reco_taun_pizero1_py','reco_taun_pizero1_pz','reco_taun_pizero1_e',
        'reco_taun_pizero2_px','reco_taun_pizero2_py','reco_taun_pizero2_pz','reco_taun_pizero2_e',
        'reco_taup_vx','reco_taup_vy','reco_taup_vz',
        'reco_taun_vx','reco_taun_vy','reco_taun_vz',
        'reco_taup_pi1_ipx','reco_taup_pi1_ipy','reco_taup_pi1_ipz',
        'reco_taun_pi1_ipx','reco_taun_pi1_ipy','reco_taun_pi1_ipz',
        'reco_Z_px','reco_Z_py','reco_Z_pz','reco_Z_e',
        'taup_pi1_vz','taup_vz',
        'z_x','z_y','z_z',
        'cosn_plus',
        'cosr_plus',
        'cosk_plus',
        'cosn_minus',
        'cosr_minus',
        'cosk_minus',
        'cosTheta',
        'cosn_plus_reco',
        'cosr_plus_reco',
        'cosk_plus_reco',
        'cosn_minus_reco',
        'cosr_minus_reco',
        'cosk_minus_reco',
        'cosTheta_reco',    
    ]

    read_root_in_chunks(filename1, treename1, output_name='output/df.pkl', nchunks=nchunks, variables=variables, verbosity=verbosity)
    read_root_in_chunks(filename2, treename2, output_name='output/df_friend.pkl', nchunks=nchunks, variables=variables, verbosity=verbosity)


    # for each chunk concatenate along columns since they have the same event order and save the concatenated dataframe
    for i in range(nchunks):
        # load the two dataframes
        df = pd.read_pickle('output/df_chunk_%i.pkl' % i)
        df_friend = pd.read_pickle('output/df_friend_chunk_%i.pkl' % i)

        # concatenate along columns
        df = pd.concat([df, df_friend], axis=1)
        # remove any duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        # the variable in the first dataframe called "taup_pi1_vz" should match the variables from the second dataframe called "taup_vz"
        # drop any events where this is not the case, printing how many events are dropped and then drop these collumns from the dataframe
        if verbosity>0: print(f"Number of events before dropping: {len(df)}")
        df = df[df["taup_pi1_vz"] == df["taup_vz"]]
        if verbosity>0: print(f"Number of events after dropping unmatched events from the two trees: {len(df)}")
        # drops nans
        df = df.dropna()
        if verbosity>0: print(f"Number of events after dropping nans: {len(df)}")
        df = df.drop(columns=["taup_pi1_vz", "taup_vz"])
        # remove tau->pipi0pi0+mnu decays for now
        #require taup_npizero < 2
        df = df[df["taup_npizero"] < 2]
        #require taun_npizero < 2
        df = df[df["taun_npizero"] < 2]

        # add the visible tau 4-vectors by summing the pi and pizero 4-vectors
        for x in ['px', 'py', 'pz', 'e']:
            df['reco_taup_vis_' + x] = df['reco_taup_pi1_' + x] + df['reco_taup_pi2_' + x] + df['reco_taup_pi3_' + x] + df['reco_taup_pizero1_' + x] + df['reco_taup_pizero2_' + x]
            df['reco_taun_vis_' + x] = df['reco_taun_pi1_' + x] + df['reco_taun_pi2_' + x] + df['reco_taun_pi3_' + x] + df['reco_taun_pizero1_' + x] + df['reco_taun_pizero2_' + x]

        # save the concatenated dataframe
        if verbosity>0: print('Writing output/df_chunk_%i.pkl' % i)
        df.to_pickle('output/df_chunk_%i.pkl' % i)
        # save as csv file as well
        #if verbosity>0: print('Writing output/df_chunk_%i.csv' % i)
        #df.to_csv('output/df_chunk_%i.csv' % i, index=False)
        # delete the two dataframes
        del df
        del df_friend
        # delete the friends pkl file
        os.remove('output/df_friend_chunk_%i.pkl' % i)

class RegressionDataset(Dataset):
    def __init__(self, dataframe, input_features, output_features):
        # Convert entire dataset to tensors at initialization (avoids slow indexing)
        self.X = torch.tensor(dataframe[input_features].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[output_features].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
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

    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        #self.input_layer = nn.Linear(input_dim, 128)
        #self.hidden_layer = nn.Linear(128, 64)
        #self.output_layer = nn.Linear(64, output_dim)
        n_nodes = int(2./3*input_dim + output_dim)
        n_nodes = 2*input_dim
        print(input_dim, output_dim, n_nodes)
        self.input_layer = nn.Linear(input_dim, input_dim)
        self.hidden_layer_1 = nn.Linear(input_dim, n_nodes)
        self.hidden_layer_2 = nn.Linear(n_nodes, n_nodes)
        self.hidden_layer_3 = nn.Linear(n_nodes, n_nodes)
        self.output_layer = nn.Linear(n_nodes, output_dim)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer_1(x))
        x = torch.relu(self.hidden_layer_2(x))
        x = torch.relu(self.hidden_layer_3(x))
        x = self.output_layer(x)
        return x

def compute_tau_four_vectors(df, y):
    """
    Compute the tau 4-vectors by summing the momenta and energy of 
    reconstructed pions, pi0s, and neutrinos.
    """
    # make dataframe with same 8 output columns and same rows as initial dataframe
    taus_df = pd.DataFrame(index=df.index, columns=['reco_taup_px', 'reco_taup_py', 'reco_taup_pz', 'reco_taup_e', 'reco_taun_px', 'reco_taun_py', 'reco_taun_pz', 'reco_taun_e'])
    ## get the gen-level neutrinos
    #taup_nu = df[['taup_nu_px', 'taup_nu_py', 'taup_nu_pz']]
    #taun_nu = df[['taun_nu_px', 'taun_nu_py', 'taun_nu_pz']]
    ## we need to compute the energy which is calculated from the momentum assuming mass = 0
    #taup_nu['taup_nu_e'] = np.sqrt(np.sum(taup_nu**2, axis=1))
    #taun_nu['taun_nu_e'] = np.sqrt(np.sum(taun_nu**2, axis=1))
    ## get the recnstructed neutrinos from the y tensor 
    reco_taup_nu = pd.DataFrame(index=df.index, columns=['reco_taup_nu_px', 'reco_taup_nu_py', 'reco_taup_nu_pz'])
    reco_taun_nu = pd.DataFrame(index=df.index, columns=['reco_taun_nu_px', 'reco_taun_nu_py', 'reco_taun_nu_pz'])
    reco_taup_nu['reco_taup_nu_px'] = y[:,0]
    reco_taup_nu['reco_taup_nu_py'] = y[:,1]
    reco_taup_nu['reco_taup_nu_pz'] = y[:,2]
    reco_taup_nu['reco_taup_nu_e'] = np.sqrt(np.sum(reco_taup_nu**2, axis=1))
    reco_taun_nu['reco_taun_nu_px'] = y[:,3]
    reco_taun_nu['reco_taun_nu_py'] = y[:,4]
    reco_taun_nu['reco_taun_nu_pz'] = y[:,5]
    reco_taun_nu['reco_taun_nu_e'] = np.sqrt(np.sum(reco_taun_nu**2, axis=1))
    #sum the momenta and energy of the pions and pi0s and neutrinos
    for x in ['px', 'py', 'pz', 'e']:
        # visible-only
        taus_df['reco_taup_vis_' + x] = df['reco_taup_pi1_' + x] + df['reco_taup_pi2_' + x] + df['reco_taup_pi3_' + x] + df['reco_taup_pizero1_' + x] + df['reco_taup_pizero2_' + x]
        taus_df['reco_taun_vis_' + x] = df['reco_taun_pi1_' + x] + df['reco_taun_pi2_' + x] + df['reco_taun_pi3_' + x] + df['reco_taun_pizero1_' + x] + df['reco_taun_pizero2_' + x]
        # visible+invisible
        taus_df['reco_taup_' + x] = taus_df['reco_taup_vis_' + x] + reco_taup_nu['reco_taup_nu_' + x]
        taus_df['reco_taun_' + x] = taus_df['reco_taun_vis_' + x] + reco_taun_nu['reco_taun_nu_' + x]
    ## now we can compute the tau mass for convenience
    taus_df['reco_taup_mass'] = np.sqrt(taus_df['reco_taup_e']**2 - (taus_df['reco_taup_px']**2 + taus_df['reco_taup_py']**2 + taus_df['reco_taup_pz']**2))
    taus_df['reco_taun_mass'] = np.sqrt(taus_df['reco_taun_e']**2 - (taus_df['reco_taun_px']**2 + taus_df['reco_taun_py']**2 + taus_df['reco_taun_pz']**2))
    return taus_df

def PolarimetricVector(pis=[], pizeros=[], nu=ROOT.TLorentzVector(0,0,0,0)):
    """
    Compute the polarimetric vector for the tau decay products.
    """

    # compute the polarimetric vector for the tau decay products
    tau = nu
    for pi in pis:
        tau += pi
    for pizero in pizeros:
        tau += pizero

    if len(pis) == 1 and len(pizeros) == 0:
        pi = pis[0]
        pi.Boost(-tau.BoostVector())
        tau_s = pi.Vect().Unit()
    elif len(pis) == 1 and len(pizeros) == 1:
        pi = pis[0]
        pi0 = pizeros[0]
        q = pi  - pi0
        P = tau
        N = tau - pi - pi0
        pv = P.M()*(2*(q*N)*q - q.Mag2()*N) * (1/ (2*(q*N)*(q*P) - q.Mag2()*(N*P)))
        pv.Boost(-tau.BoostVector())
        tau_s = pv.Vect().Unit()
    elif len(pis) == 3 and len(pizeros) == 0:
        pi1 = pis[0]
        pi2 = pis[1]
        pi3 = pis[2]

        pv =  -PolarimetricA1(tau, pi1, pi2, pi3, +1).PVC()
        pv.Boost(-tau.BoostVector())
        tau_s = pv.Vect().Unit()

    else: 
        print(f'Warning: Invalid number of pions or pi0s, Npi = {len(pis)}, Npi0 = {len(pizeros)}')
        tau_s = ROOT.TVector3(0,0,0)

    return tau_s


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--stages', '-s', help='a list of stages to run', type=int, nargs="+")
    args = argparser.parse_args()

    filename1 = "batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_10000000_Njob_10000/pythia_events.root"
    filename2 = "batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_10000000_Njob_10000/pythia_events_extravars.root"
    treename1 = "tree"
    treename2 = "new_tree"

    # stage 1: prepare dataframes
    if 1 in args.stages:
        PrepareDataframes(filename1, filename2, treename1, treename2, nchunks=20, verbosity=1)


    batch_size = 32
    input_features = [
        'reco_taup_pi1_px','reco_taup_pi1_py','reco_taup_pi1_pz','reco_taup_pi1_e',
        'reco_taup_pi2_px','reco_taup_pi2_py','reco_taup_pi2_pz','reco_taup_pi2_e',
        'reco_taup_pi3_px','reco_taup_pi3_py','reco_taup_pi3_pz','reco_taup_pi3_e',
        'reco_taup_pizero1_px','reco_taup_pizero1_py','reco_taup_pizero1_pz','reco_taup_pizero1_e',
        #'reco_taup_pizero2_px','reco_taup_pizero2_py','reco_taup_pizero2_pz','reco_taup_pizero2_e',
        'reco_taun_pi1_px','reco_taun_pi1_py','reco_taun_pi1_pz','reco_taun_pi1_e',
        'reco_taun_pi2_px','reco_taun_pi2_py','reco_taun_pi2_pz','reco_taun_pi2_e',
        'reco_taun_pi3_px','reco_taun_pi3_py','reco_taun_pi3_pz','reco_taun_pi3_e',
        'reco_taun_pizero1_px','reco_taun_pizero1_py','reco_taun_pizero1_pz','reco_taun_pizero1_e',
        #'reco_taun_pizero2_px','reco_taun_pizero2_py','reco_taun_pizero2_pz','reco_taun_pizero2_e',
        'reco_taup_vx','reco_taup_vy','reco_taup_vz',
        'reco_taun_vx','reco_taun_vy','reco_taun_vz',
        'reco_taup_pi1_ipx','reco_taup_pi1_ipy','reco_taup_pi1_ipz',
        'reco_taun_pi1_ipx','reco_taun_pi1_ipy','reco_taun_pi1_ipz',
        'reco_Z_px','reco_Z_py','reco_Z_pz','reco_Z_e']
    output_features = [
        'taup_nu_px','taup_nu_py','taup_nu_pz',
        'taun_nu_px','taun_nu_py','taun_nu_pz']
    file_paths = [f'output/df_chunk_{i}.pkl' for i in range(3)]
    dataframes = [pd.read_pickle(file) for file in file_paths]
    train_df = pd.concat(dataframes[:1], ignore_index=True)
    test_df = pd.concat(dataframes[2:], ignore_index=True)

    # stage 2: train the model
    if 2 in args.stages:


        train_dataset = RegressionDataset(train_df, input_features, output_features)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = RegressionDataset(test_df, input_features, output_features)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = SimpleNN(len(input_features), len(output_features))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 20
        loss_values = []
        val_loss_values = []

        print('Training the model...')
        current_time = time.time()
        early_stopper = EarlyStopper(patience=5, min_delta=0.)
        for epoch in range(num_epochs):
            model.train()
            running_loss= 0.0
            for i, (X, y) in enumerate(train_dataloader):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss/len(train_dataloader)
            loss_values.append(train_loss)
            
            # get the validation loss
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for i, (X, y) in enumerate(test_dataloader):
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                val_loss /= len(test_dataloader)
                # apply early stopping
                val_loss_values.append(val_loss)
                
                if early_stopper.early_stop(val_loss):
                    print(f"Early stopping triggered for epoch {epoch+1}")
                    break
                

            print(f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
        
        elapsed_time = time.time() - current_time
        print (f"Training time: {elapsed_time:.2f} seconds")
        # Save the model
        torch.save(model.state_dict(), 'model.pth')

        plt.figure()
        plt.plot(range(2, len(loss_values)+1), loss_values[1:], label='train loss')
        plt.plot(range(2, len(val_loss_values)+1), val_loss_values[1:], label='validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('loss_vs_epoch.pdf')

    # test the model
    if 3 in args.stages:
        # load the pytorch model 'model.pth'
        model = SimpleNN(len(input_features), len(output_features))
        model.load_state_dict(torch.load('model.pth'))
        model.eval()

        # apply the model to the test set
        test_dataset = RegressionDataset(test_df, input_features, output_features)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # get the predictions
        predictions = []
        with torch.no_grad():
            for i, (X, y) in enumerate(test_dataloader):
                outputs = model(X)
                predictions.append(outputs.numpy())
        predictions = np.concatenate(predictions, axis=0)
        # get the true values
        true_values = test_df[output_features].values
        # calculate the mean squared error
        mse = np.mean((predictions - true_values)**2)
        print(f'Mean squared error: {mse:.4f}')

        taus_df = compute_tau_four_vectors(test_df, predictions)
        # add tau_df to test_df
        test_df = pd.concat([test_df, taus_df], axis=1)

        # predict spin sensitive variables, note needs to be done in a loop for now

        test_df['cosn_plus_nn_reco'] = -9999.
        test_df['cosr_plus_nn_reco'] = -9999.
        test_df['cosk_plus_nn_reco'] = -9999.
        test_df['cosn_minus_nn_reco'] = -9999.
        test_df['cosr_minus_nn_reco'] = -9999.
        test_df['cosk_minus_nn_reco'] = -9999.
        test_df['cosTheta_nn_reco'] = -9999.

        for i in range(len(test_df)):
            if i > 10000: break
            if i % 1000 == 0:
                print(f'Processing event {i}/{len(test_df)}')
            # get the tau decay product 4-vectors and the neutrino 4-vectors as TLorentzVectors

            taup_pi1 = ROOT.TLorentzVector(test_df.iloc[i][['reco_taup_pi1_px']].values[0],
                                           test_df.iloc[i][['reco_taup_pi1_py']].values[0],
                                           test_df.iloc[i][['reco_taup_pi1_pz']].values[0],
                                           test_df.iloc[i][['reco_taup_pi1_e']].values[0])
            taup_pi2 = ROOT.TLorentzVector(test_df.iloc[i][['reco_taup_pi2_px']].values[0],
                                           test_df.iloc[i][['reco_taup_pi2_py']].values[0],
                                           test_df.iloc[i][['reco_taup_pi2_pz']].values[0],
                                           test_df.iloc[i][['reco_taup_pi2_e']].values[0])
            taup_pi3 = ROOT.TLorentzVector(test_df.iloc[i][['reco_taup_pi3_px']].values[0],
                                             test_df.iloc[i][['reco_taup_pi3_py']].values[0],
                                             test_df.iloc[i][['reco_taup_pi3_pz']].values[0],
                                             test_df.iloc[i][['reco_taup_pi3_e']].values[0])
            taup_pizero1 = ROOT.TLorentzVector(test_df.iloc[i][['reco_taup_pizero1_px']].values[0],
                                               test_df.iloc[i][['reco_taup_pizero1_py']].values[0],
                                               test_df.iloc[i][['reco_taup_pizero1_pz']].values[0],
                                               test_df.iloc[i][['reco_taup_pizero1_e']].values[0])

            taun_pi1 = ROOT.TLorentzVector(test_df.iloc[i][['reco_taun_pi1_px']].values[0],
                                           test_df.iloc[i][['reco_taun_pi1_py']].values[0],
                                           test_df.iloc[i][['reco_taun_pi1_pz']].values[0],
                                           test_df.iloc[i][['reco_taun_pi1_e']].values[0])
            taun_pi2 = ROOT.TLorentzVector(test_df.iloc[i][['reco_taun_pi2_px']].values[0],
                                           test_df.iloc[i][['reco_taun_pi2_py']].values[0],
                                           test_df.iloc[i][['reco_taun_pi2_pz']].values[0],
                                           test_df.iloc[i][['reco_taun_pi2_e']].values[0])
            taun_pi3 = ROOT.TLorentzVector(test_df.iloc[i][['reco_taun_pi3_px']].values[0],
                                           test_df.iloc[i][['reco_taun_pi3_py']].values[0],
                                           test_df.iloc[i][['reco_taun_pi3_pz']].values[0],
                                           test_df.iloc[i][['reco_taun_pi3_e']].values[0])
            taun_pizero1 = ROOT.TLorentzVector(test_df.iloc[i][['reco_taun_pizero1_px']].values[0],
                                               test_df.iloc[i][['reco_taun_pizero1_py']].values[0],
                                               test_df.iloc[i][['reco_taun_pizero1_pz']].values[0],
                                               test_df.iloc[i][['reco_taun_pizero1_e']].values[0])
            # get the neutrino 4-vectors
            taup_nu = ROOT.TLorentzVector(predictions[i][0], predictions[i][1], predictions[i][2], np.sqrt(np.sum(predictions[i][:3]**2)))
            taun_nu = ROOT.TLorentzVector(predictions[i][3], predictions[i][4], predictions[i][5], np.sqrt(np.sum(predictions[i][3:6]**2)))

            taup = taup_pi1 + taup_pi2 + taup_pi3 + taup_pizero1 + taup_nu
            taun = taun_pi1 + taun_pi2 + taun_pi3 + taun_pizero1 + taun_nu

            taup_pis = [ x for x in [taup_pi1, taup_pi2, taup_pi3] if x.E() > 0]
            taup_pizeros = [ x for x in [taup_pizero1] if x.E() > 0]
            taun_pis = [ x for x in [taun_pi1, taun_pi2, taun_pi3] if x.E() > 0]
            taun_pizeros = [ x for x in [taun_pizero1] if x.E() > 0]

            taup_s = PolarimetricVector(taup_pis, taup_pizeros, taup_nu)
            taun_s = PolarimetricVector(taun_pis, taun_pizeros, taun_nu)

            # p is direction of e+ beam - this will be known also for reco variables!
            p = ROOT.TVector3(test_df.iloc[i][['z_x']].values[0],
                              test_df.iloc[i][['z_y']].values[0],
                              test_df.iloc[i][['z_z']].values[0])

            # k is direction of tau+
            k = taup.Vect().Unit()
            n = (p.Cross(k)).Unit()
            cosTheta = p.Dot(k)
            r = (p - (k*cosTheta)).Unit() 

            cosn_plus = taup_s.Dot(n)
            cosr_plus = taup_s.Dot(r)
            cosk_plus = taup_s.Dot(k)
            cosn_minus = taun_s.Dot(n)
            cosr_minus = taun_s.Dot(r)
            cosk_minus = taun_s.Dot(k)

            # store these as 'xxx_nn_reco' named variables in the dataframe
            test_df.at[i,'cosn_plus_nn_reco'] = cosn_plus
            test_df.at[i,'cosr_plus_nn_reco'] = cosr_plus
            test_df.at[i,'cosk_plus_nn_reco'] = cosk_plus
            test_df.at[i,'cosn_minus_nn_reco'] = cosn_minus
            test_df.at[i,'cosr_minus_nn_reco'] = cosr_minus
            test_df.at[i,'cosk_minus_nn_reco'] = cosk_minus
            test_df.at[i,'cosTheta_nn_reco'] = cosTheta

        # save the dataframe with the new variables
        test_df.to_pickle('output/df_test_inc_nn_vars.pkl')
        # also store it as a root file
        with uproot.recreate('output/df_test_inc_nn_vars.root') as root_file:
            root_file['tree'] = test_df 
