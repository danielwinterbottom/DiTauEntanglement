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
from collections import OrderedDict
import torch.nn.init as init
from torch.optim.lr_scheduler import _LRScheduler


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
        'reco_taup_nu_px','reco_taup_nu_py','reco_taup_nu_pz',
        'reco_taun_nu_px','reco_taun_nu_py','reco_taun_nu_pz', 
        'reco_alt_taup_nu_px','reco_alt_taup_nu_py','reco_alt_taup_nu_pz',
        'reco_alt_taun_nu_px','reco_alt_taun_nu_py','reco_alt_taun_nu_pz',
        'reco_d0_taup_nu_px','reco_d0_taup_nu_py','reco_d0_taup_nu_pz',
        'reco_d0_taun_nu_px','reco_d0_taun_nu_py','reco_d0_taun_nu_pz',
        'dsign','dsign_alt',
        'dplus_taup_nu_px','dplus_taup_nu_py','dplus_taup_nu_pz',
        'dplus_taun_nu_px','dplus_taun_nu_py','dplus_taun_nu_pz',
        'dminus_taup_nu_px','dminus_taup_nu_py','dminus_taup_nu_pz',
        'dminus_taun_nu_px','dminus_taun_nu_py','dminus_taun_nu_pz',

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

        # define tau decay modes 0,1,2
        # 0 = 1pi0pi0, 1 = 1pi1pi0, 2 = 3pi
        df['taup_decay_mode'] = 0
        df.loc[(df['taup_npi'] == 1) & (df['taup_npizero'] == 1), 'taup_decay_mode'] = 1
        df.loc[(df['taup_npi'] == 3) & (df['taup_npizero'] == 0), 'taup_decay_mode'] = 2
        df['taun_decay_mode'] = 0
        df.loc[(df['taun_npi'] == 1) & (df['taun_npizero'] == 1), 'taun_decay_mode'] = 1
        df.loc[(df['taun_npi'] == 3) & (df['taun_npizero'] == 0), 'taun_decay_mode'] = 2

        # calculate delta IPs and delta SVs and store as unit vectors and magnitudes
        # for taup
        delta_ipx = df['reco_taup_pi1_ipx'] - df['reco_taun_pi1_ipx']
        delta_ipy = df['reco_taup_pi1_ipy'] - df['reco_taun_pi1_ipy']
        delta_ipz = df['reco_taup_pi1_ipz'] - df['reco_taun_pi1_ipz']
        delta_ip = np.sqrt(delta_ipx**2 + delta_ipy**2 + delta_ipz**2)
        # store the unit vector, if delta_ip = 0 then set to 0
        df['delta_ipx'] = np.where(delta_ip == 0, 0, delta_ipx/delta_ip)
        df['delta_ipy'] = np.where(delta_ip == 0, 0, delta_ipy/delta_ip)
        df['delta_ipz'] = np.where(delta_ip == 0, 0, delta_ipz/delta_ip)
        df['delta_ip_mag'] = delta_ip

        # now the same for the SVs
        delta_svx = df['reco_taup_vx'] - df['reco_taun_vx']
        delta_svy = df['reco_taup_vy'] - df['reco_taun_vy']
        delta_svz = df['reco_taup_vz'] - df['reco_taun_vz']
        delta_sv = np.sqrt(delta_svx**2 + delta_svy**2 + delta_svz**2)
        # store the unit vector, if delta_sv = 0 then set to 0
        df['delta_svx'] = np.where(delta_sv == 0, 0, delta_svx/delta_sv)
        df['delta_svy'] = np.where(delta_sv == 0, 0, delta_svy/delta_sv)
        df['delta_svz'] = np.where(delta_sv == 0, 0, delta_svz/delta_sv)
        df['delta_sv_mag'] = delta_sv


        # add the visible tau 4-vectors by summing the pi and pizero 4-vectors
        for x in ['px', 'py', 'pz', 'e']:
            df['reco_taup_vis_' + x] = df['reco_taup_pi1_' + x] + df['reco_taup_pi2_' + x] + df['reco_taup_pi3_' + x] + df['reco_taup_pizero1_' + x] + df['reco_taup_pizero2_' + x]
            df['reco_taun_vis_' + x] = df['reco_taun_pi1_' + x] + df['reco_taun_pi2_' + x] + df['reco_taun_pi3_' + x] + df['reco_taun_pizero1_' + x] + df['reco_taun_pizero2_' + x]

        # convert to the polar coordinate system and add these variables as well
        # create the polar coordinates object
        # initialised like: (self, tau_p, nu_p):
        tau_p = df[['reco_taup_vis_px', 'reco_taup_vis_py', 'reco_taup_vis_pz']].values
        nu_p = df[['taup_nu_px', 'taup_nu_py', 'taup_nu_pz']].values
        taup_polar = NeutrinoPolarCoordinates(tau_p)
        # compute the angles
        nu_p_mag, theta, phi = taup_polar.compute_angles(nu_p)
        # add to the df
        df['taup_nu_p_mag'] = nu_p_mag
        df['taup_nu_p_mag_rel'] = df['taup_nu_p_mag'] / df['reco_taup_vis_e'] 
        df['taup_nu_phi'] = phi
        df['taup_nu_theta'] = theta
        # now do the same for the taun
        tau_p = df[['reco_taun_vis_px', 'reco_taun_vis_py', 'reco_taun_vis_pz']].values
        nu_p = df[['taun_nu_px', 'taun_nu_py', 'taun_nu_pz']].values
        taun_polar = NeutrinoPolarCoordinates(tau_p)
        # compute the angles
        nu_p_mag, theta, phi = taun_polar.compute_angles(nu_p)
        # add to the df
        df['taun_nu_p_mag'] = nu_p_mag
        df['taun_nu_p_mag_rel'] = df['taun_nu_p_mag'] / df['reco_taun_vis_e'] 
        df['taun_nu_phi'] = phi
        df['taun_nu_theta'] = theta
        # now the same for reco_taup_nu_pi
        tau_p = df[['reco_taup_vis_px', 'reco_taup_vis_py', 'reco_taup_vis_pz']].values
        nu_p = df[['reco_taup_nu_px', 'reco_taup_nu_py', 'reco_taup_nu_pz']].values
        taup_polar = NeutrinoPolarCoordinates(tau_p)
        # compute the angles
        nu_p_mag, theta, phi = taup_polar.compute_angles(nu_p)
        # add to the df
        df['reco_taup_nu_p_mag'] = nu_p_mag
        df['reco_taup_nu_p_mag_rel'] = df['reco_taup_nu_p_mag'] / df['reco_taup_vis_e']
        df['reco_taup_nu_phi'] = phi
        df['reco_taup_nu_theta'] = theta
        # now do the same for reco_taun_nu_pi
        tau_p = df[['reco_taun_vis_px', 'reco_taun_vis_py', 'reco_taun_vis_pz']].values
        nu_p = df[['reco_taun_nu_px', 'reco_taun_nu_py', 'reco_taun_nu_pz']].values
        taun_polar = NeutrinoPolarCoordinates(tau_p)
        # compute the angles
        nu_p_mag, theta, phi = taun_polar.compute_angles(nu_p)
        # add to the df
        df['reco_taun_nu_p_mag'] = nu_p_mag
        df['reco_taun_nu_p_mag_rel'] = df['reco_taun_nu_p_mag'] / df['reco_taun_vis_e']
        df['reco_taun_nu_phi'] = phi
        df['reco_taun_nu_theta'] = theta

        nu_p_dplus = df[['dplus_taup_nu_px', 'dplus_taup_nu_py', 'dplus_taup_nu_pz']].values
        nu_p_dminus = df[['dminus_taup_nu_px', 'dminus_taup_nu_py', 'dminus_taup_nu_pz']].values

        nu_n_dplus = df[['dplus_taun_nu_px', 'dplus_taun_nu_py', 'dplus_taun_nu_pz']].values

        d = (nu_p_dplus - nu_p_dminus)/2
        df[['d_x', 'd_y', 'd_z']] = d

        nu_p_d0 = nu_p_dplus - d
        nu_n_d0 = nu_n_dplus + d
        df[['d0_taup_nu_px', 'd0_taup_nu_py', 'd0_taup_nu_pz']] = nu_p_d0
        df[['d0_taun_nu_px', 'd0_taun_nu_py', 'd0_taun_nu_pz']] = nu_n_d0

        # save the concatenated dataframe
        if verbosity>0: 
            # print df collumns making sure all are displayed
            pd.set_option('display.max_columns', None)
            print('Dataframe columns:')
            print(df.columns)
            print('Writing output/df_chunk_%i.pkl' % i)
        df.to_pickle('output/df_chunk_%i.pkl' % i)
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

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normlize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, input):
        x = x - self.mean
        x = x / self.std
        return x

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        print('initializing weights for layer %s' % module)
        #init.orthogonal_(module.weight)

def DeleteOldModels(model_name, val_loss_values, current_epoch):
    # check the models stored for previous epochs. If the validation loss for the current model is better then delete the previous models
    # search for all files in the current directory that start with the model name and end with .pth
    name_to_search = f'{model_name}_epoch_'
    files = [f for f in os.listdir('.') if f.startswith(name_to_search) and f.endswith('.pth')]
    current_loss = val_loss_values[-1]
    for file in files:
        # get the epoch number from the filename
        epoch = int(file.split('_')[-1].split('.')[0])
        if epoch >= current_epoch: continue
        # get the validation loss for that epoch
        val_loss = val_loss_values[epoch-1]
        # if the current loss is better than the previous loss then delete the previous model
        if current_loss < val_loss:
            os.remove(file)
            print(f'Deleted model {file} with validation loss {val_loss:.6f}')

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

class RelativeHuberLoss(nn.Module):
    def __init__(self, delta=1.0, eps=0.1):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, prediction, target):
        # Relative error
        rel_error = (prediction - target) / (torch.abs(target) + self.eps)
        abs_rel_error = torch.abs(rel_error)

        # Huber-like behavior on relative error
        mask = abs_rel_error < self.delta
        loss = torch.where(
            mask,
            0.5 * rel_error ** 2,
            self.delta * (abs_rel_error - 0.5 * self.delta)
        )

        return loss.mean()

def compute_tau_four_vectors(df, y):
    """
    Compute the tau 4-vectors by summing the momenta and energy of 
    reconstructed pions, pi0s, and neutrinos.
    """
    # make dataframe with same 8 output columns and same rows as initial dataframe
    taus_df = pd.DataFrame(index=df.index, columns=['reco_taup_px', 'reco_taup_py', 'reco_taup_pz', 'reco_taup_e', 'reco_taun_px', 'reco_taun_py', 'reco_taun_pz', 'reco_taun_e'])

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
        taus_df['reco_nn_taup_' + x] = taus_df['reco_taup_vis_' + x] + reco_taup_nu['reco_taup_nu_' + x]
        taus_df['reco_nn_taun_' + x] = taus_df['reco_taun_vis_' + x] + reco_taun_nu['reco_taun_nu_' + x]
         # store the neutrino momenta
        taus_df['reco_nn_taup_nu_' + x] = reco_taup_nu['reco_taup_nu_' + x]
        taus_df['reco_nn_taun_nu_' + x] = reco_taun_nu['reco_taun_nu_' + x]

    ## now we can compute the tau mass for convenience
    taus_df['reco_nn_taup_mass'] = np.sqrt(taus_df['reco_nn_taup_e']**2 - (taus_df['reco_nn_taup_px']**2 + taus_df['reco_nn_taup_py']**2 + taus_df['reco_nn_taup_pz']**2))
    taus_df['reco_nn_taun_mass'] = np.sqrt(taus_df['reco_nn_taun_e']**2 - (taus_df['reco_nn_taun_px']**2 + taus_df['reco_nn_taun_py']**2 + taus_df['reco_nn_taun_pz']**2))
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

class NeutrinoPolarCoordinates:
    def __init__(self, tau_p):
        """
        Class to compute the polar angles (theta, phi) of a neutrino relative to a visible tau momentum
        and perform the inverse transformation.
        
        :param tau_p: Array-like shape (N,3), visible tau momenta [px, py, pz].
        :param nu_p: Array-like shape (N,3), neutrino momenta [px, py, pz].
        """
        self.tau_p = np.array(tau_p)

    
    def compute_angles(self, nu_p):
        """
        Compute the theta and phi angles for the neutrino.
        """
        # Compute theta (opening angle)
        dot_product = np.einsum('ij,ij->i', self.tau_p, nu_p)  # Efficient row-wise dot product
        tau_norm = np.linalg.norm(self.tau_p, axis=1)
        nu_norm = np.linalg.norm(nu_p, axis=1)
        
        theta = np.arccos(np.clip(dot_product / (tau_norm * nu_norm), -1.0, 1.0))
        
        # Compute an orthonormal basis for the plane perpendicular to tau momentum
        tau_normed = self.tau_p / tau_norm[:, None]  # Normalize tau momentum

        ref_vector = np.array([0, 0, 1])
        
        e1 = np.cross(tau_normed, ref_vector)
        e1 /= np.linalg.norm(e1, axis=1)[:, None]  # Normalize
        e2 = np.cross(tau_normed, e1)
        
        # Project neutrino momentum onto the plane perpendicular to tau
        nu_proj = nu_p - (np.einsum('ij,ij->i', nu_p, tau_normed)[:, None] * tau_normed)
        
        # Compute phi using atan2
        phi = np.arctan2(np.einsum('ij,ij->i', nu_proj, e2), np.einsum('ij,ij->i', nu_proj, e1))

        nu_p_mag = np.linalg.norm(nu_p, axis=1)
        
        return nu_p_mag, theta, phi
    
    def reverse_transform(self, nu_p_mag, nu_theta, nu_phi):
        """
        Reverse the transformation to recover the neutrino momenta in the original frame.
        
        :param theta: Array-like shape (N,), opening angle.
        :param phi: Array-like shape (N,), angular position on the cone.
        :return: Reconstructed neutrino momenta (N,3)
        """
        tau_norm = np.linalg.norm(self.tau_p, axis=1)
        tau_normed = self.tau_p / tau_norm[:, None]
        
        ref_vector = np.array([0, 0, 1])
        
        e1 = np.cross(tau_normed, ref_vector)
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(tau_normed, e1)
        
        nu_proj = np.cos(nu_phi)[:, None] * e1 + np.sin(nu_phi)[:, None] * e2
        
        nu_p_reconstructed = nu_p_mag[:, None] * (np.cos(nu_theta)[:, None] * tau_normed + np.sin(nu_theta)[:, None] * nu_proj)
        
        return nu_p_reconstructed
   

class TauFrameRotator:
    """
    A class to rotate a given tau's neutrino momentum so that the z-axis is aligned with the tau's visible momentum.

    Rotation happens in two steps:
        1. Rotate around the z-axis to remove the azimuthal angle (phi).
        2. Rotate around the y-axis to align the momentum with the z-axis.

    Attributes:
        theta (float): The polar angle of the tau's visible momentum.
        phi (float): The azimuthal angle of the tau's visible momentum.

    Methods:
        set_rotation_angles(px, py, pz): Compute and store theta and phi based on tau momentum.
        apply_rotation(px, py, pz, reverse=False): Apply the stored rotation (or its inverse) to a given momentum.
        rotate_dataframe(df, nu_prefix, reverse=False): Rotate neutrino momenta in a DataFrame.
    """

    def __init__(self):
        self.theta = None
        self.phi = None

    def set_rotation_angles(self, px, py, pz):
        """Set theta and phi based on the tau's visible 3-momentum."""
        norm_xy = np.hypot(px, py)  # sqrt(px^2 + py^2)
        self.phi = np.arctan2(py, px)  # Azimuthal angle
        self.theta = np.arctan2(norm_xy, pz)  # Polar angle    

    def apply_rotation(self, px, py, pz, reverse=False):
        """Rotate a given 3-momentum using the stored theta and phi."""
        if self.theta is None or self.phi is None:
            raise ValueError("Rotation angles have not been set. Call `set_rotation_angles` first.")

        if reverse:
            theta, phi = -self.theta, -self.phi
            # Rotate around Y-axis
            cos_theta, sin_theta = np.cos(-theta), np.sin(-theta)
            px, pz = cos_theta * px + sin_theta * pz, -sin_theta * px + cos_theta * pz

            # Rotate around Z-axis
            cos_phi, sin_phi = np.cos(-phi), np.sin(-phi)
            px, py = cos_phi * px - sin_phi * py, sin_phi * px + cos_phi * py
        else:
            theta, phi = self.theta, self.phi

            # Rotate around Z-axis
            cos_phi, sin_phi = np.cos(-phi), np.sin(-phi)
            px, py = cos_phi * px - sin_phi * py, sin_phi * px + cos_phi * py

            # Rotate around Y-axis
            cos_theta, sin_theta = np.cos(-theta), np.sin(-theta)
            px, pz = cos_theta * px + sin_theta * pz, -sin_theta * px + cos_theta * pz

        return px, py, pz

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--stages', '-s', help='a list of stages to run', type=int, nargs="+")
    argparser.add_argument('--model_name', '-m', help='the name of the model output name', type=str, default='model')
    argparser.add_argument('--n_hidden_layers', help='number of hidden layers', type=int, default=6)
    argparser.add_argument('--n_nodes', help='number of nodes per layer', type=int, default=300)
    argparser.add_argument('--batch_size', help='batch size', type=int, default=1024)
    argparser.add_argument('--train_dsolution', help='train a specific solutions for the d sign (+/- 1 - other values default to the ordinary training)', default=None, type=int)
    argparser.add_argument('--loss', help='loss function to use options are MSE, MAE, or Huber', type=str, default='MSE')
    args = argparser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    filename1 = "batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_10000000_Njob_10000/pythia_events.root"
    filename2 = "batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_10000000_Njob_10000/pythia_events_extravars.root"
    treename1 = "tree"
    treename2 = "new_tree"

    standardize = 2 # 0: no standardization, 1: standardize input, 2: standardize input and output
    polar = False
    #num_epochs = 415
    num_epochs = 115
    n_hidden_layers = args.n_hidden_layers
    
    n_nodes = args.n_nodes

    # stage 1: prepare dataframes
    if 1 in args.stages:
        PrepareDataframes(filename1, filename2, treename1, treename2, nchunks=20, verbosity=1)


    batch_size = args.batch_size
    input_features = [
        'reco_taup_vis_px','reco_taup_vis_py','reco_taup_vis_pz','reco_taup_vis_e',
        'reco_taun_vis_px','reco_taun_vis_py','reco_taun_vis_pz','reco_taun_vis_e',
        'reco_taup_pi1_px','reco_taup_pi1_py','reco_taup_pi1_pz','reco_taup_pi1_e',
        'reco_taup_pi2_px','reco_taup_pi2_py','reco_taup_pi2_pz','reco_taup_pi2_e',
        'reco_taup_pi3_px','reco_taup_pi3_py','reco_taup_pi3_pz','reco_taup_pi3_e',
        'reco_taup_pizero1_px','reco_taup_pizero1_py','reco_taup_pizero1_pz','reco_taup_pizero1_e',
        'reco_taun_pi1_px','reco_taun_pi1_py','reco_taun_pi1_pz','reco_taun_pi1_e',
        'reco_taun_pi2_px','reco_taun_pi2_py','reco_taun_pi2_pz','reco_taun_pi2_e',
        'reco_taun_pi3_px','reco_taun_pi3_py','reco_taun_pi3_pz','reco_taun_pi3_e',
        'reco_taun_pizero1_px','reco_taun_pizero1_py','reco_taun_pizero1_pz','reco_taun_pizero1_e',
        'reco_taup_vx','reco_taup_vy','reco_taup_vz',
        'reco_taun_vx','reco_taun_vy','reco_taun_vz',
        'reco_taup_pi1_ipx','reco_taup_pi1_ipy','reco_taup_pi1_ipz',
        'reco_taun_pi1_ipx','reco_taun_pi1_ipy','reco_taun_pi1_ipz',
        'delta_ipx','delta_ipy','delta_ipz','delta_ip_mag',
        'delta_svx','delta_svy','delta_svz','delta_sv_mag',
        'reco_Z_px','reco_Z_py','reco_Z_pz','reco_Z_e',
        'taup_decay_mode','taun_decay_mode',
        #'reco_taup_nu_px','reco_taup_nu_py','reco_taup_nu_pz',
        #'reco_taun_nu_px','reco_taun_nu_py','reco_taun_nu_pz',
        ]

    if args.train_dsolution == 1:
        # regress the dplus solution
        output_features = [
            'dplus_taup_nu_px','dplus_taup_nu_py','dplus_taup_nu_pz',
            'dplus_taun_nu_px','dplus_taun_nu_py','dplus_taun_nu_pz']
    elif args.train_dsolution == -1:
        # regress the dminus solution
        output_features = [
            'dminus_taup_nu_px','dminus_taup_nu_py','dminus_taup_nu_pz',
            'dminus_taun_nu_px','dminus_taun_nu_py','dminus_taun_nu_pz']
    elif args.train_dsolution == 0:
        # regress the d0 solution
        output_features = [
            'd0_taup_nu_px','d0_taup_nu_py','d0_taup_nu_pz',
            'd0_taun_nu_px','d0_taun_nu_py','d0_taun_nu_pz']
    else: 
        output_features = [
            'taup_nu_px','taup_nu_py','taup_nu_pz',
            'taun_nu_px','taun_nu_py','taun_nu_pz']

    print(f'Regressing output features: {output_features}')
    file_paths = [f'output/df_chunk_{i}.pkl' for i in range(20)]
    # only read the input and output features when reading the dfs
    #dataframes = [pd.read_pickle(file)[input_features + output_features + ['reco_taup_vis_px', 'reco_taup_vis_py', 'reco_taup_vis_pz']] for file in file_paths]
    dataframes = [pd.read_pickle(file) for file in file_paths]
    train_df = pd.concat(dataframes[:1], ignore_index=True)
    test_df = pd.concat(dataframes[1:2], ignore_index=True)
            
    # stage 2: train the model

    train_dataset = RegressionDataset(train_df, input_features, output_features)

    train_in_means, train_in_stds = train_dataset.get_input_means_stds()
    train_out_means, train_out_stds = train_dataset.get_output_means_stds()


    def custom_loss(output, target, X, theta_scale=2., phi_scale=1./10, separate=False):
        # first implement L1Loss (MAE) for the 2 neutrino x,y, and z components
        loss1 = torch.abs(output[:, 0] - target[:, 0]) # taup nu px
        loss2 = torch.abs(output[:, 1] - target[:, 1]) # taup nu py
        loss3 = torch.abs(output[:, 2] - target[:, 2]) # taup nu pz
        loss4 = torch.abs(output[:, 3] - target[:, 3]) # taun nu px
        loss5 = torch.abs(output[:, 4] - target[:, 4]) # taun nu py
        loss6 = torch.abs(output[:, 5] - target[:, 5]) # taun nu pz
        # now implement the L1Loss for the 3 visible tau momenta
        l1_loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6

        X_unscaled = X * train_in_stds + train_in_means
        output_unscaled = output * train_out_stds + train_out_means
        target_unscaled = target * train_out_stds + train_out_means

        # get the visible tau vectors from X then convert to df so that they can be used in NeutrinoPolarCoordinates
        taup_vis_tensor = X_unscaled[:, 0:3] + X_unscaled[:, 4:7] + X_unscaled[:, 8:11] + X_unscaled[:, 12:15]
        taun_vis_tensor = X_unscaled[:, 16:19] + X_unscaled[:, 20:23] + X_unscaled[:, 24:27] + X_unscaled[:, 28:31]

        # get neutrinos from the output and target tensors
        reco_taup_nu_tensor = output_unscaled[:, 0:3]
        reco_taun_nu_tensor = output_unscaled[:, 3:6]

        taup_nu_tensor = target_unscaled[:, 0:3]
        taun_nu_tensor = target_unscaled[:, 3:6]

        
        taup_vis = pd.DataFrame(taup_vis_tensor.cpu(), columns=['px', 'py', 'pz'])
        taun_vis = pd.DataFrame(taun_vis_tensor.cpu(), columns=['px', 'py', 'pz'])

        reco_taup_nu = pd.DataFrame(reco_taup_nu_tensor.cpu().detach(), columns=['px', 'py', 'pz'])
        reco_taun_nu = pd.DataFrame(reco_taun_nu_tensor.cpu().detach(), columns=['px', 'py', 'pz'])

        taup_nu = pd.DataFrame(taup_nu_tensor.cpu(), columns=['px', 'py', 'pz'])
        taun_nu = pd.DataFrame(taun_nu_tensor.cpu(), columns=['px', 'py', 'pz'])
        

        # get the polar coordinates
        taup_polar = NeutrinoPolarCoordinates(taup_vis)
        taun_polar = NeutrinoPolarCoordinates(taun_vis)
        reco_taup_nu_p_mag, reco_taup_nu_theta, reco_taup_nu_phi = taup_polar.compute_angles(reco_taup_nu)
        reco_taun_nu_p_mag, reco_taun_nu_theta, reco_taun_nu_phi = taun_polar.compute_angles(reco_taun_nu)
        taup_nu_p_mag, taup_nu_theta, taup_nu_phi = taup_polar.compute_angles(taup_nu)
        taun_nu_p_mag, taun_nu_theta, taun_nu_phi = taun_polar.compute_angles(taun_nu)

        # convert to tensors
        reco_taup_nu_p_mag = torch.tensor(reco_taup_nu_p_mag, dtype=torch.float32)
        reco_taup_nu_theta = torch.tensor(reco_taup_nu_theta, dtype=torch.float32)
        reco_taup_nu_phi = torch.tensor(reco_taup_nu_phi, dtype=torch.float32)
        reco_taun_nu_p_mag = torch.tensor(reco_taun_nu_p_mag, dtype=torch.float32)
        reco_taun_nu_theta = torch.tensor(reco_taun_nu_theta, dtype=torch.float32)
        reco_taun_nu_phi = torch.tensor(reco_taun_nu_phi, dtype=torch.float32)
        taup_nu_p_mag = torch.tensor(taup_nu_p_mag, dtype=torch.float32)
        taup_nu_theta = torch.tensor(taup_nu_theta, dtype=torch.float32)
        taup_nu_phi = torch.tensor(taup_nu_phi, dtype=torch.float32)
        taun_nu_p_mag = torch.tensor(taun_nu_p_mag, dtype=torch.float32)
        taun_nu_theta = torch.tensor(taun_nu_theta, dtype=torch.float32)
        taun_nu_phi = torch.tensor(taun_nu_phi, dtype=torch.float32)

        # send all to gpu
        reco_taup_nu_p_mag = reco_taup_nu_p_mag.to(device)
        reco_taup_nu_theta = reco_taup_nu_theta.to(device)
        reco_taup_nu_phi = reco_taup_nu_phi.to(device)
        reco_taun_nu_p_mag = reco_taun_nu_p_mag.to(device)
        reco_taun_nu_theta = reco_taun_nu_theta.to(device)
        reco_taun_nu_phi = reco_taun_nu_phi.to(device)
        taup_nu_p_mag = taup_nu_p_mag.to(device)
        taup_nu_theta = taup_nu_theta.to(device)
        taup_nu_phi = taup_nu_phi.to(device)
        taun_nu_p_mag = taun_nu_p_mag.to(device)
        taun_nu_theta = taun_nu_theta.to(device)
        taun_nu_phi = taun_nu_phi.to(device)

        taup_dphi = taup_nu_phi - reco_taup_nu_phi
        taun_dphi = taun_nu_phi - reco_taun_nu_phi
        taup_dphi = (taup_dphi + torch.pi) % (2 * torch.pi) - torch.pi
        taun_dphi = (taun_dphi + torch.pi) % (2 * torch.pi) - torch.pi

        phi_loss = phi_scale*(torch.abs(taup_dphi) + torch.abs(taun_dphi))
        theta_loss = theta_scale*torch.abs(taup_nu_theta - reco_taup_nu_theta) + torch.abs(taun_nu_theta - reco_taun_nu_theta)

        # compute totals
        total_loss = torch.mean(l1_loss + theta_loss + phi_loss)
        l1_loss = torch.mean(l1_loss)
        theta_loss = torch.mean(theta_loss)
        phi_loss = torch.mean(phi_loss)

        if separate: return total_loss, l1_loss, theta_loss, phi_loss
        return total_loss


    def custom_loss_2(output, target, X, EP_loss_scale=1., mass_loss_scale=0., separate=False):

        # remove the standardization from the input and output tensors since this can break energy and momentum conservation
        X_unscaled = X * train_in_stds + train_in_means
        output_unscaled = output * train_out_stds + train_out_means
        target_unscaled = target * train_out_stds + train_out_means

        loss1 = torch.abs(output[:, 0] - target[:, 0]) # taup nu px
        loss2 = torch.abs(output[:, 1] - target[:, 1]) # taup nu py
        loss3 = torch.abs(output[:, 2] - target[:, 2]) # taup nu pz
        loss4 = torch.abs(output[:, 3] - target[:, 3]) # taun nu px
        loss5 = torch.abs(output[:, 4] - target[:, 4]) # taun nu py
        loss6 = torch.abs(output[:, 5] - target[:, 5]) # taun nu pz

        ## first implement L1Loss (MAE) for the 2 neutrino x,y, and z components
        #loss1 = torch.abs(output_unscaled[:, 0] - target_unscaled[:, 0]) # taup nu px
        #loss2 = torch.abs(output_unscaled[:, 1] - target_unscaled[:, 1]) # taup nu py
        #loss3 = torch.abs(output_unscaled[:, 2] - target_unscaled[:, 2]) # taup nu pz
        #loss4 = torch.abs(output_unscaled[:, 3] - target_unscaled[:, 3]) # taun nu px
        #loss5 = torch.abs(output_unscaled[:, 4] - target_unscaled[:, 4]) # taun nu py
        #loss6 = torch.abs(output_unscaled[:, 5] - target_unscaled[:, 5]) # taun nu pz
        # now implement the L1Loss for the 3 visible tau momenta
        l1_loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6

        # get the visible tau 4-vectors from X
        #taup_vis
        taup_vis_tensor = X_unscaled[:, 0:4] + X_unscaled[:, 4:8] + X_unscaled[:, 8:12] + X_unscaled[:, 12:16]
        taun_vis_tensor = X_unscaled[:, 16:20] + X_unscaled[:, 20:24] + X_unscaled[:, 24:28] + X_unscaled[:, 28:32]

        # get neutrino 3-vectors from the output and target tensors
        reco_taup_nu_tensor = output_unscaled[:, 0:3]
        reco_taun_nu_tensor = output_unscaled[:, 3:6]
        # add the energy by summing the momenta
        reco_taun_nu_e = torch.sqrt(reco_taun_nu_tensor[:, 0]**2 + reco_taun_nu_tensor[:, 1]**2 + reco_taun_nu_tensor[:, 2]**2)
        reco_taup_nu_e = torch.sqrt(reco_taup_nu_tensor[:, 0]**2 + reco_taup_nu_tensor[:, 1]**2 + reco_taup_nu_tensor[:, 2]**2)
        # include the energy in the 4-vectors
        reco_taup_nu_tensor = torch.cat((reco_taup_nu_tensor, reco_taup_nu_e.unsqueeze(1)), dim=1)
        reco_taun_nu_tensor = torch.cat((reco_taun_nu_tensor, reco_taun_nu_e.unsqueeze(1)), dim=1)

        taup_nu_tensor = target_unscaled[:, 0:3]
        taun_nu_tensor = target_unscaled[:, 3:6]
        taup_nu_e = torch.sqrt(taup_nu_tensor[:, 0]**2 + taup_nu_tensor[:, 1]**2 + taup_nu_tensor[:, 2]**2)
        taun_nu_e = torch.sqrt(taun_nu_tensor[:, 0]**2 + taun_nu_tensor[:, 1]**2 + taun_nu_tensor[:, 2]**2)
        taup_nu_tensor = torch.cat((taup_nu_tensor, taup_nu_e.unsqueeze(1)), dim=1)
        taun_nu_tensor = torch.cat((taun_nu_tensor, taun_nu_e.unsqueeze(1)), dim=1)
        

        ## ditau energy and momentum conservation terms
        tau_mass_exp = 1.777
        reco_taup = reco_taup_nu_tensor + taup_vis_tensor
        reco_taun = reco_taun_nu_tensor + taun_vis_tensor

        taup = taup_nu_tensor + taup_vis_tensor
        taun = taun_nu_tensor + taun_vis_tensor

        reco_total_px = reco_taup[:, 0] + reco_taun[:, 0]
        reco_total_py = reco_taup[:, 1] + reco_taun[:, 1]
        reco_total_pz = reco_taup[:, 2] + reco_taun[:, 2]
        reco_total_e  = reco_taup[:, 3] + reco_taun[:, 3]

        total_px = taup[:, 0] + taun[:, 0]
        total_py = taup[:, 1] + taun[:, 1]
        total_pz = taup[:, 2] + taun[:, 2]
        total_e = taup[:, 3] + taun[:, 3]

        Ep_conservation_loss = EP_loss_scale*(torch.abs(reco_total_e - total_e) + torch.abs(reco_total_px - total_px) + torch.abs(reco_total_py - total_py) + torch.abs(reco_total_pz - total_pz))

        # now compute a loss term for the tau masses
        # make sure sqrt is not negative
        reco_taup_mass = torch.sqrt(torch.clamp(reco_taup[:, 3]**2 - (reco_taup[:, 0]**2 + reco_taup[:, 1]**2 + reco_taup[:, 2]**2), min=0.01))
        reco_taun_mass = torch.sqrt(torch.clamp(reco_taun[:, 3]**2 - (reco_taun[:, 0]**2 + reco_taun[:, 1]**2 + reco_taun[:, 2]**2), min=0.01))

        mass_diff_taup = reco_taup_mass - tau_mass_exp
        mass_diff_taun = reco_taun_mass - tau_mass_exp

        mass_loss = mass_loss_scale*((reco_taup_mass - tau_mass_exp)**2 + (reco_taun_mass - tau_mass_exp)**2)
        

        ## now get geometric term. Computed by finding cross products of tau visible and nu's. then taking delat phiu between these two
        #reco_taup_cross = torch.cross(taup_vis_tensor[:, 0:3], reco_taup_nu_tensor[:, 0:3])
        #reco_taun_cross = torch.cross(taun_vis_tensor[:, 0:3], reco_taun_nu_tensor[:, 0:3])
        #reco_taup_cross_norm = torch.norm(reco_taup_cross, dim=1)
        #reco_taun_cross_norm = torch.norm(reco_taun_cross, dim=1)
        #reco_taup_cross_unit = reco_taup_cross / reco_taup_cross_norm.unsqueeze(1)
        #reco_taun_cross_unit = reco_taun_cross / reco_taun_cross_norm.unsqueeze(1)
#
        #reco_dot_product = torch.einsum('ij,ij->i', reco_taup_cross_unit, reco_taun_cross_unit)
        #reco_dphi = torch.acos(torch.clamp(reco_dot_product, -1.0, 1.0))
#
        #taup_cross = torch.cross(taup_vis_tensor[:, 0:3], taup_nu_tensor[:, 0:3])
        #taun_cross = torch.cross(taun_vis_tensor[:, 0:3], taun_nu_tensor[:, 0:3])
        #taup_cross_norm = torch.norm(taup_cross, dim=1)
        #taun_cross_norm = torch.norm(taun_cross, dim=1)
        #taup_cross_unit = taup_cross / taup_cross_norm.unsqueeze(1)
        #taun_cross_unit = taun_cross / taun_cross_norm.unsqueeze(1)
        #dot_product = torch.einsum('ij,ij->i', taup_cross_unit, taun_cross_unit)
        #dphi = torch.acos(torch.clamp(dot_product, -1.0, 1.0)) # limit to [-1, 1] to avoid nan from acos
#
        #geometric_loss = torch.abs(reco_dphi - dphi)
        

        # compute totals
        total_loss = torch.mean(l1_loss + Ep_conservation_loss + mass_loss)
        #total_loss = torch.mean(l1_loss+mass_loss)
        l1_loss = torch.mean(l1_loss)
        mass_loss = torch.mean(mass_loss)
        Ep_conservation_loss = torch.mean(Ep_conservation_loss)
        #geometric_loss = torch.mean(geometric_loss)


        if separate: return total_loss, l1_loss, Ep_conservation_loss, mass_loss
        return total_loss

    if 2 in args.stages:

        if standardize >= 1:
            # normalize the input features
            train_dataset.X = (train_dataset.X - train_in_means) / train_in_stds
        if standardize == 2:
            # normalize the output features
            train_dataset.y = (train_dataset.y - train_out_means) / train_out_stds

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = RegressionDataset(test_df, input_features, output_features)
        if standardize >= 1:
            # normalize the input features
            test_dataset.X = (test_dataset.X - train_in_means) / train_in_stds
        if standardize == 2:
            # normalize the output features
            test_dataset.y = (test_dataset.y - train_out_means) / train_out_stds
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = SimpleNN(len(input_features), len(output_features), n_hidden_layers=n_hidden_layers, n_nodes=n_nodes)
        
        #try:
        #    model.load_state_dict(torch.load('model_MAE.pth')) # loading best model first
        #except:
        #    model.load_state_dict(torch.load('model_MAE.pth', map_location=torch.device('cpu')))

        if args.loss == 'MSE':
            print('Using MSE loss')
            criterion = nn.MSELoss()   
        elif args.loss == 'MAE':
            print('Using MAE loss')
            criterion = nn.L1Loss() 
        elif args.loss == 'Huber':
            print('Using Huber loss')
            criterion = nn.HuberLoss(delta=1.0)
        else: 
            raise ValueError(f"Unknown loss function: {args.loss}")


        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001)
        #optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.2, min_lr=1e-6,verbose=True)
        #optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.2, min_lr=1e-6,verbose=True)    
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98) # exponentially decaying lr

        start_epochs = 5
        ramp_epochs = 10
        low_lr = 0.00001
        peak_lr = 0.01
        gamma = 0.98
        
        optimizer = optim.Adam(model.parameters(), lr=low_lr, weight_decay=1e-5)

        def combined_schedule(epoch):
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

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=combined_schedule)       


        loss_values = []
        val_loss_values = []
        running_loss_values = []
        learning_rate_values = []

        print('Training the model...')
        weights = [1,1,1,1,1,1]
        current_time = time.time()
        early_stopper = EarlyStopper(patience=10, min_delta=0.)

        def plot_loss(loss_values, val_loss_values, running_loss_values=None):
            plt.figure()
            plt.plot(range(2, len(loss_values)+1), loss_values[1:], label='train loss')
            plt.plot(range(2, len(val_loss_values)+1), val_loss_values[1:], label='validation loss')
            if running_loss_values is not None: plt.plot(range(2, len(running_loss_values)+1), running_loss_values[1:], label='running loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(f'loss_vs_epoch_{args.model_name}.pdf')
            plt.close()

        def plot_learning_rate(learning_rate_values):
            plt.figure()
            plt.plot(range(1, len(learning_rate_values)+1), learning_rate_values)
            plt.xlabel('epoch')
            plt.ylabel('learning rate')
            plt.savefig(f'learning_rate_{args.model_name}.pdf')
            plt.close()

        def SignedLog(x):
            return torch.sign(x) * torch.log(torch.abs(x) + 1)

            

        model.to(device)
        for epoch in range(num_epochs):
            #model.train()
            running_loss= 0.0
            running_loss_components = [0.0, 0.0, 0.0, 0.0]
            for i, (X, y) in enumerate(train_dataloader):
                # move data to GPU
                #print(X.shape)
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                outputs = model(X)
                train_in_stds = train_in_stds.to(device)
                train_out_stds = train_out_stds.to(device)
                train_in_means = train_in_means.to(device)
                train_out_means = train_out_means.to(device)
                mass_loss_scale=0.
                EP_loss_scale=0.
                loss = criterion(outputs, y)
                #loss = criterion(SignedLog(outputs), SignedLog(y))

                loss.backward()
                optimizer.step()
                lr = scheduler.get_last_lr()
                
                running_loss += loss.item()


            running_loss /= len(train_dataloader)
            running_loss_values.append(running_loss)
            learning_rate_values.append(lr[0])
            for i in range(len(running_loss_components)):
                running_loss_components[i] /= len(train_dataloader)
            
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
                    #loss = criterion(SignedLog(outputs), SignedLog(y))
                    train_loss += loss.item()
                train_loss /= len(train_dataloader)
                loss_values.append(train_loss)
                val_loss = 0.0
                for i, (X, y) in enumerate(test_dataloader):
                    X = X.to(device)
                    y = y.to(device)
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    #loss = criterion(SignedLog(outputs), SignedLog(y))
                    val_loss += loss.item()
                val_loss /= len(test_dataloader)
                val_loss_values.append(val_loss)
                
                if early_stopper.early_stop(val_loss):
                    print(f"Early stopping triggered for epoch {epoch+1}")
                #    break # not triggering it just printing if it would have been stopped!

                lr = scheduler.get_last_lr()
                scheduler.step()
                #scheduler.step(val_loss) # for reduce on plateau
                

            print(f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, running_loss: {running_loss:.6f}, lr = {lr[0]:.6f}')
            if running_loss_components[0] != 0.:
                print(f'Running loss components: {running_loss_components[0]:.6f}, {running_loss_components[1]:.6f}, {running_loss_components[2]:.6f}, {running_loss_components[3]:.6f}')
            if epoch > 1: plot_loss(loss_values, val_loss_values, running_loss_values)
            plot_learning_rate(learning_rate_values)
            # save model every 10 epochs
            if (epoch+1) % 5 == 0:
                torch.save(model.state_dict(), f'{args.model_name}_epoch_{epoch+1}.pth')
                # after we save the model we delete the models for the previously stored epoch if the loss is better
                #try:
                #    os.remove(f'{args.model_name}_epoch_{epoch-4}.pth')
                #except:
                #    print(f"Could not remove model {args.model_name}_epoch_{epoch-4}.pth")
                DeleteOldModels(args.model_name, val_loss_values, epoch)
                print(f'Model saved at epoch {epoch+1}')
        
        elapsed_time = time.time() - current_time
        print (f"Training time: {elapsed_time:.2f} seconds")
        # Save the model
        torch.save(model.state_dict(), f'{args.model_name}.pth')


    # test the model
    if 3 in args.stages:
        # load the pytorch model 'model.pth'
        model = SimpleNN(len(input_features), len(output_features), n_hidden_layers=n_hidden_layers, n_nodes=n_nodes)
        #model.load_state_dict(torch.load('model.pth'))
        # try the below and if it doesnt work then load to cpu
        model_path = f'{args.model_name}_epoch_415.pth'
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print(f"Loading model from {model_path} failed. Trying to load from CPU.")
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # apply the model to the test set
        test_dataset = RegressionDataset(test_df, input_features, output_features)
        if standardize >= 1:
            # apply the same normalization as for the training set
            test_dataset.X = (test_dataset.X - train_in_means) / train_in_stds
        if standardize == 2:
            # normalize the output features
            test_dataset.y = (test_dataset.y - train_out_means) / train_out_stds
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # get the predictions
        predictions = []
        custom_loss_total, l1_loss_total, theta_loss_total, phi_loss_total = 0, 0, 0, 0
        custom_loss_2_total, l1_loss_2_total, Ep_loss_2_total, mass_loss_2_total = 0, 0, 0, 0
        with torch.no_grad():
            for i, (X, y) in enumerate(test_dataloader):
                outputs = model(X)
                if standardize == 2:
                    # convert predictions back to original scale
                    custom_loss_i, l1_loss_i, theta_loss_i, phi_loss_i = 0, 0, 0, 0
                    custom_loss_2_i, l1_loss_2_i, Ep_loss_i, mass_loss_i = 0, 0, 0, 0
                    #custom_loss_i, l1_loss_i, theta_loss_i, phi_loss_i = custom_loss(outputs, y, X, separate=True)
                    #custom_loss_2_i, l1_loss_2_i, Ep_loss_i, mass_loss_i = custom_loss_2(outputs, y, X, separate=True)
                    if custom_loss_i and l1_loss_i and theta_loss_i and phi_loss_i:
                        custom_loss_total += custom_loss_i.item()
                        l1_loss_total += l1_loss_i.item()
                        theta_loss_total += theta_loss_i.item()
                        phi_loss_total += phi_loss_i.item()
                    if custom_loss_2_i and l1_loss_2_i and Ep_loss_i and mass_loss_i:
                        custom_loss_2_total += custom_loss_2_i.item()
                        l1_loss_2_total += l1_loss_2_i.item()
                        Ep_loss_2_total += Ep_loss_i.item()
                        mass_loss_2_total += mass_loss_i.item()
                    outputs = outputs * train_out_stds + train_out_means
                predictions.append(outputs.numpy())
                #print('losses:', custom_loss_i.item(), l1_loss_i.item(), theta_loss_i.item())
        predictions = np.concatenate(predictions, axis=0)
        
        print('custom loss:', custom_loss_total/len(test_dataloader))
        print('l1 loss:', l1_loss_total/len(test_dataloader))
        print('theta loss:', theta_loss_total/len(test_dataloader))
        print('phi loss:', phi_loss_total/len(test_dataloader))
        print('custom loss 2:', custom_loss_2_total/len(test_dataloader))
        print('l1 loss 2:', l1_loss_2_total/len(test_dataloader))
        print('Ep loss 2:', Ep_loss_2_total/len(test_dataloader))
        print('mass loss 2:', mass_loss_2_total/len(test_dataloader))


        # store a reco_rand_...px,py,pz columns where we randomly ick between reco_tau.. and reco_alt_tau... solutions
        # it should always choose the same solution for px, py, and pz
        rands = np.random.rand(len(test_df))
        test_df['reco_rand_taup_nu_px'] = np.where(rands < 0.5, test_df['reco_taup_nu_px'], test_df['reco_alt_taup_nu_px'])
        test_df['reco_rand_taup_nu_py'] = np.where(rands < 0.5, test_df['reco_taup_nu_py'], test_df['reco_alt_taup_nu_py'])
        test_df['reco_rand_taup_nu_pz'] = np.where(rands < 0.5, test_df['reco_taup_nu_pz'], test_df['reco_alt_taup_nu_pz'])
        test_df['reco_rand_taun_nu_px'] = np.where(rands < 0.5, test_df['reco_taun_nu_px'], test_df['reco_alt_taun_nu_px'])
        test_df['reco_rand_taun_nu_py'] = np.where(rands < 0.5, test_df['reco_taun_nu_py'], test_df['reco_alt_taun_nu_py'])
        test_df['reco_rand_taun_nu_pz'] = np.where(rands < 0.5, test_df['reco_taun_nu_pz'], test_df['reco_alt_taun_nu_pz'])     

        # get the true values

        true_values = test_df[output_features].values
        simple_predictions = test_df[['reco_taup_nu_px', 'reco_taup_nu_py', 'reco_taup_nu_pz', 'reco_taun_nu_px', 'reco_taun_nu_py', 'reco_taun_nu_pz']].values
        simple_predictions_alt = test_df[['reco_alt_taup_nu_px', 'reco_alt_taup_nu_py', 'reco_alt_taup_nu_pz', 'reco_alt_taun_nu_px', 'reco_alt_taun_nu_py', 'reco_alt_taun_nu_pz']].values
        simple_prediction_d0 = test_df[['reco_d0_taup_nu_px', 'reco_d0_taup_nu_py', 'reco_d0_taup_nu_pz', 'reco_d0_taun_nu_px', 'reco_d0_taun_nu_py', 'reco_d0_taun_nu_pz']].values
        simple_prediction_rand = test_df[['reco_rand_taup_nu_px', 'reco_rand_taup_nu_py', 'reco_rand_taup_nu_pz', 'reco_rand_taun_nu_px', 'reco_rand_taun_nu_py', 'reco_rand_taun_nu_pz']].values
        # calculate the mean squared error
        mse = np.mean((predictions - true_values)**2)
        mse_simple = np.mean((simple_predictions - true_values)**2) 
        mse_simple_alt = np.mean((simple_predictions_alt - true_values)**2)
        mse_simple_d0 = np.mean((simple_prediction_d0 - true_values)**2)
        mse_simple_rand = np.mean((simple_prediction_rand - true_values)**2)
        print(f'Mean squared error: {mse:.4f}')
        print(f'Mean squared error (simple): {mse_simple:.4f}')
        print(f'Mean squared error (simple alt): {mse_simple_alt:.4f}')
        print(f'Mean squared error (simple d0): {mse_simple_d0:.4f}')
        print(f'Mean squared error (simple rand): {mse_simple_rand:.4f}')

        # get non-NN reco values by summing reco_taup_nu.. with reco_taup_vis..
        # and reco_taun_nu.. with reco_taun_vis..
        # get the reco values
        test_df['reco_taup_nu_e'] = np.sqrt(test_df['reco_taup_nu_px']**2 + test_df['reco_taup_nu_py']**2 + test_df['reco_taup_nu_pz']**2)
        test_df['reco_taun_nu_e'] = np.sqrt(test_df['reco_taun_nu_px']**2 + test_df['reco_taun_nu_py']**2 + test_df['reco_taun_nu_pz']**2)
        for x in ['px', 'py', 'pz', 'e']:
            test_df[f'reco_taup_{x}'] = test_df[f'reco_taup_vis_{x}'] + test_df[f'reco_taup_nu_{x}']
            test_df[f'reco_taun_{x}'] = test_df[f'reco_taun_vis_{x}'] + test_df[f'reco_taun_nu_{x}']


        taus_df = compute_tau_four_vectors(test_df, predictions)

        # add tau_df to test_df
        test_df = pd.concat([test_df, taus_df], axis=1)
        # delete repeated columns
        test_df = test_df.loc[:, ~test_df.columns.duplicated()]

        
        # convert to polar coordinates
        tau_p = test_df[['reco_taup_vis_px', 'reco_taup_vis_py', 'reco_taup_vis_pz']].values
        nu_p = test_df[['reco_nn_taup_nu_px', 'reco_nn_taup_nu_py', 'reco_nn_taup_nu_pz']].values
        taup_polar = NeutrinoPolarCoordinates(tau_p)
        taup_nu_p_mag, taup_nu_theta, taup_nu_phi = taup_polar.compute_angles(nu_p)
        tau_p = test_df[['reco_taun_vis_px', 'reco_taun_vis_py', 'reco_taun_vis_pz']].values
        nu_p = test_df[['reco_nn_taun_nu_px', 'reco_nn_taun_nu_py', 'reco_nn_taun_nu_pz']].values
        taun_polar = NeutrinoPolarCoordinates(tau_p)
        taun_nu_p_mag, taun_nu_theta, taun_nu_phi = taun_polar.compute_angles(nu_p)
        
        test_df['reco_nn_taup_nu_p_mag'] = taup_nu_p_mag
        test_df['reco_nn_taup_nu_theta'] = taup_nu_theta
        test_df['reco_nn_taup_nu_phi'] = taup_nu_phi
        test_df['reco_nn_taun_nu_p_mag'] = taun_nu_p_mag
        test_df['reco_nn_taun_nu_theta'] = taun_nu_theta
        test_df['reco_nn_taun_nu_phi'] = taun_nu_phi

        # predict spin sensitive variables, note needs to be done in a loop for now

        test_df['cosn_plus_nn_reco'] = -9999.
        test_df['cosr_plus_nn_reco'] = -9999.
        test_df['cosk_plus_nn_reco'] = -9999.
        test_df['cosn_minus_nn_reco'] = -9999.
        test_df['cosr_minus_nn_reco'] = -9999.
        test_df['cosk_minus_nn_reco'] = -9999.
        test_df['cosTheta_nn_reco'] = -9999.

        for i in range(len(test_df)):
            break
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
        #test_df.to_pickle(f'output/df_test_inc_nn_vars_{args.model_name}.pkl')
        # also store it as a root file
        with uproot.recreate(f'output/df_test_inc_nn_vars_{args.model_name}.root') as root_file:
            root_file['tree'] = test_df 
