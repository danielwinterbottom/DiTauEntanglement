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

class SimpleNN(nn.Module):

    def __init__(self, input_dim, output_dim, n_hidden_layers=4, num_nodes=48):
        super(SimpleNN, self).__init__()
        n_nodes = input_dim

        layers = OrderedDict()
        layers['input'] = nn.Linear(input_dim, num_nodes)
        layers['bn_input'] = nn.BatchNorm1d(num_nodes)
        layers['relu_input'] = nn.ReLU()
        for i in range(n_hidden_layers):
            layers[f'hidden_{i+1}'] = nn.Linear(num_nodes, num_nodes)
            layers[f'hidden_bn_{i+1}'] = nn.BatchNorm1d(num_nodes)
            layers[f'hidden_relu_{i+1}'] = nn.ReLU()
        layers['output'] = nn.Linear(num_nodes, output_dim)

        self.layers = nn.Sequential(layers)
 

    def forward(self, x):
        x = self.layers(x)
        return x

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

def compute_nu_four_vectors(df, y):
    """
    compute the neutrino  4-vectors by subtracting the pi1, pi2, pi3, and pizero from the regressed taus
    """

    # make dataframe with same 8 output columns and same rows as initial dataframe
    taus_df = pd.DataFrame(index=df.index, columns=['reco_taup_nu_px', 'reco_taup_nu_py', 'reco_taup_nu_pz', 'reco_taup_nu_e', 'reco_taun_nu_px', 'reco_taun_nu_py', 'reco_taun_nu_pz', 'reco_taun_nu_e'])
    # get the regressed taus from the y tensor
    taus_df['reco_nn_taup_px'] = y[:,0]
    taus_df['reco_nn_taup_py'] = y[:,1]
    taus_df['reco_nn_taup_pz'] = y[:,2]
    taus_df['reco_nn_taup_e'] = y[:,3]
    taus_df['reco_nn_taun_px'] = y[:,4]
    taus_df['reco_nn_taun_py'] = y[:,5]
    taus_df['reco_nn_taun_pz'] = y[:,6]
    taus_df['reco_nn_taun_e'] = y[:,7]

    #sum the momenta and energy of the pions and pi0s and neutrinos
    for x in ['px', 'py', 'pz', 'e']:
        # visible-only
        taus_df['reco_taup_vis_' + x] = df['reco_taup_pi1_' + x] + df['reco_taup_pi2_' + x] + df['reco_taup_pi3_' + x] + df['reco_taup_pizero1_' + x] + df['reco_taup_pizero2_' + x]
        taus_df['reco_taun_vis_' + x] = df['reco_taun_pi1_' + x] + df['reco_taun_pi2_' + x] + df['reco_taun_pi3_' + x] + df['reco_taun_pizero1_' + x] + df['reco_taun_pizero2_' + x]
        # visible+invisible

        # compute and store the neutrino momenta
        taus_df['reco_nn_taup_nu_' + x] = taus_df['reco_nn_taup_'+x] - df['reco_taup_vis_' + x]
        taus_df['reco_nn_taun_nu_' + x] = taus_df['reco_nn_taun_'+x] - df['reco_taun_vis_' + x]

    # now we can compute the tau mass for convenience
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
    args = argparser.parse_args()

    filename1 = "batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_10000000_Njob_10000/pythia_events.root"
    filename2 = "batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_10000000_Njob_10000/pythia_events_extravars.root"
    treename1 = "tree"
    treename2 = "new_tree"

    standardize = 2 # 0: no standardization, 1: standardize input, 2: standardize input and output
    polar = False
    regress_tau = False
    num_epochs = 10
    n_hidden_layers=6
    num_nodes=300

    # stage 1: prepare dataframes
    if 1 in args.stages:
        PrepareDataframes(filename1, filename2, treename1, treename2, nchunks=20, verbosity=1)


    batch_size = 64
    input_features = [
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
        'reco_Z_px','reco_Z_py','reco_Z_pz','reco_Z_e']

    # use polar features
    if not regress_tau:
        if polar:
            output_features = [
                'taup_nu_p_mag', 'taup_nu_phi', 'taup_nu_theta',
                'taun_nu_p_mag', 'taun_nu_phi', 'taun_nu_theta']
        else:
            output_features = [
                'taup_nu_px','taup_nu_py','taup_nu_pz',
                'taun_nu_px','taun_nu_py','taun_nu_pz']
    else: 
        output_features = [
            'taup_px','taup_py','taup_pz', 'taup_e',
            'taun_px','taun_py','taun_pz', 'taun_e']
    file_paths = [f'output/df_chunk_{i}.pkl' for i in range(2)]
    # only read the input and output features when reading the dfs
    #dataframes = [pd.read_pickle(file)[input_features + output_features + ['reco_taup_vis_px', 'reco_taup_vis_py', 'reco_taup_vis_pz']] for file in file_paths]
    dataframes = [pd.read_pickle(file) for file in file_paths]
    train_df = pd.concat(dataframes[:1], ignore_index=True)
    test_df = pd.concat(dataframes[1:], ignore_index=True)
            
    # stage 2: train the model

    train_dataset = RegressionDataset(train_df, input_features, output_features)

    train_in_means, train_in_stds = train_dataset.get_input_means_stds()
    train_out_means, train_out_stds = train_dataset.get_output_means_stds()

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

        model = SimpleNN(len(input_features), len(output_features), n_hidden_layers=n_hidden_layers, num_nodes=num_nodes)
        #criterion = nn.MSELoss()   
        # use abs error instead
        criterion = nn.L1Loss()   

        def polar_loss(output, target, weights = [1,1,1,1,1,1]):
            #loss = torch.mean((output - target)**2) # implement this equivalent below put taking 3 components seperatly
            #return loss
            loss1 = (output[:, 0] - target[:, 0])**2 # p_mag
            #loss2 = (output[:, 1] - target[:, 1])**2 # phi
            dphi = output[:, 1] - target[:, 1]
            mask = torch.abs(dphi) <= 3 * torch.pi
            dphi[mask] = (dphi[mask] + torch.pi) % (2 * torch.pi) - torch.pi
            loss2 = dphi**2 # phi
            loss3 = (output[:, 2] - target[:, 2])**2 # theta
            loss4 = (output[:, 3] - target[:, 3])**2 # p_mag
            #loss5 = (output[:, 4] - target[:, 4])**2 # phi
            dphi = output[:, 4] - target[:, 4]
            mask = torch.abs(dphi) <= 3 * torch.pi
            dphi[mask] = (dphi[mask] + torch.pi) % (2 * torch.pi) - torch.pi
            loss5 = dphi**2 # phi
            loss6 = (output[:, 5] - target[:, 5])**2 # theta
            # Compute the total loss
            total_loss = torch.mean(loss1*weights[0] + loss2*weights[1] + loss3*weights[2] + loss4*weights[3] + loss5*weights[4] + loss6*weights[5])/sum(weights)
            return total_loss

        def my_loss(output, target):
            #get the loss for each component seperatly
            #taup:
            loss1 = (output[:, 0] - target[:, 0])**2 # px
            loss2 = (output[:, 1] - target[:, 1])**2 # py
            loss3 = (output[:, 2] - target[:, 2])**2 # pz
            #taun:
            loss4 = (output[:, 3] - target[:, 3])**2 # px
            loss5 = (output[:, 4] - target[:, 4])**2 # py
            loss6 = (output[:, 5] - target[:, 5])**2 # pz

            # now get loss for the momentum differences between taup and taun
            loss7 = ((output[:, 0] - output[:, 3]) - (target[:, 0] - target[:, 3]))**2 # Deltapx
            loss8 = ((output[:, 1] - output[:, 4]) - (target[:, 1] - target[:, 4]))**2 # Deltapy
            loss9 = ((output[:, 2] - output[:, 5]) - (target[:, 2] - target[:, 5]))**2 # Deltapz
            # Compute the total loss
            total_loss = torch.mean(loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9)/9
            return total_loss

        def loss_tau_regression(output, target):
            # loss of individual components
            loss1 = (output[:, 0] - target[:, 0])**2 # taup px
            loss2 = (output[:, 1] - target[:, 1])**2 # taup py
            loss3 = (output[:, 2] - target[:, 2])**2 # taup pz
            loss4 = (output[:, 3] - target[:, 3])**2 # taup e
            loss5 = (output[:, 4] - target[:, 4])**2 # taun px
            loss6 = (output[:, 5] - target[:, 5])**2 # taun py
            loss7 = (output[:, 6] - target[:, 6])**2 # taun pz
            loss8 = (output[:, 7] - target[:, 7])**2 # taun e

            # tau mass constraint terms
            tau_mass_exp = 1.777
            taup_mass2 = output[:, 3]**2 - (output[:, 0]**2 + output[:, 1]**2 + output[:, 2]**2)
            loss9 = torch.abs(taup_mass2 - tau_mass_exp**2)
            taun_mass2 = output[:, 7]**2 - (output[:, 4]**2 + output[:, 5]**2 + output[:, 6]**2)
            loss10 = torch.abs(taun_mass2 - tau_mass_exp**2)
            #loss9 = 0
            #loss10 = 0

            # loss for components of the ditau system
            loss12 = ((output[:, 0] + output[:, 4]) - (target[:, 0] + target[:, 4]))**2 # Z px
            loss13 = ((output[:, 1] + output[:, 5]) - (target[:, 1] + target[:, 5]))**2 # Z py
            loss14 = ((output[:, 2] + output[:, 6]) - (target[:, 2] + target[:, 6]))**2 # Z pz
            loss15 = ((output[:, 3] + output[:, 7]) - (target[:, 3] + target[:, 7]))**2 # Z e

            total_loss = torch.mean(loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss12 + loss13 + loss14 + loss15)/14
            return total_loss

        weight_decay = 0
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
        # Define learning rate scheduler
        # learning rate will be reduced by a factor of 10 if the loss does not improve for 5 epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, min_lr=1e-6)

        loss_values = []
        val_loss_values = []
        running_loss_values = []

        print('Training the model...')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            plt.savefig('loss_vs_epoch.pdf')

        for epoch in range(num_epochs):
            #model.train()
            running_loss= 0.0
            for i, (X, y) in enumerate(train_dataloader):
                # move data to GPU
                #print(X.shape)
                X = X.to(device)
                y = y.to(device)
                # move model to GPU
                model.to(device)
                optimizer.zero_grad()
                outputs = model(X)
                if polar: loss = polar_loss(outputs, y, weights)
                elif regress_tau: loss = loss_tau_regression(outputs, y)
                else: loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                lr = scheduler.get_last_lr()
                running_loss += loss.item()

            running_loss /= len(train_dataloader)
            running_loss_values.append(running_loss)
            
            # get the validation loss
            #model.eval()
            with torch.no_grad():
                val_loss = 0.0
                train_loss = 0.0
                for i, (X, y) in enumerate(train_dataloader):
                    X = X.to(device)
                    y = y.to(device)
                    model.to(device)
                    outputs = model(X)
                    if polar: loss = polar_loss(outputs, y, weights)
                    elif regress_tau: loss = loss_tau_regression(outputs, y)
                    else: loss = criterion(outputs, y)
                    train_loss += loss.item()
                train_loss /= len(train_dataloader)
                loss_values.append(train_loss)
                val_loss = 0.0
                for i, (X, y) in enumerate(test_dataloader):
                    X = X.to(device)
                    y = y.to(device)
                    model.to(device)
                    outputs = model(X)
                    if polar: loss = polar_loss(outputs, y, weights)
                    elif regress_tau: loss = loss_tau_regression(outputs, y)
                    else: loss = criterion(outputs, y)
                    val_loss += loss.item()
                val_loss /= len(test_dataloader)
                val_loss_values.append(val_loss)
                
                if early_stopper.early_stop(val_loss):
                    print(f"Early stopping triggered for epoch {epoch+1}")
                    break
                

            print(f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, running_loss: {running_loss:.6f}, lr = {lr[0]:.6f}')
            if epoch > 1: plot_loss(loss_values, val_loss_values, running_loss_values)
            # save model every 10 epochs
            if (epoch+1) % 5 == 0:
                torch.save(model.state_dict(), f'{args.model_name}_epoch_{epoch+1}.pth')
                print(f'Model saved at epoch {epoch+1}')
        
        elapsed_time = time.time() - current_time
        print (f"Training time: {elapsed_time:.2f} seconds")
        # Save the model
        torch.save(model.state_dict(), f'f{args.model_name}.pth')


    # test the model
    if 3 in args.stages:
        # load the pytorch model 'model.pth'
        model = SimpleNN(len(input_features), len(output_features), n_hidden_layers=n_hidden_layers, num_nodes=num_nodes)
        #model.load_state_dict(torch.load('model.pth'))
        model.load_state_dict(torch.load(f'{args.model_name}_epoch_10.pth')) 
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
        with torch.no_grad():
            for i, (X, y) in enumerate(test_dataloader):
                outputs = model(X)
                if standardize == 2:
                    # convert predictions back to original scale
                    outputs = outputs * train_out_stds + train_out_means
                predictions.append(outputs.numpy())
        predictions = np.concatenate(predictions, axis=0)
        
        # get the true values

        true_values = test_df[output_features].values
        # calculate the mean squared error
        mse = np.mean((predictions - true_values)**2)
        print(f'Mean squared error: {mse:.4f}')

        # get non-NN reco values by summing reco_taup_nu.. with reco_taup_vis..
        # and reco_taun_nu.. with reco_taun_vis..
        # get the reco values
        test_df['reco_taup_nu_e'] = np.sqrt(test_df['reco_taup_nu_px']**2 + test_df['reco_taup_nu_py']**2 + test_df['reco_taup_nu_pz']**2)
        test_df['reco_taun_nu_e'] = np.sqrt(test_df['reco_taun_nu_px']**2 + test_df['reco_taun_nu_py']**2 + test_df['reco_taun_nu_pz']**2)
        for x in ['px', 'py', 'pz', 'e']:
            test_df[f'reco_taup_{x}'] = test_df[f'reco_taup_vis_{x}'] + test_df[f'reco_taup_nu_{x}']
            test_df[f'reco_taun_{x}'] = test_df[f'reco_taun_vis_{x}'] + test_df[f'reco_taun_nu_{x}']

        if polar:
            # store the original predictions in the test_df with xx_rotated in the name
            test_df['reco_nn_taup_nu_p_mag'] = predictions[:, 0]
            test_df['reco_nn_taup_nu_phi'] = predictions[:, 1]
            test_df['reco_nn_taup_nu_theta'] = predictions[:, 2]
            test_df['reco_nn_taun_nu_p_mag'] = predictions[:, 3]
            test_df['reco_nn_taun_nu_phi'] = predictions[:, 4]
            test_df['reco_nn_taun_nu_theta'] = predictions[:, 5]
            # transform back to origional non-polar frame using NeutrinoPolarCoordinates
            taup_nu_p_mag = test_df['reco_nn_taup_nu_p_mag'].values
            taup_nu_theta = test_df['reco_nn_taup_nu_theta'].values
            taup_nu_phi = test_df['reco_nn_taup_nu_phi'].values
            taun_nu_p_mag = test_df['reco_nn_taun_nu_p_mag'].values
            taun_nu_theta = test_df['reco_nn_taun_nu_theta'].values
            taun_nu_phi = test_df['reco_nn_taun_nu_phi'].values
            taup_polar = NeutrinoPolarCoordinates(test_df[['reco_taup_vis_px', 'reco_taup_vis_py', 'reco_taup_vis_pz']].values)
            taup_nu_p = taup_polar.reverse_transform(taup_nu_p_mag, taup_nu_theta, taup_nu_phi)
            taun_polar = NeutrinoPolarCoordinates(test_df[['reco_taun_vis_px', 'reco_taun_vis_py', 'reco_taun_vis_pz']].values)
            taun_nu_p = taun_polar.reverse_transform(taun_nu_p_mag, taun_nu_theta, taun_nu_phi)
            # redefine predictions to use the rotated values
            predictions = np.concatenate([taup_nu_p[:, 0].reshape(-1, 1), taup_nu_p[:, 1].reshape(-1, 1), taup_nu_p[:, 2].reshape(-1, 1),
                                          taun_nu_p[:, 0].reshape(-1, 1), taun_nu_p[:, 1].reshape(-1, 1), taun_nu_p[:, 2].reshape(-1, 1)], axis=1)

        if not regress_tau: taus_df = compute_tau_four_vectors(test_df, predictions)
        else: taus_df = compute_nu_four_vectors(test_df, predictions)
        

        # add tau_df to test_df
        test_df = pd.concat([test_df, taus_df], axis=1)
        # delete repeated columns
        test_df = test_df.loc[:, ~test_df.columns.duplicated()]

        if not polar: # if not polar then convert and store polar coordinates as well
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
        test_df.to_pickle('output/df_test_inc_nn_vars.pkl')
        # also store it as a root file
        with uproot.recreate('output/df_test_inc_nn_vars.root') as root_file:
            root_file['tree'] = test_df 
