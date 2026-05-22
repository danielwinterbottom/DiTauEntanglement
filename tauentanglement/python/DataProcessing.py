import pandas as pd
import torch
from torch.utils.data import Dataset
import uproot
import numpy as np
import os
from tauentanglement.utils.coordinate_conversions import ConvertToPolar, ConvertToOrthonormalNRK

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
        device = y_norm.device
        return y_norm * self.output_std.to(device) + self.output_mean.to(device)


class MorphDataset(Dataset):
    def __init__(self, dataframe, input_features, output_features, context_features,
                 input_mean=None, input_std=None,
                 output_mean=None, output_std=None,
                 context_mean=None, context_std=None,
                 normalize_inputs=False, normalize_outputs=False, normalize_context=True, eps=1e-8):
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
        context_features : list[str]
            Names of context columns.
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
        c = torch.tensor(dataframe[context_features].values, dtype=torch.float32)

        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.eps = eps

        # ---- Context normalization ----
        if normalize_context:
            if context_mean is None or context_std is None:
                self.context_mean = c.mean(dim=0, keepdim=True)
                self.context_std = c.std(dim=0, keepdim=True).clamp_min(eps)
            else:
                self.context_mean = context_mean
                self.context_std = context_std.clamp_min(eps)

            c = (c - self.context_mean) / self.context_std
        else:
            self.context_mean = torch.zeros(c.shape[1])
            self.context_std = torch.ones(c.shape[1])

        ## ---- Input normalization ----
        #if normalize_inputs:
        #    if input_mean is None or input_std is None:
        #        self.input_mean = X.mean(dim=0, keepdim=True)
        #        self.input_std = X.std(dim=0, keepdim=True).clamp_min(eps)
        #    else:
        #        self.input_mean = input_mean
        #        self.input_std = input_std.clamp_min(eps)
#
        #    X = (X - self.input_mean) / self.input_std
        #else:
        #    self.input_mean = torch.zeros(X.shape[1])
        #    self.input_std = torch.ones(X.shape[1])

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

        # normalize inputs using same means as if used for outputs
        if normalize_outputs:
            X = (X - self.output_mean) / self.output_std
            #TODO: trying to understand is these really need to be standardized in consistent way with the outputs
        self.input_mean = self.output_mean
        self.input_std = self.output_std


        self.c = c
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.c[idx]

    def destandardize_outputs(self, y_norm):
        """
        Convert standardized outputs back to physical units.
        """
        device = y_norm.device
        return y_norm * self.output_std.to(device) + self.output_mean.to(device)


def convert_root_to_parquet(input_file_name, key, config, collider, use_reco=True):

    print("Converting from ROOT to Parquet...")

    tree = uproot.open(input_file_name)[config['tree_name']]
    # print(f">> Opened tree in file {input_file_name}")

    df = tree.arrays(config['Features']['dataframe_variables'], library="pd")

    ## select only DM 0 and 1
    # removing this since we want to train on other decay modes as well
    #df = df[(df['taup_npi'] == 1) & (df['taup_npizero'] < 2)]
    #df = df[(df['taun_npi'] == 1) & (df['taun_npizero'] < 2)]

    if use_reco:
        # require at least 1 pi, elec or muon for each tau
        df = df[(df['reco_taup_npi']+df['reco_taup_nmu']+df['reco_taup_nele'] > 0) & (df['reco_taun_npi']+df['reco_taun_nmu']+df['reco_taun_nele'] > 0)]

        # apply vis pT cut as well
        df = df[(df['reco_taup_vis_pT'] > 20) & (df['reco_taun_vis_pT'] > 20)]

    # now we add some booleans to the dataframe with information about the type of the tau decay 
    df['taup_haspizero'] = (df['taup_npizero'] > 0).astype(float)
    df['taun_haspizero'] = (df['taun_npizero'] > 0).astype(float)
    df['taup_is3prong'] = (df['taup_npi'] > 1).astype(float)
    df['taun_is3prong'] = (df['taun_npi'] > 1).astype(float)
    df['taup_isleptonic'] = ((df['taup_nmu'] + df['taup_nele']) > 0).astype(float)
    df['taun_isleptonic'] = ((df['taun_nmu'] + df['taun_nele']) > 0).astype(float)
    df['taup_ismuon'] = (df['taup_nmu'] > 0).astype(float)
    df['taun_ismuon'] = (df['taun_nmu'] > 0).astype(float)
    df['taup_iselectron'] = (df['taup_nele'] > 0).astype(float)
    df['taun_iselectron'] = (df['taun_nele'] > 0).astype(float)
    df['taup_ishadronic'] = (df['taup_npi']> 0).astype(float)
    df['taun_ishadronic'] = (df['taun_npi']> 0).astype(float)

    # do the same for reco if it exists
    if 'reco_taup_npizero' in df.columns:
        df['reco_taup_haspizero'] = (df['reco_taup_npizero'] > 0).astype(float)
        df['reco_taun_haspizero'] = (df['reco_taun_npizero'] > 0).astype(float)
        df['reco_taup_is3prong'] = (df['reco_taup_npi'] > 1).astype(float)
        df['reco_taun_is3prong'] = (df['reco_taun_npi'] > 1).astype(float)
        df['reco_taup_isleptonic'] = ((df['reco_taup_nmu'] + df['reco_taup_nele']) > 0).astype(float)
        df['reco_taun_isleptonic'] = ((df['reco_taun_nmu'] + df['reco_taun_nele']) > 0).astype(float)
        df['reco_taup_ismuon'] = (df['reco_taup_nmu'] > 0).astype(float)
        df['reco_taun_ismuon'] = (df['reco_taun_nmu'] > 0).astype(float)
        df['reco_taup_iselectron'] = (df['reco_taup_nele'] > 0).astype(float)
        df['reco_taun_iselectron'] = (df['reco_taun_nele'] > 0).astype(float)
        df['reco_taup_ishadronic'] = (df['reco_taup_npi']> 0).astype(float)
        df['reco_taun_ishadronic'] = (df['reco_taun_npi']> 0).astype(float)

    if collider == 'LEP':
        # also apply a reco_mass cut to select events close to the Z pole with little boost
        df = df[(df['reco_mass'] > 91)]
        df = df.drop(columns=['reco_mass'])  # now remove the reco_mass column

        # compute the d_min vector by subtracting the 2 impact parameters
        df['dmin_x'] = df['reco_taup_pi1_ipx'] - df['reco_taun_pi1_ipx']
        df['dmin_y'] = df['reco_taup_pi1_ipy'] - df['reco_taun_pi1_ipy']
        df['dmin_z'] = df['reco_taup_pi1_ipz'] - df['reco_taun_pi1_ipz']

    if collider == 'LHC': # TODO delete eventaually as this should be stored correctly now
        # recompute met_px and met_py from neutrinos as this wasn't stored properly
        df['met_px'] = df['taup_nu_px'] + df['taun_nu_px']
        df['met_py'] = df['taup_nu_py'] + df['taun_nu_py']

    if config['coordinates'] == 'polar':  # option to convert to polar coordinates

        # convert outputs
        df = ConvertToPolar(df, 'taup_nu_p')
        df = ConvertToPolar(df, 'taun_nu_p')

        ## uncomment to convert inputs as well
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

        df.to_parquet(os.path.join(config['output_dir'], key, f'full_polar_dataframe.parquet'))

    elif config['coordinates'] == 'onorm':  # option to convert to orthonormal basis

        if not use_reco: prefix= ""
        else: prefix = "reco_"

        print ("Converting to orthonormal basis"+f" using prefix {prefix} for visible tau vectors" if use_reco else " using gen-level visible tau vectors")

        # convert outputs
        # check if charged exists on dataframe and use this if so, if not use pi1
        if 'taup_charged_e' in df.columns:
            charged_name = "charged"
        else:            charged_name = "pi1"

        df = ConvertToOrthonormalNRK(
            df,
            prefix_to_convert='taup_nu_',
            charged_prefix=f"{prefix}taup_{charged_name}_",
            pi0_prefix=f"{prefix}taup_pizero1_",
            out_prefix=None,
            drop_xyz=False,
            keep_basis=True,
        )
        df = ConvertToOrthonormalNRK(
            df,
            prefix_to_convert='taun_nu_',
            charged_prefix=f"{prefix}taun_{charged_name}_",
            pi0_prefix=f"{prefix}taun_pizero1_",
            out_prefix=None,
            drop_xyz=False,
            keep_basis=True,
        )

        df.to_parquet(os.path.join(config['output_dir'], key, f'full_onorm_dataframe.parquet'))

    else:  # no conversion
        df.to_parquet(os.path.join(config['output_dir'], key, f'full_dataframe.parquet'))

    print(f">> Dataframe {key} converted and saved to {config['output_dir']}/{key}")
    print('Columns in the saved dataframe:', df.columns.tolist())
    return df

def get_train_val_test_datasets(keys, config, shuffle=True):

    # check if key is not a list, if not add it to a list
    if not isinstance(keys, list):
        keys = [keys]

    input_features = config['Features']['input_features']
    if config['inc_reco_taus']:
        input_features += config['Features']['input_reco_tau_features']

    output_features = config['Features']['output_features'][config['coordinates']]

    train_df = None
    val_df = None

    for k in keys:
        if config['coordinates'] == 'standard':
            df = pd.read_parquet(os.path.join(config['output_dir'], k, 'full_dataframe.parquet'))
        elif config['coordinates'] == 'polar':
            df = pd.read_parquet(os.path.join(config['output_dir'], k, 'full_polar_dataframe.parquet'))
        elif config['coordinates'] == 'onorm':
            df = pd.read_parquet(os.path.join(config['output_dir'], k, 'full_onorm_dataframe.parquet'))
        # add a column to identify the dataset
        df['dataset'] = k

        train_size = int(config['train_fraction'] * len(df))
        val_size = int(config['val_fraction'] * len(df))

        # check if test_fraction is defined, if not use the rest of the data for testing
        if 'test_fraction' in config:
            test_size = int(config['test_fraction'] * len(df))
        else:
            test_size = len(df) - train_size - val_size

        train_df_ = df.iloc[:train_size]
        val_df_ = df.iloc[train_size:train_size + val_size]

        if config['full_dataframe_testing']:
            test_df_ = df.copy()
        else:
            test_df_ = df.iloc[train_size + val_size:train_size + val_size + test_size]
        del df

        val_df_.to_parquet(os.path.join(config['output_dir'], k, f'val_dataframe.parquet'))
        test_df_.to_parquet(os.path.join(config['output_dir'], k, f'test_dataframe.parquet'))
        train_df_.to_parquet(os.path.join(config['output_dir'], k, f'train_dataframe.parquet'))
        print(f">> Train, validation and test dataframes for {k} saved.")
        print(f">> Train dataframe size: {len(train_df_)}, Validation dataframe size: {len(val_df_)}, Test dataframe size: {len(test_df_)}")

        if train_df is None:
            train_df = train_df_
        else:
            train_df = pd.concat([train_df, train_df_], ignore_index=True)
        if val_df is None:
            val_df = val_df_
        else:
            val_df = pd.concat([val_df, val_df_], ignore_index=True)

    if shuffle:
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)     

    print(f"Number of events in training dataframe: {len(train_df)}, validation dataframe: {len(val_df)}")
    print('Columns in training dataframe:', train_df.columns.tolist())
    print('>> Number of input features:', len(input_features))
    print('Input features:', input_features)
    print('>> Number of output features:', len(output_features))
    print('Output features:', output_features)

    # define datasets and normalize inputs and outputs
    train_dataset = RegressionDataset(train_df, input_features, output_features, normalize_inputs=True, normalize_outputs=True)
    del train_df
    in_mean, in_std = train_dataset.input_mean, train_dataset.input_std
    out_mean, out_std = train_dataset.output_mean, train_dataset.output_std
    val_dataset = RegressionDataset(val_df, input_features, output_features, normalize_inputs=True, normalize_outputs=True,
                                      input_mean=in_mean, input_std=in_std, output_mean=out_mean, output_std=out_std)
    del val_df

    return train_dataset, val_dataset, input_features, output_features

def get_test_dataset(config, norm_data):

    test_df = pd.read_parquet(config['test_dataset'])

    input_features = config['Features']['input_features']
    if config['inc_reco_taus']:
        input_features += config['Features']['input_reco_tau_features']

    output_features = config['Features']['output_features'][config['coordinates']]

    #print the names of all the columns and information on the number of events in the dataframe
    print('>> Number of events in test dataframe:', len(test_df))
    print('>> Number of input features:', len(input_features))

    in_mean  = torch.from_numpy(norm_data['input_mean'])
    in_std   = torch.from_numpy(norm_data['input_std'])
    out_mean = torch.from_numpy(norm_data['output_mean'])
    out_std  = torch.from_numpy(norm_data['output_std'])
    test_dataset = RegressionDataset(test_df, input_features, output_features, normalize_inputs=True, normalize_outputs=True,
                                      input_mean=in_mean, input_std=in_std, output_mean=out_mean, output_std=out_std)

    return test_dataset, test_df, input_features, output_features

