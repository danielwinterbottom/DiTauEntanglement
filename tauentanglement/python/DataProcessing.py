import pandas as pd
import torch
from torch.utils.data import Dataset
import uproot
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os
from tauentanglement.utils.coordinate_conversions import ConvertToPolar, ConvertToOrthonormalNRK, ConvertNRKToAngular

class RegressionDataset(Dataset):
    def __init__(self, dataframe, input_features, output_features,
                 input_mean=None, input_std=None,
                 output_mean=None, output_std=None,
                 normalize_inputs=True, normalize_outputs=False, eps=1e-8,
                 weights=None):
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
        weights : torch.Tensor or None
            Optional per-event training weight, shape [N]. If given, __getitem__
            returns (X, y, w) triples instead of (X, y) pairs -- see
            get_train_val_test_datasets/train_model for how these are combined
            from TauSpinner weight columns and used in the loss.
        """
        X = torch.tensor(dataframe[input_features].values, dtype=torch.float32)
        y = torch.tensor(dataframe[output_features].values, dtype=torch.float32)

        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.eps = eps
        self.weights = weights

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
        if self.weights is not None:
            return self.X[idx], self.y[idx], self.weights[idx]
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


def _output_path_for_coordinates(config, key):
    if config['coordinates'] == 'polar':
        fname = 'full_polar_dataframe.parquet'
    elif config['coordinates'] == 'onorm':
        fname = 'full_onorm_dataframe.parquet'
    elif config['coordinates'] == 'onorm_angular':
        fname = 'full_onorm_angular_dataframe.parquet'
    else:
        fname = 'full_dataframe.parquet'
    return os.path.join(config['output_dir'], key, fname)


def _process_chunk(df, config, collider, use_reco, prefix, charged_name, has_ts_hh, has_undecayed, has_reco_flags):
    """Everything convert_root_to_parquet used to do to the whole dataframe at
    once (row filtering, decay-mode boolean flags, collider-specific columns,
    coordinate conversion) -- applied per-chunk so convert_root_to_parquet can
    stream through the tree instead of holding it all in memory at once."""
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
    if has_reco_flags:
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

    if config['coordinates'] == 'polar':  # option to convert to polar coordinates
        df = ConvertToPolar(df, 'taup_nu_p')
        df = ConvertToPolar(df, 'taun_nu_p')

    elif config['coordinates'] in ('onorm', 'onorm_angular'):  # option to convert to orthonormal basis
        df = ConvertToOrthonormalNRK(
            df, prefix_to_convert='taup_nu_',
            charged_prefix=f"{prefix}taup_{charged_name}_", pi0_prefix=f"{prefix}taup_pizero1_",
            out_prefix=None, drop_xyz=False, keep_basis=True,
        )
        df = ConvertToOrthonormalNRK(
            df, prefix_to_convert='taun_nu_',
            charged_prefix=f"{prefix}taun_{charged_name}_", pi0_prefix=f"{prefix}taun_pizero1_",
            out_prefix=None, drop_xyz=False, keep_basis=True,
        )

        # polarimetric vectors + full tau momenta (polvec training outputs), projected
        # onto the same per-tau visible-momentum (n,r,k) basis as the neutrino above.
        # Basis vectors are already stored via the taup_nu_/taun_nu_ conversions above,
        # so keep_basis=False here to avoid redundant duplicate columns.
        if has_ts_hh:
            df = ConvertToOrthonormalNRK(
                df, prefix_to_convert='ts_hh_taup_',
                charged_prefix=f"{prefix}taup_{charged_name}_", pi0_prefix=f"{prefix}taup_pizero1_",
                out_prefix=None, drop_xyz=False, keep_basis=False, suffixes=("x", "y", "z"),
            )
            df = ConvertToOrthonormalNRK(
                df, prefix_to_convert='ts_hh_taun_',
                charged_prefix=f"{prefix}taun_{charged_name}_", pi0_prefix=f"{prefix}taun_pizero1_",
                out_prefix=None, drop_xyz=False, keep_basis=False, suffixes=("x", "y", "z"),
            )

            # ts_hh_taup/taun are (nominally) unit vectors, so their (n,r,k)
            # triplet carries a redundant degree of freedom -- collapse to the
            # 2 genuine ones (costheta, phi) for the onorm_angular training
            # target (see ConvertNRKToAngular for why this matters for
            # flow-based training). drop_nrk=False keeps the raw (n,r,k) too
            # (not used for training, same treatment as x,y,z above) so
            # downstream physics validation (e.g. the entanglement/spin-density
            # variables in evaluate_polvec.py) can use the true, unnormalized
            # projections rather than the direction-only training target.
            if config['coordinates'] == 'onorm_angular':
                df = ConvertNRKToAngular(df, prefix='ts_hh_taup_', drop_nrk=False)
                df = ConvertNRKToAngular(df, prefix='ts_hh_taun_', drop_nrk=False)

        if has_undecayed:
            df = ConvertToOrthonormalNRK(
                df, prefix_to_convert='undecayed_taup_',
                charged_prefix=f"{prefix}taup_{charged_name}_", pi0_prefix=f"{prefix}taup_pizero1_",
                out_prefix=None, drop_xyz=False, keep_basis=False,
            )
            df = ConvertToOrthonormalNRK(
                df, prefix_to_convert='undecayed_taun_',
                charged_prefix=f"{prefix}taun_{charged_name}_", pi0_prefix=f"{prefix}taun_pizero1_",
                out_prefix=None, drop_xyz=False, keep_basis=False,
            )

    return df.reset_index(drop=True)


def convert_root_to_parquet(input_file_name, key, config, collider, use_reco=True, chunk_size=None):
    """
    Streams the ROOT tree in chunks (uproot.iterate) rather than loading the
    whole thing into one pandas DataFrame -- for a many-GB file, reading it in
    one go via tree.arrays() is a hard floor on peak memory no matter how well
    process-level isolation between datasets is done (see prepare_inputs.py).
    Each chunk goes through the exact same processing as before and is appended
    directly to the output parquet file via a streaming ParquetWriter.

    chunk_size : int or None
        Number of events to read/process per chunk. Defaults to
        config['conversion_chunk_size'] if set, else 1,000,000.
    """

    print("Converting from ROOT to Parquet...")
    if chunk_size is None:
        chunk_size = config.get('conversion_chunk_size', 1_000_000)

    output_path = _output_path_for_coordinates(config, key)

    # Optionally load TauSpinner weights, polarimetric vectors, and undecayed tau
    # 4-vectors if they were produced by run_delphes.py --tauspinner.
    _AXES = ('n', 'r', 'k')
    _optional_cols = (
        [f'tauspinner_wt_alpha{a}' for a in [0, 45, 90]] +
        [f'wt_hp_{a}' for a in _AXES] +
        [f'wt_hm_{a}' for a in _AXES] +
        [f'wt_hp_{a}_hm_{b}' for a in _AXES for b in _AXES] +
        ['ts_hh_taup_x', 'ts_hh_taup_y', 'ts_hh_taup_z',
         'ts_hh_taun_x', 'ts_hh_taun_y', 'ts_hh_taun_z'] +
        ['taup_px', 'taup_py', 'taup_pz', 'taup_e',
         'taun_px', 'taun_py', 'taun_pz', 'taun_e']
    )

    writer = None
    n_events_total = 0
    columns_written = None

    # Explicitly close the uproot file handle once we're done reading from it.
    # Without this, the file's internal read/decompression buffers can linger
    # until Python's GC gets around to collecting the tree object, which is
    # what made memory usage balloon when processing multiple input files in
    # one process (each file's buffers stuck around on top of the previous
    # one's instead of being freed immediately).
    with uproot.open(input_file_name) as f:
        tree = f[config['tree_name']]
        tree_keys = set(tree.keys())

        main_cols = list(config['Features']['dataframe_variables'])
        optional_cols = [c for c in _optional_cols if c in tree_keys]
        all_cols = list(dict.fromkeys(main_cols + optional_cols))  # dedupe, keep order

        # these are all fixed properties of the file/config, not of any one
        # chunk, so determine them once up front from the branch list
        has_ts_hh = 'ts_hh_taup_x' in optional_cols
        has_undecayed = 'taup_px' in optional_cols
        charged_name = "charged" if 'taup_charged_e' in all_cols else "pi1"
        prefix = "reco_" if use_reco else ""
        has_reco_flags = 'reco_taup_npizero' in all_cols

        rename = {f'{tau}_{comp}': f'undecayed_{tau}_{comp}'
                  for tau in ['taup', 'taun']
                  for comp in ['px', 'py', 'pz', 'e']
                  if f'{tau}_{comp}' in optional_cols}

        if config['coordinates'] == 'onorm':
            print("Converting to orthonormal basis" + (f" using prefix {prefix} for visible tau vectors" if use_reco else " using gen-level visible tau vectors"))
            print(f"Using {charged_name} as the charged component for the onorm basis")

        print(f">> Reading {len(all_cols)} branches in chunks of {chunk_size} events...")
        n_events_read = 0
        for chunk in tree.iterate(all_cols, step_size=chunk_size, library="pd"):
            n_events_read += len(chunk)
            if rename:
                chunk = chunk.rename(columns=rename)

            chunk = _process_chunk(
                chunk, config, collider, use_reco, prefix, charged_name,
                has_ts_hh, has_undecayed, has_reco_flags,
            )
            if len(chunk) > 0:
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                    columns_written = chunk.columns.tolist()
                else:
                    table = table.cast(writer.schema)  # guard against dtype drift between chunks
                writer.write_table(table)
                n_events_total += len(chunk)

            # \r + no newline overwrites the previous status line instead of
            # scrolling the terminal with one line per chunk
            print(f"\r  ...read {n_events_read} events, kept {n_events_total} after cuts"
                  + " " * 10, end="", flush=True)

    print()  # move off the status line before the final summary
    if writer is not None:
        writer.close()

    print(f">> Dataframe {key} converted and saved to {output_path} ({n_events_total} events)")
    if columns_written is not None:
        print('Columns in the saved dataframe:', columns_written)

def convert_semileptonic_df(df):
    # restructure semileptonic dataframe to give the leptonic tau as "tau1" and the hadronic one as "tau2"
    # instead of taup and taun
    # check if taup is leptonic and taun is hadronic and if so relabel taup->tau1 and taun->tau2

    df_lep_tau1 = df[((df['taup_isleptonic'] == 1) & (df['taun_ishadronic'] == 1))].copy()
    df_lep_tau1 = df_lep_tau1.rename(columns=lambda x: x.replace('taup_', 'tau1_').replace('taun_', 'tau2_'))
    # store the charge of the leptonic tau in a separate column
    df_lep_tau1['tau1_charge'] = np.ones(len(df_lep_tau1))
    df_lep_tau1['tau2_charge'] = -1*np.ones(len(df_lep_tau1))

    # check if taup is hadronic and taun is leptonic and if so relabel taun->tau1 and taup->tau2
    df_lep_tau2 = df[((df['taup_ishadronic'] == 1) & (df['taun_isleptonic'] == 1))].copy()
    df_lep_tau2 = df_lep_tau2.rename(columns=lambda x: x.replace('taun_', 'tau1_').replace('taup_', 'tau2_'))
    # store the charge of the taus in a separate column
    df_lep_tau2['tau2_charge'] = -1*np.ones(len(df_lep_tau2))
    df_lep_tau2['tau1_charge'] = np.ones(len(df_lep_tau2))

    # concatenate the two dataframes
    df_out = pd.concat([df_lep_tau1, df_lep_tau2], ignore_index=True)
    # shuffle the dataframe
    df_out = df_out.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_out


def get_train_val_test_datasets(keys, config, shuffle=True, load_existing=False):

    leptonic_mode = config.get('leptonic_mode', -1)  # default to -1 if not specified i.e no selection based on whether tau is leptonic is applied
    one_prong_only = config.get('one_prong_only', False)  # option to select only one prong decays for training
    match_n_prongs = config.get('match_n_prongs', False)  # option to select only one prong decays for training
    inc_three_prongs = config.get('inc_three_prongs', False)  # option to select only events with at least 1 3-prong tau
    transformer = config.get('use_transformer', False)  # option to use transformer for conditioning

    # check if key is not a list, if not add it to a list
    if not isinstance(keys, list):
        keys = [keys]

    input_features = config['Features']['input_features']
    if config['inc_reco_taus']:
        input_features += config['Features']['input_reco_tau_features']

    output_features = config['Features']['output_features'][config['coordinates']]

    # optional per-event training weight: sum of one or more TauSpinner weight
    # columns (e.g. ['tauspinner_wt_alpha0', 'tauspinner_wt_alpha90'] to train on
    # an even mixture of CP-even and CP-odd weighted events). None/absent = no
    # weighting (all events weight 1, same as before this option existed).
    training_weight_columns = config.get('training_weight_columns', None)
    keep_columns = input_features + output_features
    if training_weight_columns:
        keep_columns = keep_columns + [c for c in training_weight_columns if c not in keep_columns]

    train_df = None
    val_df = None


    extra_name = ''
    if config['coordinates'] == 'standard':
        extra_name += 'cartesian'
    if leptonic_mode>=0:
        extra_name += f"_leptonic_mode_{leptonic_mode}"
    if inc_three_prongs:
        extra_name += "_inc_three_prongs"
    if transformer:
        print("Transformer dataset required")
        extra_name += "_transformer"

    for k in keys:

        # dataset names
        train_df_path = os.path.join(config['output_dir'], k, f'train_dataframe_{extra_name}.parquet')
        val_df_path = os.path.join(config['output_dir'], k, f'val_dataframe_{extra_name}.parquet')
        test_df_path = os.path.join(config['output_dir'], k, f'test_dataframe_{extra_name}.parquet')

        if load_existing and os.path.exists(train_df_path) and os.path.exists(val_df_path) and os.path.exists(test_df_path):

            print(">> WARNING: Loading pre-existing train and test dataframes from disk instead of creating new ones. Make sure this is intentional to avoid accidentally using wrong data for training/testing!")
            val_df_ = pd.read_parquet(val_df_path)
            train_df_ = pd.read_parquet(train_df_path)
            test_df_ = pd.read_parquet(test_df_path)

        else:
            if config['coordinates'] == 'standard':
                df = pd.read_parquet(os.path.join(config['output_dir'], k, 'full_onorm_dataframe.parquet'))
            elif config['coordinates'] == 'polar':
                df = pd.read_parquet(os.path.join(config['output_dir'], k, 'full_polar_dataframe.parquet'))
            elif config['coordinates'] == 'onorm':
                df = pd.read_parquet(os.path.join(config['output_dir'], k, 'full_onorm_dataframe.parquet'))
            elif config['coordinates'] == 'onorm_angular':
                df = pd.read_parquet(os.path.join(config['output_dir'], k, 'full_onorm_angular_dataframe.parquet'))
            # add a column to identify the dataset
            df['dataset'] = k

            if match_n_prongs:
                # only select events where number of pions and number of elecron and muons match the gen-values
                df = df[(df['taup_npi'] == df['reco_taup_npi']) & (df['taun_npi'] == df['reco_taun_npi'])]
                df = df[(df['taup_nmu'] == df['reco_taup_nmu']) & (df['taun_nmu'] == df['reco_taun_nmu'])]
                df = df[(df['taup_nele'] == df['reco_taup_nele']) & (df['taun_nele'] == df['reco_taun_nele'])]

            if leptonic_mode == 0:
                # select cases where both taus are hadronic
                df = df[(df['taup_nmu'] == 0) & (df['taup_nele'] == 0) & (df['taun_nmu'] == 0) & (df['taun_nele'] == 0)]

                #apply reco cuts as well
                df = df[(df['reco_taup_nmu'] == 0) & (df['reco_taup_nele'] == 0) & (df['reco_taun_nmu'] == 0) & (df['reco_taun_nele'] == 0)]

                if one_prong_only: # only train on 1-prong events (require both truth and reco level be 1-prong)
                    df = df[(df['taup_npi'] == 1) & (df['taun_npi'] == 1)]
                    df = df[(df['reco_taup_npi'] == 1) & (df['reco_taun_npi'] == 1)]

                if inc_three_prongs: # only train on events with at least 1 3-prong tau
                    df = df[(df['taup_npi'] > 1) | (df['taun_npi'] > 1)]
                    df = df[(df['reco_taup_npi'] > 1) | (df['reco_taun_npi'] > 1)]

            elif leptonic_mode == 1:
            # select cases where one tau is leptonic and one is hadronic
                df = df[((df['taup_nmu'] + df['taup_nele']) > 0) & ((df['taun_nmu'] + df['taun_nele']) == 0) |
                        ((df['taup_nmu'] + df['taup_nele']) == 0) & ((df['taun_nmu'] + df['taun_nele']) > 0)]

                # apply reco cuts as well
                df = df[((df['reco_taup_nmu'] + df['reco_taup_nele']) > 0) & ((df['reco_taun_nmu'] + df['reco_taun_nele']) == 0) |
                        ((df['reco_taup_nmu'] + df['reco_taup_nele']) == 0) & ((df['reco_taun_nmu'] + df['reco_taun_nele']) > 0)]

                # restructure the dataframe so that the leptonic tau is always tau1 and the hadronic tau is always tau2
                df = convert_semileptonic_df(df)

            elif leptonic_mode == 2:
                # select cases where both taus are leptonic
                df = df[(df['taup_nmu'] + df['taup_nele'] > 0) & (df['taun_nmu'] + df['taun_nele'] > 0)]

                # apply reco cuts as well
                df = df[(df['reco_taup_nmu'] + df['reco_taup_nele'] > 0) & (df['reco_taun_nmu'] + df['reco_taun_nele'] > 0)]

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

            val_df_.to_parquet(val_df_path)
            test_df_.to_parquet(test_df_path)
            train_df_.to_parquet(train_df_path)
            print(f">> Train, validation and test dataframes for {k} saved.")
            print(f">> Train dataframe size: {len(train_df_)}, Validation dataframe size: {len(val_df_)}, Test dataframe size: {len(test_df_)}")

            # after saving we can delet the test_df as we dont use it during the training
            del test_df_

        # to save memory usage we also drop anything that isn't an input out output feature from the dataframes
        #print size of dataframe in GB before dropping columns
        print(f">> Size of train dataframe before dropping columns: {train_df_.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        print(f">> Size of val dataframe before dropping columns: {val_df_.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        train_df_ = train_df_[keep_columns]
        val_df_ = val_df_[keep_columns]
        # print size of dataframe in GB after dropping columns
        print(f">> Size of train dataframe after dropping columns: {train_df_.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        print(f">> Size of val dataframe after dropping columns: {val_df_.memory_usage(deep=True).sum() / 1e9:.2f} GB")

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

    # optional per-event training weight = sum of the configured TauSpinner weight columns
    if training_weight_columns:
        train_weights = torch.tensor(train_df[training_weight_columns].sum(axis=1).values, dtype=torch.float32)
        val_weights = torch.tensor(val_df[training_weight_columns].sum(axis=1).values, dtype=torch.float32)
        print(f">> Using training weight = sum of columns {training_weight_columns} "
              f"(train mean={train_weights.mean().item():.4f}, val mean={val_weights.mean().item():.4f})")
    else:
        train_weights = None
        val_weights = None

    # define datasets and normalize inputs and outputs
    train_dataset = RegressionDataset(train_df, input_features, output_features, normalize_inputs=True, normalize_outputs=True,
                                       weights=train_weights)
    del train_df
    in_mean, in_std = train_dataset.input_mean, train_dataset.input_std
    out_mean, out_std = train_dataset.output_mean, train_dataset.output_std
    val_dataset = RegressionDataset(val_df, input_features, output_features, normalize_inputs=True, normalize_outputs=True,
                                      input_mean=in_mean, input_std=in_std, output_mean=out_mean, output_std=out_std,
                                      weights=val_weights)
    del val_df

    return train_dataset, val_dataset, input_features, output_features

def get_test_dataset(config, norm_data, oneprong=False, threeprong=False):

    if oneprong and threeprong:
        raise ValueError("Cannot select both oneprong and threeprong options simultaneously. Please choose one or neither.")

    test_df = pd.read_parquet(config['test_dataset'])

    if oneprong:
        test_df = test_df[(test_df['reco_taup_npi'] == 1) & (test_df['reco_taun_npi'] == 1)]

    if threeprong:
        # require either tau to be 3-prong
        test_df = test_df[(test_df['reco_taup_npi'] == 3) | (test_df['reco_taun_npi'] == 3)]

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
