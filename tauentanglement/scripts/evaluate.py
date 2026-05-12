import torch
import pandas as pd
import argparse
import yaml
import os
import uproot
import numpy as np
from tauentanglement.python.NN_Tools import load_model
from tauentanglement.python.DataProcessing import get_test_dataset
from tauentanglement.utils.coordinate_conversions import ConvertPredictionsToCartesian, ConvertFromOrthonormalNRK_Predictions
from tauentanglement.python.Evaluation_Tools import flow_map_predict, compute_spin_vars, save_sampled_pdfs
from tauentanglement.utils.kinematic_helpers import compute_spin_density_vars

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', help='path to the configuration file', type=str, default='tauentanglement/config/LEP.yaml', required=True)
    argparser.add_argument('--useMLP', help='whether to use a simple MLP instead of a normalizing flow', action='store_true')
    argparser.add_argument('--useCPU', help='whether to use CPU only for evaluation', action='store_true')
    args = argparser.parse_args()

    # load config
    config_file = args.config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']
    nn_config = config['SetupNN']
    use_reco = data_config['use_reco']

    if use_reco: prefix = 'reco_'
    else: prefix = ''

    # set gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.useCPU else "cpu")

    # load test dataset
    if len(data_config['datasets']) > 1:
        raise NotImplementedError("Currently only supports one dataset at a time.")
    else:
        dataset=data_config['datasets'][0]
    test_dataset, test_df, input_features, output_features = get_test_dataset(dataset, data_config)

    # load model
    output_dir = f"outputs_{nn_config['model_name']}"
    output_plots_dir = f"{output_dir}/plots"
    print(f'Evaluating final model {nn_config["model_name"]} on test dataset {data_config["test_dataset"]}')
    print(f'Number of events in test dataset: {len(test_dataset)}')
    hp = nn_config['MLP_hyperparams'] if args.useMLP else nn_config['hyperparams']
    model = load_model(hp, input_features, output_features, batch_norm=False, useMLP=args.useMLP) # base_model
    model_path = f'{output_plots_dir}/best_model.pth'
    if not os.path.exists(model_path):  # check if model exists, if not take partial model
        model_path = f'{output_plots_dir}/partial_model.pth'
    try:  # load model and optimizer
        model.load_state_dict(torch.load(model_path))
    except:
        print(f"Loading model from {model_path} failed. Trying to load from CPU.")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(">> Successfully loaded model")
    model.eval()

    # get tau pi and pizero four vectors from test_df
    true_taun_pi = test_df[['taun_pi1_e', 'taun_pi1_px', 'taun_pi1_py', 'taun_pi1_pz']].values
    true_taup_pi  = test_df[['taup_pi1_e', 'taup_pi1_px', 'taup_pi1_py', 'taup_pi1_pz']].values
    true_taun_pizero  = test_df[['taun_pizero1_e', 'taun_pizero1_px', 'taun_pizero1_py', 'taun_pizero1_pz']].values
    true_taup_pizero = test_df[['taup_pizero1_e', 'taup_pizero1_px', 'taup_pizero1_py', 'taup_pizero1_pz']].values

    if use_reco:
        reco_taun_pi = test_df[['reco_taun_pi1_e', 'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz']].values
        reco_taup_pi  = test_df[['reco_taup_pi1_e', 'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz']].values
        reco_taun_pizero  = test_df[['reco_taun_pizero1_e', 'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz']].values
        reco_taup_pizero = test_df[['reco_taup_pizero1_e', 'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz']].values


    # move X_test and model to device
    X_test, _ = test_dataset[:]
    X_test = X_test.to(device)
    model = model.to(device)

    if args.useMLP:
        # for MLP we just do a forward pass
        with torch.no_grad():
            predictions_norm = model(X_test)
        # get predictions
        predictions = test_dataset.destandardize_outputs(predictions_norm).cpu().numpy()
        samples_alt = None
        predictions_alt = None

    else:
        # move X_test and model to device
        with torch.no_grad():
            predictions_norm = model.sample(num_samples=1, context=X_test).squeeze()

        # destandardize predictions so that they are in physical units
        predictions = test_dataset.destandardize_outputs(predictions_norm).cpu().numpy()

    if data_config['coordinates'] == 'polar':
        # convert predictions back to cartesian coordinates
        predictions = ConvertPredictionsToCartesian(predictions, output_features)
    elif data_config['coordinates'] == 'onorm':
        # convert predictions back from orthonormal basis
        if use_reco:
            predictions = ConvertFromOrthonormalNRK_Predictions(predictions, taup_pi=reco_taup_pi, taup_pi0=reco_taup_pizero,
                                                                taun_pi=reco_taun_pi, taun_pi0=reco_taun_pizero)
        else:
            predictions = ConvertFromOrthonormalNRK_Predictions(predictions, taup_pi=true_taup_pi, taup_pi0=true_taup_pizero,
                                                                taun_pi=true_taun_pi, taun_pi0=true_taun_pizero)

    if not args.useMLP:
        # define alternative prediction by taking most probable value from flow
        # to do this we sample 100 times and take the case with the best log probability
        # estimate most likely solution using flow_map_predict function
        print("Computing alternative predictions using flow_map_predict...")
        _, samples_alt = flow_map_predict(  # TODO: this can probably be improved
            model,
            X_test,
            test_dataset=test_dataset,
            num_draws=100,
            chunk_size=1000 if device.type == 'cpu' else 10000,
        )

        if data_config['coordinates'] == 'polar':
            samples_alt = ConvertPredictionsToCartesian(samples_alt, output_features)
        elif data_config['coordinates'] == 'onorm':
            if use_reco:
                samples_alt = ConvertFromOrthonormalNRK_Predictions(samples_alt, taup_pi=reco_taup_pi, taup_pi0=reco_taup_pizero,
                                                                taun_pi=reco_taun_pi, taun_pi0=reco_taun_pizero)
            else:
                samples_alt = ConvertFromOrthonormalNRK_Predictions(samples_alt, taup_pi=true_taup_pi, taup_pi0=true_taup_pizero,
                                                                    taun_pi=true_taun_pi, taun_pi0=true_taun_pizero)

    if samples_alt is not None:

        # unpack MAP outputs
        alt_nubar_px = samples_alt[:,0]
        alt_nubar_py = samples_alt[:,1]
        alt_nubar_pz = samples_alt[:,2]
        alt_nu_px    = samples_alt[:,3]
        alt_nu_py    = samples_alt[:,4]
        alt_nu_pz    = samples_alt[:,5]

        # energies
        alt_nu_E    = np.sqrt(alt_nu_px**2    + alt_nu_py**2    + alt_nu_pz**2)
        alt_nubar_E = np.sqrt(alt_nubar_px**2 + alt_nubar_py**2 + alt_nubar_pz**2)

        # final MAP / "alternative" array
        predictions_alt = np.column_stack(
            (alt_nubar_E, alt_nubar_px, alt_nubar_py, alt_nubar_pz,
            alt_nu_E, alt_nu_px, alt_nu_py, alt_nu_pz)
        )
    
    # get true values
    true_values = test_df[output_features].values
    true_values_alt = test_df[['taup_nu_px', 'taup_nu_py', 'taup_nu_pz', 'taun_nu_px', 'taun_nu_py', 'taun_nu_pz']].values

    if data_config['coordinates'] == 'polar':
        # convert true values back to cartesian coordinates
        true_values = ConvertPredictionsToCartesian(true_values, output_features)
    elif data_config['coordinates'] == 'onorm':
        # convert true values back from orthonormal basis
        if use_reco:
            true_values = ConvertFromOrthonormalNRK_Predictions(true_values, taup_pi=reco_taup_pi, taup_pi0=reco_taup_pizero,
                                                                taun_pi=reco_taun_pi, taun_pi0=reco_taun_pizero)
        else:
            true_values = ConvertFromOrthonormalNRK_Predictions(true_values, taup_pi=taup_pi, taup_pi0=taup_pizero,
                                                                taun_pi=taun_pi, taun_pi0=taun_pizero)

    # predictions dont include E so we need to compute them # TODO: make function
    # compute E for nu and nubar
    nubar_px = predictions[:,0]
    nubar_py = predictions[:,1]
    nubar_pz = predictions[:,2]
    nu_px = predictions[:,3]
    nu_py = predictions[:,4]
    nu_pz = predictions[:,5]
    nu_E = np.sqrt(nu_px**2 + nu_py**2 + nu_pz**2)
    nubar_E = np.sqrt(nubar_px**2 + nubar_py**2 + nubar_pz**2)
    predictions = np.column_stack((nubar_E, nubar_px, nubar_py, nubar_pz, nu_E, nu_px, nu_py, nu_pz))

    # get E components for true values as well
    true_nubar_px = true_values[:,0]
    true_nubar_py = true_values[:,1]
    true_nubar_pz = true_values[:,2]
    true_nu_px = true_values[:,3]
    true_nu_py = true_values[:,4]
    true_nu_pz = true_values[:,5]
    true_nu_E = np.sqrt(true_nu_px**2 + true_nu_py**2 + true_nu_pz**2)
    true_nubar_E = np.sqrt(true_nubar_px**2 + true_nubar_py**2 + true_nubar_pz**2)
    true_values = np.column_stack((true_nubar_E, true_nubar_px, true_nubar_py, true_nubar_pz,
                                   true_nu_E, true_nu_px, true_nu_py, true_nu_pz))

    # get true taus by summing with pis and pizeros

    # tau- = nu + pi + pizero
    taup = true_values[:, 0:4] + true_taup_pi + true_taup_pizero
    
    # tau+ = nu + pi + pizero
    taun = true_values[:, 4:8] + true_taun_pi + true_taun_pizero
    
    # final true taus (8 columns total)
    true_taus = np.concatenate([taup, taun], axis=1)

    # now use predicted neutrino but add to true visible products to get predicted taus
    taup_pred = predictions[:, 0:4] + true_taup_pi + true_taup_pizero
    taun_pred = predictions[:, 4:8] + true_taun_pi + true_taun_pizero

    pred_taus = np.concatenate([taup_pred, taun_pred], axis=1)

    if use_reco:
        reco_taup_pred = predictions[:, 0:4] + reco_taup_pi + reco_taup_pizero
        reco_taun_pred = predictions[:, 4:8] + reco_taun_pi + reco_taun_pizero

        reco_pred_taus = np.concatenate([reco_taup_pred, reco_taun_pred], axis=1)

    # get alternative predictions
    if predictions_alt is not None:
        alt_taup_pred = predictions_alt[:, 0:4] + true_taup_pi + true_taup_pizero
        alt_taun_pred = predictions_alt[:, 4:8] + true_taun_pi + true_taun_pizero
        pred_taus_alt = np.concatenate([alt_taup_pred, alt_taun_pred], axis=1)

        if use_reco:
            alt_reco_taup_pred = predictions_alt[:, 0:4] + reco_taup_pi + reco_taup_pizero
            alt_reco_taun_pred = predictions_alt[:, 4:8] + reco_taun_pi + reco_taun_pizero
            alt_reco_pred_taus = np.concatenate([alt_reco_taup_pred, alt_reco_taun_pred], axis=1)

    # collect true and predicted nus true and predicted taus AND pi's into pandas dataframe, label the collumns

    true_taup_haspizero = test_df['taup_haspizero'].values.reshape(-1,1)
    true_taun_haspizero = test_df['taun_haspizero'].values.reshape(-1,1)

    if use_reco:
        reco_taup_haspizero = test_df['reco_taup_haspizero'].values.reshape(-1,1) if use_reco else None
        reco_taun_haspizero = test_df['reco_taun_haspizero'].values.reshape(-1,1) if use_reco else None

    results_df = pd.DataFrame(data=np.concatenate([true_values, predictions, true_taus, pred_taus, true_taun_haspizero, true_taup_haspizero,
                              true_taup_pi, true_taup_pizero, true_taun_pi, true_taun_pizero], axis=1),
                              columns=[
                                       'true_nubar_E', 'true_nubar_px', 'true_nubar_py', 'true_nubar_pz',
                                       'true_nu_E', 'true_nu_px', 'true_nu_py', 'true_nu_pz',
                                       'pred_nubar_E', 'pred_nubar_px', 'pred_nubar_py', 'pred_nubar_pz',
                                       'pred_nu_E', 'pred_nu_px', 'pred_nu_py', 'pred_nu_pz',
                                       'true_tau_plus_E',  'true_tau_plus_px',  'true_tau_plus_py',  'true_tau_plus_pz',
                                       'true_tau_minus_E', 'true_tau_minus_px', 'true_tau_minus_py', 'true_tau_minus_pz',
                                       'pred_tau_plus_E',  'pred_tau_plus_px',  'pred_tau_plus_py',  'pred_tau_plus_pz',
                                       'pred_tau_minus_E', 'pred_tau_minus_px', 'pred_tau_minus_py', 'pred_tau_minus_pz',
                                       'true_taun_haspizero', 'true_taup_haspizero',
                                       'true_taup_pi1_E', 'true_taup_pi1_px', 'true_taup_pi1_py', 'true_taup_pi1_pz',
                                       'true_taup_pizero1_E', 'true_taup_pizero1_px', 'true_taup_pizero1_py', 'true_taup_pizero1_pz',
                                       'true_taun_pi1_E', 'true_taun_pi1_px', 'true_taun_pi1_py', 'true_taun_pi1_pz',
                                       'true_taun_pizero1_E', 'true_taun_pizero1_px', 'true_taun_pizero1_py', 'true_taun_pizero1_pz',
                                       ])

    if use_reco:
        results_df_extra = pd.DataFrame(data=np.concatenate([reco_pred_taus, reco_taup_haspizero, reco_taun_haspizero,
                                                            reco_taup_pi, reco_taup_pizero, reco_taun_pi, reco_taun_pizero], axis=1),
                              columns=[
                                        'reco_pred_tau_plus_E',  'reco_pred_tau_plus_px',  'reco_pred_tau_plus_py',  'reco_pred_tau_plus_pz',
                                        'reco_pred_tau_minus_E', 'reco_pred_tau_minus_px', 'reco_pred_tau_minus_py', 'reco_pred_tau_minus_pz',
                                        'reco_taup_haspizero', 'reco_taun_haspizero',
                                        'reco_taup_pi1_E', 'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz',
                                        'reco_taup_pizero1_E', 'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz',
                                        'reco_taun_pi1_E', 'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz',
                                        'reco_taun_pizero1_E', 'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz',
                                       ])
        results_df = pd.concat([results_df, results_df_extra], axis=1)


    if predictions_alt is not None:
        results_df_extra = pd.DataFrame(data=np.concatenate([predictions_alt, pred_taus_alt], axis=1),
                              columns=[
                                        'alt_pred_nubar_E', 'alt_pred_nubar_px', 'alt_pred_nubar_py', 'alt_pred_nubar_pz',
                                        'alt_pred_nu_E', 'alt_pred_nu_px', 'alt_pred_nu_py', 'alt_pred_nu_pz',
                                        'alt_pred_tau_plus_E',  'alt_pred_tau_plus_px',  'alt_pred_tau_plus_py',  'alt_pred_tau_plus_pz',
                                        'alt_pred_tau_minus_E', 'alt_pred_tau_minus_px', 'alt_pred_tau_minus_py', 'alt_pred_tau_minus_pz',

                                       ])
        results_df = pd.concat([results_df, results_df_extra], axis=1)

        if use_reco:
            results_df_extra = pd.DataFrame(data=np.concatenate([alt_reco_pred_taus], axis=1),
                                columns=[
                                        'alt_reco_pred_tau_plus_E',  'alt_reco_pred_tau_plus_px',  'alt_reco_pred_tau_plus_py',  'alt_reco_pred_tau_plus_pz',
                                        'alt_reco_pred_tau_minus_E', 'alt_reco_pred_tau_minus_px', 'alt_reco_pred_tau_minus_py', 'alt_reco_pred_tau_minus_pz',
                                       ])
            results_df = pd.concat([results_df, results_df_extra], axis=1)

    # compute predicted mass of taus and Z boson and store on dataframe
    pred_tau_plus_mass = np.sqrt(np.maximum(pred_taus[:,0]**2 - pred_taus[:,1]**2 - pred_taus[:,2]**2 - pred_taus[:,3]**2, 0))
    pred_tau_minus_mass  = np.sqrt(np.maximum(pred_taus[:,4]**2 - pred_taus[:,5]**2 - pred_taus[:,6]**2 - pred_taus[:,7]**2, 0))
    pred_boson_mass = np.sqrt(np.maximum((pred_taus[:,0] + pred_taus[:,4])**2 - (pred_taus[:,1] + pred_taus[:,5])**2 - (pred_taus[:,2] + pred_taus[:,6])**2 - (pred_taus[:,3] + pred_taus[:,7])**2, 0))
    results_df['true_tau_plus_mass'] = np.sqrt(np.maximum(true_taus[:,0]**2 - true_taus[:,1]**2 - true_taus[:,2]**2 - true_taus[:,3]**2, 0))
    results_df['true_tau_minus_mass'] = np.sqrt(np.maximum(true_taus[:,4]**2 - true_taus[:,5]**2 - true_taus[:,6]**2 - true_taus[:,7]**2, 0))
    results_df['pred_tau_minus_mass'] = pred_tau_minus_mass
    results_df['pred_tau_plus_mass'] = pred_tau_plus_mass
    results_df['pred_boson_mass'] = pred_boson_mass

    if use_reco:
        reco_pred_tau_plus_mass = np.sqrt(np.maximum(reco_pred_taus[:,0]**2 - reco_pred_taus[:,1]**2 - reco_pred_taus[:,2]**2 - reco_pred_taus[:,3]**2, 0))
        reco_pred_tau_minus_mass  = np.sqrt(np.maximum(reco_pred_taus[:,4]**2 - reco_pred_taus[:,5]**2 - reco_pred_taus[:,6]**2 - reco_pred_taus[:,7]**2, 0))
        results_df['reco_pred_tau_plus_mass'] = reco_pred_tau_plus_mass
        results_df['reco_pred_tau_minus_mass'] = reco_pred_tau_minus_mass

        reco_pred_boson_mass = np.sqrt(np.maximum((reco_pred_taus[:,0] + reco_pred_taus[:,4])**2 - (reco_pred_taus[:,1] + reco_pred_taus[:,5])**2 - (reco_pred_taus[:,2] + reco_pred_taus[:,6])**2 - (reco_pred_taus[:,3] + reco_pred_taus[:,7])**2, 0))
        results_df['reco_pred_boson_mass'] = reco_pred_boson_mass

    if predictions_alt is not None:
        results_df['alt_pred_tau_plus_mass'] = np.sqrt(np.maximum(pred_taus_alt[:,0]**2 - pred_taus_alt[:,1]**2 - pred_taus_alt[:,2]**2 - pred_taus_alt[:,3]**2, 0))
        results_df['alt_pred_tau_minus_mass'] = np.sqrt(np.maximum(pred_taus_alt[:,4]**2 - pred_taus_alt[:,5]**2 - pred_taus_alt[:,6]**2 - pred_taus_alt[:,7]**2, 0))
        if use_reco:
            results_df['alt_reco_pred_tau_plus_mass'] = np.sqrt(np.maximum(alt_reco_pred_taus[:,0]**2 - alt_reco_pred_taus[:,1]**2 - alt_reco_pred_taus[:,2]**2 - alt_reco_pred_taus[:,3]**2, 0))
            results_df['alt_reco_pred_tau_minus_mass'] = np.sqrt(np.maximum(alt_reco_pred_taus[:,4]**2 - alt_reco_pred_taus[:,5]**2 - alt_reco_pred_taus[:,6]**2 - alt_reco_pred_taus[:,7]**2, 0))
            results_df['alt_reco_pred_boson_mass'] = np.sqrt(np.maximum((alt_reco_pred_taus[:,0] + alt_reco_pred_taus[:,4])**2 - (alt_reco_pred_taus[:,1] + alt_reco_pred_taus[:,5])**2 - (alt_reco_pred_taus[:,2] + alt_reco_pred_taus[:,6])**2 - (alt_reco_pred_taus[:,3] + alt_reco_pred_taus[:,7])**2, 0))


    # get spin vars for true_tau
    print("Computing spin variables...")
    results_df = compute_spin_vars(results_df, tau_pred_prefix='true_', tau_vis_prefix='true_')  # for true taus we can use the true visible products as well
    # get spin vars for pred_tau
    results_df = compute_spin_vars(results_df, tau_pred_prefix='pred_', tau_vis_prefix='reco_' if use_reco else 'true_')
    # get spin vars for alt_pred_tau if present
    if predictions_alt is not None:
        results_df = compute_spin_vars(results_df, tau_pred_prefix='alt_pred_', tau_vis_prefix='reco_' if use_reco else 'true_')

    # loop over dm categories and compute spin density matrix variables for each
    for dm_category in ['all','dm_0_0','dm_0_1','dm_1_1']:

        #TODO: could also do splitting based on reco dm category
        if dm_category == 'dm_0_0':
            results_df_dm = results_df[(results_df['true_taup_haspizero'] == 0) & (results_df['true_taun_haspizero'] == 0)]
        elif dm_category == 'dm_0_1':
            results_df_dm = results_df[((results_df['true_taup_haspizero'] == 0) & (results_df['true_taun_haspizero'] == 1)) | ((results_df['true_taup_haspizero'] == 1) & (results_df['true_taun_haspizero'] == 0))]
        elif dm_category == 'dm_1_1':
            results_df_dm = results_df[(results_df['true_taup_haspizero'] == 1) & (results_df['true_taun_haspizero'] == 1)]
        else:
            results_df_dm = results_df

        true_Bplus, true_Bminus, true_C, true_con, true_m12 = compute_spin_density_vars(results_df_dm, prefix='true_')
        pred_Bplus, pred_Bminus, pred_C, pred_con, pred_m12 = compute_spin_density_vars(results_df_dm, prefix='pred_')
        if predictions_alt is not None:
            alt_pred_Bplus, alt_pred_Bminus, alt_pred_C, alt_pred_con, alt_pred_m12 = compute_spin_density_vars(results_df_dm, prefix='alt_pred_')

        print('\n===== DM CATEGORY:', dm_category, '=====')
        print(f'Number of events in this category: {len(results_df_dm)}')
        print('True spin density matrix variables:')
        print(true_Bplus)
        print(true_Bminus)
        print(true_C)
        print(true_con, true_m12)
        print()
        print('NN predicted spin density matrix variables:')
        print(pred_Bplus)
        print(pred_Bminus)
        print(pred_C)
        print(pred_con, pred_m12)
        print()
        if predictions_alt is not None:
            print('Alternative NN predicted spin density matrix variables:')
            print(alt_pred_Bplus)
            print(alt_pred_Bminus)
            print(alt_pred_C)
            print(alt_pred_con, alt_pred_m12)

    # write the results dataframe to a pickle file
    results_df.to_pickle(f"{output_dir}/{data_config['test_output_name']}.pkl")

    # write root file aswell
    output_root_file = f"{output_dir}/{data_config['test_output_name']}.root"

    with uproot.recreate(output_root_file) as f:
        f["tree"] = results_df.to_dict(orient="list")
    
