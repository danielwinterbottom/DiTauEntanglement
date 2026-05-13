import torch
import pandas as pd
import argparse
import yaml
import os
import uproot
import numpy as np
from tauentanglement.python.NN_Tools import load_model
from tauentanglement.python.DataProcessing import get_test_dataset
from tauentanglement.utils.coordinate_conversions import ConvertPredictionsToCartesian, ConvertFromOrthonormalNRK_Predictions, convert_coordinates_pred
from tauentanglement.python.Evaluation_Tools import flow_map_predict, compute_spin_vars, save_sampled_pdfs, plot_spin_density_matrix
from tauentanglement.utils.kinematic_helpers import compute_spin_density_vars, add_energies_pair, add_energy, inv_mass


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', help='path to the configuration file', type=str, default='tauentanglement/config/LEP.yaml', required=True)
    argparser.add_argument('--useMLP', help='whether to use a simple MLP instead of a normalizing flow', action='store_true')
    argparser.add_argument('--useCPU', help='whether to use CPU only for evaluation', action='store_true')
    args = argparser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']

    nn_config   = config['SetupNN']
    coordinates = data_config['coordinates']
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

    output_dir = f"outputs_{nn_config['model_name']}"
    output_plots_dir = f"{output_dir}/plots"

    norm_data = np.load(f'{output_dir}/normalization_params.npz') # we get the means and stds used in training the model so that we can apply the same normalization to the test dataset
    test_dataset, test_df, input_features, output_features = get_test_dataset(dataset, data_config, norm_data)

    # load model
    print(f'Evaluating final model {nn_config["model_name"]} on test dataset {data_config["test_dataset"]}')
    print(f'Number of events in test dataset: {len(test_dataset)}')
    hp = nn_config['MLP_hyperparams'] if args.useMLP else nn_config['hyperparams']
    model = load_model(hp, input_features, output_features, batch_norm=False, useMLP=args.useMLP)
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

    # gets ips as well
    true_taun_pi_ip = test_df[['taun_pi1_ipx', 'taun_pi1_ipy', 'taun_pi1_ipz']].values
    true_taup_pi_ip = test_df[['taup_pi1_ipx', 'taup_pi1_ipy', 'taup_pi1_ipz']].values

    if use_reco:
        reco_taun_pi = test_df[['reco_taun_pi1_e', 'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz']].values
        reco_taup_pi  = test_df[['reco_taup_pi1_e', 'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz']].values
        reco_taun_pizero  = test_df[['reco_taun_pizero1_e', 'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz']].values
        reco_taup_pizero = test_df[['reco_taup_pizero1_e', 'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz']].values

        # get ips for reco pis as well
        reco_taun_pi_ip = test_df[['reco_taun_pi1_ipx', 'reco_taun_pi1_ipy', 'reco_taun_pi1_ipz']].values
        reco_taup_pi_ip = test_df[['reco_taup_pi1_ipx', 'reco_taup_pi1_ipy', 'reco_taup_pi1_ipz']].values

    # move X_test and model to device
    X_test, _ = test_dataset[:]
    del _
    X_test = X_test.to(device)
    model = model.to(device)

    samples_map = None
    sample_chunk_size = 50000 if device.type == 'cpu' else 100000
    if args.useMLP:
        # for MLP we just do a forward pass then get predictions
        with torch.no_grad():
            predictions_norm = model(X_test)
    else:
        # sample from the normflow pdf in chunks to avoid memory issues
        pred_chunks = []
        with torch.no_grad():
            for start in range(0, X_test.shape[0], sample_chunk_size):
                chunk = model.sample(num_samples=1, context=X_test[start:start + sample_chunk_size]).squeeze(1)
                pred_chunks.append(chunk.cpu())
        predictions_norm = torch.cat(pred_chunks, dim=0)
        del pred_chunks

    # destandardize predictions so that they are in physical units
    predictions = test_dataset.destandardize_outputs(predictions_norm).cpu().numpy()

    if use_reco:
      conv_kwargs = dict(coordinates=coordinates, output_features=output_features,
                    taup_pi=reco_taup_pi, taup_pizero=reco_taup_pizero, taun_pi=reco_taun_pi, taun_pizero=reco_taun_pizero)
    else:
      conv_kwargs = dict(coordinates=coordinates, output_features=output_features,
                    taup_pi=true_taup_pi, taup_pizero=true_taup_pizero, taun_pi=true_taun_pi, taun_pizero=true_taun_pizero)
    predictions = convert_coordinates_pred(predictions, **conv_kwargs)

    if not args.useMLP:
        # define alternative prediction by taking most probable value from flow, do this by sampling to find MAP estimate
        print("Computing alternative predictions using flow_map_predict...")
        map_method = nn_config.get('map_method', 'latent_zero')
        _, samples_map = flow_map_predict(
            model, X_test,
            test_dataset=test_dataset,
            num_draws=nn_config.get('map_num_draws', 100),
            chunk_size=1000 if device.type == 'cpu' else 10000,
            method=map_method,
        )
        samples_map = convert_coordinates_pred(samples_map, **conv_kwargs)
  
    # add energies (E=|p|) and build neutrino 4-vectors
    predictions     = add_energies_pair(predictions)
    predictions_map = add_energies_pair(samples_map) if samples_map is not None else None

    # also get the gen values
    true_values = convert_coordinates_pred(test_df[output_features].values, **conv_kwargs)
    true_values = add_energies_pair(true_values)
   
    # get true taus by summing with pis and pizeros
    true_taus = np.concatenate([true_values[:, 0:4] + true_taup_pi + true_taup_pizero,
                                true_values[:, 4:8] + true_taun_pi + true_taun_pizero], axis=1)
    
    # now use predicted neutrino but add to visible products to get predicted taus
    if use_reco:
        pred_taus = np.concatenate([predictions[:, 0:4] + reco_taup_pi + reco_taup_pizero,
                                predictions[:, 4:8] + reco_taun_pi + reco_taun_pizero], axis=1)
    else:
        pred_taus = np.concatenate([predictions[:, 0:4] + true_taup_pi + true_taup_pizero,
                                predictions[:, 4:8] + true_taun_pi + true_taun_pizero], axis=1)
    
    # same for the MAP predictions
    pred_taus_map = None
    if predictions_map is not None:
        if use_reco:
            pred_taus_map = np.concatenate([predictions_map[:, 0:4] + reco_taup_pi + reco_taup_pizero,
                                        predictions_map[:, 4:8] + reco_taun_pi + reco_taun_pizero], axis=1)
        else:
            pred_taus_map = np.concatenate([predictions_map[:, 0:4] + true_taup_pi + true_taup_pizero,
                                        predictions_map[:, 4:8] + true_taun_pi + true_taun_pizero], axis=1)
  
  
    # build dataframe for results
    true_taup_haspizero = test_df['taup_haspizero'].values.reshape(-1,1)
    true_taun_haspizero = test_df['taun_haspizero'].values.reshape(-1,1)

    if use_reco:
        reco_taup_haspizero = test_df['reco_taup_haspizero'].values.reshape(-1,1) if use_reco else None
        reco_taun_haspizero = test_df['reco_taun_haspizero'].values.reshape(-1,1) if use_reco else None
        
    # collect true and predicted nus, true and predicted taus, and pi's into pandas dataframe, label the columns
    results_df = pd.DataFrame(data=np.concatenate([true_values, predictions, true_taus, pred_taus, true_taun_haspizero, true_taup_haspizero,
                              true_taup_pi, true_taup_pizero, true_taun_pi, true_taun_pizero, true_taun_pi_ip, true_taup_pi_ip], axis=1),
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
                                       'true_taun_pi1_ipx', 'true_taun_pi1_ipy', 'true_taun_pi1_ipz',
                                       'true_taup_pi1_ipx', 'true_taup_pi1_ipy', 'true_taup_pi1_ipz'
                                       ])
    
    
    if use_reco:
        results_df_extra = pd.DataFrame(data=np.concatenate([reco_taup_haspizero, reco_taun_haspizero,
                                                            reco_taup_pi, reco_taup_pizero, reco_taun_pi, reco_taun_pizero, reco_taun_pi_ip, reco_taup_pi_ip], axis=1),
                              columns=[
                                        'reco_taup_haspizero', 'reco_taun_haspizero',
                                        'reco_taup_pi1_E', 'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz',
                                        'reco_taup_pizero1_E', 'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz',
                                        'reco_taun_pi1_E', 'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz',
                                        'reco_taun_pizero1_E', 'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz',
                                        'reco_taun_pi1_ipx', 'reco_taun_pi1_ipy', 'reco_taun_pi1_ipz',
                                        'reco_taup_pi1_ipx', 'reco_taup_pi1_ipy', 'reco_taup_pi1_ipz'
                                       ])
        results_df = pd.concat([results_df, results_df_extra], axis=1)

    if predictions_map is not None:
        results_df_extra = pd.DataFrame(
            data=np.concatenate([predictions_map, pred_taus_map], axis=1),
            columns=[
                'map_pred_nubar_E', 'map_pred_nubar_px', 'map_pred_nubar_py', 'map_pred_nubar_pz',
                'map_pred_nu_E', 'map_pred_nu_px', 'map_pred_nu_py', 'map_pred_nu_pz',
                'map_pred_tau_plus_E',  'map_pred_tau_plus_px',  'map_pred_tau_plus_py',  'map_pred_tau_plus_pz',
                'map_pred_tau_minus_E', 'map_pred_tau_minus_px', 'map_pred_tau_minus_py', 'map_pred_tau_minus_pz',
            ])
        results_df = pd.concat([results_df, results_df_extra], axis=1)
        
    
    # invariant masses (per tau and for the pair)
    results_df['true_tau_plus_mass']  = inv_mass(true_taus, 0)
    results_df['true_tau_minus_mass'] = inv_mass(true_taus, 4)
    results_df['pred_tau_plus_mass']  = inv_mass(pred_taus, 0)
    results_df['pred_tau_minus_mass'] = inv_mass(pred_taus, 4)
    results_df['pred_boson_mass'] = np.sqrt(np.maximum(
        (pred_taus[:,0]+pred_taus[:,4])**2 - (pred_taus[:,1]+pred_taus[:,5])**2
        - (pred_taus[:,2]+pred_taus[:,6])**2 - (pred_taus[:,3]+pred_taus[:,7])**2, 0))
    if predictions_map is not None:
        results_df['map_pred_tau_plus_mass']  = inv_mass(pred_taus_map, 0)
        results_df['map_pred_tau_minus_mass'] = inv_mass(pred_taus_map, 4)
        results_df['map_pred_boson_mass'] = np.sqrt(np.maximum(
            (pred_taus_map[:,0]+pred_taus_map[:,4])**2 - (pred_taus_map[:,1]+pred_taus_map[:,5])**2
            - (pred_taus_map[:,2]+pred_taus_map[:,6])**2 - (pred_taus_map[:,3]+pred_taus_map[:,7])**2, 0))

    # spin variables
    print("Computing spin variables...")
    results_df = compute_spin_vars(results_df, tau_pred_prefix='true_', tau_vis_prefix='true_') 
    results_df = compute_spin_vars(results_df, tau_pred_prefix='pred_',  tau_vis_prefix='reco_' if use_reco else 'true_')
    # get spin vars for MAP prediction if present
    if predictions_map is not None:
        results_df = compute_spin_vars(results_df, tau_pred_prefix='map_pred_', tau_vis_prefix='reco_' if use_reco else 'true_')

    # loop over dm categories and compute spin density matrix variables for each
    # TODO: could also do splitting based on reco dm category - both give us different but useful information
    dm_masks = {
        'all':    results_df,
        'dm_0_0': results_df[(results_df['true_taup_haspizero'] == 0) & (results_df['true_taun_haspizero'] == 0)],
        'dm_0_1': results_df[((results_df['true_taup_haspizero'] == 0) & (results_df['true_taun_haspizero'] == 1)) |
                              ((results_df['true_taup_haspizero'] == 1) & (results_df['true_taun_haspizero'] == 0))],
        'dm_1_1': results_df[(results_df['true_taup_haspizero'] == 1) & (results_df['true_taun_haspizero'] == 1)],
    }
    spin_plot_dir = f"{output_plots_dir}/spin_density"
    for dm_category, results_df_dm in dm_masks.items():
        true_Bplus, true_Bminus, true_C, true_con, true_m12 = compute_spin_density_vars(results_df_dm, prefix='true_')
        pred_Bplus, pred_Bminus, pred_C, pred_con, pred_m12 = compute_spin_density_vars(results_df_dm, prefix='pred_')
        if predictions_map is not None:
            map_pred_Bplus, map_pred_Bminus, map_pred_C, map_pred_con, map_pred_m12 = compute_spin_density_vars(results_df_dm, prefix='map_pred_')

        print('\n===== DM CATEGORY:', dm_category, '=====')
        print(f'Number of events in this category: {len(results_df_dm)}')
        print('\n True spin density matrix variables:')
        print(true_Bplus)
        print(true_Bminus)
        print(true_C)
        print(true_con, true_m12)
        print()

        print('\n Sampled predicted spin density matrix variables:')
        print(pred_Bplus)
        print(pred_Bminus)
        print(pred_C)
        print(pred_con, pred_m12)
        
        if predictions_map is not None:
            print('\n MAP estimate spin density matrix variables:')
            print(map_pred_Bplus)
            print(map_pred_Bminus)
            print(map_pred_C)
            print(map_pred_con, map_pred_m12)

        # collect results for plotting
        plot_results = {'True': (true_Bplus, true_Bminus, true_C, true_con, true_m12)}
        plot_results['Sampled'] = (pred_Bplus, pred_Bminus, pred_C, pred_con, pred_m12)
        if predictions_map is not None:
            plot_results['MAP'] = (map_pred_Bplus, map_pred_Bminus, map_pred_C, map_pred_con, map_pred_m12)
        plot_spin_density_matrix(plot_results, dm_category, outdir=spin_plot_dir)

    # write the results dataframe to a parquet file
    results_df.to_parquet(f"{output_dir}/{data_config['test_output_name']}.parquet")

    # write root file aswell
    with uproot.recreate(f"{output_dir}/{data_config['test_output_name']}.root") as f:
        f.mktree('tree', results_df.to_dict(orient="list"))

if __name__ == "__main__":
    main()
