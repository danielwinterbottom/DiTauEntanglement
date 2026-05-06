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
from tauentanglement.python.Evaluation_Tools import flow_map_predict, compute_spin_vars, save_sampled_pdfs
from tauentanglement.utils.kinematic_helpers import compute_spin_density_vars


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
    collider    = config['collider']

    # set gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.useCPU else "cpu")

    # load test dataset
    if len(data_config['datasets']) > 1:
        raise NotImplementedError("Currently only supports one dataset at a time.")

    dataset = data_config['datasets'][0]
    test_dataset, test_df, input_features, output_features = get_test_dataset(dataset, data_config)

    # load model
    output_dir = f"outputs_{nn_config['model_name']}"
    output_plots_dir = f"{output_dir}/plots"
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
    if collider == 'LHC':
        taun_pi = test_df[['taun_pi1_e', 'taun_pi1_px', 'taun_pi1_py', 'taun_pi1_pz']].values
        taup_pi  = test_df[['taup_pi1_e', 'taup_pi1_px', 'taup_pi1_py', 'taup_pi1_pz']].values
        taun_pizero  = test_df[['taun_pizero1_e', 'taun_pizero1_px', 'taun_pizero1_py', 'taun_pizero1_pz']].values
        taup_pizero = test_df[['taup_pizero1_e', 'taup_pizero1_px', 'taup_pizero1_py', 'taup_pizero1_pz']].values
    elif collider == 'LEP':
        taun_pi = test_df[['reco_taun_pi1_e', 'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz']].values
        taup_pi  = test_df[['reco_taup_pi1_e', 'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz']].values
        taun_pizero  = test_df[['reco_taun_pizero1_e', 'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz']].values
        taup_pizero = test_df[['reco_taup_pizero1_e', 'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz']].values
    else:
        raise ValueError(f"Collider {collider} not recognized. Should be 'LHC' or 'LEP'.")

    # move X_test and model to device
    X_test, _ = test_dataset[:]
    X_test = X_test.to(device)
    model = model.to(device)

    samples_alt = None
    if args.useMLP:
        # for MLP we just do a forward pass then get predictions
        with torch.no_grad():
            predictions_norm = model(X_test)
    else:
        # sample from the normflow pdf
        with torch.no_grad():
            predictions_norm = model.sample(num_samples=1, context=X_test).squeeze()

    # destandardize predictions so that they are in physical units
    predictions = test_dataset.destandardize_outputs(predictions_norm).cpu().numpy()

    conv_kwargs = dict(coordinates=coordinates, output_features=output_features,
                    taup_pi=taup_pi, taup_pizero=taup_pizero, taun_pi=taun_pi, taun_pizero=taun_pizero)
    predictions = convert_coordinates_pred(predictions, **conv_kwargs)

    if not args.useMLP:
        # define alternative prediction by taking most probable value from flow, do this by sampling to find MAP estimate
        print("Computing alternative predictions using flow_map_predict...")
        _, samples_alt = flow_map_predict(
            model, X_test,
            test_dataset=test_dataset,
            num_draws=10,
            chunk_size=1000 if device.type == 'cpu' else 10000,
        )
        samples_alt = convert_coordinates_pred(samples_alt, **conv_kwargs)

    # add energies (E=|p|) and build neutrino 4-vectors
    predictions     = add_energies_pair(predictions)
    predictions_alt = add_energies_pair(samples_alt) if samples_alt is not None else None

    # also get the gen values
    true_values = convert_coordinates_pred(test_df[output_features].values, **conv_kwargs)
    true_values = add_energies_pair(true_values)


    # tau 4-vectors: nu + pi + pizero
    true_taus = np.concatenate([true_values[:, 0:4] + taup_pi + taup_pizero,
                                true_values[:, 4:8] + taun_pi + taun_pizero], axis=1)
    pred_taus = np.concatenate([predictions[:, 0:4] + taup_pi + taup_pizero,
                                predictions[:, 4:8] + taun_pi + taun_pizero], axis=1)
    pred_taus_alt = None
    if predictions_alt is not None:
        pred_taus_alt = np.concatenate([predictions_alt[:, 0:4] + taup_pi + taup_pizero,
                                        predictions_alt[:, 4:8] + taun_pi + taun_pizero], axis=1)

    # analytical LEP predictions
    if collider == 'LEP':
        reco_taup_nu     = add_energy(test_df[['reco_taup_nu_px', 'reco_taup_nu_py', 'reco_taup_nu_pz']].values)
        reco_taun_nu     = add_energy(test_df[['reco_taun_nu_px', 'reco_taun_nu_py', 'reco_taun_nu_pz']].values)
        reco_alt_taup_nu = add_energy(test_df[['reco_alt_taup_nu_px', 'reco_alt_taup_nu_py', 'reco_alt_taup_nu_pz']].values)
        reco_alt_taun_nu = add_energy(test_df[['reco_alt_taun_nu_px', 'reco_alt_taun_nu_py', 'reco_alt_taun_nu_pz']].values)
        ana_pred_values     = np.concatenate([reco_taup_nu, reco_taun_nu], axis=1)
        ana_alt_pred_values = np.concatenate([reco_alt_taup_nu, reco_alt_taun_nu], axis=1)
        ana_pred_taus     = np.concatenate([reco_taup_nu     + taup_pi + taup_pizero,
                                            reco_taun_nu     + taun_pi + taun_pizero], axis=1)
        ana_alt_pred_taus = np.concatenate([reco_alt_taun_nu + taun_pi + taun_pizero,
                                            reco_alt_taup_nu + taup_pi + taup_pizero], axis=1)


    # build dataframe for results
    taup_haspizero = test_df['taup_haspizero'].values.reshape(-1, 1)
    taun_haspizero = test_df['taun_haspizero'].values.reshape(-1, 1)

    results_df = pd.DataFrame(
        data=np.concatenate([true_values, predictions, true_taus, pred_taus,
                             taun_haspizero, taup_haspizero,
                             taup_pi, taup_pizero, taun_pi, taun_pizero], axis=1),
        columns=[
            'true_nubar_E', 'true_nubar_px', 'true_nubar_py', 'true_nubar_pz',
            'true_nu_E', 'true_nu_px', 'true_nu_py', 'true_nu_pz',
            'pred_nubar_E', 'pred_nubar_px', 'pred_nubar_py', 'pred_nubar_pz',
            'pred_nu_E', 'pred_nu_px', 'pred_nu_py', 'pred_nu_pz',
            'true_tau_plus_E',  'true_tau_plus_px',  'true_tau_plus_py',  'true_tau_plus_pz',
            'true_tau_minus_E', 'true_tau_minus_px', 'true_tau_minus_py', 'true_tau_minus_pz',
            'pred_tau_plus_E',  'pred_tau_plus_px',  'pred_tau_plus_py',  'pred_tau_plus_pz',
            'pred_tau_minus_E', 'pred_tau_minus_px', 'pred_tau_minus_py', 'pred_tau_minus_pz',
            'taun_haspizero', 'taup_haspizero',
            'taup_pi1_E', 'taup_pi1_px', 'taup_pi1_py', 'taup_pi1_pz',
            'taup_pizero1_E', 'taup_pizero1_px', 'taup_pizero1_py', 'taup_pizero1_pz',
            'taun_pi1_E', 'taun_pi1_px', 'taun_pi1_py', 'taun_pi1_pz',
            'taun_pizero1_E', 'taun_pizero1_px', 'taun_pizero1_py', 'taun_pizero1_pz',
        ])

    if collider == 'LEP':
        ana_results_df = pd.DataFrame(
            data=np.concatenate([ana_pred_values, ana_alt_pred_values, ana_pred_taus, ana_alt_pred_taus], axis=1),
            columns=[
                'ana_pred_nubar_E', 'ana_pred_nubar_px', 'ana_pred_nubar_py', 'ana_pred_nubar_pz',
                'ana_pred_nu_E', 'ana_pred_nu_px', 'ana_pred_nu_py', 'ana_pred_nu_pz',
                'ana_alt_pred_nubar_E', 'ana_alt_pred_nubar_px', 'ana_alt_pred_nubar_py', 'ana_alt_pred_nubar_pz',
                'ana_alt_pred_nu_E', 'ana_alt_pred_nu_px', 'ana_alt_pred_nu_py', 'ana_alt_pred_nu_pz',
                'ana_pred_tau_plus_E',      'ana_pred_tau_plus_px',      'ana_pred_tau_plus_py',      'ana_pred_tau_plus_pz',
                'ana_pred_tau_minus_E',     'ana_pred_tau_minus_px',     'ana_pred_tau_minus_py',     'ana_pred_tau_minus_pz',
                'ana_alt_pred_tau_plus_E',  'ana_alt_pred_tau_plus_px',  'ana_alt_pred_tau_plus_py',  'ana_alt_pred_tau_plus_pz',
                'ana_alt_pred_tau_minus_E', 'ana_alt_pred_tau_minus_px', 'ana_alt_pred_tau_minus_py', 'ana_alt_pred_tau_minus_pz',
            ])
        results_df = pd.concat([results_df, ana_results_df], axis=1)

    if predictions_alt is not None:
        results_df_extra = pd.DataFrame(
            data=np.concatenate([predictions_alt, pred_taus_alt], axis=1),
            columns=[
                'alt_pred_nubar_E', 'alt_pred_nubar_px', 'alt_pred_nubar_py', 'alt_pred_nubar_pz',
                'alt_pred_nu_E', 'alt_pred_nu_px', 'alt_pred_nu_py', 'alt_pred_nu_pz',
                'alt_pred_tau_plus_E',  'alt_pred_tau_plus_px',  'alt_pred_tau_plus_py',  'alt_pred_tau_plus_pz',
                'alt_pred_tau_minus_E', 'alt_pred_tau_minus_px', 'alt_pred_tau_minus_py', 'alt_pred_tau_minus_pz',
            ])
        results_df = pd.concat([results_df, results_df_extra], axis=1)

    # invariant masses (per tau and for the pair)
    results_df['true_tau_plus_mass']  = inv_mass(true_taus, 0)
    results_df['true_tau_minus_mass'] = inv_mass(true_taus, 4)
    results_df['pred_tau_plus_mass']  = inv_mass(pred_taus, 0)
    results_df['pred_tau_minus_mass'] = inv_mass(pred_taus, 4)
    results_df['pred_z_mass'] = np.sqrt(np.maximum(
        (pred_taus[:,0]+pred_taus[:,4])**2 - (pred_taus[:,1]+pred_taus[:,5])**2
        - (pred_taus[:,2]+pred_taus[:,6])**2 - (pred_taus[:,3]+pred_taus[:,7])**2, 0))
    if predictions_alt is not None:
        results_df['alt_pred_tau_plus_mass']  = inv_mass(pred_taus_alt, 0)
        results_df['alt_pred_tau_minus_mass'] = inv_mass(pred_taus_alt, 4)
    if collider == 'LEP':
        results_df['ana_pred_tau_plus_mass']  = inv_mass(ana_pred_taus, 0)
        results_df['ana_pred_tau_minus_mass'] = inv_mass(ana_pred_taus, 4)

    # spin variables
    print("Computing spin variables...")
    results_df = compute_spin_vars(results_df, tau_prefix='true_')
    results_df = compute_spin_vars(results_df, tau_prefix='pred_')

    if collider == 'LEP':
        results_df = compute_spin_vars(results_df, tau_prefix='ana_pred_')

    if predictions_alt is not None:
        results_df = compute_spin_vars(results_df, tau_prefix='alt_pred_')

    # spin density matrix per DM category
    dm_masks = {
        'all':    results_df,
        'dm_0_0': results_df[(results_df['taup_haspizero'] == 0) & (results_df['taun_haspizero'] == 0)],
        'dm_0_1': results_df[((results_df['taup_haspizero'] == 0) & (results_df['taun_haspizero'] == 1)) |
                              ((results_df['taup_haspizero'] == 1) & (results_df['taun_haspizero'] == 0))],
        'dm_1_1': results_df[(results_df['taup_haspizero'] == 1) & (results_df['taun_haspizero'] == 1)],
    }
    for dm_category, results_df_dm in dm_masks.items():
        true_Bplus, true_Bminus, true_C, true_con, true_m12 = compute_spin_density_vars(results_df_dm, prefix='true_')
        if collider == 'LEP':
            ana_pred_Bplus, ana_pred_Bminus, ana_pred_C, ana_pred_con, ana_pred_m12 = compute_spin_density_vars(results_df_dm, prefix='ana_pred_')
        pred_Bplus, pred_Bminus, pred_C, pred_con, pred_m12 = compute_spin_density_vars(results_df_dm, prefix='pred_')
        if predictions_alt is not None:
            alt_pred_Bplus, alt_pred_Bminus, alt_pred_C, alt_pred_con, alt_pred_m12 = compute_spin_density_vars(results_df_dm, prefix='alt_pred_')


        print('\n===== DM CATEGORY:', dm_category, '=====')
        print('True spin density matrix variables:')
        print(true_Bplus)
        print(true_Bminus)
        print(true_C)
        print(true_con, true_m12)
        print()
        if collider == 'LEP':
            print('\nAnalytical predicted spin density matrix variables:')
            print(ana_pred_Bplus)
            print(ana_pred_Bminus)
            print(ana_pred_C)
            print(ana_pred_con, ana_pred_m12)
        print('\nNN predicted spin density matrix variables:')
        print(pred_Bplus)
        print(pred_Bminus)
        print(pred_C)
        print(pred_con, pred_m12)
        if predictions_alt is not None:
            print('\nAlternative NN predicted spin density matrix variables:')
            print(alt_pred_Bplus)
            print(alt_pred_Bminus)
            print(alt_pred_C)
            print(alt_pred_con, alt_pred_m12)


    # write the results dataframe to a parquet file
    results_df.to_parquet(f"{output_dir}/output_results.parquet")

    # write root file aswell
    with uproot.recreate(f"{output_dir}/output_results.root") as f:
        f["tree"] = results_df.to_dict(orient="list")

    # make plots of the samples PDFs vs the analytical solutions for some events
    if not args.useMLP and collider == 'LEP':
        for event_number in [0, 1, 2, 3, 4]:
            save_sampled_pdfs(
                model=model,
                device=device,
                dataset=test_dataset,
                df=results_df,
                output_features=['nubar_px', 'nubar_py', 'nubar_pz', 'nu_px', 'nu_py', 'nu_pz'],
                event_number=event_number,
                num_samples=50000,
                bins=100,
                outdir=f"{output_dir}/pdf_slices_sampled",
                use_polar=True if coordinates == 'polar' else False,
                use_onorm=True if coordinates == 'onorm' else False,
            )

if __name__ == "__main__":
    main()