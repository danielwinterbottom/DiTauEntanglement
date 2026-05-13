import torch
import pandas as pd
import argparse
from tauentanglement.python.DataProcessing import get_train_val_test_datasets
from tauentanglement.python.NN_Tools import setup_model_and_training, train_model
import yaml
import os
import numpy as np

# TODO: Transfer hyperparam optimisation code here and make work with new def


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', help='path to the configuration file', type=str, default='tauentanglement/config/LEP.yaml', required=True)
    argparser.add_argument('--useMLP', help='whether to use a simple MLP instead of a normalizing flow', action='store_true')
    args = argparser.parse_args()

    config_file = args.config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']
    nn_config = config['SetupNN']

    # make output directory called outputs_{model_name}, with plots subdirectory
    output_dir = f"outputs_{nn_config['model_name']}"
    output_plots_dir = f"{output_dir}/plots"
    os.makedirs(output_plots_dir, exist_ok=True)

    # gpu or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Get train and validation datasets

    datasets = data_config['datasets']
    train_dataset, val_dataset, input_features, output_features = get_train_val_test_datasets(datasets, data_config)

    # store the means and stds used for normalization
    in_mean, in_std = train_dataset.input_mean, train_dataset.input_std
    out_mean, out_std = train_dataset.output_mean, train_dataset.output_std

    np.savez(f'{output_dir}/normalization_params.npz',
             input_mean=in_mean, input_std=in_std,
             output_mean=out_mean, output_std=out_std)

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # Get hyperparameters
    hp = nn_config['MLP_hyperparams'] if args.useMLP else nn_config['hyperparams']
    hp['num_epochs'] = nn_config['n_epochs']
    print(f"Hyperparameters: {hp}")

    # Setup model
    model, optimizer, train_loader, val_loader, scheduler, es = setup_model_and_training(hp, train_dataset, val_dataset, input_features, output_features, nn_config['model_name'], reload=nn_config['reload'], reload_scheduler=nn_config['reload_scheduler'], batch_norm=False, useMLP=args.useMLP)
    print("Model and training setup complete.")

    # Train
    best_val_loss, history = train_model(model, optimizer, train_loader, val_loader, num_epochs=nn_config['n_epochs'], device=device, verbose=True, output_plots_dir=output_plots_dir,save_every_N=1, scheduler=scheduler, early_stopper=es, useMLP=args.useMLP)

    model_name = nn_config['model_name']
    torch.save(model.state_dict(), f'{output_dir}/{model_name}.pth')


