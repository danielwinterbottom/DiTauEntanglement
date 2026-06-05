import torch
import argparse
from tauentanglement.python.DataProcessing import get_train_val_test_datasets
from tauentanglement.python.NN_Tools import setup_model_and_training, train_model
import yaml
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def powers_of_two(min_power, max_power):
    return [2 ** n for n in range(min_power, max_power + 1)]


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', help='path to the configuration file', type=str, required=True)
    argparser.add_argument('--n_trials', '-t', help='number of Optuna hyperparameter optimisation trials', type=int, default=100)
    argparser.add_argument('--useMLP', help='whether to use a simple MLP instead of a normalizing flow', action='store_true')
    argparser.add_argument('--loadDS', help='whether to load existing train/val datasets or recreate', action='store_true')
    args = argparser.parse_args()

    import optuna
    import optuna.visualization.matplotlib as optplt

    config_file = args.config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']
    nn_config = config['SetupNN']

    output_dir = f"outputs_{nn_config['model_name']}_OPTIMISATION"
    output_plots_dir = f"{output_dir}/plots"
    os.makedirs(output_plots_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datasets = data_config['datasets']
    train_dataset, val_dataset, input_features, output_features = get_train_val_test_datasets(datasets, data_config, load_existing=args.loadDS)

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # n_epochs_opt lets you run fewer epochs per trial than full training to speed up the search
    n_epochs_opt = nn_config.get('n_epochs_opt', nn_config['n_epochs'])
    print(f"Running {args.n_trials} trials with {n_epochs_opt} epochs each.")

    def live_plot_callback(study, trial):
        plt.figure(figsize=(6, 4))
        optplt.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(output_plots_dir, "optimization_history.pdf"))
        plt.close()

        losses = np.array([t.value for t in study.trials if t.value is not None])
        high_cut = np.quantile(losses, 0.95)
        trimmed_trials = [t for t in study.trials if t.value is not None and t.value <= high_cut]
        filtered_study = optuna.create_study(direction=study.direction)
        filtered_study.add_trials(trimmed_trials)

        plt.figure(figsize=(6, 4))
        optplt.plot_optimization_history(filtered_study)
        plt.tight_layout()
        plt.savefig(os.path.join(output_plots_dir, "optimization_history_filtered.pdf"))
        plt.close()

    def objective(trial):
        print(f"\nStarting trial {trial.number}...")

        hp = {
            'batch_size': trial.suggest_categorical('batch_size', powers_of_two(13, 15)),
            'num_layers': trial.suggest_int('num_layers', 2, 10),
            'num_bins': trial.suggest_int('num_bins', 8, 24, step=4),
            'tail_bound': trial.suggest_float('tail_bound', 3.0, 5.0, step=1.0),
            'hidden_size': trial.suggest_categorical('hidden_size', powers_of_two(6, 9)),
            'num_blocks': trial.suggest_int('num_blocks', 1, 4),
            'lr': trial.suggest_categorical('lr', [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'condition_net_hidden_size': trial.suggest_categorical('condition_net_hidden_size', powers_of_two(6, 9)),
            'condition_net_num_blocks': trial.suggest_int('condition_net_num_blocks', 0, 4),
            'condition_net_output_size': trial.suggest_categorical('condition_net_output_size', powers_of_two(5, 8)),
            'num_epochs': 1,
        }

        model, optimizer, train_loader, val_loader, scheduler, es = setup_model_and_training(
            hp, train_dataset, val_dataset, input_features, output_features,
            nn_config['model_name']+'_OPTIMISATION', verbose=False, useMLP=args.useMLP
        )

        num_params = sum(p.numel() for p in model.parameters())
        trial.set_user_attr("n_params", num_params)

        best_val_loss, history = train_model(
            model, optimizer, train_loader, val_loader,
            num_epochs=n_epochs_opt, device=device, verbose=False,
            output_plots_dir=None, save_every_N=None, scheduler=scheduler,
            recompute_train_loss=False, early_stopper=es, useMLP=args.useMLP,
            optuna_trial=trial
        )

        best_trial_loss_path = os.path.join(output_dir, "best_trial_loss.txt")
        if os.path.exists(best_trial_loss_path):
            with open(best_trial_loss_path, "r") as f:
                best_trial_loss = float(f.read())
        else:
            best_trial_loss = float("inf")
        if best_val_loss < best_trial_loss:
            with open(best_trial_loss_path, "w") as f:
                f.write(str(best_val_loss))
            torch.save(model.state_dict(), os.path.join(output_dir, "best_trial_model.pth"))

        return best_val_loss

    db_path = os.path.join(output_dir, "nn_optuna_study.db")
    study = optuna.create_study(
        study_name=nn_config['model_name'] + "_HyperparamOPTIMISATION",
        direction="minimize",
        storage=f"sqlite:///{db_path}?timeout=10000",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    )
    study.optimize(objective, n_trials=args.n_trials, callbacks=[live_plot_callback])

    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Validation loss: {best_trial.value:.6f}")
    print("  Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
