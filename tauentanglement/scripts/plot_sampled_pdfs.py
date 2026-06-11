import torch
import argparse
import yaml
import os
import numpy as np
from tauentanglement.python.NN_Tools import load_model
from tauentanglement.python.DataProcessing import get_test_dataset
from tauentanglement.python.Evaluation_Tools import save_sampled_pdfs_LHC, flow_map_predict


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', required=True, type=str,
                           help='path to the config file used for training/evaluation')
    argparser.add_argument('--output_name', '-o', required=True, type=str,
                           help='test_output_name used in evaluate.py (sets output subdirectory name)')
    argparser.add_argument('--events', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                           help='event indices to plot')
    argparser.add_argument('--num_samples', type=int, default=50000,
                           help='number of flow samples per event')
    argparser.add_argument('--bins', type=int, default=100)
    argparser.add_argument('--no_map', action='store_true',
                           help='skip MAP estimate overlay')
    argparser.add_argument('--useCPU', action='store_true')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']
    nn_config = config['SetupNN']

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.useCPU else 'cpu')

    output_dir = f"outputs_{nn_config['model_name']}"
    output_plots_dir = f"{output_dir}/plots"

    input_features = data_config['Features']['input_features']
    output_features = data_config['Features']['output_features'][data_config['coordinates']]
    is_transformer = nn_config.get('use_transformer', False)
    leptonic_mode = data_config.get('leptonic_mode', 0)
    hp = nn_config['hyperparams']

    model = load_model(hp, input_features, output_features, batch_norm=False,
                       useMLP=False, useTransformer=is_transformer,
                       useTransformerMLP=False, leptonic_mode=leptonic_mode)

    model_path = f'{output_plots_dir}/best_model.pth'
    if not os.path.exists(model_path):
        model_path = f'{output_plots_dir}/partial_model.pth'
    print(f"Loading model from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path))
    except Exception:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model = model.to(device)
    print(">> Successfully loaded model")

    norm_data = np.load(f'{output_dir}/normalization_params.npz')

    data_config['test_dataset'] = data_config['test_dataset'] if not isinstance(data_config['test_dataset'], list) else data_config['test_dataset'][0]
    data_config['test_output_name'] = args.output_name
    test_dataset, _, _, _ = get_test_dataset(data_config, norm_data, oneprong=False)

    outdir = f"{output_dir}/pdf_slices_sampled_{args.output_name}"
    print(f"Saving plots to {outdir}/")

    map_method  = nn_config.get('map_method', 'gradient')
    map_num_draws = nn_config.get('map_num_draws', 100)

    for event_number in args.events:
        print(f"  Sampling event {event_number}...")

        map_value = None
        if not args.no_map:
            X_event = test_dataset.X[event_number].unsqueeze(0).to(device)
            _, map_pred = flow_map_predict(
                model, X_event,
                test_dataset=test_dataset,
                num_draws=map_num_draws,
                chunk_size=1,
                method=map_method,
            )
            map_value = map_pred[0]

        save_sampled_pdfs_LHC(
            model=model,
            device=device,
            dataset=test_dataset,
            output_features=output_features,
            event_number=event_number,
            num_samples=args.num_samples,
            bins=args.bins,
            outdir=outdir,
            map_value=map_value,
        )

    print("Done.")


if __name__ == '__main__':
    main()
