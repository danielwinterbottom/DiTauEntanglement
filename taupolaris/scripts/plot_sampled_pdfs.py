import torch
import argparse
import yaml
import os
import numpy as np
from taupolaris.python.NN_Tools import load_model, get_device
from taupolaris.python.DataProcessing import get_test_dataset
from taupolaris.python.Evaluation_Tools import save_sampled_pdfs_LHC, flow_map_predict
import matplotlib.pyplot as plt
import mplhep as hep
from tqdm import tqdm
plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 16})

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
    argparser.add_argument('--scan_pdf', action='store_true')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['Data']
    nn_config = config['SetupNN']

    device = torch.device('cpu') if args.useCPU else get_device()

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
    model = model.float().to(device)  # nflows' StandardNormal buffer is float64; MPS needs float32
    print(">> Successfully loaded model")

    norm_data = np.load(f'{output_dir}/normalization_params.npz')

    data_config['test_dataset'] = data_config['test_dataset'] if not isinstance(data_config['test_dataset'], list) else data_config['test_dataset'][0]
    data_config['test_output_name'] = args.output_name
    test_dataset, test_df, _, _ = get_test_dataset(data_config, norm_data, oneprong=False)

    outdir = f"{output_dir}/pdf_slices_sampled_{args.output_name}"
    os.makedirs(outdir, exist_ok=True)
    print(f"Saving plots to {outdir}/")

    map_method  = nn_config.get('map_method', 'gradient')
    map_num_draws = nn_config.get('map_num_draws', 100)

    for event_number in args.events:
        print(f"  Sampling event {event_number}...")

        map_value = None
        if not args.no_map:
            X_event = test_dataset.X[event_number].unsqueeze(0).to(device)
            map_norm, map_pred, log_prob_z = flow_map_predict(
                model, X_event,
                test_dataset=test_dataset,
                num_draws=map_num_draws,
                chunk_size=1,
                method=map_method,
                return_log_prob=True
            )
            map_value = map_pred[0]


            print("\n\nINFO!!!")
            print(f"At best standardised x ({map_norm}), found likelihood: {log_prob_z}")
            print("-"*60, '\n')

            if args.scan_pdf:
                # SCAN LOG
                for tau in ['taun', 'taup']:
                    if tau == 'taun':
                        scan_idx = 5 # 6th element
                    else:
                        scan_idx = 2 # 3rd element
                    x_map_norm = map_norm[0].to(device)
                    shifts = np.arange(-1, 1, 0.01)
                    scan_log_p = []
                    x_shifted_phys = []
                    print(f">> Profiling likelihood for k in {tau} (others frozen)")
                    with torch.no_grad():
                        for shift in tqdm(shifts):
                            x_shifted = x_map_norm.clone()
                            x_shifted[scan_idx] += float(shift)
                            z, logabsdet_enc = model.encode(x_shifted.unsqueeze(0), context=X_event)
                            log_pz = -0.5 * (z ** 2).sum(dim=-1)
                            log_p = log_pz + logabsdet_enc
                            # print(f">> for x ({x_shifted}) have logp: {log_p.item()}")
                            scan_log_p.append((log_p).item())
                            x_shifted_phys.append(test_dataset.destandardize_outputs(x_shifted.unsqueeze(0).cpu())[0][scan_idx])

                    scan_log_p = np.array(scan_log_p)

                    # plot the scan
                    x_range = [x_map_norm[scan_idx].item()+shift for shift in shifts]

                    fig,ax = plt.subplots(figsize=(6,6))
                    plt.plot(x_range, scan_log_p)
                    ax.axhline(y=float(log_prob_z), color='red', linestyle='--', label='MAP log p')
                    ax.legend()
                    plt.xlabel(f"k component {tau} (entry {scan_idx+1}) - standardised")
                    plt.ylabel('log(p)')
                    plt.savefig(os.path.join(outdir, f"profile_event{event_number}_{tau}_standardised.pdf"))
                    # unstandardised
                    fig,ax = plt.subplots(figsize=(6,6))
                    plt.plot(x_shifted_phys, scan_log_p)
                    ax.axhline(y=float(log_prob_z), color='red', linestyle='--', label='MAP log p')
                    ax.legend()
                    plt.xlabel(f"k component {tau} (entry {scan_idx+1}) - physical")
                    plt.ylabel('log(p)')
                    plt.savefig(os.path.join(outdir, f"profile_event{event_number}_{tau}_physical.pdf"))



        # Scan
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
            df=test_df,
            coordinates=data_config['coordinates']
        )

    print("Done.")


if __name__ == '__main__':
    main()
