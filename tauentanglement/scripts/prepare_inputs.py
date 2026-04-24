import yaml
import os
import argparse
from tauentanglement.python.DataProcessing import RegressionDataset, convert_root_to_parquet

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', help='path to the configuration file', type=str, default='tauentanglement/config/LEP.yaml', required=True)
    args = argparser.parse_args()

    config_file = args.config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)['Data']

    # prepare the datasets
    datasets_to_process = config['datasets']

    for key in datasets_to_process:
        file_path = config['available_files'][key]
        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist. Skipping dataset {key}.")
            continue
        # make output directory for the dataset
        os.makedirs(os.path.join(config['output_dir'], key), exist_ok=True)
        # convert root file to parquet
        df = convert_root_to_parquet(file_path, key, config)