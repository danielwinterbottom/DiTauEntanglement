import yaml
import os
import argparse
import multiprocessing
from taupolaris.python.DataProcessing import convert_root_to_parquet


def _prepare_one_dataset(file_path, key, config_data, collider, use_reco):
    """Runs in its own OS process (see main loop below), so all memory used
    while reading/converting this dataset's ROOT tree is fully reclaimed by
    the OS as soon as this process exits. Doing every dataset in one shared
    long-lived process doesn't work: pandas/numpy's allocator doesn't hand
    freed heap pages back to the OS eagerly, so RSS climbs across files even
    when each DataFrame is properly dereferenced -- a separate process per
    dataset is what actually bounds memory, not just closing file handles."""
    os.makedirs(os.path.join(config_data['output_dir'], key), exist_ok=True)
    convert_root_to_parquet(file_path, key, config_data, collider, use_reco=use_reco)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', help='path to the configuration file', type=str, default='taupolaris/config/LEP.yaml', required=True)
    args = argparser.parse_args()

    config_file = args.config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config_data = config['Data']

    # prepare the datasets
    datasets_to_process = config_data['datasets']
    collider = config['collider']

    for key in datasets_to_process:
        file_path = config_data['available_files'][key]
        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist. Skipping dataset {key}.")
            continue

        # one dataset at a time (not in parallel -- that would multiply peak
        # memory instead of bounding it) in a dedicated subprocess
        print(f">> Preparing dataset '{key}' in a separate process...")
        proc = multiprocessing.Process(
            target=_prepare_one_dataset,
            args=(file_path, key, config_data, collider, config_data['use_reco']),
        )
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Preparing dataset '{key}' failed (subprocess exit code {proc.exitcode}).")
        print(f">> Finished dataset '{key}'.")
