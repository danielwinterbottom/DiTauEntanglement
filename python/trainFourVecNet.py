import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class FourVecDataset(Dataset):
    def __init__(self, dataframe, output_features):
        # Convert entire dataset to tensors at initialization (avoids slow indexing)
        #self.X = torch.tensor(dataframe[['reco_taup_vis_px','reco_taup_vis_py','reco_taup_vis_pz','reco_taup_vis_e']].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[output_features].values, dtype=torch.float32)

        particle_names = ['reco_taup_vis', 'reco_taun_vis', 'reco_Z']
        component_order = ['e', 'px', 'py', 'pz']
        features = []
        for part in particle_names:
            part_features = [dataframe[f"{part}_{comp}"].values for comp in component_order]
            stacked = torch.tensor(part_features).T  # shape: (num_events, 4)
            features.append(stacked)

        fourvec_tensor = torch.stack(features, dim=1)

        self.X = fourvec_tensor

        means, std = self.get_input_means_stds()

    def get_input_means_stds(self):
        """
        Compute the means and standard deviations of the input features.
        We use the mean and std of the E components for the scaling
        """
        E = self.X[..., 0]
        means = E.mean(dim=0)
        stds = E.std(dim=0)
        return means, stds

    def get_output_means_stds(self):
        """
        Compute the means and standard deviations of the output features.
        """
        means = self.y.mean(dim=0)
        stds = self.y.std(dim=0)
        return means, stds

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--stages', '-s', help='a list of stages to run', type=int, nargs="+")
    argparser.add_argument('--model_name', '-m', help='the name of the model output name', type=str, default='model')
    argparser.add_argument('--n_hidden_layers', help='number of hidden layers', type=int, default=6)
    argparser.add_argument('--n_nodes', help='number of nodes per layer', type=int, default=300)
    argparser.add_argument('--batch_size', help='batch size', type=int, default=1024)
    argparser.add_argument('--train_dsolution', help='train a specific solutions for the d sign (+/- 1 - other values default to the ordinary training)', default=None, type=int)
    argparser.add_argument('--loss', help='loss function to use options are MSE, MAE, or Huber', type=str, default='MSE')
    args = argparser.parse_args()

    file_paths = [f'output/df_chunk_{i}.pkl' for i in range(20)]
    # only read the input and output features when reading the dfs

    dataframes = [pd.read_pickle(file) for file in file_paths]
    train_df = pd.concat(dataframes[:1], ignore_index=True)
    test_df = pd.concat(dataframes[1:2], ignore_index=True)

    if args.train_dsolution == 1:
        # regress the dplus solution
        output_features = [
            'dplus_taup_nu_px','dplus_taup_nu_py','dplus_taup_nu_pz',
            'dplus_taun_nu_px','dplus_taun_nu_py','dplus_taun_nu_pz']
    elif args.train_dsolution == -1:
        # regress the dminus solution
        output_features = [
            'dminus_taup_nu_px','dminus_taup_nu_py','dminus_taup_nu_pz',
            'dminus_taun_nu_px','dminus_taun_nu_py','dminus_taun_nu_pz']
    elif args.train_dsolution == 0:
        # regress the d0 solution
        output_features = [
            'd0_taup_nu_px','d0_taup_nu_py','d0_taup_nu_pz',
            'd0_taun_nu_px','d0_taun_nu_py','d0_taun_nu_pz']
    else: 
        output_features = [
            'taup_nu_px','taup_nu_py','taup_nu_pz',
            'taun_nu_px','taun_nu_py','taun_nu_pz']

    train_dataset = FourVecDataset(train_df, output_features)