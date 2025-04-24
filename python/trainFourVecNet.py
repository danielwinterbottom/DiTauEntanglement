import argparse
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))
from FourVecNN import *
from collections import OrderedDict
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class FourVecDataset(Dataset):
    def __init__(self, dataframe, output_features):
        # Convert entire dataset to tensors at initialization (avoids slow indexing)
        #self.X = torch.tensor(dataframe[['reco_taup_vis_px','reco_taup_vis_py','reco_taup_vis_pz','reco_taup_vis_e']].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[output_features].values, dtype=torch.float32)

        #particle_names = ['reco_taup_vis', 'reco_taun_vis', 'reco_Z']
        particle_names = ['reco_taup_pi1', 'reco_taup_pi2', 'reco_taup_pi3', 'reco_taup_pizero1',
                         'reco_taun_pi1', 'reco_taun_pi2', 'reco_taun_pi3', 'reco_taun_pizero1',  
                         'reco_Z']
        component_order = ['e', 'px', 'py', 'pz']
        #'delta_ipx','delta_ipy','delta_ipz','delta_ip_mag',
        features = []
        for part in particle_names:
            part_features = [dataframe[f"{part}_{comp}"].values for comp in component_order]
            stacked = torch.tensor(part_features).T  # shape: (num_events, 4)
            features.append(stacked)
        
        # the IP (between lead pi tracks) and SVs are defined as 3-vectors so we will take the first component as the magnitude
        # and the other three components as the x, y, and z components
        threevec_names = ['delta_sv','delta_ip']
        component_order = ['_mag', 'x', 'y', 'z']
        for part in threevec_names:
            part_features = [dataframe[f"{part}{comp}"].values for comp in component_order]
            # the x y anf z components need to be multiplied by the mag component
            part_features[1:] = [part_features[0] * part_features[i] for i in range(1, 4)]
            stacked = torch.tensor(part_features).T
            features.append(stacked)

        fourvec_tensor = torch.stack(features, dim=1)

        self.X = fourvec_tensor

    def get_input_means_stds_minmax(self):
        """
        Compute the means and standard deviations of the input features.
        We use the mean and std of the E components for the scaling
        """
        E = self.X[..., 0]
        means = E.mean(dim=0)
        stds = E.std(dim=0)
        minE = E.min(dim=0)[0]
        maxE = E.max(dim=0)[0]
        return means, stds, minE, maxE

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

class NN(nn.Module):

    def __init__(self, input_dim, output_dim, n_hidden_layers=1, n_nodes=10):
        super(NN, self).__init__()

        layers = OrderedDict()

        layers['input'] = FourVecLinear(input_dim, n_nodes)
        layers['input_activation'] = FourVecReLUE(n_nodes)
        for i in range(n_hidden_layers):
            layers[f'hidden_{i+1}'] = FourVecLinear(n_nodes, n_nodes)
            #layers[f'hidden_bn_{i+1}'] = FourVecBatchNormE(n_nodes)
            layers[f'hidden_activation_{i+1}'] = FourVecReLUE(n_nodes)
            #layers[f'hidden_activation_{i+1}'] = FourVecLeakyReLUE(n_nodes)
        layers['flatten'] = nn.Flatten(start_dim=1)
        layers['linear'] = nn.Linear(n_nodes * 4, n_nodes * 4)
        #layers['linear_bn'] = nn.BatchNorm1d(n_nodes * 4)
        layers['linear_activation'] = nn.ReLU()
        #layers['linear_activation'] = nn.LeakyReLU()
        layers['output'] = nn.Linear(n_nodes * 4, output_dim)

        self.layers = nn.Sequential(layers)

        # print a summry of the model
        print('Model summary:')
        print(self.layers)
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Number of parameters: {n_params}')
 

    def forward(self, x):
        x = self.layers(x)
        return x

def DeleteOldModels(model_name, val_loss_values, current_epoch):
    # check the models stored for previous epochs. If the validation loss for the current model is better then delete the previous models
    # search for all files in the current directory that start with the model name and end with .pth
    name_to_search = f'{model_name}_epoch_'
    files = [f for f in os.listdir('.') if f.startswith(name_to_search) and f.endswith('.pth')]
    current_loss = val_loss_values[-1]
    for file in files:
        # get the epoch number from the filename
        epoch = int(file.split('_')[-1].split('.')[0])
        if epoch >= current_epoch: continue
        # get the validation loss for that epoch
        val_loss = val_loss_values[epoch-1]
        # if the current loss is better than the previous loss then delete the previous model
        if current_loss < val_loss:
            os.remove(file)
            print(f'Deleted model {file} with validation loss {val_loss:.6f}')

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--stages', '-s', help='a list of stages to run', type=int, nargs="+")
    argparser.add_argument('--model_name', '-m', help='the name of the model output name', type=str, default='model')
    argparser.add_argument('--n_hidden_layers', help='number of hidden layers', type=int, default=6)
    argparser.add_argument('--n_nodes', help='number of nodes per layer', type=int, default=50)
    argparser.add_argument('--batch_size', help='batch size', type=int, default=1024)
    argparser.add_argument('--train_dsolution', help='train a specific solutions for the d sign (+/- 1 - other values default to the ordinary training)', default=None, type=int)
    argparser.add_argument('--loss', help='loss function to use options are MSE, MAE, or Huber', type=str, default='Huber')
    args = argparser.parse_args()

    standardize = 2 # 0: no standardization, 1: standardize input, 2: standardize input and output

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    #output_features = ['reco_taup_vis_px','reco_taup_vis_py','reco_taup_vis_pz','reco_taup_vis_e']

    print(f'Regressing output features: {output_features}')

    train_dataset = FourVecDataset(train_df, output_features)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = FourVecDataset(test_df, output_features)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_in_means, train_in_stds, train_in_min, train_in_max = train_dataset.get_input_means_stds_minmax()
    train_out_means, train_out_stds = train_dataset.get_output_means_stds()

    if standardize >= 1:
        # standardize the input 4-vectors based on the E component
        # scale 4-vectors by the max of the E component
        train_dataset.X = train_dataset.X / train_in_max.view(1, -1, 1)
        test_dataset.X = test_dataset.X / train_in_max.view(1, -1, 1)

    if standardize == 2:
        # normalize the output features
        train_dataset.y = (train_dataset.y - train_out_means) / train_out_stds
        test_dataset.y = (test_dataset.y - train_out_means) / train_out_stds

    # get the length of the second dimension
    N_input_features = train_dataset.X.shape[1]

    model = NN(N_input_features, len(output_features), args.n_hidden_layers, args.n_nodes)

    if 1 in args.stages:

        def plot_loss(loss_values, val_loss_values, running_loss_values=None):
            plt.figure()
            plt.plot(range(2, len(loss_values)+1), loss_values[1:], label='train loss')
            plt.plot(range(2, len(val_loss_values)+1), val_loss_values[1:], label='validation loss')
            if running_loss_values is not None: plt.plot(range(2, len(running_loss_values)+1), running_loss_values[1:], label='running loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            #print(f'saving plot: loss_vs_epoch_{args.model_name}.pdf')
            plt.savefig(f'loss_vs_epoch_{args.model_name}.pdf')
            plt.close()
            # also store the loss values in a text file
            with open(f'loss_values_{args.model_name}.txt', 'w') as f:
                f.write('epoch,train_loss,val_loss,running_loss\n')
                for i in range(len(loss_values)):
                    if running_loss_values is not None:
                        f.write(f'{i+1},{loss_values[i]},{val_loss_values[i]},{running_loss_values[i]}\n')
                    else:
                        f.write(f'{i+1},{loss_values[i]},{val_loss_values[i]},\n')

        def plot_learning_rate(learning_rate_values):
            plt.figure()
            plt.plot(range(1, len(learning_rate_values)+1), learning_rate_values)
            plt.xlabel('epoch')
            plt.ylabel('learning rate')
            plt.savefig(f'learning_rate_{args.model_name}.pdf')
            plt.close()
            # also store the learning rate values in a text file
            with open(f'learning_rate_values_{args.model_name}.txt', 'w') as f:
                f.write('epoch,learning_rate\n')
                for i in range(len(learning_rate_values)):
                    f.write(f'{i+1},{learning_rate_values[i]}\n')

        start_epochs = 2
        ramp_epochs = 5
        low_lr = 0.00001
        peak_lr = 0.001
        gamma = 0.98
        
        optimizer = optim.Adam(model.parameters(), lr=low_lr, weight_decay=1e-5)

        def combined_schedule(epoch):
            ramp_factor = peak_lr / low_lr
            if epoch < start_epochs:
                return 1.0  # phase 1: constant low_lr
            elif epoch < start_epochs + ramp_epochs:
                # phase 2: linear ramp
                ramp_progress = (epoch - start_epochs) / ramp_epochs
                return 1.0 + (ramp_factor - 1.0) * ramp_progress
            else:
                # phase 3: exponential decay from peak_lr
                decay_steps = epoch - (start_epochs + ramp_epochs)
                return ramp_factor * (gamma ** decay_steps)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=combined_schedule)

        if args.loss == 'MSE':
            print('Using MSE loss')
            criterion = nn.MSELoss()   
        elif args.loss == 'MAE':
            print('Using MAE loss')
            criterion = nn.L1Loss() 
        elif args.loss == 'Huber':
            print('Using Huber loss')
            criterion = nn.HuberLoss(delta=1.0)
        else: 
            raise ValueError(f"Unknown loss function: {args.loss}")


        model.to(device)

        loss_values = []
        val_loss_values = []
        running_loss_values = []
        learning_rate_values = []

        num_epochs = 100

        for epoch in range(num_epochs):
            #model.train()
            running_loss= 0.0
            running_loss_components = [0.0, 0.0, 0.0, 0.0]
            for i, (X, y) in enumerate(train_dataloader):
                # move data to GPU
                #print(X.shape)
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()


            running_loss /= len(train_dataloader)
            running_loss_values.append(running_loss)
            
            for i in range(len(running_loss_components)):
                running_loss_components[i] /= len(train_dataloader)
            
            # get the validation loss
            #model.eval()
            model.to(device)
            with torch.no_grad():
                val_loss = 0.0
                train_loss = 0.0
                for i, (X, y) in enumerate(train_dataloader):
                    X = X.to(device)
                    y = y.to(device)
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    train_loss += loss.item()
                train_loss /= len(train_dataloader)
                loss_values.append(train_loss)
                val_loss = 0.0
                for i, (X, y) in enumerate(test_dataloader):
                    X = X.to(device)
                    y = y.to(device)
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                val_loss /= len(test_dataloader)
                val_loss_values.append(val_loss)
                

            lr = scheduler.get_last_lr()
            learning_rate_values.append(lr[0])
            scheduler.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, running_loss: {running_loss:.6f}, lr = {lr[0]:.6f}')
            #if epoch > 1: 
            plot_loss(loss_values, val_loss_values, running_loss_values)
            plot_learning_rate(learning_rate_values)

            if (epoch+1) % 5 == 0:
                torch.save(model.state_dict(), f'{args.model_name}_epoch_{epoch+1}.pth')
                DeleteOldModels(args.model_name, val_loss_values, epoch)
                print(f'Model saved at epoch {epoch+1}')

        # save the model
        torch.save(model.state_dict(), f'{args.model_name}.pth')



    if 2 in args.stages:
        # load the model
        try:
            model.load_state_dict(torch.load(f'{args.model_name}.pth'))
        except:
            #load from cpu
            model.load_state_dict(torch.load(f'{args.model_name}.pth', map_location=torch.device('cpu')))

        # print the activation functions biases
        #print('Model biases:')
        #for name, param in model.named_parameters():
        #    if 'bias' in name and 'activation' in name:
        #        print(f'{name}: {param.data}')

        predictions = []
        # apply the model to the test set
        model.eval()

        with torch.no_grad():
            for i, (X, y) in enumerate(test_dataloader):
                outputs = model(X)
                if standardize == 2:
                    # convert predictions back to original scale
                    outputs = outputs * train_out_stds + train_out_means
                predictions.append(outputs.numpy())
        predictions = np.concatenate(predictions, axis=0)

        true_values = test_df[output_features]
        print(true_values)
        print(predictions)