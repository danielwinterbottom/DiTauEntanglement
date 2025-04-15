import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import OrderedDict
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# first generate some random data which has a gaussian distribution
import argparse

class SmearedGaussianDataset(Dataset):
    def __init__(self, n_samples=10000, 
                 mu_y=0.0, sigma_y=0.1,
                 mu_noise=0.2, sigma_noise=0.2,
                 device='cpu'):
        super().__init__()
        self.device = device

        # Sample y ~ N(mu_y, sigma_y)
        self.y = torch.normal(mean=mu_y, std=sigma_y, size=(n_samples,), device=device)

        # Sample noise ~ N(mu_noise, sigma_noise)
        noise = torch.normal(mean=mu_noise, std=sigma_noise, size=(n_samples,), device=device)

        # Generate X = y + noise
        self.X = self.y + noise

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx].unsqueeze(0)

class MixtureGaussianDataset(Dataset):
    def __init__(self, n_samples=10000, 
                 mu1=-1.0, sigma1=0.1,
                 mu2=1.0, sigma2=0.1,
                 mix_prob=0.5,
                 mu_noise=0.1, sigma_noise=0.2,
                 device='cpu'):
        super().__init__()
        self.device = device

        # Decide which Gaussian to sample from
        component_choices = torch.bernoulli(torch.full((n_samples,), mix_prob)).to(device)

        # Sample y from the chosen component
        y1 = torch.normal(mu1, sigma1, size=(n_samples,), device=device)
        y2 = torch.normal(mu2, sigma2, size=(n_samples,), device=device)
        self.y = torch.where(component_choices == 1, y1, y2)

        # Add Gaussian noise
        noise = torch.normal(mu_noise, sigma_noise, size=(n_samples,), device=device)
        self.X = self.y + noise

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.y[idx].unsqueeze(0)

class MixtureGaussianDatasetDegenerate(Dataset):
    def __init__(self, n_samples=10000, 
                 sigma1=0.01,
                 sigma2=0.01,
                 mix_prob=0.5,
                 mu_noise=0.0, sigma_noise=0.2,
                 extra_var = False, eff=1.0,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.extra_var = extra_var
        mu1 = 1.0
        mu2 = -mu1
        # Decide which Gaussian to sample from
        component_choices = torch.bernoulli(torch.full((n_samples,), mix_prob)).to(device)

        # Sample y from the chosen component
        y1 = torch.normal(mu1, sigma1, size=(n_samples,), device=device)
        y2 = torch.normal(mu2, sigma2, size=(n_samples,), device=device)
        self.y = torch.where(component_choices == 1, y1, y2)

        # for the X we will assume that the model can't differentiate between + and - y values
        # X will equal the absolue value of y + noise

        # Add Gaussian noise
        noise = torch.normal(mu_noise, sigma_noise, size=(n_samples,), device=device)
        self.X = torch.abs(self.y) + noise

        # if extra_var is True then we will add a new variable that can differentiate between the +ve and -ve sign which some inaccuracy
        if extra_var:
            # the extra var will equal -1 or 1
            # sample a random number, if this is larger than 0.7 then set the new var sign to be equal to the sign of y, if not take the opposite sign
            extra_var_col = torch.bernoulli(torch.full((n_samples,), eff)).to(device)
            extra_var_col = torch.where(extra_var_col == 1, 1, -1)
            # multiply the extra var by the sign of y
            self.X = self.X * torch.sign(self.y) * extra_var_col


    def get_input_means_stds(self):
        """
        Compute the means and standard deviations of the input features.
        """
        means = self.X.mean(dim=0)
        stds = self.X.std(dim=0)
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
        #if self.extra_var: return self.X[idx], self.y[idx]
        return self.X[idx].unsqueeze(0), self.y[idx].unsqueeze(0)        


argparse = argparse.ArgumentParser(description='Nflow example')
argparse.add_argument('--loss', type=str, default='MSE', help='loss function to use: MSE, MAE')
#argparse.
args = argparse.parse_args()

dataset_double_degenerate = MixtureGaussianDatasetDegenerate(n_samples=100000, device=device)

in_means, in_stds = dataset_double_degenerate.get_input_means_stds()
out_means, out_stds = dataset_double_degenerate.get_output_means_stds()
dataset_double_degenerate.X = (dataset_double_degenerate.X - in_means) / in_stds
dataset_double_degenerate.y = (dataset_double_degenerate.y - out_means) / out_stds

dataloader_double_degenerate = DataLoader(dataset_double_degenerate, batch_size=512, shuffle=True)

# make a function to plot the data y and X seperatly, and the 2D histogram of the data
def plot_data(X, y, title="data_distribution"):
    plt.figure(figsize=(12, 6))

    # Plot y
    plt.subplot(1, 2, 1)
    plt.hist(y.cpu().numpy(), bins=50, density=True, alpha=0.6, color='g')
    plt.title('y Distribution')
    plt.xlabel('y')
    plt.ylabel('Density')

    # Plot X
    plt.subplot(1, 2, 2)
    plt.hist(X.cpu().numpy(), bins=50, density=True, alpha=0.6, color='b')
    plt.title('X Distribution')
    plt.xlabel('X')
    plt.ylabel('Density')

    plt.suptitle(title)
    plt.savefig(f"{title}.pdf")

# define NN to regress y given X

class SimpleNN(nn.Module):

    def __init__(self, input_dim, output_dim, n_hidden_layers, n_nodes):
        super(SimpleNN, self).__init__()

        layers = OrderedDict()
        layers['input'] = nn.Linear(input_dim, n_nodes)
        layers['bn_input'] = nn.BatchNorm1d(n_nodes)
        layers['relu_input'] = nn.ReLU()
        for i in range(n_hidden_layers):
            layers[f'hidden_{i+1}'] = nn.Linear(n_nodes, n_nodes)
            layers[f'hidden_bn_{i+1}'] = nn.BatchNorm1d(n_nodes)
            layers[f'hidden_relu_{i+1}'] = nn.ReLU()
        layers['output'] = nn.Linear(n_nodes, output_dim)

        self.layers = nn.Sequential(layers)

        # print a summry of the model
        print('Model summary:')
        print(self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x

# define training loop function that will loop over the data in batches and train the model

def train_model(model, dataloader, n_epochs=50, lr=0.01):
    print(f"Training model on {len(dataloader.dataset)} samples")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if args.loss == 'MAE': criterion = nn.L1Loss()
    elif args.loss == 'MSE': criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {args.loss}. Use 'MAE' or 'MSE'.")
    model.to(device)
    for epoch in range(n_epochs):
        for i, (X, y) in enumerate(dataloader):

            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            #lr = scheduler.get_last_lr()

        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
        #print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}, LR: {lr[0]:.6f}')
        #scheduler.step()

model_double_degenerate = SimpleNN(input_dim=1, output_dim=1, n_hidden_layers=2, n_nodes=20).to(device)
train_model(model_double_degenerate, dataloader_double_degenerate, n_epochs=50, lr=0.01)
torch.save(model_double_degenerate.state_dict(), f'Nflow_example_NN_model_double_degenerate_{args.loss}.pth')

# now apply the models to the data and plot the results
def plot_model_predictions(model, dataloader, title, in_means, in_stds, out_means, out_stds):
    model.eval()
    predictions = []
    X_vals = []
    y_vals = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            # Get model predictions
            y_pred = model(X)

            predictions.append(y_pred.cpu().numpy())
            X_vals.append(X.cpu().numpy())
            y_vals.append(y.cpu().numpy())

    # Concatenate all predictions   

    predictions = np.concatenate(predictions, axis=0)
    X = np.concatenate(X_vals, axis=0)
    y = np.concatenate(y_vals, axis=0)
    # undo standardization
    predictions = predictions * out_stds.numpy() + out_means.numpy()
    X = X * in_stds.numpy() + in_means.numpy()
    y = y * out_stds.numpy() + out_means.numpy()


    # plot the X, y, and y_pred histograms in seperate sub figures
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.hist(X[:, 0], bins=50, density=False, alpha=0.6, color='b')
    plt.title('X')
    plt.xlabel('X')
    plt.ylabel('Density')
    plt.subplot(1, 3, 2)
    plt.hist(y, bins=50, density=False, alpha=0.6, color='g')
    plt.title('y true')
    plt.xlabel('y')
    plt.ylabel('Density')
    plt.subplot(1, 3, 3)
    plt.hist(predictions, bins=50, density=False, alpha=0.6, color='r')
    plt.title('y_pred')
    plt.xlabel('y pred')
    plt.ylabel('Density')
    plt.suptitle(title)
    plt.savefig(f"{title}.pdf")

    # make 2D plot of the data
    plt.figure(figsize=(8, 8))
    plt.hist2d(X.squeeze(), y.squeeze(), bins=50, density=True, cmap='Blues')
    plt.colorbar()
    plt.title('X vs y true')
    plt.xlabel('X')
    plt.ylabel('y true')
    plt.savefig(f"{title}_2D.pdf")

    plt.figure(figsize=(8, 8))
    plt.hist2d(predictions.squeeze(), y.squeeze(), bins=50, density=True, cmap='Blues')
    plt.colorbar()
    plt.title('y pred vs y true')
    plt.xlabel('y pred')
    plt.ylabel('y true')
    plt.savefig(f"{title}_pred_vs_true_2D.pdf")

plot_model_predictions(model_double_degenerate, dataloader_double_degenerate, title=f"model_predictions_{args.loss}", in_means=in_means, in_stds=in_stds, out_means=out_means, out_stds=out_stds)

# make plots for cases where there is some seperation between the signs:

for eff in [0.5,0.7,0.9,1.0]:

    print (f"Training model with extra var eff: {eff}")

    eff_str = str(eff).replace('.', 'p')

    dataset_double_seperable = MixtureGaussianDatasetDegenerate(n_samples=100000, device=device, extra_var=True, eff=eff)
    #dataset_double_seperable = MixtureGaussianDataset(n_samples=100000)
    in_means_2, in_stds_2 = dataset_double_seperable.get_input_means_stds()
    out_means_2, out_stds_2 = dataset_double_seperable.get_output_means_stds()
    dataset_double_seperable.X = (dataset_double_seperable.X - in_means_2) / in_stds_2
    dataset_double_seperable.y = (dataset_double_seperable.y - out_means_2) / out_stds_2
    dataloader_double_seperable = DataLoader(dataset_double_seperable, batch_size=512, shuffle=True)
    
    model_double_seperable = SimpleNN(input_dim=1, output_dim=1, n_hidden_layers=2, n_nodes=20).to(device)
    train_model(model_double_seperable, dataloader_double_seperable, n_epochs=50, lr=0.01)
    torch.save(model_double_seperable.state_dict(), f'Nflow_example_NN_model_double_seperable_{args.loss}.pth')
    plot_model_predictions(model_double_seperable, dataloader_double_seperable, title=f"model_predictions_double_seperable_eff{eff_str}_{args.loss}", in_means=in_means_2, in_stds=in_stds_2, out_means=out_means_2, out_stds=out_stds_2)
    
    