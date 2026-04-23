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
            extra_var_col = torch.where(extra_var_col == 1, 1, -1) * torch.sign(self.y)
            # multiply the extra var by the sign of y
            #self.X = self.X * torch.sign(self.y) * extra_var_col
            # add a column to X which =  torch.sign(self.y) * extra_var_col
            self.X = torch.cat((self.X.unsqueeze(1), extra_var_col.unsqueeze(1)), dim=1)


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
        if self.extra_var: return self.X[idx], self.y[idx].unsqueeze(0)
        else: return self.X[idx].unsqueeze(0), self.y[idx].unsqueeze(0)        


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

class QuantileLoss(nn.Module):
    def __init__(self, quantile: float):
        super().__init__()
        self.quantile = quantile

    def forward(self, preds, targets):
        errors = targets - preds
        loss = torch.max(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        return loss.mean()

class SimpleNN(nn.Module):

    def __init__(self, input_dim, output_dim, n_hidden_layers, n_nodes, sigmoid=False, activation='ReLU'):
        super(SimpleNN, self).__init__()

        layers = OrderedDict()
        layers['input'] = nn.Linear(input_dim, n_nodes)
        #layers['bn_input'] = nn.BatchNorm1d(n_nodes)
        if activation == 'LeakyReLU': layers['relu_input'] = nn.LeakyReLU(negative_slope=0.2)
        else: layers['relu_input'] = nn.ReLU()
        for i in range(n_hidden_layers):
            layers[f'hidden_{i+1}'] = nn.Linear(n_nodes, n_nodes)
            #layers[f'hidden_bn_{i+1}'] = nn.BatchNorm1d(n_nodes)
            if activation == 'LeakyReLU': layers[f'hidden_relu_{i+1}'] = nn.LeakyReLU(negative_slope=0.2)
            else: layers[f'hidden_relu_{i+1}'] = nn.ReLU()
        
        layers['output'] = nn.Linear(n_nodes, output_dim)
        if sigmoid:
            layers['sigmoid'] = nn.Sigmoid()

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
    elif args.loss.startswith('Quantile_'):
        # extract the quantile from the loss str
        quantile = float(args.loss.split('_')[1])
        criterion = QuantileLoss(quantile=quantile)
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

###model_double_degenerate = SimpleNN(input_dim=1, output_dim=1, n_hidden_layers=2, n_nodes=20).to(device)
###train_model(model_double_degenerate, dataloader_double_degenerate, n_epochs=50, lr=0.01)
###torch.save(model_double_degenerate.state_dict(), f'Nflow_example_NN_model_double_degenerate_{args.loss}.pth')

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
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.hist(X[:, 0], bins=50, density=False, alpha=0.6, color='b')
    plt.title('X1')
    plt.xlabel('X1')
    plt.ylabel('Density')
    plt.subplot(2, 2, 2)
    plt.hist(X[:, 1], bins=50, density=False, alpha=0.6, color='b')
    plt.title('X2')
    plt.xlabel('X2')
    plt.ylabel('Density')
    plt.subplot(2, 2, 3)
    plt.hist(y, bins=50, density=False, alpha=0.6, color='g')
    plt.title('y true')
    plt.xlabel('y true')
    plt.ylabel('Density')
    plt.subplot(2, 2, 4)
    plt.hist(predictions, bins=50, density=False, alpha=0.6, color='r')
    plt.title('y_pred')
    plt.xlabel('y pred')
    plt.ylabel('Density')
    plt.suptitle(title)
    plt.savefig(f"{title}.pdf")

    # make 2D plot of the data
    plt.figure(figsize=(8, 8))
    plt.hist2d(X[:, 0], y.squeeze(), bins=50, density=True, cmap='Blues')
    plt.colorbar()
    plt.title('X1 vs y true')
    plt.xlabel('X1')
    plt.ylabel('y true')
    plt.savefig(f"{title}_2D.pdf")

    # make 2D plot of the data
    plt.figure(figsize=(8, 8))
    plt.hist2d(X[:, 1], y.squeeze(), bins=50, density=True, cmap='Blues')
    plt.colorbar()
    plt.title('X2 vs y true')
    plt.xlabel('X2')
    plt.ylabel('y true')
    plt.savefig(f"{title}_2D.pdf")

    plt.figure(figsize=(8, 8))
    plt.hist2d(predictions.squeeze(), y.squeeze(), bins=50, density=True, cmap='Blues')
    plt.colorbar()
    plt.title('y pred vs y true')
    plt.xlabel('y pred')
    plt.ylabel('y true')
    plt.savefig(f"{title}_pred_vs_true_2D.pdf")

###plot_model_predictions(model_double_degenerate, dataloader_double_degenerate, title=f"model_predictions_{args.loss}", in_means=in_means, in_stds=in_stds, out_means=out_means, out_stds=out_stds)

# make plots for cases where there is some seperation between the signs:

#for eff in [0.5,0.7,0.9,1.0]:
for eff in [0.9]:

    print (f"Training model with extra var eff: {eff}")

    eff_str = str(eff).replace('.', 'p')

    dataset_double_seperable = MixtureGaussianDatasetDegenerate(n_samples=100000, device=device, extra_var=True, eff=eff)
    in_means_2, in_stds_2, out_means_2, out_stds_2 = None, None, None, None
    in_means_2, in_stds_2 = dataset_double_seperable.get_input_means_stds()
    out_means_2, out_stds_2 = dataset_double_seperable.get_output_means_stds()
    dataset_double_seperable.X = (dataset_double_seperable.X - in_means_2) / in_stds_2
    dataset_double_seperable.y = (dataset_double_seperable.y - out_means_2) / out_stds_2
    dataloader_double_seperable = DataLoader(dataset_double_seperable, batch_size=1024, shuffle=True)
    
    model_double_seperable = SimpleNN(input_dim=2, output_dim=1, n_hidden_layers=2, n_nodes=20).to(device)
    train_model(model_double_seperable, dataloader_double_seperable, n_epochs=50, lr=0.01)
    torch.save(model_double_seperable.state_dict(), f'NN_model_double_seperable_eff{eff_str}_{args.loss}.pth')
    plot_model_predictions(model_double_seperable, dataloader_double_seperable, title=f"model_predictions_double_seperable_eff{eff_str}_{args.loss}", in_means=in_means_2, in_stds=in_stds_2, out_means=out_means_2, out_stds=out_stds_2)

    continue

    # now training an adversarial method
    print('Training adversarial model...')
    
    G_model = SimpleNN(input_dim=2, output_dim=1, n_hidden_layers=2, n_nodes=20, activation='LeakyReLU').to(device)
    D_model = SimpleNN(input_dim=3, output_dim=1,n_hidden_layers=2, n_nodes=20, sigmoid=True).to(device)

    num_epochs = 10
    gamma = (0.0001/0.01)**(1/num_epochs)

    G_optimizer = torch.optim.Adam(G_model.parameters(), lr=0.001)
    D_optimizer = torch.optim.Adam(D_model.parameters(), lr=0.001)
    #G_scheduler = optim.lr_scheduler.ExponentialLR(G_optimizer, gamma=gamma)
    #D_scheduler = optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=gamma)
    
    # we could initialize the weights using the previous training if we want e.g:
    #try:
    #    G_model.load_state_dict(torch.load(f'NN_model_double_seperable_eff{eff_str}_{args.loss}.pth')) # loading best model first
    #except:
    #    G_model.load_state_dict(torch.load(f'NN_model_double_seperable_eff{eff_str}_{args.loss}.pth', map_location=torch.device('cpu')))

    adversarial_criterion = nn.BCELoss()

    
    for epoch in range(num_epochs):
        D_running_loss = 0.0
        G_running_loss = 0.0
        for i, (X, y_true) in enumerate(dataloader_double_seperable):

            # Update D network: minimize -(log(D(x)) + log(1 - D(G(z))))

#            D_optimizer.zero_grad()
#
#            # split X in half randomly
#            X = X.to(device)
#            y_true = y_true.to(device)
#            #split X and y_pred in half to produce real and fake
#            # For X_real we will add the true y values as additional columns
#            # For X_fake we will add the predicted y values from G_model as the columns
#            X_real = X[:int(X.size(0)/2)]
#            X_fake = X[int(X.size(0)/2):]
#            y_real = y_true[:int(X.size(0)/2)]
#            #X_real = X
#            #X_fake = X
#            #y_real = y_true
#            y_fake = G_model(X_fake)
#            X_real = torch.cat((X_real, y_real), dim=1)
#            # now we will generate the fake y values using the G_model
#            # now we will concatenate the X and y_fake values
#            X_fake = torch.cat((X_fake, y_fake), dim=1)
#
#
#            # now we create labels for real (=1) and fake (=0) data
#            y_real_labels = torch.ones(X_real.size(0)).unsqueeze(1)
#            y_fake_labels = torch.zeros(X_fake.size(0)).unsqueeze(1)
#            # move labels to device
#            y_real_labels = y_real_labels.to(device)
#            y_fake_labels = y_fake_labels.to(device)
#
#            D_loss_real = adversarial_criterion(D_model(X_real), y_real_labels)
#            D_loss_real.backward()
#            D_loss_fake = adversarial_criterion(D_model(X_fake.detach()), y_fake_labels)
#            D_loss_fake.backward()
#            D_loss = D_loss_real + D_loss_fake
#            D_optimizer.step()
#
#            D_running_loss += D_loss.item()
#
#            # Update G network: maximize log(D(G(z)))
#            G_optimizer.zero_grad()
#            y_fake_labels = torch.ones(y_fake.size(0)).unsqueeze(1) # fake labels are real for G cost
#            y_fake_labels = y_fake_labels.to(device)
#            G_loss = adversarial_criterion(D_model(X_fake), y_fake_labels)
#            G_loss.backward()
#            G_optimizer.step()
#
#            G_running_loss += G_loss.item()
#

            y_label_real = torch.ones(X.size(0)).unsqueeze(1).to(device)
            y_label_fake = torch.zeros(X.size(0)).unsqueeze(1).to(device)

            # train G:
            G_optimizer.zero_grad()
            X = X.to(device)
            y_true = y_true.to(device)
            y_fake = G_model(X)
            # concatenate X and y_fake
            X_fake = torch.cat((X, y_fake), dim=1)
            # create labels for real data
            G_loss = adversarial_criterion(D_model(X_fake), y_label_real)
            G_loss.backward()
            G_optimizer.step()
            G_running_loss += G_loss.item()

            # train D:
            D_optimizer.zero_grad()
            #print(f"X: {X.shape}, y_true: {y_true.shape}, y_fake: {y_fake.shape}")
            #print("y_label_real: ", y_label_real.shape, "y_label_fake: ", y_label_fake.shape)
            D_loss_real = adversarial_criterion(D_model(torch.cat((X, y_true), dim=1)), y_label_real)
            D_loss_fake = adversarial_criterion(D_model(torch.cat((X, y_fake.detach()), dim=1)), y_label_fake)

            D_loss = (D_loss_real + D_loss_fake)/2

            D_loss.backward()
            D_optimizer.step()
            D_running_loss += D_loss.item()

        D_running_loss /= len(dataloader_double_seperable)
        G_running_loss /= len(dataloader_double_seperable)
        #G_scheduler.step()
        #D_scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {D_running_loss:.4f}, G Loss: {G_running_loss:.4f}')

    torch.save(G_model.state_dict(), f'NN_adversarial_model_double_seperable_eff{eff_str}_{args.loss}.pth')
    plot_model_predictions(G_model, dataloader_double_seperable, title=f"adversarial_model_predictions_double_seperable_eff{eff_str}_{args.loss}", in_means=in_means_2, in_stds=in_stds_2, out_means=out_means_2, out_stds=out_stds_2)
      

