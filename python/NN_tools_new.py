import torch
import torch.nn as nn
import nflows
from nflows.nn import nets
from torch.utils.data import Dataset
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.lu import LULinear
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform, AffineCouplingTransform
from nflows.flows.base import Flow

class RegressionDataset(Dataset):
    def __init__(self, dataframe, input_features, output_features, 
                 input_mean=None, input_std=None, 
                 output_mean=None, output_std=None,
                 normalize_inputs=True, normalize_outputs=False, eps=1e-8):
        """
        A regression dataset that can standardize features using provided means/stds.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data.
        input_features : list[str]
            Names of input columns.
        output_features : list[str]
            Names of target columns.
        input_mean, input_std : torch.Tensor or None
            If given, used for input normalization.
            If None, computed from the current dataframe.
        output_mean, output_std : torch.Tensor or None
            Same as above, for outputs.
        normalize_inputs, normalize_outputs : bool
            Whether to apply standardization.
        eps : float
            Small value to prevent division by zero.
        """
        X = torch.tensor(dataframe[input_features].values, dtype=torch.float32)
        y = torch.tensor(dataframe[output_features].values, dtype=torch.float32)

        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.eps = eps

        # ---- Input normalization ----
        if normalize_inputs:
            if input_mean is None or input_std is None:
                self.input_mean = X.mean(dim=0, keepdim=True)
                self.input_std = X.std(dim=0, keepdim=True).clamp_min(eps)
            else:
                self.input_mean = input_mean
                self.input_std = input_std.clamp_min(eps)

            X = (X - self.input_mean) / self.input_std
        else:
            self.input_mean = torch.zeros(X.shape[1])
            self.input_std = torch.ones(X.shape[1])

        # ---- Output normalization ----
        if normalize_outputs:
            if output_mean is None or output_std is None:
                self.output_mean = y.mean(dim=0, keepdim=True)
                self.output_std = y.std(dim=0, keepdim=True).clamp_min(eps)
            else:
                self.output_mean = output_mean
                self.output_std = output_std.clamp_min(eps)

            y = (y - self.output_mean) / self.output_std
        else:
            self.output_mean = torch.zeros(y.shape[1])
            self.output_std = torch.ones(y.shape[1])

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def destandardize_outputs(self, y_norm):
        """
        Convert standardized outputs back to physical units.
        """
        device = y_norm.device
        return y_norm * self.output_std.to(device) + self.output_mean.to(device)


class MorphDataset(Dataset):
    def __init__(self, dataframe, input_features, output_features, context_features,
                 input_mean=None, input_std=None, 
                 output_mean=None, output_std=None,
                 context_mean=None, context_std=None,
                 normalize_inputs=False, normalize_outputs=False, normalize_context=True, eps=1e-8):
        """
        A regression dataset that can standardize features using provided means/stds.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input data.
        input_features : list[str]
            Names of input columns.
        output_features : list[str]
            Names of target columns.
        context_features : list[str]
            Names of context columns.
        input_mean, input_std : torch.Tensor or None
            If given, used for input normalization.
            If None, computed from the current dataframe.
        output_mean, output_std : torch.Tensor or None
            Same as above, for outputs.
        normalize_inputs, normalize_outputs : bool
            Whether to apply standardization.
        eps : float
            Small value to prevent division by zero.
        """
        X = torch.tensor(dataframe[input_features].values, dtype=torch.float32)
        y = torch.tensor(dataframe[output_features].values, dtype=torch.float32)
        c = torch.tensor(dataframe[context_features].values, dtype=torch.float32)

        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.eps = eps

        # ---- Context normalization ----
        if normalize_context:
            if context_mean is None or context_std is None:
                self.context_mean = c.mean(dim=0, keepdim=True)
                self.context_std = c.std(dim=0, keepdim=True).clamp_min(eps)
            else:
                self.context_mean = context_mean
                self.context_std = context_std.clamp_min(eps)

            c = (c - self.context_mean) / self.context_std
        else:
            self.context_mean = torch.zeros(c.shape[1])
            self.context_std = torch.ones(c.shape[1])

        ## ---- Input normalization ----
        #if normalize_inputs:
        #    if input_mean is None or input_std is None:
        #        self.input_mean = X.mean(dim=0, keepdim=True)
        #        self.input_std = X.std(dim=0, keepdim=True).clamp_min(eps)
        #    else:
        #        self.input_mean = input_mean
        #        self.input_std = input_std.clamp_min(eps)
#
        #    X = (X - self.input_mean) / self.input_std
        #else:
        #    self.input_mean = torch.zeros(X.shape[1])
        #    self.input_std = torch.ones(X.shape[1])

        # ---- Output normalization ----
        if normalize_outputs:
            if output_mean is None or output_std is None:
                self.output_mean = y.mean(dim=0, keepdim=True)
                self.output_std = y.std(dim=0, keepdim=True).clamp_min(eps)
            else:
                self.output_mean = output_mean
                self.output_std = output_std.clamp_min(eps)

            y = (y - self.output_mean) / self.output_std
        else:
            self.output_mean = torch.zeros(y.shape[1])
            self.output_std = torch.ones(y.shape[1])

        # normalize inputs using same means as if used for outputs
        if normalize_outputs:
            X = (X - self.output_mean) / self.output_std  
            #TODO: trying to understand is these really need to be standardized in consistent way with the outputs  
        self.input_mean = self.output_mean
        self.input_std = self.output_std


        self.c = c
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.c[idx]

    def destandardize_outputs(self, y_norm):
        """
        Convert standardized outputs back to physical units.
        """
        device = y_norm.device
        return y_norm * self.output_std.to(device) + self.output_mean.to(device)

# normalizing flow definitions

def NormalizingFlow(input_size=8, 
                    context_features=6, 
                    num_layers=8, 
                    num_bins=8, 
                    tail_bound=2.0, 
                    hidden_size=64, 
                    num_blocks=2,
                    affine_hidden_size=64,
                    affine_num_blocks=2,
                    batch_norm=False,
                    #activation=nn.PReLU()):
                    #activation=nn.LeakyReLU(0.1)):
                    activation=nn.ReLU()):
    """Creates a normalizing flow model using nflows library."""
    transforms = []

    def create_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features, context_features=context_features,
            hidden_features=hidden_size, num_blocks=num_blocks,
            use_batch_norm=batch_norm,
            activation=activation)

    def create_affine_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features, context_features=context_features,
            hidden_features=affine_hidden_size, num_blocks=affine_num_blocks,
            use_batch_norm=batch_norm,
            activation=activation)

    for i in range(num_layers):
        mask = nflows.utils.torchutils.create_mid_split_binary_mask(input_size) 
        # if features are ordered so that tau- features come first and then tau+ features and they have the same number 
        # then this mask will alternate which use tau is used to learn the spline parameters for the other tau  
        transforms.append(ReversePermutation(features=input_size))
        transforms.append(nflows.transforms.LULinear(input_size))
        transforms.append(AffineCouplingTransform(mask, create_affine_net))
        transforms.append(ReversePermutation(features=input_size))
        transforms.append(nflows.transforms.LULinear(input_size))
        transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask, create_net, tails='linear', num_bins=num_bins, tail_bound=tail_bound))

    transform = CompositeTransform(transforms)
    distribution = StandardNormal([input_size])
    flow = Flow(transform, distribution)
    return flow 

def NormalizingFlowNew(input_size=8, 
                    context_features=6, 
                    num_layers=8, 
                    num_bins=8, 
                    tail_bound=2.0, 
                    hidden_size=64, 
                    num_blocks=2,
                    batch_norm=False,
                    #activation=nn.PReLU()):
                    #activation=nn.LeakyReLU(0.1)):
                    activation=nn.ReLU()):
    """Creates a normalizing flow model using nflows library."""
    transforms = []

    def create_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features, context_features=context_features,
            hidden_features=hidden_size, num_blocks=num_blocks,
            use_batch_norm=batch_norm,
            activation=activation)

    mask = nflows.utils.torchutils.create_mid_split_binary_mask(input_size)

    for i in range(num_layers):
        if i > 0: transforms.append(nflows.transforms.LULinear(input_size))
        transforms.append(ReversePermutation(features=input_size))
        transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask, create_net, tails='linear', num_bins=num_bins, tail_bound=tail_bound))
        transforms.append(ReversePermutation(features=input_size))
        transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask, create_net, tails='linear', num_bins=num_bins, tail_bound=tail_bound))

    transform = CompositeTransform(transforms)
    distribution = StandardNormal([input_size])
    flow = Flow(transform, distribution)
    return flow 

class ConditionalFlow(nn.Module):
    def __init__(self, 
                 input_dim,
                 raw_condition_dim,
                 context_dim,
                 cond_hidden_dim=64,
                 cond_num_blocks=2,
                 batch_norm=False,
                 activation=nn.ReLU(),
                 **flow_kwargs
    ):
        super().__init__()

        # if cond_num_blocks = 0 we just skip the condition network
        if cond_num_blocks == 0:
            self.condition_net = nn.Identity()
            context_dim = raw_condition_dim

        else:
            self.condition_net = nets.ResidualNet(
                in_features=raw_condition_dim,
                out_features=context_dim,
                hidden_features=cond_hidden_dim,
                num_blocks=cond_num_blocks,
                activation=activation,
                use_batch_norm=batch_norm
            )

        self.flow = NormalizingFlowNew(
            input_size=input_dim,
            context_features=context_dim,
            activation=activation,
            **flow_kwargs
        )

    def log_prob(self, inputs, context):
        cond_embed = self.condition_net(context)
        return self.flow.log_prob(inputs=inputs, context=cond_embed)

    def sample(self, num_samples, context):
        cond_embed = self.condition_net(context)
        return self.flow.sample(num_samples=num_samples, context=cond_embed)

    def encode(self, inputs, context):
        """
        Map data space -> latent space (x -> z)
        """
        cond_embed = self.condition_net(context)
        z, logabsdet = self.flow._transform.forward(inputs, context=cond_embed)
        return z, logabsdet

    def decode(self, z, context):
        """
        Map latent space -> data space (z -> x)
        """
        cond_embed = self.condition_net(context)
        x, logabsdet = self.flow._transform.inverse(z, context=cond_embed)
        return x, logabsdet
    

class ConditionalMorphingFlow(nn.Module):
    """
    A pair of conditional normalizing flows following the same pattern as your ConditionalFlow:
      - self.condition_net produces context embeddings
      - self.flow_data  models analytic estimates    p_data(x | context)
      - self.flow_truth models MC truth distribution p_truth(x | context)
    Both flows map x -> z into the SAME latent normal distribution.
    
    Flow-based morphing:
        z_data = f_data.forward(x_data | context)
        x_corr = f_truth.inverse(z_data | context)
    """

    def __init__(
        self,
        input_dim,
        raw_condition_dim,
        context_dim,
        cond_hidden_dim=64,
        cond_num_blocks=2,
        batch_norm=False,
        activation=nn.ReLU(),
        flow_kwargs_data=None,      # kwargs for data flow
        flow_kwargs_truth=None,     # kwargs for truth flow
    ):
        super().__init__()

        if flow_kwargs_data is None:
            flow_kwargs_data = {}
        if flow_kwargs_truth is None:
            flow_kwargs_truth = {}

        # ----------------------------------------------------
        # 1. Condition network
        # ----------------------------------------------------
        self.condition_net = nets.ResidualNet(
            in_features=raw_condition_dim,
            out_features=context_dim,
            hidden_features=cond_hidden_dim,
            num_blocks=cond_num_blocks,
            activation=activation,
            use_batch_norm=batch_norm
        )

        # ----------------------------------------------------
        # 2. Two conditional flows
        # ----------------------------------------------------
        self.flow_data = NormalizingFlowNew(
            input_size=input_dim,
            context_features=context_dim,
            **flow_kwargs_data
        )

        self.flow_truth = NormalizingFlowNew(
            input_size=input_dim,
            context_features=context_dim,
            **flow_kwargs_truth
        )

    # ======================================================
    #  LOG PROBABILITIES
    # ======================================================

    def log_prob_data(self, inputs, context):
        """Log-probability under the analytic flow."""
        cond_embed = self.condition_net(context)
        return self.flow_data.log_prob(inputs=inputs, context=cond_embed)

    def log_prob_truth(self, inputs, context):
        """Log-probability under the truth flow."""
        cond_embed = self.condition_net(context)
        return self.flow_truth.log_prob(inputs=inputs, context=cond_embed)

    # ======================================================
    #  SAMPLING
    # ======================================================

    def sample_data(self, num_samples, context):
        cond_embed = self.condition_net(context)
        return self.flow_data.sample(num_samples=num_samples, context=cond_embed)

    def sample_truth(self, num_samples, context):
        cond_embed = self.condition_net(context)
        return self.flow_truth.sample(num_samples=num_samples, context=cond_embed)

    # ======================================================
    #  FLOW-BASED MORPHING (DATA → TRUTH)
    # ======================================================

    def morph_data_to_truth(self, x_data, context):
        """
        Takes analytic (data-like) inputs x_data and returns a corrected version
        distributed like the truth flow:
        
            z = f_data.forward(x_data | context)
            x_corr = f_truth.inverse(z | context)
        """
        with torch.no_grad():
            cond_embed = self.condition_net(context)

            # x -> z using data flow
            z, _ = self.flow_data._transform.forward(
                x_data, context=cond_embed
            )

            # z -> corrected x using inverse of truth flow
            x_corr, _ = self.flow_truth._transform.inverse(
                z, context=cond_embed
            )

        return x_corr

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def reset(self):
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss)*(1. + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False