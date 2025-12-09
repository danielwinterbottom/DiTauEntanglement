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
                    batch_norm=False):
    """Creates a normalizing flow model using nflows library."""
    transforms = []

    def create_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features, context_features=context_features,
            hidden_features=hidden_size, num_blocks=num_blocks,
            use_batch_norm=batch_norm,
            activation=nn.ReLU())

    def create_affine_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features, context_features=context_features,
            hidden_features=affine_hidden_size, num_blocks=affine_num_blocks,
            use_batch_norm=batch_norm,
            activation=nn.ReLU())

    for _ in range(num_layers):
        mask = nflows.utils.torchutils.create_mid_split_binary_mask(input_size) 
        # if features are ordered so that tau- features come first and then tau+ features and they have the same number 
        # then this mask will alternate which use tau is used to learn the spline parameters for the other tau  
        transforms.append(ReversePermutation(features=input_size))
        transforms.append(nflows.transforms.LULinear(input_size))
        transforms.append(AffineCouplingTransform(mask, create_affine_net))
        transforms.append(ReversePermutation(features=input_size))
        transforms.append(nflows.transforms.LULinear(input_size))
        transforms.append(PiecewiseRationalQuadraticCouplingTransform(mask, create_net, tails='linear', num_bins=num_bins, tail_bound=tail_bound))
        # add a LU triangular layer
        #transforms.append(nflows.transforms.LULinear(input_size))

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
                 batch_norm=False,
                 **flow_kwargs
    ):
        super().__init__()

        self.condition_net = nets.ResidualNet(
            in_features=raw_condition_dim,
            out_features=context_dim,
            hidden_features=cond_hidden_dim,
            num_blocks=2,
            activation=nn.ReLU(),
            use_batch_norm=batch_norm
        )

        self.flow = NormalizingFlow(
            input_size=input_dim,
            context_features=context_dim,
            **flow_kwargs
        )

    def log_prob(self, inputs, context):
        cond_embed = self.condition_net(context)
        return self.flow.log_prob(inputs=inputs, context=cond_embed)

    def sample(self, num_samples, context):
        cond_embed = self.condition_net(context)
        return self.flow.sample(num_samples=num_samples, context=cond_embed)