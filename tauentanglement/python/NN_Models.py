import torch
import torch.nn as nn
import nflows
from nflows.nn import nets
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.lu import LULinear
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform, AffineCouplingTransform
from nflows.flows.base import Flow


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

def MLP(input_size=8,
    num_blocks=3,
    hidden_size=64,
    output_size=8,
    batch_norm=False,
    activation=nn.ReLU()):

    # define a simple feed-forward neural network using resisual blocks
    def create_net(in_features, out_features):
        return nets.ResidualNet(
            in_features, out_features,
            hidden_features=hidden_size, num_blocks=num_blocks,
            use_batch_norm=batch_norm,
            activation=activation)

    net = create_net(input_size, output_size)
    return net

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

    def sample_and_log_prob(self, num_samples, context):
        """
        Sample and compute log-prob in a single forward pass.
        Returns samples [B, num_samples, F] and log_prob [B, num_samples].
        More efficient than separate sample() + log_prob() calls.
        """
        cond_embed = self.condition_net(context)
        return self.flow.sample_and_log_prob(num_samples=num_samples, context=cond_embed)
    

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
            if self.counter > self.patience:
                return True
        return False
