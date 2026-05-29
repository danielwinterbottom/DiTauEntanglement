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

class ParticleTransformerCondition(nn.Module):
    """
    Tokenises reco tau decay products as particles and encodes with a transformer.
    Expects input features ordered as in the LHC config:
        [taup_haspizero, taun_haspizero, met_px, met_py,
         taup_pi1_(px,py,pz,e,ipx,ipy,ipz),
         taun_pi1_(px,py,pz,e,ipx,ipy,ipz),
         taup_pizero1_(px,py,pz,e),
         taun_pizero1_(px,py,pz,e)]
    -> only works with DM0/1 currently
    """

    def __init__(self, context_dim=256, d_model=64, nhead=4, num_layers=3, dropout=0.0):
        super().__init__()
        self.pion_proj   = nn.Linear(7, d_model)   # 4-mom + IP (shared taup/taun)
        self.pizero_proj = nn.Linear(4, d_model)   # 4-mom only (shared taup/taun)
        self.met_proj    = nn.Linear(2, d_model)   # px, py
        self.type_emb    = nn.Embedding(5, d_model) # learned per-token-type embedding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu', norm_first=True,
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj  = nn.Linear(d_model, context_dim)

    def forward(self, x):
        B, device = x.shape[0], x.device
        type_embs = self.type_emb(torch.arange(5, device=device))  # [5, d_model]

        tokens = torch.stack([
            self.pion_proj(x[:, 4:11])    + type_embs[0],  # taup pion
            self.pion_proj(x[:, 11:18])   + type_embs[1],  # taun pion
            self.pizero_proj(x[:, 18:22]) + type_embs[2],  # taup pi0
            self.pizero_proj(x[:, 22:26]) + type_embs[3],  # taun pi0
            self.met_proj(x[:, 2:4])      + type_embs[4],  # MET
        ], dim=1)

        # True = ignore this token (PyTorch convention)
        pad_mask = torch.zeros(B, 5, dtype=torch.bool, device=device)
        pad_mask[:, 2] = ~x[:, 0].bool()  # taup pi0 absent when haspizero=0
        pad_mask[:, 3] = ~x[:, 1].bool()  # taun pi0 absent when haspizero=0

        out = self.transformer(tokens, src_key_padding_mask=pad_mask)

        # mean pool over present tokens only
        present = (~pad_mask).float().unsqueeze(-1)
        context = (out * present).sum(dim=1) / present.sum(dim=1)
        return self.output_proj(context)

# Lucas TODO: can be merged with main one probably
class ConditionalFlowTransformer(nn.Module):
    """
    Replacement for ConditionalFlow using a particle transformer
    condition network and NormalizingFlowNew as the flow.
    """

    def __init__(self, input_dim, context_dim=256, d_model=64, nhead=4,
                 num_transformer_layers=3, dropout=0.0, **flow_kwargs):
        super().__init__()
        self.condition_net = ParticleTransformerCondition(
            context_dim=context_dim, d_model=d_model,
            nhead=nhead, num_layers=num_transformer_layers, dropout=dropout,
        )
        self.flow = NormalizingFlowNew(
            input_size=input_dim,
            context_features=context_dim,
            **flow_kwargs
        )

    def log_prob(self, inputs, context):
        return self.flow.log_prob(inputs=inputs, context=self.condition_net(context))

    def sample(self, num_samples, context):
        return self.flow.sample(num_samples=num_samples, context=self.condition_net(context))

    def encode(self, inputs, context):
        cond_embed = self.condition_net(context)
        z, logabsdet = self.flow._transform.forward(inputs, context=cond_embed)
        return z, logabsdet

    def decode(self, z, context):
        cond_embed = self.condition_net(context)
        x, logabsdet = self.flow._transform.inverse(z, context=cond_embed)
        return x, logabsdet

    def sample_and_log_prob(self, num_samples, context):
        return self.flow.sample_and_log_prob(num_samples=num_samples, context=self.condition_net(context))


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
    

