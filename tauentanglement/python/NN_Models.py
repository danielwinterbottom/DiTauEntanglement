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
import time

# Model Definitions

def NormalizingFlow(input_size=8,
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
    Tokenises reco tau decay products and encodes with a transformer.
    Tokens: [taup_pi1, taun_pi1, taup_ip, taun_ip,
             taup_pi2, taun_pi2, taup_pi3, taun_pi3,
             taup_pi0, taun_pi0, MET, taup_sv, taun_sv]
    pi2/pi3/sv masked when is_3prong=0; pi0 masked when haspizero=0.
    """

    _taup_pi1_feats = ['reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz', 'reco_taup_pi1_e']
    _taun_pi1_feats = ['reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz', 'reco_taun_pi1_e']
    _taup_charged_ip_feats = ['reco_taup_charged_ipx', 'reco_taup_charged_ipy', 'reco_taup_charged_ipz']
    _taun_charged_ip_feats = ['reco_taun_charged_ipx', 'reco_taun_charged_ipy', 'reco_taun_charged_ipz']
    _taup_pi2_feats = ['reco_taup_pi2_px', 'reco_taup_pi2_py', 'reco_taup_pi2_pz', 'reco_taup_pi2_e']
    _taun_pi2_feats = ['reco_taun_pi2_px', 'reco_taun_pi2_py', 'reco_taun_pi2_pz', 'reco_taun_pi2_e']
    _taup_pi3_feats = ['reco_taup_pi3_px', 'reco_taup_pi3_py', 'reco_taup_pi3_pz', 'reco_taup_pi3_e']
    _taun_pi3_feats = ['reco_taun_pi3_px', 'reco_taun_pi3_py', 'reco_taun_pi3_pz', 'reco_taun_pi3_e']
    _taup_pizero_feats = ['reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz', 'reco_taup_pizero1_e']
    _taun_pizero_feats = ['reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz', 'reco_taun_pizero1_e']
    _met_feats = ['reco_met_px', 'reco_met_py']
    _taup_sv_feats = ['reco_taup_sv_x', 'reco_taup_sv_y', 'reco_taup_sv_z']
    _taun_sv_feats = ['reco_taun_sv_x', 'reco_taun_sv_y', 'reco_taun_sv_z']

    def __init__(self, input_features, context_dim=256, d_model=64, nhead=4, num_layers=3, dropout=0.0):
        super().__init__()
        feat_idx = {name: i for i, name in enumerate(input_features)}

        # load appropriate columns
        self.taup_pi1_idx = [feat_idx[f] for f in self._taup_pi1_feats]
        self.taun_pi1_idx = [feat_idx[f] for f in self._taun_pi1_feats]
        self.taup_charged_ip_idx = [feat_idx[f] for f in self._taup_charged_ip_feats]
        self.taun_charged_ip_idx = [feat_idx[f] for f in self._taun_charged_ip_feats]
        self.taup_pi2_idx = [feat_idx[f] for f in self._taup_pi2_feats]
        self.taun_pi2_idx = [feat_idx[f] for f in self._taun_pi2_feats]
        self.taup_pi3_idx = [feat_idx[f] for f in self._taup_pi3_feats]
        self.taun_pi3_idx = [feat_idx[f] for f in self._taun_pi3_feats]
        self.taup_pizero_idx = [feat_idx[f] for f in self._taup_pizero_feats]
        self.taun_pizero_idx = [feat_idx[f] for f in self._taun_pizero_feats]
        self.met_idx = [feat_idx[f] for f in self._met_feats]
        self.taup_sv_idx = [feat_idx[f] for f in self._taup_sv_feats]
        self.taun_sv_idx = [feat_idx[f] for f in self._taun_sv_feats]
        self.taup_haspizero_idx = feat_idx['reco_taup_haspizero']
        self.taun_haspizero_idx = feat_idx['reco_taun_haspizero']
        self.taup_is3prong_idx = feat_idx['reco_taup_is3prong']
        self.taun_is3prong_idx = feat_idx['reco_taun_is3prong']

        # projections to the model dimensionality
        self.pi_proj = nn.Linear(4, d_model)  # 4-momentum (shared: pi1, pi2, pi3, pi0)
        self.ip_proj = nn.Linear(3, d_model)  # impact parameter
        self.met_proj = nn.Linear(2, d_model)
        self.sv_proj = nn.Linear(3, d_model)  # secondary vertex position
        self.type_emb = nn.Embedding(13, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu', norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, context_dim)

    def forward(self, x):
        B, device = x.shape[0], x.device
        type_embs = self.type_emb(torch.arange(13, device=device))

        tokens = torch.stack([
            self.pi_proj(x[:, self.taup_pi1_idx]) + type_embs[0],
            self.pi_proj(x[:, self.taun_pi1_idx]) + type_embs[1],
            self.ip_proj(x[:, self.taup_charged_ip_idx]) + type_embs[2],
            self.ip_proj(x[:, self.taun_charged_ip_idx]) + type_embs[3],
            self.pi_proj(x[:, self.taup_pi2_idx]) + type_embs[4],
            self.pi_proj(x[:, self.taun_pi2_idx]) + type_embs[5],
            self.pi_proj(x[:, self.taup_pi3_idx]) + type_embs[6],
            self.pi_proj(x[:, self.taun_pi3_idx]) + type_embs[7],
            self.pi_proj(x[:, self.taup_pizero_idx]) + type_embs[8],
            self.pi_proj(x[:, self.taun_pizero_idx]) + type_embs[9],
            self.met_proj(x[:, self.met_idx]) + type_embs[10],
            self.sv_proj(x[:, self.taup_sv_idx]) + type_embs[11],
            self.sv_proj(x[:, self.taun_sv_idx]) + type_embs[12],
        ], dim=1)

        # Mask undefined columns (DM dependent, True = ignore token)
        pad_mask = torch.zeros(B, 13, dtype=torch.bool, device=device)
        pad_mask[:, 4] = ~x[:, self.taup_is3prong_idx].bool()   # taup pi2
        pad_mask[:, 5] = ~x[:, self.taun_is3prong_idx].bool()   # taun pi2
        pad_mask[:, 6] = ~x[:, self.taup_is3prong_idx].bool()   # taup pi3
        pad_mask[:, 7] = ~x[:, self.taun_is3prong_idx].bool()   # taun pi3
        pad_mask[:, 8] = ~x[:, self.taup_haspizero_idx].bool()  # taup pi0
        pad_mask[:, 9] = ~x[:, self.taun_haspizero_idx].bool()  # taun pi0
        pad_mask[:, 11] = ~x[:, self.taup_is3prong_idx].bool()  # taup SV
        pad_mask[:, 12] = ~x[:, self.taun_is3prong_idx].bool()  # taun SV

        out = self.transformer(tokens, src_key_padding_mask=pad_mask)

        # mean pool over present tokens only
        present = (~pad_mask).float().unsqueeze(-1)
        context = (out * present).sum(dim=1) / present.sum(dim=1)
        return self.output_proj(context)

class ConditionalFlow(nn.Module):
    def __init__(self,
                 input_dim,
                 raw_condition_dim=None,
                 context_dim=256,
                 cond_hidden_dim=64,
                 cond_num_blocks=2,
                 batch_norm=False,
                 activation=nn.ReLU(),
                 use_transformer=False,
                 input_features=None,
                 d_model=64,
                 nhead=4,
                 num_transformer_layers=3,
                 dropout=0.0,
                 **flow_kwargs
    ):
        super().__init__()

        if use_transformer:
            print("!! INFO: Using Transformer for conditioning")
            self.condition_net = ParticleTransformerCondition(
                input_features=input_features,
                context_dim=context_dim, d_model=d_model,
                nhead=nhead, num_layers=num_transformer_layers, dropout=dropout,
            )
        elif cond_num_blocks == 0:
            print("!! INFO: No conditioning network")
            self.condition_net = nn.Identity()
            context_dim = raw_condition_dim
        else:
            print("!! INFO: Using ResidualNet for conditioning")
            self.condition_net = nets.ResidualNet(
                in_features=raw_condition_dim,
                out_features=context_dim,
                hidden_features=cond_hidden_dim,
                num_blocks=cond_num_blocks,
                activation=activation,
                use_batch_norm=batch_norm
            )

        self.flow = NormalizingFlow(
            input_size=input_dim,
            context_features=context_dim,
            activation=activation,
            **flow_kwargs
        )

    def log_prob(self, inputs, context):
        cond_embed = self.condition_net(context)
        log_prob = self.flow.log_prob(inputs=inputs, context=cond_embed)
        return log_prob

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
        #t0 = time.time()
        cond_embed = self.condition_net(context)
        #t1 = time.time()
        x, logabsdet = self.flow._transform.inverse(z, context=cond_embed)
        #t2 = time.time()
        #print(f"Conditioning time: {t1 - t0:.4f} s, Flow decode time: {t2 - t1:.4f} s")
        return x, logabsdet

    def sample_and_log_prob(self, num_samples, context):
        """
        Sample and compute log-prob in a single forward pass.
        Returns samples [B, num_samples, F] and log_prob [B, num_samples].
        More efficient than separate sample() + log_prob() calls.
        """
        cond_embed = self.condition_net(context)
        return self.flow.sample_and_log_prob(num_samples=num_samples, context=cond_embed)
