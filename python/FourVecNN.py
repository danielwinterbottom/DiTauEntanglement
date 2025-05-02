import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from collections import OrderedDict

class FourVecLinear(nn.Module):
    """
    A linear layer for 4-vectors of the form [E, px, py, pz].

    This layer takes a batch of 4-vectors shaped (batch_size, in_features, 4)
    and outputs a new set of 4-vectors shaped (batch_size, out_features, 4).
    Each input 4-vector is scaled by a learnable weight and summed over the input dimension
    to produce an output 4-vector for each feature.


    Args:
        in_features (int): Number of input 4-vectors per sample.
        out_features (int): Number of output 4-vectors per sample.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        weighted = torch.einsum('oi,bif->bof', self.weight, x)

        return weighted


    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'

class FourVecReLUE(nn.Module):
    """
    A custom activation function for 4-vectors that applies the ReLU function only to the energy component (E) of the 4-vector.

    For each input 4-vector [E, px, py, pz], it computes:
        z = ReLU(E)
        then it applies:
        E = z
        px = px if E > 0 else 0
        py = py if E > 0 else 0
        pz = pz if E > 0 else 0
    The output 4-vector is then [z, px, py, pz].

    This ensures the direction of the 4-vector remains the same,
    but its magnitude is affected by the nonlinearity of ReLU on the energy component.

    """

    def __init__(self, num_vecs: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_vecs, 1))
        bound = 1 / math.sqrt(num_vecs)
        nn.init.uniform_(self.bias, bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = x[..., 0:1] + self.bias
        p = x[..., 1:]

        E_relu = torch.relu(E)
        mask = (E > 0).float()  # 1 where E > 0, else 0

        p_masked = p * mask     # Zero momenta where E <= 0

        return torch.cat([E_relu, p_masked], dim=-1) 

class FourVecLeakyReLUE(nn.Module):
    """
    A custom activation function for 4-vectors that applies the LeakyReLU function only to the energy component (E) of the 4-vector.

    For each input 4-vector [E, px, py, pz], it computes:
        z = LeakyReLU(E, negative_slope)
        then it applies:
        E = z
        px = px if E > 0 else px * negative_slope
        py = py if E > 0 else py * negative_slope
        pz = pz if E > 0 else pz * negative_slope

    The output 4-vector is then [z, px, py, pz].

    This ensures the direction of the 4-vector remains the same,
    but its magnitude is affected by the nonlinearity of LeakyReLU on the energy component.
    """

    def __init__(self, num_vecs: int, negative_slope: float = 0.01):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, num_vecs, 1))
        bound = 1 / math.sqrt(num_vecs)
        nn.init.uniform_(self.bias, bound, bound)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = x[..., 0:1] + self.bias
        p = x[..., 1:]

        # Apply LeakyReLU to energy component
        E_leaky = torch.where(E > 0, E, self.negative_slope * E)

        # Apply the same scaling to the momentum components
        p_scaled = torch.where(E > 0, p, p * self.negative_slope)

        return torch.cat([E_leaky, p_scaled], dim=-1)

class FourVecBatchNormE(nn.Module):
    """
    A custom BatchNorm for 4-vectors that normalizes only the energy (E) component,
    and scales the full 4-vector [E, px, py, pz] proportionally.

    Input shape: (batch_size, num_vectors, 4)
    """

    def __init__(self, num_vecs, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_vecs = num_vecs
        # BatchNorm1d over the E components from all 4-vectors
        self.bn_E = nn.BatchNorm1d(num_vecs, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, F = x.shape
        assert F == 4 and N == self.num_vecs, "Expected shape (B, num_vecs, 4)"

        E = x[..., 0]  # shape (B, N)
        p = x[..., 1:]  # shape (B, N, 3)

        # Normalize E: reshape to (B, N) and apply BN1d across dim=0
        E_norm = self.bn_E(E)  # shape (B, N)

        # Compute scale: new E / original E
        scale = E_norm / (E + 1e-6)  # avoid div by zero, shape (B, N)

        # Expand scale for broadcasting over 3 momentum components
        scale = scale.unsqueeze(-1)  # (B, N, 1)

        p_scaled = p * scale  # scale px, py, pz proportionally
        E_norm = E_norm.unsqueeze(-1)  # (B, N, 1)

        return torch.cat([E_norm, p_scaled], dim=-1)  # (B, N, 4)

class FourVecLinearResidualBlock(nn.Module):
    """
    A residual block for 4-vectors that applies a linear transformation followed by a FourVecReLUE activation.
    The input is added to the output of the block, allowing for identity mapping.
    """

    def __init__(self, N_features: int):
        super().__init__()
        layers = OrderedDict()
        self.linear1 = FourVecLinear(N_features, N_features)
        self.activation = FourVecLeakyReLUE(N_features)
        self.linear2 = FourVecLinear(N_features, N_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        return out + identity

if __name__ == '__main__':

    # Create a batch of 4-vectors (shape: batch_size, in_features, 4)
    batch_size = 2
    in_features = 3
    out_features = 2

    x = torch.randn(batch_size, in_features, 4)  # Random input tensor

    layer = FourVecLinear(in_features, out_features)
    output = layer(x)

    print("Input 4-vectors (x):")
    print(x)

    print(x.shape)

    print("\nWeight matrix:")
    print(layer.weight)
    print("\nBias matrix:")
    print(layer.bias)

    print(f"\nOutput:")
    print(output)

    relu_layer = FourVecReLUE(out_features)
    relu_output = relu_layer(output)
    print("\nOutput after FourVecReLUE activation:")
    print(relu_output)

    leaky_relu_layer = FourVecLeakyReLU(out_features,negative_slope=0.1)
    leaky_relu_output = leaky_relu_layer(output)
    print("\nOutput after FourVecLeakyReLU activation:")
    print(leaky_relu_output)


    print('FourVecBatchNormE checks:')

    num_vecs = 3 
    bn = FourVecBatchNormE(num_vecs=num_vecs)

    # Sample input: shape (batch_size=2, num_vecs=3, features=4)
    # Each 4-vector is [E, px, py, pz]
    x = torch.tensor([
        [[10.0, 1.0, 2.0, 3.0],
         [20.0, 2.0, 4.0, 6.0],
         [30.0, 3.0, 6.0, 9.0]],

         [[30.0, 1.0, 2.0, 3.0],
         [50.0, 2.0, 4.0, 6.0],
         [70.0, 3.0, 6.0, 9.0]],

        [[15.0, 1.5, 3.0, 4.5],
         [25.0, 2.5, 5.0, 7.5],
         [35.0, 3.5, 7.0, 10.5]]
    ], dtype=torch.float32)

    # get just the E component
    E = x[..., 0]  # shape (batch_size, num_vecs)
    varE = torch.var(input=E, dim=0, unbiased=False)  # shape (num_vecs)
    meanE = torch.mean(input=E, dim=0)  # shape (num_vecs)
    print("\nE mean:")
    print(meanE)
    print("\nE variance:")
    print(varE)
    bn_E = nn.BatchNorm1d(num_vecs)
    bn_E.train()  # Set to training mode
    E_norm = bn_E(E)  # shape (batch_size, num_vecs)
    print("\nE components:")
    print(E)
    print("\nNormalized E components:")
    print(E_norm)


    # Enable training mode to use batch stats
    bn.train()

    # Apply the normalization
    x_norm = bn(x)

    # Print outputs
    print("Input:")
    print(x)
    print("\nNormalized Output:")
    print(x_norm)

    # Additional test: verify proportional scaling
    # px / E should be the same as px' / E' for each 4-vector
    ratios_before = x[..., 1:] / x[..., 0:1]
    ratios_after = x_norm[..., 1:] / x_norm[..., 0:1]

    print("\nMomentum / Energy ratios before:")
    print(ratios_before)
    print("\nMomentum / Energy ratios after:")
    print(ratios_after)

    # Check if ratios are (approximately) preserved
    print("\nRatio difference (should be near zero):")
    print((ratios_before - ratios_after).abs().max())