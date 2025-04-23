import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class FourVecLinear(nn.Module):
    """
    A linear layer for 4-vectors of the form [E, px, py, pz].

    This layer takes a batch of 4-vectors shaped (batch_size, in_features, 4)
    and outputs a new set of 4-vectors shaped (batch_size, out_features, 4).
    Each input 4-vector is scaled by a learnable weight and summed over the input dimension
    to produce an output 4-vector for each feature.

    Bias behavior is controlled by `bias_type`, and the bias is applied **to the output 4-vectors** after the weighted sum:

        bias_type = 0: No bias is added.

        bias_type = 1: A single scalar bias is learned per output feature. This scalar is **added equally to all four components**
                       (E, px, py, pz) of each output 4-vector. The output becomes:
                       [E + b, px + b, py + b, pz + b], where b is the learned scalar for that output feature.

        bias_type = 2: A scalar bias is learned per output feature and **applied along the direction of the output 4-vector**.
                       The bias is scaled in the direction of the output 4-vector (before the bias is applied), effectively shifting
                       the vector along its own direction.

        bias_type = 3: A scalar bias is learned per output feature and **only added to the E (energy) component** of the output 4-vector.
                       The (px, py, pz) components remain unchanged. The output becomes:
                        [E + b, px, py, pz], where b is the learned scalar for that output feature.

        bias_type = 4: A full 4-component bias vector is learned per output feature, allowing each of (E, px, py, pz)
                       to receive a distinct bias term. The output becomes:
                       [E + b0, px + b1, py + b2, pz + b3], where b0 through b3 are the individual learned biases.

    Args:
        in_features (int): Number of input 4-vectors per sample.
        out_features (int): Number of output 4-vectors per sample.
        bias_type (int, optional): Type of bias to apply (0, 1, or 2). Default: 1.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias_type: int = 3,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.bias_type = bias_type

        # Initialize weights
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # Initialize bias based on bias_type
        if bias_type in [1,2,3]:
            self.bias = nn.Parameter(torch.empty((out_features), **factory_kwargs))  # same bias for all elements
        elif bias_type == 4:
            self.bias = nn.Parameter(torch.empty((out_features, 4), **factory_kwargs))  # unique bias for each component
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        weighted = torch.einsum('oi,bif->bof', self.weight, x)

        if self.bias_type == 1 and self.bias is not None:
            weighted = weighted + self.bias.view(1, -1, 1)
        elif self.bias_type == 2 and self.bias is not None:
            # for this option we will keep length of bias vector equal to b, but the direction will the the same as the input
            # get the direction of the output vector before the bias is applied
            norm = torch.norm(weighted, dim=2, keepdim=True)
            norm = torch.clamp(norm, min=1e-6) # Avoid division by zero
            weighted_dir = weighted / norm
            # apply bias in the direction of the output vector before the bias is applied
            weighted = weighted +  self.bias.view(1, -1, 1) * weighted_dir
        elif self.bias_type == 3 and self.bias is not None:
            # for this option we add the bias only to the E component of the 4-vectors
            # make a object the same shape as weighted but with px,py,pz = 0, and E=1
            bias_term = torch.zeros_like(weighted)
            bias_term[:, :, 0] = 1
            weighted = weighted +  self.bias.view(1, -1, 1) * bias_term
        elif self.bias_type == 4 and self.bias is not None:
            # for this option we generate 4 bias terms or each component of the output vector
            weighted = weighted + self.bias.view(1, -1, 4) 

        return weighted


    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias_type={self.bias_type}'

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

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = x[..., 0:1]  # Shape (..., 1)
        p = x[..., 1:]   # Shape (..., 3)

        E_relu = torch.relu(E)  # Shape (..., 1)
        mask = (E > 0).float()  # 1 where E > 0, else 0; shape (..., 1)

        p_masked = p * mask     # Zero momenta where E <= 0

        return torch.cat([E_relu, p_masked], dim=-1)  # Shape (..., 4)

class FourVecLeakyReLU(nn.Module):
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

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        E = x[..., 0:1]  # Shape (..., 1)
        p = x[..., 1:]   # Shape (..., 3)

        # Apply LeakyReLU to energy component
        E_leaky = torch.where(E > 0, E, self.negative_slope * E)  # Shape (..., 1)

        # Apply the same scaling to the momentum components
        p_scaled = torch.where(E > 0, p, p * self.negative_slope)  # Shape (..., 3)

        return torch.cat([E_leaky, p_scaled], dim=-1)  # Shape (..., 4)

if __name__ == '__main__':

    # Create a batch of 4-vectors (shape: batch_size, in_features, 4)
    batch_size = 2
    in_features = 3
    out_features = 2
    bias_type = 3

    x = torch.randn(batch_size, in_features, 4)  # Random input tensor

    layer = FourVecLinear(in_features, out_features, bias_type=bias_type)
    output = layer(x)

    print("Input 4-vectors (x):")
    print(x)

    print("\nWeight matrix:")
    print(layer.weight)
    print("\nBias matrix:")
    print(layer.bias)

    print(f"\nOutput with bias_type={bias_type}:")
    print(output)

    relu_layer = FourVecReLUE()
    relu_output = relu_layer(output)
    print("\nOutput after FourVecReLUE activation:")
    print(relu_output)

    leaky_relu_layer = FourVecLeakyReLU(negative_slope=0.1)
    leaky_relu_output = leaky_relu_layer(output)
    print("\nOutput after FourVecLeakyReLU activation:")
    print(leaky_relu_output)
