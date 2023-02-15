import torch
import torch.nn as nn

from functions import Identity, LeakySoftplus, InvertedLeakySoftplus
from torch.nn.parameter import Parameter
        
class LUNet(nn.Module):
    def __init__(self, num_lu_blocks=1, layer_size = 2, device="cuda:0"):
        """init LUNet with given number of blocks of LU layers"""
        print("... initialized LUNet")
        super(LUNet, self).__init__()
        
        """masks to zero out gradients"""
        self.mask_triu = torch.triu(torch.ones(layer_size, layer_size)).bool()
        self.mask_tril = torch.tril(torch.ones(layer_size, layer_size)).bool().fill_diagonal_(False)
        self.nonlinearity = LeakySoftplus()
        self.layer_size = layer_size
        
        """create LU modules"""
        self.intermediate_lu_blocks = nn.ModuleList()
        """adding number of LU Blocks"""
        for _ in range(num_lu_blocks):
            """init upper triangular weight matrix U without bias"""
            self.intermediate_lu_blocks.append(nn.Linear(layer_size, layer_size, bias=False))
            upper = self.intermediate_lu_blocks[-1]
            with torch.no_grad():
                upper.weight.copy_(torch.triu(upper.weight))
            upper.weight.register_hook(get_zero_grad_hook(self.mask_triu, device))
            """init lower triangular weight matrix L with bias"""
            self.intermediate_lu_blocks.append(nn.Linear(layer_size, layer_size))
            lower = self.intermediate_lu_blocks[-1]
            with torch.no_grad():
                lower.weight.copy_(torch.tril(lower.weight))
                lower.weight.copy_(lower.weight.fill_diagonal_(1))
            lower.weight.register_hook(get_zero_grad_hook(self.mask_tril, device))
        
          
        """adding one final LU block = extra block"""
        self.final_lu_block = nn.ModuleList()
        """init upper triangular weight matrix U without bias"""
        self.final_lu_block.append(nn.Linear(layer_size, layer_size, bias=False))
        upper = self.final_lu_block[-1]
        with torch.no_grad():
            upper.weight.copy_(torch.triu(upper.weight))
        upper.weight.register_hook(get_zero_grad_hook(self.mask_triu, device))
        """init lower triangular weight matrix L with bias"""
        self.final_lu_block.append(nn.Linear(layer_size, layer_size))
        lower = self.final_lu_block[-1]
        with torch.no_grad():
            lower.weight.copy_(torch.tril(lower.weight))
            lower.weight.copy_(lower.weight.fill_diagonal_(1))
        lower.weight.register_hook(get_zero_grad_hook(self.mask_tril, device))
    
        """adding some identity layers to later access and extract activations"""
        self.storage = nn.ModuleList()
        for _ in range(num_lu_blocks+1):
            self.storage.append(Identity())

    def forward(self, x):
        """build network"""
        x = torch.flatten(x, 1)
        for i, layer in enumerate(self.intermediate_lu_blocks):
            x = layer(x)
            if i % 2 != 0: # after one L and U matrix
                x = self.storage[int(i/2)](x)
                """apply non-linear activation"""
                x = self.nonlinearity(x)
        """final LU block without activation"""
        for i, layer in enumerate(self.final_lu_block):
            x = layer(x)
        x = self.storage[-1](x)
        return x

def get_zero_grad_hook(mask, device="cuda:0"):
    """zero out gradients"""
    def hook(grad):
        return grad * mask.to(device)
    return hook


class LUNetInverse(nn.Module):
    def __init__(self, num_lu_blocks=1, layer_size=2, device="cpu"):
        """init inverted LUNet with given number of blocks of LU layers"""
        print("... initialized inverted LUNet")
        super(LUNetInverse, self).__init__()
        self.nonlinearity = InvertedLeakySoftplus()
        self.lu_layers = nn.ModuleList()
        for _ in range(num_lu_blocks + 1):
            self.lu_layers.append(InverseLinear(layer_size, device=device))
            self.lu_layers.append(InverseLinear(layer_size, bias=False, device=device))
                 
    def forward(self, x):
        for i in range(0, len(self.lu_layers), 2):
            if i == 0: # final lu block without activation
                x = self.lu_layers[i](x) # inverted L layer
                x = self.lu_layers[i + 1](x) # inverted U layer
            else:
                x = self.nonlinearity(x) # inverted activation function
                x = self.lu_layers[i](x) # inverted L layer
                x = self.lu_layers[i + 1](x) # inverted U layer
        return x


class InverseLinear(nn.Module):
    """applies the inverse of a linear transformation to incoming data: x=(z-b)A^(-1)"""
    def __init__(self, num_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(InverseLinear, self).__init__()
        if bias:
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.weight = Parameter(torch.empty((num_features, num_features), **factory_kwargs))
        self.device = device

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            x = torch.t(input.to(self.device))
        elif self.bias is not None:
            x = torch.t(input.to(self.device) - self.bias.to(self.device))
        x = torch.linalg.solve(self.weight, x)
        return torch.t(x)