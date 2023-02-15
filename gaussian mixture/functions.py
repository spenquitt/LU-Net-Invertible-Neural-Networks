from turtle import forward
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from scipy import interpolate

class LeakySoftplus(nn.Module):
    def __init__(self, alpha: float = 0.1) -> None:
        super(LeakySoftplus, self).__init__()
        self.alpha = alpha
        
    def forward(self, input: Tensor) -> Tensor:
        softplus = torch.log1p(torch.exp(-torch.abs(input))) + torch.maximum(input, torch.tensor(0))
        output = self.alpha * input + (1-self.alpha) * softplus
        return output

def lifted_sigmoid(x, alpha=0.1):
    """derivative of leaky softplus"""
    return alpha + (1-alpha) * torch.sigmoid(x)

class InvertedLeakySoftplus(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super(InvertedLeakySoftplus,self).__init__()
        self.alpha = alpha
    
    def forward(self, input: Tensor):
        x = torch.arange(-1000., 1000.001, 0.001)
        activation = LeakySoftplus()
        y = activation(x)
        tck = interpolate.splrep(y, x, s=0)
        """data is first moved to cpu and then converted to numpy array"""
        yfit = interpolate.splev(input.cpu().detach().numpy(), tck, der=0)
        return torch.tensor(yfit, dtype=torch.float32)
    
class InvertedLLayer(nn.Module):
    def __init__(self, inverted_weight = None, inverted_bias = None) -> None:
        super(InvertedLLayer, self).__init__()
        self.inverted_weight = inverted_weight
        self.inverted_bias = inverted_bias
        
    def forward(self, input: Tensor, device="cuda:0") -> Tensor:
        input = input.to(device)
        y_tilde = torch.t(input - self.inverted_bias)
        x_tilde = torch.linalg.solve(self.inverted_weight, y_tilde)
        x = torch.t(x_tilde)
        return x

class InvertedULayer(nn.Module):
    def __init__(self, inverted_weight = None) -> None:
        super(InvertedULayer, self).__init__()
        self.inverted_weight = inverted_weight
        
    def forward(self, input: Tensor, device="cuda:0") -> Tensor:
        input = input.to(device)
        y_tilde = torch.t(input)
        x_tilde = torch.linalg.solve(self.inverted_weight, y_tilde)
        x = torch.t(x_tilde)
        return x

"""loss function"""
def log_likelihood(output, model, layers):
    """compute the log likelihood with change of variables formula, average per pixel"""
    N, D = output.shape # batch size and single output size
    
    """First summand"""
    constant = torch.from_numpy(np.array(0.5 * D * N * np.log(np.pi))).type(torch.float64)
    
    """Second summand"""
    sum_squared_mappings = torch.square(output)
    sum_squared_mappings = torch.sum(sum_squared_mappings)
    sum_squared_mappings = 0.5 * sum_squared_mappings
    
    """Third summand"""
    """log diagonals of U"""
    log_diagonals_triu = []
    for param in model.parameters():
        if len(param.shape) == 2 and param[1,0] == 0: # if upper triangular and matrix
            log_diagonals_triu.append(torch.log(torch.abs(torch.diag(param))))

    log_derivatives = []
    for i in range((len(layers) - 1) * 2):
        """layers are outputs of the L-Layer"""
        """lifted sigmoid = derivative of leaky softplus"""
        if i % 2 != 0:
            log_derivatives.append(torch.log(torch.abs(lifted_sigmoid(layers["intermediate_lu_blocks.{}".format(i)][0]))))
    log_derivatives.append(torch.log(torch.abs(lifted_sigmoid(layers["final_lu_block.1"][0]))))
    
    """lu-blocks 1,...,M-1"""
    volume_corr = 0
    for l in range(len(log_diagonals_triu) - 1):
        summand = torch.zeros(N, D).to("cuda:0")
        summand = summand + log_derivatives[l]
        summand = summand + log_diagonals_triu[l]
        volume_corr = volume_corr + torch.sum(summand)
        
    
    """lu-block M"""
    last = log_diagonals_triu[len(log_diagonals_triu) - 1]
    last = N * torch.sum(last)
    
    output = constant + sum_squared_mappings - last - volume_corr
    return output
 