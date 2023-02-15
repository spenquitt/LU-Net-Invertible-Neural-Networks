import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from scipy import interpolate
from utils import register_activation_hooks

class Dequantization(object):
    def __init__(self,  alpha=1e-5, quants=256):
        self.alpha = alpha
        self.quants = quants

    def __call__(self, tensor, reverse=False):
        if not reverse:
            tensor = tensor * 255
            z = self.dequant(tensor)
            z = self.sigmoid(z, reverse=True)
        else:
            z = self.sigmoid(tensor, reverse=False)
            z = z * self.quants
            z = torch.floor(z).clamp(min=0, max=self.quants-1).to(torch.int32)
        return z

    def __repr__(self):
        return self.__class__.__name__ + '(alpha={0}, quants={1})'.format(self.alpha, self.quants)

    def sigmoid(self, z, reverse=False):
        if not reverse:
            z = torch.sigmoid(z)
            z = (z - 0.5 * self.alpha) / (1 - self.alpha)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            z = torch.log(z) - torch.log(1-z)
        return z

    def dequant(self, z):
        z = z + torch.rand_like(z).detach()
        z = z / self.quants
        return z

class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input

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
        yfit = interpolate.splev(input.cpu().detach().numpy(), tck, der=0)
        return torch.tensor(yfit, dtype=torch.float32)
    
def neg_log_likelihood(output, model, layers, alpha=1):
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
    for i in range(len(layers)):
        """layers are outputs of the L-Layer"""
        """lifted sigmoid = derivative of leaky softplus"""
        log_derivatives.append(torch.log(torch.abs(lifted_sigmoid(layers[i]))))
    
    """lu-blocks 1,...,M-1"""
    volume_corr = 0
    for l in range(len(log_diagonals_triu) - 1):
        summand = torch.zeros(N, D).to("cuda:0")
        summand = summand + log_derivatives[l]
        summand = summand + alpha * log_diagonals_triu[l]
        volume_corr = volume_corr + torch.sum(summand)
        
    
    """lu-block M"""
    last = alpha * log_diagonals_triu[len(log_diagonals_triu) - 1]
    last = N * torch.sum(last)
    
    output = constant + sum_squared_mappings - last - volume_corr
    return output
    
def bpd_per_image(inputs, model):
    device = next(model.parameters()).device
    storage = register_activation_hooks(model, layer_name="storage")
    outputs = model(inputs.to(device))
    weighted_sums = []
    for key in sorted(storage.keys()):
        weighted_sums.append(storage[key])
    
    N, D = outputs.shape
    constant = torch.from_numpy(np.array(0.5 * D * np.log(np.pi))).type(torch.float64)
    
    sum_squared_mappings = torch.square(outputs)
    sum_squared_mappings = torch.sum(sum_squared_mappings, dim=1)
    sum_squared_mappings = 0.5 * sum_squared_mappings

    log_diagonals_triu = []
    for param in model.parameters():
        if len(param.shape) == 2 and param[1,0] == 0: # if upper triangular and matrix
            log_diagonals_triu.append(torch.log(torch.abs(torch.diag(param))))

    log_derivatives = []
    for i in range(len(weighted_sums)):
        log_derivatives.append(torch.log(lifted_sigmoid(weighted_sums[i])))
    
    volume_corr = torch.zeros(N).to(device)
    for l in range(len(log_diagonals_triu) - 1):
        summand = torch.zeros(N, D).to(device)
        summand = summand + log_derivatives[l]
        summand = summand + log_diagonals_triu[l]
        volume_corr += torch.sum(summand, dim=1)
    volume_corr += torch.sum(log_diagonals_triu[-1])
    
    nll = constant + sum_squared_mappings - volume_corr
    bpd = nll * np.log2(np.exp(1)) / D
    return bpd.cpu().detach().numpy()