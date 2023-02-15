import torch
import torch.nn as nn
from functions import LeakySoftplus
from functools import partial
        
class LUNet(nn.Module):
    def __init__(self, num_lu_blocks=1, layer_size = 2, device="cuda:0"):
        """init LUNet with given number of blocks of LU layers"""
        print("... initialized LUNet")
        super(LUNet, self).__init__()
        
        """masks to zero out gradients"""
        self.mask_triu = torch.triu(torch.ones(layer_size, layer_size)).bool()
        self.mask_tril = torch.tril(torch.ones(layer_size, layer_size)).bool().fill_diagonal_(False)
        self.nonlinearity = LeakySoftplus()
        
        """create LU modules"""
        self.intermediate_lu_blocks = nn.ModuleList()
        """adding number of LU Blocks"""
        for i in range(num_lu_blocks):
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
          
        """Adding one final LU block = extra block"""
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
    
    def forward(self, x):
        """build network"""
        x = torch.flatten(x, 1)
        for i, layer in enumerate(self.intermediate_lu_blocks):
            x = layer(x)
            if i % 2 != 0: # after one L and U matrix
                """apply non-linear activation"""
                x = self.nonlinearity(x)
        """final LU block without activation"""
        for i, layer in enumerate(self.final_lu_block):
            x = layer(x)
        return x

def get_zero_grad_hook(mask, device="cuda:0"):
    """zero out gradients"""
    def hook(grad):
        return grad * mask.to(device)
    return hook


"""
* helper functions to store activations and parameters in intermediate layers of the model
* use forward hooks for this, which are functions executed automatically during forward pass
* in PyTorch hooks are registered for nn.Module and are triggered by forward pass of object
"""

def save_activations(activations_dict, name, blu, bla, out):
	activations_dict[name].append(out)
 
def register_activation_hooks(model, layers_to_save):
    """register forward hooks in specified layers"""
    activations_dict = {name: [] for name in layers_to_save}
    for name, module in model.named_modules():
        if name in layers_to_save:
            module.register_forward_hook(partial(save_activations, activations_dict, name))
    return activations_dict