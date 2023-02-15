import torch
import torch.nn as nn
from functions import InvertedLeakySoftplus, InvertedLLayer, InvertedULayer
from visuals import generation_gaussian
from model import LUNet
import matplotlib.pyplot as plt
from pathlib import Path
from gaussian_mixture import generate_gaussian

class InvertedLUNet(nn.Module):
    def __init__(self, num_lu_blocks=1, save_path="./checkpoints/lunet.pth"):
        """init inverted LuNet with given numer of blocks of LU layers"""
        print("... initialized inverted LUNet")
        super(InvertedLUNet, self).__init__()
        self.inverted_nonlinearity = InvertedLeakySoftplus()
        
        """initialize the weights and bias"""
        params = torch.load(save_path)
        bias = []
        l_weight = []
        u_weight = []
        
        for param in reversed(params.values()):
            if len(param.shape) == 1: # bias
                bias.append(param)
            elif len(param.shape) == 2 and param[1,0] == 0: # U weight
                u_weight.append(param)
            else: # L weight
                l_weight.append(param)
        
        self.inverted_lu_blocks = nn.ModuleList()
        for i in range(num_lu_blocks + 1):
            self.inverted_lu_blocks.append(InvertedLLayer(l_weight[i], bias[i]))
            self.inverted_lu_blocks.append(InvertedULayer(u_weight[i]))
            
                 
    def forward(self, x, device="cuda:0"): 
        x = x.to(device)       
        count = 0
        for i in range(0, len(self.inverted_lu_blocks), 2):
            if i == 0: # final lu block without activation
                x = self.inverted_lu_blocks[i](x, device)  #inverted L layer
                x = self.inverted_lu_blocks[i + 1](x, device)  # inverted U layer
                count += 1
            else:
                x = self.inverted_nonlinearity(x)   # inverted activation function
                x = self.inverted_lu_blocks[i](x, device)  #inverted L layer
                x = self.inverted_lu_blocks[i + 1](x, device)  # inverted U layer
                count += 1
        return x
 
 
"""testing trained net with gaussian data"""
def testing_gaussian_network(save_path, num_lu_blocks, layer_size, device="cuda:0"):
    gaussian_test_data = generate_gaussian(layer_size, numb_data=1000)
    
    """plot gaussian test data"""
    plt.scatter(gaussian_test_data[:, 0], gaussian_test_data[:, 1], s=40, cmap='viridis', alpha=0.5)
    plot_name = "blobs_test_gaussian.png"
    path = Path("outputs/generation gaussian/" + plot_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path))
    plt.close()
    
    """initialize LU net"""
    gaussian_test_data = torch.Tensor(gaussian_test_data).to(device)
    network = LUNet(num_lu_blocks, layer_size)
    params = torch.load(save_path)
    network.load_state_dict(params)
    network = network.to(device)
    output = network(gaussian_test_data)
    
    """inititalize inverted LU net"""
    inverted_network = InvertedLUNet(num_lu_blocks, save_path).to(device)
    
    """generation process"""
    generation_gaussian(inverted_network, output)

