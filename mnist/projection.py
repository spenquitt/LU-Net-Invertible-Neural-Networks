import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np
import scipy.stats as stats

def projection (norm_output, data, target, layer_size = 2):
    for i in range(1, 11):
        v = torch.normal(-2, 3, size=(layer_size, 1)).to("cuda:0")
        norm_v = v / (torch.norm(v))
        norm_v = norm_v.reshape(layer_size).to("cuda:0")
        list = []
        
        for output in norm_output:
            dot = torch.dot(output, norm_v)
            list.append(dot)
        
        T = torch.tensor(list)
        plot_projection(T, "outputs/projection/{}/target_{}/".format(data, target), 'projection_{}.png'.format(i))


def plot_projection(output, save_dir, save_name):
    output_data = output.detach().flatten().cpu().numpy()
    """remove outlier"""
    upper = np.quantile(output_data, 0.997)
    lower = np.quantile(output_data, 0.003)
    output_data = output_data[np.logical_and(output_data > lower, output_data < upper)]

    plt.hist(output_data, density=True, label="projection distribution")

    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), label="pdf standard normal")
    plt.legend()

    plt.xlim(-3.3, 3.3)
    plt.ylim(0, 0.6)
    save_path = Path(save_dir + save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()
                          