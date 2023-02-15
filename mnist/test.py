import torch
from functions import neg_log_likelihood
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path
from utils import register_activation_hooks

"""testing with MNIST and Fashion MNIST data"""
def testing_routine(model, test_loader, dataset, target, plot_name=None, return_bpd=False):
    model.eval()
    device = next(model.parameters()).device
    storage = register_activation_hooks(model, layer_name="storage")
    test_nll = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            weighted_sums = []
            for key in sorted(storage.keys()):
                weighted_sums.append(storage[key])
            loss = neg_log_likelihood(output, model, weighted_sums)
            batch_size, layer_size = output.shape
            test_nll += loss / (batch_size * layer_size)
    nat = test_nll / len(test_loader)
    bpd = nat * np.log2(np.exp(1))
    print('Test set : nll nats : {:.4f}'.format(nat))
    if plot_name is not None:
        plot_distribution(output, "outputs/distribution/{}/target_{}/".format(dataset, target), plot_name, title="nll bpd : {:.3f}".format(bpd))
    if return_bpd:
        return bpd.cpu().detach().numpy()
    
def plot_distribution(output, save_dir, save_name, title=""):
    output_data = output.detach().flatten().cpu().numpy()
    """remove outlier"""
    upper = np.quantile(output_data, 0.997)
    lower = np.quantile(output_data, 0.003)
    output_data = output_data[np.logical_and(output_data > lower, output_data < upper)]

    plt.hist(output_data, density=True, label="forward distribution")

    x = np.linspace(-3, 3, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), label="pdf standard normal")
    plt.legend()

    plt.xlim(-3.3, 3.3)
    plt.ylim(0, 0.6)
    plt.title(title)
    save_path = Path(save_dir + save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()