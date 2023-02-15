import torch
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs

def generate_gaussian(d, numb_data = 9000):
    """generate two dimensional gaussian data"""
    if d == 2:
        # Generate some data
        X_new, y_true = make_blobs(n_samples=numb_data, centers=4,
                       cluster_std=0.20, random_state=1)
        plt.scatter(X_new[:, 0], X_new[:, 1], s=40, cmap='viridis', alpha=0.5)
        plot_name = "blobs_gaussian.png"
        save_path = Path("outputs/generation gaussian/" + plot_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path))
        plt.close()
        
    # shuffle the data tensor
    X = X_new
    X = torch.tensor(X)
    X=X[torch.randperm(X.size()[0])]
    
    return X


