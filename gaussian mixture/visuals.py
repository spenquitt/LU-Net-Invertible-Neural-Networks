from __future__ import print_function
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
import gaussian_mixture
from sklearn import mixture


"""plotting loss"""
def init_loss_dict(num_epochs):
    loss_dict = {}  # empty loss dictionary
    for i in range(1, num_epochs + 1):
        loss_dict["Epoch {}".format(i)] = {}
    torch.save(loss_dict, 'train_loss_data.json')
        
def add_batch_loss(epoch, batch, loss):
    loss_dict = torch.load('train_loss_data.json')
    loss_dict["Epoch {}".format(epoch)].update({"Batch {}".format(batch): loss})
    torch.save(loss_dict, 'train_loss_data.json')
    
def loss_plotten():
    loss_dict = torch.load('train_loss_data.json')
    loss = []
    for i in loss_dict.values():
        for j in i.values():
            loss.append(j)
    plt.plot(loss)
    plt.title("Average training loss per pixel")
    save_name = "loss_plot.png"
    save_path = Path("outputs/loss plot/" + save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()


"""generation of gaussian data"""
def generation_gaussian(inverted_network, output):
    """singular value decomposition"""
    m = torch.mean(output, 0)
    cov = torch.cov(torch.t(output))
    U, S, Vh = torch.linalg.svd(cov)
    print("SVD correct: " + str(torch.dist(cov, U @ torch.diag(S) @ Vh)))
    cov_sqrt = U @ torch.sqrt(torch.diag(S)) @ Vh
    print("Correctness square value: " + str(torch.dist(cov, cov_sqrt @ cov_sqrt)))

    """generation"""
    normal_corr = torch.randn(output.shape[0], output.shape[1]).to("cuda:0") @ cov_sqrt + m
    output_inverted = inverted_network(normal_corr)
    
    """remove outlier"""
    output_inverted = output_inverted.detach().flatten().cpu().numpy()
    """only plot the main area of the four centers to remove outliers"""
    output_inverted = output_inverted[np.logical_and(output_inverted > -30, output_inverted < 30)]
    output_inverted = torch.tensor(output_inverted).reshape([-1, 2])
    plt.scatter(output_inverted[:, 0], output_inverted[:, 1], alpha=0.5)
    plot_name = "inv_gaussian.png"
    save_path = Path("outputs/generation gaussian/" + plot_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()
    
    """adding heatmap"""
    with open("config.yaml", "r") as yamlfile:
        cfg = edict(yaml.load(yamlfile, Loader=yaml.FullLoader))
        
    # fit a Gaussian Mixture Model with four components
    train_loader = gaussian_mixture.generate_gaussian(cfg[cfg.model].layer_size)
    X_new = output_inverted
    clf = mixture.GaussianMixture(n_components=4, covariance_type="full")
    clf.fit(torch.Tensor(train_loader))
    
    """X_new and trainloader as contour one on colored and one in black"""
    # display predicted scores by the model as a contour plot
    x = np.linspace(-11.9, -0.1)
    y = np.linspace(-9.9, 5.9)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    nll = -clf.score_samples(XX) # nll per example
    Z = nll.reshape(X.shape)
        
    # colored contours
    plt.contourf(X, Y, Z, levels = 50, cmap="RdBu")
    
    clf_train = mixture.GaussianMixture(n_components=4, covariance_type="full")
    clf_train.fit(X_new)
    nll = -clf_train.score_samples(XX) 
    Z = nll.reshape(X.shape)
        
    plt.contour(X, Y, Z, levels=40, colors="black")
    
    plot_name = "contour_gaussian_{}.png".format(cfg[cfg.model].num_lu_blocks)
    save_path = Path("outputs/generation gaussian/" + plot_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()
    
    # fit a Gaussian Mixture Model with four components
    train_loader = gaussian_mixture.generate_gaussian(cfg[cfg.model].layer_size)
    X_new = output_inverted
    clf = mixture.GaussianMixture(n_components=4, covariance_type="full")
    clf.fit(torch.Tensor(train_loader))

    # display predicted scores by the model as a contour plot
    x = np.linspace(-11.9, -0.1)
    y = np.linspace(-9.9, 5.9)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    nll = -clf.score_samples(XX) # nll per example
    Z = nll.reshape(X.shape)

    # colored contours
    plt.contourf(X, Y, Z, levels = 50, cmap="RdBu")
    # black and white contours
    # plt.contour(X, Y, Z, levels = 50, colors="black")
    plt.scatter(train_loader[:, 0], train_loader[:, 1], s=40, color='green', alpha=0.2, label="Training data")
    plt.legend()
    
    plot_name = "contour_gaussian_train_data.png"
    save_path = Path("outputs/generation gaussian/" + plot_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path))
    plt.close()
        