import yaml
import hydra
import torch
import time
import numpy as np

from pathlib import Path
from torch import nn
from torchvision import datasets, transforms
from easydict import EasyDict as edict
from torch.distributions.multivariate_normal import MultivariateNormal

from utils import register_activation_hooks
from functions import neg_log_likelihood
from functions import Dequantization
from model import LUNet, LUNetInverse


def main():
    """load config"""
    with open("config.yaml", "r") as yamlfile:
        cfg = edict(yaml.load(yamlfile, Loader=yaml.FullLoader))
    start_speed_test(cfg)


def start_speed_test(cfg):
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    available = torch.cuda.is_available()
    curr_device = torch.cuda.current_device()
    device_count = torch.cuda.device_count() 
    device_name =  torch.cuda.get_device_name(0)

    print(f'Cuda available: {available}')
    print(f'Current device: {curr_device}')
    print(f'Device: {device}')
    print(f'Device count: {device_count}')
    print(f'Device name: {device_name}')
    
    model = hydra.utils.instantiate(cfg[cfg.model]).to(device) # initialize LUNet

    print("Number of model parameters: {:,}".format(count_parameters(model)))

    train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    """Experiments on MNIST and FashionMNIST"""
    if cfg.data == 'MNIST' or cfg.data == 'FashionMNIST':
        transform_train=transforms.Compose([
            transforms.ToTensor(),
            Dequantization()
            ])
        
        """load datasets"""
        if cfg.data == 'MNIST':
            print("Experiments on MNIST")
            train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
        
        if cfg.data == 'FashionMNIST':
            print("Experiments on FashionMNIST")
            train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform_train)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)

    """training loop"""
    model.train()
    times = []
    for _ in range(1, cfg.num_epochs + 1):
        start_time = time.time()
        training_routine(model, train_loader, optimizer, cfg.loss_alpha)
        times.append(time.time() - start_time)
    average_time_epoch = sum(times) / len(times)
    average_time_batch = average_time_epoch / cfg.batch_size
    print("Average training time per epoch: {:.4f} sec over {} epochs".format(average_time_epoch, cfg.num_epochs))
    print("Average training time per batch: {:.4f} sec over {} epochs".format(average_time_batch, cfg.num_epochs))

    """initialize model"""
    load_path =  Path(cfg.checkpoints_dir) / Path("{}/target_{}/best.pth".format(cfg.data, cfg.mnist_target))
    state_dict = torch.load(load_path)
    model = LUNet(cfg[cfg.model].num_lu_blocks, cfg[cfg.model].layer_size, device)
    model.to(device).eval()
    model.load_state_dict(state_dict)

    """initialize reverse model"""
    load_path_inverse = str(load_path).replace("best", "best_inverse")
    state_dict_inverse_lunet = torch.load(load_path_inverse)
    reversed_model = LUNetInverse(cfg[cfg.model].num_lu_blocks, cfg[cfg.model].layer_size, device)
    reversed_model.to(device).eval()
    reversed_model.load_state_dict(state_dict_inverse_lunet)
    
    """initialize latent distribution to sample from"""
    load_path_stats = str(load_path).replace("best", "best_stats")
    stats = torch.load(load_path_stats)
    mean = stats["latent_mean"]
    covariance = stats["latent_covariance"]
    dist = MultivariateNormal(mean, covariance)

    """generating new images by sampling from latent distribution"""
    times = []
    for _ in range(cfg.num_runs):
        start_time = time.time()
        random = dist.sample(sample_shape=torch.Size([cfg.num_samples]))
        with torch.no_grad():
            reversed_model(random.float().to(device))
        times.append(time.time() - start_time)
    average_time_image = sum(times) / (len(times) * cfg.num_samples)
    print("Average sampling time per image: {:.4f} sec over {} runs".format(average_time_image, cfg.num_runs))

    """density estimation"""
    times = []
    model.eval()
    for _ in range(cfg.num_runs):
        start_time = time.time()
        testing_routine(model, train_loader)
        times.append(time.time() - start_time)
    average_time_image = sum(times) / (len(times) * cfg.batch_size)
    print("Average density estiamtion time per image: {:.4f} sec over {} runs".format(average_time_image, cfg.num_runs))


def training_routine(model, train_loader, optimizer, loss_alpha):
    """training for MNIST and Fashion MNIST"""
    device = next(model.parameters()).device
    storage = register_activation_hooks(model, layer_name="storage")
    for inputs, _ in train_loader:
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs.to(device))
        weighted_sums = []
        for key in sorted(storage.keys()):
            weighted_sums.append(storage[key])
        loss = neg_log_likelihood(output, model, weighted_sums, loss_alpha)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()

def testing_routine(model, test_loader):
    device = next(model.parameters()).device
    storage = register_activation_hooks(model, layer_name="storage")
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            weighted_sums = []
            for key in sorted(storage.keys()):
                weighted_sums.append(storage[key])
            nat = neg_log_likelihood(output, model, weighted_sums)
            bpd = nat * np.log2(np.exp(1))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    main()
