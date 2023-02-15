import yaml
import hydra
import torch

from torch import nn
from tqdm import tqdm
from pathlib import Path
from torchvision import datasets, transforms
from easydict import EasyDict as edict

from test import testing_routine
from utils import register_activation_hooks, reverse_state_dict_lunet, mutlivariate_normal_params
from functions import neg_log_likelihood
from functions import Dequantization


def main():
    """load config"""
    with open("config.yaml", "r") as yamlfile:
        cfg = edict(yaml.load(yamlfile, Loader=yaml.FullLoader))
    start_training(cfg)


def start_training(cfg):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(cfg.random_seed)
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = hydra.utils.instantiate(cfg[cfg.model]).to(device) # initialize LUNet
    
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
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            Dequantization()
        ])

        """load datasets"""
        if cfg.data == 'MNIST':
            print("Experiments on MNIST")
            train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
            test_dataset = datasets.MNIST('data', train=False, transform=transform_test)

        if cfg.data == 'FashionMNIST':
            print("Experiments on FashionMNIST")
            train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform_train)
            test_dataset = datasets.FashionMNIST('data', train=False, transform=transform_test)

        if isinstance(cfg.mnist_target, int):
            print("-> subset with target {}".format(cfg.mnist_target))
            idx = train_dataset.targets==cfg.mnist_target
            train_dataset.targets = train_dataset.targets[idx]
            train_dataset.data = train_dataset.data[idx]
            idx = test_dataset.targets==cfg.mnist_target
            test_dataset.targets = test_dataset.targets[idx]
            test_dataset.data = test_dataset.data[idx]
        elif cfg.mnist_target != "All":
            print("-> subset with target {}".format(cfg.mnist_target))
            idx = (train_dataset.targets==cfg.mnist_target[0]) | (train_dataset.targets==cfg.mnist_target[-1])
            train_dataset.targets = train_dataset.targets[idx]
            train_dataset.data = train_dataset.data[idx]
            idx = (test_dataset.targets==cfg.mnist_target[0]) | (test_dataset.targets==cfg.mnist_target[-1])
            test_dataset.targets = test_dataset.targets[idx]
            test_dataset.data = test_dataset.data[idx]
            
        """initialize data loaders"""
        train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        
        """start training"""
        start_training_mnist(model, train_loader, cfg, test_loader)

def start_training_mnist(model, train_loader, cfg, validation_loader=None):
    """start training for MNIST and Fashion MNIST"""
    device = next(model.parameters()).device
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    best_bpd = 100000 # set to high score as bpd is aimed to be minimized during training

    """saving paths"""
    if cfg.save_model:
        save_dir = Path(cfg.checkpoints_dir) / Path("{}/target_{}".format(cfg.data, cfg.mnist_target))
        save_dir.mkdir(parents=True, exist_ok=True)
        last_save_path = save_dir / Path("last.pth")
        best_save_path = save_dir / Path("best.pth")
        best_inverse_save_path = save_dir / Path("best_inverse.pth")
        best_stats_save_path = save_dir / Path("best_stats.pth")

    """training and validation loops"""
    for epoch in range(1, cfg.num_epochs + 1):
        print("\nEpoch {}:".format(epoch))
        training_routine(model, train_loader, optimizer, cfg.loss_alpha)
        if cfg.save_model:
            state_dict = model.state_dict()
            torch.save(state_dict, last_save_path)
        if validation_loader is not None:
            bpd = testing_routine(model, validation_loader, cfg.data, cfg.mnist_target, "epoch_{:03d}.png".format(epoch), return_bpd=True)
            if bpd < best_bpd:
                best_bpd = bpd
                print("In epoch {}, new best bdp : {:.4f}".format(epoch, best_bpd))
                if cfg.save_model:
                    epoch_save_path = save_dir / Path("epoch_{:02d}_bpd_{:.4f}.pth".format(epoch, best_bpd))
                    state_dict = model.state_dict()
                    torch.save(state_dict, epoch_save_path)
                    torch.save(state_dict, best_save_path)
                    print("Saved checkpoint:", epoch_save_path)
        scheduler.step()

    """save the checkpoint for LUNet inverse and statistics of latent representations"""
    if cfg.save_model:
        try: state_dict # if training loop is skipped
        except NameError: 
            state_dict = torch.load(best_save_path)
            model.load_state_dict(state_dict)
        state_dict_inverse_lunet = reverse_state_dict_lunet(state_dict)
        stats_dict = mutlivariate_normal_params(train_loader, model, device)
        torch.save(state_dict_inverse_lunet, best_inverse_save_path)
        torch.save(stats_dict, best_stats_save_path)


def training_routine(model, train_loader, optimizer, loss_alpha):
    """training for MNIST and Fashion MNIST"""
    model.train()
    device = next(model.parameters()).device
    storage = register_activation_hooks(model, layer_name="storage")
    train_nll = 0
    for inputs, _ in tqdm(train_loader):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs.to(device))
        weighted_sums = []
        for key in sorted(storage.keys()):
            weighted_sums.append(storage[key])
        loss = neg_log_likelihood(output, model, weighted_sums, loss_alpha)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()
        
        nll = neg_log_likelihood(output, model, weighted_sums)
        batch_size, layer_size = output.shape
        train_nll += nll / (batch_size * layer_size)
    print('Train set: nll nats : {:.4f}'.format(train_nll / len(train_loader)))

if __name__ == '__main__':
    main()
