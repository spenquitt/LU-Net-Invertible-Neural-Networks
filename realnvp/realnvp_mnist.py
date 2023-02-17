import torch
import yaml
import numpy as np
import time

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
from easydict import EasyDict as edict

from model import RealNVP, loss_fn

"""
Code adapted from: https://github.com/bjlkeng/sandbox/tree/master/realnvp
"""

def main():
    """load config"""
    with open("config.yaml", "r") as yamlfile:
        cfg = edict(yaml.load(yamlfile, Loader=yaml.FullLoader))
    start_training(cfg)


class Dequantization(object):
    def __init__(self,  alpha=0.05, quants=256):
        self.alpha = alpha
        self.quants = quants

    def __call__(self, tensor, reverse=False):
        if not reverse:
            tensor = tensor * 255.
            x = tensor + torch.rand_like(tensor)
            x = torch.logit(self.alpha + (1-self.alpha) * x / self.quants)
            return x
        else:
            x = torch.floor(self.quants / (1-self.alpha) * (torch.sigmoid(tensor) - self.alpha))
            return torch.clip(x, min=0, max=255) / 255


def training_routine(dataloader, model, loss_fn, optimizer):
    device = next(model.parameters()).device
    batch_size = len(next(iter(dataloader))[0])
    progress = tqdm(dataloader)
    bpd_sum_over_batches = 0
    for it, (inputs,_) in enumerate(progress):
        y, s, norms, scale = model(inputs.to(device))
        loss, _ = loss_fn(y, s, norms, scale, batch_size)
        bpd_sum_over_batches += (loss.item() + (784 * np.log(255))) / (784 * np.log(2))
        progress.set_postfix({'bits/pixel': bpd_sum_over_batches / (it+1)})
        optimizer.zero_grad()
        loss.backward()
        start_time = time.time()
        optimizer.step()
        end_time = time.time()
        print("Optimizer time: " + str(end_time - start_time))
        exit()
        

def testing_routine(dataloader, model, loss_fn, return_bpd=True):
    device = next(model.parameters()).device
    batch_size = len(next(iter(dataloader))[0])
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        model.validate()
        for inputs, _ in dataloader:
            y, s, norms, scale = model(inputs.to(device))
            loss, _ = loss_fn(y, s, norms, scale, batch_size)
            test_loss += loss
        model.train()
    test_loss /= num_batches
    test_loss += 784 * np.log(255)
    bpd = test_loss / (np.log(2) * 784)
    print(f"Test Error: \n Avg loss: {test_loss:.2f}; bits/pixel: {bpd:.2f} \n")
    if return_bpd:
        return bpd
    else:
        return test_loss



def start_training(cfg):
    available = torch.cuda.is_available()
    curr_device = torch.cuda.current_device()
    device = torch.device("cuda:0" if available else "cpu")
    device_count = torch.cuda.device_count() 
    device_name =  torch.cuda.get_device_name(0)

    print(f'Cuda available: {available}')
    print(f'Current device: {curr_device}')
    print(f'Device: {device}')
    print(f'Device count: {device_count}')
    print(f'Device name: {device_name}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        Dequantization(),
        ])
    if cfg.dataset == "MNIST":
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)

        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    elif cfg.dataset == "FashionMNIST":
        train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('data', train=False, transform=transform)


    if isinstance(cfg.mnist_target, int):
        print("Experiments on MNIST, subset with {}".format(cfg.mnist_target))
        idx = train_dataset.targets==cfg.mnist_target
        train_dataset.targets = train_dataset.targets[idx]
        train_dataset.data = train_dataset.data[idx]
        idx = test_dataset.targets==cfg.mnist_target
        test_dataset.targets = test_dataset.targets[idx]
        test_dataset.data = test_dataset.data[idx]
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

    model = RealNVP(num_coupling=cfg.num_coupling, num_final_coupling=cfg.num_final_coupling, planes=cfg.num_planes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    model.train()

    best_validation_bpd = 100000
    CHECKPOINTS_DIR = Path('{}/{}/target_{}'.format(cfg.checkpoint_dir, cfg.dataset, cfg.mnist_target))
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        training_routine(train_loader, model, loss_fn, optimizer)
        validation_bpd = testing_routine(test_loader, model, loss_fn)
        
        if cfg.save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_bpd': validation_bpd,
            }, CHECKPOINTS_DIR / Path('last.pth'))

            if validation_bpd < best_validation_bpd:
                best_validation_bpd = validation_bpd
                best_path = CHECKPOINTS_DIR / Path(f'best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_bpd': validation_bpd,
                }, best_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_bpd': validation_bpd,
                }, CHECKPOINTS_DIR / Path('epoch_{:02d}_bpd_{:.4f}.pth'.format(epoch+1, validation_bpd)))
        
        scheduler.step()

    print("Done.")

if __name__ == '__main__':
    main()