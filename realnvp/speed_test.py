import torch
import yaml
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from easydict import EasyDict as edict
from realnvp_mnist import testing_routine
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
from model import RealNVP, loss_fn
from torch.utils.data import DataLoader
import time

def main():
    """load config"""
    with open("config.yaml", "r") as yamlfile:
        cfg = edict(yaml.load(yamlfile, Loader=yaml.FullLoader))
    start_speed_test(cfg)

def start_speed_test(cfg):
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

    """Loading data"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        Dequantization(),
        ])
    if cfg.dataset == "MNIST":
        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    elif cfg.dataset == "FashionMNIST":
        train_dataset = datasets.FashionMNIST('data', train=True, transform=transform)
        
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    """initialize model"""
    model = RealNVP(num_coupling=cfg.num_coupling, num_final_coupling=cfg.num_final_coupling, planes=cfg.num_planes).to(device)
    best_path = Path("{}/{}/target_{}/best.pth".format(cfg.checkpoint_dir, cfg.dataset, cfg.mnist_target))
    checkpoint = torch.load(best_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_bpd']
    print("Checkpoint loaded from training epoch {} with validation bpd {:.4f}".format(epoch, loss.item()))
    model.validate()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    
    gpu_memory_before = torch.cuda.memory_allocated(curr_device)
    print(f'GPU memory before training: {gpu_memory_before}')
    
    model.train()

    times = []
    for _ in tqdm(range(1, cfg.num_epochs + 1)):
        start_time = time.time()
        training_routine(train_loader, model, loss_fn, optimizer)
        scheduler.step()
        times.append(time.time() - start_time)
    average_time_epoch = sum(times) / len(times)
    average_time_batch = average_time_epoch / cfg.batch_size
    print("Average training time per epoch: {:.4f} sec over {} epochs".format(average_time_epoch, cfg.num_epochs))
    print("Average training time per batch: {:.4f} sec over {} epochs".format(average_time_batch, cfg.num_epochs))
    gpu_memory_after = torch.cuda.memory_allocated(curr_device)
    print(f'GPU memory after training: {gpu_memory_after}')
    print(f'Training GPU memory usage: {gpu_memory_after-gpu_memory_before}')


    """generating new images by sampling from latent distribution"""
    model.eval()
    times = []
    for _ in tqdm(range(cfg.num_runs)):
        start_time = time.time()
        with torch.no_grad():
            random_latents = torch.normal(
                torch.zeros(cfg.num_samples, 784),
                torch.ones(cfg.num_samples, 784))
        reversed = model(random_latents.to(device))
        samples = Dequantization().__call__(reversed, reverse=True)
        times.append(time.time() - start_time)
    average_time_image = sum(times) / (len(times) * cfg.num_samples)
    print("Average sampling time per image: {:.4f} sec over {} runs".format(average_time_image, cfg.num_runs))

    """density estimation"""
    times = []
    model.eval()
    for _ in tqdm(range(cfg.num_runs)):
        start_time = time.time()
        testing_routine(train_loader, model, loss_fn)
        times.append(time.time() - start_time)
    average_time_image = sum(times) / (len(times) * cfg.batch_size)
    print("Average density estiamtion time per image: {:.4f} sec over {} runs".format(average_time_image, cfg.num_runs))
    
    gpu_memory_end = torch.cuda.memory_allocated(curr_device)
    print(f'GPU memory usage: {gpu_memory_end-gpu_memory_before}')


def training_routine(dataloader, model, loss_fn, optimizer):
    device = next(model.parameters()).device
    batch_size = len(next(iter(dataloader))[0])
    for inputs,_ in dataloader:
        y, s, norms, scale = model(inputs.to(device))
        loss, _ = loss_fn(y, s, norms, scale, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def testing_routine(dataloader, model, loss_fn):
    device = next(model.parameters()).device
    batch_size = len(next(iter(dataloader))[0])
    with torch.no_grad():
        model.validate()
        for inputs, _ in dataloader:
            y, s, norms, scale = model(inputs.to(device))
            loss, _ = loss_fn(y, s, norms, scale, batch_size)
            bpd = loss * np.log(255)  / (np.log(2) * 784)


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


if __name__ == '__main__':
    main()

