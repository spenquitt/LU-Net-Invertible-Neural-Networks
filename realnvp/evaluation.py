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


def main():
    """load config"""
    with open("config.yaml", "r") as yamlfile:
        cfg = edict(yaml.load(yamlfile, Loader=yaml.FullLoader))
    start_evaluation(cfg)


def start_evaluation(cfg):
    torch.manual_seed(cfg.random_seed)
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
        test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    elif cfg.dataset == "FashionMNIST":
        test_dataset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)


    if isinstance(cfg.mnist_target, int):
        print("Experiments on MNIST, subset with {}".format(cfg.mnist_target))
        idx = test_dataset.targets==cfg.mnist_target
        test_dataset.targets = test_dataset.targets[idx]
        test_dataset.data = test_dataset.data[idx]

    """initialize model"""
    model = RealNVP(num_coupling=cfg.num_coupling, num_final_coupling=cfg.num_final_coupling, planes=cfg.num_planes).to(device)
    best_path = Path("{}/{}/target_{}/best.pth".format(cfg.checkpoint_dir, cfg.dataset, cfg.mnist_target))
    checkpoint = torch.load(best_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_bpd']
    print("Checkpoint loaded from training epoch {} with validation bpd {:.4f}".format(epoch, loss.item()))
    model.validate()

    """compute negative log likelihood on test set, report in bits per dimension (bpd)"""
    test_bpd_scores = []
    for i in range(cfg.num_runs):
        print("Run {:2d} :".format(i+1))
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
        test_bpd = testing_routine(test_loader, model, loss_fn)
        test_bpd_scores.append(test_bpd.item())
        print("Test set : bits/dim : {:.04f} \n".format(test_bpd))
    test_bpd_mean = np.nanmean(test_bpd_scores)
    test_bpd_std = np.nanstd(test_bpd_scores)
    print("Averaged over {} runs.".format(cfg.num_runs))
    print("Test set : bits/dim : {:.04f} +- {:.04f} \n".format(test_bpd_mean, test_bpd_std))

    """generating new images by sampling from latent distribution"""
    model.eval()
    with torch.no_grad():
        random_latents = torch.normal(
            torch.zeros(cfg.num_samples, 784),
            torch.ones(cfg.num_samples, 784))
        reversed = model(random_latents.to(device))
        samples = Dequantization().__call__(reversed, reverse=True)
        samples = np.uint8(np.squeeze(samples.cpu().numpy()) * 255)

    """compute likelihoods of samples"""
    model.validate()
    bpd_scores = []
    for i in tqdm(range(len(reversed))):
        with torch.no_grad():
            x = torch.unsqueeze(reversed[i], 0)
            latents, s, norms, scale = model(x.to(device))
            loss, _ = loss_fn(latents, s, norms, scale, 1)
            bpd_scores.append((loss.item() + (784 * np.log(255))) / (784 * np.log(2)))

    """saving new images as PNGs"""
    save_dir = Path("{}/{}/sampling/target_{}/".format(cfg.outputs_dir, cfg.dataset, cfg.mnist_target))
    save_dir.mkdir(parents=True, exist_ok=True)
    mp_args = [(samples[i], 
                save_dir / Path("{:.4f}.png".format(bpd_scores[i]))) 
                for i in range(len(samples))]
    with Pool(cfg.num_cores) as pool:
        pool.starmap(save_image_from_numpy, tqdm(mp_args, total=mp_args.__len__()), chunksize = 4)
    print("Saved generated images at :", save_dir)

    """saving original images as PNGs"""
    test_batch, _ = next(iter(test_loader))
    originals = Dequantization().__call__(test_batch, reverse=True)
    originals = np.uint8(np.squeeze(originals.cpu().numpy()) * 255)
    
    """compute likelihoods of samples"""
    bpd_scores = []
    for i in tqdm(range(len(test_batch))):
        with torch.no_grad():
            x = torch.unsqueeze(test_batch[i], 0)    
            latents, s, norms, scale = model(x.to(device))
            loss, _ = loss_fn(latents, s, norms, scale, 1)
            bpd_scores.append((loss.item() + (784 * np.log(255))) / (784 * np.log(2)))
    
    save_dir = Path("{}/{}/original/target_{}/".format(cfg.outputs_dir, cfg.dataset, cfg.mnist_target))
    save_dir.mkdir(parents=True, exist_ok=True)
    mp_args = [(originals[i], 
                save_dir / Path("{:.4f}.png".format(bpd_scores[i]))) 
                for i in range(len(originals))]
    with Pool(cfg.num_cores) as pool:
        pool.starmap(save_image_from_numpy, tqdm(mp_args, total=mp_args.__len__()), chunksize = 4)
    print("Saved generated images at :", save_dir)
    

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


def save_image_from_numpy(np_array, save_path):
    Image.fromarray(np_array).save(save_path)


def reconstruct(input_vector):
    image = input_vector.cpu().detach().reshape(-1, 28, 28)
    image = Dequantization().__call__(image, reverse=True)
    return np.uint8(np.squeeze(image.numpy()))

if __name__ == '__main__':
    main()

