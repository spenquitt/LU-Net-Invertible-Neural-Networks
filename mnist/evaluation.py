from __future__ import print_function
import torch
import yaml
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from easydict import EasyDict as edict
from functions import Dequantization, bpd_per_image
from model import LUNet, LUNetInverse
from test import testing_routine
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import save_image_from_numpy
from multiprocessing import Pool
from tqdm import tqdm
import projection


def main():
    """load config"""
    with open("config.yaml", "r") as yamlfile:
        cfg = edict(yaml.load(yamlfile, Loader=yaml.FullLoader))
    start_evaluation(cfg)

def start_evaluation(cfg):
    torch.manual_seed(cfg.random_seed)
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    test_kwargs = {'batch_size': cfg.test_batch_size, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4, 'pin_memory': True}
        test_kwargs.update(cuda_kwargs)

    """Loading data"""
    transform_test = transforms.Compose([transforms.ToTensor(), Dequantization()])
    
    if cfg.data == 'MNIST':
        test_dataset = datasets.MNIST('data', train=False, transform=transform_test)
    if cfg.data == 'FashionMNIST':
        test_dataset = datasets.FashionMNIST('data', train=False, transform=transform_test)

    if cfg.mnist_target != "All":
        print("-> subset with target {}".format(cfg.mnist_target))
        idx = test_dataset.targets==cfg.mnist_target
        test_dataset.targets = test_dataset.targets[idx]
        test_dataset.data = test_dataset.data[idx]

    """initialize model"""
    load_path =  Path(cfg.checkpoints_dir) / Path("{}/target_{}/best.pth".format(cfg.data, cfg.mnist_target))
    state_dict = torch.load(load_path)
    model = LUNet(cfg[cfg.model].num_lu_blocks, cfg[cfg.model].layer_size, device)
    model.to(device).eval()
    model.load_state_dict(state_dict)
    
    """projection"""
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2000, shuffle=True)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data = example_data.to(device)
    output = model(example_data)
    projection.projection(output, cfg.data, cfg.mnist_target, cfg[cfg.model].layer_size)
    
    """
    compute negative log likelihood on test set, report in bits per dimension (bpd)
    * note that the bpd score may vary in different runs as the score is affected 
      by the (stochastic) input dequantization method with uniform noise
    """
    test_bpd_scores = []
    for i in range(cfg.num_runs):
        print("Run {:2d} :".format(i+1))
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        test_bpd = testing_routine(model, test_loader, cfg.data, cfg.mnist_target, return_bpd=True)
        test_bpd_scores.append(test_bpd)
        print("Test set : bits/dim : {:.04f} \n".format(test_bpd))
    test_bpd_mean = np.mean(test_bpd_scores)
    test_bpd_std = np.std(test_bpd_scores)
    print("Averaged over {} runs.".format(cfg.num_runs))
    print("Test set : bits/dim : {:.04f} +- {:.04f} \n".format(test_bpd_mean, test_bpd_std))

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
    random = dist.sample(sample_shape=torch.Size([cfg.num_samples]))
    with torch.no_grad():
        generated = reversed_model(random.float().to(device))
    
    generated_images = reconstruct(generated)
    """saving new images as PNGs"""
    bpd_scores = bpd_per_image(generated, model)
    save_dir = Path("outputs/{}/sampling/target_{}/".format(cfg.data, cfg.mnist_target))
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Saved generated images at :", save_dir)
    mp_args = [(generated_images[i], 
                save_dir / Path("{:.4f}.png".format(bpd_scores[i]))) 
                for i in range(len(generated))]
    with Pool(cfg.num_cores) as pool:
        pool.starmap(save_image_from_numpy, tqdm(mp_args, total=mp_args.__len__()), chunksize = 4)

    """saving original images as PNGs"""
    test_batch, _ = next(iter(test_loader))
    original_images = reconstruct(test_batch)
    bpd_scores = bpd_per_image(test_batch, model)
    save_dir = Path("outputs/{}/original/target_{}/".format(cfg.data, cfg.mnist_target))
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Saved original images at :", save_dir)
    mp_args = [(original_images[i], 
                save_dir / Path("{:.4f}.png".format(bpd_scores[i]))) 
                for i in range(len(generated))]
    with Pool(cfg.num_cores) as pool:
        pool.starmap(save_image_from_numpy, tqdm(mp_args, total=mp_args.__len__()), chunksize = 4)

    """interpolate in latent space"""
    save_dir = Path("outputs/{}/interpolation/target_{}/".format(cfg.data, cfg.mnist_target))
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Saved interpolation images at :", save_dir)

    """define the other classes to interpolate to"""
    if cfg.another_mnist_target is None:
        another_mnist_target_list = [i for i in range(10)]
        another_mnist_target_list.remove(cfg.mnist_target)
    elif isinstance(cfg.another_mnist_target, int):
        another_mnist_target_list = [cfg.another_mnist_target]
    else:
        another_mnist_target_list = cfg.another_mnist_target

    """generating and saving interpolated images as PNGs"""
    for the_other_target in another_mnist_target_list:
        if cfg.data == 'MNIST':
            test_dataset = datasets.MNIST('data', train=False, transform=transform_test)
        if cfg.data == 'FashionMNIST':
            test_dataset = datasets.FashionMNIST('data', train=False, transform=transform_test)
        all_idx = np.array([i for i in range(len(test_dataset))])

        idx = all_idx[(test_dataset.targets==cfg.mnist_target).tolist()]
        random = torch.randint(0, len(idx), (1,)).item()
        #example_1 = test_dataset[idx[random]][0] # random example
        example_1 = test_dataset[idx[0]][0]
        input_1 = example_1.view(1, 1, -1).to(device)
        latent_1 = model(input_1)

        idx = all_idx[(test_dataset.targets==the_other_target).tolist()]
        random = torch.randint(0, len(idx), (1,)).item()
        example_2 = test_dataset[idx[random]][0] # random example
        input_2 = example_2.view(1, 1, -1).to(device)
        latent_2 = model(input_2)

        latents_list = []
        interpol_coefs = np.linspace(0, 1, 21)
        for alpha in interpol_coefs: # interpolation between latent_1 and latent_2
            latents_list.append((1-alpha) * latent_1 + alpha * latent_2)
        latents = torch.cat(latents_list, dim = 0)

        with torch.no_grad():
            reversed = reversed_model(latents)
        interpolated_images = reconstruct(reversed)
        
        print("interpolated from {} -> {}".format(cfg.mnist_target, the_other_target))
        mp_args = [(interpolated_images[i], 
                    save_dir / Path("{}_alpha_{:.02f}.png".format(the_other_target, interpol_coefs[i]))) 
                    for i in range(len(interpolated_images))]
        with Pool(cfg.num_cores) as pool:
            pool.starmap(save_image_from_numpy, tqdm(mp_args, total=mp_args.__len__()), chunksize = 4)


def reconstruct(input_vector):
    image = input_vector.cpu().detach().reshape(-1, 28, 28)
    image = Dequantization().__call__(image, reverse=True)
    return np.uint8(np.squeeze(image.numpy()))

if __name__ == '__main__':
    main()
