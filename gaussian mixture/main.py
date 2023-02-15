from __future__ import print_function
import hydra
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from train import training_routine_gaussian
from test import testing_routine_gaussian
from visuals import init_loss_dict, loss_plotten
from model_inverted import testing_gaussian_network
import gaussian_mixture

@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    torch.set_default_dtype(torch.float64)
    
    torch.manual_seed(cfg.random_seed)
    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    """Before cuda, here changing gpu"""
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = hydra.utils.instantiate(cfg[cfg.model]).to(device)
    
    """Loading gaussian data"""
    train_loader = gaussian_mixture.generate_gaussian(cfg[cfg.model].layer_size)
    test_loader = gaussian_mixture.generate_gaussian(cfg[cfg.model].layer_size, numb_data=1000)
    
    """trained models are only tested"""
    if (cfg.train_or_test == 'test'):
        testing_gaussian_network("checkpoints/Gaussian/experiment{}.pth".format(cfg.checkpoint_number), cfg[cfg.model].num_lu_blocks, cfg[cfg.model].layer_size, device)
        exit()
            
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.gamma)
    init_loss_dict(cfg.num_epochs)
    
    for epoch in range(1, cfg.num_epochs + 1):
        print("\nEpoch {}:".format(epoch))
        training_routine_gaussian(model, device, train_loader, optimizer, epoch, cfg.batch_size)
        testing_routine_gaussian(model, device, test_loader, cfg.batch_size, plot_name="epoch_{:03d}.png".format(epoch))
        scheduler.step()
    
    if cfg.save_model:
        save_path = Path(cfg.checkpoints_dir) / Path("{}/experiment{}.pth".format(cfg.data, cfg.checkpoint_number))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print("Saved checkpoint:", save_path)
    
    loss_plotten()
    
    testing_gaussian_network(save_path, cfg[cfg.model].num_lu_blocks, cfg[cfg.model].layer_size, device)
        
if __name__ == '__main__':
    main()
