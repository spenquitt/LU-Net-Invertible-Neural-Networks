import hydra

from omegaconf import DictConfig

from realnvp_mnist import start_training

@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    start_training(cfg)

if __name__ == '__main__':
    main()

