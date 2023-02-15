from __future__ import print_function
import hydra

from omegaconf import DictConfig

from train import start_training
from evaluation import start_evaluation

@hydra.main(config_path=".", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    start_training(cfg)
    start_evaluation(cfg)

if __name__ == '__main__':
    main()
