import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../config", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    pass
    # load data

    # load model

    # run model on data

    # evaluate model
