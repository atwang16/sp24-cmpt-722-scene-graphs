import json

import hydra
import numpy as np
from omegaconf import DictConfig

from text2sg.data import get_dataloader
from text2sg import models


@hydra.main(version_base=None, config_path="../../config", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    # load data
    loader = get_dataloader(cfg.data)

    # load model
    model = getattr(models, cfg.model.module)(cfg.model)

    # run model on data
    distances = []
    for inp_description, target in loader:
        # load model
        pred_scene_graph = model.parse(inp_description)

        # evaluate model
        distances.append(pred_scene_graph.compute_distance(target))

    print("Average scene graph distance", np.mean(np.array(distances)))
