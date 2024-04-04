import json

import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from text2sg.data import get_dataloader
from text2sg.models import sg_parser


@hydra.main(version_base=None, config_path="../../config", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    # load data
    loader = get_dataloader(cfg.data)

    # load model
    model: sg_parser.BaseSceneParser = getattr(sg_parser, cfg.model.module)(cfg.model)

    # run model on data
    distances = []
    for idx, (id_, inp_description, target) in tqdm(enumerate(loader)):
        # load model
        pred_scene_graph = model.parse(inp_description)
        pred_scene_graph.id = id_

        # evaluate model
        if idx % cfg.viz_frequency == 0:
            pred_scene_graph.visualize()
        distances.append(pred_scene_graph.compute_distance(target))

    print(f"Average scene graph distance: {np.mean(np.array(distances))}")
