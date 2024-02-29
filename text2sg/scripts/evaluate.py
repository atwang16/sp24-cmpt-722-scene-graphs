import json

import hydra
import numpy as np
from omegaconf import DictConfig

from text2sg.data.vg import parse_vg_region_descriptions, parse_vg_scene_graphs
from text2sg.models import LLMBasedT2SGParser


def evaluate_scene_graph_parsing(cfg: DictConfig):
    # load data
    # TODO: split this out if we end up having more than one dataset here
    with open(cfg.data.scene_graphs, "r") as f:
        scene_graphs = json.load(f)
    with open(cfg.data.region_descriptions, "r") as f:
        region_descriptions = json.load(f)

        # preprocess data
        scene_graphs = parse_vg_scene_graphs(scene_graphs)
        region_descriptions = parse_vg_region_descriptions(region_descriptions)

    distances = []
    for gt_scene_graph, region_description in zip(scene_graphs, region_descriptions):
        # load model
        model = LLMBasedT2SGParser(cfg.model)
        pred_scene_graph = model(region_description)

        # evaluate model
        distances.append(pred_scene_graph.compute_distance(gt_scene_graph))

    print("Average scene graph distance", np.mean(np.array(distances)))


def evaluate_scene_generation(cfg: DictConfig):
    pass


@hydra.main(version_base=None, config_path="../../config", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    pass
    # load data

    # load model

    # run model on data

    # evaluate model
