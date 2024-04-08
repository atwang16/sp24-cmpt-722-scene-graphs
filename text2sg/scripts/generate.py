import json
import os

import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from text2sg.data import get_dataloader
from text2sg.models import sg_parser
from text2sg.utils import InvalidSceneGraphError


@hydra.main(version_base=None, config_path="../../config", config_name="inference_config")
def main(cfg: DictConfig) -> None:
    load_dotenv()

    prompt = input("Enter a prompt: ")

    # make output visualization directory
    output_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(output_dir, exist_ok=True)

    # load data
    loader = get_dataloader(cfg.data)

    # load model
    model: sg_parser.BaseSceneParser = getattr(sg_parser, cfg.model.module)(cfg.model)

    # run model on data
    pred_scene_graph = model.parse(prompt)
    if cfg.validate:
        pred_scene_graph.validate(
            allowed_objects=[obj_type.replace("_", " ") for obj_type in loader.object_types],
            allowed_relationships=loader.predicate_types,
        )
    pred_scene_graph.visualize(output_dir)
    print(pred_scene_graph.export(format="json"))


if __name__ == "__main__":
    main()
