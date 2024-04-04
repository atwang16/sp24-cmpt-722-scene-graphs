import json
import os

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

    # make output visualization directory
    output_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(output_dir, exist_ok=True)

    # load data
    loader = get_dataloader(cfg.data)

    # load model
    model: sg_parser.BaseSceneParser = getattr(sg_parser, cfg.model.module)(cfg.model)

    # run model on data
    distances = []
    visualized_scenes = {}
    for idx, (id_, inp_description, target) in tqdm(enumerate(loader)):
        # load model
        pred_scene_graph = model.parse(inp_description)
        pred_scene_graph.id = id_

        # evaluate model
        if idx % cfg.viz_frequency == 0:
            visualized_scenes[id_] = inp_description
            pred_scene_graph.visualize(output_dir)
        distances.append(pred_scene_graph.compute_distance(target))

        # end if we have enough samples
        if idx + 1 >= cfg.num_samples:
            break

    print(f"Average scene graph distance: {np.mean(np.array(distances))}")
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(
            {
                "run": cfg.run_name,
                "ave_distance": distances,
                "num_samples": cfg.num_samples,
                "visualized": visualized_scenes,
            },
            f,
            indent=4,
        )
