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
    scenes = {}
    commonscenes_output = {"scans": []}
    error_count = 0
    processed_count = 0
    for idx, (id_, inp_description, target) in tqdm(enumerate(loader)):
        # load model
        try:
            pred_scene_graph = model.parse(inp_description)
            pred_scene_graph.id = id_
            pred_scene_graph.validate(
                allowed_objects=[obj_type.replace("_", " ") for obj_type in loader.object_types],
                allowed_relationships=loader.predicate_types,
            )
        except InvalidSceneGraphError as e:
            print(f"[ERROR] {e}")
            error_count += 1
        else:
            # evaluate model
            if idx % cfg.viz_frequency == 0:
                pred_scene_graph.visualize(output_dir)
            scenes[id_] = {
                "id": id_,
                "description": inp_description,
                "scene_graph": pred_scene_graph.export(format="json"),
            }
            distances.append(pred_scene_graph.compute_distance(target))
            commonscenes_output["scans"].append(pred_scene_graph.export(format="commonscenes"))

        processed_count += 1
        # end if we have enough samples
        if idx + 1 - error_count >= cfg.num_samples:
            break

    print(f"Average scene graph distance: {np.mean(np.array(distances))}")
    print(f"Scene graph success rate: {(1 - error_count / processed_count) * 100:.2f}%")
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(
            {
                "run": cfg.run_name,
                "ave_distance": distances,
                "num_samples": cfg.num_samples,
                "visualized": scenes,
            },
            f,
            indent=4,
        )

    with open(os.path.join(output_dir, "commonscenes_relationships.json"), "w") as f:
        json.dump(commonscenes_output, f, indent=4)


if __name__ == "__main__":
    main()
