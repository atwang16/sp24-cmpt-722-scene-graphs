import json

from omegaconf import DictConfig
from tqdm import tqdm

from text2sg.utils import SceneGraph


class VisualGenomeDataset:
    def __init__(self, cfg: DictConfig):
        with open(cfg.scene_graphs, "r") as f:
            scene_graphs = json.load(f)
        with open(cfg.region_descriptions, "r") as f:
            region_descriptions = json.load(f)

            # preprocess data
        self.scene_graphs = self.parse_vg_scene_graphs(scene_graphs)
        self.region_descriptions = self.parse_vg_region_descriptions(region_descriptions)

    def __len__(self):
        return len(self.scene_graphs)

    def __getitem__(self, index):
        return f"vg_{index}", self.region_descriptions[index], self.scene_graphs[index]

    @staticmethod
    def parse_vg_scene_graphs(scene_graphs) -> list[SceneGraph]:
        """
        Parse the Visual Genome scene graphs into a format that is compatible with the text2sg library.

        Args:
            scene_graphs (list[dict[str, Any]]): A list of scene graphs in the Visual Genome format.

        Returns:
            list[dict[str, Any]]: A list of scene graphs in the text2sg format.
        """
        parsed_scene_graphs = []
        for scene_graph in tqdm(scene_graphs):
            parsed_scene_graph = {
                "objects": [
                    {
                        "id": obj["object_id"],
                        "name": obj["names"][0],
                        "attributes": obj.get("attributes", []),
                    }
                    for obj in scene_graph["objects"]
                ],
                "relationships": [
                    {
                        "id": rel["relationship_id"],
                        "type": rel["predicate"].lower(),
                        "subject_id": rel["subject_id"],
                        "target_id": rel["object_id"],
                    }
                    for rel in scene_graph["relationships"]
                ],
            }
            parsed_scene_graphs.append(SceneGraph.from_json(parsed_scene_graph))
        return parsed_scene_graphs

    @staticmethod
    def parse_vg_region_descriptions(region_descriptions) -> list[str]:
        """
        Parse the Visual Genome region descriptions into a format that is compatible with the text2sg library.

        Args:
            region_descriptions (list[dict[str, Any]]): A list of region descriptions in the Visual Genome format.

        Returns:
            list[str]: A list of region descriptions in the text2sg format.
        """
        descriptions = []
        for image in tqdm(region_descriptions):
            for region in image["regions"]:
                full_description = ". ".join([region["phrase"].strip() for region_description in region_descriptions])
            descriptions.append(full_description)

        return descriptions
