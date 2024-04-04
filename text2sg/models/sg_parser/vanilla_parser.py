import sng_parser

from text2sg.utils import SceneGraph
from .base import BaseSceneParser


class VanillaParser(BaseSceneParser):
    def parse(self, text: str):
        graph = sng_parser.parse(text)

        scene_graph = {
            "objects": [],
            "relationships": [],
        }

        for i, entity in enumerate(graph["entities"]):
            scene_graph["objects"].append(
                {
                    "id": i + 1,
                    "name": entity["head"],
                    "attributes": [modifier["span"] for modifier in entity["modifiers"] if modifier["dep"] != "det"],
                }
            )

        for relation in graph["relations"]:
            scene_graph["relationships"].append(
                {
                    "type": relation["relation"],
                    "subject_id": relation["subject"] + 1,
                    "target_id": relation["object"] + 1,
                }
            )

        return SceneGraph.from_json(scene_graph)
