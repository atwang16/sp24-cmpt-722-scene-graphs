import json
import os

from omegaconf import DictConfig
from openai import OpenAI

from text2sg.utils import SceneGraph
from .base import BaseSceneParser


OV_SCENE_GRAPH_PROMPT = """Please follow the examples in the Visual Genome dataset and generate a scene graph that best describes the following text:
"{}"
Return the output in a JSON format according to the following format:
{{
  "room_type": type of room, such as bedroom,
  "objects": [
    {{
      "id": id of object,
      "name": name of object as string,
      "attributes": array of string,
    }}
  ],
  "relationships": [
    {{
        "type": type of relationship as string,
        "subject_id": id of object which is the subject of the relationship,
        "target_id": id of object which is the target of the relationship
    }}
  ]
}}"""

INSTRUCTSCENE_SCENE_GRAPH_PROMPT = """Please follow the examples in the Visual Genome dataset and generate a scene graph that best describes the following text:
"{}"
Return the output in a JSON format according to the following format:
{{
  "room_type": bedroom, diningroom, or livingroom,
  "objects": [
    {{
      "id": id of object,
      "name": name of object as string,
      "attributes": array of string,
    }}
  ],
  "relationships": [
    {{
        "type": type of relationship as string,
        "subject_id": id of object which is the subject of the relationship,
        "target_id": id of object which is the target of the relationship
    }}
  ]
}}

The object ID should start with 0 and increment. Every subject_id and target_id in relationships should correspond to an existing object ID.

The object name must be one of "armchair", "bookcase", "cabinet", "ceiling lamp", "chaise longue sofa", "chinese chair", "coffee table", "console table", "corner/side table", "desk", "dining chair", "dining table", "l-shaped sofa", "lazy sofa", "lounge chair", "loveseat sofa", "multi-seat sofa", "pendant lamp", "round end table", "shelf", "stool", "tv stand", "wardrobe", or "wine cabinet".

The relationship type must be one of "above", "left of", "in front of", "closely left of", "closely in front of", "below", "right of", "behind", "closely right of", or "closely behind". Pick the one that matches the best for each relationship.
"""


COMMONSCENES_SCENE_GRAPH_PROMPT = """Please follow the examples in the Visual Genome dataset and generate a scene graph that best describes the following text:
"{}"
Return the output in a JSON format according to the following format:
{{
  "room_type": bedroom, diningroom, or livingroom,
  "objects": [
    {{
      "id": id of object,
      "name": name of object as string,
      "attributes": array of string,
    }}
  ],
  "relationships": [
    {{
        "type": type of relationship as string,
        "subject_id": id of object which is the subject of the relationship,
        "target_id": id of object which is the target of the relationship
    }}
  ]
}}

The object ID should start with 0 and increment. Every subject_id and target_id in relationships should correspond to an existing object ID.

The object name must be one of "armchair", "bookcase", "cabinet", "ceiling lamp", "chaise longue sofa", "chinese chair", "coffee table", "console table", "corner/side table", "desk", "dining chair", "dining table", "l-shaped sofa", "lazy sofa", "lounge chair", "loveseat sofa", "multi-seat sofa", "pendant lamp", "round end table", "shelf", "stool", "tv stand", "wardrobe", or "wine cabinet".

The relationship type must be one of "left", "right", "front", "behind", "close by", "above", "standing on", "bigger than", "smaller than", "taller than", "shorter than', "symmetrical to", "same style as", "same super category as", or "same material as". Pick the one that matches the best for each relationship.
"""


class LLMSceneParser(BaseSceneParser):
    """
    Parse an unstructured scene description into a structured scene specification which can be used by downstream
    modules.

    The current implementation of this class involves very basic room parsing for the layout module, but in the future
    we would want to have more sophisticated parsing of the scene type, shape, and objects generated, such as with a
    scene graph or other representation.
    """

    prompts = {
        "open_vocabulary": OV_SCENE_GRAPH_PROMPT,
        "instructscene": INSTRUCTSCENE_SCENE_GRAPH_PROMPT,
        "commonscenes": COMMONSCENES_SCENE_GRAPH_PROMPT,
    }

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.prompt = LLMSceneParser.prompts[cfg.prompt_type]

    def parse(self, text: str) -> SceneGraph:
        """Parse scene description into a structured scene specification.

        The parsing function currently expects a "text" type scene specification and expects to find one of the
        following key phrases—"living room", "dining room", or "bedroom"—in the text description, which is used to
        specify the scene room type to generate. As of now, the scene parsing is fairly rudimentary and does not use
        any other information in the scene specification.

        :param scene_spec: unstructured scene specification
        :raises ValueError: scene spec type or room type not supported for parsing
        :return: structured scene specification
        """
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        inp = self.prompt.format(text)

        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant helping a user generate a semantic scene graph from a text description of a scene.",
                },
                {"role": "user", "content": inp},
            ],
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content
        try:
            output_json = json.loads(raw_output)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse scene graph response from OpenAI as JSON:\n{raw_output}")

        return SceneGraph.from_json(output_json)
