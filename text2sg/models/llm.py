import json
import os

from omegaconf import DictConfig
from openai import OpenAI

from text2sg.utils import SceneGraph

SCENE_GRAPH_PROMPT = """Please follow the examples in the Visual Genome dataset and generate a scene graph that best describes the following text:
"{}"
Return the output in a JSON format according to the following format:
{{
  "room_type": one of 'bedroom', 'livingroom', 'diningroom', or 'other',
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


class LLMBasedT2SGParser:
    def __init__(self, cfg: DictConfig):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_name = cfg.model_name

    def __call__(self, text: str) -> SceneGraph:
        inp = SCENE_GRAPH_PROMPT.format(text)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant helping a user generate a semantic scene graph from a text "
                        "description of a scene."
                    ),
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
