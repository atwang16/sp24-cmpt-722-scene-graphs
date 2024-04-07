import json
import os
import pickle
from typing import Optional

import numpy as np
from openai import OpenAI
from omegaconf import DictConfig
from nltk.corpus import cmudict

from text2sg.utils import SceneGraph


PROMPT = "Rephrase the following text to make it more natural, while making as minimal changes to the content as possible and keeping the entire text in one paragraph: \n\n{}"


class CommonScenesThreedFrontDataset:
    """
    Much of this code is taken/adapted from InstructScene

    TODO: we could turn this into a real pytorch dataset, but we're not really using it for training and are not
    generating tensors, so it would really only be for convenient parallelization.
    """

    def __init__(
        self,
        cfg: DictConfig,
        seed: Optional[int] = None,
    ) -> None:
        self.seed = seed
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.object_types: list[str] = list(cfg.object_types)
        self.prompt = PROMPT

        with open(cfg.relationships, "r") as f:
            self.relationships = json.load(f)

        self.existing_prompts_path = cfg.existing_prompts
        if os.path.exists(cfg.existing_prompts):
            with open(cfg.existing_prompts, "r") as g:
                self.existing_prompts = json.load(g)
        else:
            self.existing_prompts = {}

    @property
    def predicate_types(self):
        return [
            "none",
            "left",
            "right",
            "front",
            "behind",
            "close by",
            "above",
            "standing on",
            "bigger than",
            "smaller than",
            "taller than",
            "shorter than",
            "symmetrical to",
            "same style as",
            "same super category as",
            "same material as",
        ]

    @staticmethod
    def reverse_rel(rel: str) -> str:
        return {
            "none": "none",
            "left": "right",
            "right": "left",
            "front": "behind",
            "behind": "front",
            "close by": "far away",
            "above": "below",
            "standing on": "supporting",
            "bigger than": "smaller than",
            "smaller than": "bigger than",
            "taller than": "shorter than",
            "shorter than": "taller than",
            "symmetrical to": "symmetrical to",
            "same style as": "same style as",
            "same super category as": "same super category as",
            "same material as": "same material as",
        }[rel]

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        scene_graph = self.relationships["scans"][idx]
        scene_id = scene_graph["scan"]

        descriptions = {"obj_class_ids": {}, "obj_relations": []}
        for object_id, object_name in scene_graph["objects"].items():
            descriptions["obj_class_ids"][int(object_id)] = self.object_types.index(object_name)
        for object_id, target_id, predicate_id, _ in scene_graph["relationships"]:
            descriptions["obj_relations"].append((int(object_id), int(predicate_id), int(target_id)))

        if scene_id in self.existing_prompts:
            rephrased_text = self.existing_prompts[scene_id]
            selected_relation_indices = list(range(len(descriptions["obj_relations"])))
            selected_relations = [descriptions["obj_relations"][idx] for idx in selected_relation_indices]
            selected_relations = [
                (int(s), int(p), int(o)) for s, p, o in selected_relations
            ]  # e.g., [(4, 2, 18), ...]; 4, 18 are class ids; 2 is predicate id
        else:
            text, selected_relations, _ = self._fill_templates(
                descriptions, self.object_types, self.predicate_types, None
            )
            rephrased_text = self.rephrase(text)
            self.existing_prompts[scene_id] = rephrased_text
            with open(self.existing_prompts_path, "w") as f:
                json.dump(self.existing_prompts, f)

        # build ground truth scene graph
        scene_graph = {
            "objects": [],
            "relationships": [],
        }
        added_objects = set()
        for idx, (subject_id, relationship_id, target_id) in enumerate(selected_relations):
            for id_ in (subject_id, target_id):
                if id_ not in added_objects:
                    scene_graph["objects"].append(
                        {
                            "id": id_,
                            "name": self.object_types[descriptions["obj_class_ids"][id_]],
                            "attributes": [],
                        }
                    )
                    added_objects.add(id_)
            scene_graph["relationships"].append(
                {
                    "id": idx + 1,
                    "type": self.predicate_types[relationship_id],
                    "subject_id": subject_id,
                    "target_id": target_id,
                }
            )

        return scene_id, rephrased_text, SceneGraph.from_json(scene_graph)

    """
    Taken from https://stackoverflow.com/questions/20336524/verify-correct-use-of-a-and-an-in-english-texts-python
    """

    @staticmethod
    def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
        for syllables in pronunciations.get(word, []):
            return syllables[0][-1].isdigit()

    @staticmethod
    def get_article(word):
        word = word.split(" ")[0]
        article = "an" if CommonScenesThreedFrontDataset.starts_with_vowel_sound(word) else "a"
        return article

    def _fill_templates(
        self,
        desc: dict[str, list],
        object_types: list[str],
        predicate_types: list[str],
        object_descs: Optional[list[str]] = None,
        num_relations: tuple[int, int] = (1, 2),
        return_obj_ids=False,
    ) -> tuple[str, dict[int, int], list[tuple[int, int, int]], list[tuple[str, str]]]:
        if object_descs is None:
            assert object_types is not None

        if self.seed is not None:
            np.random.seed(self.seed)

        obj_class_ids = desc["obj_class_ids"]  # map from object index to class id

        # Describe the relations between the main objects and others
        # selected_relation_indices = np.random.choice(
        #     len(desc["obj_relations"]),
        #     min(np.random.choice(num_relations), len(desc["obj_relations"])),
        #     replace=False,
        # )
        selected_relation_indices = list(range(len(desc["obj_relations"])))
        selected_relations = [desc["obj_relations"][idx] for idx in selected_relation_indices]
        selected_relations = [
            (int(s), int(p), int(o)) for s, p, o in selected_relations
        ]  # e.g., [(4, 2, 18), ...]; 4, 18 are class ids; 2 is predicate id
        selected_descs = []
        selected_sentences = []
        selected_object_ids = []  # e.g., [0, ...]; 0 is object id
        for idx in selected_relation_indices:
            s, p, o = desc["obj_relations"][idx]
            s, p, o = int(s), int(p), int(o)
            if object_descs is None:
                s_name = object_types[obj_class_ids[s]].replace("_", " ")
                o_name = object_types[obj_class_ids[o]].replace("_", " ")
                p_str = predicate_types[p]
                if np.random.rand() > 0.5:
                    subject = f"{CommonScenesThreedFrontDataset.get_article(s_name).replace('a', 'A')} {s_name}"
                    predicate = f" is {p_str} "
                    object = f"{CommonScenesThreedFrontDataset.get_article(o_name)} {o_name}."
                else:  # 50% of the time to reverse the order
                    subject = f"{CommonScenesThreedFrontDataset.get_article(o_name).replace('a', 'A')} {o_name}"
                    predicate = f" is {CommonScenesThreedFrontDataset.reverse_rel(p_str)} "
                    object = f"{CommonScenesThreedFrontDataset.get_article(s_name)} {s_name}."
            else:
                if np.random.rand() < 0.75:
                    s_name = object_descs[s]
                else:  # 25% of the time to use the object type as the description
                    s_name = object_types[obj_class_ids[s]].replace("_", " ")
                    s_name = f"{CommonScenesThreedFrontDataset.get_article(s_name)} {s_name}"  # "a" or "an" is added
                if np.random.rand() < 0.75:
                    o_name = object_descs[o]
                else:
                    o_name = object_types[obj_class_ids[o]].replace("_", " ")
                    o_name = f"{CommonScenesThreedFrontDataset.get_article(o_name)} {o_name}"

                p_str = predicate_types[p]
                rev_p_str = CommonScenesThreedFrontDataset.reverse_rel(p_str)

                if p_str in ["left of", "right of"]:
                    if np.random.rand() < 0.5:
                        p_str = "to the " + p_str
                        rev_p_str = "to the " + rev_p_str
                elif p_str in ["closely left of", "closely right of"]:
                    if np.random.rand() < 0.25:
                        p_str = "closely to the " + p_str.split(" ")[-2] + " of"
                        rev_p_str = "closely to the " + rev_p_str.split(" ")[-2] + " of"
                    elif np.random.rand() < 0.5:
                        p_str = "to the close " + p_str.split(" ")[-2] + " of"
                        rev_p_str = "to the close " + rev_p_str.split(" ")[-2] + " of"
                    elif np.random.rand() < 0.75:
                        p_str = "to the near " + p_str.split(" ")[-2] + " of"
                        rev_p_str = "to the near " + rev_p_str.split(" ")[-2] + " of"

                if np.random.rand() < 0.5:
                    verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                    if "lamp" in s_name:
                        verbs += ["Hang", "Install"]
                    verb = verbs[np.random.choice(len(verbs))]
                    subject = f"{verb} {s_name}"
                    predicate = f" {p_str} "
                    object = f"{o_name}."
                    selected_descs.append((s_name, o_name))
                    selected_object_ids.append(s)
                else:  # 50% of the time to reverse the order
                    verbs = ["Place", "Put", "Position", "Arrange", "Add", "Set up"]
                    if "lamp" in o_name:
                        verbs += ["Hang", "Install"]
                    verb = verbs[np.random.choice(len(verbs))]
                    subject = f"{verb} {o_name}"
                    predicate = f" {rev_p_str} "
                    object = f"{s_name}."
                    selected_descs.append((o_name, s_name))
                    selected_object_ids.append(o)
            selected_sentences.append(subject + predicate + object)

        text = ""
        conjunctions = [" Then, ", " Next, ", " Additionally, ", " Finally, ", " And ", " "]
        for i, sentence in enumerate(selected_sentences):
            if i == 0:
                text += sentence
            else:
                conjunction = conjunctions[np.random.choice(len(conjunctions))]
                while conjunction == " Finally, " and i != len(selected_sentences) - 1:
                    # "Finally" should be used only in the last sentence
                    conjunction = conjunctions[np.random.choice(len(conjunctions))]
                if conjunction != " ":
                    sentence = sentence[0].lower() + sentence[1:]
                text += conjunction + sentence

        if return_obj_ids:
            return text, selected_relations, selected_descs, selected_object_ids
        else:
            return (
                text,
                selected_relations,
                selected_descs,
            )  # return `selected_relations`, `selected_descs` for evaluation

    def rephrase(self, text: str):

        inp = self.prompt.format(text)

        response = self.client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an annotator helping to rephrase a text prompt to make it more natural.",
                },
                {"role": "user", "content": inp},
            ],
        )

        out = response.choices[0].message.content
        return out
