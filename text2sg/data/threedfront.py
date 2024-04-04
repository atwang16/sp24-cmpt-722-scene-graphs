import json
import os
import pickle
from typing import Optional

import numpy as np
from omegaconf import DictConfig
from nltk.corpus import cmudict

from text2sg.utils import SceneGraph


class ThreedFrontDataset:
    """
    Much of this code is taken/adapted from InstructScene

    TODO: we could turn this into a real pytorch dataset, but we're not really using it for training and are not
    generating tensors, so it would really only be for convenient parallelization.
    """

    def __init__(
        self,
        cfg: DictConfig,
        scene_ids: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.seed = seed

        self._parse_train_stats(cfg.dataset_directory, cfg.train_stats)

        if scene_ids:
            self._tags = sorted([oi for oi in os.listdir(cfg.dataset_directory) if oi.split("_")[1] in scene_ids])
        else:
            self._tags = sorted([oi for oi in os.listdir(cfg.dataset_directory)])

        self._dataset = []
        for pi in self._tags:
            self._dataset.append(
                {
                    "id": pi,
                    "description_path": os.path.join(cfg.dataset_directory, pi, "descriptions.pkl"),
                    "models_info_path": os.path.join(cfg.dataset_directory, pi, "models_info.pkl"),
                }
            )

    @property
    def object_types(self):
        return self._object_types

    @property
    def predicate_types(self):
        return [
            "above",
            "left of",
            "in front of",
            "closely left of",
            "closely in front of",
            "below",
            "right of",
            "behind",
            "closely right of",
            "closely behind",
        ]

    def __len__(self):
        return len(self._dataset)

    def _parse_train_stats(self, dataset_directory, train_stats):
        with open(os.path.join(dataset_directory, train_stats), "r") as f:
            train_stats = json.load(f)
        self._centroids = train_stats["bounds_translations"]
        self._centroids = (np.array(self._centroids[:3]), np.array(self._centroids[3:]))
        self._sizes = train_stats["bounds_sizes"]
        self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
        self._angles = train_stats["bounds_angles"]
        self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))

        self._class_labels = train_stats["class_labels"]
        self._object_types = train_stats["object_types"]
        self._class_frequencies = train_stats["class_frequencies"]
        self._class_order = train_stats["class_order"]
        self._count_furniture = train_stats["count_furniture"]

        ################################ For InstructScene BEGIN ################################

        self._openshape_vitg14 = None
        if "bounds_openshape_vitg14_features" in train_stats:
            self._openshape_vitg14 = train_stats["bounds_openshape_vitg14_features"]
            self._openshape_vitg14 = (np.array(self._openshape_vitg14[0]), np.array(self._openshape_vitg14[1]))

        ################################ For InstructScene END ################################

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        description_path = sample["description_path"]
        models_info_path = sample["models_info_path"]

        with open(description_path, "rb") as f:
            descriptions = pickle.load(f)

        with open(models_info_path, "rb") as f:
            models_info = pickle.load(f)

        # object_descs = [
        #     np.random.choice((model_info["blip_caption"], model_info["msft_caption"], model_info["chatgpt_caption"]))
        #     for model_info in models_info
        # ]
        object_descs = [mi["chatgpt_caption"] for mi in models_info]

        text, selected_relations, selected_descs = self._fill_templates(
            descriptions, self.object_types, self.predicate_types, object_descs, num_relations=(2, 4)
        )

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

        return sample["id"], text, SceneGraph.from_json(scene_graph)

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
        article = "an" if ThreedFrontDataset.starts_with_vowel_sound(word) else "a"
        return article

    @staticmethod
    def reverse_rel(rel: str) -> str:
        return {
            "above": "below",
            "below": "above",
            "in front of": "behind",
            "behind": "in front of",
            "left of": "right of",
            "right of": "left of",
            "closely in front of": "closely behind",
            "closely behind": "closely in front of",
            "closely left of": "closely right of",
            "closely right of": "closely left of",
        }[rel]

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
        selected_relation_indices = np.random.choice(
            len(desc["obj_relations"]),
            min(np.random.choice(num_relations), len(desc["obj_relations"])),
            replace=False,
        )
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
                    subject = f"{ThreedFrontDataset.get_article(s_name).replace('a', 'A')} {s_name}"
                    predicate = f" is {p_str} "
                    object = f"{ThreedFrontDataset.get_article(o_name)} {o_name}."
                else:  # 50% of the time to reverse the order
                    subject = f"{ThreedFrontDataset.get_article(o_name).replace('a', 'A')} {o_name}"
                    predicate = f" is {ThreedFrontDataset.reverse_rel(p_str)} "
                    object = f"{ThreedFrontDataset.get_article(s_name)} {s_name}."
            else:
                if np.random.rand() < 0.75:
                    s_name = object_descs[s]
                else:  # 25% of the time to use the object type as the description
                    s_name = object_types[obj_class_ids[s]].replace("_", " ")
                    s_name = f"{ThreedFrontDataset.get_article(s_name)} {s_name}"  # "a" or "an" is added
                if np.random.rand() < 0.75:
                    o_name = object_descs[o]
                else:
                    o_name = object_types[obj_class_ids[o]].replace("_", " ")
                    o_name = f"{ThreedFrontDataset.get_article(o_name)} {o_name}"

                p_str = predicate_types[p]
                rev_p_str = ThreedFrontDataset.reverse_rel(p_str)

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
