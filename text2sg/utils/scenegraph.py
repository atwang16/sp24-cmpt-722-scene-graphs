import os
from dataclasses import dataclass, field
from pprint import pprint
from typing import Generator, Optional, Self

import numpy as np
from pyvis.network import Network

from text2sg.utils.commonscenes import relationship_map


class InvalidSceneGraphError(Exception):
    pass


@dataclass
class Object:
    id: int
    name: str
    attributes: list[str]
    relationships: list["Relationship"] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

    def compute_kernel(self, other: Self) -> float:
        if self.embedding and other.embedding:
            return (
                1
                + self.embedding * other.embedding / (np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding))
            ) / 2
        else:
            return 2 / 3 * (1 if self.name == other.name else 0) + 1 / 3 * (
                1 if set(self.attributes) == set(other.attributes) else 0
            )

    def __eq__(self, __other: object) -> bool:
        return (
            isinstance(__other, Object)
            and self.id == __other.id
            and self.name == __other.name
            and set(self.attributes) == set(__other.attributes)
        )


@dataclass
class Relationship:
    id: int
    type: str
    subject: Object
    target: Object
    embedding: Optional[np.ndarray] = None

    def get_end(self, obj: Object) -> Object:
        if obj != self.subject and obj != self.target:
            raise ValueError(f"Object {obj} is not part of the relationship {self}")
        return self.target if self.subject == obj else self.subject

    def compute_kernel(self, other: Self) -> float:
        if self.embedding and other.embedding:
            return (
                1
                + self.embedding * other.embedding / (np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding))
            ) / 2
        else:
            return 1 if self.type == other.type else 0


@dataclass
class SceneGraph:
    id: Optional[str]
    objects: list[Object]
    relationships: list[Relationship]

    def validate(
        self, allowed_objects: Optional[list[str]] = None, allowed_relationships: Optional[list[str]] = None
    ) -> bool:
        # check object names
        for obj in self.objects:
            if allowed_objects is not None and obj.name not in allowed_objects:
                raise InvalidSceneGraphError(f"Found invalid object name: {obj.name}")
            if not isinstance(obj.id, int):
                raise InvalidSceneGraphError(f"Found object_id which is not an integer: {obj.id}")

        object_ids = set([obj.id for obj in self.objects])

        # check relationships
        for rel in self.relationships:
            if allowed_relationships is not None and rel.type not in allowed_relationships:
                raise InvalidSceneGraphError(f"Found invalid relationship type: {rel.type}")
            if rel.subject.id not in object_ids:
                raise InvalidSceneGraphError(f"Found subject_id which does not correspond to a real object: {rel}")
            if rel.target.id not in object_ids:
                raise InvalidSceneGraphError(f"Found target_id which does not correspond to a real object: {rel}")

        return True

    @classmethod
    def from_json(cls, data: dict, id: Optional[str] = None) -> Self:
        objects = {obj["id"]: Object(**obj) for obj in data["objects"]}
        relationships = []
        for rel in data["relationships"]:
            try:
                subject = objects[rel["subject_id"]]
            except KeyError:
                raise InvalidSceneGraphError(
                    f"Found subject_id which does not correspond to a real object: {rel}, object_ids={objects.keys()}"
                )
            try:
                target = objects[rel["target_id"]]
            except KeyError:
                raise InvalidSceneGraphError(
                    f"Found target_id which does not correspond to a real object: {rel}, object_ids={objects.keys()}"
                )
            relationships.append(
                Relationship(
                    id=rel.get("id"),
                    type=rel["type"],
                    subject=subject,
                    target=target,
                )
            )
            subject.relationships.append(relationships[-1])
            target.relationships.append(relationships[-1])
        return cls(
            id=id,
            objects=list(objects.values()),
            relationships=relationships,
        )

    def _compute_graph_kernel(self, other: Self, max_length: int = 4) -> float:
        """Compute similarity metric between two semantic scene graphs based on a p-length random walk.

        This implementation is based on https://graphics.stanford.edu/~mdfisher/Data/GraphKernel.pdf. The algorithm
        defines a p-th order walk graph kernel k_G^p as a similarity metric between two graphs G_a and G_b:

        k_G^p(G_a, G_b) := sum_{r \in G_a} sum_{s \in G_b} k_R^p(G_a, G_b, r, s)

        where r and s are starting nodes in the respective graphs.

        We define the pth-order kernel k_R^p as

        k_R^p(G_a, G_b, r, s) := \sum_{all combinations of paths in G_a and G_b starting at r, s}
            k_node(r_p, s_p) \prod_{i=1}^{p-1} k_node(r_i, s_i) k_{edge}(e_i, f_i)

        which recursively can be defined as

        k_R^p(G_a, G_b, r, s) := k_node(r, s) * sum_{r' \in G_a} sum_{s' \in G_b} k_edge(e, f) k_R^{p-1}(G_a, G_b, r', s')

        k_R^p then is intuitively the sum of similarities of all paths of length p starting from r and s, and k_G^p is
        the sum over all starting nodes r and s.

        :param other: a semantic scene graph to compare to
        :param max_length: maximum path length, defaults to 4
        :return: kernel similarity value
        """
        cached_path_distances = {}

        def _compute_rooted_walk_graph_kernel(start_1: Object, start_2: Object, max_length: int) -> float:
            if (start_1.id, start_2.id, max_length) in cached_path_distances:
                return cached_path_distances[(start_1.id, start_2.id, max_length)]

            node_sim = start_1.compute_kernel(start_2)

            if node_sim == 0 or max_length == 0:
                cached_path_distances[(start_1.id, start_2.id, max_length)] = node_sim
                return node_sim

            if start_1.relationships or start_2.relationships:
                total_sim = 0.0
                for rel_1 in start_1.relationships:
                    for rel_2 in start_2.relationships:
                        edge_sim = rel_1.compute_kernel(rel_2)

                        total_sim += edge_sim * _compute_rooted_walk_graph_kernel(
                            rel_1.get_end(start_1), rel_2.get_end(start_2), max_length - 1
                        )
            else:
                total_sim = 1.0

            total_sim *= node_sim
            cached_path_distances[(start_1.id, start_2.id, max_length)] = total_sim
            return total_sim

        graph_kernel = 0
        for obj_1 in self.objects:
            for obj_2 in other.objects:
                graph_kernel += _compute_rooted_walk_graph_kernel(obj_1, obj_2, max_length=max_length)
        return graph_kernel

    def _compute_graph_kernel_normalized(self, other: Self, max_length: int = 4) -> float:
        return self._compute_graph_kernel(other, max_length) / np.maximum(
            self._compute_graph_kernel(self, max_length), other._compute_graph_kernel(other, max_length)
        )

    def compute_distance(self, other: Self, max_length: int = 4) -> float:
        return 1 - self._compute_graph_kernel_normalized(other, max_length)

    def visualize(self, output_dir: Optional[str] = None):
        net = Network(directed=True)
        for obj in self.objects:
            net.add_node(obj.id, label=obj.name, color="blue")
            for attribute in obj.attributes:
                net.add_node(f"{obj.id}-{attribute}", label=attribute, color="yellow")
                net.add_edge(obj.id, f"{obj.id}-{attribute}")

        for relationship in self.relationships:
            net.add_edge(relationship.subject.id, relationship.target.id, label=relationship.type)
        if self.id:
            output_path = (
                os.path.join(output_dir, f"{self.id}_scene_graph.html")
                if output_dir is not None
                else f"{self.id}_scene_graph.html"
            )
            net.show(output_path, notebook=False)
        else:
            output_path = (
                os.path.join(output_dir, "scene_graph.html") if output_dir is not None else f"scene_graph.html"
            )
            net.show(output_path, notebook=False)

    def export(self, format="json"):
        if format == "json":
            output = {"objects": [], "relationships": []}
            for obj in self.objects:
                output["objects"].append({"id": obj.id, "name": obj.name, "attributes": obj.attributes})
            for rel in self.relationships:
                output["relationships"].append(
                    {"id": rel.id, "type": rel.type, "subject_id": rel.subject.id, "target_id": rel.target.id}
                )

        elif format == "commonscenes":
            output = {"scan": self.id, "objects": {}, "relationships": []}
            for obj in self.objects:
                output["objects"][obj.id] = obj.name
            for rel in self.relationships:
                # TODO: -1 is not a great default value but there is no guarantee as of now that the generated scene
                # graphs will be restricted to the relationship types. The only one we can "enforce" this for is the
                # LLM method, since we can prompt it to restrict its relationship set to the allowed ones.
                output["relationships"].append(
                    [rel.subject.id, rel.target.id, relationship_map.get(rel.type, -1), rel.type]
                )

        else:
            raise NotImplementedError(f"Format not supported: {format}")

        return output
