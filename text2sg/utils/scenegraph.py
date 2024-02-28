from dataclasses import dataclass
from typing import Self


@dataclass
class Object:
    id: str
    name: str
    attributes: list[str]


@dataclass
class Relationship:
    type: str
    subject: Object
    target: Object


@dataclass
class SceneGraph:
    room_type: str
    objects: list[Object]
    relationships: list[Relationship]

    @classmethod
    def from_json(cls, data: dict) -> Self:
        objects = {obj["id"]: Object(**obj) for obj in data["objects"]}
        relationships = []
        for rel in data["relationships"]:
            relationships.append(
                Relationship(type=rel["type"], subject=objects[rel["subject_id"]], target=objects[rel["target_id"]])
            )
        return cls(
            room_type=data["room_type"],
            objects=list(objects.values()),
            relationships=relationships,
        )
