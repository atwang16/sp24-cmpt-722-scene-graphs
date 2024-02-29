from text2sg.utils import SceneGraph


def parse_vg_scene_graphs(scene_graphs) -> list[SceneGraph]:
    """
    Parse the Visual Genome scene graphs into a format that is compatible with the text2sg library.

    Args:
        scene_graphs (list[dict[str, Any]]): A list of scene graphs in the Visual Genome format.

    Returns:
        list[dict[str, Any]]: A list of scene graphs in the text2sg format.
    """
    parsed_scene_graphs = []
    for scene_graph in scene_graphs:
        parsed_scene_graph = {
            "objects": [
                {
                    "id": obj["object_id"],
                    "name": obj["names"][0],
                    "attributes": obj["attributes"],
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


def parse_vg_region_descriptions(region_descriptions) -> list[str]:
    """
    Parse the Visual Genome region descriptions into a format that is compatible with the text2sg library.

    Args:
        region_descriptions (list[dict[str, Any]]): A list of region descriptions in the Visual Genome format.

    Returns:
        list[str]: A list of region descriptions in the text2sg format.
    """
    descriptions = []
    for image in region_descriptions:
        for region in image["regions"]:
            full_description = ". ".join([region["phrase"].strip() for region_description in region_descriptions])
        descriptions.append(full_description)
    
    return descriptions
