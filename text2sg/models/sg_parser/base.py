from omegaconf import DictConfig

from text2sg.utils import SceneGraph


class BaseSceneParser:
    """
    Parse an unstructured scene description into a structured scene specification which can be used by downstream
    modules.

    The current implementation of this class involves very basic room parsing for the layout module, but in the future
    we would want to have more sophisticated parsing of the scene type, shape, and objects generated, such as with a
    scene graph or other representation.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def parse(self, text: str) -> SceneGraph:
        """Parse scene description into a structured scene specification.

        :param scene_spec: unstructured scene specification
        :raises ValueError: scene spec type or room type not supported for parsing
        :return: structured scene specification
        """
        raise NotImplementedError
