import sys

from .instructscene_3dfront import InstructSceneThreedFrontDataset
from .commonscenes_3dfront import CommonScenesThreedFrontDataset
from .vg import VisualGenomeDataset

__all__ = ["InstructSceneThreedFrontDataset", "VisualGenomeDataset", "CommonScenesThreedFrontDataset"]


def get_dataloader(config):
    """Get the dataloader for the specified dataset."""
    try:
        return getattr(sys.modules[__name__], config.module)(config)
    except KeyError:
        raise ValueError(f"Dataset {config.module} not supported")
