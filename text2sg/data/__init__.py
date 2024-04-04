import sys

from .threedfront import ThreedFrontDataset
from .vg import VisualGenomeDataset

__all__ = ["ThreedFrontDataset", "VisualGenomeDataset"]


def get_dataloader(config):
    """Get the dataloader for the specified dataset."""
    try:
        return sys.modules[config.module](config)
    except KeyError:
        raise ValueError(f"Dataset {config.module} not supported")
