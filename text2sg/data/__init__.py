import sys

from .threedfront import ThreedFrontDataset
from .vg import VisualGenomeDataset

__all__ = ["ThreedFrontDataset", "VisualGenomeDataset"]


def get_dataloader(config):
    """Get the dataloader for the specified dataset."""
    try:
        return getattr(sys.modules[__name__], config.module)(config)
    except KeyError:
        raise ValueError(f"Dataset {config.module} not supported")
