"""Dataset download utilities."""

from graph_moes.download.download_graphbench_datasets import (
    download_graphbench_dataset,
    download_graphbench_datasets,
)
from graph_moes.download.download_ogb_datasets import download_ogb_datasets
from graph_moes.download.load_graphbench import (
    get_graphbench_dataset_info,
    get_graphbench_evaluator,
    load_graphbench_dataset,
)

__all__ = [
    "download_ogb_datasets",
    "download_graphbench_datasets",
    "download_graphbench_dataset",
    "load_graphbench_dataset",
    "get_graphbench_dataset_info",
    "get_graphbench_evaluator",
]
