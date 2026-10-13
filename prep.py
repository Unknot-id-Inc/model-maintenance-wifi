"""Public preprocessing entry points.

All raw and prepared data interpretation lives in :mod:`gufu_data`; this module
keeps the historical ``prep`` import path without maintaining a second parser.
"""

from gufu_data import (
    PreparedDataset,
    PreprocessingConfig,
    load_legacy_pickle_pair,
    load_prepared_dataset,
    prepare_dataset_files,
    save_prepared_dataset,
)

__all__ = [
    "PreparedDataset",
    "PreprocessingConfig",
    "load_legacy_pickle_pair",
    "load_prepared_dataset",
    "prepare_dataset_files",
    "save_prepared_dataset",
]
