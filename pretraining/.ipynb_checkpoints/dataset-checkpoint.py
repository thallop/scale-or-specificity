from __future__ import annotations

from pathlib import Path
from typing import Optional

from ssl4eo_s12_dataset import build_ssl4eos12_dataset


def load_allowed_ids(file_path: Optional[str]) -> Optional[set[str]]:
    """
    Load a text file containing one sample ID per line.

    Args:
        file_path: Path to the text file. If None, no filtering is used.

    Returns:
        A set of allowed sample IDs, or None if no file path is provided.
    """
    if file_path is None:
        return None

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Allowed ID file not found: {path}")

    allowed_ids = {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    }

    return allowed_ids


def extract_sample_ids(batch: dict) -> list[str]:
    """
    Extract sample IDs from a raw WebDataset batch.

    Args:
        batch: Raw batch from WebDataset.

    Returns:
        List of sample IDs.
    """
    if "sample_id" in batch:
        values = batch["sample_id"]
    elif "__key__" in batch:
        values = batch["__key__"]
    else:
        raise KeyError("Neither 'sample_id' nor '__key__' was found in the batch.")

    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]

    try:
        return [str(v) for v in values.tolist()]
    except AttributeError:
        return [str(v) for v in values]


def build_croma_webdataset(
    root: str,
    raw_batch_size: int,
    split: str = "train",
    shuffle: Optional[bool] = None,
    shardshuffle: int = 100,
    deterministic: bool = False,
    seed: Optional[int] = None,
):
    """
    Build a WebDataset pipeline for SSL4EO-S12 v1.1.

    Args:
        root: Dataset root directory.
        raw_batch_size: Batch size at raw sample level.
        split: Dataset split.
        shuffle: Whether to shuffle samples and shards.
        shardshuffle: Number of shards to shuffle.
        deterministic: Whether WebDataset shuffling should be deterministic.
        seed: Random seed used by WebDataset.

    Returns:
        A batched WebDataset pipeline.
    """
    dataset = build_ssl4eos12_dataset(
        path=root,
        modalities=["S2L2A", "S1GRD"],
        split=split,
        batch_size=raw_batch_size,
        return_metadata=False,
        reindex_seasonal=False,
        shuffle=shuffle,
        shardshuffle=shardshuffle,
        deterministic=deterministic,
        seed=seed,
        partial=(split == "val"),
    )

    return dataset