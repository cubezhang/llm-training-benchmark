#!/usr/bin/env python3
"""WikiText-103 loading helpers shared by training and evaluation."""

from __future__ import annotations

from pathlib import Path


WIKITEXT_FILES = {
    "train": (
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ),
    "validation": ("validation-00000-of-00001.parquet",),
    "test": ("test-00000-of-00001.parquet",),
}


def resolve_local_wikitext_files(dataset_dir: str, split: str) -> list[str]:
    if split not in WIKITEXT_FILES:
        raise ValueError(f"Unsupported WikiText split: {split}")
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DATASET_DIR does not exist: {root}")
    files = [root / name for name in WIKITEXT_FILES[split]]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"DATASET_DIR is missing {split} parquet file(s): " + ", ".join(missing)
        )
    return [str(path) for path in files]


def load_wikitext(split: str, dataset_dir: str | None = None):
    from datasets import load_dataset

    if dataset_dir:
        files = resolve_local_wikitext_files(dataset_dir, split)
        print(
            f"Loading WikiText-103 {split} from local parquet: {dataset_dir}",
            flush=True,
        )
        return load_dataset(
            "parquet",
            data_files={split: files},
            split=split,
        )
    print(f"Loading WikiText-103 {split} from Hugging Face", flush=True)
    return load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=split,
    )
