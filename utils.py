"""Validation and logging helpers for the GCE HDF5 converter."""

from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path

LOGGER_NAME = "gce_hdf5_converter"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the package logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    return logger


def validate_input_path(path: str) -> Path:
    """Ensure the input CSV exists and is a readable file."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Input path is not a file: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise PermissionError(f"Cannot read input file: {resolved}")
    return resolved


def validate_output_path(path: str) -> Path:
    """Ensure the output path is writable and parent directories exist."""
    resolved = Path(path).expanduser().resolve()
    parent = resolved.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"Output path is a directory, not a file: {resolved}")
    if resolved.exists() and not os.access(resolved, os.W_OK):
        raise PermissionError(f"Cannot write output file: {resolved}")
    if parent and not os.access(parent, os.W_OK):
        raise PermissionError(f"Cannot write to output directory: {parent}")
    return resolved


def validate_csv(path: Path) -> list[str]:
    """Validate CSV structure and return column names."""
    if path.stat().st_size == 0:
        raise ValueError(f"Input CSV is empty: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input CSV has no header row: {path}") from exc

        if not header or all(not cell.strip() for cell in header):
            raise ValueError(f"Input CSV header is missing or blank: {path}")

        columns = [cell.strip() for cell in header]
        if len(set(columns)) != len(columns):
            raise ValueError(f"Input CSV contains duplicate column names: {path}")

        row_count = 0
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            row_count += 1

    if row_count == 0:
        raise ValueError(f"Input CSV contains no data rows: {path}")

    return columns


def sanitize_dataset_name(name: str) -> str:
    """Return a safe HDF5 dataset label."""
    cleaned = re.sub(r"[^\w.-]", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if not cleaned:
        raise ValueError(f"Invalid dataset name: {name!r}")
    if cleaned[0].isdigit():
        cleaned = f"dataset_{cleaned}"
    return cleaned
