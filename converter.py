#!/usr/bin/env python3
"""Convert CSV simulation exports to HDF5 format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from utils import (
    sanitize_dataset_name,
    setup_logging,
    validate_csv,
    validate_input_path,
    validate_output_path,
)

CONVERTER_VERSION = "1.0.0"
DEFAULT_DATASET_NAME = "data"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert CSV simulation exports to HDF5 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output HDF5 file",
    )
    parser.add_argument(
        "--compression",
        choices=["gzip", "none"],
        default="gzip",
        help="HDF5 compression algorithm",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Name of the HDF5 dataset to create",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def _compression_kwargs(compression: str) -> dict:
    """Return h5py dataset kwargs for the selected compression mode."""
    if compression == "none":
        return {}
    return {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def convert_csv_to_hdf5(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    compression: str,
) -> None:
    """Read a CSV file and write numeric data to HDF5."""
    columns = validate_csv(input_path)
    frame = pd.read_csv(input_path)

    if frame.empty:
        raise ValueError(f"No data rows found in CSV: {input_path}")

    numeric_frame = frame.select_dtypes(include=[np.number])
    if numeric_frame.empty:
        raise ValueError(
            f"CSV contains no numeric columns suitable for HDF5 storage: {input_path}"
        )

    array = numeric_frame.to_numpy(dtype=np.float64)
    dataset_label = sanitize_dataset_name(dataset_name)

    with h5py.File(output_path, "w") as handle:
        handle.attrs["source_file"] = input_path.name
        handle.attrs["row_count"] = int(array.shape[0])
        handle.attrs["column_count"] = int(array.shape[1])
        handle.attrs["converter_version"] = CONVERTER_VERSION

        dataset = handle.create_dataset(
            dataset_label,
            data=array,
            **_compression_kwargs(compression),
        )
        dataset.attrs["columns"] = np.array(
            list(numeric_frame.columns),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        dataset.attrs["source_columns"] = np.array(
            columns,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


def main(argv: list[str] | None = None) -> int:
    """Run the CSV to HDF5 conversion CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging(verbose=args.verbose)

    try:
        input_path = validate_input_path(args.input)
        output_path = validate_output_path(args.output)
        logger.info("Reading CSV: %s", input_path)
        convert_csv_to_hdf5(
            input_path=input_path,
            output_path=output_path,
            dataset_name=args.dataset_name,
            compression=args.compression,
        )
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    except OSError as exc:
        logger.error("Failed to write HDF5 output: %s", exc)
        return 1

    logger.info("Wrote HDF5 file: %s", output_path)
    logger.info("Compression: %s", args.compression)
    return 0


if __name__ == "__main__":
    sys.exit(main())
