import csv
import json
import os
import re
import warnings
from collections import defaultdict
from typing import Literal, Optional, Union

import numpy as np

from hdf5_io import GroupedArrays, save_arrays, save_grouped

HasHeader = Union[bool, Literal["auto"]]

_RUN_ID_HEADER_RE = re.compile(r"^#\s*RUN_ID:\s*(.+)$", re.IGNORECASE)
_SIM_TYPES = ("log", "state", "estimator", "isf", "pair", "planewind")
_SIM_FILENAME_RE = re.compile(
    r"^sim-(?P<type>log|state|estimator|isf|pair|planewind)-"
    r"(?P<T>[^-]+)-(?P<L>[^-]+)-(?P<u>[^-]+)-(?P<t>[^-]+)-"
    r"(?P<run_id>RUN_\d+)$",
    re.IGNORECASE,
)
_NUMERIC_SIM_TYPES = frozenset({"estimator", "isf", "pair", "planewind"})


def _first_line_tokens(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        return []
    if "," in line and "\t" not in line and " " not in line.strip(","):
        return [token.strip() for token in line.split(",") if token.strip()]
    return line.split()


def _line_is_numeric(tokens: list[str]) -> bool:
    if not tokens:
        return False
    for token in tokens:
        try:
            float(token)
        except ValueError:
            return False
    return True


def _detect_header(path: str, has_header: HasHeader) -> bool:
    if has_header is True:
        return True
    if has_header is False:
        return False
    tokens = _first_line_tokens(path)
    return bool(tokens) and not _line_is_numeric(tokens)


def sanitize_dataset_name(name: str) -> str:
    cleaned = re.sub(r"[^\w.-]", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if not cleaned:
        raise ValueError(f"Invalid dataset label: {name!r}")
    return cleaned


def is_sim_dat(path: str) -> bool:
    if not path.lower().endswith(".dat"):
        return False
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            return bool(_RUN_ID_HEADER_RE.match(stripped))
    return False


def _read_run_id(path: str) -> Optional[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = _RUN_ID_HEADER_RE.match(line.strip())
            if match:
                return match.group(1).strip()
    return None


def _comment_header_tokens(line: str) -> Optional[list[str]]:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    body = stripped.lstrip("#").strip()
    if not body or body.upper().startswith("RUN_ID:"):
        return None
    tokens = body.split()
    if len(tokens) >= 2:
        return tokens
    return None


def parse_sim_dat(path: str) -> tuple[np.ndarray, list[str], dict[str, str]]:
    """Load a simulation estimator .dat file with comment metadata and column headers."""
    metadata: dict[str, str] = {}
    column_names: Optional[list[str]] = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                break
            run_id_match = _RUN_ID_HEADER_RE.match(stripped)
            if run_id_match:
                metadata["RUN_ID"] = run_id_match.group(1).strip()
                continue
            header_tokens = _comment_header_tokens(stripped)
            if header_tokens:
                column_names = header_tokens

    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 0:
        arr = np.array([arr])
    elif arr.ndim == 1:
        if column_names and len(column_names) > 1:
            arr = arr.reshape(-1, len(column_names))
        else:
            arr = arr.reshape(-1, 1)

    if column_names and arr.ndim == 2 and arr.shape[1] != len(column_names):
        raise ValueError(
            f"Column count mismatch in {path}: "
            f"expected {len(column_names)}, got {arr.shape[1]}"
        )

    if not column_names:
        column_names = [f"col_{i}" for i in range(arr.shape[1])]

    return arr, column_names, metadata


def sim_columns_to_datasets(
    arr: np.ndarray,
    column_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Split a simulation table into one 1D dataset per column heading."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D simulation array, got shape {arr.shape}")

    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, dict] = {}
    key_counts: dict[str, int] = {}

    for index, column_name in enumerate(column_names):
        base_key = sanitize_dataset_name(column_name)
        if base_key in arrays:
            count = key_counts.get(base_key, 1)
            while True:
                key = f"{base_key}_{count}"
                if key not in arrays:
                    break
                count += 1
            key_counts[base_key] = count + 1
        else:
            key = base_key
        arrays[key] = arr[:, index]
        attrs[key] = {"column_name": column_name, "column_index": index}

    return arrays, attrs


def sim_table_to_group(
    arr: np.ndarray,
    column_names: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Union[str, int]]]:
    """Store a simulation table as one 2D values array (gzip-friendly layout)."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D simulation array, got shape {arr.shape}")
    group_meta: dict[str, Union[str, int]] = {
        "column_names": json.dumps(column_names),
        "n_rows": int(arr.shape[0]),
        "n_columns": int(arr.shape[1]),
    }
    return {"values": arr}, group_meta


def parse_sim_filename(path: str) -> dict[str, str]:
    """Parse sim-<type>-T-L-u-t-RUN_ID.dat filename components."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _SIM_FILENAME_RE.match(stem)
    if not match:
        raise ValueError(
            f"Filename does not match sim-<type>-T-L-u-t-RUN_ID pattern: {path}"
        )
    return {key: value for key, value in match.groupdict().items()}


def is_sim_filename(path: str) -> bool:
    stem = os.path.splitext(os.path.basename(path))[0]
    return bool(_SIM_FILENAME_RE.match(stem))


def parse_sim_log(path: str) -> tuple[dict[str, str], dict[str, str], Optional[str]]:
    """Parse a simulation log: Simulation Parameters as dict, other sections as text."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    run_id: Optional[str] = None
    sim_params: dict[str, str] = {}
    preamble_lines: list[str] = []
    acceptance_lines: list[str] = []
    estimator_lines: list[str] = []

    section = "preamble"

    for line in lines:
        if "Begin Simulation Parameters" in line:
            section = "params"
            continue
        if "End Simulation Parameters" in line:
            section = "between"
            continue
        if "Begin Acceptance Data" in line:
            section = "acceptance"
            continue
        if "End Acceptance Data" in line:
            section = "between"
            continue
        if "Begin Estimator Data" in line:
            section = "estimator"
            continue
        if "End Estimator Data" in line:
            section = "between"
            continue

        if section == "preamble":
            preamble_lines.append(line)
            run_id_match = _RUN_ID_HEADER_RE.match(line.strip())
            if run_id_match:
                run_id = run_id_match.group(1).strip()
        elif section == "params":
            parts = re.split(r"\s*:\s*", line, maxsplit=1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if key:
                    sim_params[key] = value
        elif section == "acceptance":
            acceptance_lines.append(line)
        elif section == "estimator":
            estimator_lines.append(line)

    sections: dict[str, str] = {}
    preamble = "\n".join(preamble_lines).strip()
    if preamble:
        sections["preamble"] = preamble
    acceptance = "\n".join(acceptance_lines).strip()
    if acceptance:
        sections["acceptance"] = acceptance
    estimator = "\n".join(estimator_lines).strip()
    if estimator:
        sections["estimator"] = estimator

    return sim_params, sections, run_id


_PAIR_LINE_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")


def _is_binary_row(line: str) -> bool:
    stripped = line.strip().rstrip("]")
    tokens = stripped.split()
    return bool(tokens) and all(token in {"0", "1"} for token in tokens)


def _parse_int_line(line: str) -> Optional[np.ndarray]:
    tokens = line.strip().split()
    if not tokens:
        return None
    try:
        values = [int(token) for token in tokens]
    except ValueError:
        return None
    return np.array(values, dtype=np.int64)


def parse_sim_state(path: str) -> tuple[dict[str, Union[np.ndarray, str]], dict[str, Union[str, int]]]:
    """Parse a simulation state file into arrays for each logical section."""
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if not lines or not lines[0].strip():
        raise ValueError(f"Empty state file: {path}")

    try:
        leading_value = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Expected integer on first line of state file: {path}") from exc

    idx = 1
    pair_rows: list[list[int]] = []
    while idx < len(lines):
        match = _PAIR_LINE_RE.match(lines[idx])
        if not match:
            break
        pair_rows.append([int(match.group(1)), int(match.group(2))])
        idx += 1

    grid_spec = lines[idx].strip() if idx < len(lines) else ""
    idx += 1

    coordinate_lines: list[str] = []
    while idx < len(lines) and not _is_binary_row(lines[idx]):
        if lines[idx].strip():
            coordinate_lines.append(lines[idx])
        idx += 1

    matrix_rows: list[list[int]] = []
    while idx < len(lines) and _is_binary_row(lines[idx]):
        tokens = lines[idx].strip().rstrip("]").split()
        matrix_rows.append([int(token) for token in tokens])
        idx += 1

    identifiers: Optional[np.ndarray] = None
    for line in reversed(lines):
        parsed = _parse_int_line(line)
        if parsed is not None and parsed.size >= 20:
            identifiers = parsed
            break

    datasets: dict[str, Union[np.ndarray, str]] = {
        "pairs": np.array(pair_rows, dtype=np.int64),
    }
    if matrix_rows:
        datasets["matrix"] = np.array(matrix_rows, dtype=np.int64)
    if identifiers is not None and identifiers.size:
        datasets["identifiers"] = identifiers
    if coordinate_lines:
        datasets["coordinates_raw"] = "\n".join(coordinate_lines)

    attrs: dict[str, Union[str, int]] = {
        "leading_value": leading_value,
    }
    if grid_spec:
        attrs["grid_spec"] = grid_spec

    return datasets, attrs


def parse_sim_numeric(
    path: str,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, str], dict[str, Union[str, int]]]:
    """Parse a numeric simulation measurement file into a single 2D values dataset."""
    arr, column_names, metadata = parse_sim_dat(path)
    arrays, table_attrs = sim_table_to_group(arr, column_names)
    return arrays, column_names, metadata, table_attrs


def convert_sim_file_to_hdf5(
    dat_path: str,
    output_dir: str,
    compression: Optional[str] = None,
) -> str:
    """Convert one simulation .dat file to output_dir/RUN_ID.h5 with per-column datasets."""
    filename_meta = parse_sim_filename(dat_path)
    arr, column_names, file_meta = parse_sim_dat(dat_path)

    run_id_from_name = filename_meta["run_id"]
    run_id_from_file = file_meta.get("RUN_ID")
    if run_id_from_file and run_id_from_file.lower() != run_id_from_name.lower():
        raise ValueError(
            f"RUN_ID mismatch for {dat_path}: "
            f"filename={run_id_from_name}, file={run_id_from_file}"
        )

    arrays, attrs = sim_columns_to_datasets(arr, column_names)
    file_attrs = {
        "RUN_ID": run_id_from_name,
        "T": filename_meta["T"],
        "L": filename_meta["L"],
        "u": filename_meta["u"],
        "t": filename_meta["t"],
        "source_file": os.path.basename(dat_path),
    }

    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f"{run_id_from_name}.h5")
    save_arrays(
        h5_path,
        arrays,
        compression=compression,
        attrs=attrs,
        file_attrs=file_attrs,
    )
    return h5_path


def _validate_run_id_bucket(files: list[tuple[str, dict[str, str]]]) -> None:
    """Ensure all files in a run_id bucket share T, L, u, t."""
    if not files:
        return
    ref = files[0][1]
    for path, meta in files[1:]:
        for key in ("T", "L", "u", "t"):
            if meta[key] != ref[key]:
                raise ValueError(
                    f"Inconsistent {key} for RUN_ID {ref['run_id']}: "
                    f"{os.path.basename(path)}={meta[key]!r}, "
                    f"{os.path.basename(files[0][0])}={ref[key]!r}"
                )


_LOG_SECTION_H5_NAMES = {
    "preamble": "log_preamble",
    "acceptance": "log_acceptance",
    "estimator": "log_estimator",
}


def _build_run_id_groups(
    files: list[tuple[str, dict[str, str]]],
) -> tuple[
    GroupedArrays,
    dict[str, dict],
    dict[str, dict[str, dict]],
    dict[str, Union[str, int]],
    dict[str, str],
]:
    """Build HDF5 group data for one run_id bucket (no file write)."""
    _validate_run_id_bucket(files)

    ref_meta = files[0][1]
    run_id = ref_meta["run_id"]
    groups: dict[str, dict[str, Union[np.ndarray, str]]] = {}
    group_attrs: dict[str, dict] = {}
    dataset_attrs: dict[str, dict[str, dict]] = {}
    root_datasets: dict[str, str] = {}
    source_files: list[str] = []
    seen_types: set[str] = set()
    log_source_file: Optional[str] = None
    sim_params: dict[str, str] = {}

    for path, filename_meta in files:
        file_type = filename_meta["type"].lower()
        basename = os.path.basename(path)
        source_files.append(basename)

        if file_type in seen_types:
            raise ValueError(
                f"Duplicate sim-{file_type} file for RUN_ID {run_id}: {basename}"
            )
        seen_types.add(file_type)

        if file_type in _NUMERIC_SIM_TYPES:
            arrays, column_names, file_meta, table_attrs = parse_sim_numeric(path)
            run_id_from_file = file_meta.get("RUN_ID")
            if run_id_from_file and run_id_from_file.lower() != run_id.lower():
                raise ValueError(
                    f"RUN_ID mismatch for {path}: "
                    f"filename={run_id}, file={run_id_from_file}"
                )
            groups[file_type] = arrays
            group_attrs[file_type] = {
                "source_file": basename,
                "delimiter": "\\t",
                **table_attrs,
            }

        elif file_type == "log":
            with open(path, encoding="utf-8") as log_file:
                log_source_raw = log_file.read()
            parsed_params, sections, run_id_from_file = parse_sim_log(path)
            if run_id_from_file and run_id_from_file.lower() != run_id.lower():
                raise ValueError(
                    f"RUN_ID mismatch for {path}: "
                    f"filename={run_id}, file={run_id_from_file}"
                )
            log_source_file = basename
            sim_params = parsed_params
            root_datasets["log_source_raw"] = log_source_raw
            for section_key, h5_name in _LOG_SECTION_H5_NAMES.items():
                if section_key in sections:
                    root_datasets[h5_name] = sections[section_key]

        elif file_type == "state":
            state_datasets, state_attrs = parse_sim_state(path)
            groups["state"] = state_datasets
            group_attrs["state"] = {
                "source_file": basename,
                **state_attrs,
            }

        else:
            raise ValueError(f"Unsupported simulation file type: {file_type}")

    missing_types = set(_SIM_TYPES) - seen_types
    if missing_types:
        warnings.warn(
            f"RUN_ID {run_id}: missing file type(s): {', '.join(sorted(missing_types))}",
            stacklevel=2,
        )

    run_attrs: dict[str, Union[str, int]] = {
        "RUN_ID": run_id,
        "T": ref_meta["T"],
        "L": ref_meta["L"],
        "u": ref_meta["u"],
        "t": ref_meta["t"],
        "source_files": json.dumps(sorted(source_files)),
    }
    if log_source_file:
        run_attrs["log_source_file"] = log_source_file
    run_attrs.update(sim_params)
    return groups, group_attrs, dataset_attrs, run_attrs, root_datasets


def _convert_run_id_bucket(
    files: list[tuple[str, dict[str, str]]],
    output_dir: str,
    compression: Optional[str] = None,
) -> str:
    """Merge all simulation file types for one RUN_ID into a single grouped HDF5 file."""
    groups, group_attrs, dataset_attrs, run_attrs, root_datasets = _build_run_id_groups(files)
    run_id = run_attrs["RUN_ID"]

    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f"{run_id}.h5")
    save_grouped(
        h5_path,
        groups,
        file_attrs=run_attrs,
        group_attrs=group_attrs,
        dataset_attrs=dataset_attrs,
        root_datasets=root_datasets or None,
        compression=compression,
    )
    return h5_path


def convert_batch(
    input_dir: str,
    output_dir: str,
    compression: Optional[str] = None,
) -> list[str]:
    """Convert simulation .dat files in a folder to one grouped .h5 per RUN_ID."""
    buckets: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)

    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path) or not name.lower().endswith(".dat"):
            continue
        if not is_sim_filename(path):
            continue
        meta = parse_sim_filename(path)
        buckets[meta["run_id"]].append((path, meta))

    if not buckets:
        raise ValueError(f"No simulation .dat files found in {input_dir}")

    return [
        _convert_run_id_bucket(files, output_dir, compression=compression)
        for _, files in sorted(buckets.items())
    ]


def _sim_file_attrs(metadata: dict[str, str]) -> Optional[dict]:
    if not metadata:
        return None
    return dict(metadata)


def sim_dataset_label(path: str, name: Optional[str] = None) -> str:
    if name:
        return sanitize_dataset_name(name)
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("-")
    label_parts: list[str] = []
    for part in parts:
        if re.fullmatch(r"[A-Za-z]+", part):
            label_parts.append(part.lower())
        else:
            break
    if label_parts:
        return sanitize_dataset_name("_".join(label_parts))
    return sanitize_dataset_name(stem)


def unique_dataset_label(
    path: str,
    used_labels: set[str],
    name: Optional[str] = None,
) -> str:
    """Return a unique dataset label, disambiguating duplicate simulation files with RUN_ID."""
    label = text_dataset_label(path, name)
    if label not in used_labels:
        used_labels.add(label)
        return label

    if is_sim_dat(path):
        run_id = _read_run_id(path) or "unknown"
        suffix = re.sub(r"[^\w]", "", run_id)[:8] or "dup"
        unique = sanitize_dataset_name(f"{label}_{suffix}")
        counter = 2
        while unique in used_labels:
            unique = sanitize_dataset_name(f"{label}_{suffix}_{counter}")
            counter += 1
        used_labels.add(unique)
        return unique

    raise ValueError(f"Duplicate dataset label: {label}")


def text_dataset_label(path: str, name: Optional[str] = None) -> str:
    if name:
        return sanitize_dataset_name(name)
    if is_sim_dat(path):
        return sim_dataset_label(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    return sanitize_dataset_name(stem)


def _load_plain_text_table(
    path: str,
    delimiter: Optional[str] = None,
    has_header: HasHeader = "auto",
) -> tuple[np.ndarray, Optional[list[str]]]:
    use_header = _detect_header(path, has_header)
    column_names: Optional[list[str]] = None
    skiprows = 0

    if use_header:
        column_names = _first_line_tokens(path)
        skiprows = 1

    load_delimiter = delimiter
    if load_delimiter is None and column_names:
        with open(path, encoding="utf-8") as f:
            first_line = f.readline()
        if "," in first_line:
            load_delimiter = ","

    arr = np.loadtxt(path, delimiter=load_delimiter, skiprows=skiprows)
    if arr.ndim == 0:
        arr = np.array([arr])
    elif arr.ndim == 1 and column_names and len(column_names) > 1:
        arr = arr.reshape(-1, len(column_names))

    return arr, column_names


def load_text_table(
    path: str,
    delimiter: Optional[str] = None,
    has_header: HasHeader = "auto",
) -> tuple[np.ndarray, Optional[list[str]]]:
    """Load a numeric plain-text table and optional column labels."""
    arr, column_names, _ = _load_input_file(path, delimiter=delimiter, has_header=has_header)
    return arr, column_names


def _load_input_file(
    path: str,
    delimiter: Optional[str] = None,
    has_header: HasHeader = "auto",
) -> tuple[np.ndarray, Optional[list[str]], dict[str, str]]:
    if is_sim_dat(path) and has_header is not False:
        arr, column_names, metadata = parse_sim_dat(path)
        return arr, column_names, metadata

    arr, column_names = _load_plain_text_table(
        path, delimiter=delimiter, has_header=has_header
    )
    return arr, column_names, {}


def _build_attrs(
    label: str,
    column_names: Optional[list[str]],
    metadata: dict[str, str],
) -> Optional[dict[str, dict]]:
    attrs: dict = {}
    if column_names:
        attrs["column_names"] = column_names
    attrs.update(metadata)
    if not attrs:
        return None
    return {label: attrs}


def load_manifest(path: str) -> dict[str, str]:
    """Load label,path pairs from a CSV manifest."""
    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"Manifest row must have label,path: {row}")
            label = sanitize_dataset_name(row[0].strip())
            txt_path = row[1].strip()
            if label in mapping:
                raise ValueError(f"Duplicate dataset label in manifest: {label}")
            mapping[label] = txt_path
    if not mapping:
        raise ValueError(f"No entries found in manifest: {path}")
    return mapping


def convert_text_to_hdf5(
    txt_path: str,
    h5_path: str,
    dataset_name: Optional[str] = None,
    delimiter: Optional[str] = None,
    has_header: HasHeader = "auto",
    compression: Optional[str] = None,
) -> list[str]:
    """Convert one plain-text file to labeled HDF5 dataset(s)."""
    arr, column_names, metadata = _load_input_file(
        txt_path, delimiter=delimiter, has_header=has_header
    )

    if is_sim_dat(txt_path) and has_header is not False and column_names:
        arrays, attrs = sim_columns_to_datasets(arr, column_names)
        save_arrays(
            h5_path,
            arrays,
            compression=compression,
            attrs=attrs,
            file_attrs=_sim_file_attrs(metadata),
        )
        return list(arrays.keys())

    label = text_dataset_label(txt_path, dataset_name)
    save_arrays(
        h5_path,
        {label: arr},
        compression=compression,
        attrs=_build_attrs(label, column_names, metadata),
    )
    return [label]


def convert_text_files_to_hdf5(
    mapping: dict[str, str],
    h5_path: str,
    delimiter: Optional[str] = None,
    has_header: HasHeader = "auto",
    compression: Optional[str] = None,
) -> list[str]:
    """Convert multiple labeled plain-text files into one HDF5 file."""
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, dict] = {}
    file_attrs: dict[str, str] = {}
    used_labels: set[str] = set()

    for manifest_label, txt_path in mapping.items():
        dataset_label = unique_dataset_label(txt_path, used_labels, name=manifest_label)
        arr, column_names, metadata = _load_input_file(
            txt_path, delimiter=delimiter, has_header=has_header
        )

        if is_sim_dat(txt_path) and has_header is not False and column_names:
            column_arrays, column_attrs = sim_columns_to_datasets(arr, column_names)
            for key, column_arr in column_arrays.items():
                if key in arrays:
                    raise ValueError(
                        f"Duplicate dataset key {key!r} when merging simulation file {txt_path!r}. "
                        "Use one .dat file per .h5 file."
                    )
                arrays[key] = column_arr
                attrs[key] = column_attrs[key]
            if "RUN_ID" in metadata:
                if file_attrs and file_attrs.get("RUN_ID") != metadata["RUN_ID"]:
                    raise ValueError("Multiple simulation files with different RUN_ID values in one .h5 file.")
                file_attrs["RUN_ID"] = metadata["RUN_ID"]
            continue

        arrays[dataset_label] = arr
        file_attrs_for_table = _build_attrs(dataset_label, column_names, metadata)
        if file_attrs_for_table:
            attrs.update(file_attrs_for_table)

    save_arrays(
        h5_path,
        arrays,
        compression=compression,
        attrs=attrs or None,
        file_attrs=file_attrs or None,
    )
    return list(arrays.keys())
