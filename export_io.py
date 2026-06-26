import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import h5py
import numpy as np

from hdf5_io import is_run_id
from text_io import (
    _LOG_SECTION_H5_NAMES,
    _NUMERIC_SIM_TYPES,
    is_sim_filename,
    parse_sim_dat,
    parse_sim_filename,
    parse_sim_state,
)

MEASUREMENT_GROUPS = ("estimator", "isf", "pair", "planewind", "state")
_LOG_H5_TO_SECTION = {value: key for key, value in _LOG_SECTION_H5_NAMES.items()}


def build_sim_filename(
    file_type: str,
    T: str,
    L: str,
    u: str,
    t: str,
    run_id: str,
) -> str:
    return f"sim-{file_type}-{T}-{L}-{u}-{t}-{run_id}.dat"


def _read_dataset_text(item: h5py.Dataset) -> str:
    if item.attrs.get("text_format") == "utf-8-bytes":
        return item[()].tobytes().decode("utf-8")
    value = item[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "U", "O"}:
        if value.shape == ():
            return str(value)
    return str(value)


def _is_nested_run_id_layout(f: h5py.File) -> bool:
    top_keys = list(f.keys())
    if not top_keys:
        return False
    if top_keys[0] in MEASUREMENT_GROUPS:
        return False
    if not is_run_id(top_keys[0]):
        return False
    run_grp = f[top_keys[0]]
    return isinstance(run_grp, h5py.Group) and any(
        name in run_grp for name in MEASUREMENT_GROUPS
    )


def _resolve_export_root(f: h5py.File) -> tuple[h5py.Group, dict[str, Any]]:
    if _is_nested_run_id_layout(f):
        run_id_keys = sorted(k for k in f.keys() if is_run_id(k))
        if len(run_id_keys) != 1:
            raise ValueError(
                "Nested HDF5 with multiple RUN_ID groups is not supported; "
                "use one .h5 file per RUN_ID"
            )
        run_grp = f[run_id_keys[0]]
        if not isinstance(run_grp, h5py.Group):
            raise ValueError(f"Expected group at /{run_id_keys[0]}")
        attrs = dict(run_grp.attrs)
        return run_grp, attrs

    attrs = dict(f.attrs)
    return f, attrs


def _attr_str(attrs: dict[str, Any], key: str) -> str:
    value = attrs.get(key)
    if value is None:
        raise ValueError(f"Missing required attribute: {key}")
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_table_from_group(grp: h5py.Group) -> tuple[np.ndarray, list[str]]:
    if "values" in grp and isinstance(grp["values"], h5py.Dataset):
        values = grp["values"][()]
        if "column_names" not in grp.attrs:
            raise ValueError(f"Missing column_names attribute in /{grp.name}")
        raw = grp.attrs["column_names"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        column_names = json.loads(raw)
        return values, column_names

    legacy_keys = sorted(
        name
        for name in grp.keys()
        if isinstance(grp[name], h5py.Dataset) and name != "values"
    )
    if not legacy_keys:
        raise ValueError(f"No table data found in /{grp.name}")

    arrays = [grp[name][()] for name in legacy_keys]
    expected_rows = arrays[0].shape[0]
    for name, arr in zip(legacy_keys, arrays):
        if arr.shape != (expected_rows,):
            raise ValueError(
                f"Legacy column {name} in /{grp.name} has shape {arr.shape}, "
                f"expected ({expected_rows},)"
            )
    values = np.column_stack(arrays)
    column_names = legacy_keys
    return values, column_names


def export_numeric_dat(
    path: str,
    run_id: str,
    column_names: list[str],
    values: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        out.write(f"# RUN_ID: {run_id}\n")
        header = " ".join(f"{name:>15}" for name in column_names)
        out.write(f"# {header}\n")
        for row in values:
            parts = [f"{value:15.8E}" for value in row]
            out.write(" ".join(parts) + "\n")


def _reconstruct_log_text(root: h5py.Group, attrs: dict[str, Any]) -> str:
    if "log_source_raw" in root and isinstance(root["log_source_raw"], h5py.Dataset):
        return _read_dataset_text(root["log_source_raw"])

    lines: list[str] = []
    if "log_preamble" in root and isinstance(root["log_preamble"], h5py.Dataset):
        preamble = _read_dataset_text(root["log_preamble"]).strip()
        if preamble:
            lines.append(preamble)

    lines.append("")
    lines.append("---------- Begin Simulation Parameters ----------")
    lines.append("")
    skip_keys = {
        "RUN_ID",
        "T",
        "L",
        "u",
        "t",
        "source_files",
        "log_source_file",
        "run_ids",
        "n_runs",
    }
    for key in sorted(attrs):
        if key in skip_keys:
            continue
        value = attrs[key]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        lines.append(f"{key:<28}\t:\t{value}")
    lines.append("")
    lines.append("---------- End Simulation Parameters --------------")
    lines.append("")

    section_markers = {
        "acceptance": (
            "---------- Begin Acceptance Data ---------------",
            "---------- End Acceptance Data -----------------",
        ),
        "estimator": (
            "---------- Begin Estimator Data ----------------",
            "---------- End Estimator Data ------------------",
        ),
    }
    for h5_name, section_key in _LOG_H5_TO_SECTION.items():
        if section_key == "preamble":
            continue
        if h5_name not in root or not isinstance(root[h5_name], h5py.Dataset):
            continue
        begin, end = section_markers[section_key]
        body = _read_dataset_text(root[h5_name]).strip()
        lines.append(begin)
        lines.append("")
        if body:
            lines.append(body)
        lines.append("")
        lines.append(end)
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def export_log_dat(path: str, root: h5py.Group, attrs: dict[str, Any]) -> None:
    text = _reconstruct_log_text(root, attrs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        out.write(text)


def export_state_dat(path: str, state_grp: h5py.Group) -> None:
    leading_value = int(state_grp.attrs.get("leading_value", 0))
    pairs = state_grp["pairs"][()] if "pairs" in state_grp else np.empty((0, 2), dtype=np.int64)
    matrix = state_grp["matrix"][()] if "matrix" in state_grp else None
    identifiers = (
        state_grp["identifiers"][()] if "identifiers" in state_grp else None
    )
    grid_spec = ""
    if "grid_spec" in state_grp.attrs:
        raw = state_grp.attrs["grid_spec"]
        grid_spec = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    coordinates_raw = ""
    if "coordinates_raw" in state_grp:
        coordinates_raw = _read_dataset_text(state_grp["coordinates_raw"])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        out.write(f"{leading_value}\n")
        for row in pairs:
            out.write(f"{int(row[0]):>14}\t{int(row[1]):>14}\n")
        if grid_spec:
            out.write(f"{grid_spec}\n")
        if coordinates_raw:
            out.write(coordinates_raw.rstrip("\n"))
            if not coordinates_raw.endswith("\n"):
                out.write("\n")
        if matrix is not None:
            rows = matrix.tolist()
            for index, row in enumerate(rows):
                tokens = " ".join(str(int(value)) for value in row)
                suffix = " ]" if index == len(rows) - 1 else " "
                out.write(f"  {tokens}{suffix}\n")
        if identifiers is not None and identifiers.size:
            out.write(" ".join(str(int(value)) for value in identifiers) + " ")


def _output_basename(
    file_type: str,
    attrs: dict[str, Any],
    grp: Optional[h5py.Group],
) -> str:
    if file_type == "log" and "log_source_file" in attrs:
        return _attr_str(attrs, "log_source_file")
    if grp is not None and "source_file" in grp.attrs:
        return _attr_str(dict(grp.attrs), "source_file")

    T = _attr_str(attrs, "T")
    L = _attr_str(attrs, "L")
    u = _attr_str(attrs, "u")
    t = _attr_str(attrs, "t")
    run_id = _attr_str(attrs, "RUN_ID")
    return build_sim_filename(file_type, T, L, u, t, run_id)


def export_run_h5(h5_path: str, output_dir: str) -> list[str]:
    written: list[str] = []
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        root, attrs = _resolve_export_root(f)
        run_id = _attr_str(attrs, "RUN_ID")

        log_name = _output_basename("log", attrs, None)
        log_path = os.path.join(output_dir, log_name)
        if any(name in root for name in ("log_source_raw", "log_preamble", "log_acceptance", "log_estimator")):
            export_log_dat(log_path, root, attrs)
            written.append(log_path)

        for file_type in sorted(_NUMERIC_SIM_TYPES):
            if file_type not in root:
                continue
            grp = root[file_type]
            if not isinstance(grp, h5py.Group):
                raise ValueError(f"Expected group /{file_type}")
            values, column_names = read_table_from_group(grp)
            out_name = _output_basename(file_type, attrs, grp)
            out_path = os.path.join(output_dir, out_name)
            export_numeric_dat(out_path, run_id, column_names, values)
            written.append(out_path)

        if "state" in root:
            state_grp = root["state"]
            if not isinstance(state_grp, h5py.Group):
                raise ValueError("Expected /state group")
            out_name = _output_basename("state", attrs, state_grp)
            out_path = os.path.join(output_dir, out_name)
            export_state_dat(out_path, state_grp)
            written.append(out_path)

    if not written:
        raise ValueError(f"No exportable simulation content found in {h5_path}")
    return written


@dataclass
class CompareResult:
    ok: bool
    file_type: str
    message: str


def _detect_file_type(path: str) -> str:
    meta = parse_sim_filename(path)
    return meta["type"].lower()


def _compare_numeric(orig_path: str, exported_path: str) -> CompareResult:
    orig_arr, orig_cols, _ = parse_sim_dat(orig_path)
    exp_arr, exp_cols, _ = parse_sim_dat(exported_path)
    if orig_cols != exp_cols:
        missing = [name for name in orig_cols if name not in exp_cols]
        return CompareResult(
            ok=False,
            file_type="numeric",
            message=f"column name mismatch (missing in export: {missing[:5]})",
        )
    if orig_arr.shape != exp_arr.shape:
        return CompareResult(
            ok=False,
            file_type="numeric",
            message=f"shape mismatch {orig_arr.shape} vs {exp_arr.shape}",
        )
    if not np.allclose(orig_arr, exp_arr, rtol=0, atol=0, equal_nan=True):
        diff = np.where(~np.isclose(orig_arr, exp_arr, rtol=0, atol=0, equal_nan=True))
        if diff[0].size:
            row = int(diff[0][0])
            col = int(diff[1][0])
            return CompareResult(
                ok=False,
                file_type="numeric",
                message=(
                    f"value mismatch at row {row}, column {orig_cols[col]}: "
                    f"{orig_arr[row, col]!r} vs {exp_arr[row, col]!r}"
                ),
            )
        return CompareResult(ok=False, file_type="numeric", message="value mismatch")
    return CompareResult(ok=True, file_type="numeric", message="data equivalent")


def _compare_log(orig_path: str, exported_path: str) -> CompareResult:
    with open(orig_path, "rb") as orig_file, open(exported_path, "rb") as exp_file:
        orig_bytes = orig_file.read()
        exp_bytes = exp_file.read()
    if orig_bytes == exp_bytes:
        return CompareResult(ok=True, file_type="log", message="byte-identical")
    if orig_bytes.replace(b"\r\n", b"\n") == exp_bytes.replace(b"\r\n", b"\n"):
        return CompareResult(ok=True, file_type="log", message="byte-identical (normalized newlines)")
    return CompareResult(
        ok=False,
        file_type="log",
        message=f"byte mismatch (orig {len(orig_bytes)} bytes, export {len(exp_bytes)} bytes)",
    )


def _compare_state(orig_path: str, exported_path: str) -> CompareResult:
    orig_data, orig_attrs = parse_sim_state(orig_path)
    exp_data, exp_attrs = parse_sim_state(exported_path)

    for key in ("leading_value", "grid_spec"):
        if orig_attrs.get(key) != exp_attrs.get(key):
            return CompareResult(
                ok=False,
                file_type="state",
                message=f"attribute {key!r} mismatch: {orig_attrs.get(key)!r} vs {exp_attrs.get(key)!r}",
            )

    for key in ("pairs", "matrix", "identifiers"):
        if key not in orig_data and key not in exp_data:
            continue
        if key not in orig_data or key not in exp_data:
            return CompareResult(ok=False, file_type="state", message=f"missing dataset {key!r}")
        orig_arr = orig_data[key]
        exp_arr = exp_data[key]
        if not isinstance(orig_arr, np.ndarray) or not isinstance(exp_arr, np.ndarray):
            if orig_arr != exp_arr:
                return CompareResult(ok=False, file_type="state", message=f"{key} mismatch")
            continue
        if not np.array_equal(orig_arr, exp_arr):
            return CompareResult(ok=False, file_type="state", message=f"{key} array mismatch")

    orig_coords = orig_data.get("coordinates_raw", "")
    exp_coords = exp_data.get("coordinates_raw", "")
    if orig_coords != exp_coords:
        return CompareResult(ok=False, file_type="state", message="coordinates_raw mismatch")
    return CompareResult(ok=True, file_type="state", message="data equivalent")


def compare_dat_files(orig_path: str, exported_path: str) -> CompareResult:
    file_type = _detect_file_type(orig_path)
    if file_type == "log":
        return _compare_log(orig_path, exported_path)
    if file_type == "state":
        return _compare_state(orig_path, exported_path)
    if file_type in _NUMERIC_SIM_TYPES:
        return _compare_numeric(orig_path, exported_path)
    return CompareResult(ok=False, file_type=file_type, message=f"unsupported type {file_type!r}")


def compare_dat_folders(orig_dir: str, exported_dir: str) -> list[tuple[str, CompareResult]]:
    results: list[tuple[str, CompareResult]] = []
    for name in sorted(os.listdir(orig_dir)):
        if not name.lower().endswith(".dat"):
            continue
        orig_path = os.path.join(orig_dir, name)
        if not os.path.isfile(orig_path) or not is_sim_filename(orig_path):
            continue
        exported_path = os.path.join(exported_dir, name)
        if not os.path.isfile(exported_path):
            results.append(
                (name, CompareResult(ok=False, file_type="?", message="missing exported file"))
            )
            continue
        results.append((name, compare_dat_files(orig_path, exported_path)))
    return results


def run_roundtrip(
    input_dir: str,
    work_dir: str = "roundtrip_work",
    compression: Optional[str] = None,
) -> tuple[list[str], list[tuple[str, CompareResult]]]:
    from text_io import convert_batch

    h5_dir = os.path.join(work_dir, "h5")
    dat_dir = os.path.join(work_dir, "dat")
    os.makedirs(h5_dir, exist_ok=True)
    os.makedirs(dat_dir, exist_ok=True)

    h5_paths = convert_batch(input_dir, h5_dir, compression=compression)
    exported_all: list[str] = []
    for h5_path in h5_paths:
        exported_all.extend(export_run_h5(h5_path, dat_dir))

    compare_results: list[tuple[str, CompareResult]] = []
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith(".dat"):
            continue
        orig_path = os.path.join(input_dir, name)
        if not os.path.isfile(orig_path) or not is_sim_filename(orig_path):
            continue
        exported_path = os.path.join(dat_dir, name)
        if not os.path.isfile(exported_path):
            compare_results.append(
                (name, CompareResult(ok=False, file_type="?", message="missing exported file"))
            )
            continue
        compare_results.append((name, compare_dat_files(orig_path, exported_path)))

    return exported_all, compare_results
