import os
import re
from typing import Any, Optional, Union

import h5py
import numpy as np

GroupedArrays = dict[str, dict[str, Union[np.ndarray, str]]]
NestedRuns = dict[str, GroupedArrays]
RECOMMENDED_COMPRESSION = "none"
MIN_COMPRESS_BYTES = 4096
DEFAULT_COMPRESSION = "gzip"
DEFAULT_COMPRESSION_OPTS = 4
DEFAULT_SHUFFLE = True
_RUN_ID_RE = re.compile(r"^RUN_\d+$", re.IGNORECASE)


def _resolve_compression(
    compression: Optional[str],
) -> tuple[Optional[str], Optional[int], Optional[bool]]:
    """Resolve compression settings. None or 'none' disables compression."""
    if compression is None or compression == "none":
        return None, None, None
    if compression == "gzip":
        return DEFAULT_COMPRESSION, DEFAULT_COMPRESSION_OPTS, DEFAULT_SHUFFLE
    if compression == "gzip-1":
        return DEFAULT_COMPRESSION, 1, DEFAULT_SHUFFLE
    if compression == "gzip-9":
        return DEFAULT_COMPRESSION, 9, DEFAULT_SHUFFLE
    if compression == "lzf":
        return "lzf", None, None
    raise ValueError(f"Unsupported compression: {compression!r}")


def _dataset_kwargs(compression: Optional[str], nbytes: int = 0) -> dict[str, Any]:
    comp, comp_opts, shuffle = _resolve_compression(compression)
    if comp is None:
        return {}
    if nbytes < MIN_COMPRESS_BYTES:
        return {}
    kwargs: dict[str, Any] = {"compression": comp}
    if comp_opts is not None:
        kwargs["compression_opts"] = comp_opts
    if shuffle is not None:
        kwargs["shuffle"] = shuffle
    return kwargs


def describe_compression(compression: Optional[str]) -> str:
    """Human-readable label for success messages."""
    if compression is None or compression == "none":
        return "none (largest files)"
    comp, comp_opts, shuffle = _resolve_compression(compression)
    if comp is None:
        return "none (largest files)"
    details: list[str] = []
    if comp_opts is not None:
        details.append(f"level {comp_opts}")
    if shuffle:
        details.append("shuffle")
    if details:
        return f"{comp} ({', '.join(details)})"
    return comp


def save_arrays(
    path: str,
    arrays: Union[np.ndarray, dict[str, np.ndarray]],
    compression: Optional[str] = None,
    attrs: Optional[dict[str, dict]] = None,
    file_attrs: Optional[dict] = None,
) -> None:
    """Save one or more NumPy arrays to an HDF5 file."""
    if isinstance(arrays, np.ndarray):
        arrays = {"data": arrays}

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with h5py.File(path, "w") as f:
        if file_attrs:
            for key, value in file_attrs.items():
                f.attrs[key] = value
        for name, arr in arrays.items():
            dset = f.create_dataset(name, data=arr, **_dataset_kwargs(compression, arr.nbytes))
            if attrs and name in attrs:
                for key, value in attrs[name].items():
                    dset.attrs[key] = value


def load_arrays(
    path: str,
    names: Optional[list[str]] = None,
) -> Union[np.ndarray, dict[str, np.ndarray]]:
    """Load NumPy arrays from an HDF5 file."""
    with h5py.File(path, "r") as f:
        keys = names or list(f.keys())
        loaded = {k: f[k][()] for k in keys}

    if set(loaded.keys()) == {"data"}:
        return loaded["data"]
    return loaded


def _write_dataset(
    parent: h5py.Group,
    name: str,
    data: Union[np.ndarray, str],
    compression: Optional[str],
    attrs: Optional[dict] = None,
) -> None:
    if isinstance(data, str):
        byte_len = len(data.encode("utf-8"))
        kwargs = _dataset_kwargs(compression, byte_len)
        if byte_len >= MIN_COMPRESS_BYTES and kwargs:
            byte_arr = np.frombuffer(data.encode("utf-8"), dtype=np.uint8)
            dset = parent.create_dataset(name, data=byte_arr, **kwargs)
            dset.attrs["text_format"] = "utf-8-bytes"
        else:
            arr = np.array(data, dtype=h5py.string_dtype(encoding="utf-8"))
            dset = parent.create_dataset(name, data=arr)
    else:
        kwargs = _dataset_kwargs(compression, data.nbytes)
        dset = parent.create_dataset(name, data=data, **kwargs)
    if attrs:
        for key, value in attrs.items():
            dset.attrs[key] = value


def _read_dataset(item: h5py.Dataset) -> Any:
    if item.attrs.get("text_format") == "utf-8-bytes":
        return item[()].tobytes().decode("utf-8")
    value = item[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "U", "O"}:
        if value.shape == ():
            return str(value)
    return value


def _write_type_groups(
    parent: h5py.Group,
    groups: GroupedArrays,
    group_attrs: Optional[dict[str, dict]] = None,
    dataset_attrs: Optional[dict[str, dict[str, dict]]] = None,
    compression: Optional[str] = None,
) -> None:
    for group_name, datasets in groups.items():
        grp = parent.create_group(group_name)
        if group_attrs and group_name in group_attrs:
            for key, value in group_attrs[group_name].items():
                grp.attrs[key] = value
        for dset_name, arr in datasets.items():
            attrs = None
            if dataset_attrs and group_name in dataset_attrs:
                attrs = dataset_attrs[group_name].get(dset_name)
            _write_dataset(grp, dset_name, arr, compression, attrs=attrs)


def save_grouped(
    path: str,
    groups: GroupedArrays,
    file_attrs: Optional[dict] = None,
    group_attrs: Optional[dict[str, dict]] = None,
    dataset_attrs: Optional[dict[str, dict[str, dict]]] = None,
    root_datasets: Optional[dict[str, Union[np.ndarray, str]]] = None,
    compression: Optional[str] = None,
) -> None:
    """Save nested groups of datasets (and optional string datasets) to HDF5."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with h5py.File(path, "w") as f:
        if file_attrs:
            for key, value in file_attrs.items():
                f.attrs[key] = value
        _write_type_groups(f, groups, group_attrs, dataset_attrs, compression)
        for name, data in (root_datasets or {}).items():
            _write_dataset(f, name, data, compression)


def save_nested_grouped(
    path: str,
    runs: NestedRuns,
    file_attrs: Optional[dict] = None,
    run_attrs: Optional[dict[str, dict]] = None,
    group_attrs: Optional[dict[str, dict[str, dict]]] = None,
    dataset_attrs: Optional[dict[str, dict[str, dict[str, dict]]]] = None,
    compression: Optional[str] = None,
) -> None:
    """Save one HDF5 file with a top-level group per run_id."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with h5py.File(path, "w") as f:
        if file_attrs:
            for key, value in file_attrs.items():
                f.attrs[key] = value
        for run_id, groups in runs.items():
            run_grp = f.create_group(run_id)
            if run_attrs and run_id in run_attrs:
                for key, value in run_attrs[run_id].items():
                    run_grp.attrs[key] = value
            run_group_attrs = group_attrs.get(run_id) if group_attrs else None
            run_dataset_attrs = dataset_attrs.get(run_id) if dataset_attrs else None
            _write_type_groups(
                run_grp,
                groups,
                run_group_attrs,
                run_dataset_attrs,
                compression,
            )


def load_grouped(path: str) -> dict[str, dict[str, Any]]:
    """Load all groups and datasets from a grouped HDF5 file."""
    loaded: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as f:
        for group_name in f.keys():
            grp = f[group_name]
            if not isinstance(grp, h5py.Group):
                continue
            loaded[group_name] = {}
            for dset_name in grp.keys():
                item = grp[dset_name]
                if isinstance(item, h5py.Dataset):
                    loaded[group_name][dset_name] = _read_dataset(item)
    return loaded


def load_nested_grouped(path: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Load a nested RUN_ID HDF5 file: run_id -> type -> dataset."""
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    with h5py.File(path, "r") as f:
        for run_id in f.keys():
            run_grp = f[run_id]
            if not isinstance(run_grp, h5py.Group):
                continue
            loaded[run_id] = {}
            for type_name in run_grp.keys():
                type_grp = run_grp[type_name]
                if not isinstance(type_grp, h5py.Group):
                    continue
                loaded[run_id][type_name] = {}
                for dset_name in type_grp.keys():
                    item = type_grp[dset_name]
                    if isinstance(item, h5py.Dataset):
                        loaded[run_id][type_name][dset_name] = _read_dataset(item)
    return loaded


def is_run_id(name: str) -> bool:
    return bool(_RUN_ID_RE.match(name))
