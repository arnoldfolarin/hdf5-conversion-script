import argparse
import json
import sys

import h5py

from hdf5_io import is_run_id
from text_io import sanitize_dataset_name

MEASUREMENT_GROUPS = ("estimator", "isf", "pair", "planewind", "state")
EXPECTED_ESTIMATOR_COLUMNS = [
    "K",
    "V",
    "V_ext",
    "V_int",
    "E",
    "E_mu",
    "K_N",
    "V_N",
    "E_N",
    "N",
    "N_2",
    "density",
]
LOG_ROOT_DATASETS = ("log_acceptance", "log_estimator")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify grouped RUN_ID HDF5 output (all simulation measurement types).",
    )
    parser.add_argument(
        "h5_file",
        nargs="?",
        default="output/sim.h5",
        help="Path to HDF5 file (default: output/sim.h5)",
    )
    return parser


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


def _print_run_attrs(grp: h5py.Group) -> None:
    for key in ("RUN_ID", "T", "L", "u", "t", "source_files"):
        if key in grp.attrs:
            print(f"  {key}: {grp.attrs[key]}")


def _print_file_attrs(f: h5py.File) -> None:
    print("file attrs:")
    for key in ("run_ids", "n_runs", "RUN_ID", "T", "L", "u", "t", "source_files", "log_source_file"):
        if key in f.attrs:
            print(f"  {key}: {f.attrs[key]}")


def _dataset_text_length(item: h5py.Dataset) -> int:
    if item.attrs.get("text_format") == "utf-8-bytes":
        return len(item[()].tobytes().decode("utf-8"))
    raw = item[()]
    return len(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))


def _check_log_at_root(root: h5py.Group, prefix: str) -> None:
    if "log" in root and isinstance(root["log"], h5py.Group):
        raise SystemExit(
            f"FAIL: {prefix}/log group should not exist "
            "(use file attrs for sim params and log_* root datasets)"
        )

    if "Temperature" not in root.attrs:
        raise SystemExit(f"FAIL: missing Temperature file attribute")
    if "log_source_file" not in root.attrs:
        raise SystemExit(f"FAIL: missing log_source_file file attribute")

    print(f"\n{prefix} (log metadata at file level)")
    print(f"  Temperature: {root.attrs['Temperature']}")
    sim_attr_count = len(root.attrs) - len(
        [k for k in ("RUN_ID", "T", "L", "u", "t", "source_files", "log_source_file") if k in root.attrs]
    )
    print(f"  simulation parameter attrs: {sim_attr_count}")

    for dset_name in LOG_ROOT_DATASETS:
        if dset_name not in root:
            raise SystemExit(f"FAIL: missing root dataset {dset_name}")
        item = root[dset_name]
        if not isinstance(item, h5py.Dataset):
            raise SystemExit(f"FAIL: {dset_name} is not a dataset")
        print(f"  {dset_name}: length={_dataset_text_length(item)}")

    if "log_preamble" in root:
        item = root["log_preamble"]
        if isinstance(item, h5py.Dataset):
            print(f"  log_preamble: length={_dataset_text_length(item)}")

    if "log_source_raw" in root:
        item = root["log_source_raw"]
        if isinstance(item, h5py.Dataset):
            print(f"  log_source_raw: length={_dataset_text_length(item)} (exact .dat export)")


def _check_type_groups(root: h5py.Group, prefix: str) -> None:
    present_groups = [name for name in MEASUREMENT_GROUPS if name in root]
    missing_groups = [name for name in MEASUREMENT_GROUPS if name not in root]
    if missing_groups:
        print(f"WARN: missing groups under {prefix}: {missing_groups}")

    for group_name in MEASUREMENT_GROUPS:
        if group_name not in root:
            continue
        grp = root[group_name]
        if not isinstance(grp, h5py.Group):
            raise SystemExit(f"FAIL: {prefix}/{group_name} is not a group")

        print(f"\n{prefix}/{group_name}/")
        if "source_file" in grp.attrs:
            print(f"  source_file: {grp.attrs['source_file']}")

        if group_name == "state":
            if "leading_value" in grp.attrs:
                print(f"  leading_value: {grp.attrs['leading_value']}")
            if "grid_spec" in grp.attrs:
                print(f"  grid_spec: {grp.attrs['grid_spec']}")
            for dset_name in ("pairs", "matrix", "identifiers"):
                if dset_name in grp:
                    data = grp[dset_name]
                    print(f"  {dset_name}: shape={data.shape}, dtype={data.dtype}")
            if "coordinates_raw" in grp:
                raw_len = _dataset_text_length(grp["coordinates_raw"])
                print(f"  coordinates_raw: length={raw_len}")
            if "pairs" not in grp:
                raise SystemExit(f"FAIL: {prefix}/state/pairs dataset missing")
            continue

        _check_numeric_table_group(grp, group_name, prefix.rstrip("/") or "/")

    if len(present_groups) < len(MEASUREMENT_GROUPS):
        raise SystemExit(
            f"FAIL: expected {len(MEASUREMENT_GROUPS)} measurement groups under {prefix}, "
            f"found {len(present_groups)}"
        )


def _column_names_from_group(grp: h5py.Group) -> list[str]:
    if "column_names" not in grp.attrs:
        raise SystemExit(f"FAIL: missing column_names group attribute")
    raw = grp.attrs["column_names"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _sanitized_column_names(grp: h5py.Group) -> list[str]:
    return [sanitize_dataset_name(name) for name in _column_names_from_group(grp)]


def _check_numeric_table_group(grp: h5py.Group, group_name: str, prefix: str) -> None:
    if "values" not in grp:
        legacy = [k for k in grp.keys() if k != "values"]
        if legacy and "values" not in grp:
            raise SystemExit(
                f"FAIL: {prefix}/{group_name} uses legacy per-column layout; "
                "re-convert with current convert_batch.py (expects values dataset)"
            )
        raise SystemExit(f"FAIL: {prefix}/{group_name}/values dataset missing")

    values = grp["values"]
    if not isinstance(values, h5py.Dataset):
        raise SystemExit(f"FAIL: {prefix}/{group_name}/values is not a dataset")

    column_names = _column_names_from_group(grp)
    sanitized = _sanitized_column_names(grp)
    n_rows, n_cols = values.shape
    print(f"  values: shape=({n_rows}, {n_cols}), dtype={values.dtype}", end="")
    if values.compression:
        print(f", compression={values.compression}")
    else:
        print()
    print(f"  columns: {len(column_names)}")

    if group_name == "estimator":
        missing = [name for name in EXPECTED_ESTIMATOR_COLUMNS if name not in sanitized]
        if missing:
            raise SystemExit(f"FAIL: missing estimator columns in table: {missing}")
        print(f"  estimator rows: {n_rows}")


def _check_grouped_run_id(root: h5py.Group, h5_file: str, label: str = "") -> None:
    prefix = f"/{label}" if label else "/"
    if label:
        print(f"\n=== RUN_ID {label} ===")
        _print_run_attrs(root)
    _check_log_at_root(root, prefix.rstrip("/") or "/")
    _check_type_groups(root, prefix.rstrip("/") or "/")


def _check_nested_combined(f: h5py.File, h5_file: str) -> None:
    _print_file_attrs(f)
    run_id_keys = sorted(k for k in f.keys() if is_run_id(k))
    if not run_id_keys:
        raise SystemExit("FAIL: no RUN_ID groups found in nested HDF5 file")

    for run_id in run_id_keys:
        run_grp = f[run_id]
        if not isinstance(run_grp, h5py.Group):
            raise SystemExit(f"FAIL: /{run_id} is not a group")
        _check_grouped_run_id(run_grp, h5_file, label=run_id)

    print(f"\nOK: verified {len(run_id_keys)} RUN_ID run(s) in {h5_file}")


def _check_flat_estimator(h5_file: str) -> None:
    from hdf5_io import load_arrays

    data = load_arrays(h5_file)
    if not isinstance(data, dict):
        raise SystemExit("FAIL: expected a dict of per-column datasets")

    missing = [name for name in EXPECTED_ESTIMATOR_COLUMNS if name not in data]
    if missing:
        raise SystemExit(f"FAIL: missing datasets: {missing}")

    expected_rows = data[EXPECTED_ESTIMATOR_COLUMNS[0]].shape[0]
    print("datasets:", ", ".join(sorted(data.keys())))
    for name in EXPECTED_ESTIMATOR_COLUMNS:
        arr = data[name]
        print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.shape != (expected_rows,):
            raise SystemExit(f"FAIL: expected {name} shape ({expected_rows},), got {arr.shape}")

    print(f"OK: loaded {len(EXPECTED_ESTIMATOR_COLUMNS)} column datasets from {h5_file}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    h5_file = args.h5_file

    with h5py.File(h5_file, "r") as f:
        if _is_nested_run_id_layout(f):
            _check_nested_combined(f, h5_file)
            return

        top_keys = list(f.keys())
        if top_keys and isinstance(f[top_keys[0]], h5py.Group):
            _print_file_attrs(f)
            _check_grouped_run_id(f, h5_file)
            print(f"\nOK: verified grouped RUN_ID HDF5 at {h5_file}")
            return

    _check_flat_estimator(h5_file)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code:
            sys.exit(exc.code)
