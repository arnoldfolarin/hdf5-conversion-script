import argparse
import os
import sys
from typing import Optional

from cli_prompts import (
    print_menu,
    prompt_compression,
    prompt_menu_choice,
    prompt_path,
)
from export_io import export_run_h5, run_roundtrip
from hdf5_io import RECOMMENDED_COMPRESSION, describe_compression
from text_io import convert_batch


def _format_size_mb(path: str) -> str:
    return f"{os.path.getsize(path) / (1024 * 1024):.1f} MB"


def cmd_dat_to_h5(
    input_dir: str,
    output_dir: str,
    compression: Optional[str],
) -> int:
    outputs = convert_batch(input_dir, output_dir, compression=compression)
    label = describe_compression(compression)
    abs_output_dir = os.path.abspath(output_dir)
    print(f"\nSuccess: converted {len(outputs)} HDF5 file(s) with compression={label}")
    print(f"Output folder: {abs_output_dir}")
    for path in outputs:
        print(f"  {os.path.abspath(path)}  ({_format_size_mb(path)})")
    if sys.platform == "win32":
        print(f"\nOpen output folder: explorer {abs_output_dir}")
    if outputs:
        print(f"\nVerify: python validate_output.py {os.path.abspath(outputs[0])}")
    return 0


def cmd_h5_to_dat(h5_path: str, output_dir: str) -> int:
    outputs = export_run_h5(h5_path, output_dir)
    abs_output_dir = os.path.abspath(output_dir)
    print(f"\nSuccess: exported {len(outputs)} .dat file(s)")
    print(f"Output folder: {abs_output_dir}")
    for path in outputs:
        print(f"  {os.path.abspath(path)}")
    if sys.platform == "win32":
        print(f"\nOpen output folder: explorer {abs_output_dir}")
    return 0


def cmd_roundtrip(input_dir: str, work_dir: str, compression: Optional[str]) -> int:
    exported, results = run_roundtrip(input_dir, work_dir=work_dir, compression=compression)
    passed = sum(1 for _, result in results if result.ok)
    failed = len(results) - passed

    print(f"\nRound-trip: {passed} passed, {failed} failed ({len(exported)} files exported)")
    print(f"Work folder: {os.path.abspath(work_dir)}")
    for name, result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"  [{status}] {name} ({result.file_type}): {result.message}")

    return 0 if failed == 0 else 1


def run_interactive_flow(
    compression: Optional[str],
    *,
    default_dat_dir: str = "",
    default_h5_out: str = "output",
    default_h5_file: str = "",
    default_dat_out: str = "exported_dat",
    default_work_dir: str = "roundtrip_work",
) -> int:
    print_menu()
    choice = prompt_menu_choice()
    if choice == "4":
        print("Bye.")
        return 0
    if choice == "1":
        input_dir = prompt_path("Folder with .dat files", default=default_dat_dir)
        output_dir = prompt_path("Output folder for .h5 files", default=default_h5_out)
        return cmd_dat_to_h5(input_dir, output_dir, compression)
    if choice == "2":
        h5_path = prompt_path("HDF5 file (.h5)", default=default_h5_file)
        output_dir = prompt_path("Output folder for .dat files", default=default_dat_out)
        return cmd_h5_to_dat(h5_path, output_dir)
    if choice == "3":
        input_dir = prompt_path("Folder with .dat files", default=default_dat_dir)
        work_dir = prompt_path("Work folder for round-trip", default=default_work_dir)
        return cmd_roundtrip(input_dir, work_dir, compression)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert simulation .dat files to HDF5 and back; verify round-trip.",
    )
    subparsers = parser.add_subparsers(dest="command")

    dat_to_h5 = subparsers.add_parser("dat-to-h5", help="Convert .dat folder to HDF5")
    dat_to_h5.add_argument("input", help="Input folder with .dat files")
    dat_to_h5.add_argument("output", help="Output folder for .h5 files")
    dat_to_h5.add_argument(
        "--compression",
        choices=["gzip-1", "gzip", "gzip-9"],
        help="HDF5 compression",
    )
    dat_to_h5.add_argument(
        "--no-compression",
        action="store_true",
        help="No compression (largest files)",
    )

    h5_to_dat = subparsers.add_parser("h5-to-dat", help="Export HDF5 to .dat files")
    h5_to_dat.add_argument("h5_file", help="Input .h5 file (one RUN_ID)")
    h5_to_dat.add_argument("output", help="Output folder for .dat files")

    roundtrip = subparsers.add_parser(
        "roundtrip",
        help="Convert .dat → .h5 → .dat and compare",
    )
    roundtrip.add_argument("input", help="Input folder with .dat files")
    roundtrip.add_argument(
        "--work-dir",
        default="roundtrip_work",
        help="Temporary output folder (default: roundtrip_work)",
    )
    roundtrip.add_argument(
        "--compression",
        choices=["gzip-1", "gzip", "gzip-9"],
        help="HDF5 compression for intermediate .h5",
    )
    roundtrip.add_argument(
        "--no-compression",
        action="store_true",
        help="No compression for intermediate .h5",
    )

    return parser


def resolve_compression_arg(args: argparse.Namespace) -> tuple[Optional[str], bool]:
    if getattr(args, "no_compression", False) and getattr(args, "compression", None):
        raise ValueError("cannot use --no-compression with --compression")
    if getattr(args, "no_compression", False):
        return RECOMMENDED_COMPRESSION, True
    if getattr(args, "compression", None):
        return args.compression, True
    return RECOMMENDED_COMPRESSION, False


def interactive_main() -> int:
    try:
        compression = prompt_compression() if sys.stdin.isatty() else RECOMMENDED_COMPRESSION
        return run_interactive_flow(compression)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    if argv is None and len(sys.argv) == 1:
        return interactive_main()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        return interactive_main()

    try:
        compression, explicit_compression = resolve_compression_arg(args)
        if (
            args.command in {"dat-to-h5", "roundtrip"}
            and not explicit_compression
            and sys.stdin.isatty()
        ):
            compression = prompt_compression()
        if args.command == "dat-to-h5":
            return cmd_dat_to_h5(args.input, args.output, compression)
        if args.command == "h5-to-dat":
            return cmd_h5_to_dat(args.h5_file, args.output)
        if args.command == "roundtrip":
            return cmd_roundtrip(args.input, args.work_dir, compression)
        parser.error(f"unknown command {args.command!r}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
