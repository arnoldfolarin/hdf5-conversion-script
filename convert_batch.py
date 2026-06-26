import argparse
import os
import sys
from typing import Optional

from cli_prompts import prompt_compression, prompt_path
from hdf5_io import RECOMMENDED_COMPRESSION, describe_compression
from text_io import convert_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch convert simulation .dat files to one HDF5 per RUN_ID.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input folder containing .dat files (omit for interactive prompts)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output folder for .h5 files (omit for interactive prompts)",
    )
    parser.add_argument(
        "--compression",
        choices=["gzip-1", "gzip", "gzip-9"],
        help="HDF5 compression (skips compression menu)",
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="No compression, largest files (skips compression menu)",
    )
    parser.add_argument(
        "--batch-only",
        action="store_true",
        help="Skip direction menu; convert .dat to HDF5 immediately",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def resolve_compression(args: argparse.Namespace) -> tuple[Optional[str], bool]:
    """Return compression setting and whether it was set explicitly on the CLI."""
    if args.no_compression and args.compression:
        raise ValueError("cannot use --no-compression with --compression")
    if args.no_compression:
        return RECOMMENDED_COMPRESSION, True
    if args.compression:
        return args.compression, True
    return RECOMMENDED_COMPRESSION, False


def _format_size_mb(path: str) -> str:
    size_bytes = os.path.getsize(path)
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def print_success(
    outputs: list[str],
    output_dir: str,
    compression: Optional[str],
) -> None:
    label = describe_compression(compression)
    abs_output_dir = os.path.abspath(output_dir)
    abs_outputs = [os.path.abspath(path) for path in outputs]

    print(f"\nSuccess: converted {len(outputs)} file(s) with compression={label}")
    print("\nOutput folder on this computer:")
    print(f"  {abs_output_dir}")
    print("\nFiles written:")
    for path in abs_outputs:
        print(f"  {path}  ({_format_size_mb(path)})")
    if sys.platform == "win32":
        print("\nOpen output folder in File Explorer:")
        print(f"  explorer {abs_output_dir}")
    if abs_outputs:
        print(f"\nVerify: python validate_output.py {abs_outputs[0]}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        compression, explicit_compression = resolve_compression(args)
        interactive = sys.stdin.isatty() and not args.batch_only

        if args.input is None and args.output is None:
            if not interactive:
                print("Simulation batch converter — interactive mode")
                input_dir = prompt_path("Folder with .dat files")
                output_dir = prompt_path("Output folder for .h5 files", default="output")
            else:
                input_dir = ""
                output_dir = "output"
        elif args.input is None or args.output is None:
            parser.error("provide both input and output folders, or run with no arguments")
        else:
            input_dir = args.input
            output_dir = args.output

        if not explicit_compression and sys.stdin.isatty():
            compression = prompt_compression()

        if interactive:
            from interconvert import run_interactive_flow

            return run_interactive_flow(
                compression,
                default_dat_dir=input_dir,
                default_h5_out=output_dir,
            )

        outputs = convert_batch(input_dir, output_dir, compression=compression)
        print_success(outputs, output_dir, compression)
        return 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
