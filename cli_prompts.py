from typing import Optional

from hdf5_io import RECOMMENDED_COMPRESSION


def prompt_path(label: str, default: str = "") -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{label}{suffix}: ").strip()
        if not value:
            value = default
        if value:
            return value
        print("Please enter a path.")


def prompt_compression() -> Optional[str]:
    print("\nCompression (1 = largest file, 4 = smallest file):")
    print("  1) No compression  — largest files, fastest I/O")
    print("  2) Light           — gzip level 1 + shuffle")
    print("  3) Balanced        — gzip level 4 + shuffle")
    print("  4) Maximum         — gzip level 9 + shuffle (smallest files)")
    print("  (columns under 4 KB are always stored uncompressed)")
    while True:
        choice = input("Enter 1-4 [1]: ").strip() or "1"
        if choice == "1":
            return RECOMMENDED_COMPRESSION
        if choice == "2":
            return "gzip-1"
        if choice == "3":
            return "gzip"
        if choice == "4":
            return "gzip-9"
        print("Please enter 1, 2, 3, or 4.")


def print_menu() -> None:
    print("\nWhat do you want to do?")
    print("  1) .dat folder  ->  HDF5 (.h5 per RUN_ID)")
    print("  2) HDF5 file    ->  .dat folder")
    print("  3) Round-trip test (.dat -> .h5 -> .dat, then compare)")
    print("  4) Quit")


def prompt_menu_choice() -> str:
    while True:
        choice = input("Enter 1-4 [1]: ").strip() or "1"
        if choice in {"1", "2", "3", "4"}:
            return choice
        print("Please enter 1, 2, 3, or 4.")
