#!/usr/bin/env python3
"""
Script to extract features from all UZF files in dataset_train directory.

This script runs "python -m circuit_detector features" for each UZF file found
in the dataset_train directory and saves the output to corresponding text files.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def find_uzf_files(dataset_dir):
    """
    Find all UZF files in the dataset directory recursively.

    Args:
        dataset_dir: Path to the dataset directory

    Returns:
        List of Path objects for all UZF files found
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return []

    # Find all .uzf files recursively
    uzf_files = list(dataset_path.rglob("*.uzf"))

    print(f"Found {len(uzf_files)} UZF files in {dataset_dir}")
    return uzf_files


def extract_features_to_file(uzf_file, dataset_dir, output_dir, verbose=False):
    """
    Extract features from a single UZF file and save to text file.

    Args:
        uzf_file: Path to the UZF file
        dataset_dir: Base dataset directory path
        output_dir: Base output directory to save the text file
        verbose: Whether to use verbose output

    Returns:
        True if successful, False otherwise
    """
    try:
        # Calculate relative path from dataset_dir to uzf_file
        relative_path = uzf_file.relative_to(dataset_dir)

        # Create output path maintaining directory structure
        output_relative_dir = relative_path.parent
        output_filename = uzf_file.stem + "_features.txt"

        # Create full output path
        output_subdir = output_dir / output_relative_dir
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_path = output_subdir / output_filename

        # Construct the command
        cmd = [
            sys.executable, "-m", "circuit_detector", "features",
            "--uzf-file", str(uzf_file)
        ]

        if verbose:
            cmd.append("--verbose")

        # Run the command and capture output
        print(f"Processing: {str(relative_path)} -> {str(output_relative_dir / output_filename)}")

        with open(output_path, "w", encoding="utf-8") as output_file:
            result = subprocess.run(
                cmd,
                stdout=output_file,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=Path.cwd()
            )

        if result.returncode != 0:
            print(f"Error processing {uzf_file.name}: {result.stderr}")
            return False

        return True

    except Exception as e:
        print(f"Exception processing {uzf_file.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract features from all UZF files in dataset")
    parser.add_argument("--dataset-dir", default="dataset_train",
                        help="Path to dataset directory containing UZF files (default: dataset_train)")
    parser.add_argument("--output-dir", default="feature_reports",
                        help="Directory to save feature report files (default: feature_reports)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Generate verbose feature reports")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what files would be processed without actually processing them")

    args = parser.parse_args()

    # Setup directories
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    # Find all UZF files
    uzf_files = find_uzf_files(dataset_dir)

    if not uzf_files:
        print("No UZF files found. Exiting.")
        return 1

    if args.dry_run:
        print(f"\nDry run - would process these {len(uzf_files)} files:")
        for uzf_file in uzf_files:
            try:
                relative_path = uzf_file.relative_to(dataset_dir)
                output_relative_dir = relative_path.parent
                output_filename = uzf_file.stem + "_features.txt"
                output_path = output_relative_dir / output_filename
                print(f"  {relative_path.as_posix()} -> {output_path.as_posix()}")
            except Exception as e:
                print(f"  Error showing {uzf_file.name}: {e}")
        return 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Process each UZF file
    successful = 0
    failed = 0

    for i, uzf_file in enumerate(uzf_files, 1):
        print(f"[{i}/{len(uzf_files)}] ", end="")

        if extract_features_to_file(uzf_file, dataset_dir, output_dir, args.verbose):
            successful += 1
        else:
            failed += 1

    # Summary
    print("\nFeature extraction complete:")
    print(f"  Successfully processed: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
