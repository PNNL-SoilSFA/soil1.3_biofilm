#!/usr/bin/env python3
"""Run Experiment 1.3.2's raw image-processing workflow (Activity_1.3_rep7)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Experiment 1.3.2 download root or its Activity_1.3_rep7 directory.",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional exact results directory (overrides the unique default run directory).")
    parser.add_argument("--run-id", help="Optional readable name for the default results subdirectory; cannot be used with --output-dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from experiment_analysis import BiofilmInputError, discover_experiment_1_3_2_replicate, process_raw_experiment, resolve_reviewer_output_directory
    except ModuleNotFoundError as exc:
        print(f"Missing analysis dependency ({exc.name}). Run 'uv sync' first.", file=sys.stderr)
        return 3

    try:
        output = resolve_reviewer_output_directory(args.output_dir, experiment="experiment_1.3.2", run_id=args.run_id)
        raw_directory = discover_experiment_1_3_2_replicate(args.data_dir)
        records = process_raw_experiment(raw_directory, output, replicate="rep7")
        print(f"Experiment 1.3.2 complete: {len(records)} raw samples written to {output}")
        return 0
    except ModuleNotFoundError as exc:
        print(f"Missing analysis dependency ({exc.name}). Run 'uv sync' first.", file=sys.stderr)
        return 3
    except BiofilmInputError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
