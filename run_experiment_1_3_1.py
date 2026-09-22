#!/usr/bin/env python3
"""Run Experiment 1.3.1: raw processing for three directories and both spatial analyses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_seed(value: str) -> int | None:
    """Parse an integer seed or the explicit non-deterministic ``none`` mode."""
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seed must be an integer or 'none'.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the Experiment 1.3.1 download containing Activity_1.3_rep1 through rep3.")
    parser.add_argument("--output-dir", type=Path, help="Optional exact results directory (overrides the unique default run directory).")
    parser.add_argument("--run-id", help="Optional readable name for the default results subdirectory; cannot be used with --output-dir.")
    parser.add_argument("--sampling-window-size", type=int, default=480, help="Outlier-excluded random-sampling window size (default: 480).")
    parser.add_argument("--full-window-size", type=int, default=500, help="Outlier-inclusive full spatial window size (default: 500).")
    parser.add_argument("--trials", type=int, default=100, help="Random spatial-sampling trials (default: 100).")
    parser.add_argument("--seed", type=parse_seed, metavar="INTEGER|none", default=None, help="Optional random seed for outlier-excluded sampling; omit it or use 'none' for a non-deterministic draw.")
    parser.add_argument("--pixel-size", type=float, default=0.125, help="Pixel side length (default: 0.125).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.sampling_window_size, args.full_window_size, args.trials, args.pixel_size) <= 0:
        raise SystemExit("Window sizes, --trials, and --pixel-size must be positive.")
    try:
        from experiment_analysis import (
            BiofilmInputError,
            discover_experiment_1_3_1_replicates,
            discover_raw_samples,
            EXPERIMENT_1_3_1_REP3_SKIPPED_ACQUISITIONS,
            full_spatial_tables,
            load_experiment_1_3_1_metadata,
            process_raw_experiment,
            random_spatial_table,
            resolve_reviewer_output_directory,
            temporal_tables,
            write_table,
            write_spatial_plot,
            write_temporal_plot,
        )
    except ModuleNotFoundError as exc:
        print(f"Missing analysis dependency ({exc.name}). Run 'uv sync' first.", file=sys.stderr)
        return 3

    try:
        output = resolve_reviewer_output_directory(args.output_dir, experiment="experiment_1.3.1", run_id=args.run_id)
        metadata = load_experiment_1_3_1_metadata()
        raw_directories = discover_experiment_1_3_1_replicates(args.data_dir)
        # Check every replicate's file set before creating any result directory.
        # In particular, this prevents a missing tile in a later replicate from
        # leaving the earlier replicates partially processed.
        for index, raw_directory in enumerate(raw_directories, start=1):
            excluded_acquisitions = EXPERIMENT_1_3_1_REP3_SKIPPED_ACQUISITIONS if index == 3 else ()
            discover_raw_samples(
                raw_directory,
                replicate=f"rep{index}",
                metadata=metadata,
                excluded_acquisition_names=excluded_acquisitions,
            )

        records = []
        for index, raw_directory in enumerate(raw_directories, start=1):
            excluded_acquisitions = EXPERIMENT_1_3_1_REP3_SKIPPED_ACQUISITIONS if index == 3 else ()
            records.extend(
                process_raw_experiment(
                    raw_directory,
                    output / "processed" / f"rep{index}",
                    replicate=f"rep{index}",
                    metadata=metadata,
                    excluded_acquisition_names=excluded_acquisitions,
                )
            )
        pixel_area = args.pixel_size**2
        density, volume = temporal_tables(records, pixel_area=pixel_area)
        without_outliers = [record for record in records if record.include_without_outlier]
        if not without_outliers:
            raise BiofilmInputError("No rows are marked include_without_outlier=true for PostAnalysis_spatial_sampling.")
        sampling_volume, sampling_density = random_spatial_table(without_outliers, pixel_area=pixel_area, pixel_size=args.pixel_size, window_size=args.sampling_window_size, trials=args.trials, seed=args.seed)
        full_volume, full_density, delta_volume, delta_density = full_spatial_tables(records, pixel_area=pixel_area, pixel_size=args.pixel_size, window_size=args.full_window_size)
        without_output = output / "PostAnalysis_spatial_sampling"
        with_output = output / "PostAnalysis_with_outlier"
        write_table(density, without_output / "pop_densities_wo_outliers")
        write_table(volume, without_output / "pop_volumes_wo_outliers")
        write_table(sampling_volume, without_output / "spatial" / f"pop_vol_1d_df_{args.sampling_window_size}_wo_outliers")
        write_table(sampling_density, without_output / "spatial" / f"pop_den_1d_df_{args.sampling_window_size}_wo_outliers")
        write_temporal_plot(volume, without_output / "plots" / "temporal_biomass.png", title="Experiment 1.3.1: temporal biomass without outliers")
        write_spatial_plot(sampling_volume, without_output / "plots" / "spatial_biomass.png", title="Experiment 1.3.1: random spatial sampling without outliers")
        write_table(density, with_output / "pop_densities")
        write_table(volume, with_output / "pop_volumes")
        write_table(full_volume, with_output / "spatial" / f"pop_vol_1d_df_{args.full_window_size}")
        write_table(full_density, with_output / "spatial" / f"pop_den_1d_df_{args.full_window_size}")
        write_table(delta_volume, with_output / "spatial" / f"delta_pop_vol_1d_df_{args.full_window_size}")
        write_table(delta_density, with_output / "spatial" / f"delta_pop_den_1d_df_{args.full_window_size}")
        write_temporal_plot(volume, with_output / "plots" / "temporal_biomass.png", title="Experiment 1.3.1: temporal biomass including outliers")
        write_spatial_plot(full_volume, with_output / "plots" / "spatial_biomass.png", title="Experiment 1.3.1: full spatial biomass including outliers")
        print(f"Experiment 1.3.1 complete: {len(records)} raw samples and both spatial-analysis datasets written to {output}")
        return 0
    except ModuleNotFoundError as exc:
        print(f"Missing analysis dependency ({exc.name}). Run 'uv sync' first.", file=sys.stderr)
        return 3
    except BiofilmInputError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
