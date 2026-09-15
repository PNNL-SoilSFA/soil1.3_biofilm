"""Reusable input handling and core calculations for the two biofilm papers.

The legacy ``PostAnalysis_*.py`` files are notebook exports.  This module is
the non-interactive, path-independent implementation used by the command-line
runners. It discovers the files included in the published data download.
"""

from __future__ import annotations

import csv
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Collection, Iterable, Sequence

import numpy as np
import pandas as pd
from skimage import filters, io, restoration
import tifffile


SPECIES = ("Variovorax", "Sphingopyxis", "Rhodococcus")
SUBSTRATE_NAMES = {
    "Nag": "N-acetylglucosamine",
    "Chito5": "Chitopentose",
    "Chitin": "Chitin",
}


class BiofilmInputError(ValueError):
    """Raised when a reviewer-facing input directory is incomplete or invalid."""


@dataclass(frozen=True)
class SampleRecord:
    replicate: str
    sample_id: str
    substrate: str
    time_hours: float
    mask_file: Path
    processed_file: Path
    include_without_outlier: bool


MASK_NAME = re.compile(
    r"^(?P<sample_id>.+)_(?P<substrate>Nag|Chito5|Chitin)_1_w1SD_BF_stitched_mask\.tif$",
    re.IGNORECASE,
)
EXPERIMENT_1_3_2_REPLICATE_NAME = re.compile(r"Activity_1\.3_rep7(?:_.*)?", re.IGNORECASE)
EXPERIMENT_1_3_1_REPLICATE_NAME = re.compile(r"Activity_1\.3_rep(?P<replicate>[1-3])(?:_.*)?", re.IGNORECASE)
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*$")
EXPERIMENT_1_3_1_REP3_SKIPPED_ACQUISITIONS = frozenset({"02.09.22_11AM_Chito5"})


def _canonical_substrate(value: str) -> str:
    return {"nag": "Nag", "chito5": "Chito5", "chitin": "Chitin"}[value.lower()]


def resolve_reviewer_output_directory(
    output_directory: Path | None,
    *,
    experiment: str,
    run_id: str | None,
) -> Path:
    """Return an explicit output directory without overwriting default runs.

    A caller-supplied ``--output-dir`` is used exactly as requested. Otherwise
    each run is placed in a timestamped child directory under ``results``;
    ``--run-id`` supplies a readable child-directory name instead.
    """
    if output_directory is not None:
        if run_id is not None:
            raise BiofilmInputError("--run-id cannot be combined with --output-dir; include the run name in --output-dir instead.")
        return output_directory.expanduser().resolve()
    if run_id is not None:
        if not RUN_ID.fullmatch(run_id):
            raise BiofilmInputError("--run-id must start with a letter or number and contain only letters, numbers, '.', '_', or '-'.")
        label = run_id
    else:
        label = datetime.now().astimezone().strftime("run-%Y%m%dT%H%M%S%f%z")
    output = (Path("results") / experiment / label).expanduser().resolve()
    suffix = 2
    while output.exists():
        output = output.with_name(f"{label}-{suffix}")
        suffix += 1
    return output


def _sample_layout(mask_file: Path, *, sample_id: str, substrate: str) -> str:
    """Classify the two published raw-data layouts from a mask's location."""
    if mask_file.parent.parent.name.lower() == substrate.lower():
        return "substrate-first"
    if mask_file.parent.name.lower() == f"{sample_id}_{substrate}".lower():
        return "acquisition-first"
    raise BiofilmInputError(
        "Mask is not in a supported published layout: "
        f"{mask_file}. Expected either <substrate>/<sample>/ or <sample>_<substrate>/ ."
    )


def _channel_tile_name(base_name: str, channel: str, index: int) -> str:
    """Return the underscore-separated name used by published channel TIFFs."""
    return f"{base_name}_{channel.replace(' ', '_')}_s{index}.tif".lower()


def discover_raw_samples(
    data_directory: Path,
    *,
    replicate: str,
    metadata: dict[tuple[str, str, str], tuple[float, bool]] | None = None,
    patch_number: int = 8,
    excluded_acquisition_names: Collection[str] = (),
) -> list[tuple[SampleRecord, tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]]:
    """Discover masks and the 3 × 8 TIFF tiles from a published data directory.

    ``excluded_acquisition_names`` is used only for explicit historical
    per-acquisition skips made by a runner.
    """
    data_directory = data_directory.expanduser().resolve()
    if not data_directory.is_dir():
        raise BiofilmInputError(f"Data directory does not exist: {data_directory}")
    discovered = []
    layouts: set[str] = set()
    for mask_file in sorted(data_directory.rglob("*")):
        if not mask_file.is_file():
            continue
        # macOS resource-fork sidecars can accompany the published TIFFs.
        if mask_file.name.startswith("._"):
            continue
        match = MASK_NAME.match(mask_file.name)
        if not match:
            continue
        sample_id = match["sample_id"]
        substrate = _canonical_substrate(match["substrate"])
        acquisition_name = f"{sample_id}_{substrate}"
        if acquisition_name in excluded_acquisition_names:
            continue
        layouts.add(_sample_layout(mask_file, sample_id=sample_id, substrate=substrate))
        time_hours, include_without_outlier = (metadata or {}).get((replicate, sample_id, substrate), (float("nan"), True))
        base_name = f"{sample_id}_{substrate}_1"
        file_index = {
            path.name.lower(): path
            for path in mask_file.parent.iterdir()
            if path.is_file() and not path.name.startswith("._")
        }
        channel_tiles = []
        for channel in ("w4SD RFP", "w3SD GFP", "w2SD DAPI"):
            tiles = tuple(file_index.get(_channel_tile_name(base_name, channel, index)) for index in range(1, patch_number + 1))
            if any(tile is None for tile in tiles):
                missing = [str(index + 1) for index, tile in enumerate(tiles) if tile is None]
                raise BiofilmInputError(f"Missing {channel} tile(s) {', '.join(missing)} for {mask_file}.")
            channel_tiles.append(tiles)
        discovered.append(
            (
                SampleRecord(replicate, sample_id, substrate, time_hours, mask_file, Path(), include_without_outlier),
                channel_tiles[0], channel_tiles[1], channel_tiles[2],
            )
        )
    if not discovered:
        raise BiofilmInputError(f"No BF stitched masks were found under {data_directory}.")
    if len(layouts) != 1:
        raise BiofilmInputError(
            f"Mixed raw-data layouts found under {data_directory}: {', '.join(sorted(layouts))}."
        )
    return discovered


def load_experiment_1_3_1_metadata() -> dict[tuple[str, str, str], tuple[float, bool]]:
    """Load the fixed sample/time/outlier assignments for Experiment 1.3.1."""
    metadata_file = Path(__file__).with_name("experiment_1.3.1_metadata.csv")
    with metadata_file.open(newline="", encoding="utf-8") as handle:
        return {
            (row["replicate"], row["sample_id"], row["substrate"]): (float(row["time_hours"]), row["include_without_outlier"].lower() == "true")
            for row in csv.DictReader(handle)
        }


def discover_experiment_1_3_2_replicate(data_directory: Path) -> Path:
    """Find Experiment 1.3.2's rep7 directory at or immediately below a download root."""
    data_directory = data_directory.expanduser().resolve()
    if not data_directory.is_dir():
        raise BiofilmInputError(f"Data directory does not exist: {data_directory}")
    if EXPERIMENT_1_3_2_REPLICATE_NAME.fullmatch(data_directory.name):
        return data_directory
    short_term_replicates = sorted(
        path.name
        for path in data_directory.iterdir()
        if path.is_dir() and EXPERIMENT_1_3_1_REPLICATE_NAME.fullmatch(path.name)
    )
    if short_term_replicates:
        raise BiofilmInputError(
            "Experiment 1.3.2 input must not mix layouts. "
            f"Found Experiment 1.3.1 directory/directories: {', '.join(short_term_replicates)}."
        )
    candidates = sorted(
        path for path in data_directory.iterdir()
        if path.is_dir() and EXPERIMENT_1_3_2_REPLICATE_NAME.fullmatch(path.name)
    )
    if len(candidates) != 1:
        raise BiofilmInputError(
            f"Expected one Activity_1.3_rep7 directory at or immediately below {data_directory}; found {len(candidates)}."
        )
    return candidates[0]


def discover_experiment_1_3_1_replicates(data_directory: Path) -> list[Path]:
    """Find Experiment 1.3.1's three conventional replicate directories in a download."""
    data_directory = data_directory.expanduser().resolve()
    if not data_directory.is_dir():
        raise BiofilmInputError(f"Data directory does not exist: {data_directory}")
    long_term_replicates = sorted(
        path.name
        for path in data_directory.iterdir()
        if path.is_dir() and EXPERIMENT_1_3_2_REPLICATE_NAME.fullmatch(path.name)
    )
    if long_term_replicates:
        raise BiofilmInputError(
            "Experiment 1.3.1 input must not mix layouts. "
            f"Found Experiment 1.3.2 directory/directories: {', '.join(long_term_replicates)}."
        )
    matches: list[Path] = []
    for replicate in range(1, 4):
        candidates = sorted(
            path for path in data_directory.iterdir()
            if path.is_dir()
            and (match := EXPERIMENT_1_3_1_REPLICATE_NAME.fullmatch(path.name))
            and int(match["replicate"]) == replicate
        )
        if len(candidates) != 1:
            raise BiofilmInputError(f"Expected one Activity_1.3_rep{replicate} directory beneath {data_directory}; found {len(candidates)}.")
        matches.append(candidates[0])
    return matches


def _normalize(image: np.ndarray, *, min_range: float = 32, scale_to: int = 65535) -> np.ndarray:
    lower, upper = np.min(image), np.max(image)
    return np.asarray((image - lower) / max(upper - lower, min_range) * scale_to, dtype=np.uint16)


def _filter(image: np.ndarray, *, minimum_threshold: float) -> np.ndarray:
    try:
        threshold = max(filters.threshold_otsu(image), minimum_threshold)
    except ValueError:
        threshold = image.min() + minimum_threshold
    return np.where(image > threshold, image, 0)


def _simple_stitch(images: Sequence[np.ndarray], *, horizontal_margin: int, vertical_margin: int) -> np.ndarray:
    if len(images) != 8:
        raise BiofilmInputError(f"Expected eight image tiles, found {len(images)}.")
    upper = np.hstack([image[:, :-horizontal_margin] if index < 3 else image for index, image in enumerate(images[:4])])
    lower = np.hstack([image[:, :-horizontal_margin] if index < 3 else image for index, image in enumerate(images[4:])])
    return np.vstack((upper[vertical_margin:, :], lower[vertical_margin:, :]))


def _stitch_z_stacks(tile_files: Sequence[Path], *, horizontal_margin: int, vertical_margin: int) -> np.ndarray:
    for tile_file in tile_files:
        if not tile_file.is_file():
            raise BiofilmInputError(f"Missing raw TIFF tile: {tile_file}")
    stacks = [np.asarray(io.imread(tile_file)) for tile_file in tile_files]
    if any(stack.ndim != 3 for stack in stacks):
        raise BiofilmInputError("Each raw TIFF tile must be a three-dimensional z-stack.")
    if len({stack.shape for stack in stacks}) != 1:
        raise BiofilmInputError("All eight TIFF tiles for a channel must have matching dimensions.")
    return np.asarray([
        _simple_stitch([stack[z_index] for stack in stacks], horizontal_margin=horizontal_margin, vertical_margin=vertical_margin)
        for z_index in range(stacks[0].shape[0])
    ])


def process_raw_experiment(
    raw_directory: Path,
    output_directory: Path,
    *,
    replicate: str,
    metadata: dict[tuple[str, str, str], tuple[float, bool]] | None = None,
    patch_number: int = 8,
    horizontal_margin: int = 233,
    vertical_margin: int = 0,
    rolling_ball_sizes: Sequence[int] = (7, 9, 5),
    minimum_thresholds: Sequence[int] = (32, 48, 128),
    sharpening_threshold: float = 0.01,
    excluded_acquisition_names: Collection[str] = (),
) -> list[SampleRecord]:
    """Execute the non-interactive core of ``RawImageProcessing.py``.

    Results use the legacy processed-image filenames; the input data is found
    directly from the published download's TIFF names.
    """
    if len(rolling_ball_sizes) != 3 or len(minimum_thresholds) != 3:
        raise BiofilmInputError("Exactly three per-channel background and threshold settings are required.")
    raw_records = discover_raw_samples(
        raw_directory,
        replicate=replicate,
        metadata=metadata,
        patch_number=patch_number,
        excluded_acquisition_names=excluded_acquisition_names,
    )
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    processed_records: list[SampleRecord] = []
    for record, rfp_tiles, gfp_tiles, dapi_tiles in raw_records:
        if not record.mask_file.is_file():
            raise BiofilmInputError(f"Missing mask for {record.sample_id}: {record.mask_file}")
        raw_mask = np.asarray(io.imread(record.mask_file))
        if raw_mask.ndim != 2:
            raise BiofilmInputError(f"Raw mask must be two-dimensional: {record.mask_file}")
        mask = (raw_mask == 0).astype(np.uint8)
        channels: list[np.ndarray] = []
        sharp_channels: list[np.ndarray] = []
        base_name = f"{record.sample_id}_{record.substrate}_1"
        for channel_index, (channel_name, tiles) in enumerate(zip(("w4SD RFP", "w3SD GFP", "w2SD DAPI"), (rfp_tiles, gfp_tiles, dapi_tiles), strict=True)):
            projected = np.max(_stitch_z_stacks(tiles, horizontal_margin=horizontal_margin, vertical_margin=vertical_margin), axis=0)
            if projected.shape != mask.shape:
                raise BiofilmInputError(f"Stitched image and mask dimensions differ for {record.sample_id}.")
            masked = np.where(mask, projected - restoration.rolling_ball(projected, radius=rolling_ball_sizes[channel_index]), 0)
            filtered = _filter(masked, minimum_threshold=minimum_thresholds[channel_index])
            normalized = _normalize(filtered)
            sharp = np.where(normalized > int(normalized.max() * sharpening_threshold), normalized.max(), 0).astype(np.uint16)
            tifffile.imwrite(output_directory / f"{base_name}_{channel_name}_masked.TIF", masked)
            tifffile.imwrite(output_directory / f"{base_name}_{channel_name}_processed.TIF", normalized)
            tifffile.imwrite(output_directory / f"{base_name}_{channel_name}_sharp.TIF", sharp)
            np.savetxt(output_directory / f"{base_name}_{channel_name}_processed.csv", normalized, delimiter=",")
            channels.append(normalized)
            sharp_channels.append(sharp)
        combined = np.stack(channels, axis=-1)
        combined_sharp = np.stack(sharp_channels, axis=-1)
        combined_tiff = output_directory / f"{base_name}_combined_processed.TIF"
        processed_pickle = output_directory / f"{base_name}_combined_processed.pkl"
        tifffile.imwrite(combined_tiff, combined)
        tifffile.imwrite(output_directory / f"{base_name}_combined_processed_sharp.TIF", combined_sharp)
        with processed_pickle.open("wb") as handle:
            pickle.dump(combined, handle)
        write_raw_preview(combined, output_directory / f"{base_name}_preview.png")
        processed_records.append(
            SampleRecord(
                replicate=record.replicate,
                sample_id=record.sample_id,
                substrate=record.substrate,
                time_hours=record.time_hours,
                mask_file=record.mask_file,
                processed_file=processed_pickle,
                include_without_outlier=record.include_without_outlier,
            )
        )
    return processed_records


def _plotting():
    """Import a non-interactive plotting backend only when figures are requested."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_raw_preview(combined_image: np.ndarray, path: Path) -> None:
    """Write a viewable RGB image and its width-wise population-density profile."""
    plt = _plotting()
    path.parent.mkdir(parents=True, exist_ok=True)
    image_max = combined_image.max()
    display = combined_image / image_max if image_max else combined_image
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(display)
    axes[0].set_title("Combined processed image")
    axes[0].set_axis_off()
    profile = _density(np.count_nonzero(combined_image, axis=0))
    for index, species in enumerate(SPECIES):
        axes[1].plot(profile[:, index], label=species)
    axes[1].set_xlabel("Image x-coordinate (pixels)")
    axes[1].set_ylabel("Relative abundance")
    axes[1].set_title("Population-density profile")
    axes[1].legend()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _load_arrays(record: SampleRecord) -> tuple[np.ndarray, np.ndarray]:
    for file_path, label in ((record.mask_file, "mask"), (record.processed_file, "processed image")):
        if not file_path.is_file():
            raise BiofilmInputError(
                f"Missing {label} for {record.replicate}/{record.sample_id}: {file_path}"
            )
    mask = np.asarray(io.imread(record.mask_file))
    with record.processed_file.open("rb") as handle:
        rgb = np.asarray(pickle.load(handle))
    if mask.ndim != 2:
        raise BiofilmInputError(f"Mask must be two-dimensional: {record.mask_file}")
    if rgb.ndim != 3 or rgb.shape[-1] != len(SPECIES):
        raise BiofilmInputError(
            f"Processed image must have shape (height, width, {len(SPECIES)}): {record.processed_file}"
        )
    if mask.shape != rgb.shape[:2]:
        raise BiofilmInputError(
            f"Mask/image dimensions differ for {record.replicate}/{record.sample_id}."
        )
    # This matches the legacy notebooks: zero pixels are the habitat mask.
    return (mask == 0).astype(np.uint8), rgb


def _identity(record: SampleRecord) -> dict[str, object]:
    return {
        "Rep": record.replicate,
        "Growth Substrate": SUBSTRATE_NAMES.get(record.substrate, record.substrate),
        "Time": record.time_hours,
    }


def _sliding_sum(image: np.ndarray, window_size: int) -> np.ndarray:
    values = np.count_nonzero(image, axis=0)
    if len(values) <= window_size:
        raise BiofilmInputError(
            f"Image width ({len(values)}) must exceed window size ({window_size})."
        )
    # Retain the original notebook's left-edge window convention.
    return np.asarray([values[i : i + window_size].sum(axis=0) for i in range(len(values) - window_size)])


def _density(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.full_like(values, np.nan, dtype=float), where=totals != 0)


def temporal_tables(records: Iterable[SampleRecord], *, pixel_area: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return population-density and volume tables for discovered image records."""
    density_rows: list[dict[str, object]] = []
    volume_rows: list[dict[str, object]] = []
    for record in records:
        mask, rgb = _load_arrays(record)
        volumes = rgb.sum(axis=(0, 1)) * pixel_area
        total = volumes.sum()
        identity = _identity(record)
        volume_rows.append({**identity, **dict(zip(SPECIES, volumes)), "Volume": mask.sum() * pixel_area})
        density_rows.append(
            {
                **identity,
                **dict(zip(SPECIES, np.divide(volumes, total, out=np.full(3, np.nan), where=total != 0))),
                "Volume": mask.sum() * pixel_area,
            }
        )
    density = pd.DataFrame(density_rows)
    volume = pd.DataFrame(volume_rows)
    volume["Total"] = volume[list(SPECIES)].sum(axis=1)
    for species in SPECIES:
        volume[f"{species} / Volume"] = volume[species] / volume["Volume"]
    return density, volume


def random_spatial_table(
    records: Iterable[SampleRecord], *, pixel_area: float, pixel_size: float, window_size: int, trials: int, seed: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce Experiment 1.3.1's randomly sampled spatial volume and density data."""
    rng = np.random.default_rng(seed)
    volume_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    for record in records:
        mask, rgb = _load_arrays(record)
        mask_volume = _sliding_sum(mask, window_size) * pixel_area
        population_volume = _sliding_sum(rgb, window_size) * pixel_area
        population_density = _density(_sliding_sum(rgb, window_size))
        sample_count = len(mask_volume) // window_size
        if sample_count == 0:
            raise BiofilmInputError(f"No non-overlapping sampling positions for {record.sample_id}.")
        for trial in range(trials):
            for position in rng.choice(len(mask_volume), size=sample_count, replace=False):
                identity = _identity(record)
                common = {**identity, "Radius": position * pixel_size, "Volume": mask_volume[position], "window_position": position, "Trial": trial}
                volume_rows.append({**common, **dict(zip(SPECIES, population_volume[position]))})
                density_rows.append({**common, **dict(zip(SPECIES, population_density[position]))})
    return pd.DataFrame(volume_rows), pd.DataFrame(density_rows)


def full_spatial_tables(
    records: Iterable[SampleRecord], *, pixel_area: float, pixel_size: float, window_size: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce Experiment 1.3.1's full spatial and delta tables at legacy step sizes."""
    volume_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    delta_volume_rows: list[dict[str, object]] = []
    delta_density_rows: list[dict[str, object]] = []
    for record in records:
        mask, rgb = _load_arrays(record)
        mask_volume = _sliding_sum(mask, window_size) * pixel_area
        population_volume = _sliding_sum(rgb, window_size) * pixel_area
        population_density = _density(_sliding_sum(rgb, window_size))
        identity = _identity(record)
        for position in range(len(mask_volume)):
            common = {**identity, "Radius": position * pixel_size, "Volume": mask_volume[position]}
            volume_rows.append({**common, **dict(zip(SPECIES, population_volume[position]))})
            density_rows.append({**common, **dict(zip(SPECIES, population_density[position]))})
            for step in (1, 10, 100):
                if position >= step:
                    delta_common = {**common, "delta": step, "Volume": mask_volume[position] - mask_volume[position - step]}
                    delta_volume_rows.append({**delta_common, **dict(zip(SPECIES, population_volume[position] - population_volume[position - step]))})
                    delta_density_rows.append({**delta_common, **dict(zip(SPECIES, population_density[position] - population_density[position - step]))})
    return (
        pd.DataFrame(volume_rows),
        pd.DataFrame(density_rows),
        pd.DataFrame(delta_volume_rows),
        pd.DataFrame(delta_density_rows),
    )


def write_table(table: pd.DataFrame, path: Path) -> None:
    """Write a pickle and a transparent TSV counterpart for every output table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(path.with_suffix(".pkl"))
    table.to_csv(path.with_suffix(".tsv"), sep="\t", index=False)


def write_temporal_plot(volume_table: pd.DataFrame, path: Path, *, title: str) -> None:
    """Write a static temporal biomass summary suitable for opening without Jupyter."""
    plt = _plotting()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for substrate, group in volume_table.groupby("Growth Substrate", sort=True):
        # ``groupby(sort=True)`` preserves the legacy ascending time order.
        # Avoiding a redundant DataFrame.sort_values call also keeps pandas'
        # static type checker from treating the result as a Series.
        summary = group.groupby("Time", as_index=False, sort=True)["Total"].mean()
        axis.plot(summary["Time"], summary["Total"], marker="o", label=substrate)
    axis.set_title(title)
    axis.set_xlabel("Time (hours)")
    axis.set_ylabel("Mean total biomass")
    axis.legend(title="Growth substrate")
    figure.savefig(path, dpi=200)
    plt.close(figure)


def write_spatial_plot(spatial_table: pd.DataFrame, path: Path, *, title: str) -> None:
    """Write a static spatial biomass summary for either Experiment 1.3.1 branch."""
    plt = _plotting()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = spatial_table.copy()
    table["Total"] = table[list(SPECIES)].sum(axis=1)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for substrate, group in table.groupby("Growth Substrate", sort=True):
        # ``groupby(sort=True)`` preserves the legacy ascending radius order.
        summary = group.groupby("Radius", as_index=False, sort=True)["Total"].mean()
        axis.plot(summary["Radius"], summary["Total"], label=substrate)
    axis.set_title(title)
    axis.set_xlabel("Position along sampling axis (physical units)")
    axis.set_ylabel("Mean total biomass in window")
    axis.legend(title="Growth substrate")
    figure.savefig(path, dpi=200)
    plt.close(figure)
