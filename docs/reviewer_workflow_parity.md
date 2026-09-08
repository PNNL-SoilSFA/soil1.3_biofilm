# Historical-to-runner workflow parity

## Claim and boundary

The runner-based workflow preserves the *executable scientific core* of the historical notebook exports while making it non-interactive and independent of the original workstation paths:

- `biofilm-experiment-1.3.1` processes the three short-term replicates and produces both spatial-analysis branches.
- `biofilm-experiment-1.3.2` processes long-term replicate 7 only. It has no spatial-analysis branch.

The historical files remain the scientific record. Their exploratory figures, interactive display cells, and downstream statistical/figure-production cells are deliberately not reproduced as a CLI side effect. This is not a claim of byte-for-byte equality of figures or legacy output directories. The parity claim is instead the same raw-image transformations, sample/timepoint/outlier assignments, and temporal/spatial data calculations documented below.

## Implementation map

| Historical workflow | Runner implementation | Parity evidence |
| --- | --- | --- |
| `RawImageProcessing.py`: three channels in `w4SD RFP`, `w3SD GFP`, `w2SD DAPI` order; eight tiles; 4×2 stitch with 233-pixel horizontal overlap; max-z projection | `experiment_analysis.process_raw_experiment`, `_stitch_z_stacks`, `_simple_stitch` | Defaults and channel order are retained. `checks/test_scientific_parity.py` checks the legacy normalization and stitch equations; `checks/test_runner_workflows.py` processes all required tiles through both runners. |
| `RawImageProcessing.py`: rolling-ball radii `[7, 9, 5]`, minimum thresholds `[32, 48, 128]`, normalization floor 32 / scale 65535, sharp cutoff 0.01 | `process_raw_experiment` defaults, `_filter`, `_normalize` | The values are explicit function defaults. The shared tests exercise normalization and the real processing path. |
| `RawImageProcessing.py`: mask convention is zero-valued habitat, and legacy processed TIFF/CSV/pickle products are emitted | `process_raw_experiment`, `_load_arrays`, `SampleRecord` | The runner reads the same mask semantics and writes the same per-channel and `*_combined_processed` product names, under reviewer-selected results directories. |
| `PostAnalysis_spatial_sampling.py`: 1.3.1 temporal tables, outlier-excluded samples, 480-pixel windows, 100 trials, 0.125-pixel length/0.015625-pixel area | `load_experiment_1_3_1_metadata`, `temporal_tables`, `random_spatial_table`; 1.3.1 runner defaults | The metadata test reads both notebook sample lists and proves that `experiment_1.3.1_metadata.csv` exactly preserves times and the without-outlier membership. Omitting `--seed`, or using `--seed none`, requests a non-deterministic draw; an integer seed enables reproducibility. |
| `PostAnalysis_with_outlier.py`: 1.3.1 temporal tables and complete 500-pixel spatial/delta tables at steps 1, 10, and 100 | `temporal_tables`, `full_spatial_tables`; 1.3.1 runner defaults | The shared parity test exercises the temporal, sliding-window, density, and delta formulas. |
| Historical hard-coded dataset paths and manual file lists | `discover_experiment_1_3_1_replicates`, `discover_experiment_1_3_2_replicate`, `discover_raw_samples` | The runner checks the two observed published layouts separately and fails a mixed root before processing. It writes outside the downloaded input. |

## Layout contract exercised by tests

The tests create their files in temporary directories and delete them automatically. No downloaded data, image products, or test fixtures are committed.

| Experiment | Synthetic directory and filename convention | What is proved |
| --- | --- | --- |
| 1.3.1 | `Activity_1.3_rep1/Chitin/11.22.21_17/11.22.21_17_Chitin_1_w1SD BF_stitched_mask.tif` (and equivalent rep2/rep3 files) | Substrate-first discovery, all three replicate runners, raw processing, both spatial branches, random-window output with optional integer seeding, and outputs outside the download. |
| 1.3.2 | `Activity_1.3_rep7/10.19.22_11AM_TP8_Chitin/10.19.22_11AM_TP8_Chitin_1_w1SD BF_stitched_mask.tif` | Acquisition-first discovery from either the download root or rep7 itself, raw processing, legacy combined output naming, and outputs outside the download. |

Each synthetic sample includes the mask plus all 24 required raw TIFF files: three channels times eight tiles. The tile dimensions deliberately create a stitched width of 965 pixels, which is sufficient to exercise the 480- and 500-pixel spatial defaults.

## Repeatable verification

After `uv sync`, run:

```bash
uv run python checks/test_scientific_parity.py
uv run python checks/test_runner_workflows.py
uv run biofilm-experiment-1.3.1 --help
uv run biofilm-experiment-1.3.2 --help
uv run python -m compileall -q .
```

The tests provide code- and workflow-level evidence without the public download. A full reviewer-release validation additionally requires running each applicable command against its downloaded public package and comparing the generated data tables to the retained historical analysis outputs; the public inputs are intentionally not stored in this repository.
