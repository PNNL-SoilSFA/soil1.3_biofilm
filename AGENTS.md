# SoilChip reviewer-workflow guide

## Purpose

This repository supports reproducible analysis of a SoilChip incubation microbiology experiment. Reviewers download the public data package, then run the supplied command-line workflow to reproduce the analysis outputs for two distinct experiments:

| Experiment | Incubation | Source data                                                       | Runner                                                 |
| ---------- | ---------- | ----------------------------------------------------------------- | ------------------------------------------------------ |
| 1.3.1      | Short-term | `Activity_1.3_rep1`, `Activity_1.3_rep2`, and `Activity_1.3_rep3` | `run_experiment_1_3_1.py` / `biofilm-experiment-1.3.1` |
| 1.3.2      | Long-term  | `Activity_1.3_rep7`                                               | `run_experiment_1_3_2.py` / `biofilm-experiment-1.3.2` |

The downloaded data are inputs, not repository content. Do not add raw, derived, or reviewer-downloaded data files to version control.

## Scientific and implementation contract

- The reviewer runners must implement the same science as the historical notebook exports: `RawImageProcessing.py`, `PostAnalysis_spatial_sampling.py`, and `PostAnalysis_with_outlier.py`.
- Preserve those historical scripts as the record of the original analysis. Make reproducibility improvements in the runner modules or `experiment_analysis.py`; do not silently rewrite the historical scripts.
- `experiment_analysis.py` is the shared non-interactive implementation used by both runners.
- Experiment 1.3.1 runs raw processing for all three short-term replicates, then both spatial-analysis branches. Experiment 1.3.2 currently runs raw image processing for long-term replicate 7 only; do not imply that it has a spatial-analysis branch.
- Keep the 1.3.1 metadata assignments in `experiment_1.3.1_metadata.csv` aligned with the original post-analysis notebooks. Changes to sample, timepoint, or outlier inclusion must be traceable and documented.

## Data layouts are intentionally different

- **Experiment 1.3.1:** the download root contains `Activity_1.3_rep1` through `Activity_1.3_rep3`. Each replicate is organized by substrate, then acquisition directory (for example, `Chitin/11.22.21_17`).
- **Experiment 1.3.2:** the download root contains `Activity_1.3_rep7`, or the reviewer may pass that directory itself. Its acquisition directories are directly below the replicate root (for example, `10.19.22_11AM_TP8_Chitin`).
- Consult `ai/activity_replicate_comparison/README.md` and its inventories before changing discovery or filename logic. Do not normalize these two layouts into one assumed structure without validating both.

## Reviewer-facing expectations

- The README is the primary human-facing run guide. Keep its install and command examples accurate.
- Both runners must require only `--data-dir` for normal reviewer use. Keep sensible, documented defaults for outputs and scientific parameters.
- Validate missing files, incompatible image/mask dimensions, and an input directory that mixes the two experiment layouts before writing partial results whenever practical.
- Write results under `results/experiment_1.3.1` or `results/experiment_1.3.2` by default, unless the reviewer supplies `--output-dir`. Do not write into the downloaded data directory.
- Document randomized-analysis seed behavior. Experiment 1.3.1 defaults to an unseeded (`None`) random draw; reviewers may supply an integer `--seed` when reproducibility is required, and that seed must be retained with the run record.

## Working conventions

- Use `uv sync` to create the locked project environment and `uv run ...` to execute commands.
- Keep 1.3.1 and 1.3.2 paths, outputs, and validation separate unless a shared behavior is genuinely layout-independent.
- Prefer explicit errors that identify the expected download layout and the missing or incompatible input.
- Keep open implementation and reviewer-workflow questions in `ai/questions/`. Keep observed replicate-layout comparisons in `ai/activity_replicate_comparison/`.
- Do not modify generated inventories in `ai/activity_replicate_comparison/` by hand. Regenerate them from their documented source process when that comparison is intentionally refreshed.

## Before submitting a change

- Run the relevant runner's help command: `uv run biofilm-experiment-1.3.1 --help` and/or `uv run biofilm-experiment-1.3.2 --help`.
- Run focused checks that do not require the public download when possible, such as `uv run python -m compileall -q .`.
- For changes affecting discovery, processing, outputs, metadata, or scientific parameters, test against the appropriate downloaded data package before representing the workflow as reviewer-ready.
- Update `README.md` and the relevant AI helper note when a reviewer-visible behavior or a documented experiment assumption changes.
