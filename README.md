# Code for soil1.3_biofilm papers

## Reproducible command-line workflows

Install the locked project environment with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

`RawImageProcessing.py`, `PostAnalysis_spatial_sampling.py`, and `PostAnalysis_with_outlier.py` are notebook exports retained unchanged as the record of the original work. The executable files below use a shared non-interactive implementation of their core processing steps; they do not alter or replace the notebooks.

### Paper / Experiment 1.3.1 — three-replicate processing and spatial analysis

This paper accepts the path to its unpacked data download, which contains the three conventional `Activity_1.3_rep1`, `Activity_1.3_rep2`, and `Activity_1.3_rep3` directories. Each of these stores acquisition folders below the substrate directory (for example, `Chitin/11.22.21_17`). For each replicate, the command runs the core of `RawImageProcessing.py`, then performs both Experiment 1.3.1 spatial workflows: the random outlier-excluded analysis represented by `PostAnalysis_spatial_sampling.py` and the full, outlier-inclusive analysis represented by `PostAnalysis_with_outlier.py`.

```bash
uv run biofilm-experiment-1.3.1 \
  --data-dir /path/to/unpacked/experiment_1.3.1_data
```

### Paper / Experiment 1.3.2 — raw image processing

This paper uses raw image processing only at present. Supply either the unpacked data-download root or its `Activity_1.3_rep7` directory. Experiment 1.3.2 stores acquisition folders directly beneath that replicate directory (for example, `10.19.22_11AM_TP8_Chitin`). The executable discovers the published mask and TIFF filenames, then writes processed TIFFs, per-channel data, combined pickles, and one `<sample>_preview.png` for every processed image. The PNG combines a viewable RGB result with its population-density profile, so it can be reviewed without opening Jupyter. No spatial-analysis workflow is assumed for Experiment 1.3.2.

```bash
uv run biofilm-experiment-1.3.2 \
  --data-dir /path/to/unpacked/experiment_1.3.2_data
```

By default, each invocation writes to a new timestamped subdirectory below `./results/experiment_1.3.1/` or `./results/experiment_1.3.2/`, so a later run does not overwrite an earlier one. Use `--run-id <name>` to give the default subdirectory a readable name, or `--output-dir <path>` to select one exact output directory; `--run-id` and `--output-dir` cannot be combined. Experiment 1.3.1 writes the processed images beneath `processed/rep1` through `processed/rep3`. It also creates separate `PostAnalysis_spatial_sampling/` and `PostAnalysis_with_outlier/` output directories. Each branch includes `plots/temporal_biomass.png` and `plots/spatial_biomass.png`, which external reviewers can open directly without running a notebook. Its defaults preserve the historical 480-pixel/100-trial sampling analysis and the 500-pixel full spatial analysis.

Experiment 1.3.1 accepts `--seed INTEGER|none` for its randomized, outlier-excluded spatial sampling. Omitting the option, or using `--seed none`, requests a fresh non-deterministic draw. Supplying the same integer reproduces the same random-window selection. An unseeded result cannot be reproduced unless the resulting random-window table is retained. Experiment 1.3.2 does not accept a seed because its current raw-image-processing workflow has no randomized analysis step.

For example, a labeled and explicitly seeded 1.3.1 run is:

```bash
uv run biofilm-experiment-1.3.1 \
  --data-dir /path/to/unpacked/experiment_1.3.1_data \
  --run-id reviewer-seed-17 \
  --seed 17
```

For an intentionally unseeded draw, replace the last line with `--seed none`.

The Experiment 1.3.1 sample/timepoint/outlier assignments are bundled in [experiment_1.3.1_metadata.csv](experiment_1.3.1_metadata.csv), transcribed from the two original post-analysis notebooks. Reviewers do not edit it or create any manifest; both commands need only the downloaded-data directory path. The runners validate missing files, incompatible mask/image sizes, and a mixed Experiment 1.3.2/Experiment 1.3.1 layout before writing results.

## Workflow verification

[`docs/reviewer_workflow_parity.md`](docs/reviewer_workflow_parity.md) maps the historical notebook exports to the runner implementation and records the scope of the parity claim. It also documents the temporary synthetic filename/layout checks. Run them after `uv sync` without downloading public data:

```bash
uv run python checks/test_scientific_parity.py
uv run python checks/test_runner_workflows.py
```

See [`checks/README.md`](checks/README.md) for the complete test command list,
the temporary input directory trees, and the temporary output locations. See
[`docs/file_category_provenance.md`](docs/file_category_provenance.md) for the
origin and role of downloaded versus runner-generated filename categories.

## File-category provenance

Treat the downloaded data package as read-only input, even when a filename
looks processed. The runner reads the stitched bright-field mask
(`*_w1SD BF_stitched_mask` or `*_MASK`) and the 24 raw fluorescence z-stacks
per sample (`w2SD DAPI`, `w3SD GFP`, and `w4SD RFP`, each at sites `s1`–`s8`).
Downloaded `_stitched`, per-site `_wo_bg`, and per-site `_masked` products are
not generated or consumed by the reviewer runner.

The runner writes its own channel-level `_masked`, `_processed`, `_sharp`, and
`_combined_processed` products to the selected results directory. Experiment
1.3.1 additionally writes its `PostAnalysis_spatial_sampling/` and
`PostAnalysis_with_outlier/` tables and plots there. The detailed category,
origin, and usage table is in
[`docs/file_category_provenance.md`](docs/file_category_provenance.md).


Released under the Creative Commons 0 Public Domain Dedication: 
https://creativecommons.org/publicdomain/zero/1.0/
 
This material is free to use, and attribution is always appreciated.  Attribution may be as follows:

Authored by Song Feng at the Pacific Northwest National Laboratory, operated by Battelle for the U.S. Department of Energy.<br>
Please cite the following in your work:<br>
> Song Feng, Natalie Sadler, Arunima Bhattacharjee, Ryan McClure, Kirsten Hofmockel. (2026) Imaging and analysis data of short-term co-culture in a SoilChip.<br>
> PNNL DataHub: https://data.pnnl.gov/group/nodes/dataset/34902<br>
> DOI: https://doi.org/10.25584/3375157<br>


### Disclaimer:
This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor the Contractor, nor any or their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

<div align="center">

PACIFIC NORTHWEST NATIONAL LABORATORY<br>
operated by<br>
BATTELLE<br>
for the<br>
UNITED STATES DEPARTMENT OF ENERGY<br>
under Contract DE-AC05-76RL01830<br>

</div>
