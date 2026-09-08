# Test suite

The tests are standalone Python `unittest` scripts. They use the locked `uv`
environment and require no public-data download. Their generated TIFFs,
pickles, tables, plots, and directories are temporary test artifacts; nothing
is written to the repository or retained when a test finishes.

## Run the checks

From the repository root, install the locked environment once:

```bash
uv sync
```

Then run the checks individually:

```bash
uv run python checks/layout_discovery.py
uv run python checks/test_scientific_parity.py
uv run python checks/test_runner_workflows.py
```

The first check verifies filename discovery without importing the image stack.
The second reads the historical notebook exports without executing them, then
checks metadata, raw-image helper equations, and temporal/spatial equations.
The third is the end-to-end check: it creates complete synthetic raw samples,
runs the actual Experiment 1.3.1 and 1.3.2 runner entry points, and inspects
their output products.

Useful supporting checks are:

```bash
uv run biofilm-experiment-1.3.1 --help
uv run biofilm-experiment-1.3.2 --help
uv run python -m compileall -q .
```

## Synthetic input directories

`test_runner_workflows.py` passes the following temporary `--data-dir`
directories to the runners. Each synthetic acquisition contains one
`*_w1SD BF_stitched_mask.tif` file and 24 fluorescence z-stacks: three
channels (`w4SD RFP`, `w3SD GFP`, `w2SD DAPI`) times eight sites (`s1`–`s8`).

```text
<temporary-directory>/download/                         # passed to 1.3.1
├── Activity_1.3_rep1/Chitin/11.22.21_17/
│   ├── 11.22.21_17_Chitin_1_w1SD BF_stitched_mask.tif
│   └── 11.22.21_17_Chitin_1_<channel>_s<1..8>.tif
├── Activity_1.3_rep2/Chitin/12.16.21_5PM/
│   └── 12.16.21_5PM_Chitin_1_<...>
└── Activity_1.3_rep3/Chitin/02.10.22_5PM/
    └── 02.10.22_5PM_Chitin_1_<...>

<temporary-directory>/download/                         # passed to 1.3.2
└── Activity_1.3_rep7/
    └── 10.19.22_11AM_TP8_Chitin/
        ├── 10.19.22_11AM_TP8_Chitin_1_w1SD BF_stitched_mask.tif
        └── 10.19.22_11AM_TP8_Chitin_1_<channel>_s<1..8>.tif
```

For Experiment 1.3.2, the same test executes the runner twice: once with
`<temporary-directory>/download` and once with
`<temporary-directory>/download/Activity_1.3_rep7`. This exercises both
accepted `--data-dir` forms.

`layout_discovery.py` creates smaller, empty placeholder files in the same
two layout conventions. `test_scientific_parity.py` creates only a temporary
mask TIFF and combined-image pickle because it tests calculations after raw
processing.

## Test output locations

The test runner deliberately overrides the normal runner defaults with paths
under its temporary directory:

| Test | Runner output path while the test is running | Kept after test? |
| --- | --- | --- |
| Experiment 1.3.1 end-to-end test | `<temporary-directory>/outputs-1.3.1/` | No — the `TemporaryDirectory` context removes it. |
| Experiment 1.3.2, download-root input | `<temporary-directory>/outputs-1.3.2-root/` | No — the `TemporaryDirectory` context removes it. |
| Experiment 1.3.2, direct-rep7 input | `<temporary-directory>/outputs-1.3.2-direct/` | No — the `TemporaryDirectory` context removes it. |
| Discovery and calculation tests | No runner output directory; only temporary fixtures | No — removed automatically. |

The end-to-end test asserts that no `.pkl` files are written below the
synthetic download directory. Test results are printed to the terminal by
`unittest`; a successful run exits with status 0.

For a retained manual run on public data, use either a readable `--run-id` or
a deliberate `--output-dir`. Without either option, the runners write to a
new timestamped child directory under `results/experiment_1.3.1` or
`results/experiment_1.3.2` relative to the directory where the command is
run. See the repository README for those commands.
