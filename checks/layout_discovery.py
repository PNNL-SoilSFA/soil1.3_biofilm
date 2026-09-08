"""Regression checks for the two published raw-data directory layouts."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


def _install_analysis_import_stubs() -> None:
    """Keep path-discovery checks independent of the image-processing stack."""
    numpy = ModuleType("numpy")
    setattr(numpy, "ndarray", object)
    pandas = ModuleType("pandas")
    setattr(pandas, "DataFrame", object)
    skimage = ModuleType("skimage")
    setattr(skimage, "filters", ModuleType("skimage.filters"))
    setattr(skimage, "io", ModuleType("skimage.io"))
    setattr(skimage, "restoration", ModuleType("skimage.restoration"))
    sys.modules.setdefault("numpy", numpy)
    sys.modules.setdefault("pandas", pandas)
    sys.modules.setdefault("skimage", skimage)
    sys.modules.setdefault("skimage.filters", skimage.filters)
    sys.modules.setdefault("skimage.io", skimage.io)
    sys.modules.setdefault("skimage.restoration", skimage.restoration)
    sys.modules.setdefault("tifffile", ModuleType("tifffile"))


_install_analysis_import_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_analysis import (
    discover_experiment_1_3_1_replicates,
    discover_experiment_1_3_2_replicate,
    discover_raw_samples,
)


def _write_sample(directory: Path, sample_id: str, substrate: str, *, mask_case: str) -> None:
    directory.mkdir(parents=True)
    base = f"{sample_id}_{substrate}_1"
    (directory / f"{base}_w1SD BF_stitched_{mask_case}.tif").touch()
    for channel in ("w4SD RFP", "w3SD GFP", "w2SD DAPI"):
        for index in range(1, 9):
            (directory / f"{base}_{channel}_s{index}.TIF").touch()


class LayoutDiscoveryTest(unittest.TestCase):
    def test_experiment_1_3_2_acquisition_first_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            replicate = root / "Activity_1.3_rep7"
            _write_sample(replicate / "10.11.22_10PM_TP1_Chito5", "10.11.22_10PM_TP1", "Chito5", mask_case="mask")
            self.assertEqual(discover_experiment_1_3_2_replicate(root), replicate.resolve())
            records = discover_raw_samples(replicate, replicate="rep7")
            self.assertEqual([(record.sample_id, record.substrate) for record, *_ in records], [("10.11.22_10PM_TP1", "Chito5")])

    def test_experiment_1_3_1_substrate_first_layout_and_replicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index in range(1, 4):
                (root / f"Activity_1.3_rep{index}").mkdir()
            replicate = root / "Activity_1.3_rep1"
            _write_sample(replicate / "Chitin" / "11.18.21_23", "11.18.21_23", "Chitin", mask_case="MASK")
            self.assertEqual(discover_experiment_1_3_1_replicates(root), [(root / f"Activity_1.3_rep{index}").resolve() for index in range(1, 4)])
            records = discover_raw_samples(replicate, replicate="rep1")
            self.assertEqual([(record.sample_id, record.substrate) for record, *_ in records], [("11.18.21_23", "Chitin")])


if __name__ == "__main__":
    unittest.main()
