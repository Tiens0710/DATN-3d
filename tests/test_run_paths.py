import tempfile
import sys
import types
import unittest
from pathlib import Path

# These tests exercise pure path/prompt helpers and never perform HTTP calls.
sys.modules.setdefault("requests", types.ModuleType("requests"))

from src.generator_2d import build_object_jobs
from src.generator_3d import _path_within


class RunPathTests(unittest.TestCase):
    def test_crop_path_must_stay_inside_active_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            valid = root / "crop.png"
            valid.touch()

            self.assertEqual(_path_within(str(valid), str(root)), valid.resolve())
            with self.assertRaises(ValueError):
                _path_within(str(Path(directory) / "other.png"), str(root))

    def test_object_id_cannot_escape_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                build_object_jobs(
                    "one chair",
                    [{"id": "../outside", "label": "chair"}],
                    object_image_dir=directory,
                )


if __name__ == "__main__":
    unittest.main()
