import tempfile
import unittest
import zipfile
from pathlib import Path


class RunArchiveTests(unittest.TestCase):
    def test_full_archive_contains_pipeline_artifacts_without_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            outputs = root / "outputs"
            crops = root / "crops"
            models = root / "models"
            for path in (outputs, crops, models):
                path.mkdir(parents=True)
            (root / "generation_metadata.json").write_text("{}", encoding="utf-8")
            (crops / "chair.png").write_bytes(b"png")
            (models / "chair.glb").write_bytes(b"glb")
            (outputs / "scene_combined.glb").write_bytes(b"scene")

            archive_path = outputs / "pipeline_full_results.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for file_path in sorted(root.rglob("*")):
                    if not file_path.is_file() or file_path.resolve() == archive_path.resolve():
                        continue
                    archive.write(file_path, file_path.relative_to(root).as_posix())

            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())

            self.assertIn("generation_metadata.json", names)
            self.assertIn("crops/chair.png", names)
            self.assertIn("models/chair.glb", names)
            self.assertIn("outputs/scene_combined.glb", names)
            self.assertNotIn("outputs/pipeline_full_results.zip", names)


if __name__ == "__main__":
    unittest.main()
