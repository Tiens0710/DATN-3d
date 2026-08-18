import ast
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

def load_crop_helper():
    source = Path("worker_sam2_dino.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    selected = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_mask_extent", "_save_trellis_rgba_crop"}
    ]
    namespace = {"np": np, "Image": Image, "os": __import__("os")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "crop_helpers", "exec"), namespace)
    return namespace["_save_trellis_rgba_crop"]


def load_generator_helpers():
    source = Path("src/generator_2d.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    constants = {"_OBJECT_ALIASES", "_GLOBAL_PROMPT_MARKERS", "_CLAUSE_BREAKS"}
    functions = {
        "_object_descriptor",
        "_isolation_background",
        "_object_category",
        "_object_prompt",
        "build_object_jobs",
    }
    selected = []
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    namespace = {
        "os": os,
        "re": re,
        "Path": Path,
        "OBJECT_IMAGE_DIR": "/tmp/object_images",
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), "generator_helpers", "exec"), namespace)
    return namespace


class ObjectAssetGuardTests(unittest.TestCase):
    def test_descriptive_coffee_table_uses_table_structure_rules(self):
        helpers = load_generator_helpers()
        positive, negative = helpers["_object_prompt"](
            "coffee table",
            "A sofa with one wooden coffee table in front.",
            ["sofa", "coffee table"],
        )

        self.assertIn("Use a conventional rectangular tabletop", positive)
        self.assertIn("four straight vertical legs", positive)
        self.assertIn("wireframe", negative)
        self.assertIn("round tabletop", negative)

    def test_clean_background_retry_uses_a_different_seed(self):
        helpers = load_generator_helpers()
        build_jobs = helpers["build_object_jobs"]
        with tempfile.TemporaryDirectory() as directory:
            original = build_jobs(
                "one coffee table",
                [{"id": "coffee_table_1", "label": "coffee table"}],
                directory,
            )[0]
            retry = build_jobs(
                "one coffee table",
                [
                    {
                        "id": "coffee_table_1",
                        "label": "coffee table",
                        "retry_clean_background": True,
                    }
                ],
                directory,
            )[0]

        self.assertNotEqual(original["seed"], retry["seed"])
        self.assertEqual(retry["seed"] - original["seed"], 10_000)

    def test_text_flow_crop_is_tight_not_a_full_transparent_canvas(self):
        save_crop = load_crop_helper()
        image = Image.new("RGB", (400, 400), "white")
        alpha = np.zeros((400, 400), dtype=np.uint8)
        alpha[120:280, 150:250] = 255

        with tempfile.TemporaryDirectory() as directory:
            crop_path, final_box, crop = save_crop(image, alpha, directory, "object")

            self.assertTrue(Path(crop_path).is_file())
            self.assertEqual(final_box, [150, 120, 250, 280])
            self.assertLess(crop.width, 400)
            self.assertLess(crop.height, 400)
            self.assertGreater((np.asarray(crop.getchannel("A")) >= 32).mean(), 0.35)


if __name__ == "__main__":
    unittest.main()
