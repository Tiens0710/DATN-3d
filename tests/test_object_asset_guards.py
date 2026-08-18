import ast
import hashlib
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
        "_generation_attempt",
        "_stable_object_seed",
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
        "hashlib": hashlib,
        "re": re,
        "Path": Path,
        "OBJECT_IMAGE_DIR": "/tmp/object_images",
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), "generator_helpers", "exec"), namespace)
    return namespace


class ObjectAssetGuardTests(unittest.TestCase):
    def test_vietnamese_sofa_does_not_add_a_separate_chair_guard(self):
        source = Path("server.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        selected = [
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_contains_prompt_term", "_ensure_furniture_details"}
        ]
        namespace = {"re": re}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "prompt_guard", "exec"), namespace)
        optimized = namespace["_ensure_furniture_details"](
            "Một ghế sofa, một bàn trà và một đèn sàn.",
            "Exactly one sofa, one coffee table and one floor lamp.",
        )

        self.assertNotIn("complete chair", optimized)
        self.assertIn("complete sofa", optimized)
        self.assertIn("complete floor lamp", optimized)

    def test_vietnamese_armchair_does_not_trigger_table_guard(self):
        source = Path("server.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        selected = [
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_contains_prompt_term", "_ensure_furniture_details"}
        ]
        namespace = {"re": re}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "armchair_guard", "exec"), namespace)
        optimized = namespace["_ensure_furniture_details"](
            "Một chiếc ghế bành hiện đại bằng gỗ.",
            "Exactly one modern wooden armchair, no other objects.",
        )

        self.assertNotIn("complete rectangular table", optimized)
        self.assertIn("complete chair", optimized)

    def test_descriptive_coffee_table_uses_table_structure_rules(self):
        helpers = load_generator_helpers()
        positive, negative = helpers["_object_prompt"](
            "coffee table",
            "A sofa with one wooden coffee table in front.",
            ["sofa", "coffee table"],
        )

        self.assertIn("use a conventional rectangular tabletop", positive)
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
        self.assertNotEqual(retry["seed"], original["seed"])

    def test_seed_depends_on_object_identity_not_list_slot(self):
        helpers = load_generator_helpers()
        seed = helpers["_stable_object_seed"]
        self.assertNotEqual(
            seed("scene", "sofa_1", "sofa", 0),
            seed("scene", "floor_lamp_3", "floor lamp", 0),
        )

    def test_prompts_reject_ground_and_sofa_bed_shapes(self):
        helpers = load_generator_helpers()
        _positive, table_negative = helpers["_object_prompt"](
            "coffee table", "one wooden coffee table", ["coffee table"]
        )
        sofa_positive, sofa_negative = helpers["_object_prompt"](
            "sofa", "one cream sofa", ["sofa"]
        )

        self.assertIn("rug", table_negative)
        self.assertIn("floor", table_negative)
        self.assertIn("two seat cushions", sofa_positive)
        self.assertIn("bed frame", sofa_negative)

    def test_floor_attachment_guard_rejects_a_broad_lower_sheet(self):
        source = Path("worker_sam2_dino.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        selected = [
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_mask_extent", "_looks_like_floor_attachment"}
        ]
        namespace = {"np": np}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "floor_guard", "exec"), namespace)
        guard = namespace["_looks_like_floor_attachment"]

        rug = np.zeros((200, 200), dtype=np.uint8)
        rug[30:135, 65:135] = 255
        rug[120:190, 20:180] = 255
        valid_table = np.zeros((200, 200), dtype=np.uint8)
        valid_table[40:85, 25:175] = 255
        valid_table[85:175, 35:50] = 255
        valid_table[85:175, 150:165] = 255
        valid_cabinet = np.zeros((200, 200), dtype=np.uint8)
        valid_cabinet[25:180, 45:155] = 255

        self.assertTrue(guard(rug))
        self.assertFalse(guard(valid_table))
        self.assertFalse(guard(valid_cabinet))

    def test_floor_lamp_shape_guard_rejects_a_solid_block(self):
        source = Path("worker_sam2_dino.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        selected = [
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_mask_extent", "_category_shape_warning"}
        ]
        namespace = {"np": np}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "shape_guard", "exec"), namespace)
        guard = namespace["_category_shape_warning"]

        block = np.zeros((220, 160), dtype=np.uint8)
        block[20:205, 45:115] = 255
        lamp = np.zeros((220, 160), dtype=np.uint8)
        lamp[15:55, 35:125] = 255
        lamp[50:185, 76:84] = 255
        lamp[180:205, 55:105] = 255

        self.assertIsNotNone(guard(block, "floor lamp"))
        self.assertIsNone(guard(lamp, "floor lamp"))

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
