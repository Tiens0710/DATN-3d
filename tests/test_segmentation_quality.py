import ast
import unittest
from pathlib import Path

import numpy as np


def load_mask_helpers():
    source = Path("worker_sam2_dino.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    selected = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_mask_extent",
            "_mask_completeness",
            "_select_complete_mask",
            "_is_reusable_alpha",
        }
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "mask_helpers", "exec"), namespace)
    return namespace


class SegmentationQualityTests(unittest.TestCase):
    def test_reusable_alpha_rejects_empty_and_full_canvas_masks(self):
        helpers = load_mask_helpers()
        is_reusable = helpers["_is_reusable_alpha"]

        empty = np.zeros((100, 100), dtype=np.uint8)
        full = np.full((100, 100), 255, dtype=np.uint8)
        object_alpha = np.zeros((100, 100), dtype=np.uint8)
        object_alpha[20:80, 30:70] = 255

        self.assertFalse(is_reusable(empty))
        self.assertFalse(is_reusable(full))
        self.assertTrue(is_reusable(object_alpha))

    def test_complete_chair_mask_wins_over_higher_score_seat_only_mask(self):
        helpers = load_mask_helpers()
        partial = np.zeros((100, 100), dtype=bool)
        partial[48:92, 30:70] = True
        complete = np.zeros((100, 100), dtype=bool)
        complete[10:92, 30:70] = True

        best = helpers["_select_complete_mask"](
            np.stack([partial, complete]),
            np.asarray([0.96, 0.86]),
            [25, 5, 75, 95],
        )

        self.assertEqual(best, 1)


if __name__ == "__main__":
    unittest.main()
