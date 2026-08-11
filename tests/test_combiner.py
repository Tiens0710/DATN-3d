import sys
import types
import unittest

import numpy as np

try:
    import trimesh  # noqa: F401
except ModuleNotFoundError:
    sys.modules["trimesh"] = types.SimpleNamespace(Trimesh=object)

from src.combiner import (
    _category_scale,
    _layout_dimensions,
    _position_entries,
    _relations_connect_all,
    _semantic_relations,
)


class SceneCombinerTests(unittest.TestCase):
    def test_layout_dimensions_limit_noisy_mesh_depth(self):
        dimensions = _layout_dimensions(
            np.array([2.1, 4.8, 0.85]),
            "sofa",
            1.0,
        )

        self.assertAlmostEqual(dimensions[0], 2.10)
        self.assertAlmostEqual(dimensions[1], 0.90 * 1.20)
        self.assertAlmostEqual(dimensions[2], 0.85)

    def test_primary_axis_scaling_uses_realistic_furniture_size(self):
        sofa_scale = _category_scale(np.array([0.8, 0.7, 0.9]), "sofa", 1.0)
        lamp_scale = _category_scale(np.array([0.5, 0.5, 0.8]), "floor_lamp", 1.0)

        self.assertAlmostEqual(0.8 * sofa_scale, 2.10)
        self.assertAlmostEqual(0.8 * lamp_scale, 1.65)

    def test_living_room_uses_functional_positions_and_floor_alignment(self):
        entries = {
            "sofa_1": self._entry("sofa", 2.1, 0.9, 0.85),
            "table_1": self._entry("table", 1.2, 0.72, 0.75),
            "lamp_1": self._entry("floor_lamp", 0.38, 0.38, 1.65),
        }
        relations, inferred = _semantic_relations(
            entries,
            [
                {"subject": "sofa_1", "relation": "next_to", "object": "table_1"},
                {"subject": "table_1", "relation": "next_to", "object": "lamp_1"},
            ],
        )
        _position_entries(entries, relations)

        self.assertLess(entries["table_1"]["position"][2], entries["sofa_1"]["position"][2])
        self.assertGreater(entries["lamp_1"]["position"][0], entries["sofa_1"]["position"][0])
        self.assertEqual({entry["position"][1] for entry in entries.values()}, {0.0})
        self.assertIn("table_in_front_of_sofa", inferred)
        self.assertIn("lamp_beside_sofa", inferred)

    def test_complete_gemini_layout_is_not_overridden(self):
        entries = {
            "sofa_1": self._entry("sofa", 2.1, 0.9, 0.85),
            "table_1": self._entry("table", 1.2, 0.72, 0.75),
            "lamp_1": self._entry("floor_lamp", 0.38, 0.38, 1.65),
        }
        gemini_relations = [
            {"subject": "table_1", "relation": "left_of", "object": "sofa_1"},
            {"subject": "lamp_1", "relation": "behind", "object": "sofa_1"},
        ]

        self.assertTrue(_relations_connect_all(entries, gemini_relations))
        relations, inferred = _semantic_relations(
            entries,
            gemini_relations,
            allow_inference=False,
        )

        self.assertEqual(relations, gemini_relations)
        self.assertEqual(inferred, [])

    @staticmethod
    def _entry(category, width, depth, height):
        return {
            "category": category,
            "width": width,
            "depth": depth,
            "height": height,
            "scene_scale": 1.0,
            "position": np.zeros(3, dtype=float),
        }


if __name__ == "__main__":
    unittest.main()
