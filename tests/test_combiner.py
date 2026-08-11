import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ModuleNotFoundError:
    sys.modules["trimesh"] = types.SimpleNamespace(Trimesh=object)
    TRIMESH_AVAILABLE = False

from src.combiner import (
    _as_mesh,
    _category_scale,
    _constrain_floor_plan,
    _layout_dimensions,
    _position_entries,
    _relations_connect_all,
    _semantic_relations,
    combine_scene_meshes,
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

    def test_floor_plan_constrains_extreme_object_spacing(self):
        entries = {
            "sofa_1": self._entry("sofa", 2.1, 0.9, 0.85),
            "table_1": self._entry("coffee_table", 1.1, 0.65, 0.45),
            "lamp_1": self._entry("floor_lamp", 0.38, 0.38, 1.65),
        }
        entries["table_1"]["position"][:] = [0.0, 0.0, -12.0]
        entries["lamp_1"]["position"][:] = [15.0, 0.0, 0.0]
        relations = [
            {"subject": "table_1", "relation": "in_front_of", "object": "sofa_1"},
            {"subject": "lamp_1", "relation": "right_of", "object": "sofa_1"},
        ]

        _constrain_floor_plan(entries, relations, "sofa_1")

        self.assertLessEqual(abs(entries["table_1"]["position"][2]), 1.225)
        self.assertLessEqual(abs(entries["lamp_1"]["position"][0]), 1.69)
        self.assertEqual({entry["position"][1] for entry in entries.values()}, {0.0})

    @unittest.skipUnless(TRIMESH_AVAILABLE, "trimesh is required for GLB integration tests")
    def test_glb_scene_graph_transform_is_applied_when_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transformed.glb"
            source = trimesh.Scene()
            transform = np.eye(4)
            transform[:3, 3] = [10.0, 20.0, 30.0]
            source.add_geometry(
                trimesh.creation.box(extents=[2.0, 1.0, 3.0]),
                node_name="box",
                transform=transform,
            )
            path.write_bytes(source.export(file_type="glb"))

            loaded = _as_mesh(str(path))

            np.testing.assert_allclose(
                loaded.bounds,
                [[9.0, 19.5, 28.5], [11.0, 20.5, 31.5]],
                atol=1e-6,
            )

    @unittest.skipUnless(TRIMESH_AVAILABLE, "trimesh is required for GLB integration tests")
    def test_exported_room_scene_is_y_up_and_uses_z_for_depth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_specs = {
                "sofa_1": ("sofa", [2.10, 0.85, 0.90]),
                "table_1": ("coffee table", [1.10, 0.45, 0.65]),
                "lamp_1": ("floor lamp", [0.38, 1.65, 0.38]),
            }
            models = []
            for name, (label, extents) in model_specs.items():
                path = root / f"{name}.glb"
                mesh = trimesh.creation.box(extents=extents)
                mesh.apply_translation([0.0, extents[1] / 2, 0.0])
                path.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
                models.append({"name": name, "label": label, "model_path": str(path)})

            output_path = root / "room.glb"
            combine_scene_meshes(
                models,
                str(output_path),
                layout={
                    "relation_source": "gemini_structured",
                    "relations": [
                        {"subject": "table_1", "relation": "in_front_of", "object": "sofa_1"},
                        {"subject": "lamp_1", "relation": "right_of", "object": "sofa_1"},
                    ],
                },
            )

            report = json.loads(output_path.with_suffix(".layout.json").read_text())
            positions = {
                item["name"]: np.asarray(item["position_xyz"], dtype=float)
                for item in report["objects"]
            }
            self.assertEqual(report["coordinate_system"], {
                "x": "left-right",
                "y": "up",
                "z": "front-back",
            })
            self.assertTrue(all(abs(position[1]) < 1e-8 for position in positions.values()))
            self.assertLess(positions["table_1"][2], positions["sofa_1"][2])
            self.assertGreater(positions["lamp_1"][0], positions["sofa_1"][0])

            exported = trimesh.load(output_path, force="scene", process=False)
            self.assertEqual(len(exported.geometry), 3)
            for geometry in exported.geometry.values():
                self.assertAlmostEqual(float(geometry.bounds[0, 1]), 0.0, places=5)

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
