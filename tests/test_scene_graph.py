import unittest

from src.layout import compute_layout
from src.parser import parse_scene_graph, parse_scene_graph_from_objects


class SceneGraphTests(unittest.TestCase):
    def test_vietnamese_fallback_keeps_one_table_and_one_chair(self):
        graph = parse_scene_graph(
            "mot chiec ban go va mot chiec ghe go dat ben canh ban"
        )

        self.assertEqual(
            [node["label"] for node in graph["nodes"]],
            ["table", "chair"],
        )

    def test_repeated_objects_expand_without_duplicate_reference(self):
        graph = parse_scene_graph_from_objects(
            "two chairs beside one table",
            [
                {"label": "chair", "count": 2},
                {"label": "table", "count": 1},
            ],
            relations=[{"subject": 0, "relation": "next_to", "object": 1}],
        )

        self.assertEqual([node["label"] for node in graph["nodes"]], ["chair", "chair", "table"])
        self.assertEqual(len(graph["edges"]), 2)
        self.assertEqual({edge["subject"] for edge in graph["edges"]}, {"chair_1", "chair_2"})

    def test_unspecified_multi_object_graph_is_connected(self):
        graph = parse_scene_graph_from_objects(
            "a chair, a table, and a lamp",
            [{"label": "chair"}, {"label": "table"}, {"label": "lamp"}],
        )

        self.assertEqual(len(graph["edges"]), 2)
        self.assertEqual(graph["edges"][0]["object"], graph["edges"][1]["subject"])

    def test_layout_preserves_all_relations_and_stays_in_canvas(self):
        graph = parse_scene_graph_from_objects(
            "a lamp on a table beside a chair",
            [{"label": "lamp"}, {"label": "table"}, {"label": "chair"}],
            relations=[
                {"subject": 0, "relation": "on_top_of", "object": 1},
                {"subject": 1, "relation": "next_to", "object": 2},
            ],
        )
        layout = compute_layout(graph)

        self.assertEqual(len(layout["relations"]), 2)
        for box in layout["layout"].values():
            self.assertGreaterEqual(box["x"], 0)
            self.assertGreaterEqual(box["y"], 0)
            self.assertLessEqual(box["x"] + box["w"], 512)
            self.assertLessEqual(box["y"] + box["h"], 512)

    def test_gemini_metric_placements_survive_parser_and_layout(self):
        graph = parse_scene_graph_from_objects(
            "a sofa with a coffee table in front",
            [{"label": "sofa"}, {"label": "coffee table"}],
            relations=[{"subject": 1, "relation": "in_front_of", "object": 0}],
            placements=[
                {"object": 0, "position_xyz": [0, 0, 0], "rotation_y_degrees": 0},
                {"object": 1, "position_xyz": [0, 0, -1.1], "rotation_y_degrees": 10},
            ],
            parser_source="gemini_structured",
        )
        layout = compute_layout(graph)

        self.assertEqual(
            layout["placements"],
            [
                {"object_id": "sofa_1", "position_xyz": [0.0, 0.0, 0.0], "rotation_y_degrees": 0.0},
                {
                    "object_id": "coffee_table_2",
                    "position_xyz": [0.0, 0.0, -1.1],
                    "rotation_y_degrees": 10.0,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
