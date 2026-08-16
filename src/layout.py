"""Build a semantic top-view layout for the 2D preview and scene assembly."""

from __future__ import annotations

import math


CANVAS_SIZE = 512
CANVAS_MARGIN = 28
LAYOUT_GAP = 0.20

# Approximate X/Z footprints in metres. The preview is a top view, so height
# is intentionally ignored here; the real 3D dimensions are handled by the
# scene combiner later in the pipeline.
OBJECT_FOOTPRINTS = {
    "sofa": (2.10, 0.90),
    "bed": (2.05, 1.65),
    "wardrobe": (1.40, 0.65),
    "cabinet": (0.95, 0.45),
    "bookshelf": (0.95, 0.40),
    "shelf": (0.90, 0.35),
    "coffee_table": (1.10, 0.65),
    "dining_table": (1.55, 0.90),
    "table": (1.20, 0.72),
    "nightstand": (0.50, 0.42),
    "dresser": (1.10, 0.50),
    "desk": (1.20, 0.60),
    "chair": (0.52, 0.58),
    "armchair": (0.85, 0.85),
    "stool": (0.42, 0.42),
    "bench": (1.10, 0.42),
    "ottoman": (0.65, 0.65),
    "floor_lamp": (0.38, 0.38),
    "lamp": (0.32, 0.32),
    "plant": (0.50, 0.50),
    "object": (0.80, 0.80),
}

ANCHOR_PRIORITY = (
    "bed",
    "sofa",
    "dining_table",
    "table",
    "desk",
    "cabinet",
    "wardrobe",
)

INVERSE_RELATIONS = {
    "left_of": "right_of",
    "right_of": "left_of",
    "in_front_of": "behind",
    "behind": "in_front_of",
    "on_top_of": "under",
    "under": "on_top_of",
    "next_to": "next_to",
}


def _canonical_category(node: dict) -> str:
    text = " ".join(
        str(node.get(key, ""))
        for key in ("label", "description", "full")
    ).lower().replace("_", " ").replace("-", " ")

    if "bedside table" in text or "nightstand" in text:
        return "nightstand"
    if "floor lamp" in text or "standing lamp" in text:
        return "floor_lamp"
    if "coffee table" in text or "tea table" in text:
        return "coffee_table"
    if "dining table" in text:
        return "dining_table"
    if "armchair" in text or "lounge chair" in text:
        return "armchair"
    for category in (
        "wardrobe", "bookshelf", "dresser", "cabinet", "shelf", "sofa",
        "bed", "desk", "chair", "stool", "bench", "ottoman", "plant",
        "lamp", "table",
    ):
        if category in text:
            return category
    return "object"


def _footprints(nodes: list[dict]) -> dict[str, tuple[float, float]]:
    return {
        node["id"]: OBJECT_FOOTPRINTS.get(
            _canonical_category(node), OBJECT_FOOTPRINTS["object"]
        )
        for node in nodes
    }


def _placement_map(scene_graph: dict, node_ids: set[str]) -> tuple[dict, dict]:
    positions = {}
    rotations = {}
    placements = scene_graph.get("placements", []) or []
    nodes = list(scene_graph.get("nodes", []))

    for placement in placements:
        if not isinstance(placement, dict):
            continue
        object_id = placement.get("object_id")
        if not object_id and isinstance(placement.get("object"), int):
            index = placement["object"]
            if 0 <= index < len(nodes):
                object_id = nodes[index].get("id")
        if object_id not in node_ids:
            continue

        raw_position = placement.get("position_xyz")
        if not isinstance(raw_position, (list, tuple)) or len(raw_position) < 3:
            continue
        try:
            position = [float(raw_position[index]) for index in range(3)]
            rotation = float(placement.get("rotation_y_degrees", 0.0))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in position + [rotation]):
            continue
        positions[object_id] = (position[0], position[2])
        rotations[object_id] = rotation

    # A malformed planner sometimes returns [0, 0, 0] for every object. That
    # is not useful as a visual layout, so fall back to semantic relations.
    if len(positions) > 1 and len(set(positions.values())) == 1:
        return {}, {}
    return positions, rotations


def _preferred_next_to(subject: dict, target: dict) -> str:
    """Turn an underspecified next_to edge into a useful top-view direction."""
    subject_category = _canonical_category(subject)
    target_category = _canonical_category(target)

    if subject_category in {"coffee_table", "ottoman"} and target_category in {
        "sofa", "bed", "armchair"
    }:
        return "in_front_of"
    if subject_category in {"nightstand", "floor_lamp", "lamp", "plant"} and target_category in {
        "bed", "sofa", "armchair"
    }:
        return "right_of"
    if subject_category in {"wardrobe", "cabinet", "bookshelf", "shelf"} and target_category in {
        "bed", "sofa"
    }:
        return "behind"
    if subject_category in {"chair", "armchair", "stool"} and target_category in {
        "table", "dining_table", "desk"
    }:
        return "left_of"
    return "right_of"


def _relative_position(
    subject: dict,
    target: dict,
    relation: str,
    positions: dict,
    footprints: dict,
) -> tuple[float, float]:
    target_x, target_z = positions[target["id"]]
    subject_width, subject_depth = footprints[subject["id"]]
    target_width, target_depth = footprints[target["id"]]
    relation = relation if relation != "next_to" else _preferred_next_to(subject, target)

    if relation == "left_of":
        return (
            target_x - (target_width + subject_width) / 2 - LAYOUT_GAP,
            target_z,
        )
    if relation == "right_of":
        return (
            target_x + (target_width + subject_width) / 2 + LAYOUT_GAP,
            target_z,
        )
    if relation == "in_front_of":
        return (
            target_x,
            target_z - (target_depth + subject_depth) / 2 - LAYOUT_GAP,
        )
    if relation == "behind":
        return (
            target_x,
            target_z + (target_depth + subject_depth) / 2 + LAYOUT_GAP,
        )
    # on_top_of and under share the same X/Z footprint in a top view. The
    # vertical distinction is still preserved in scene_graph.placements.
    return target_x, target_z


def _anchor_node(nodes: list[dict]) -> dict:
    for category in ANCHOR_PRIORITY:
        for node in nodes:
            if _canonical_category(node) == category:
                return node
    return nodes[0]


def _build_world_positions(
    nodes: list[dict],
    edges: list[dict],
    explicit_positions: dict,
    footprints: dict,
) -> dict:
    positions = dict(explicit_positions)
    anchor = _anchor_node(nodes)
    positions.setdefault(anchor["id"], (0.0, 0.0))

    by_id = {node["id"]: node for node in nodes}
    # Propagate positions through the relation graph. Explicit Gemini
    # placements remain authoritative while missing entries are inferred.
    for _ in range(max(2, len(nodes) * 2)):
        changed = False
        for edge in edges:
            subject = by_id.get(edge.get("subject"))
            target = by_id.get(edge.get("object"))
            if not subject or not target:
                continue
            relation = str(edge.get("relation", "next_to"))
            if target["id"] in positions and subject["id"] not in positions:
                positions[subject["id"]] = _relative_position(
                    subject, target, relation, positions, footprints
                )
                changed = True
            elif subject["id"] in positions and target["id"] not in positions:
                inverse = INVERSE_RELATIONS.get(relation, "next_to")
                positions[target["id"]] = _relative_position(
                    target, subject, inverse, positions, footprints
                )
                changed = True
        if not changed:
            break

    # Disconnected nodes are placed around the anchor rather than in a row.
    anchor_x, anchor_z = positions[anchor["id"]]
    for index, node in enumerate(nodes):
        if node["id"] in positions:
            continue
        angle = index * 2.4
        radius = 1.8 + (index // 3) * 0.8
        positions[node["id"]] = (
            anchor_x + math.cos(angle) * radius,
            anchor_z + math.sin(angle) * radius,
        )
    return positions


def _to_canvas_boxes(
    nodes: list[dict],
    positions: dict,
    rotations: dict,
    footprints: dict,
) -> dict:
    extents = []
    for node in nodes:
        x, z = positions[node["id"]]
        width, depth = footprints[node["id"]]
        extents.append((x - width / 2, x + width / 2, z - depth / 2, z + depth / 2))

    min_x = min(item[0] for item in extents)
    max_x = max(item[1] for item in extents)
    min_z = min(item[2] for item in extents)
    max_z = max(item[3] for item in extents)
    world_width = max(max_x - min_x, 0.1)
    world_depth = max(max_z - min_z, 0.1)
    usable = CANVAS_SIZE - CANVAS_MARGIN * 2
    scale = min(usable / world_width, usable / world_depth)
    scale = min(max(scale, 28.0), 125.0)

    scene_center_x = (min_x + max_x) / 2
    scene_center_z = (min_z + max_z) / 2
    center = CANVAS_SIZE / 2
    boxes = {}
    for node in nodes:
        object_id = node["id"]
        x, z = positions[object_id]
        width, depth = footprints[object_id]
        box_width = max(24.0, width * scale)
        box_height = max(24.0, depth * scale)
        center_x = center + (x - scene_center_x) * scale
        center_y = center + (scene_center_z - z) * scale
        boxes[object_id] = {
            "x": round(center_x - box_width / 2, 2),
            "y": round(center_y - box_height / 2, 2),
            "w": round(box_width, 2),
            "h": round(box_height, 2),
            "rotation_y_degrees": round(rotations.get(object_id, 0.0), 2),
            "world_position_xz": [round(x, 3), round(z, 3)],
        }
    return boxes


def compute_layout(scene_graph: dict) -> dict:
    """Compute a top-view layout that reflects semantic 3D placement."""
    nodes = list(scene_graph.get("nodes", []))
    edges = list(scene_graph.get("edges", []))
    if not nodes:
        return {"layout": {}, "objects": [], "relation": "single"}

    relation = str(scene_graph.get("relation", ""))
    subject_id = None
    target_id = None
    if edges:
        relation = str(edges[0].get("relation", relation or "next_to"))
        subject_id = edges[0].get("subject")
        target_id = edges[0].get("object")
    if len(nodes) == 1:
        relation = "single"
    elif not relation:
        relation = "next_to"

    relations = [
        {
            "subject": edge.get("subject"),
            "relation": str(edge.get("relation", "next_to")),
            "object": edge.get("object"),
        }
        for edge in edges
        if edge.get("subject") and edge.get("object")
    ]

    node_ids = {node["id"] for node in nodes}
    footprints = _footprints(nodes)
    explicit_positions, rotations = _placement_map(scene_graph, node_ids)
    positions = _build_world_positions(nodes, edges, explicit_positions, footprints)
    boxes = _to_canvas_boxes(nodes, positions, rotations, footprints)

    objects = [
        {
            "id": node["id"],
            "label": node["label"],
            "description": node.get("description", scene_graph.get("raw", "")),
        }
        for node in nodes
    ]
    return {
        "layout": boxes,
        "objects": objects,
        "relation": relation,
        "subject_id": subject_id or nodes[0]["id"],
        "target_id": target_id if len(nodes) > 1 else None,
        "relations": relations,
        "placements": list(scene_graph.get("placements", [])),
        "relation_source": scene_graph.get("parser_source", "deterministic_fallback"),
        "mode": "objectwise",
        "canvas": {
            "width": CANVAS_SIZE,
            "height": CANVAS_SIZE,
            "projection": "top_down_xz",
        },
    }
