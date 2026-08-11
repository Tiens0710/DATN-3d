import json
from pathlib import Path

import numpy as np
import trimesh


# Approximate real-world dimensions in metres: width, depth, height.
# Uniform scaling preserves the shape produced by TRELLIS.
TARGET_DIMENSIONS = {
    "sofa": (2.10, 0.90, 0.85),
    "armchair": (0.85, 0.85, 0.95),
    "chair": (0.52, 0.58, 0.90),
    "coffee_table": (1.10, 0.65, 0.45),
    "dining_table": (1.55, 0.90, 0.75),
    "table": (1.20, 0.72, 0.75),
    "bed": (2.05, 1.65, 0.60),
    "nightstand": (0.50, 0.42, 0.55),
    "cabinet": (0.95, 0.45, 1.40),
    "floor_lamp": (0.38, 0.38, 1.65),
    "lamp": (0.32, 0.32, 0.55),
    "plant": (0.50, 0.50, 1.25),
}

# The axis that best represents each object's real-world scale.
PRIMARY_SCALE_AXIS = {
    "sofa": 0,
    "armchair": 2,
    "chair": 2,
    "coffee_table": 0,
    "dining_table": 0,
    "table": 0,
    "bed": 0,
    "nightstand": 2,
    "cabinet": 2,
    "floor_lamp": 2,
    "lamp": 2,
    "plant": 2,
}

HORIZONTAL_ALIGNMENT_CATEGORIES = {
    "sofa",
    "coffee_table",
    "dining_table",
    "table",
    "cabinet",
}

BACKREST_CATEGORIES = {"sofa", "armchair", "chair"}


def _as_mesh(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    if not loaded.geometry:
        raise ValueError(f"GLB has no geometry: {path}")
    # A GLB can store orientation, scale and translation in its scene graph.
    # Flattening raw geometries would silently discard those node transforms.
    return loaded.to_geometry()


def _category(label: str, description: str = "") -> str:
    text = f"{label} {description}".lower().replace("-", " ").replace("_", " ")
    if any(token in text for token in ("sofa", "couch", "settee")):
        return "sofa"
    if "armchair" in text or "lounge chair" in text:
        return "armchair"
    if any(token in text for token in ("coffee table", "tea table", "ban tra")):
        return "coffee_table"
    if "dining table" in text or "ban an" in text:
        return "dining_table"
    if any(token in text for token in ("nightstand", "bedside table", "bedside cabinet")):
        return "nightstand"
    if any(token in text for token in ("floor lamp", "standing lamp")):
        return "floor_lamp"
    for category in ("chair", "table", "bed", "cabinet", "lamp", "plant"):
        if category in text:
            return category
    return "object"


def _robust_bounds(mesh: trimesh.Trimesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) < 200:
        return np.asarray(mesh.bounds, dtype=float)
    lower = np.quantile(vertices, 0.005, axis=0)
    upper = np.quantile(vertices, 0.995, axis=0)
    if np.any(upper - lower <= 1e-6):
        return np.asarray(mesh.bounds, dtype=float)
    return np.stack([lower, upper])


def _category_scale(dimensions: np.ndarray, category: str, scene_scale: float) -> float:
    target = TARGET_DIMENSIONS.get(category)
    if not target or np.any(dimensions <= 1e-6):
        return 1.0
    target_dimensions = np.asarray(target, dtype=float) * scene_scale
    primary_axis = PRIMARY_SCALE_AXIS.get(category)
    if primary_axis is not None:
        ratio = target_dimensions[primary_axis] / dimensions[primary_axis]
    else:
        ratio = float(np.median(target_dimensions / dimensions))
    # TRELLIS exports are not guaranteed to use one unit convention. Some GLBs
    # arrive close to metres while others are normalized to a unit cube. The
    # old upper bound of 12 left normalized meshes about 8-20x too small after
    # the base 0.01 scene conversion, while placement still used metre-sized
    # furniture dimensions. Keep a broad corruption guard but allow the model
    # to reach the requested real-world primary dimension.
    return float(np.clip(ratio, 0.002, 500.0))


def _layout_dimensions(
    mesh_dimensions: np.ndarray,
    category: str,
    scene_scale: float,
) -> np.ndarray:
    """Bound noisy TRELLIS extents before converting relations to distances."""
    target = TARGET_DIMENSIONS.get(category)
    if not target:
        return np.asarray(mesh_dimensions, dtype=float)
    expected = np.asarray(target, dtype=float) * scene_scale
    return np.clip(mesh_dimensions, expected * 0.80, expected * 1.20)


def _canonicalize_orientation(mesh: trimesh.Trimesh, category: str) -> trimesh.Trimesh:
    """Align elongated furniture to X and seat fronts toward negative Z."""
    mesh = mesh.copy()
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) < 4:
        return mesh

    if category in HORIZONTAL_ALIGNMENT_CATEGORIES:
        horizontal = vertices[:, [0, 2]]
        centered = horizontal - np.median(horizontal, axis=0)
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[-1] > max(eigenvalues[0] * 1.10, 1e-10):
            principal = eigenvectors[:, -1]
            yaw = float(np.arctan2(principal[1], principal[0]))
            mesh.apply_transform(trimesh.transformations.rotation_matrix(yaw, [0, 1, 0]))

    if category in BACKREST_CATEGORIES:
        vertices = np.asarray(mesh.vertices, dtype=float)
        y_min = float(vertices[:, 1].min())
        y_max = float(vertices[:, 1].max())
        high = vertices[:, 1] >= y_min + 0.62 * (y_max - y_min)
        if np.count_nonzero(high) >= 3:
            high_z = float(np.median(vertices[high, 2]))
            center_z = float((vertices[:, 2].min() + vertices[:, 2].max()) / 2)
            # Backrests are the dominant high geometry. Keep them on +Z so
            # the usable front side of every seat consistently faces -Z.
            if high_z < center_z:
                mesh.apply_transform(
                    trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
                )
    return mesh


def _prepare_mesh(
    mesh: trimesh.Trimesh,
    category: str,
    scale_factor: float,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    mesh = _canonicalize_orientation(mesh, category)
    bounds = _robust_bounds(mesh)
    full_bounds = np.asarray(mesh.bounds, dtype=float)
    # GLB/model-viewer uses Y-up: X is left/right, Y is height, Z is depth.
    mesh.apply_translation(
        [
            -float((bounds[0, 0] + bounds[1, 0]) / 2),
            -float(full_bounds[0, 1]),
            -float((bounds[0, 2] + bounds[1, 2]) / 2),
        ]
    )
    mesh.apply_scale(scale_factor)

    raw_extents = _robust_bounds(mesh)[1] - _robust_bounds(mesh)[0]
    dimensions = raw_extents[[0, 2, 1]]  # width, depth, height
    if category in TARGET_DIMENSIONS and np.all(dimensions > 1e-6):
        scene_scale = max(scale_factor / 0.01, 1e-3)
        uniform_scale = _category_scale(dimensions, category, scene_scale)
        mesh.apply_scale(uniform_scale)
        scaled_bounds = _robust_bounds(mesh)
        scaled_extents = scaled_bounds[1] - scaled_bounds[0]
        dimensions = scaled_extents[[0, 2, 1]]

    return mesh, dimensions


def _find_entry(entries: dict, categories: set[str]) -> str | None:
    return next(
        (name for name, entry in entries.items() if entry["category"] in categories),
        None,
    )


def _semantic_relations(
    entries: dict,
    relations: list[dict],
    allow_inference: bool = True,
) -> tuple[list[dict], list[str]]:
    """Complete vague object lists with common interior-design relationships."""
    valid_names = set(entries)
    normalized = []
    for edge in relations:
        subject = edge.get("subject")
        target = edge.get("object")
        if subject in valid_names and target in valid_names and subject != target:
            normalized.append(
                {
                    "subject": subject,
                    "relation": str(edge.get("relation", "next_to")),
                    "object": target,
                }
            )

    inferred = []
    if not allow_inference:
        return normalized, inferred
    sofa = _find_entry(entries, {"sofa"})
    table = _find_entry(entries, {"coffee_table", "table", "dining_table"})
    lamp = _find_entry(entries, {"floor_lamp", "lamp"})
    plant = _find_entry(entries, {"plant"})
    bed = _find_entry(entries, {"bed"})
    nightstand = _find_entry(entries, {"nightstand"})

    def pair_edges(first: str | None, second: str | None) -> list[dict]:
        return [
            edge
            for edge in normalized
            if first and second and {edge["subject"], edge["object"]} == {first, second}
        ]

    # Furniture scenes need a geometry safety layer even when Gemini returned a
    # connected graph. Otherwise relation chains can place a lamp relative to a
    # table that is already relative to a sofa, making the final scene enormous.
    sofa_table_edges = pair_edges(sofa, table)
    if sofa and table:
        normalized = [
            edge
            for edge in normalized
            if {edge["subject"], edge["object"]} != {sofa, table}
        ]
        normalized.insert(0, {"subject": table, "relation": "in_front_of", "object": sofa})
        if not sofa_table_edges or any(edge["relation"] != "in_front_of" for edge in sofa_table_edges):
            inferred.append("table_in_front_of_sofa")

    if sofa and lamp:
        normalized = [
            edge
            for edge in normalized
            if lamp not in (edge["subject"], edge["object"])
        ]
        normalized.insert(1, {"subject": lamp, "relation": "right_of", "object": sofa})
        inferred.append("lamp_beside_sofa")
    if sofa and plant:
        normalized = [
            edge
            for edge in normalized
            if plant not in (edge["subject"], edge["object"])
        ]
        normalized.insert(1, {"subject": plant, "relation": "left_of", "object": sofa})
        inferred.append("plant_beside_sofa")
    if bed and nightstand:
        normalized = [
            edge
            for edge in normalized
            if {edge["subject"], edge["object"]} != {bed, nightstand}
        ]
        normalized.insert(0, {"subject": nightstand, "relation": "right_of", "object": bed})
        inferred.append("nightstand_beside_bed")

    # Remove duplicate edges after semantic replacement.
    unique = []
    seen = set()
    for edge in normalized:
        key = (edge["subject"], edge["relation"], edge["object"])
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique, inferred


def _relation_offset(subject: dict, target: dict, relation: str, side_index: int = 0) -> np.ndarray:
    scene_scale = max(subject["scene_scale"], target["scene_scale"])
    clearance = 0.06 * scene_scale
    horizontal = subject["width"] / 2 + target["width"] / 2 + clearance
    depth_clearance = 0.10 * scene_scale
    depth = subject["depth"] / 2 + target["depth"] / 2 + depth_clearance
    if relation == "left_of":
        return np.array([-horizontal, 0.0, 0.0])
    if relation == "right_of":
        return np.array([horizontal, 0.0, 0.0])
    if relation == "in_front_of":
        return np.array([0.0, 0.0, -depth])
    if relation == "behind":
        return np.array([0.0, 0.0, depth])
    if relation == "on_top_of":
        return np.array([0.0, target["height"] + 0.02 * target["scene_scale"], 0.0])
    if relation == "under":
        return np.array([0.0, -subject["height"] - 0.02 * subject["scene_scale"], 0.0])
    side = -1.0 if side_index % 2 == 0 else 1.0
    return np.array([side * horizontal, 0.0, 0.0])


def _choose_anchor(entries: dict) -> str:
    priority = ("sofa", "bed", "dining_table", "table", "coffee_table", "chair")
    for category in priority:
        match = _find_entry(entries, {category})
        if match:
            return match
    return max(entries, key=lambda name: entries[name]["width"] * entries[name]["depth"])


def _constrain_floor_plan(entries: dict, relations: list[dict], anchor: str) -> None:
    """Keep relation chains compact enough for a practical room composition."""
    adjacency = {name: set() for name in entries}
    for edge in relations:
        subject = edge.get("subject")
        target = edge.get("object")
        if subject in adjacency and target in adjacency and subject != target:
            adjacency[subject].add(target)
            adjacency[target].add(subject)

    hops = {anchor: 0}
    pending = [anchor]
    while pending:
        current = pending.pop(0)
        for neighbor in adjacency[current]:
            if neighbor not in hops:
                hops[neighbor] = hops[current] + 1
                pending.append(neighbor)

    anchor_entry = entries[anchor]
    for name, entry in entries.items():
        if name == anchor:
            continue
        hop_count = max(1, hops.get(name, 1))
        scene_scale = max(anchor_entry["scene_scale"], entry["scene_scale"])
        x_limit = hop_count * (
            anchor_entry["width"] / 2 + entry["width"] / 2 + 0.45 * scene_scale
        )
        z_limit = hop_count * (
            anchor_entry["depth"] / 2 + entry["depth"] / 2 + 0.45 * scene_scale
        )
        entry["position"][0] = float(np.clip(entry["position"][0], -x_limit, x_limit))
        entry["position"][2] = float(np.clip(entry["position"][2], -z_limit, z_limit))


def _position_entries(entries: dict, relations: list[dict]) -> None:
    anchor = _choose_anchor(entries)
    positioned = {anchor}
    entries[anchor]["position"][:] = 0.0
    side_counts = {}

    for _ in range(max(1, len(entries) * 2)):
        changed = False
        for edge in relations:
            subject_id = edge["subject"]
            target_id = edge["object"]
            subject_positioned = subject_id in positioned
            target_positioned = target_id in positioned
            if subject_positioned == target_positioned:
                continue
            relation = edge["relation"]
            pair_key = (target_id, relation)
            side_index = side_counts.get(pair_key, 0)
            offset = _relation_offset(entries[subject_id], entries[target_id], relation, side_index)
            if target_positioned:
                entries[subject_id]["position"] = entries[target_id]["position"] + offset
                positioned.add(subject_id)
                side_counts[pair_key] = side_index + 1
            else:
                entries[target_id]["position"] = entries[subject_id]["position"] - offset
                positioned.add(target_id)
            changed = True
        if not changed:
            break

    # Disconnected objects form a second row behind the main composition.
    cursor_x = 0.0
    back_z = max((entry["depth"] for entry in entries.values()), default=1.0) + 0.35
    for name, entry in entries.items():
        if name in positioned:
            continue
        entry["position"][:] = [cursor_x, 0.0, back_z]
        cursor_x += entry["width"] + 0.25 * entry["scene_scale"]
        positioned.add(name)

    # Floor-standing objects must share y=0 in GLB coordinates.
    stacked = {edge["subject"] for edge in relations if edge["relation"] == "on_top_of"}
    for name, entry in entries.items():
        if name not in stacked:
            entry["position"][1] = 0.0

    _constrain_floor_plan(entries, relations, anchor)


def _apply_gemini_placements(
    entries: dict,
    placements: list[dict],
    relations: list[dict],
) -> bool:
    """Apply Gemini's complete metric 3D plan, correcting only invalid geometry."""
    placement_by_id = {}
    for placement in placements:
        object_id = str(placement.get("object_id", ""))
        try:
            position = np.asarray(placement["position_xyz"], dtype=float)
            rotation = float(placement.get("rotation_y_degrees", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if object_id in entries and position.shape == (3,) and np.all(np.isfinite(position)):
            placement_by_id[object_id] = (position, rotation)

    if set(placement_by_id) != set(entries):
        return False

    anchor = _choose_anchor(entries)
    anchor_position = placement_by_id[anchor][0]
    for name, entry in entries.items():
        position, rotation = placement_by_id[name]
        position = position.copy()
        position[0] -= anchor_position[0]
        position[2] -= anchor_position[2]
        entry["position"] = position
        entry["rotation_y_degrees"] = float(((rotation + 180.0) % 360.0) - 180.0)

    # Preserve Gemini's composition while fitting pathological coordinates into
    # a practical room envelope.
    horizontal = np.asarray(
        [[entry["position"][0], entry["position"][2]] for entry in entries.values()]
    )
    span = np.ptp(horizontal, axis=0)
    max_span = 5.0 * max((entry["scene_scale"] for entry in entries.values()), default=1.0)
    largest_span = float(max(span))
    if largest_span > max_span:
        compression = max_span / largest_span
        for entry in entries.values():
            entry["position"][[0, 2]] *= compression

    stacked = {edge["subject"] for edge in relations if edge.get("relation") == "on_top_of"}
    for name, entry in entries.items():
        if name not in stacked:
            entry["position"][1] = 0.0
        else:
            entry["position"][1] = max(0.0, float(entry["position"][1]))

    # Resolve only actual solid overlap. The direction chosen by Gemini is
    # retained; the movable relation subject receives the minimum correction.
    for edge in relations:
        subject_id = edge.get("subject")
        target_id = edge.get("object")
        if subject_id not in entries or target_id not in entries:
            continue
        subject = entries[subject_id]
        target = entries[target_id]
        relation = str(edge.get("relation", "next_to"))
        if relation == "on_top_of":
            subject["position"][1] = target["position"][1] + target["height"]
            continue
        if relation == "under":
            continue

        delta = subject["position"] - target["position"]
        required_x = subject["width"] / 2 + target["width"] / 2 + 0.06 * subject["scene_scale"]
        required_z = subject["depth"] / 2 + target["depth"] / 2 + 0.06 * subject["scene_scale"]
        overlap_x = required_x - abs(float(delta[0]))
        overlap_z = required_z - abs(float(delta[2]))
        if overlap_x <= 0 or overlap_z <= 0:
            continue

        if relation in {"left_of", "right_of"}:
            sign = -1.0 if relation == "left_of" else 1.0
            subject["position"][0] = target["position"][0] + sign * required_x
        elif relation in {"in_front_of", "behind"}:
            sign = -1.0 if relation == "in_front_of" else 1.0
            subject["position"][2] = target["position"][2] + sign * required_z
        elif overlap_x <= overlap_z:
            sign = -1.0 if delta[0] < 0 else 1.0
            subject["position"][0] = target["position"][0] + sign * required_x
        else:
            sign = -1.0 if delta[2] < 0 else 1.0
            subject["position"][2] = target["position"][2] + sign * required_z
    return True


def _relations_connect_all(entries: dict, relations: list[dict]) -> bool:
    if len(entries) <= 1:
        return True
    adjacency = {name: set() for name in entries}
    for edge in relations:
        subject = edge.get("subject")
        target = edge.get("object")
        if subject in adjacency and target in adjacency and subject != target:
            adjacency[subject].add(target)
            adjacency[target].add(subject)
    visited = set()
    pending = [next(iter(entries))]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency[current] - visited)
    return len(visited) == len(entries)


def combine_scene_meshes(
    models: list,
    output_scene_path: str,
    layout: dict | None = None,
    scale_factor: float = 0.01,
    allowed_root: str | None = None,
) -> bool:
    """Compose TRELLIS meshes into a relation-aware, room-like 3D scene."""
    if not models:
        raise ValueError("No models were provided for scene composition")

    layout = layout or {}
    entries = {}
    for model in models:
        path = model.get("model_path")
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Missing GLB for scene composition: {path}")
        resolved_path = Path(path).resolve()
        if allowed_root:
            resolved_root = Path(allowed_root).resolve()
            if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
                raise ValueError(f"Model path is outside the active run: {path}")

        name = str(model["name"])
        label = str(model.get("label", name))
        category = _category(label, str(model.get("description", "")))
        mesh, dimensions = _prepare_mesh(_as_mesh(str(resolved_path)), category, scale_factor)
        scene_scale = max(scale_factor / 0.01, 1e-3)
        placement_dimensions = _layout_dimensions(dimensions, category, scene_scale)
        entries[name] = {
            "model": model,
            "mesh": mesh,
            "category": category,
            "width": float(placement_dimensions[0]),
            "depth": float(placement_dimensions[1]),
            "height": float(placement_dimensions[2]),
            "mesh_dimensions": dimensions,
            "scene_scale": scene_scale,
            "position": np.zeros(3, dtype=float),
        }

    relation = str(layout.get("relation", "next_to"))
    relations = list(layout.get("relations") or [])
    names = list(entries)
    if not relations and len(names) > 1:
        relations = [
            {"subject": names[index + 1], "relation": relation, "object": names[index]}
            for index in range(len(names) - 1)
        ]
    relation_source = str(layout.get("relation_source", "deterministic_fallback"))
    gemini_complete = (
        relation_source == "gemini_structured"
        and _relations_connect_all(entries, relations)
    )
    placements = list(layout.get("placements") or [])
    gemini_placement_applied = (
        relation_source == "gemini_structured"
        and _apply_gemini_placements(entries, placements, relations)
    )
    relations, inferred_rules = _semantic_relations(
        entries,
        relations,
        allow_inference=not gemini_placement_applied,
    )
    if not gemini_placement_applied:
        _position_entries(entries, relations)

    scene = trimesh.Scene()
    report = {
        "relation": relation,
        "relations": relations,
        "inferred_rules": inferred_rules,
        "relation_source": relation_source,
        "gemini_layout_applied": gemini_complete,
        "gemini_placement_applied": gemini_placement_applied,
        "coordinate_system": {"x": "left-right", "y": "up", "z": "front-back"},
        "objects": [],
    }
    for name, entry in entries.items():
        rotation_y = float(entry.get("rotation_y_degrees", 0.0))
        if abs(rotation_y) > 1e-8:
            entry["mesh"].apply_transform(
                trimesh.transformations.rotation_matrix(
                    np.radians(rotation_y),
                    [0, 1, 0],
                )
            )
        entry["mesh"].apply_translation(entry["position"])
        scene.add_geometry(entry["mesh"], node_name=name)
        report["objects"].append(
            {
                "name": name,
                "label": entry["model"].get("label", name),
                "category": entry["category"],
                "position_xyz": [round(float(value), 6) for value in entry["position"]],
                "rotation_y_degrees": round(rotation_y, 6),
                "dimensions_wdh": [
                    round(entry["width"], 6),
                    round(entry["depth"], 6),
                    round(entry["height"], 6),
                ],
                "mesh_dimensions_wdh": [
                    round(float(value), 6) for value in entry["mesh_dimensions"]
                ],
            }
        )

    output_path = Path(output_scene_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(scene.export(file_type="glb"))
    output_path.with_suffix(".layout.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return output_path.is_file()
