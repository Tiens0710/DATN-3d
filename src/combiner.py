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


def _as_mesh(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    geometries = list(loaded.geometry.values())
    if not geometries:
        raise ValueError(f"GLB has no geometry: {path}")
    return trimesh.util.concatenate(geometries)


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


def _prepare_mesh(
    mesh: trimesh.Trimesh,
    category: str,
    scale_factor: float,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    mesh = mesh.copy()
    bounds = np.asarray(mesh.bounds, dtype=float)
    mesh.apply_translation(
        [
            -float((bounds[0, 0] + bounds[1, 0]) / 2),
            -float((bounds[0, 1] + bounds[1, 1]) / 2),
            -float(bounds[0, 2]),
        ]
    )
    mesh.apply_scale(scale_factor)

    dimensions = np.asarray(mesh.extents, dtype=float)
    target = TARGET_DIMENSIONS.get(category)
    if target and np.all(dimensions > 1e-6):
        scene_scale = max(scale_factor / 0.01, 1e-3)
        target_dimensions = np.asarray(target, dtype=float) * scene_scale
        ratios = target_dimensions / dimensions
        # The median resists a single noisy TRELLIS axis without distorting mesh geometry.
        uniform_scale = float(np.clip(np.median(ratios), 0.08, 12.0))
        mesh.apply_scale(uniform_scale)
        dimensions = np.asarray(mesh.extents, dtype=float)

    return mesh, dimensions


def _find_entry(entries: dict, categories: set[str]) -> str | None:
    return next(
        (name for name, entry in entries.items() if entry["category"] in categories),
        None,
    )


def _semantic_relations(entries: dict, relations: list[dict]) -> tuple[list[dict], list[str]]:
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

    # A vague "sofa and table" should form a seating area, not a horizontal catalog row.
    sofa_table_edges = pair_edges(sofa, table)
    if sofa and table:
        if not sofa_table_edges:
            normalized.append({"subject": table, "relation": "in_front_of", "object": sofa})
            inferred.append("table_in_front_of_sofa")
        elif all(edge["relation"] == "next_to" for edge in sofa_table_edges):
            for edge in sofa_table_edges:
                edge.update({"subject": table, "relation": "in_front_of", "object": sofa})
            inferred.append("table_in_front_of_sofa")

    if sofa and lamp and not pair_edges(sofa, lamp):
        normalized = [
            edge
            for edge in normalized
            if not (lamp in (edge["subject"], edge["object"]) and edge["relation"] == "next_to")
        ]
        normalized.append({"subject": lamp, "relation": "right_of", "object": sofa})
        inferred.append("lamp_beside_sofa")
    if sofa and plant and not pair_edges(sofa, plant):
        normalized = [
            edge
            for edge in normalized
            if not (plant in (edge["subject"], edge["object"]) and edge["relation"] == "next_to")
        ]
        normalized.append({"subject": plant, "relation": "right_of", "object": sofa})
        inferred.append("plant_beside_sofa")
    if bed and nightstand and not pair_edges(bed, nightstand):
        normalized.append({"subject": nightstand, "relation": "right_of", "object": bed})
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
    clearance = 0.16 * max(subject["scene_scale"], target["scene_scale"])
    horizontal = subject["width"] / 2 + target["width"] / 2 + clearance
    depth = subject["depth"] / 2 + target["depth"] / 2 + clearance
    if relation == "left_of":
        return np.array([-horizontal, 0.0, 0.0])
    if relation == "right_of":
        return np.array([horizontal, 0.0, 0.0])
    if relation == "in_front_of":
        return np.array([0.0, -depth, 0.0])
    if relation == "behind":
        return np.array([0.0, depth, 0.0])
    if relation == "on_top_of":
        return np.array([0.0, 0.0, target["height"] + 0.02 * target["scene_scale"]])
    if relation == "under":
        return np.array([0.0, 0.0, -subject["height"] - 0.02 * subject["scene_scale"]])
    side = -1.0 if side_index % 2 == 0 else 1.0
    return np.array([side * horizontal, 0.0, 0.0])


def _choose_anchor(entries: dict) -> str:
    priority = ("sofa", "bed", "dining_table", "table", "coffee_table", "chair")
    for category in priority:
        match = _find_entry(entries, {category})
        if match:
            return match
    return max(entries, key=lambda name: entries[name]["width"] * entries[name]["depth"])


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
    back_y = max((entry["depth"] for entry in entries.values()), default=1.0) + 0.35
    for name, entry in entries.items():
        if name in positioned:
            continue
        entry["position"][:] = [cursor_x, back_y, 0.0]
        cursor_x += entry["width"] + 0.25 * entry["scene_scale"]
        positioned.add(name)

    # Floor-standing objects must share z=0. Only stacking relations may lift them.
    stacked = {edge["subject"] for edge in relations if edge["relation"] == "on_top_of"}
    for name, entry in entries.items():
        if name not in stacked:
            entry["position"][2] = 0.0


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
        entries[name] = {
            "model": model,
            "mesh": mesh,
            "category": category,
            "width": float(dimensions[0]),
            "depth": float(dimensions[1]),
            "height": float(dimensions[2]),
            "scene_scale": max(scale_factor / 0.01, 1e-3),
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
    relations, inferred_rules = _semantic_relations(entries, relations)
    _position_entries(entries, relations)

    scene = trimesh.Scene()
    report = {
        "relation": relation,
        "relations": relations,
        "inferred_rules": inferred_rules,
        "coordinate_system": {"x": "left-right", "y": "front-back", "z": "up"},
        "objects": [],
    }
    for name, entry in entries.items():
        entry["mesh"].apply_translation(entry["position"])
        scene.add_geometry(entry["mesh"], node_name=name)
        report["objects"].append(
            {
                "name": name,
                "label": entry["model"].get("label", name),
                "category": entry["category"],
                "position_xyz": [round(float(value), 6) for value in entry["position"]],
                "dimensions_xyz": [
                    round(entry["width"], 6),
                    round(entry["depth"], 6),
                    round(entry["height"], 6),
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
