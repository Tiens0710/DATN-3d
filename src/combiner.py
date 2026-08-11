import json
from pathlib import Path

import numpy as np
import trimesh


def _as_mesh(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    geometries = list(loaded.geometry.values())
    if not geometries:
        raise ValueError(f"GLB has no geometry: {path}")
    return trimesh.util.concatenate(geometries)


def _center_on_floor(mesh: trimesh.Trimesh, scale_factor: float) -> trimesh.Trimesh:
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
    return mesh


def combine_scene_meshes(
    models: list,
    output_scene_path: str,
    layout: dict | None = None,
    scale_factor: float = 0.01,
    allowed_root: str | None = None,
) -> bool:
    """Compose separate TRELLIS meshes using the original relation-aware notebook rule."""
    if not models:
        raise ValueError("No models were provided for scene composition")

    layout = layout or {}
    relation = str(layout.get("relation", "next_to"))
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
        mesh = _center_on_floor(_as_mesh(str(resolved_path)), scale_factor)
        bounds = np.asarray(mesh.bounds, dtype=float)
        entries[model["name"]] = {
            "model": model,
            "mesh": mesh,
            "width": float(bounds[1, 0] - bounds[0, 0]),
            "depth": float(bounds[1, 1] - bounds[0, 1]),
            "height": float(bounds[1, 2] - bounds[0, 2]),
            "position": np.zeros(3, dtype=float),
        }

    names = list(entries)
    subject_id = layout.get("subject_id") if layout.get("subject_id") in entries else names[0]
    target_id = layout.get("target_id") if layout.get("target_id") in entries else None
    if target_id == subject_id:
        target_id = None
    if target_id is None and len(names) > 1:
        target_id = next(name for name in names if name != subject_id)

    gap = 0.12 * scale_factor
    relations = list(layout.get("relations") or [])
    if not relations and target_id:
        relations = [{"subject": subject_id, "relation": relation, "object": target_id}]

    positioned = set()
    for edge in relations:
        edge_subject_id = edge.get("subject")
        edge_target_id = edge.get("object")
        if edge_subject_id not in entries or edge_target_id not in entries:
            continue
        edge_relation = str(edge.get("relation", "next_to"))
        subject = entries[edge_subject_id]
        target = entries[edge_target_id]
        subject_was_positioned = edge_subject_id in positioned
        target_was_positioned = edge_target_id in positioned
        if not target_was_positioned and not subject_was_positioned:
            target["position"][:] = 0.0
            positioned.add(edge_target_id)

        horizontal = subject["width"] / 2 + target["width"] / 2 + gap
        depth = subject["depth"] / 2 + target["depth"] / 2 + gap
        if edge_relation in ("next_to", "left_of"):
            offset = np.array([-horizontal, 0.0, 0.0])
        elif edge_relation == "right_of":
            offset = np.array([horizontal, 0.0, 0.0])
        elif edge_relation == "in_front_of":
            offset = np.array([0.0, -depth, 0.0])
        elif edge_relation == "behind":
            offset = np.array([0.0, depth, 0.0])
        elif edge_relation == "on_top_of":
            offset = np.array([0.0, 0.0, target["height"] + gap])
        elif edge_relation == "under":
            # Keep the lower object on the floor and lift the upper object.
            if target_was_positioned or not subject_was_positioned:
                subject["position"] = target["position"].copy()
                subject["position"][2] = 0.0
                target["position"][2] = subject["height"] + gap
            else:
                target["position"] = subject["position"].copy()
                target["position"][2] = subject["height"] + gap
            positioned.update((edge_subject_id, edge_target_id))
            continue
        else:
            offset = np.array([-horizontal, 0.0, 0.0])

        if target_was_positioned or not subject_was_positioned:
            subject["position"] = target["position"] + offset
        elif subject_was_positioned and not target_was_positioned:
            target["position"] = subject["position"] - offset
        positioned.update((edge_subject_id, edge_target_id))

    cursor_x = max((entry["position"][0] + entry["width"] / 2 for entry in entries.values()), default=0.0)
    for name, entry in entries.items():
        if name in positioned:
            continue
        entry["position"][0] = cursor_x + gap + entry["width"] / 2
        cursor_x = entry["position"][0] + entry["width"] / 2

    scene = trimesh.Scene()
    report = {
        "relation": relation,
        "relations": relations,
        "subject_id": subject_id,
        "target_id": target_id,
        "objects": [],
    }
    for name, entry in entries.items():
        entry["mesh"].apply_translation(entry["position"])
        scene.add_geometry(entry["mesh"], node_name=name)
        report["objects"].append(
            {
                "name": name,
                "label": entry["model"].get("label", name),
                "position_xyz": [float(value) for value in entry["position"]],
            }
        )

    output_path = Path(output_scene_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(scene.export(file_type="glb"))
    output_path.with_suffix(".layout.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path.is_file()
