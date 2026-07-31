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
        mesh = _center_on_floor(_as_mesh(path), scale_factor)
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
    subject = entries[subject_id]
    target = entries[target_id] if target_id else None
    if target:
        horizontal = subject["width"] / 2 + target["width"] / 2 + gap
        depth = subject["depth"] / 2 + target["depth"] / 2 + gap
        if relation in ("next_to", "left_of"):
            subject["position"][0], target["position"][0] = -horizontal / 2, horizontal / 2
        elif relation == "right_of":
            subject["position"][0], target["position"][0] = horizontal / 2, -horizontal / 2
        elif relation == "in_front_of":
            subject["position"][1], target["position"][1] = -depth / 2, depth / 2
        elif relation == "behind":
            subject["position"][1], target["position"][1] = depth / 2, -depth / 2
        elif relation == "on_top_of":
            subject["position"][2] = target["height"] + gap
        elif relation == "under":
            # Keep both origins aligned on the floor. This mirrors the notebook's
            # conservative rule; an impossible under-relation stays visible for QA.
            subject["position"][:] = 0.0
            target["position"][:] = 0.0
        else:
            subject["position"][0], target["position"][0] = -horizontal / 2, horizontal / 2

    cursor_x = max((entry["position"][0] + entry["width"] / 2 for entry in entries.values()), default=0.0)
    for name, entry in entries.items():
        if name in {subject_id, target_id}:
            continue
        entry["position"][0] = cursor_x + gap + entry["width"] / 2
        cursor_x = entry["position"][0] + entry["width"] / 2

    scene = trimesh.Scene()
    report = {"relation": relation, "subject_id": subject_id, "target_id": target_id, "objects": []}
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
