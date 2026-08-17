"""Small, dependency-free GLB material editing helpers.

The TRELLIS worker already exports a textured GLB.  These helpers update the
glTF PBR factors in the JSON chunk without loading the mesh or re-running
TRELLIS, so the original geometry and binary texture data remain unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


GLB_MAGIC = 0x46546C67  # b"glTF"
JSON_CHUNK_TYPE = 0x4E4F534A  # b"JSON"


def _normalise_material_value(value: Mapping[str, Any]) -> dict[str, Any]:
    color = value.get("base_color", [1.0, 1.0, 1.0, 1.0])
    if not isinstance(color, Sequence) or isinstance(color, (str, bytes)):
        raise ValueError("base_color must be an array of 3 or 4 numbers")
    if len(color) not in {3, 4}:
        raise ValueError("base_color must contain 3 or 4 numbers")

    normalised_color = [float(channel) for channel in color]
    if len(normalised_color) == 3:
        normalised_color.append(1.0)
    if any(channel < 0.0 or channel > 1.0 for channel in normalised_color):
        raise ValueError("base_color channels must be between 0 and 1")

    roughness = float(value.get("roughness", 0.8))
    metallic = float(value.get("metallic", 0.0))
    if not 0.0 <= roughness <= 1.0:
        raise ValueError("roughness must be between 0 and 1")
    if not 0.0 <= metallic <= 1.0:
        raise ValueError("metallic must be between 0 and 1")

    return {
        "base_color": normalised_color,
        "roughness": roughness,
        "metallic": metallic,
    }


def patch_glb_materials(
    source_path: str | Path,
    output_path: str | Path,
    material_values: Sequence[Mapping[str, Any]],
) -> int:
    """Write PBR factors into a GLB and return the number of materials updated.

    ``material_values`` uses the glTF material order.  Entries beyond the
    number of materials in the GLB are rejected so a caller cannot silently
    edit a different material than the one selected in the viewer.
    """

    if not material_values:
        raise ValueError("At least one material value is required")

    source = Path(source_path)
    output = Path(output_path)
    raw = source.read_bytes()
    if len(raw) < 20:
        raise ValueError("File is too small to be a valid GLB")

    header = int.from_bytes(raw[0:4], "little")
    version = int.from_bytes(raw[4:8], "little")
    if header != GLB_MAGIC or version != 2:
        raise ValueError("Only version 2 GLB files are supported")

    json_length = int.from_bytes(raw[12:16], "little")
    json_type = int.from_bytes(raw[16:20], "little")
    if json_type != JSON_CHUNK_TYPE:
        raise ValueError("GLB JSON chunk is missing")

    json_start = 20
    json_end = json_start + json_length
    if json_end > len(raw):
        raise ValueError("GLB JSON chunk is truncated")

    try:
        document = json.loads(raw[json_start:json_end].decode("utf-8").rstrip(" \t\r\n\x00"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid GLB JSON chunk: {exc}") from exc

    materials = document.get("materials")
    if not isinstance(materials, list):
        raise ValueError("GLB does not contain materials")
    if len(material_values) > len(materials):
        raise ValueError(
            f"Received {len(material_values)} material values for {len(materials)} GLB materials"
        )

    updated = 0
    for index, value in enumerate(material_values):
        if not isinstance(value, Mapping):
            raise ValueError(f"Material {index} must be an object")
        normalised = _normalise_material_value(value)
        pbr = materials[index].setdefault("pbrMetallicRoughness", {})
        pbr["baseColorFactor"] = normalised["base_color"]
        pbr["roughnessFactor"] = normalised["roughness"]
        pbr["metallicFactor"] = normalised["metallic"]
        updated += 1

    encoded_json = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    padded_length = (len(encoded_json) + 3) // 4 * 4
    padded_json = encoded_json + b" " * (padded_length - len(encoded_json))
    suffix = raw[json_end:]

    output_bytes = bytearray(12 + 8 + len(padded_json) + len(suffix))
    output_bytes[0:12] = raw[0:12]
    output_bytes[12:20] = raw[12:20]
    output_bytes[8:12] = len(output_bytes).to_bytes(4, "little")
    output_bytes[12:16] = len(padded_json).to_bytes(4, "little")
    output_bytes[20:20 + len(padded_json)] = padded_json
    output_bytes[20 + len(padded_json):] = suffix

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(output_bytes)
    temporary.replace(output)
    return updated
