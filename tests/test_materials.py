import json
import tempfile
import unittest
from pathlib import Path

from src.materials import patch_glb_materials


def _make_minimal_glb(materials):
    document = {
        "asset": {"version": "2.0"},
        "materials": materials,
    }
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total_length = 12 + 8 + len(payload)
    return (
        b"glTF"
        + (2).to_bytes(4, "little")
        + total_length.to_bytes(4, "little")
        + len(payload).to_bytes(4, "little")
        + (0x4E4F534A).to_bytes(4, "little")
        + payload
    )


class MaterialPatchTests(unittest.TestCase):
    def test_patch_glb_updates_pbr_factors_and_preserves_material_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.glb"
            output = root / "edited.glb"
            source.write_bytes(
                _make_minimal_glb([
                    {
                        "name": "mattress",
                        "pbrMetallicRoughness": {
                            "baseColorFactor": [0, 0, 0, 1],
                            "roughnessFactor": 1,
                            "metallicFactor": 0,
                        },
                    }
                ])
            )

            self.assertEqual(
                patch_glb_materials(
                    source,
                    output,
                    [{"base_color": [1, 1, 1, 1], "roughness": 0.82, "metallic": 0.05}],
                ),
                1,
            )
            raw = output.read_bytes()
            json_length = int.from_bytes(raw[12:16], "little")
            document = json.loads(raw[20:20 + json_length].decode("utf-8").strip())
            pbr = document["materials"][0]["pbrMetallicRoughness"]
            self.assertEqual(pbr["baseColorFactor"], [1.0, 1.0, 1.0, 1.0])
            self.assertEqual(pbr["roughnessFactor"], 0.82)
            self.assertEqual(pbr["metallicFactor"], 0.05)


if __name__ == "__main__":
    unittest.main()
