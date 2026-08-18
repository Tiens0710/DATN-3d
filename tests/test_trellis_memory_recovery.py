import unittest
from pathlib import Path

from src.generator_3d import _is_cuda_oom_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrellisMemoryRecoveryTests(unittest.TestCase):
    def test_cuda_oom_messages_are_detected(self):
        self.assertTrue(_is_cuda_oom_error("OutOfMemoryError: CUDA out of memory"))
        self.assertTrue(_is_cuda_oom_error("cuda out of memory while allocating"))
        self.assertFalse(_is_cuda_oom_error("crop validation failed"))

    def test_allocator_is_configured_before_torch_import(self):
        source = (PROJECT_ROOT / "worker_trellis.py").read_text(encoding="utf-8")
        allocator = 'os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF"'
        self.assertLess(source.index(allocator), source.index("import torch"))

    def test_t4_defaults_use_fast_texture_and_request_offload(self):
        source = (PROJECT_ROOT / "worker_trellis.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("TRELLIS_TEXTURE_MODE", "fast")', source)
        self.assertIn('"TRELLIS_OFFLOAD_BETWEEN_REQUESTS", "1"', source)
        self.assertIn('_move_pipeline("cpu")', source)


if __name__ == "__main__":
    unittest.main()
