from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.rooms.room_screening.build_inventory import load_houses


class InventoryContractTests(unittest.TestCase):
    def test_reads_explicit_scene_list_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "houses.txt"
            path.write_text("# selected scenes\nhm3d_test_00001_synthetic\nhm3d_test_00001_synthetic\n", encoding="utf-8")
            self.assertEqual(load_houses(path, None), ["hm3d_test_00001_synthetic"])

    def test_rejects_path_like_scene_suffix(self):
        with self.assertRaisesRegex(ValueError, "invalid HM3D"):
            load_houses(None, ["hm3d_test_00001_../../outside"])

    def test_requires_at_least_one_scene(self):
        with self.assertRaisesRegex(ValueError, "provide --houses-file"):
            load_houses(None, None)


if __name__ == "__main__":
    unittest.main()
