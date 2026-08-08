from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
TOOL_PATH = REPOSITORY / "tools/qa/derive_native_occluder_evidence.py"
SPEC = importlib.util.spec_from_file_location("derive_native_occluder_evidence", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


CHAIR = "native_static_object::Meshes/05_chair/Round_Table_Chair_01"
TABLE = "native_static_object::Meshes/07_table/Round_Table"


def test_unique_static_occluder_requires_every_hidden_pixel() -> None:
    assert TOOL._admit_unique_occluder(
        occluded_pixels=1063,
        grouped={CHAIR: 1063},
    ) == [CHAIR]
    assert TOOL._admit_unique_occluder(
        occluded_pixels=1189,
        grouped={CHAIR: 1186, TABLE: 3},
    ) == []
    assert TOOL._admit_unique_occluder(
        occluded_pixels=1063,
        grouped={CHAIR: 1062},
    ) == []


def test_unique_static_occluder_rejects_tiny_hidden_regions() -> None:
    assert TOOL._admit_unique_occluder(
        occluded_pixels=TOOL.MIN_OCCLUDED_PIXELS - 1,
        grouped={CHAIR: TOOL.MIN_OCCLUDED_PIXELS - 1},
    ) == []
