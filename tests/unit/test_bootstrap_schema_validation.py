from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "scripts" / "validate_schemas.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_schemas", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_schema(path: Path, *, schema_id: str, value_type: str = "object") -> None:
    path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": schema_id,
                "type": value_type,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_repository_schema_set_is_draft_2020_12_and_hashable() -> None:
    module = _load_script()
    report = module.validate_schema_directory(REPOSITORY_ROOT / "schemas")
    assert report["status"] == "pass", report["errors"]
    assert report["schema_count"] == len(report["files"])
    assert report["schema_count"] > 0
    assert len(report["set_sha256"]) == 64
    assert "avengine_release_manifest_v1.schema.json" in {
        record["path"] for record in report["files"]
    }


def test_schema_validation_rejects_duplicate_ids(tmp_path: Path) -> None:
    module = _load_script()
    schema_id = "https://avengine.local/schema/duplicate.schema.json"
    _write_schema(tmp_path / "a.json", schema_id=schema_id)
    _write_schema(tmp_path / "b.json", schema_id=schema_id)
    report = module.validate_schema_directory(tmp_path)
    assert report["status"] == "fail"
    assert any("duplicate $id" in error for error in report["errors"])


def test_schema_validation_rejects_wrong_dialect_and_invalid_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "$id": "https://avengine.local/schema/invalid.schema.json",
                "type": "not-a-json-schema-type",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = _load_script().validate_schema_directory(tmp_path)
    assert report["status"] == "fail"
    assert any("Draft 2020-12 canonical URI" in error for error in report["errors"])
    assert any("invalid Draft 2020-12 schema" in error for error in report["errors"])
