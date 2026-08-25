#!/usr/bin/env python3
"""The static admission runner could never validate a real watertight manifest.

blender_create_watertight_textured_proxy_mesh.py always writes attribute_input
with a fourth key, same_as_geometry_input, while emitter_contract
validate_file_record requires the field set to be exactly path/sha256/
size_bytes.  The runner's own tests stub the watertight command, so the first
real run is the first time the two met.

The fix validates the flag instead of ignoring it: a static object has no
separate attribute source, so the flag must be present and true.  That is
strictly stronger than what the runner intended to check.
"""

from __future__ import annotations

from pathlib import Path

TOOL = Path("/data/jzy/code/SPEAR-lead-b/tools/run_controlled_static_object_admission.py")

OLD = """        emitter_contract.validate_file_record(
            payload.get("attribute_input"),
            label="watertight attribute input",
            expected_path=pixal_path,
        )
"""

NEW = """        attribute_input = payload.get("attribute_input")
        # The watertight tool always adds same_as_geometry_input to this
        # record, and validate_file_record requires an exact field set, so the
        # flag is checked here and removed before the record is validated.  A
        # static object has no separate attribute source: the flag must say so.
        if (
            not isinstance(attribute_input, Mapping)
            or attribute_input.get("same_as_geometry_input") is not True
        ):
            raise AdmissionError(
                "watertight attribute input must reuse the approved geometry"
            )
        emitter_contract.validate_file_record(
            {
                key: value
                for key, value in attribute_input.items()
                if key != "same_as_geometry_input"
            },
            label="watertight attribute input",
            expected_path=pixal_path,
        )
"""

text = TOOL.read_text(encoding="utf-8")
if text.count(OLD) != 1:
    raise SystemExit(f"anchor matched {text.count(OLD)} times")
TOOL.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("patched", TOOL)
