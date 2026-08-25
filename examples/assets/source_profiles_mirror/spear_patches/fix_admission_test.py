#!/usr/bin/env python3
"""Make the watertight stub emit what the real watertight tool emits.

The stub wrote attribute_input as a bare file record.  The real tool always
adds same_as_geometry_input, so the runner's validation of that field was
never exercised against reality and shipped broken.
"""

from __future__ import annotations

from pathlib import Path

TEST = Path(
    "/data/jzy/code/SPEAR-lead-b/tests/tools/"
    "test_run_controlled_static_object_admission.py"
)

OLD = '        "attribute_input": _record(source),\n'
NEW = (
    "        # The real watertight tool always adds this fourth key; keeping the\n"
    "        # stub identical to it is what makes the runner's validation real.\n"
    '        "attribute_input": {**_record(source), "same_as_geometry_input": True},\n'
)

EXTRA_TEST = '''

def test_watertight_attribute_input_must_reuse_the_approved_geometry(tmp_path):
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "admission"

    def mutate(command, stage):
        if stage != "watertight":
            return None
        manifest = _argument(command, "--manifest")
        payload = contracts.load_json(manifest)
        payload["attribute_input"]["same_as_geometry_input"] = False
        _write_json(manifest, payload)
        return None

    with _stub_commands(after=mutate):
        code = admission.main(
            [
                "--decision-batch",
                str(fixture["decision_batch"]),
                "--plan",
                str(fixture["plan"]),
                "--output-root",
                str(output_root),
            ]
        )
    assert code != 0
'''

text = TEST.read_text(encoding="utf-8")
if text.count(OLD) != 1:
    raise SystemExit(f"stub anchor matched {text.count(OLD)} times")
TEST.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("patched the watertight stub")
