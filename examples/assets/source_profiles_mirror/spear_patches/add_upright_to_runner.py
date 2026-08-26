#!/usr/bin/env python3
"""Let the admission runner pass heading evidence v2 through.

The runner re-authenticates the authority before staging it, so it has to know
the v2 field set too. It deliberately does not re-derive the correction: the
finalizer owns that check, and duplicating a numeric gate in two places is how
they drift apart.
"""

from __future__ import annotations

from pathlib import Path

TOOL = Path("/data/jzy/code/SPEAR-lead-b/tools/run_controlled_static_object_admission.py")

OLD = '''    if (
        set(payload) != required
        or payload.get("schema") != "avengine_static_heading_review_v1"
        or payload.get("target_front_axis") != "positive-x"
        or payload.get("decision") != "approved_for_positive_x_normalization"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise AdmissionError("static heading authority contract is invalid")
'''

NEW = '''    schema = payload.get("schema")
    if schema == "avengine_static_heading_review_v2":
        # v2 adds the reviewed upright correction. Its numbers are checked by
        # the finalizer, which is the tool that applies them; re-deriving the
        # same gate here would only give the two a way to disagree.
        required = required | {"reviewed_upright_correction"}
    elif schema != "avengine_static_heading_review_v1":
        raise AdmissionError("static heading authority contract is invalid")
    if (
        set(payload) != required
        or payload.get("target_front_axis") != "positive-x"
        or payload.get("decision") != "approved_for_positive_x_normalization"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise AdmissionError("static heading authority contract is invalid")
'''

text = TOOL.read_text(encoding="utf-8")
if text.count(OLD) != 1:
    raise SystemExit(f"anchor matched {text.count(OLD)} times")
TOOL.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("admission runner accepts heading evidence v2")
