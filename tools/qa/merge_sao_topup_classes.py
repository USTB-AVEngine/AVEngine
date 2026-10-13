"""Merge owner-approved SAO catalog classes while preserving untouched JSON bytes."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import tempfile

CLASS_MERGE_MAP = {
    "telephone": "telephone_bell_ringing",
    "phone_ring": "telephone_bell_ringing",
}
FROM_CLASS = "telephone"
TO_CLASS = "telephone_bell_ringing"
CHANGED_FIELDS = {
    "sound_class", "effective_qa21_answer_class",
    "merged_from_sound_class", "admission_gate_class",
}
DECODER = json.JSONDecoder()


def _skip(text, pos):
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    return pos


def _members(text, start=0):
    pos = _skip(text, start)
    if text[pos:pos + 1] != "{":
        raise ValueError("expected JSON object")
    pos = _skip(text, pos + 1)
    members = {}
    while text[pos:pos + 1] != "}":
        key_start = pos
        key, pos = DECODER.raw_decode(text, pos)
        if not isinstance(key, str) or key in members:
            raise ValueError("invalid or duplicate object key")
        pos = _skip(text, pos)
        if text[pos:pos + 1] != ":":
            raise ValueError("expected colon")
        value_start = _skip(text, pos + 1)
        _, value_end = DECODER.raw_decode(text, value_start)
        members[key] = (key_start, value_start, value_end)
        pos = _skip(text, value_end)
        if text[pos:pos + 1] == "}":
            break
        if text[pos:pos + 1] != ",":
            raise ValueError("expected object separator")
        pos = _skip(text, pos + 1)
    if text[pos:pos + 1] != "}":
        raise ValueError("unterminated object")
    return members, pos


def record_spans(text):
    members, _ = _members(text)
    if "records" not in members:
        raise ValueError("catalog records missing")
    _, start, end = members["records"]
    pos = _skip(text, start)
    if text[pos:pos + 1] != "[":
        raise ValueError("records must be an array")
    pos = _skip(text, pos + 1)
    rows = []
    while text[pos:pos + 1] != "]":
        value_start = pos
        row, value_end = DECODER.raw_decode(text, pos)
        if not isinstance(row, dict):
            raise ValueError("record must be an object")
        rows.append((value_start, value_end, row))
        pos = _skip(text, value_end)
        if text[pos:pos + 1] == "]":
            break
        if text[pos:pos + 1] != ",":
            raise ValueError("expected records separator")
        pos = _skip(text, pos + 1)
    if pos + 1 != end:
        raise ValueError("records array span mismatch")
    return rows


def _apply(text, edits):
    ordered = sorted(edits)
    previous_end = 0
    parts = []
    for start, end, replacement in ordered:
        if start < previous_end:
            raise ValueError("overlapping JSON edits")
        parts.extend((text[previous_end:start], replacement))
        previous_end = end
    parts.append(text[previous_end:])
    return "".join(parts)


def _edit_object(text, values):
    members, close = _members(text)
    edits = []
    additions = {}
    for key, value in values.items():
        rendered = json.dumps(value, ensure_ascii=False, indent=2)
        if key in members:
            _, start, end = members[key]
            edits.append((start, end, rendered))
        else:
            additions[key] = rendered
    if additions:
        first_key = next(iter(members.values()))[0] if members else close
        indent = text[text.rfind("\n", 0, first_key) + 1:first_key]
        if indent.strip():
            indent = "  "
        insertion = max(span[2] for span in members.values()) if members else close
        suffix = ""
        for key, rendered in additions.items():
            rendered = rendered.replace("\n", "\n" + indent)
            suffix += ("," if members or suffix else "") + "\n" + indent
            suffix += json.dumps(key) + ": " + rendered
        edits.append((insertion, insertion, suffix))
    return _apply(text, edits)


def _origin(row):
    for key in ("original_group", "source_original_id", "fsd50k_id", "source_origin"):
        if row.get(key):
            return str(row[key])
    return str(row["sound_asset_id"])


def merge_document(raw, *, decision_time, input_path):
    text = raw.decode("utf-8")
    document = json.loads(text)
    if "scope_change_6" in document:
        raise ValueError("scope_change_6 already exists; use the immutable v12 input")
    before = record_spans(text)
    ids = [r["sound_asset_id"] for _, _, r in before]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source IDs")
    before_counts = collections.Counter(r["sound_class"] for _, _, r in before)
    if not before_counts[FROM_CLASS] or not before_counts[TO_CLASS]:
        raise ValueError("both merge classes must be present")
    changed = []
    edits = []
    for start, end, row in before:
        if row["sound_class"] != FROM_CLASS:
            continue
        if row.get("admission_gate_class", FROM_CLASS) != FROM_CLASS:
            raise ValueError("existing admission gate class conflicts with telephone")
        if row.get("merged_from_sound_class", FROM_CLASS) != FROM_CLASS:
            raise ValueError("existing merge provenance conflicts with telephone")
        fields = {
            "sound_class": TO_CLASS,
            "effective_qa21_answer_class": TO_CLASS,
            "merged_from_sound_class": FROM_CLASS,
            "admission_gate_class": FROM_CLASS,
        }
        edits.append((start, end, _edit_object(text[start:end], fields)))
        changed.append(row)
    generated = sum(r.get("source_type") == "generated_audio" for r in changed)
    decision = {
        "decision": 'owner 2026-10-13: "电话声和电话铃可以合并"; merge telephone into telephone_bell_ringing',
        "owner_decision_time": decision_time,
        "owner_decision_time_precision": "approximate minute, as supplied by owner relay",
        "relay_time": "2026-10-13 19:5x Asia/Singapore",
        "merged_from_sound_class": FROM_CLASS,
        "merged_into_sound_class": TO_CLASS,
        "merged_records": len(changed),
        "merged_real_records": len(changed) - generated,
        "merged_generated_records": generated,
        "admission_gate_class": FROM_CLASS,
        "admission_rerun": False,
        "thresholds_changed": False,
        "classes_before": len(before_counts),
        "classes_after": len(before_counts) - 1,
        "records_before": len(before),
        "records_after": len(before),
        "input": {"path": str(input_path), "sha256": hashlib.sha256(raw).hexdigest()},
        "historical_metadata": "All earlier class names in prompts, model/admission receipts and segmentation stay as admitted; only the four top-level record fields change.",
    }
    merged = _apply(text, edits)
    merged = _edit_object(merged, {"scope_change_6": decision})
    after = record_spans(merged)
    if len(after) != len(before):
        raise ValueError("record count changed")
    unchanged_bytes = 0
    for (start, end, old), (new_start, new_end, new) in zip(before, after):
        old_raw, new_raw = text[start:end], merged[new_start:new_end]
        if old["sound_class"] != FROM_CLASS:
            if old_raw.encode("utf-8") != new_raw.encode("utf-8"):
                raise ValueError("untouched record bytes changed")
            unchanged_bytes += 1
        else:
            wanted = dict(old)
            wanted.update({
                "sound_class": TO_CLASS,
                "effective_qa21_answer_class": TO_CLASS,
                "merged_from_sound_class": FROM_CLASS,
                "admission_gate_class": FROM_CLASS,
            })
            if new != wanted:
                raise ValueError("unexpected changed-record content")
            old_members, _ = _members(old_raw)
            new_members, _ = _members(new_raw)
            for key, (_, a, b) in old_members.items():
                if key not in CHANGED_FIELDS:
                    _, na, nb = new_members[key]
                    if old_raw[a:b].encode("utf-8") != new_raw[na:nb].encode("utf-8"):
                        raise ValueError("historical field bytes changed")
    old_members, _ = _members(text)
    new_members, _ = _members(merged)
    for key, (_, start, end) in old_members.items():
        if key != "records":
            _, new_start, new_end = new_members[key]
            if text[start:end] != merged[new_start:new_end]:
                raise ValueError("historical top-level metadata changed")
    after_counts = collections.Counter(r["sound_class"] for _, _, r in after)
    if after_counts[TO_CLASS] != before_counts[FROM_CLASS] + before_counts[TO_CLASS]:
        raise ValueError("merged class count mismatch")
    combined = [r for _, _, r in after if r["sound_class"] == TO_CLASS]
    summary = {
        "status": "pass",
        "record_count": len(after), "class_count": len(after_counts),
        "class_counts": dict(sorted(after_counts.items())),
        "merged_records": len(changed),
        "merged_real_records": len(changed) - generated,
        "merged_generated_records": generated,
        "merged_sound_asset_ids": [r["sound_asset_id"] for r in changed],
        "telephone_bell_ringing_records_before": before_counts[TO_CLASS],
        "telephone_bell_ringing_records_after": after_counts[TO_CLASS],
        "telephone_bell_ringing_distinct_originals": len({_origin(r) for r in combined}),
        "distinct_original_key_policy": "same as v12: original_group, source_original_id, fsd50k_id, source_origin, sound_asset_id",
        "unchanged_records_byte_identical": unchanged_bytes,
        "changed_records_other_field_values_byte_identical": True,
        "historical_top_level_values_byte_identical": True,
        "ids_and_record_order_preserved": [r["sound_asset_id"] for _, _, r in after] == ids,
        "admission_rerun": False, "new_audio_files": 0,
    }
    return merged.encode("utf-8"), summary


def _publish(path, payload):
    """Commit a complete sibling without replacing an existing artifact."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"completed artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--cci-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pull-json-path", required=True)
    parser.add_argument("--owner-decision-time", default="2026-10-13 19:4x Asia/Singapore")
    args = parser.parse_args()
    original = args.input.read_bytes()
    cci_original = args.cci_input.read_bytes()
    main_raw, summary = merge_document(original, decision_time=args.owner_decision_time, input_path=args.input)
    cci_raw, cci_summary = merge_document(cci_original, decision_time=args.owner_decision_time, input_path=args.cci_input)
    if summary["record_count"] != 2124 or summary["class_count"] != 26:
        raise ValueError("v13 requires 2124 records and 26 classes")
    main_rows = [r for _, _, r in record_spans(main_raw.decode("utf-8"))]
    cci_rows = [r for _, _, r in record_spans(cci_raw.decode("utf-8"))]
    if [r["sound_asset_id"] for r in main_rows] != [r["sound_asset_id"] for r in cci_rows]:
        raise ValueError("CCI IDs/order differ")
    if summary["class_counts"] != cci_summary["class_counts"]:
        raise ValueError("CCI class counts differ")
    for left, right in zip(main_rows, cci_rows):
        if any(left.get(k) != right.get(k) for k in CHANGED_FIELDS):
            raise ValueError("CCI merge fields differ")
    o = args.output
    main_path = o / "source_admission_v13_classes26_english_sao.json"
    cci_path = o / "cci_source_admission_v13_classes26_english_sao.json"
    if Path(args.pull_json_path).name != cci_path.name or "\n" in args.pull_json_path:
        raise ValueError("pull list must contain only the new CCI JSON")
    products = {
        main_path: main_raw,
        cci_path: cci_raw,
        o / "class_merge_map_v13.json": (json.dumps(CLASS_MERGE_MAP, ensure_ascii=False, indent=2) + "\n").encode(),
        o / "cci_pull_list_v13.txt": (args.pull_json_path + "\n").encode(),
    }
    summary.update({
        "inputs": [{"path": str(p), "sha256": hashlib.sha256(raw).hexdigest()}
                   for p, raw in ((args.input, original), (args.cci_input, cci_original))],
        "outputs": [{"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                    for p, raw in products.items()],
        "cci_validation": cci_summary,
        "pull_file_count": 1,
        "cci_transfer_performed": False,
        "gpu_used": False,
        "code_file": str(Path(__file__).resolve()),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    for path, payload in products.items():
        _publish(path, payload)
    _publish(o / "DELIVERY_v13.json", (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode())
    if args.input.read_bytes() != original or args.cci_input.read_bytes() != cci_original:
        raise ValueError("v12 input changed during merge")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
