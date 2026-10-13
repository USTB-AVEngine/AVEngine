"""Byte-preserving class relabeling leaves original admission evidence intact."""
import json

import pytest

from tools.qa.merge_sao_topup_classes import merge_document, record_spans, _publish

pytestmark = pytest.mark.fast_unit


def fixture():
    # Noncanonical formatting, escaped Unicode and exponent spelling are deliberate.
    return (
        '{\n "profile" : "v12",\n "records": [\n'
        '  {"sound_asset_id":"a", "sound_class":"telephone",'
        '"effective_qa21_answer_class":"telephone","original_group":"fsd:1",'
        '"evidence":{"sound_class":"telephone","target":1.917129977e-1},'
        '"title":"\\u7535\\u8bdd"},\n'
        '  { "sound_asset_id" : "b", "sound_class" : "telephone_bell_ringing",'
        '"original_group":"fsd:2", "merged_from_sound_class":"phone_ring" },\n'
        '  {"sound_asset_id":"c","sound_class":"telephone",'
        '"source_type":"generated_audio","original_group":"sao:3",'
        '"allowed_splits":["train"],"license":"Stability AI Community License"}\n'
        ' ],\n "scope_change_2": { "merged_records" : 12 }\n}\n'
    ).encode()


def merge(raw):
    return merge_document(raw, decision_time="2026-10-13 19:4x Asia/Singapore", input_path="v12.json")


def test_preserves_other_record_and_historical_field_bytes():
    raw = fixture()
    result, receipt = merge(raw)
    assert b'{ "sound_asset_id" : "b", "sound_class" : "telephone_bell_ringing","original_group":"fsd:2", "merged_from_sound_class":"phone_ring" }' in result
    assert b'"evidence":{"sound_class":"telephone","target":1.917129977e-1}' in result
    assert b'"title":"\\u7535\\u8bdd"' in result
    assert b'"scope_change_2": { "merged_records" : 12 }' in result
    rows = json.loads(result)["records"]
    assert rows[0]["admission_gate_class"] == "telephone"
    assert rows[0]["effective_qa21_answer_class"] == "telephone_bell_ringing"
    assert rows[2]["allowed_splits"] == ["train"]
    assert rows[2]["license"] == "Stability AI Community License"
    assert rows[1]["merged_from_sound_class"] == "phone_ring"
    assert receipt["unchanged_records_byte_identical"] == 1
    assert receipt["merged_real_records"] == 1
    assert receipt["merged_generated_records"] == 1
    assert receipt["telephone_bell_ringing_records_after"] == 3
    assert receipt["telephone_bell_ringing_distinct_originals"] == 3
    assert receipt["ids_and_record_order_preserved"] is True
    assert receipt["new_audio_files"] == 0


def test_counts_shared_original_only_once():
    raw = fixture().replace(b'"original_group":"fsd:2"', b'"original_group":"fsd:1"')
    _, receipt = merge(raw)
    assert receipt["telephone_bell_ringing_records_after"] == 3
    assert receipt["telephone_bell_ringing_distinct_originals"] == 2


def test_duplicate_ids_rejected():
    raw = fixture().replace(b'"sound_asset_id" : "b"', b'"sound_asset_id" : "a"')
    with pytest.raises(ValueError, match="duplicate source IDs"):
        merge(raw)


def test_conflicting_admission_gate_not_overwritten():
    raw = fixture().replace(b'"original_group":"fsd:1"', b'"original_group":"fsd:1","admission_gate_class":"smoke_alarm"')
    with pytest.raises(ValueError, match="admission gate class conflicts"):
        merge(raw)


def test_completed_v13_cannot_be_used_as_v12():
    result, _ = merge(fixture())
    with pytest.raises(ValueError, match="scope_change_6 already exists"):
        merge(result)


def test_existing_gate_is_kept_and_compact_json_is_supported():
    value = json.loads(fixture())
    value["records"][0]["admission_gate_class"] = "telephone"
    result, _ = merge(json.dumps(value).encode())
    assert json.loads(result)["records"][0]["admission_gate_class"] == "telephone"
    assert len(record_spans(result.decode())) == 3


def test_publication_never_replaces_completed_output(tmp_path):
    target = tmp_path / "out.json"
    _publish(target, b'{"complete":true}\n')
    _publish(target, b'{"complete":true}\n')
    with pytest.raises(FileExistsError):
        _publish(target, b'{"complete":false}\n')
    assert target.read_bytes() == b'{"complete":true}\n'


def test_whitespace_after_records_array_is_preserved():
    raw = fixture().replace(b" ],", b" ] \\t,")
    # Use actual JSON whitespace rather than a backslash outside a string.
    raw = raw.replace(b"\\t", b"\t")
    result, _ = merge(raw)
    assert b" ] \t," in result
