from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.qa import binding_delivery


pytestmark = pytest.mark.fast_unit


def _question_item(question_id: str = "private-1") -> dict:
    options = [
        {"value": "yes", "label_en": "yes", "label_zh": "是"},
        {"value": "no", "label_en": "no", "label_zh": "否"},
    ]
    return {
        "schema": "avengine_qa_unified_question_v1",
        "status": "pass",
        "qa_id": "QA-01",
        "question_id": question_id,
        "forms": {
            "mcq": {
                "question_en": "Which answer?",
                "question_zh": "哪个答案？",
                "answer_type": "choice",
                "options": options,
                "gold": {"correct_index": 0, "value": "yes"},
            },
            "open": {
                "question_en": "Is it yes?",
                "question_zh": "是否为是？",
                "answer_type": "closed_set",
                "truth": "yes",
                "classes": {"yes": ["yes", "是"], "no": ["no", "否"]},
            },
        },
    }


def test_rewrite_paths_is_relative_to_the_copied_json_directory() -> None:
    source = Path("/source/facts/sample.json")
    path_map = {
        str(source): {
            "source_path": str(source),
            "delivered_path": "catalog/media/video.mp4",
        }
    }

    rewritten = binding_delivery._rewrite_paths(
        {"media": str(source)},
        path_map,
        destination_base=Path("catalog"),
        source_base=Path("/source/facts"),
    )

    assert rewritten == {"media": "media/video.mp4"}


def test_catalog_scoring_resolves_questions_relative_to_catalog_root(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    catalog_root = root / "catalog"
    (catalog_root / "questions").mkdir(parents=True)
    (catalog_root / "questions/sample_000001.json").write_text(
        json.dumps(
            {
                "schema": "avengine_qa_question_set_v1",
                "episode_id": "fixture",
                "items": [_question_item()],
            }
        ),
        encoding="utf-8",
    )
    catalog = {
        "status": "research_candidate",
        "records": [
            {
                "sample_id": "sample_000001",
                "group_id": "fixture-group",
                "member_id": "member-1",
                "facts_path": "../native/facts/facts_0001.json",
                "questions_path": "questions/sample_000001.json",
                "public_question_ids": ["public-1"],
            }
        ],
    }
    (catalog_root / "catalog_index.json").write_text(
        json.dumps(catalog), encoding="utf-8"
    )

    result = binding_delivery._validate_catalog_scoring(root, catalog)

    assert result["status"] == "pass"
    assert result["forms"]["mcq"]["correct"] == 1
    assert result["forms"]["open"]["correct"] == 1


def test_validate_core_shape_accepts_confined_relative_facts_link(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    (root / "core").mkdir(parents=True)
    (root / "native/facts").mkdir(parents=True)
    (root / "native/facts/facts_0001.json").write_text("{}", encoding="utf-8")
    core = {
        "groups": [
            {
                "group_id": "fixture-group",
                "members": [
                    {
                        "member_id": "member-1",
                        "facts_path": "../native/facts/facts_0001.json",
                        "media": {
                            "video_path": "media/video.mp4",
                            "audio_path": "media/audio.wav",
                        },
                    }
                ],
            }
        ]
    }
    # Shape validation only checks media path form; existence is checked by the
    # media validator, so synthetic media need not be encoded in this unit test.
    binding_delivery._validate_core_shape(core, root)


def test_dataset_index_configuration_is_validated() -> None:
    with pytest.raises(binding_delivery.BindingDeliveryError, match="unknown dataset index"):
        binding_delivery._dataset_index_config({"layouts": ["binaural"]})
    with pytest.raises(binding_delivery.BindingDeliveryError, match="must sum to 1.0"):
        binding_delivery._dataset_index_config(
            {"split_policy": {"mode": "by_world_key", "fractions": {"train": 0.5}}}
        )
    with pytest.raises(binding_delivery.BindingDeliveryError, match="non-empty unique list"):
        binding_delivery._dataset_index_config({"audio_layouts": ["binaural", "binaural"]})

    settings = binding_delivery._dataset_index_config({"split_policy": {"mode": "single"}})
    assert settings["split_policy"] == {"mode": "single"}


def test_world_keys_are_opaque_and_one_per_engine_world() -> None:
    keys = binding_delivery._world_keys(
        ["world_b", "world_a", "world_b"], prefix="world_", seed="fixed"
    )

    assert sorted(keys) == ["world_a", "world_b"]
    assert sorted(keys.values()) == ["world_0001", "world_0002"]
    assert keys == binding_delivery._world_keys(
        ["world_a", "world_b"], prefix="world_", seed="fixed"
    )


def test_splits_keep_one_world_inside_one_split() -> None:
    policy = {
        "mode": "by_world_key",
        "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "seed": "fixed",
    }
    assigned = binding_delivery._assign_splits([f"world_{i:04d}" for i in range(1, 11)], policy)

    assert len(assigned) == 10
    counts: dict[str, int] = {}
    for name in assigned.values():
        counts[name] = counts.get(name, 0) + 1
    assert sum(counts.values()) == 10
    assert counts["train"] == 8

    single = binding_delivery._assign_splits(["world_0001"], {"mode": "single", "name": "all"})
    assert single == {"world_0001": "all"}


def test_media_payload_index_collapses_identical_copies(tmp_path: Path) -> None:
    (tmp_path / "media").mkdir()
    (tmp_path / "media/a.mp4").write_bytes(b"same-visual-episode")
    (tmp_path / "media/b.mp4").write_bytes(b"same-visual-episode")
    (tmp_path / "media/c.mp4").write_bytes(b"different-episode--")

    index = binding_delivery._MediaPayloadIndex(tmp_path)
    assert index.canonical("media/a.mp4") == "media/a.mp4"
    assert index.canonical("media/b.mp4") == "media/a.mp4"
    assert index.canonical("media/c.mp4") == "media/c.mp4"

    report = index.report()
    assert report["distinct_payloads"] == 2
    assert report["redundant_delivered_files"] == 1
    assert report["redundant_to_canonical"] == {"media/b.mp4": "media/a.mp4"}


def test_core_free_join_accepts_only_episode_records(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog_index.json"
    catalog.write_text(
        json.dumps(
            {
                "status": "research_candidate",
                "records": [
                    {
                        "sample_id": "sample_000001",
                        "record_kind": "core_group_member",
                        "group_id": "group",
                        "member_id": "v0_a0",
                        "facts_path": "facts.json",
                        "questions_path": "questions/sample_000001.json",
                        "public_question_ids": ["question_000001"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(binding_delivery.BindingDeliveryError, match="only record_kind 'episode'"):
        binding_delivery._join_inputs(None, catalog)

    document = json.loads(catalog.read_text())
    document["records"][0].update(
        {"record_kind": "episode", "group_id": None, "member_id": None,
         "episode_id": "episode_a", "core_sample_id": None}
    )
    catalog.write_text(json.dumps(document), encoding="utf-8")

    core, _, joined = binding_delivery._join_inputs(None, catalog)
    assert core is None
    assert len(joined) == 1
    assert joined[0]["group_id"] is None
    assert joined[0]["episode_id"] == "episode_a"
    assert joined[0]["core_facts_path"] is None


def _attachment_skeleton(tmp_path: Path, *, content_sha: str) -> Path:
    """Minimal delivered package with one member and its research report."""

    root = tmp_path / "delivery"
    (root / "catalog").mkdir(parents=True)
    (root / "native/readbacks").mkdir(parents=True)
    (root / "provenance").mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"schema": binding_delivery.DELIVERY_SCHEMA, "status": "intermediate_delivery"}),
        encoding="utf-8",
    )
    (root / "catalog/catalog_index.json").write_text(
        json.dumps(
            {
                "status": "research_candidate",
                "records": [
                    {
                        "sample_id": "sample_000001",
                        "facts_path": "../native/facts/facts_0001.json",
                        "questions_path": "questions/sample_000001.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "native/facts").mkdir(parents=True)
    (root / "native/facts/facts_0001.json").write_text(
        json.dumps(
            {
                "episode_id": "episode_a",
                "audio": {"sample_count": 160000, "sample_rate_hz": 16000},
                "source_paths": {"research_report": "/producer/v0_a0/research_report.json"},
            }
        ),
        encoding="utf-8",
    )
    (root / "native/readbacks/readback_0001_research_report.json").write_text(
        json.dumps(
            {
                "audio_program_metadata": {
                    "program_id": "episode_a_audio_program",
                    "revision": "v1",
                    "program_content_sha256": content_sha,
                    "timeline": {"sample_count": 160000, "frame_count": 150},
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "provenance/path_map.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source_path": "/producer/v0_a0/research_report.json",
                        "delivered_path": "native/readbacks/readback_0001_research_report.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return root


def _layout_receipt(tmp_path: Path, *, content_sha: str, mixture: Path) -> Path:
    receipt = tmp_path / "research_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "audio_program_record": {
                    "program_id": "episode_a_audio_program",
                    "revision": "v1",
                    "program_content_sha256": content_sha,
                    "timeline": {"sample_count": 160000, "frame_count": 150},
                },
                "audio": {
                    "layout_delivery": {
                        "ambisonics": {
                            "layout_id": "rlr_foa_acn_n3d_world_v1",
                            "channel_count": 4,
                            "channel_labels": ["W", "Y", "Z", "X"],
                            "channel_order": "ACN",
                            "normalization": "N3D",
                            "coordinate_frame": "avengine_world",
                            "mixture": {"path": str(mixture), "sample_rate_hz": 16000},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return receipt


def test_attaching_another_members_render_is_refused(tmp_path: Path) -> None:
    root = _attachment_skeleton(tmp_path, content_sha="a" * 64)
    mixture = tmp_path / "foa_mixture.wav"
    mixture.write_bytes(b"RIFF----WAVEfake")
    receipt = _layout_receipt(tmp_path, content_sha="b" * 64, mixture=mixture)

    with pytest.raises(
        binding_delivery.BindingDeliveryError, match="not this member's own render"
    ) as error:
        binding_delivery.attach_audio_layout(
            root, sample_id="sample_000001", layout="ambisonics", receipt=receipt
        )
    assert "program_content_sha256" in str(error.value)
    assert not (root / "attachments").exists()


def test_attaching_a_matching_render_reads_the_declared_layout(tmp_path: Path) -> None:
    root = _attachment_skeleton(tmp_path, content_sha="a" * 64)
    mixture = tmp_path / "foa_mixture.wav"
    mixture.write_bytes(b"RIFF----WAVEfake")
    receipt = _layout_receipt(tmp_path, content_sha="a" * 64, mixture=mixture)

    attachment = binding_delivery.attach_audio_layout(
        root, sample_id="sample_000001", layout="ambisonics", receipt=receipt
    )

    assert attachment["attached_view_of"] == "binaural"
    assert attachment["declaration"]["normalization"] == "N3D"
    assert attachment["declaration"]["channel_order"] == "ACN"
    assert attachment["declaration"]["coordinate_frame"] == "avengine_world"
    delivered = root / attachment["delivered_path"]
    assert delivered.read_bytes() == mixture.read_bytes()


def test_an_undeclared_layout_is_not_attached(tmp_path: Path) -> None:
    root = _attachment_skeleton(tmp_path, content_sha="a" * 64)
    mixture = tmp_path / "foa_mixture.wav"
    mixture.write_bytes(b"RIFF----WAVEfake")
    receipt = _layout_receipt(tmp_path, content_sha="a" * 64, mixture=mixture)
    document = json.loads(receipt.read_text())
    document["audio"]["layout_delivery"]["ambisonics"].pop("normalization")
    receipt.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        binding_delivery.BindingDeliveryError,
        match=r"is incomplete: missing \['normalization'\]",
    ):
        binding_delivery.attach_audio_layout(
            root, sample_id="sample_000001", layout="ambisonics", receipt=receipt
        )


def test_a_mislabelled_layout_declaration_is_refused(tmp_path: Path) -> None:
    root = _attachment_skeleton(tmp_path, content_sha="a" * 64)
    mixture = tmp_path / "foa_mixture.wav"
    mixture.write_bytes(b"RIFF----WAVEfake")
    receipt = _layout_receipt(tmp_path, content_sha="a" * 64, mixture=mixture)
    document = json.loads(receipt.read_text())
    document["audio"]["layout_delivery"]["ambisonics"]["normalization"] = "SN3D"
    receipt.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        binding_delivery.BindingDeliveryError, match="disagrees with the runtime layout contract"
    ):
        binding_delivery.attach_audio_layout(
            root, sample_id="sample_000001", layout="ambisonics", receipt=receipt
        )
