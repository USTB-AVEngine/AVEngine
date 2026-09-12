"""A0 integration: declared branches and CLI arguments reach their consumers."""
from contextlib import nullcontext
import json
from pathlib import Path
import pytest
from avengine import cli
from avengine.dataset.production_spec import QaTargetSpec


def test_target_branch_survives_serialization_and_legacy_absence():
    value = {"qa_id": "QA-06", "target_instance_ids": ["instance_a"], "branch": "moving"}
    target = QaTargetSpec.from_mapping(value, instance_ids=["instance_a"])
    saved = target.to_dict()
    assert saved["branch"] == "moving"
    assert QaTargetSpec.from_mapping(saved, instance_ids=["instance_a"]).branch == "moving"
    del value["branch"]
    assert "branch" not in QaTargetSpec.from_mapping(value, instance_ids=["instance_a"]).to_dict()


@pytest.mark.parametrize("normalization", ["native_n3d", "sn3d"])
def test_dynamic_audio_cli_forwards_foa_normalization(monkeypatch, tmp_path, normalization):
    seen = {}
    def render(**kwargs):
        seen.update(kwargs)
        return {"status": "research_candidate", "research_only": True,
                "rir": {"keyframe_count": 1}, "audio_program": {"event_count": 1},
                "audio": {"layouts": {"binaural": {}, "ambisonics": {}}}}
    monkeypatch.setattr(cli, "render_current_mp3d_dynamic_audio", render)
    monkeypatch.setattr(cli, "_require_ignored_or_external_output", lambda value: Path(value))
    monkeypatch.setattr(cli, "_temporary_native_audio_environment", lambda **kwargs: nullcontext())
    args = ["m5", "render-current-mp3d-dynamic-audio"]
    for option in ("visual-capture-dir", "m1-request", "simulation-request", "package-manifest",
                   "audio-program", "runtime-prefix", "rlr-sdk-root"):
        args += ["--" + option, "unused_by_native_stub"]
    args += ["--output", str(tmp_path / "fresh"), "--layouts", "binaural,ambisonics",
             "--foa-normalization", normalization]
    assert cli.main(args) == 0
    assert seen["foa_normalization"] == normalization
    assert tuple(seen["layouts"]) == ("binaural", "ambisonics")


def test_dataset_export_cli_calls_public_api(monkeypatch, tmp_path, capsys):
    from avengine.qa import binding_delivery
    seen = {}
    def export(core, catalog, output):
        seen.update(core=core, catalog=catalog, output=output)
        return {"status": "pass"}
    monkeypatch.setattr(binding_delivery, "export_binding_delivery", export)
    monkeypatch.setattr(binding_delivery, "build_dataset_index", lambda output, config: {"status": "pass"})
    assert cli.main(["dataset", "export-delivery", "--catalog-index", str(tmp_path / "catalog.json"),
                     "--output", str(tmp_path / "delivery"), "--build-index"]) == 0
    assert seen["core"] is None
    assert seen["catalog"] == tmp_path / "catalog.json"
    assert json.loads(capsys.readouterr().out)["dataset_index"]["status"] == "pass"


def test_dataset_output_does_not_overwrite_an_existing_result(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("retained")
    with pytest.raises(FileExistsError):
        cli._dataset_result({"replacement": True}, output)
    assert output.read_text() == "retained"


def test_dataset_media_api_paths_are_json_serializable(monkeypatch, tmp_path, capsys):
    from avengine.dataset import qa_dataset_reader
    class Reader:
        def media(self, sample):
            assert sample == "sample_a"
            return {"audio": {"path": tmp_path / "audio.wav"}}
    monkeypatch.setattr(qa_dataset_reader, "open_qa_dataset", lambda root, config: Reader())
    assert cli.main(["dataset", "inspect", "media", "--root", str(tmp_path), "--sample", "sample_a"]) == 0
    assert json.loads(capsys.readouterr().out)["audio"]["path"] == str(tmp_path / "audio.wav")
