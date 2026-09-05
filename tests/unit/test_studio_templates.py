"""Template argv builders: room-family coverage and the submit-time contract.

These test the builder layer directly rather than through the HTTP server,
because the properties that matter are the builder's own: paths are validated
at submit time so a bad request is an HTTP 400 rather than a queued corpse,
the output is fresh, the override policy is closed, and the argv targets the
tool each template claims to wrap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avengine.studio.config import load_studio_config
from avengine.studio.templates import StudioTemplateError, build_template_argv

REPOSITORY = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path, task_templates: dict) -> object:
    for name in ("review", "tasks", "scenes"):
        (tmp_path / name).mkdir(exist_ok=True)
    room_registry = tmp_path / "room_registry.json"
    room_registry.write_text(json.dumps({"schema": "any", "rooms": []}))
    registries = {}
    for name in ("source_endpoints", "sound_assets", "entity_assets"):
        target = tmp_path / f"{name}.json"
        target.write_text("{}")
        registries[name] = str(target)
    config_path = tmp_path / "studio_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "avengine_studio_config_v1",
                "repository_root": str(REPOSITORY),
                "python_executable": "/usr/bin/python3",
                "review_root": str(tmp_path / "review"),
                "tasks_root": str(tmp_path / "tasks"),
                "scenes_root": str(tmp_path / "scenes"),
                "host": "127.0.0.1",
                "port": 0,
                "main_branch": "main",
                "room_registry": str(room_registry),
                "registries": registries,
                "task_templates": task_templates,
            }
        )
    )
    return load_studio_config(config_path)


def _hm3d_inputs(tmp_path: Path) -> dict:
    inputs = tmp_path / "inputs"
    values: dict[str, str] = {}
    for key in ("runtime_prefix", "magnum_site", "rlr_sdk_root", "hm3d_root",
                "scene_dir", "bank", "asset_dir"):
        target = inputs / key
        target.mkdir(parents=True)
        values[key] = str(target)
    for key in ("scene", "navmesh", "dataset_config", "materials_json",
                "hrtf", "room_manifest", "material_rules", "scene_metadata",
                "uproject", "unreal_editor", "audio_python"):
        target = inputs / f"{key}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
        values[key] = str(target)
    episode_root = inputs / "episode_root"
    episode_root.mkdir()
    values["episode_root"] = str(episode_root)
    return values


def test_hm3d_room_prepare_builds_and_rejects_a_bad_split(tmp_path: Path) -> None:
    values = _hm3d_inputs(tmp_path)
    defaults = {
        key: values[key]
        for key in ("runtime_prefix", "magnum_site", "rlr_sdk_root",
                    "hm3d_root", "scene_dir")
    }
    config = _config(tmp_path, {"hm3d_room_prepare": defaults})
    argv = build_template_argv(
        config, "hm3d_room_prepare", {"split": "train"}, tmp_path / "out"
    )
    assert argv[1].endswith("tools/rooms/emit_hm3d_room_manifest.py")
    assert argv[argv.index("--split") + 1] == "train"
    assert "--connectivity-samples" in argv

    with pytest.raises(StudioTemplateError, match="split must be one of"):
        build_template_argv(
            config, "hm3d_room_prepare", {"split": "weekend"}, tmp_path / "out2"
        )


def test_semantic_package_serves_hm3d_and_mp3d_rooms_alike(tmp_path: Path) -> None:
    """One template covers both datasets; the room manifest carries identity."""

    values = _hm3d_inputs(tmp_path)
    config = _config(
        tmp_path,
        {
            "semantic_acoustic_package": {
                "room_manifest": values["room_manifest"],
                "material_rules": values["material_rules"],
            }
        },
    )
    other_room = tmp_path / "other_room_manifest.json"
    other_room.write_text("{}")
    argv = build_template_argv(
        config,
        "semantic_acoustic_package",
        {"room_manifest": str(other_room), "seed": 7, "package_id": "abc"},
        tmp_path / "out",
    )
    assert argv[1].endswith("tools/acoustics/compile_semantic_research_package.py")
    assert argv[argv.index("--room-manifest") + 1] == str(other_room.resolve())
    assert argv[argv.index("--seed") + 1] == "7"
    assert argv[argv.index("--package-id") + 1] == "abc"


def test_hm3d_route_bank_groups_its_three_outputs_under_one_fresh_dir(
    tmp_path: Path,
) -> None:
    values = _hm3d_inputs(tmp_path)
    defaults = {
        key: values[key]
        for key in ("runtime_prefix", "magnum_site", "rlr_sdk_root",
                    "scene", "navmesh")
    }
    config = _config(tmp_path, {"hm3d_route_bank": defaults})
    out = tmp_path / "out"
    argv = build_template_argv(config, "hm3d_route_bank", {}, out)
    resolved = str(out.resolve())
    assert argv[argv.index("--bank-dir") + 1] == f"{resolved}/bank"
    assert argv[argv.index("--topdown-dir") + 1] == f"{resolved}/topdown"
    assert argv[argv.index("--report") + 1] == f"{resolved}/route_report.json"

    out.mkdir()
    with pytest.raises(StudioTemplateError, match="fresh"):
        build_template_argv(config, "hm3d_route_bank", {}, out)


def test_hm3d_episode_flags_follow_the_overrides(tmp_path: Path) -> None:
    values = _hm3d_inputs(tmp_path)
    defaults = {
        key: values[key]
        for key in ("runtime_prefix", "magnum_site", "rlr_sdk_root", "bank",
                    "asset_dir", "dataset_config", "materials_json", "hrtf",
                    "audio_python")
    }
    defaults["scene_id"] = "TEEsavR23oF"
    config = _config(tmp_path, {"hm3d_episode": defaults})
    argv = build_template_argv(
        config,
        "hm3d_episode",
        {
            "episode_id": "ep_003",
            "aim_open": True,
            "place_at_emitter": True,
            "overhead_m": 2.5,
        },
        tmp_path / "out",
    )
    assert argv[1].endswith("tools/studio/run_hm3d_episode.py")
    assert "--aim-open" in argv and "--place-at-emitter" in argv
    assert argv[argv.index("--episode-id") + 1] == "ep_003"
    assert "--episode-index" not in argv
    assert argv[argv.index("--overhead-m") + 1] == "2.5"

    plain = build_template_argv(config, "hm3d_episode", {}, tmp_path / "out2")
    assert "--aim-open" not in plain and "--episode-index" in plain


def test_kujiale_templates_build_against_their_own_tools(tmp_path: Path) -> None:
    values = _hm3d_inputs(tmp_path)
    config = _config(
        tmp_path,
        {
            "kujiale_route_bank": {"scene_metadata": values["scene_metadata"]},
            "kujiale_visual_episode": {
                "episode_root": values["episode_root"],
                "uproject": values["uproject"],
                "unreal_editor": values["unreal_editor"],
            },
        },
    )
    bank = build_template_argv(config, "kujiale_route_bank", {}, tmp_path / "b")
    assert bank[1].endswith("tools/routes/compile_kujiale_feasibility_bank.py")
    assert "--minimum-clearance-m" not in bank

    episode = build_template_argv(
        config, "kujiale_visual_episode", {"rpc_port": 39380}, tmp_path / "e"
    )
    assert episode[1].endswith("tools/rooms/run_spear_residential_episode.py")
    assert episode[episode.index("--rpc-port") + 1] == "39380"


def test_a_missing_input_path_fails_at_submit_not_in_the_queue(
    tmp_path: Path,
) -> None:
    values = _hm3d_inputs(tmp_path)
    config = _config(
        tmp_path,
        {
            "semantic_acoustic_package": {
                "room_manifest": str(tmp_path / "inputs" / "gone.json"),
                "material_rules": values["material_rules"],
            }
        },
    )
    with pytest.raises(StudioTemplateError, match="does not exist"):
        build_template_argv(
            config, "semantic_acoustic_package", {}, tmp_path / "out"
        )


def test_mp3d_room_identity_is_overridable_for_other_rooms(tmp_path: Path) -> None:
    """The MP3D templates must reach rooms beyond the configured default."""

    from avengine.studio.templates import TEMPLATE_OVERRIDABLE_KEYS

    for template in ("mp3d_route_author", "mp3d_end_to_end"):
        keys = TEMPLATE_OVERRIDABLE_KEYS[template]
        assert {"room_manifest", "package_manifest", "mp3d_root"} <= keys
    assert {"m1_request", "package_manifest"} <= TEMPLATE_OVERRIDABLE_KEYS[
        "mp3d_dynamic_audio"
    ]


def test_hm3d_end_to_end_leaves_only_the_house_to_choose(tmp_path: Path) -> None:
    """The one-click chain: paths validated, output fresh, and the room
    decision closed off - the chain itself picks the room."""

    values = _hm3d_inputs(tmp_path)
    defaults = {
        key: values[key]
        for key in (
            "scene_dir", "hm3d_root", "runtime_prefix", "magnum_site",
            "rlr_sdk_root", "material_rules", "audio_python", "asset_dir",
            "dataset_config", "materials_json", "hrtf",
        )
    }
    defaults["split"] = "val"
    train_dataset_config = tmp_path / "hm3d_all.scene_dataset_config.json"
    train_dataset_config.write_text("{}", encoding="utf-8")
    defaults["dataset_config_by_split"] = {"train": str(train_dataset_config)}
    config = _config(tmp_path, {"hm3d_end_to_end": defaults})
    argv = build_template_argv(config, "hm3d_end_to_end", {}, tmp_path / "out")
    assert argv[1].endswith("tools/studio/run_hm3d_end_to_end.py")
    assert "--split" in argv and "val" in argv
    assert str((tmp_path / "out").resolve()) in argv
    assert argv[argv.index("--dataset-config") + 1] != str(train_dataset_config)
    with pytest.raises(StudioTemplateError, match="overrides not allowed"):
        build_template_argv(
            config, "hm3d_end_to_end", {"room_bounds": [0, 0, 1, 1]},
            tmp_path / "out2",
        )
    with pytest.raises(StudioTemplateError, match="split must be one of"):
        build_template_argv(
            config, "hm3d_end_to_end", {"split": "vale"}, tmp_path / "out3"
        )

    train_argv = build_template_argv(
        config,
        "hm3d_end_to_end",
        {"split": "train"},
        tmp_path / "train-out",
    )
    assert train_argv[train_argv.index("--dataset-config") + 1] == str(
        train_dataset_config.resolve()
    )



def test_hm3d_clock_overrides_propagate_and_reject_mismatched_duration(
    tmp_path: Path,
) -> None:
    values = _hm3d_inputs(tmp_path)
    defaults = {
        key: values[key]
        for key in (
            "scene_dir", "hm3d_root", "runtime_prefix", "magnum_site",
            "rlr_sdk_root", "material_rules", "audio_python", "asset_dir",
            "dataset_config", "materials_json", "hrtf",
        )
    }
    defaults["split"] = "val"
    config = _config(tmp_path, {"hm3d_end_to_end": defaults})
    argv = build_template_argv(
        config,
        "hm3d_end_to_end",
        {
            "frame_count": 150,
            "frame_rate_hz": 15,
            "sample_rate_hz": 16_000,
            "clip_seconds": 10,
        },
        tmp_path / "out",
    )
    assert argv[argv.index("--frame-count") + 1] == "150"
    assert argv[argv.index("--frame-rate-hz") + 1] == "15"
    assert argv[argv.index("--sample-rate") + 1] == "16000"
    assert argv[argv.index("--clip-seconds") + 1] == "10"

    with pytest.raises(StudioTemplateError, match="clip_seconds"):
        build_template_argv(
            config,
            "hm3d_end_to_end",
            {"frame_count": 150, "frame_rate_hz": 15, "clip_seconds": 5},
            tmp_path / "out_bad",
        )
