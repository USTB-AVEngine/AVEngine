from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from avengine.studio.config import StudioConfigError, load_studio_config
from avengine.studio.server import StudioHTTPServer


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _init_git_repo(repo: Path) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", "main"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=studio-test",
            "-c",
            "user.email=studio-test@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "seed",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return head.stdout.strip()


def _write_capture(review_root: Path, name: str) -> None:
    capture = review_root / name
    capture.mkdir(parents=True)
    (capture / "frame_records.json").write_text("{}", encoding="utf-8")
    _write_json(
        (capture / "research_receipt.json"),
        {
            "status": "research_only",
            "research_only": True,
            "capture": {"frame_count": 75, "modalities": ["rgb"]},
            "selected_room": {"room_id": "habitat_mp3d_example_17DRP5sb8fy"},
        },
    )


@pytest.fixture()
def studio_config_path(tmp_path: Path) -> Path:
    from test_studio_scenes import make_bundle

    repo = tmp_path / "repo"
    main_sha = _init_git_repo(repo)

    static_root = repo / "tools/studio/static"
    static_root.mkdir(parents=True)
    (static_root / "studio.html").write_text("<html>studio</html>", encoding="utf-8")

    scenes_root = tmp_path / "scenes"
    make_bundle(scenes_root, "room_a")

    review_root = tmp_path / "review"
    _write_capture(review_root, f"room_visual_{main_sha[:7]}_ok")
    _write_capture(review_root, "room_visual_deadbee1_offmain")

    room_registry = _write_json(
        repo / "examples/rooms/room_registry.json",
        {
            "records": [
                {"room_id": "blender_custom_two_zone_v1", "provider_id": "blender_custom"},
                {"room_id": "habitat_mp3d_example_17DRP5sb8fy", "provider_id": "mp3d"},
            ]
        },
    )
    registries = {
        name: _write_json(
            repo / f"examples/registries/{name}.json", {"records": [{"id": name}]}
        )
        for name in ("source_endpoints", "sound_assets", "entity_assets")
    }

    # dummy inputs so template validation passes; the queued CLI run then fails
    inputs = {}
    for key in (
        "visual_capture_dir",
        "m1_request",
        "simulation_request",
        "package_manifest",
        "audio_program",
        "beagle_audio",
        "hrtf",
        "runtime_prefix",
        "rlr_sdk_root",
    ):
        if key in ("visual_capture_dir", "runtime_prefix", "rlr_sdk_root"):
            target = tmp_path / "inputs" / key
            target.mkdir(parents=True)
        else:
            target = _write_json(tmp_path / "inputs" / f"{key}.json", {})
        inputs[key] = str(target)
    inputs["source_endpoint_registry"] = str(registries["source_endpoints"])
    inputs["sound_asset_registry"] = str(registries["sound_assets"])

    return _write_json(
        tmp_path / "studio_config.json",
        {
            "schema": "avengine_studio_config_v1",
            "repository_root": str(repo),
            "python_executable": sys.executable,
            "review_root": str(review_root),
            "tasks_root": str(tmp_path / "tasks"),
            "scenes_root": str(scenes_root),
            "host": "127.0.0.1",
            "port": 0,
            "main_branch": "main",
            "room_registry": str(room_registry),
            "registries": {name: str(path) for name, path in registries.items()},
            "room_policy": {
                "banned_provider_ids": ["blender_custom"],
                "excluded_room_id_substrings": ["skokloster"],
            },
            "task_templates": {"mp3d_dynamic_audio": inputs},
        },
    )


@pytest.fixture()
def studio_server(studio_config_path: Path):
    config = load_studio_config(studio_config_path)
    server = StudioHTTPServer(config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _get_json(server: StudioHTTPServer, path: str) -> dict:
    port = server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(server: StudioHTTPServer, path: str, payload: dict):
    port = server.server_address[1]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_refuses_non_loopback_host(studio_config_path: Path) -> None:
    config = load_studio_config(studio_config_path)
    with pytest.raises(StudioConfigError):
        StudioHTTPServer(replace(config, host="0.0.0.0"))


def test_health_rooms_registries_and_captures(studio_server: StudioHTTPServer) -> None:
    health = _get_json(studio_server, "/api/health")
    assert health["status"] == "ok"
    assert health["main_commit_count"] == 1
    assert health["templates"] == ["mp3d_dynamic_audio"]

    rooms = _get_json(studio_server, "/api/rooms")
    statuses = {r["room_id"]: r["studio_status"] for r in rooms["records"]}
    assert statuses["blender_custom_two_zone_v1"] == "banned"
    assert statuses["habitat_mp3d_example_17DRP5sb8fy"] == "available"

    summaries = _get_json(studio_server, "/api/registries")
    assert set(summaries["registries"]) == {
        "source_endpoints",
        "sound_assets",
        "entity_assets",
    }
    passthrough = _get_json(studio_server, "/api/registries/sound_assets")
    assert passthrough == {"records": [{"id": "sound_assets"}]}

    captures = _get_json(studio_server, "/api/captures")
    names = [c["name"] for c in captures["captures"]]
    assert any("_ok" in name for name in names)
    assert not any("offmain" in name for name in names)
    assert captures["excluded_off_main"] == 1
    all_captures = _get_json(studio_server, "/api/captures?include=all")
    assert len(all_captures["captures"]) == 2


def test_submit_rejects_unknown_override_key(studio_server: StudioHTTPServer) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_json(
            studio_server,
            "/api/tasks",
            {"template": "mp3d_dynamic_audio", "overrides": {"hrtf": "/elsewhere"}},
        )
    assert excinfo.value.code == 400
    body = json.loads(excinfo.value.read().decode("utf-8"))
    assert "overrides not allowed" in body["error"]
    # a rejected submission leaves no task behind
    assert _get_json(studio_server, "/api/tasks")["tasks"] == []


def test_submit_queues_and_runs_cli_subprocess(studio_server: StudioHTTPServer) -> None:
    status, body = _post_json(
        studio_server,
        "/api/tasks",
        {"template": "mp3d_dynamic_audio", "overrides": {}},
    )
    assert status == 201
    task_id = body["task"]["task_id"]
    final = studio_server.task_queue.wait(task_id, timeout_s=120.0)
    # dummy inputs cannot satisfy the fail-closed CLI verb: the HTTP → queue →
    # subprocess chain is what this test verifies, not the render itself
    assert final == "fail"
    detail = _get_json(studio_server, f"/api/tasks/{task_id}")
    assert detail["task"]["status"] == "fail"
    argv = detail["task"]["argv"]
    assert argv[1:5] == ["-m", "avengine.cli", "m5", "render-current-mp3d-dynamic-audio"]

    port = studio_server.server_address[1]
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/api/tasks/{task_id}/log?tail=50"
    ) as response:
        log_text = response.read().decode("utf-8")
    assert log_text.strip()


def test_unknown_routes_and_tasks_return_404(studio_server: StudioHTTPServer) -> None:
    for path in ("/api/nope", "/api/tasks/does-not-exist"):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get_json(studio_server, path)
        assert excinfo.value.code == 404


def test_scene_endpoints_and_static(studio_server: StudioHTTPServer) -> None:
    scenes = _get_json(studio_server, "/api/scenes")
    assert [scene["room_id"] for scene in scenes["scenes"]] == ["room_a"]
    bundle = _get_json(studio_server, "/api/scenes/room_a/bundle.json")
    assert bundle["schema"] == "avengine_studio_scene_bundle_v1"

    port = studio_server.server_address[1]
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/api/scenes/room_a/files/mesh_positions.bin"
    ) as response:
        assert len(response.read()) == 4 * 3 * 4  # four float32 xyz vertices
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/scenes/room_a/files/secrets.txt"
        )
    assert excinfo.value.code == 404

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/studio") as response:
        assert b"studio" in response.read()


def test_validate_endpoint(studio_server: StudioHTTPServer) -> None:
    status, result = _post_json(
        studio_server,
        "/api/validate",
        {
            "room_id": "room_a",
            "points": [
                {"label": "good", "position_m": [0.75, 0.0, 0.75]},
                {"label": "far", "position_m": [50.0, 0.0, 50.0]},
            ],
        },
    )
    assert status == 200
    by_label = {record["label"]: record for record in result["points"]}
    assert by_label["good"]["ok"] is True
    assert by_label["far"]["ok"] is False
    assert result["all_ok"] is False


def test_artifact_endpoint_is_sandboxed(studio_server: StudioHTTPServer, tmp_path) -> None:
    import sys as _sys

    task_id = studio_server.task_queue.submit(
        template="echo",
        argv_builder=lambda output: [_sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
    )
    assert studio_server.task_queue.wait(task_id, timeout_s=30.0) == "pass"
    record = studio_server.task_queue.get(task_id)
    output_dir = Path(record["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "steps.json").write_text("{}", encoding="utf-8")

    port = studio_server.server_address[1]
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/api/tasks/{task_id}/artifact?path=steps.json"
    ) as response:
        assert response.read() == b"{}"
    with pytest.raises(urllib.error.HTTPError) as escape:
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/tasks/{task_id}/artifact"
            "?path=../../task.json"
        )
    assert escape.value.code == 400
