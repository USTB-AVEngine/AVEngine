"""Loopback-only stdlib HTTP server for the Studio backend.

GET  /                        minimal status page
GET  /studio                  3D scene editor (single-page, vendored three.js)
GET  /studio/static/<name>    editor assets from tools/studio/static/
GET  /api/health
GET  /api/rooms               room registry with studio_status annotations
GET  /api/registries          registry summaries
GET  /api/registries/<name>   full registry passthrough
GET  /api/captures            review base captures (?include=all keeps
                              captures whose commit is known off main)
GET  /api/scenes              available 3D scene bundles
GET  /api/scenes/<id>/bundle.json
GET  /api/scenes/<id>/files/<name>   raw mesh buffers
POST /api/validate            draft placement check {room_id, points}
GET  /api/templates           task templates, defaults, overridable keys
GET  /api/tasks               task list (newest first)
POST /api/tasks               {"template": ..., "overrides": {...}}
GET  /api/tasks/<id>
GET  /api/tasks/<id>/log      text tail (?tail=N)
GET  /api/tasks/<id>/artifact?path=<rel>   file from the task directory
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from avengine.studio.catalog import (
    annotate_room_registry,
    list_review_captures,
    load_registry_json,
    main_branch_commit_shas,
)
from avengine.studio.config import LOOPBACK_HOSTS, StudioConfig, StudioConfigError
from avengine.studio.scenes import (
    StudioSceneError,
    list_scene_bundles,
    load_draft_obstacle_grid,
    load_scene_bundle,
    scene_dataset_file_path,
    scene_file_path,
)
from avengine.studio.production import (
    ProductionError,
    board_rows,
    load_room_curation,
    load_sound_asset_catalog,
    load_sound_library,
    review_queue,
    sound_asset_file,
    sound_library_file,
    write_human_verdict,
    write_room_verdict,
)
from avengine.studio.tasks import StudioTaskError, StudioTaskQueue
from avengine.studio.templates import (
    TEMPLATE_OVERRIDABLE_KEYS,
    StudioTemplateError,
    build_template_argv,
)
from avengine.studio.validation import check_points

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".bin": "application/octet-stream",
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".png": "image/png",
    ".log": "text/plain; charset=utf-8",
    ".npy": "application/octet-stream",
    ".glb": "model/gltf-binary",
}

# artifact files a task may expose to the browser
_ARTIFACT_SUFFIXES = frozenset({".mp4", ".json", ".wav", ".png", ".log"})

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

_STATUS_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>AVEngine Studio</title>
<style>
body{font-family:system-ui,'PingFang SC',sans-serif;margin:2rem;max-width:70rem;background:#fbfaf7;color:#2c2c2a}
h1{font-size:1.3rem} h2{font-size:1.05rem;margin-top:1.6rem}
table{border-collapse:collapse;width:100%;font-size:.85rem}
td,th{border:1px solid #d8d5cc;padding:.3rem .5rem;text-align:left}
code{background:#f1efe8;padding:.1rem .3rem;border-radius:3px}
a.big{display:inline-block;margin:.6rem 0;padding:.5rem 1rem;background:#534ab7;color:#fff;border-radius:6px;text-decoration:none}
.badge{padding:.05rem .4rem;border-radius:3px;font-size:.78rem}
.pass{background:#e1f5ee}.fail{background:#fae7e7}.running{background:#fdf3d8}.queued{background:#eeedfe}
</style></head><body>
<h1>AVEngine Studio</h1>
<p>
<a class="big" href="/studio/submit">提交任务</a>
<a class="big" href="/studio">3D 场景编辑器</a>
<a class="big" href="/studio/assets">声源资产台</a>
<a class="big" href="/studio/sounds">声音素材库</a>
<a class="big" href="/studio/board">进度看板</a>
<a class="big" href="/studio/review">音视频验收台</a>
<a class="big" href="/studio/rooms">房间初筛台</a>
</p>
<h2>任务</h2><div id="tasks">加载中…</div>
<script>
async function refresh(){
  const tasks=await (await fetch('/api/tasks')).json();
  document.getElementById('tasks').innerHTML=
    '<table><tr><th>task</th><th>模板</th><th>状态</th><th>创建</th><th>结束</th><th>日志</th></tr>'+
    tasks.tasks.map(t=>`<tr><td>${t.task_id}</td><td>${t.template}</td>`+
      `<td><span class="badge ${t.status}">${t.status}</span></td>`+
      `<td>${t.created_at||''}</td><td>${t.finished_at||''}</td>`+
      `<td><a href="/api/tasks/${t.task_id}/log?tail=100">tail</a></td></tr>`).join('')+'</table>';
}
refresh(); setInterval(refresh, 5000);
</script></body></html>
"""


class StudioHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: StudioConfig) -> None:
        if config.host not in LOOPBACK_HOSTS:
            raise StudioConfigError(
                f"refusing to bind non-loopback host {config.host!r}; "
                "preview through an ssh -L tunnel"
            )
        self.studio_config = config
        self.task_queue = StudioTaskQueue(config.tasks_root)
        self.main_commit_shas = main_branch_commit_shas(
            config.repository_root, branch=config.main_branch
        )
        self.static_root = (config.repository_root / "tools/studio/static").resolve()
        self._grid_cache: dict[str, object] = {}
        super().__init__((config.host, config.port), StudioRequestHandler)

    def draft_grid(self, room_id: str):
        if room_id not in self._grid_cache:
            bundle = load_scene_bundle(self.studio_config.scenes_root, room_id)
            self._grid_cache[room_id] = load_draft_obstacle_grid(bundle)
        return self._grid_cache[room_id]


class StudioRequestHandler(BaseHTTPRequestHandler):
    server_version = "AVEngineStudio/0.2"
    protocol_version = "HTTP/1.1"

    # -- transport helpers ---------------------------------------------------

    # big immutable-per-version binaries get long caching; everything else
    # (pages, scripts, JSON) must always revalidate so edits show up
    _LONG_CACHE_SUFFIXES = frozenset({".bin", ".glb", ".png", ".mp4", ".wav"})

    def _send_payload(
        self,
        body: bytes,
        content_type: str,
        status: int,
        *,
        cache_control: str = "no-cache",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: object, *, status: int = 200) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self._send_payload(body, "application/json; charset=utf-8", status)

    def _send_text(self, text: str, *, status: int = 200) -> None:
        self._send_payload(text.encode("utf-8"), "text/plain; charset=utf-8", status)

    def _send_error_json(self, message: str, *, status: int) -> None:
        self._send_json({"error": message}, status=status)

    def _send_file(self, path: Path, *, status: int = 200) -> None:
        content_type = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        body = path.read_bytes()
        cache = (
            "max-age=604800"
            if path.suffix in self._LONG_CACHE_SUFFIXES
            else "no-cache"
        )
        self._send_payload(body, content_type, status, cache_control=cache)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise StudioTemplateError("request body must be a JSON object")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise StudioTemplateError("request body must be a JSON object")
        return payload

    # -- routing ---------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        self._route("POST")

    def _route(self, method: str) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if method == "GET" and path == "/":
                # The owner's whole interface: one page, one click per house.
                # The navigation hub with every workstation lives on at
                # /studio/hub for the classmates who operate the stages.
                self._handle_static("home.html")
            elif method == "GET" and path == "/studio/hub":
                self._send_payload(
                    _STATUS_PAGE.encode("utf-8"), "text/html; charset=utf-8", 200
                )
            elif method == "GET" and path == "/studio":
                self._handle_static("studio.html")
            elif method == "GET" and path == "/studio/assets":
                self._handle_static("assets.html")
            elif method == "GET" and path == "/studio/submit":
                self._handle_static("submit.html")
            elif method == "GET" and path == "/studio/sounds":
                self._handle_static("sounds.html")
            elif method == "GET" and path == "/studio/board":
                self._handle_static("board.html")
            elif method == "GET" and path == "/studio/review":
                self._handle_static("review.html")
            elif method == "GET" and path == "/studio/rooms":
                self._handle_static("rooms.html")
            elif method == "GET" and path == "/api/room-curation":
                self._handle_room_curation()
            elif method == "POST" and path == "/api/room-curation/verdict":
                self._handle_room_curation_verdict()
            elif method == "GET" and path.startswith("/studio/static/"):
                self._handle_static(path.removeprefix("/studio/static/"))
            elif method == "GET" and path == "/api/health":
                self._handle_health()
            elif method == "GET" and path == "/api/rooms":
                self._handle_rooms()
            elif method == "GET" and path == "/api/registries":
                self._handle_registry_summaries()
            elif method == "GET" and path.startswith("/api/registries/"):
                self._handle_registry(path.removeprefix("/api/registries/"))
            elif method == "GET" and path == "/api/captures":
                self._handle_captures(query)
            elif method == "GET" and path == "/api/scenes":
                self._handle_scenes()
            elif method == "GET" and path.startswith("/api/scenes/"):
                self._handle_scene_file(path.removeprefix("/api/scenes/"))
            elif method == "POST" and path == "/api/validate":
                self._handle_validate()
            elif method == "GET" and path == "/api/sounds":
                self._handle_sound_list()
            elif method == "GET" and path.startswith("/api/sounds/") and path.endswith("/audio"):
                sound_id = path.removeprefix("/api/sounds/").removesuffix("/audio").strip("/")
                self._handle_sound_audio(sound_id)
            elif method == "POST" and path == "/api/programs":
                self._handle_program_author()
            elif method == "GET" and path == "/api/templates":
                self._handle_templates()
            elif method == "GET" and path == "/api/sound-assets":
                self._handle_sound_asset_catalog()
            elif method == "GET" and path == "/api/sound-assets/file":
                self._handle_sound_asset_file(query)
            elif method == "GET" and path == "/api/hm3d-scenes":
                self._handle_hm3d_scenes()
            elif method == "GET" and path == "/api/sound-library":
                self._handle_sound_library()
            elif method == "GET" and path == "/api/sound-library/file":
                self._handle_sound_library_file(query)
            elif method == "GET" and path == "/api/board":
                self._handle_board()
            elif method == "GET" and path == "/api/review-queue":
                self._handle_review_queue()
            elif method == "POST" and path == "/api/sweeps":
                self._handle_sweep_submit()
            elif method == "GET" and path == "/api/sweeps":
                self._handle_sweep_list()
            elif method == "GET" and path == "/api/tasks":
                self._handle_task_list()
            elif method == "POST" and path == "/api/tasks":
                self._handle_task_submit()
            elif method == "GET" and path.startswith("/api/tasks/") and path.endswith("/log"):
                task_id = path.removeprefix("/api/tasks/").removesuffix("/log").strip("/")
                self._handle_task_log(task_id, query)
            elif (
                method == "POST"
                and path.startswith("/api/tasks/")
                and path.endswith("/verdict")
            ):
                task_id = (
                    path.removeprefix("/api/tasks/").removesuffix("/verdict").strip("/")
                )
                self._handle_task_verdict(task_id)
            elif (
                method == "GET"
                and path.startswith("/api/tasks/")
                and path.endswith("/artifact")
            ):
                task_id = (
                    path.removeprefix("/api/tasks/").removesuffix("/artifact").strip("/")
                )
                self._handle_task_artifact(task_id, query)
            elif method == "GET" and path.startswith("/api/tasks/"):
                self._handle_task_detail(path.removeprefix("/api/tasks/"))
            else:
                self._send_error_json(f"no route for {method} {path}", status=404)
        except (StudioTaskError, StudioSceneError) as exc:
            self._send_error_json(str(exc), status=404)
        except (
            StudioTemplateError,
            StudioConfigError,
            ProductionError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            self._send_error_json(str(exc), status=400)
        except BrokenPipeError:
            pass
        except Exception as exc:  # surface, never crash the connection thread
            self._send_error_json(f"{type(exc).__name__}: {exc}", status=500)

    # -- handlers ----------------------------------------------------------------

    def _server(self) -> StudioHTTPServer:
        return self.server  # type: ignore[return-value]

    def _handle_static(self, name: str) -> None:
        if not _SAFE_NAME.match(name) or Path(name).suffix not in (".html", ".js", ".css"):
            self._send_error_json(f"not a servable static file: {name}", status=404)
            return
        path = self._server().static_root / name
        if not path.is_file():
            self._send_error_json(f"static file not found: {name}", status=404)
            return
        self._send_file(path)

    def _handle_health(self) -> None:
        server = self._server()
        config = server.studio_config
        self._send_json(
            {
                "status": "ok",
                "repository_root": str(config.repository_root),
                "review_root": str(config.review_root),
                "tasks_root": str(config.tasks_root),
                "scenes_root": str(config.scenes_root) if config.scenes_root else None,
                "main_branch": config.main_branch,
                "main_commit_count": (
                    len(server.main_commit_shas)
                    if server.main_commit_shas is not None
                    else None
                ),
                "task_count": len(server.task_queue.list_tasks()),
                "templates": sorted(config.task_templates),
            }
        )

    def _handle_rooms(self) -> None:
        config = self._server().studio_config
        registry = load_registry_json(config.room_registry_path)
        self._send_json(annotate_room_registry(registry, config.room_policy))

    def _handle_registry_summaries(self) -> None:
        config = self._server().studio_config
        summaries = {}
        for name, path in sorted(config.registry_paths.items()):
            registry = load_registry_json(path)
            record_key, record_count = None, None
            for key, value in registry.items():
                if isinstance(value, list):
                    record_key, record_count = key, len(value)
                    break
            summaries[name] = {
                "path": str(path),
                "schema": registry.get("schema"),
                "registry_id": registry.get("registry_id"),
                "record_key": record_key,
                "record_count": record_count,
            }
        self._send_json({"registries": summaries})

    def _handle_registry(self, name: str) -> None:
        config = self._server().studio_config
        path = config.registry_paths.get(name)
        if path is None:
            self._send_error_json(
                f"unknown registry {name!r}; available: {sorted(config.registry_paths)}",
                status=404,
            )
            return
        self._send_json(load_registry_json(path))

    def _handle_captures(self, query: dict[str, list[str]]) -> None:
        server = self._server()
        entries = list_review_captures(
            server.studio_config.review_root, server.main_commit_shas
        )
        include_all = query.get("include", [""])[0] == "all"
        if include_all:
            captures = entries
        else:
            captures = [e for e in entries if e["commit_on_main"] is not False]
        self._send_json(
            {
                "captures": captures,
                "total_scanned": len(entries),
                "excluded_off_main": len(entries) - len(captures) if not include_all else 0,
                "note": (
                    "default listing excludes captures whose commit is known to be "
                    "off the main branch; pass ?include=all to see them"
                ),
            }
        )

    def _handle_scenes(self) -> None:
        config = self._server().studio_config
        if config.scenes_root is None:
            self._send_json({"scenes": [], "note": "no scenes_root configured"})
            return
        self._send_json({"scenes": list_scene_bundles(config.scenes_root)})

    def _handle_scene_file(self, remainder: str) -> None:
        config = self._server().studio_config
        if config.scenes_root is None:
            raise StudioSceneError("no scenes_root configured")
        parts = remainder.split("/")
        if len(parts) == 2 and parts[1] == "bundle.json":
            room_id, file_name = parts[0], "bundle.json"
        elif len(parts) == 3 and parts[1] == "files":
            room_id, file_name = parts[0], parts[2]
        elif len(parts) >= 3 and parts[1] == "dataset":
            room_id = parts[0]
            relative = "/".join(parts[2:])
            if not _SAFE_NAME.match(room_id) or not all(
                _SAFE_NAME.match(part) for part in parts[2:]
            ):
                raise StudioSceneError("invalid scene path")
            self._send_file(
                scene_dataset_file_path(config.scenes_root, room_id, relative)
            )
            return
        else:
            raise StudioSceneError(f"no scene route for {remainder!r}")
        if not _SAFE_NAME.match(room_id) or not _SAFE_NAME.match(file_name):
            raise StudioSceneError("invalid scene path")
        self._send_file(scene_file_path(config.scenes_root, room_id, file_name))

    def _handle_validate(self) -> None:
        server = self._server()
        body = self._read_json_body()
        room_id = str(body.get("room_id") or "")
        points = body.get("points")
        if not isinstance(points, list) or not points:
            raise StudioTemplateError("points must be a non-empty list")
        grid = server.draft_grid(room_id)
        if grid is None:
            self._send_json(
                {
                    "claim": "no draft obstacle snapshot for this scene; the "
                    "authoring verbs validate on submission",
                    "all_ok": None,
                    "points": [],
                }
            )
            return
        minimum = float(body.get("minimum_rigid_clearance_m", 0.0))
        self._send_json(
            check_points(grid, points, minimum_rigid_clearance_m=minimum)
        )

    def _sound_registry_records(self) -> dict[str, dict]:
        config = self._server().studio_config
        registry_path = config.registry_paths.get("sound_assets")
        if registry_path is None:
            raise StudioTemplateError("no sound_assets registry configured")
        registry = load_registry_json(registry_path)
        return {record["sound_asset_id"]: record for record in registry["sound_assets"]}

    def _handle_sound_list(self) -> None:
        from avengine.studio.programs import resolve_dry_audio_path

        config = self._server().studio_config
        sounds = []
        for sound_id, record in sorted(self._sound_registry_records().items()):
            dry = record.get("dry_audio", {})
            entry = {
                "sound_asset_id": sound_id,
                "semantic_sound_class": record.get("semantic_sound_class"),
                "uri": dry.get("uri"),
                "sample_count": dry.get("sample_count"),
                "duration_s": (
                    round(dry["sample_count"] / dry["sample_rate_hz"], 2)
                    if dry.get("sample_count") and dry.get("sample_rate_hz")
                    else None
                ),
                "rights_status": record.get("provenance", {}).get("rights_status"),
                "permitted_event_usage": record.get("permitted_event_usage"),
            }
            try:
                entry["resolved_path"] = str(
                    resolve_dry_audio_path(
                        record,
                        config.repository_root,
                        external_paths=config.external_sound_assets,
                    )
                )
                entry["available"] = True
            except Exception as exc:  # noqa: BLE001 - listing stays best-effort
                entry["available"] = False
                entry["error"] = str(exc)
            sounds.append(entry)
        self._send_json({"sounds": sounds, "note": "音频按需经 /api/sounds/<id>/audio 懒加载"})

    def _handle_sound_audio(self, sound_id: str) -> None:
        from avengine.studio.programs import resolve_dry_audio_path

        config = self._server().studio_config
        record = self._sound_registry_records().get(sound_id)
        if record is None:
            raise StudioTaskError(f"unknown sound asset: {sound_id}")
        self._send_file(
            resolve_dry_audio_path(
                record,
                config.repository_root,
                external_paths=config.external_sound_assets,
            )
        )

    def _handle_program_author(self) -> None:
        from datetime import datetime, timezone

        from avengine.studio.programs import (
            StudioProgramError,
            build_turn_taking_program,
            persist_program,
        )

        server = self._server()
        config = server.studio_config
        body = self._read_json_body()
        candidates = body.get("candidates")
        sound_by_endpoint = body.get("sounds")
        if not isinstance(candidates, list) or not candidates:
            raise StudioTemplateError("candidates must be a non-empty list")
        if not isinstance(sound_by_endpoint, dict):
            raise StudioTemplateError("sounds must be an object endpoint→sound_asset_id")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        program_id = f"studio_turn_taking_{stamp}"
        try:
            program = build_turn_taking_program(
                program_id=program_id,
                candidate_source_endpoint_ids=[str(item) for item in candidates],
                sound_by_endpoint={str(k): str(v) for k, v in sound_by_endpoint.items()},
                source_endpoint_registry_path=config.registry_paths["source_endpoints"],
                sound_asset_registry_path=config.registry_paths["sound_assets"],
                repository_root=config.repository_root,
                external_sound_asset_paths=config.external_sound_assets,
                event_count=int(body.get("event_count", 6)),
                event_samples=int(body.get("event_samples", 8000)),
                linear_gain=float(body.get("linear_gain", 0.2)),
            )
            path = persist_program(program, config.tasks_root / "_programs")
        except StudioProgramError as exc:
            raise StudioTemplateError(str(exc)) from exc
        self._send_json(
            {
                "program_id": program["program_id"],
                "program_path": str(path),
                "program_content_sha256": program["program_content_sha256"],
                "event_count": len(program["events"]),
                "admission_state": program["admission_state"],
            },
            status=201,
        )

    def _handle_templates(self) -> None:
        config = self._server().studio_config
        templates = {}
        for name, defaults in sorted(config.task_templates.items()):
            templates[name] = {
                "defaults": defaults,
                "overridable_keys": sorted(TEMPLATE_OVERRIDABLE_KEYS.get(name, ())),
            }
        self._send_json({"templates": templates})

    def _task_view(self, record: dict) -> dict:
        view = dict(record)
        receipt = Path(record["output_dir"]) / "research_receipt.json"
        view["receipt_path"] = str(receipt) if receipt.is_file() else None
        artifacts = []
        output_dir = Path(record["output_dir"])
        if output_dir.is_dir():
            for path in sorted(output_dir.rglob("*")):
                if path.is_file() and path.suffix in _ARTIFACT_SUFFIXES:
                    artifacts.append(str(path.relative_to(output_dir)))
                if len(artifacts) >= 60:
                    break
        view["artifacts"] = artifacts
        return view

    def _handle_task_list(self) -> None:
        self._send_json(
            {"tasks": [self._task_view(r) for r in self._server().task_queue.list_tasks()]}
        )

    def _handle_task_detail(self, task_id: str) -> None:
        self._send_json({"task": self._task_view(self._server().task_queue.get(task_id))})

    def _handle_task_log(self, task_id: str, query: dict[str, list[str]]) -> None:
        tail = int(query.get("tail", ["100"])[0])
        self._send_text(self._server().task_queue.read_log_tail(task_id, max_lines=tail))

    def _handle_task_artifact(self, task_id: str, query: dict[str, list[str]]) -> None:
        record = self._server().task_queue.get(task_id)
        relative = (query.get("path") or [""])[0]
        if not relative:
            raise StudioTemplateError("artifact requires ?path=<relative file>")
        output_dir = Path(record["output_dir"]).resolve()
        target = (output_dir / relative).resolve()
        if not str(target).startswith(str(output_dir) + "/") and target != output_dir:
            raise StudioTemplateError("artifact path escapes the task output")
        if target.suffix not in _ARTIFACT_SUFFIXES or not target.is_file():
            raise StudioTaskError(f"no such artifact: {relative}")
        self._send_file(target)

    def _sound_asset_index(self) -> Path:
        index = self._server().studio_config.sound_asset_index
        if index is None:
            raise ProductionError(
                "this deployment declares no sound_asset_index; add it to the "
                "studio config to enable the asset deck"
            )
        return index

    def _handle_sound_asset_catalog(self) -> None:
        self._send_json(load_sound_asset_catalog(self._sound_asset_index()))

    def _handle_sound_asset_file(self, query: dict[str, list[str]]) -> None:
        relative = (query.get("path") or [""])[0]
        if not relative:
            raise ProductionError("asset file requires ?path=<relative file>")
        self._send_file(sound_asset_file(self._sound_asset_index(), relative))

    def _handle_hm3d_scenes(self) -> None:
        """Annotated HM3D scenes, so picking one is a dropdown, not a path.

        The root comes from the room-prepare template's own default - the
        same value the submitted task will resolve against - and a scene
        qualifies only when its semantic annotations are on disk, because a
        scene without them cannot pass the acoustic stage anyway.
        """

        config = self._server().studio_config
        root_value = (config.task_templates.get("hm3d_room_prepare") or {}).get(
            "hm3d_root"
        )
        if not root_value:
            raise ProductionError(
                "the hm3d_room_prepare template declares no hm3d_root default"
            )
        root = Path(str(root_value))
        scenes = []
        for split in ("val", "train", "minival"):
            split_dir = root / split
            if not split_dir.is_dir():
                continue
            for scene_dir in sorted(split_dir.iterdir()):
                if not scene_dir.is_dir() or "-" not in scene_dir.name:
                    continue
                scene_id = scene_dir.name.split("-", 1)[1]
                if not (scene_dir / f"{scene_id}.semantic.txt").is_file():
                    continue
                scenes.append(
                    {
                        "name": scene_dir.name,
                        "split": split,
                        "scene_dir": str(scene_dir),
                        "scene_id": scene_id,
                    }
                )
        self._send_json({"hm3d_root": str(root), "scenes": scenes})

    def _sound_library_root(self) -> Path:
        root = self._server().studio_config.sound_library_root
        if root is None:
            raise ProductionError(
                "this deployment declares no sound_library_root; add it to "
                "the studio config to enable the sound library"
            )
        return root

    def _handle_sound_library(self) -> None:
        config = self._server().studio_config
        self._send_json(
            load_sound_library(
                self._sound_library_root(), config.sound_asset_index
            )
        )

    def _handle_sound_library_file(self, query: dict[str, list[str]]) -> None:
        relative = (query.get("path") or [""])[0]
        if not relative:
            raise ProductionError("sound-library file requires ?path=<relative>")
        self._send_file(sound_library_file(self._sound_library_root(), relative))

    def _curation_dir(self) -> Path:
        # Beside the tasks root, like the queue itself: survives restarts,
        # needs no schema, and a verdict is one file a person can read.
        return self._server().studio_config.tasks_root.parent / "room_curation"

    def _handle_room_curation(self) -> None:
        self._send_json(
            load_room_curation(
                self._server().task_queue.list_tasks(), self._curation_dir()
            )
        )

    def _handle_room_curation_verdict(self) -> None:
        body = self._read_json_body()
        written = write_room_verdict(
            self._curation_dir(),
            house=str(body.get("house") or ""),
            room_label=str(body.get("room_label") or ""),
            verdict=str(body.get("verdict") or ""),
            note=str(body.get("note") or ""),
            author=str(body.get("author") or "studio"),
        )
        self._send_json({"verdict": written})

    def _handle_board(self) -> None:
        self._send_json(board_rows(self._server().task_queue.list_tasks()))

    def _handle_review_queue(self) -> None:
        self._send_json(review_queue(self._server().task_queue.list_tasks()))

    def _handle_task_verdict(self, task_id: str) -> None:
        record = self._server().task_queue.get(task_id)
        body = self._read_json_body()
        written = write_human_verdict(
            Path(record["task_dir"]),
            verdict=str(body.get("verdict") or ""),
            note=str(body.get("note") or ""),
            author=str(body.get("author") or "studio"),
        )
        self._send_json({"task_id": task_id, "verdict": written})

    def _handle_sweep_submit(self) -> None:
        """Cartesian-product batch submission: one task per sweep point."""
        import itertools
        from datetime import datetime, timezone

        server = self._server()
        config = server.studio_config
        body = self._read_json_body()
        template = str(body.get("template") or "")
        base_overrides = body.get("base_overrides") or {}
        sweep = body.get("sweep")
        if not isinstance(base_overrides, dict):
            raise StudioTemplateError("base_overrides must be an object")
        if not isinstance(sweep, dict) or not sweep:
            raise StudioTemplateError(
                "sweep must be a non-empty object of key -> list of values"
            )
        keys = sorted(sweep)
        value_lists = []
        for key in keys:
            values = sweep[key]
            if not isinstance(values, list) or not values:
                raise StudioTemplateError(f"sweep[{key!r}] must be a non-empty list")
            value_lists.append(values)
        combos = list(itertools.product(*value_lists))
        if len(combos) > 64:
            raise StudioTemplateError(
                f"sweep would enqueue {len(combos)} tasks; the cap is 64"
            )
        batch_id = str(
            body.get("batch_id")
            or "sweep-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        task_ids = []
        for combo in combos:
            point = dict(zip(keys, combo))
            overrides = {**base_overrides, **point}
            task_ids.append(
                server.task_queue.submit(
                    template=template,
                    argv_builder=lambda output_path, ov=overrides: build_template_argv(
                        config, template, ov, output_path
                    ),
                    cwd=config.repository_root,
                    metadata={
                        "overrides": overrides,
                        "submitted_via": "sweep",
                        "batch_id": batch_id,
                        "sweep_point": point,
                    },
                )
            )
        self._send_json(
            {"batch_id": batch_id, "task_count": len(task_ids), "task_ids": task_ids},
            status=201,
        )

    def _handle_sweep_list(self) -> None:
        server = self._server()
        batches: dict[str, dict] = {}
        for record in server.task_queue.list_tasks():
            batch_id = (record.get("metadata") or {}).get("batch_id")
            if not batch_id:
                continue
            batch = batches.setdefault(
                batch_id,
                {"batch_id": batch_id, "template": record["template"],
                 "status_counts": {}, "tasks": []},
            )
            batch["status_counts"][record["status"]] = (
                batch["status_counts"].get(record["status"], 0) + 1
            )
            batch["tasks"].append(
                {
                    "task_id": record["task_id"],
                    "status": record["status"],
                    "sweep_point": (record.get("metadata") or {}).get("sweep_point"),
                }
            )
        self._send_json({"sweeps": sorted(batches.values(), key=lambda b: b["batch_id"], reverse=True)})

    def _handle_task_submit(self) -> None:
        server = self._server()
        config = server.studio_config
        body = self._read_json_body()
        template = str(body.get("template") or "")
        overrides = body.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise StudioTemplateError("overrides must be an object")
        task_id = server.task_queue.submit(
            template=template,
            argv_builder=lambda output_path: build_template_argv(
                config, template, overrides, output_path
            ),
            cwd=config.repository_root,
            metadata={"overrides": overrides, "submitted_via": "http"},
        )
        self._send_json(
            {"task": self._task_view(server.task_queue.get(task_id))}, status=201
        )
