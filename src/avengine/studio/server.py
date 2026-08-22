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
    scene_file_path,
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
<p><a class="big" href="/studio">打开 3D 场景编辑器 →</a></p>
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

    def _send_payload(self, body: bytes, content_type: str, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
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
        self._send_payload(body, content_type, status)

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
                self._send_payload(
                    _STATUS_PAGE.encode("utf-8"), "text/html; charset=utf-8", 200
                )
            elif method == "GET" and path == "/studio":
                self._handle_static("studio.html")
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
            elif method == "GET" and path == "/api/templates":
                self._handle_templates()
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
        except (StudioTemplateError, StudioConfigError, json.JSONDecodeError, ValueError) as exc:
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
