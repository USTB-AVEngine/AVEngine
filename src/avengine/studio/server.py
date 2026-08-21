"""Loopback-only stdlib HTTP server for the Studio backend skeleton.

GET  /                       minimal status page
GET  /api/health
GET  /api/rooms              room registry with studio_status annotations
GET  /api/registries         registry summaries
GET  /api/registries/<name>  full registry passthrough
GET  /api/captures           review base captures (?include=all keeps
                             captures whose commit is known off main)
GET  /api/templates          task templates, defaults, overridable keys
GET  /api/tasks              task list (newest first)
POST /api/tasks              {"template": ..., "overrides": {...}}
GET  /api/tasks/<id>
GET  /api/tasks/<id>/log     text tail (?tail=N)
"""

from __future__ import annotations

import json
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
from avengine.studio.tasks import StudioTaskError, StudioTaskQueue
from avengine.studio.templates import (
    TEMPLATE_OVERRIDABLE_KEYS,
    StudioTemplateError,
    build_template_argv,
)

_STATUS_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>AVEngine Studio</title>
<style>
body{font-family:system-ui,'PingFang SC',sans-serif;margin:2rem;max-width:70rem;background:#fbfaf7;color:#2c2c2a}
h1{font-size:1.3rem} h2{font-size:1.05rem;margin-top:1.6rem}
table{border-collapse:collapse;width:100%;font-size:.85rem}
td,th{border:1px solid #d8d5cc;padding:.3rem .5rem;text-align:left}
code{background:#f1efe8;padding:.1rem .3rem;border-radius:3px}
button{padding:.4rem .8rem;cursor:pointer}
.badge{padding:.05rem .4rem;border-radius:3px;font-size:.78rem}
.pass{background:#e1f5ee}.fail{background:#fae7e7}.running{background:#fdf3d8}.queued{background:#eeedfe}
</style></head><body>
<h1>AVEngine Studio · Session 1 后端骨架</h1>
<p>只读目录 API + 串行渲染队列。接口清单见 <code>/api/health</code>、<code>/api/rooms</code>、
<code>/api/registries</code>、<code>/api/captures</code>、<code>/api/templates</code>、<code>/api/tasks</code>。</p>
<p><button onclick="submitDefault()">用默认模板提交一次 MP3D 动态音频渲染</button>
<span id="submitResult"></span></p>
<h2>任务</h2><div id="tasks">加载中…</div>
<h2>底片捕获（默认排除已知偏离 main 的 commit）</h2><div id="captures">加载中…</div>
<script>
async function refresh(){
  const tasks=await (await fetch('/api/tasks')).json();
  document.getElementById('tasks').innerHTML=
    '<table><tr><th>task</th><th>模板</th><th>状态</th><th>创建</th><th>结束</th><th>日志</th></tr>'+
    tasks.tasks.map(t=>`<tr><td>${t.task_id}</td><td>${t.template}</td>`+
      `<td><span class="badge ${t.status}">${t.status}</span></td>`+
      `<td>${t.created_at||''}</td><td>${t.finished_at||''}</td>`+
      `<td><a href="/api/tasks/${t.task_id}/log?tail=100">tail</a></td></tr>`).join('')+'</table>';
  const caps=await (await fetch('/api/captures')).json();
  document.getElementById('captures').innerHTML=
    '<table><tr><th>名称</th><th>房间</th><th>帧数</th><th>commit</th><th>可信</th></tr>'+
    caps.captures.map(c=>`<tr><td>${c.name}</td><td>${c.room_id||''}</td>`+
      `<td>${c.frame_count||''}</td><td>${c.commit||''}</td><td>${c.trusted}</td></tr>`).join('')+'</table>';
}
async function submitDefault(){
  const r=await fetch('/api/tasks',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({template:'mp3d_dynamic_audio',overrides:{}})});
  const body=await r.json();
  document.getElementById('submitResult').textContent=
    r.ok?('已入队：'+body.task.task_id):('失败：'+(body.error||r.status));
  refresh();
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
        super().__init__((config.host, config.port), StudioRequestHandler)


class StudioRequestHandler(BaseHTTPRequestHandler):
    server_version = "AVEngineStudio/0.1"
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
            elif method == "GET" and path == "/api/templates":
                self._handle_templates()
            elif method == "GET" and path == "/api/tasks":
                self._handle_task_list()
            elif method == "POST" and path == "/api/tasks":
                self._handle_task_submit()
            elif method == "GET" and path.startswith("/api/tasks/") and path.endswith("/log"):
                task_id = path.removeprefix("/api/tasks/").removesuffix("/log").strip("/")
                self._handle_task_log(task_id, query)
            elif method == "GET" and path.startswith("/api/tasks/"):
                self._handle_task_detail(path.removeprefix("/api/tasks/"))
            else:
                self._send_error_json(f"no route for {method} {path}", status=404)
        except StudioTaskError as exc:
            self._send_error_json(str(exc), status=404)
        except (StudioTemplateError, StudioConfigError, json.JSONDecodeError, ValueError) as exc:
            self._send_error_json(str(exc), status=400)
        except BrokenPipeError:
            pass
        except Exception as exc:  # surface, never crash the connection thread
            self._send_error_json(f"{type(exc).__name__}: {exc}", status=500)

    # -- handlers ----------------------------------------------------------------

    def _handle_health(self) -> None:
        server: StudioHTTPServer = self.server  # type: ignore[assignment]
        config = server.studio_config
        self._send_json(
            {
                "status": "ok",
                "repository_root": str(config.repository_root),
                "review_root": str(config.review_root),
                "tasks_root": str(config.tasks_root),
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
        server: StudioHTTPServer = self.server  # type: ignore[assignment]
        config = server.studio_config
        registry = load_registry_json(config.room_registry_path)
        self._send_json(annotate_room_registry(registry, config.room_policy))

    def _handle_registry_summaries(self) -> None:
        server: StudioHTTPServer = self.server  # type: ignore[assignment]
        config = server.studio_config
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
        server: StudioHTTPServer = self.server  # type: ignore[assignment]
        path = server.studio_config.registry_paths.get(name)
        if path is None:
            self._send_error_json(
                f"unknown registry {name!r}; available: "
                f"{sorted(server.studio_config.registry_paths)}",
                status=404,
            )
            return
        self._send_json(load_registry_json(path))

    def _handle_captures(self, query: dict[str, list[str]]) -> None:
        server: StudioHTTPServer = self.server  # type: ignore[assignment]
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

    def _handle_templates(self) -> None:
        server: StudioHTTPServer = self.server  # type: ignore[assignment]
        config = server.studio_config
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
        return view

    def _handle_task_list(self) -> None:
        server: StudioHTTPServer = self.server  # type: ignore[assignment]
        self._send_json(
            {"tasks": [self._task_view(r) for r in server.task_queue.list_tasks()]}
        )

    def _handle_task_detail(self, task_id: str) -> None:
        server: StudioHTTPServer = self.server  # type: ignore[assignment]
        self._send_json({"task": self._task_view(server.task_queue.get(task_id))})

    def _handle_task_log(self, task_id: str, query: dict[str, list[str]]) -> None:
        server: StudioHTTPServer = self.server  # type: ignore[assignment]
        tail = int(query.get("tail", ["100"])[0])
        self._send_text(server.task_queue.read_log_tail(task_id, max_lines=tail))

    def _handle_task_submit(self) -> None:
        server: StudioHTTPServer = self.server  # type: ignore[assignment]
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
