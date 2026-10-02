#!/usr/bin/env python3
"""Local media + API proxy for the room-curation review page.

The studio server's static route only serves files from the owner's checkout
(/data/jzy/code/AVEngine-lead-a/tools/studio/static), which teammates cannot
write. This small server gives the review page its own same-origin root on
127.0.0.1:8766:

  GET /curation_review.html  -> this checkout's review page
  GET /prescreen.js          -> media root prescreen.js (JSONP suggestions)
  GET /media/<house>/R<n>.mp4 -> rendered tour clips
  /api/*                     -> proxied verbatim to the studio server (8765),
                                so the page never trips CORS

Run with: nohup python3 tools/rooms/serve_curation_media.py > logs/curation_media.log 2>&1 &
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.rooms.runtime_config import (RUNTIME_PREFIX, MAGNUM_SITE, RLR_SDK_ROOT, MP3D_ROOT, TASKS_ROOT, MEDIA_ROOT, ROOM_PYTHON)

import http.client
import http.server
import json
import os
import re
import urllib.parse
from pathlib import Path

HOST = "127.0.0.1"
PORT = int(os.environ.get("CURATION_MEDIA_PORT", "8766"))
STUDIO_ORIGIN = "127.0.0.1:8765"
REPO_ROOT = Path(__file__).resolve().parents[2]
MEDIA_ROOT = Path(str(MEDIA_ROOT))
OVERHEAD_ROOT = Path(os.environ.get("CURATION_OVERHEAD_ROOT", str(Path.home() / "room_review_overheads/20260908")))
# 系统章文件目录（server 实时读这里；删除文件 = 撤销回未审）。
VERDICT_DIR = Path("/data/avengine_external/studio/room_curation")

SUFFIX_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".webp": "image/webp",
    ".json": "application/json; charset=utf-8",
}


class Handler(http.server.BaseHTTPRequestHandler):
    # 默认是 HTTP/1.0（不支持持久连接）。浏览器 <video> 靠 HTTP/1.1 的
    # keep-alive 连续分段请求视频字节（首个 206 后继续拉下一段），HTTP/1.0
    # 应答会让 Chrome 拿到一段 206 后直接放弃加载。必须声明 1.1。
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 访问日志走 stderr（nohup 重定向到 curation_media.log）
        http.server.BaseHTTPRequestHandler.log_message(self, fmt, *args)

    def _send_file(self, path: Path, content_type: str) -> None:
        """发文件，支持浏览器视频所需的单范围 Range 请求。"""
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            try:
                spec = range_header[6:].split(",", 1)[0].strip()
                if "-" not in spec:
                    raise ValueError
                start_s, end_s = spec.split("-", 1)
                if start_s == "":
                    length = int(end_s)
                    if length <= 0:
                        raise ValueError
                    start = max(0, size - length)
                else:
                    start = int(start_s)
                    end = int(end_s) if end_s else size - 1
                if start < 0 or start >= size or end < start:
                    raise ValueError
                end = min(end, size - 1)
                status = 206
            except (TypeError, ValueError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        with path.open("rb") as stream:
            stream.seek(start)
            payload = stream.read(end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        # 页面和映射需要即时更新；重复浏览的媒体由浏览器短时缓存。
        cache_seconds = 86400 if "/all_objects_20260909/" in self.path and path.suffix in (".png", ".webp") else 300 if path.suffix in (".png", ".mp4") else 0
        self.send_header("Cache-Control", f"private, max-age={cache_seconds}" if cache_seconds else "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _not_found(self, detail: str = "") -> None:
        body = f"404 {detail}".encode()
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _undo_verdict(self) -> None:
        """撤销盖章:删除 <house>__<label>.json。

        studio server 的 /api/room-curation 实时读 room_curation 目录,
        文件消失后该房间自动回到未审状态。house/label 走白名单,防路径穿越。
        """
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(body)
        except ValueError:
            self._not_found("undo: bad json body")
            return
        house = str(payload.get("house") or "")
        label = str(payload.get("room_label") or "")
        if not (house and label and re.fullmatch(r"[A-Za-z0-9_]+", house)
                and re.fullmatch(r"[A-Za-z0-9_]+", label)):
            self._not_found("undo: bad house/room_label")
            return
        verdict_file = VERDICT_DIR / f"{house}__{label}.json"
        try:
            verdict_file.unlink()
        except FileNotFoundError:
            self._not_found(f"undo: no verdict file for {house} {label}")
            return
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy_api(self) -> None:
        """把 /api/* 原样转发给 studio server（GET/POST 都透传）。"""
        connection = http.client.HTTPConnection(STUDIO_ORIGIN, timeout=120)
        path = self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        body = None
        if self.command in ("POST", "PUT"):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            headers.setdefault("Content-Type", "application/json")
        try:
            connection.request(self.command, path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self.send_response(response.status)
            self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except OSError as error:
            self._not_found(f"studio proxy failed: {error}")
        finally:
            connection.close()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path.startswith('/overheads/'):
            file_path = (OVERHEAD_ROOT / urllib.parse.unquote(path[len('/overheads/'):])).resolve()
            if OVERHEAD_ROOT.resolve() not in file_path.parents or file_path.suffix not in ('.png', '.webp', '.json') or not file_path.is_file():
                self._not_found('overhead image or mapping unavailable')
                return
            self._send_file(file_path, SUFFIX_TYPES[file_path.suffix])
            return
        if path.startswith("/api/"):
            self._proxy_api()
            return
        if path == "/curation_review.html":
            self._send_file(REPO_ROOT / "tools/studio/static/curation_review.html",
                            "text/html; charset=utf-8")
            return
        if path == "/test_video.html":
            # 排障用极简页：隔离审阅页逻辑，单测 <video> 链路。
            self._send_file(MEDIA_ROOT / "test_video.html",
                            "text/html; charset=utf-8")
            return
        if path == "/prescreen.js":
            self._send_file(MEDIA_ROOT / "prescreen.js",
                            "application/javascript; charset=utf-8")
            return
        if path.startswith("/media/"):
            relative = urllib.parse.unquote(path[len("/media/"):])
            file_path = (MEDIA_ROOT / relative).resolve()
            if not str(file_path).startswith(str(MEDIA_ROOT.resolve())) or not file_path.is_file():
                self._not_found("media file missing; render it first")
                return
            self._send_file(file_path, SUFFIX_TYPES.get(file_path.suffix, "application/octet-stream"))
            return
        self._not_found("unknown path (try /curation_review.html)")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/room-curation/verdict/undo":
            self._undo_verdict()
            return
        if path.startswith("/api/"):
            self._proxy_api()
            return
        self._not_found("POST only supported for /api/*")


def main() -> None:
    server = http.server.ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"curation media server on http://{HOST}:{PORT}/curation_review.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
