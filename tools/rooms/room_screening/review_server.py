#!/usr/bin/env python3
"""Serve one local room-screening review manifest without copying its assets."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

SCHEMA_VERSION = "room_screening_review_manifest_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEEDBACK = REPOSITORY_ROOT / "tmp/room_screening/review_feedback.json"
UI_PATH = Path(__file__).with_name("review.html")


def child_file(root: Path, relative: str) -> Path | None:
    """Resolve an asset below root; reject traversal and symlink escapes."""
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def read_manifest(path: Path, asset_root: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION!r}")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("manifest.items must be a non-empty list")
    ids = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{index}] must be an object")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"items[{index}].id must be a non-empty string")
        ids.append(item_id)
        image = item.get("image_file")
        if not isinstance(image, str) or child_file(asset_root, image) is None:
            raise ValueError(f"items[{index}].image_file is missing or outside asset root")
        for key in ("geometry_file", "video_file"):
            value = item.get(key)
            if value is not None and (not isinstance(value, str) or child_file(asset_root, value) is None):
                raise ValueError(f"items[{index}].{key} is missing or outside asset root")
    if len(ids) != len(set(ids)):
        raise ValueError("manifest item IDs must be unique")
    choices = manifest.get("choices", [])
    if not isinstance(choices, list) or any(not isinstance(x, str) for x in choices):
        raise ValueError("manifest.choices must be a list of strings")
    return manifest


class ReviewApp:
    def __init__(self, manifest_path: Path, asset_root: Path, feedback_path: Path):
        self.manifest_path = manifest_path.expanduser().resolve()
        self.asset_root = asset_root.expanduser().resolve()
        self.feedback_path = feedback_path.expanduser()
        if not self.feedback_path.is_absolute():
            self.feedback_path = (REPOSITORY_ROOT / self.feedback_path).resolve()
        else:
            self.feedback_path = self.feedback_path.resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")
        if not self.asset_root.is_dir():
            raise NotADirectoryError(f"asset root not found: {self.asset_root}")
        self.manifest = read_manifest(self.manifest_path, self.asset_root)
        self.allowed_ids = {item["id"] for item in self.manifest["items"]}
        self.choices = set(self.manifest.get("choices", []))
        self.lock = threading.Lock()

    @staticmethod
    def send_bytes(handler, data: bytes, content_type: str, status: int = 200, headers=None):
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (headers or {}).items():
            handler.send_header(key, value)
        handler.end_headers()
        if handler.command != "HEAD":
            handler.wfile.write(data)

    def send_file(self, handler, path: Path):
        size = path.stat().st_size
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        range_header = handler.headers.get("Range")
        start, end, status = 0, size - 1, 200
        if range_header and content_type.startswith("video/"):
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or (not match.group(1) and not match.group(2)) or size == 0:
                self.send_bytes(handler, b"", "text/plain", 416)
                return
            if not match.group(1):
                suffix_length = int(match.group(2))
                if suffix_length <= 0:
                    self.send_bytes(handler, b"", "text/plain", 416)
                    return
                start = max(size - suffix_length, 0)
                end = size - 1
            else:
                start = int(match.group(1))
                end = min(int(match.group(2) or size - 1), size - 1)
            if start >= size or end < start:
                self.send_bytes(handler, b"", "text/plain", 416)
                return
            status = 206
        length = max(0, end - start + 1)
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(length))
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Content-Type-Options", "nosniff")
        if content_type.startswith("video/"):
            handler.send_header("Accept-Ranges", "bytes")
        if status == 206:
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        handler.end_headers()
        if handler.command == "HEAD":
            return
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                remaining -= len(chunk)

    def route(self, handler):
        path = unquote(urlsplit(handler.path).path)
        if path in ("/", "/review.html"):
            return self.send_file(handler, UI_PATH)
        if path == "/api/manifest":
            return self.send_bytes(handler, json.dumps(self.manifest, ensure_ascii=False).encode(),
                                   "application/json; charset=utf-8")
        if path == "/api/feedback":
            data = self.feedback_path.read_bytes() if self.feedback_path.is_file() else b"{}"
            return self.send_bytes(handler, data, "application/json; charset=utf-8")
        if path.startswith("/asset/"):
            file_path = child_file(self.asset_root, path[len("/asset/"):])
            if file_path is not None:
                return self.send_file(handler, file_path)
        return self.send_bytes(handler, b"Not found", "text/plain; charset=utf-8", 404)

    def post_feedback(self, handler):
        length_text = handler.headers.get("Content-Length", "0")
        try:
            length = int(length_text)
            if length <= 0 or length > 20000:
                raise ValueError("request body must be between 1 and 20000 bytes")
            payload = json.loads(handler.rfile.read(length))
            item_id = payload["id"]
            feedback = payload["feedback"]
            if item_id not in self.allowed_ids or not isinstance(feedback, dict):
                raise ValueError("unknown item or invalid feedback")
            assessment = str(feedback.get("assessment", ""))[:200]
            note = str(feedback.get("note", ""))
            if len(note) > 5000:
                raise ValueError("note exceeds 5000 characters")
            if self.choices and assessment and assessment not in self.choices:
                raise ValueError("assessment is not listed in manifest choices")
            row = {
                "assessment": assessment,
                "note": note,
                "updated_at": str(feedback.get("updated_at", ""))[:100],
            }
            with self.lock:
                current = json.loads(self.feedback_path.read_text()) if self.feedback_path.is_file() else {}
                current[item_id] = row
                self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.feedback_path.with_suffix(self.feedback_path.suffix + ".tmp")
                temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n")
                os.replace(temporary, self.feedback_path)
            self.send_bytes(handler, b'{"ok":true}', "application/json; charset=utf-8")
        except Exception as error:
            body = json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False).encode()
            self.send_bytes(handler, body, "application/json; charset=utf-8", 400)


def make_handler(app: ReviewApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AVEngineRoomReview/1.0"

        def do_GET(self):
            app.route(self)

        def do_HEAD(self):
            app.route(self)

        def do_POST(self):
            if urlsplit(self.path).path == "/api/feedback":
                app.post_feedback(self)
            else:
                app.send_bytes(self, b"Not found", "text/plain; charset=utf-8", 404)

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args))

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="review batch JSON manifest")
    parser.add_argument("--asset-root", type=Path, required=True, help="root for image/video/geometry files")
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK,
                        help="autosave JSON path; default is repository tmp/room_screening/")
    parser.add_argument("--host", default="127.0.0.1", help="bind locally; do not expose review notes publicly")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    app = ReviewApp(args.manifest, args.asset_root, args.feedback)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Room review page: http://{args.host}:{server.server_port}/", flush=True)
    print(f"Loaded {len(app.manifest['items'])} items; feedback path: {app.feedback_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
