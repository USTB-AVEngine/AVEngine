#!/usr/bin/env python3
"""Serve one room-split review site and autosave the reviewer's marks to a JSON file (standard library only).

usage: python3 serve_review.py [--site DIR] [--feedback FILE] [--host 127.0.0.1] [--port 8790]
The site directory is the one build_review_site.py wrote (index.html, site_data.json, img/). Every click in the page is
saved to --feedback (default <site>/feedback/review_feedback.json, written atomically) and appended to a .log.jsonl next
to it, so nothing is lost if the browser storage is cleared. Bind stays on 127.0.0.1; open it through ssh -L or VS Code.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ASSESSMENTS = {"", "ok", "bad", "unsure"}
STATIC_SUFFIXES = {".html", ".jpg", ".jpeg", ".png", ".json", ".css", ".js", ".md", ".csv"}
MAX_BODY = 4 * 1024 * 1024
MAX_NOTE = 5000


class ReviewStore:
    def __init__(self, site: Path, feedback: Path):
        self.site = site.resolve()
        data = json.loads((self.site / "site_data.json").read_text(encoding="utf-8"))
        self.build_id = data["build_id"]
        self.reasons = set(data["reasons"])
        self.ids = {r["id"] for c in data["cards"] for r in c["rooms"]} | {"dropped:" + c["card"] for c in data["cards"]}
        self.feedback = feedback.resolve()
        self.log = self.feedback.with_suffix(".log.jsonl")
        self.lock = threading.Lock()

    def read(self) -> dict:
        if not self.feedback.is_file():
            return {}
        return json.loads(self.feedback.read_text(encoding="utf-8")).get("feedback", {})

    def clean(self, item_id, record) -> dict:
        if item_id not in self.ids:
            raise ValueError(f"unknown item id: {item_id}")
        if not isinstance(record, dict):
            raise ValueError("feedback must be an object")
        assessment = str(record.get("assessment", ""))
        if assessment not in ASSESSMENTS:
            raise ValueError("assessment must be ok, bad, unsure or empty")
        codes = record.get("reason_codes", [])
        if not isinstance(codes, list) or any(not isinstance(c, str) or c not in self.reasons for c in codes):
            raise ValueError("reason_codes must come from the site's reason list")
        note = str(record.get("note", ""))
        if len(note) > MAX_NOTE:
            raise ValueError("note is longer than 5000 characters")
        return dict(assessment=assessment, reason_codes=list(dict.fromkeys(codes)), note=note,
                    reviewer=str(record.get("reviewer", ""))[:100], updated_at=str(record.get("updated_at", ""))[:40])

    def write(self, items: dict) -> int:
        cleaned = {k: self.clean(k, v) for k, v in items.items()}
        with self.lock:
            current = self.read()
            changed = 0
            for k, v in cleaned.items():
                old = current.get(k)
                if old is None or v["updated_at"] >= old.get("updated_at", ""):
                    current[k] = v
                    changed += 1
            self.feedback.parent.mkdir(parents=True, exist_ok=True)
            doc = dict(schema="avengine_room_split_review_feedback_v1", build_id=self.build_id,
                       saved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), feedback=current)
            tmp = self.feedback.with_name(self.feedback.name + ".tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            os.replace(tmp, self.feedback)
            with self.log.open("a", encoding="utf-8") as fh:
                for k, v in cleaned.items():
                    fh.write(json.dumps(dict(id=k, **v), ensure_ascii=False) + "\n")
        return changed


def make_handler(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AVEngineRoomSplitReview/1.0"

        def send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json") else "max-age=3600")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def send_json(self, code, obj):
            self.send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/feedback":
                return self.send_json(200, store.read())
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (store.site / rel).resolve()
            try:
                target.relative_to(store.site)
            except ValueError:
                return self.send(404, b"Not found", "text/plain; charset=utf-8")
            private = {store.feedback, store.log, store.feedback.with_name(store.feedback.name + ".tmp")}
            if target.suffix.lower() not in STATIC_SUFFIXES or not target.is_file() or target in private:
                return self.send(404, b"Not found", "text/plain; charset=utf-8")
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/"):
                ctype += "; charset=utf-8"
            return self.send(200, target.read_bytes(), ctype)

        do_HEAD = do_GET

        def do_POST(self):
            path = urlsplit(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY:
                    raise ValueError("request body must be between 1 byte and 4 MB")
                payload = json.loads(self.rfile.read(length))
                if path == "/api/feedback":
                    changed = store.write({payload["id"]: payload["feedback"]})
                elif path == "/api/feedback/bulk":
                    if not isinstance(payload.get("items"), dict):
                        raise ValueError("items must be an object")
                    changed = store.write(payload["items"])
                else:
                    return self.send(404, b"Not found", "text/plain; charset=utf-8")
                self.send_json(200, dict(ok=True, changed=changed))
            except Exception as error:  # report every validation problem to the page
                self.send_json(400, dict(ok=False, error=str(error)))

        def log_message(self, fmt, *args):
            if self.command == "POST":
                print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--feedback", type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args(argv)
    feedback = args.feedback or args.site / "feedback" / "review_feedback.json"
    store = ReviewStore(args.site, feedback)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"room review: http://{args.host}:{server.server_port}/  ({len(store.ids)} items, saving to {store.feedback})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
