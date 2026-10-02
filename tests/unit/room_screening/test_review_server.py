from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from tools.rooms.room_screening.review_server import ReviewApp, make_handler, read_manifest
from tools.rooms.room_screening.build_review_manifest import build_manifest


class ReviewServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.assets = root / "assets"
        self.assets.mkdir()
        (self.assets / "topdown.png").write_bytes(b"synthetic-image")
        (self.assets / "walkthrough.mp4").write_bytes(b"0123456789")
        self.manifest_path = root / "manifest.json"
        self.manifest_path.write_text(json.dumps({
            "schema_version": "room_screening_review_manifest_v1",
            "title": "Synthetic test batch",
            "choices": ["keep", "review"],
            "items": [{"id": "case-1", "image_file": "topdown.png", "video_file": "walkthrough.mp4"}],
        }), encoding="utf-8")
        self.feedback_path = root / "out" / "feedback.json"
        self.app = ReviewApp(self.manifest_path, self.assets, self.feedback_path)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.opener = build_opener(ProxyHandler({}))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_serves_ui_manifest_and_assets(self):
        with self.opener.open(self.base + "/") as response:
            self.assertEqual(response.status, 200)
            self.assertIn("候选区域".encode(), response.read())
        with self.opener.open(self.base + "/api/manifest") as response:
            self.assertEqual(json.load(response)["items"][0]["id"], "case-1")
        with self.opener.open(self.base + "/asset/topdown.png") as response:
            self.assertEqual(response.read(), b"synthetic-image")

    def test_autosaves_feedback_only_to_configured_output(self):
        payload = json.dumps({"id": "case-1", "feedback": {"assessment": "keep", "note": "synthetic note"}}).encode()
        request = Request(self.base + "/api/feedback", data=payload, headers={"Content-Type": "application/json"})
        with self.opener.open(request) as response:
            self.assertEqual(json.load(response), {"ok": True})
        self.assertEqual(json.loads(self.feedback_path.read_text())["case-1"]["note"], "synthetic note")

    def test_video_supports_byte_ranges_without_full_file_response(self):
        request = Request(self.base + "/asset/walkthrough.mp4", headers={"Range": "bytes=3-6"})
        with self.opener.open(request) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.headers["Content-Range"], "bytes 3-6/10")
            self.assertEqual(response.read(), b"3456")

    def test_rejects_asset_path_escape(self):
        with self.assertRaises(HTTPError) as error:
            self.opener.open(self.base + "/asset/%2e%2e/manifest.json")
        self.assertEqual(error.exception.code, 404)

    def test_manifest_rejects_duplicate_ids_and_missing_assets(self):
        doc = {"schema_version": "room_screening_review_manifest_v1", "items": [
            {"id": "dup", "image_file": "topdown.png"},
            {"id": "dup", "image_file": "topdown.png"},
        ]}
        self.manifest_path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique"):
            read_manifest(self.manifest_path, self.assets)
        doc["items"] = [{"id": "missing", "image_file": "absent.png"}]
        self.manifest_path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or outside"):
            read_manifest(self.manifest_path, self.assets)

    def test_manifest_builder_converts_local_paths_and_rejects_escapes(self):
        source = {"title": "fixture", "items": [{
            "id": "case-a", "image_path": str(self.assets / "topdown.png"),
            "video_path": "walkthrough.mp4", "house": "synthetic-house",
        }]}
        result = build_manifest(source, self.assets)
        self.assertEqual(result["items"][0]["image_file"], "topdown.png")
        self.assertEqual(result["items"][0]["video_file"], "walkthrough.mp4")
        source["items"][0]["image_path"] = str(Path(self.temp.name) / "outside.png")
        with self.assertRaisesRegex(ValueError, "outside --asset-root"):
            build_manifest(source, self.assets)


if __name__ == "__main__":
    unittest.main()
