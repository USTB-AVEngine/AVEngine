"""Private Unix-socket Blender RPC endpoint for local SkinTokens inference.

The socket lives in a runner-owned 0700 directory and is chmod 0600 before
requests are served. Pickle is trusted only across this OS-protected parent /
child channel; no TCP listener is created.
"""

from __future__ import annotations

import argparse
import http.server
import os
from pathlib import Path
import pickle
import socketserver
import stat
import sys
import traceback
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPOSITORY / "src" / "avengine" / "assets"
sys.path.insert(0, str(ASSET_ROOT))

from skintokens.rig_package.parser.bpy import BpyParser, transfer_rigging  # noqa: E402


def _validate_socket_parent(socket_path: Path) -> None:
    parent = socket_path.parent
    if not parent.is_dir():
        raise ValueError(f"private RPC directory is missing: {parent}")
    mode = stat.S_IMODE(parent.stat().st_mode)
    if mode != 0o700:
        raise PermissionError(
            f"private RPC directory must be mode 0700, got {mode:04o}: {parent}"
        )
    if parent.stat().st_uid != os.getuid():
        raise PermissionError(f"private RPC directory is not owned by this user: {parent}")
    if socket_path.exists() or socket_path.is_symlink():
        raise FileExistsError(f"refusing to replace existing RPC socket: {socket_path}")


def _read_payload(handler: http.server.BaseHTTPRequestHandler) -> Any:
    length = int(handler.headers.get("Content-Length", "0"))
    if length < 0:
        raise ValueError("negative Content-Length")
    return pickle.loads(handler.rfile.read(length))


def _write_payload(handler: http.server.BaseHTTPRequestHandler, value: Any) -> None:
    body = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "AVEngineSkinTokensRPCUnix/1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[SKINTOKENS_BPY] " + (fmt % args), flush=True)

    def do_GET(self) -> None:
        if self.path == "/ping":
            body = b"pong"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404, "unknown endpoint")

    def do_POST(self) -> None:
        if self.path not in {"/load", "/export", "/transfer"}:
            self.send_error(404, "unknown endpoint")
            return
        try:
            payload = _read_payload(self)
            if self.path == "/load":
                if not isinstance(payload, str):
                    raise ValueError("/load expects a filepath string")
                print(f"[SKINTOKENS_BPY] load {payload}", flush=True)
                result = BpyParser.load(payload)
            elif self.path == "/export":
                if not isinstance(payload, dict):
                    raise ValueError("/export expects a mapping")
                output = Path(payload["filepath"]).expanduser().resolve()
                if output.exists():
                    raise FileExistsError(
                        f"refusing to overwrite existing export: {output}"
                    )
                print(f"[SKINTOKENS_BPY] export {output}", flush=True)
                BpyParser.export(**payload)
                result = "ok"
            else:
                if not isinstance(payload, dict):
                    raise ValueError("/transfer expects a mapping")
                output = Path(payload["export_path"]).expanduser().resolve()
                if output.exists():
                    raise FileExistsError(
                        f"refusing to overwrite existing transfer: {output}"
                    )
                print(f"[SKINTOKENS_BPY] transfer {output}", flush=True)
                transfer_rigging(**payload)
                result = "ok"
            _write_payload(self, result)
        except Exception as error:
            message = {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            print(message["traceback"], file=sys.stderr, flush=True)
            _write_payload(self, message)


class _Server(socketserver.UnixStreamServer):
    allow_reuse_address = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=False)
    parser.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    if argv is None:
        raw = sys.argv
        argv = raw[raw.index("--") + 1:] if "--" in raw else []
    args = parser.parse_args(argv)
    if args.port is not None:
        parser.error("--port is deprecated; the RPC requires a private Unix socket")
    if args.socket is None:
        parser.error("--socket is required")
    socket_path = args.socket.expanduser().resolve()
    _validate_socket_parent(socket_path)
    old_umask = os.umask(0o077)
    try:
        httpd = _Server(str(socket_path), _Handler)
    finally:
        os.umask(old_umask)
    os.chmod(socket_path, 0o600)
    print(f"SKINTOKENS_BPY_READY {socket_path}", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        print("SKINTOKENS_BPY_STOPPED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
