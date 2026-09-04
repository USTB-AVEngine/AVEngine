#!/usr/bin/env python3
"""Launch the AVEngine Studio backend server (loopback only).

Example:
    python tools/studio/run_studio_server.py \
        --config tools/studio/studio_config_48g.json

Preview from a workstation through a tunnel:
    ssh -L 8765:127.0.0.1:8765 <server>   # then open http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path


def _bootstrap_repository_source(config_path: str | Path) -> Path:
    """Select the configured AVEngine source before importing AVEngine.

    The launcher itself can be invoked from an arbitrary checkout or with a
    stale editable install. Read only the small bootstrap field with stdlib,
    put that repository's ``src`` first for this process and its children,
    then make the same repository the process working directory. The full
    config loader remains the authority for all other validation.
    """

    config_file = Path(config_path).expanduser().resolve()
    try:
        payload = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read Studio config {config_file}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Studio config must be a JSON object: {config_file}")
    raw_root = payload.get("repository_root")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise SystemExit(
            f"Studio config repository_root must be a non-empty string: {config_file}"
        )
    repository_root = Path(raw_root).expanduser()
    if not repository_root.is_absolute():
        repository_root = config_file.parent / repository_root
    repository_root = repository_root.resolve()
    source_root = repository_root / "src"
    if not repository_root.is_dir():
        raise SystemExit(f"configured repository_root is not a directory: {repository_root}")
    if not source_root.is_dir():
        raise SystemExit(f"configured repository has no src directory: {source_root}")

    source = str(source_root)

    def is_source_path(value: str) -> bool:
        try:
            return Path(value or ".").expanduser().resolve() == source_root
        except OSError:
            return False

    sys.path[:] = [source] + [entry for entry in sys.path if not is_source_path(entry)]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    existing_entries = (
        existing_pythonpath.split(os.pathsep) if existing_pythonpath else []
    )
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [source] + [entry for entry in existing_entries if not is_source_path(entry)]
    )
    os.chdir(repository_root)
    return repository_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="studio config JSON")
    parser.add_argument("--port", type=int, help="override the configured port")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()

    configured_root = _bootstrap_repository_source(config_path)
    from avengine.studio.config import load_studio_config

    config = load_studio_config(config_path)
    if config.repository_root != configured_root:
        raise SystemExit(
            "Studio config repository_root changed between bootstrap and full load: "
            f"{configured_root} != {config.repository_root}"
        )
    if args.port is not None:
        config = replace(config, port=args.port)

    from avengine.studio.server import StudioHTTPServer

    server = StudioHTTPServer(config)
    host, port = server.server_address[:2]
    print(
        json.dumps(
            {
                "url": f"http://{host}:{port}/",
                "tunnel_hint": f"ssh -L {port}:127.0.0.1:{port} <server>",
                "tasks_root": str(config.tasks_root),
                "templates": sorted(config.task_templates),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
