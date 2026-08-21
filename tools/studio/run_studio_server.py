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
import sys
from dataclasses import replace

from avengine.studio.config import load_studio_config
from avengine.studio.server import StudioHTTPServer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="studio config JSON")
    parser.add_argument("--port", type=int, help="override the configured port")
    args = parser.parse_args()

    config = load_studio_config(args.config)
    if args.port is not None:
        config = replace(config, port=args.port)

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
