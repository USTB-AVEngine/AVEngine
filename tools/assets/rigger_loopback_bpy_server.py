"""Loopback-only launcher for the vendored TokenRig bpy server.

The vendored server asks bottle for host 0.0.0.0.  On a dual-stack host tornado
binds :: first and then fails to bind 0.0.0.0 with EADDRINUSE inside the server
thread, so the process prints "Listening" and then serves nothing.  Forcing the
loopback address binds exactly one socket, which is all the local client needs.
"""

import bottle

_original_run = bottle.run


def _loopback_run(app=None, **kwargs):
    kwargs["host"] = "127.0.0.1"
    return _original_run(app, **kwargs)


bottle.run = _loopback_run

from src.server.bpy_server import run  # noqa: E402

if __name__ == "__main__":
    run()
