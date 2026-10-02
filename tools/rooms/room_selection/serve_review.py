"""Read-only HTTP viewer; review decisions are exported in the browser."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument("--port", type=int, default=8787)
    a = p.parse_args()
    root = a.root.resolve()
    media = a.media_root.resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def translate_path(self, path):
            clean = unquote(path.split("?", 1)[0])
            if clean.startswith("/media/"):
                target = (media / clean[len("/media/") :]).resolve()
                if not target.is_relative_to(media):
                    return str(root / "missing")
                return str(target)
            return super().translate_path(path)

    print(f"http://127.0.0.1:{a.port}/review.html", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
