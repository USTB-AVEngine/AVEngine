"""Load AVEngine/paths.yaml and expose as environment variables prefixed
with AVENGINE_.

Usage:
    # In shell (export vars into current shell):
    eval "$(python3 scripts/load_paths.py --export)"

    # In Python:
    from load_paths import load_paths_env
    load_paths_env()          # side-effect: sets os.environ vars

    # CLI validation:
    python3 scripts/load_paths.py --validate
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATHS_YAML = os.path.join(os.path.dirname(_HERE), "paths.yaml")


def _flatten(d, prefix="AVENGINE"):
    """Flatten nested YAML dict into env-var-friendly UPPER_SNAKE keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}_{k.upper()}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key))
        elif v is None:
            continue  # skip null / commented-out entries
        else:
            out[key] = str(v)
    return out


def load_paths_dict(yaml_path=_PATHS_YAML):
    """Return the flat env-var dict without setting anything."""
    with open(yaml_path) as f:
        d = yaml.safe_load(f) or {}
    d.pop("version", None)
    return _flatten(d)


def load_paths_env(yaml_path=_PATHS_YAML):
    """Set env vars in-process. Idempotent (later calls overwrite)."""
    env = load_paths_dict(yaml_path)
    for k, v in env.items():
        os.environ[k] = v
    return env


def _validate():
    """Warn (not fail) on missing REQUIRED paths."""
    env = load_paths_dict()
    required_keys = ("AVENGINE_UNREAL_ENGINE_DIR", "AVENGINE_AUDIO_CORPUS")
    any_missing = False
    for k in required_keys:
        p = env.get(k)
        if not p:
            print(f"[paths] MISSING {k} (not in paths.yaml)", file=sys.stderr)
            any_missing = True
        elif not os.path.exists(p):
            print(f"[paths] {k}={p} does not exist on this machine",
                  file=sys.stderr)
            any_missing = True
        else:
            print(f"[paths] {k}={p} OK")
    return 0 if not any_missing else 1


def _export():
    """Print `export KEY=VALUE` lines for `eval`."""
    env = load_paths_dict()
    for k, v in env.items():
        # simple shell-escape (paths.yaml values shouldn't have quotes/spaces)
        print(f'export {k}="{v}"')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", action="store_true", help="print shell export lines")
    p.add_argument("--validate", action="store_true", help="check REQUIRED paths exist")
    args = p.parse_args()
    if args.export:
        _export()
    elif args.validate:
        sys.exit(_validate())
    else:
        for k, v in load_paths_dict().items():
            print(f"{k}={v}")


if __name__ == "__main__":
    main()
