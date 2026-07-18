#!/usr/bin/env bash
# Reproducible fast bootstrap for the Habitat-native AVEngine repository.
# Native Habitat/RLR builds and external datasets are separate opt-in layers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_PATH="${AVENGINE_VENV:-$REPOSITORY_ROOT/.venv}"
PROFILE="habitat_native"
DRY_RUN=0
SKIP_TESTS=0
CLONE_RUNTIME=0

usage() {
    echo "Usage: $0 [--dry-run] [--skip-tests] [--clone-runtime] [--profile habitat_native|legacy_optional] [--python PATH] [--venv PATH]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --clone-runtime) CLONE_RUNTIME=1; shift ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --venv) VENV_PATH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[setup] unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [ "$PROFILE" != "habitat_native" ] && [ "$PROFILE" != "legacy_optional" ]; then
    echo "[setup] unsupported profile: $PROFILE" >&2
    exit 2
fi

run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '[setup] DRY-RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

echo "[setup] profile=$PROFILE"
if [ "$PROFILE" = "legacy_optional" ]; then
    echo "[setup] legacy_optional is documentation-only; no UE/SPEAR/gpuRIR dependency is installed."
    echo "[setup] see docs/legacy/OPTIONAL_BACKENDS.md"
fi

run "$PYTHON_BIN" -m venv "$VENV_PATH"
VENV_PYTHON="$VENV_PATH/bin/python"
run "$VENV_PYTHON" -m pip install --upgrade pip
run "$VENV_PYTHON" -m pip install -e "$REPOSITORY_ROOT[test]"

if [ "$DRY_RUN" = "0" ]; then
    eval "$("$VENV_PYTHON" "$SCRIPT_DIR/load_paths.py" --export)"
fi

if [ "$CLONE_RUNTIME" = "1" ]; then
    if [ "$DRY_RUN" = "1" ]; then
        echo "[setup] DRY-RUN: would clone/verify the manifest-pinned Habitat runtime"
    else
        readarray -t RUNTIME_VALUES < <("$VENV_PYTHON" - "$REPOSITORY_ROOT/manifest.yaml" <<'PY'
import pathlib, sys, yaml
manifest = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = manifest["repositories"]["habitat_runtime"]
print(runtime["url"])
print(runtime["commit"])
print(runtime["default_path"])
PY
        )
        RUNTIME_URL="${RUNTIME_VALUES[0]}"
        RUNTIME_COMMIT="${RUNTIME_VALUES[1]}"
        RUNTIME_PATH="${AVENGINE_HABITAT_RUNTIME_ROOT:-$REPOSITORY_ROOT/${RUNTIME_VALUES[2]}}"
        if [ ! -d "$RUNTIME_PATH/.git" ]; then
            git clone "$RUNTIME_URL" "$RUNTIME_PATH"
        fi
        git -C "$RUNTIME_PATH" fetch origin "$RUNTIME_COMMIT"
        CURRENT="$(git -C "$RUNTIME_PATH" rev-parse HEAD)"
        if [ "$CURRENT" != "$RUNTIME_COMMIT" ]; then
            echo "[setup] runtime exists at $CURRENT; refusing to change it automatically." >&2
            echo "[setup] create a clean worktree at pinned commit $RUNTIME_COMMIT or update AVENGINE_HABITAT_RUNTIME_ROOT." >&2
            exit 1
        fi
    fi
fi

run "$VENV_PYTHON" "$SCRIPT_DIR/load_paths.py" --validate --layer fast_unit
run "$VENV_PYTHON" "$SCRIPT_DIR/validate_schemas.py"
if [ "$SKIP_TESTS" = "0" ]; then
    run "$VENV_PYTHON" -m pytest -q tests/unit -m "not integration and not canary"
fi

echo "[setup] fast Habitat-native bootstrap complete."
echo "[setup] Native Habitat/RLR, Blender and media canaries remain separate test layers."
