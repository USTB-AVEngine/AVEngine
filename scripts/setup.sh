#!/usr/bin/env bash
# Reproducible single-source AVEngine bootstrap.
# Current profiles run directly in one validated Conda environment. Native
# builds, external UE installations, datasets, and the user-installed
# Habitat/Magnum/RLR inputs remain separate layers. This script never creates a
# venv, clones, fetches, or resolves a sibling source checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN=""
PYTHON_SOURCE=""
SELECTED_CONDA_PREFIX=""
PROFILE="fast_unit"
DRY_RUN=0
SKIP_TESTS=0

usage() {
    echo "Usage: $0 [--dry-run] [--skip-tests] [--profile fast_unit|native_external|habitat_native|legacy_optional] [--python CONDA_PYTHON]"
    echo "       Select Conda Python with --python, AVENGINE_CONDA_PYTHON, or an active CONDA_PREFIX."
    echo "       --clone-runtime and --venv are retired and exit with status 2."
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --clone-runtime)
            echo "[setup] --clone-runtime was removed: provide an installed non-Git runtime through AVENGINE_HABITAT_RUNTIME_PREFIX instead." >&2
            exit 2
            ;;
        --venv)
            echo "[setup] --venv was removed: current bootstrap runs directly in the selected Conda environment." >&2
            exit 2
            ;;
        --profile)
            if [ "$#" -lt 2 ]; then
                echo "[setup] --profile requires a value." >&2
                exit 2
            fi
            PROFILE="$2"
            shift 2
            ;;
        --python)
            if [ "$#" -lt 2 ]; then
                echo "[setup] --python requires a Conda Python path." >&2
                exit 2
            fi
            PYTHON_BIN="$2"
            PYTHON_SOURCE="--python"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[setup] unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

case "$PROFILE" in
    habitat_native)
        echo "[setup] profile habitat_native is retained as a compatibility alias for native_external." >&2
        PROFILE="native_external"
        ;;
    fast_unit|native_external|legacy_optional) ;;
    *)
        echo "[setup] unsupported profile: $PROFILE" >&2
        exit 2
        ;;
esac

VALIDATION_LAYER="$PROFILE"
if [ "$PROFILE" = "legacy_optional" ]; then
    VALIDATION_LAYER="fast_unit"
fi

# Do not let an inherited checkout-era root reach installation, tests, or a
# later child process. Current native inputs have distinct explicit variables.
for retired_variable in AVENGINE_HABITAT_RUNTIME_ROOT AVENGINE_SPEAR_ROOT; do
    if [ -n "${!retired_variable:-}" ]; then
        echo "[setup] ignoring retired $retired_variable; it is not a current bootstrap input." >&2
        unset "$retired_variable"
    fi
done

resolve_conda_python() {
    local active_prefix
    local selected_executable
    local selected_prefix

    if [ -z "$PYTHON_BIN" ]; then
        if [ -n "${AVENGINE_CONDA_PYTHON:-}" ]; then
            PYTHON_BIN="$AVENGINE_CONDA_PYTHON"
            PYTHON_SOURCE="AVENGINE_CONDA_PYTHON"
        elif [ -n "${CONDA_PREFIX:-}" ]; then
            PYTHON_BIN="$CONDA_PREFIX/bin/python"
            PYTHON_SOURCE="CONDA_PREFIX"
        elif [ -n "${PYTHON:-}" ]; then
            PYTHON_BIN="$PYTHON"
            PYTHON_SOURCE="PYTHON"
        else
            echo "[setup] current bootstrap requires --python, AVENGINE_CONDA_PYTHON, or an active CONDA_PREFIX; it will not create a venv." >&2
            exit 2
        fi
    fi

    if [ ! -x "$PYTHON_BIN" ]; then
        echo "[setup] selected Conda Python is not executable ($PYTHON_SOURCE): $PYTHON_BIN" >&2
        exit 2
    fi

    readarray -t python_identity < <(
        "$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.executable).resolve()); print(Path(sys.prefix).resolve()); print(Path(sys.base_prefix).resolve())'
    )
    if [ "${#python_identity[@]}" -ne 3 ]; then
        echo "[setup] unable to inspect selected Python as a Conda environment: $PYTHON_BIN" >&2
        exit 2
    fi

    selected_executable="${python_identity[0]}"
    selected_prefix="${python_identity[1]}"
    if [ ! -d "$selected_prefix/conda-meta" ]; then
        echo "[setup] selected Python must resolve to a Conda environment with conda-meta: $selected_executable (prefix $selected_prefix)" >&2
        exit 2
    fi
    case "$selected_executable" in
        "$selected_prefix"/*) ;;
        *)
            echo "[setup] selected Python escapes its Conda prefix: $selected_executable (prefix $selected_prefix)" >&2
            exit 2
            ;;
    esac

    if [ -n "${CONDA_PREFIX:-}" ]; then
        if ! active_prefix="$(cd "$CONDA_PREFIX" && pwd -P)"; then
            echo "[setup] active CONDA_PREFIX is not a directory: $CONDA_PREFIX" >&2
            exit 2
        fi
        if [ "$active_prefix" != "$selected_prefix" ]; then
            echo "[setup] selected Conda Python prefix differs from active CONDA_PREFIX: $selected_prefix != $active_prefix" >&2
            exit 2
        fi
    fi

    PYTHON_BIN="$selected_executable"
    SELECTED_CONDA_PREFIX="$selected_prefix"
}

resolve_conda_python

require_native_directory() {
    local owner="$1"
    local raw="$2"
    if [ ! -d "$raw" ]; then
        echo "[setup] $owner must be an existing directory: $raw" >&2
        exit 2
    fi
    if ! NATIVE_RESOLVED_PATH="$(cd "$raw" && pwd -P)"; then
        echo "[setup] unable to resolve $owner: $raw" >&2
        exit 2
    fi
}

reject_native_git_ancestor() {
    local owner="$1"
    local candidate="$2"
    local parent
    while :; do
        if [ -e "$candidate/.git" ] || [ -L "$candidate/.git" ]; then
            echo "[setup] $owner must resolve outside a Git checkout: $candidate" >&2
            exit 2
        fi
        parent="$(dirname "$candidate")"
        if [ "$parent" = "$candidate" ]; then
            return
        fi
        candidate="$parent"
    done
}

preflight_native_external() {
    local habitat_prefix
    local magnum_site
    local mp3d_root
    local rlr_root
    local required_file

    require_native_directory \
        AVENGINE_HABITAT_RUNTIME_PREFIX \
        "$AVENGINE_HABITAT_RUNTIME_PREFIX"
    habitat_prefix="$NATIVE_RESOLVED_PATH"
    reject_native_git_ancestor AVENGINE_HABITAT_RUNTIME_PREFIX "$habitat_prefix"

    require_native_directory \
        AVENGINE_HABITAT_MAGNUM_PYTHON_SITE \
        "$AVENGINE_HABITAT_MAGNUM_PYTHON_SITE"
    magnum_site="$NATIVE_RESOLVED_PATH"
    reject_native_git_ancestor AVENGINE_HABITAT_MAGNUM_PYTHON_SITE "$magnum_site"

    require_native_directory AVENGINE_MP3D_ROOT "$AVENGINE_MP3D_ROOT"
    mp3d_root="$NATIVE_RESOLVED_PATH"
    reject_native_git_ancestor AVENGINE_MP3D_ROOT "$mp3d_root"
    if [ ! -d "$mp3d_root/scene_datasets" ]; then
        echo "[setup] AVENGINE_MP3D_ROOT must contain scene_datasets: $mp3d_root" >&2
        exit 2
    fi

    require_native_directory AVENGINE_RLR_SDK_ROOT "$AVENGINE_RLR_SDK_ROOT"
    rlr_root="$NATIVE_RESOLVED_PATH"
    reject_native_git_ancestor AVENGINE_RLR_SDK_ROOT "$rlr_root"
    for required_file in \
        headers/RLRAudioPropagation.h \
        libs/linux/x64/libRLRAudioPropagation.so \
        LICENSE \
        README.md; do
        if [ ! -f "$rlr_root/$required_file" ]; then
            echo "[setup] AVENGINE_RLR_SDK_ROOT is missing $required_file: $rlr_root" >&2
            exit 2
        fi
    done
}

if [ "$PROFILE" = "native_external" ]; then
    missing_inputs=()
    for required_variable in \
        AVENGINE_HABITAT_RUNTIME_PREFIX \
        AVENGINE_HABITAT_MAGNUM_PYTHON_SITE \
        AVENGINE_MP3D_ROOT \
        AVENGINE_RLR_SDK_ROOT; do
        if [ -z "${!required_variable:-}" ]; then
            missing_inputs+=("$required_variable")
        fi
    done
    if [ "${#missing_inputs[@]}" -gt 0 ]; then
        echo "[setup] native_external requires explicit: ${missing_inputs[*]}" >&2
        exit 2
    fi
    preflight_native_external
fi

run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '[setup] DRY-RUN:'
        printf ' %q' "$@"
        printf '
'
    else
        "$@"
    fi
}

echo "[setup] profile=$PROFILE"
echo "[setup] conda_prefix=$SELECTED_CONDA_PREFIX python=$PYTHON_BIN source=$PYTHON_SOURCE"
if [ "$PROFILE" = "legacy_optional" ]; then
    echo "[setup] legacy_optional does not install UE, datasets, gpuRIR or generative-model dependencies."
    echo "[setup] see docs/legacy/OPTIONAL_BACKENDS.md"
fi
if [ "$PROFILE" = "native_external" ]; then
    echo "[setup] native_external validates explicit non-Git Habitat/Magnum/MP3D/RLR paths; it does not build or clone them."
fi

run "$PYTHON_BIN" -m pip install -e "$REPOSITORY_ROOT[test]"

if [ "$DRY_RUN" = "0" ]; then
    eval "$("$PYTHON_BIN" "$SCRIPT_DIR/load_paths.py" --export)"
fi

run "$PYTHON_BIN" "$SCRIPT_DIR/load_paths.py" --validate --layer "$VALIDATION_LAYER"
run "$PYTHON_BIN" "$SCRIPT_DIR/validate_schemas.py"
if [ "$SKIP_TESTS" = "0" ]; then
    run "$PYTHON_BIN" -m pytest -q tests/unit -m "not integration and not canary"
fi

echo "[setup] AVEngine bootstrap complete for profile=$PROFILE."
echo "[setup] Native Habitat/RLR, external UE, Blender and media canaries remain separate test layers."
