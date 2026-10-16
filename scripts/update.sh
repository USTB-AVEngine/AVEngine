#!/usr/bin/env bash
# Re-run the current Habitat-native bootstrap. Dependency pin changes are made
# in manifest.yaml; this helper never checks out or overwrites a runtime tree.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/setup.sh" "$@"
