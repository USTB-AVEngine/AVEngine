#!/bin/bash
# Convenience alias for `bash scripts/setup.sh --update`.
# Only affects cloned external repos (symlinks are left alone; use
# `git pull` in the linked directory instead).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/setup.sh" --update "$@"
