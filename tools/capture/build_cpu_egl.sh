#!/usr/bin/env bash
# Build only into a caller-owned task directory, never an installed runtime.
set -euo pipefail
if [ "$#" -ne 1 ]; then echo 'usage: build_cpu_egl.sh OUTPUT_DIRECTORY' >&2; exit 2; fi
output=$(realpath -m "$1")
repo=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$output"
test ! -e "$output/libegl_cpu_only.so"
cc -shared -fPIC -O2 -Wl,-Bsymbolic -o "$output/libegl_cpu_only.so" "$repo/native/software_egl/cpu_only_egl.c" -ldl
printf '%s\n' "$output/libegl_cpu_only.so"
