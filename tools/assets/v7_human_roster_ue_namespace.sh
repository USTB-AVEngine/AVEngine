#!/usr/bin/env bash
set -euo pipefail
T="$1"
shift
if [[ ! -d "$T/ue_project" ]]; then exit 2; fi
mkdir -p "$T/ue_tmp"
exec /usr/bin/bwrap --die-with-parent --unshare-user --ro-bind / / --dev /dev --proc /proc --dev-bind /dev/nvidiactl /dev/nvidiactl --dev-bind /dev/nvidia1 /dev/nvidia1 --dev-bind /dev/nvidia-uvm /dev/nvidia-uvm --dev-bind /dev/nvidia-uvm-tools /dev/nvidia-uvm-tools --bind "$T" "$T" --bind "$T/ue_tmp" /tmp --chdir "$T" --unsetenv DISPLAY "$@"
