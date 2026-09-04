#!/bin/zsh
# Download the HM3D val split into the shared dataset root.
#
# Credentials are read from ~/.hm3d_token (line 1 = API token id, line 2 =
# secret) rather than being passed on a command line, so they never appear in a
# process listing, a shell history or a log.
#
# habitat-sim here is 0.2.2, which SoundSpaces 2.0 pins, and its downloader
# names the HM3D pieces separately - there is no combined "_full" uid on this
# version. val is the split that covers the 36 scenes the local JAEGER data
# uses, all of which are in the 00800-00894 range.
set -eu
TOKEN_FILE=${HOME}/.hm3d_token
if [ ! -f "$TOKEN_FILE" ]; then
  echo "missing $TOKEN_FILE (line 1 = token id, line 2 = token secret)" >&2
  exit 2
fi
ID=$(sed -n '1p' "$TOKEN_FILE")
SECRET=$(sed -n '2p' "$TOKEN_FILE")
if [ -z "$ID" ] || [ -z "$SECRET" ]; then
  echo "$TOKEN_FILE must hold the token id on line 1 and the secret on line 2" >&2
  exit 2
fi

DEST=${AVENGINE_HM3D_DATA_ROOT:-/data/datasets/habitat_data}
PY=${AVENGINE_HABITAT_DOWNLOAD_PYTHON:-/data/jzy/miniconda3/envs/ss2/bin/python}
if [ ! -x "$PY" ]; then
  echo "Habitat downloader Python is not executable: $PY" >&2
  exit 2
fi
mkdir -p "$DEST"

for uid in hm3d_val_habitat hm3d_val_glb hm3d_val_configs; do
  echo "=== $uid  $(date +%H:%M:%S)"
  for attempt in 1 2 3; do
    if $PY -m habitat_sim.utils.datasets_download \
         --username "$ID" --password "$SECRET" \
         --uids "$uid" --data-path "$DEST"; then
      break
    fi
    echo "--- $uid attempt $attempt failed, retrying"
    sleep 20
  done
done

echo "=== downloaded"
du -sh "$DEST"
find "$DEST" -maxdepth 4 -type d -name hm3d | head
