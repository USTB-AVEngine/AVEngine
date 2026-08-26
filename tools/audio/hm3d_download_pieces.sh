#!/bin/bash
# Fetch HM3D pieces for one split.
#
# habitat-sim's own downloader builds a curl command line holding
# --user id:secret and hands it to a shell, so the credentials land in that
# subprocess's argv where anyone on the box can read them out of ps. This runs
# the same transfers through curl -K, where they sit in a 0600 config file for
# the life of the transfer and never appear in an argument list.
#
# Two things about the naming are load-bearing. Semantic annots are .tar.gz at
# v0.1 and a plain .tar at v0.2, so the extension follows the version. And the
# archives arrive with different prefixes - some carry hm3d/<split>/, some carry
# the scene directories bare - so each one is inspected before extraction rather
# than assumed.
#
# usage: hm3d_download_pieces.sh <split> <piece> [piece ...]
#   piece is glb | habitat | configs | semantic-annots-v0.2 | semantic-configs-v0.2
set -u

SPLIT=${1:?usage: $0 <split> <piece> [piece ...]}
shift
[ $# -ge 1 ] || { echo "name at least one piece" >&2; exit 2; }

TOKEN_FILE=${HOME}/.hm3d_token
DEST=/data/datasets/habitat_data
HM3D=$DEST/versioned_data/hm3d-1.0/hm3d

[ -f "$TOKEN_FILE" ] || { echo "missing $TOKEN_FILE" >&2; exit 2; }
ID=$(sed -n '1p' "$TOKEN_FILE")
SECRET=$(sed -n '2p' "$TOKEN_FILE")
[ -n "$ID" ] && [ -n "$SECRET" ] || { echo "$TOKEN_FILE needs id then secret" >&2; exit 2; }

WORK=$(mktemp -d "$DEST/.hm3ddl.XXXXXX")
CONF=$WORK/curl.conf
umask 077
printf 'user = "%s:%s"\nlocation\nfail\nshow-error\n' "$ID" "$SECRET" > "$CONF"
chmod 600 "$CONF"
unset ID SECRET
trap 'shred -u "$CONF" 2>/dev/null || rm -f "$CONF"; rm -rf "$WORK"' EXIT

mkdir -p "$HM3D/$SPLIT"
base=https://api.matterport.com/resources/habitat
for piece in "$@"; do
  ext=.tar
  case "$piece" in *-v0.1) ext=.tar.gz;; esac
  case "$piece" in semantic-annots-v0.1) ext=.tar.gz;; esac
  url="$base/hm3d-$SPLIT-$piece$ext"
  out="$WORK/hm3d-$SPLIT-$piece$ext"
  echo "=== $SPLIT $piece  $(date +%H:%M:%S)"
  if ! curl -K "$CONF" --progress-bar -o "$out" "$url"; then
    echo "--- failed: $url" >&2
    continue
  fi
  ls -l "$out"
  first=$(tar -tf "$out" 2>/dev/null | head -1)
  echo "    first entry: ${first:-<unreadable>}"
  if [ -z "$first" ]; then
    echo "--- not a readable archive, skipping extraction" >&2
    continue
  fi
  case "$first" in
    hm3d/*) tar -xf "$out" -C "$DEST/versioned_data/hm3d-1.0/.." ;;
    *scene_dataset_config.json) tar -xf "$out" -C "$HM3D" ;;
    *) tar -xf "$out" -C "$HM3D/$SPLIT" ;;
  esac
  echo "    extracted"
  rm -f "$out"
done

echo "=== result for $SPLIT"
echo "  scene directories: $(ls -d "$HM3D/$SPLIT"/*/ 2>/dev/null | wc -l)"
echo "  uncompressed glb:  $(ls "$HM3D/$SPLIT"/*/*.glb 2>/dev/null | grep -vc basis || true)"
echo "  semantic glb:      $(ls "$HM3D/$SPLIT"/*/*.semantic.glb 2>/dev/null | wc -l)"
du -sh "$HM3D"
