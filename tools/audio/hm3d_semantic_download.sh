#!/bin/bash
# Fetch the HM3D semantic annotations for the val split.
#
# habitat-sim's own downloader builds a curl command line holding
# --user id:secret and hands it to a shell, so the credentials land in that
# subprocess's argv where anyone on the box can read them out of ps. This runs
# the same transfer through curl -K, where they sit in a 0600 config file and
# never appear in an argument list.
#
# Version matters and the extension changes with it: v0.1 ships annots as
# .tar.gz, v0.2 as plain .tar. v0.2 is what is wanted - v0.1 annotates only 20
# of the 36 val scenes JAEGER uses, while v0.2 covers the split, which is also
# why JAEGER reporting exactly 36 test scenes points at v0.2.
set -u

TOKEN_FILE=${HOME}/.hm3d_token
DEST=/data/datasets/habitat_data
HM3D=$DEST/versioned_data/hm3d-1.0/hm3d

[ -f "$TOKEN_FILE" ] || { echo "missing $TOKEN_FILE" >&2; exit 2; }
ID=$(sed -n '1p' "$TOKEN_FILE")
SECRET=$(sed -n '2p' "$TOKEN_FILE")
[ -n "$ID" ] && [ -n "$SECRET" ] || { echo "$TOKEN_FILE needs id on line 1, secret on line 2" >&2; exit 2; }

WORK=$(mktemp -d "$DEST/.hm3dsem.XXXXXX")
CONF=$WORK/curl.conf
umask 077
printf 'user = "%s:%s"\nlocation\nfail\nsilent\nshow-error\n' "$ID" "$SECRET" > "$CONF"
chmod 600 "$CONF"
unset ID SECRET
trap 'shred -u "$CONF" 2>/dev/null || rm -f "$CONF"; rm -rf "$WORK"' EXIT

echo "=== clearing any earlier semantic version so the tree holds one only"
find "$HM3D/val" -maxdepth 2 -name '*.semantic.glb' -delete
find "$HM3D/val" -maxdepth 2 -name '*.semantic.txt' -delete
rm -f "$HM3D/val"/hm3d_annotated_*.json "$HM3D"/hm3d_annotated_*.json

base=https://api.matterport.com/resources/habitat
for fmt in annots configs; do
  url="$base/hm3d-val-semantic-$fmt-v0.2.tar"
  out="$WORK/hm3d-val-semantic-$fmt-v0.2.tar"
  echo "=== $fmt v0.2  $(date +%H:%M:%S)"
  curl -K "$CONF" -o "$out" "$url" || { echo "failed: $fmt" >&2; exit 1; }
  ls -l "$out"
  first=$(tar -tf "$out" | head -1)
  echo "    first entry: $first"
  if [ "$fmt" = configs ]; then
    tar -xf "$out" -C "$HM3D"
  else
    tar -xf "$out" -C "$HM3D/val"
  fi
done

echo "=== result"
ls "$HM3D/val/00800-TEEsavR23oF/"
ls "$HM3D" | head
echo "--- scenes with semantics: $(ls "$HM3D"/val/*/*.semantic.glb 2>/dev/null | wc -l)"
miss=0
for s in $(ls /data/datasets/JAEGER/simulation_ds/test); do
  n=${s#*-}
  [ -f "$HM3D/val/$s/$n.semantic.glb" ] || { echo "MISSING $s"; miss=$((miss+1)); }
done
echo "--- JAEGER scenes missing semantics: $miss of 36"
du -sh "$HM3D"
