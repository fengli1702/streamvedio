#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/tools/split_manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "manifest not found: $MANIFEST" >&2
  exit 1
fi
while IFS=$'\t' read -r rel size sha glob n; do
  [[ "$rel" == "relpath" ]] && continue
  out="$ROOT/$rel"
  dir="$(dirname "$out")"
  mkdir -p "$dir"
  tmp="$out.reassemble_tmp"
  cat $ROOT/$glob > "$tmp"
  got_size=$(stat -c %s "$tmp")
  got_sha=$(sha256sum "$tmp" | awk '{print $1}')
  if [[ "$got_size" != "$size" || "$got_sha" != "$sha" ]]; then
    echo "verify failed: $rel" >&2
    rm -f "$tmp"
    exit 2
  fi
  mv "$tmp" "$out"
  echo "restored $rel"
done < "$MANIFEST"
