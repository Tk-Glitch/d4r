#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WINEGCC="${WINEGCC:-winegcc}"
WINEBUILD="${WINEBUILD:-winebuild}"
BRIDGE_ROOT="$ROOT/build/wine-nvcuda"

mkdir -p "$BRIDGE_ROOT/x86_64-unix" "$BRIDGE_ROOT/x86_64-windows"
"$WINEGCC" -m64 -shared -o "$BRIDGE_ROOT/x86_64-unix/nvcuda.dll" \
  "$ROOT/tools/wine_nvcuda_bridge.c" \
  -Wb,--export="$ROOT/tools/wine_nvcuda_bridge.spec" -ldl -lpthread
"$WINEBUILD" --dll --fake-module -E "$ROOT/tools/wine_nvcuda_bridge.spec" \
  -F nvcuda.dll -b x86_64-windows -o "$BRIDGE_ROOT/x86_64-windows/nvcuda.dll"
printf 'Built Wine nvcuda bridge in %s\n' "$BRIDGE_ROOT"
