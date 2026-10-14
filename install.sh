#!/usr/bin/env bash
# NVIDIA installer: beside the proxy release's d4r/ or inside the ASI's d4r/.
# Installs NVIDIA's DLLs and then builds the native engine's models from them on this machine.
set -euo pipefail
DLSS_URL=https://raw.githubusercontent.com/NVIDIA/DLSS/a291cc7d2cc642a51566f3dfd5376f635cd1b284/lib/Windows_x86_64/rel/nvngx_dlss.dll
DLSS_SHA256=be6e434a94ca32499515eb62ca0e6c274526055d568d0426e4c652dcdfb6ee6e
DLSSD_URL=https://raw.githubusercontent.com/NVIDIA/DLSS/374959484e79a640feaba44c93ac8cfb0a03f5b5/lib/Windows_x86_64/rel/nvngx_dlssd.dll
DLSSD_SHA256=4bc7ea5fcb2f32cf86bc2cb072e8d2860914a0be511cbdcb2fc206c1d9d83b80
DRIVER_URL=https://us.download.nvidia.com/Windows/596.36/596.36-desktop-win10-win11-64bit-international-dch-whql.exe
DRIVER_SHA256=4a1793d2bc7792f43270fe8286ec381ae24409fe29049dbe1eb8530ed89e19cf
NGX_SHA256=0e3423b3b7afa1019c7ecac053a8408df8e5d9d9d904434705c6aaeacec02fef

fail() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }
usage() {
  printf '%s\n' 'Usage: bash install.sh [--no-engine | --engine-only] [--rebuild-engine] [extracted-release-directory]' \
    'Default: the directory containing this script, regardless of your working directory.' \
    '1. Downloads official NVIDIA DLSS 310.7.0 (Super Resolution), DLSS Ray Reconstruction 310.9.1 and NGX from' \
    '   driver 596.36 (~1 GiB in all). Existing matching DLLs are kept; differing DLLs are backed up.' \
    '2. Builds the native engine'"'"'s models ([Engine] in d4r/d4r.ini) from those DLLs into d4r/engine:' \
    '   Ray Reconstruction D, E and F from nvngx_dlssd.dll; Super Resolution K, M and L (and their LDR variants)' \
    '   by rendering one test frame per model through d4r under GE-Proton (about a minute each).' \
    '   --no-engine skips this, --engine-only does only this, --rebuild-engine rebuilds models already built.' \
    'Requires Bash, curl, sha256sum, 7zz/7z/7za, python3 with numpy, and GE-Proton 11 in Steam (or' \
    'D4R_PROTON=/path/to/GE-Proton11-3). No sudo needed. NVIDIA license terms apply.'
}
DO_DLLS=1 DO_ENGINE=1 REBUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --no-engine) DO_ENGINE=0; shift ;;
    --engine-only) DO_DLLS=0; shift ;;
    --rebuild-engine) REBUILD=1; shift ;;
    -*) usage >&2; exit 2 ;;
    *) break ;;
  esac
done
[[ $# -le 1 ]] || { usage >&2; exit 2; }
ROOT=$(cd -- "${1:-$(dirname -- "${BASH_SOURCE[0]}")}" && pwd)
RUNTIME="$ROOT/d4r"
[[ ! -f "$ROOT/d4r.ini" ]] || RUNTIME="$ROOT"
[[ -f "$RUNTIME/nvngx.dll" && -f "$RUNTIME/d4r.ini" ]] || fail "Extract the release ZIP first; no d4r runtime found in $ROOT"
command -v sha256sum >/dev/null || fail 'Missing sha256sum (install coreutils).'
valid() { [[ -f "$1" ]] && [[ $(sha256sum -- "$1" | cut -d' ' -f1) == "$2" ]]; }

install_dlls() {
  local need_dlss=1 need_dlssd=1 need_ngx=1 tool
  valid "$RUNTIME/nvngx_dlss.dll" "$DLSS_SHA256" && need_dlss=0
  valid "$RUNTIME/nvngx_dlssd.dll" "$DLSSD_SHA256" && need_dlssd=0
  valid "$RUNTIME/ngx/_nvngx.dll" "$NGX_SHA256" && need_ngx=0
  if (( need_dlss == 0 && need_dlssd == 0 && need_ngx == 0 )); then
    echo 'NVIDIA DLLs are already installed and verified.'
    return
  fi
  command -v curl >/dev/null || fail 'Missing curl. Install it and run this script again.'
  SEVENZIP=''
  if (( need_ngx )); then
    for tool in 7zz 7z 7za; do
      if command -v "$tool" >/dev/null; then SEVENZIP=$tool; break; fi
    done
    [[ -n "$SEVENZIP" ]] || fail 'Missing 7-Zip. Install 7zip (or p7zip) and run this script again.'
  fi
  echo 'Downloading NVIDIA files under NVIDIA license terms:'
  echo 'DLSS: https://github.com/NVIDIA/DLSS/blob/v310.9.1/LICENSE.txt'
  echo 'Driver: https://www.nvidia.com/en-us/drivers/nvidia-license/'
  if (( need_dlss )); then
    echo 'Downloading DLSS 310.7.0...'
    download "$DLSS_URL" "$TMP/nvngx_dlss.dll"
    verify "$TMP/nvngx_dlss.dll" "$DLSS_SHA256"
  fi
  if (( need_dlssd )); then
    echo 'Downloading DLSS Ray Reconstruction 310.9.1...'
    download "$DLSSD_URL" "$TMP/nvngx_dlssd.dll"
    verify "$TMP/nvngx_dlssd.dll" "$DLSSD_SHA256"
  fi
  if (( need_ngx )); then
    echo 'Downloading NVIDIA driver 596.36 (~915 MiB); extracting NGX without running the installer...'
    download "$DRIVER_URL" "$TMP/driver.exe"
    verify "$TMP/driver.exe" "$DRIVER_SHA256"
    "$SEVENZIP" e -y -bso0 -bsp0 "-o$TMP/ngx" "$TMP/driver.exe" 'Display.Driver/_nvngx.dll' || fail 'Could not extract NGX; no DLLs installed.'
    verify "$TMP/ngx/_nvngx.dll" "$NGX_SHA256"
  fi
  # Verify all downloads before modifying any destination.
  (( need_dlss == 0 )) || place "$TMP/nvngx_dlss.dll" "$RUNTIME/nvngx_dlss.dll"
  (( need_dlssd == 0 )) || place "$TMP/nvngx_dlssd.dll" "$RUNTIME/nvngx_dlssd.dll"
  (( need_ngx == 0 )) || place "$TMP/ngx/_nvngx.dll" "$RUNTIME/ngx/_nvngx.dll"
  echo 'NVIDIA DLLs installed and verified.'
}

TMP=$(mktemp -d)
STAGED=''
trap '[[ -z "$STAGED" ]] || rm -f -- "$STAGED"; rm -rf -- "$TMP"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
download() {
  curl --fail --location --show-error --retry 3 --connect-timeout 30 \
    --proto '=https' --proto-redir '=https' --output "$2" "$1" || fail "Download failed: $1. Run the installer again to retry."
}
verify() { valid "$1" "$2" || fail "SHA-256 mismatch for $1; no DLLs installed."; }
# Stage on the destination filesystem for atomic replacement and retain a unique backup of a differing file.
place() {
  local src=$1 dst=$2 backup
  mkdir -p -- "$(dirname -- "$dst")"
  [[ ! -d "$dst" ]] || fail "Destination is a directory: $dst"
  STAGED=$(mktemp "${dst}.install.XXXXXX")
  cp -- "$src" "$STAGED"
  chmod 644 "$STAGED"
  if [[ -e "$dst" || -L "$dst" ]]; then
    backup=$(mktemp "${dst}.backup.XXXXXX")
    cp -p -- "$dst" "$backup"
    printf 'Saved previous DLL: %s\n' "$backup"
  fi
  mv -f -- "$STAGED" "$dst"
  STAGED=''
  printf 'Installed: %s\n' "$dst"
}

(( DO_DLLS == 0 )) || install_dlls
if (( DO_ENGINE == 0 )); then
  echo 'Follow the included README for Steam launch options.'
  exit 0
fi

# The engine's models, built from the DLLs above. A stamp of the DLLs and the builder skips an unchanged rebuild.
BUILDER="$RUNTIME/tools/build-engine-models.sh"
[[ -f "$BUILDER" ]] || fail "missing $BUILDER (this release has no engine-model builder; use --no-engine)"
STAMP="$RUNTIME/engine/.d4r-models"
stamp() {
  { sha256sum -- "$RUNTIME/nvngx_dlss.dll" "$RUNTIME/nvngx_dlssd.dll" 2>/dev/null | cut -d' ' -f1
    (cd "$RUNTIME/tools" && find . -type f ! -name '*.pyc' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum); } | sha256sum | cut -d' ' -f1
}
models=(rr-d rr-e rr-f k k-ldr m m-ldr l l-ldr)
complete=1
for m in "${models[@]}"; do [[ -d "$RUNTIME/engine/$m" ]] || complete=0; done
if (( REBUILD == 0 && complete )) && [[ -f "$STAMP" && "$(cat "$STAMP")" == "$(stamp)" ]]; then
  echo 'Engine models are already built for these DLLs (--rebuild-engine builds them again).'
  exit 0
fi
rm -f -- "$STAMP"
failed=()
bash "$BUILDER" --rr-only "$RUNTIME" || failed+=('Ray Reconstruction')
bash "$BUILDER" --sr-only "$RUNTIME" || failed+=('Super Resolution')
if (( ${#failed[@]} )); then
  printf 'install.sh: the engine models for %s were not built (reason above). d4r works without them on its\n' \
    "$(IFS=/; echo "${failed[*]}")" >&2
  printf 'normal path; fix the cause and run: bash install.sh --engine-only\n' >&2
  exit 1
fi
stamp > "$STAMP"
echo 'Done. Follow the included README for Steam launch options; set [Engine] Enabled = true in d4r/d4r.ini to use the engine.'
