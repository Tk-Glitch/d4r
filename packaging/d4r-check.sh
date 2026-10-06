#!/bin/sh
# Checks an d4r install: run it from the game folder (the one holding the game's .exe and d4r/),
# or give that folder as the argument. It only reads files and prints what it finds.
# usage: sh d4r/d4r-check.sh [GAME_FOLDER] [ROCM_DIR]
GAME="${1:-.}"
D4R="$GAME/d4r"
# ROCm: the argument, else d4r.ini's RocmDir, else the bundled d4r/rocm, else D4R_ROCM_DIR, else /opt/rocm
ini_rocm=$(sed -n 's/^[[:space:]]*RocmDir[[:space:]]*=[[:space:]]*\([^;#]*\).*/\1/p' "$D4R/d4r.ini" 2>/dev/null | tail -1 | sed 's/[[:space:]]*$//')
[ "$ini_rocm" = auto ] && ini_rocm=
case "$ini_rocm" in "~/"*) ini_rocm="$HOME/${ini_rocm#\~/}" ;; esac
[ -z "$ini_rocm" ] && [ -d "$D4R/rocm/lib" ] && ini_rocm="$D4R/rocm"
ROCM="${2:-${ini_rocm:-${D4R_ROCM_DIR:-/opt/rocm}}}"
problems=0
ok() { printf '  ok       %s\n' "$1"; }
bad() { printf '  MISSING  %s\n' "$1"; problems=$((problems + 1)); }
note() { printf '  note     %s\n' "$1"; }
misconfigured() { printf '  CONFIG   %s\n' "$1"; problems=$((problems + 1)); }
external_backend=$(awk '
  { sub(/\r$/, ""); line = $0; sub(/^[ \t]+/, "", line); sub(/[ \t]+$/, "", line) }
  line ~ /^\[/ { dlss = (tolower(line) == "[dlss]"); next }
  dlss && tolower(line) ~ /^allowexternalbackend[ \t]*=/ {
    count++; sub(/^[^=]*=[ \t]*/, "", line); value = tolower(line)
  }
  END { if (count == 1 && value == "true") print "true" }
' "$GAME/OptiScaler.ini" 2>/dev/null)

printf 'd4r install in %s\n' "$(cd "$GAME" 2>/dev/null && pwd || echo "$GAME")"
[ -d "$D4R" ] || { printf 'no d4r folder here; run this from the folder with the game .exe\n'; exit 1; }
ls "$GAME"/*.exe >/dev/null 2>&1 && ok "game executable next to d4r/" || note "no .exe next to d4r/ (it must be the game's main executable folder)"
for f in dxgi.dll OptiScaler.ini d3d12.dll d3d12core.dll d4r/nvngx.dll d4r/nvcuda.dll d4r/zluda/libcuda.so d4r/d4r.ini; do
  [ -f "$GAME/$f" ] && ok "$f" || bad "$f (re-extract the d4r zip)"
done
# File presence alone does not establish NGX routing: OptiDllPath=d4r can still
# select system32/_nvngx.dll before d4r/nvngx.dll. Check the per-DLL override.
if [ -f "$GAME/OptiScaler.ini" ]; then
  nvngx_path=$(awk '
    { sub(/\r$/, ""); line = $0; sub(/^[ \t]+/, "", line); sub(/[ \t]+$/, "", line) }
    line ~ /^\[/ { libraries = (tolower(line) == "[libraries]"); next }
    libraries && tolower(line) ~ /^nvngxpath[ \t]*=/ {
      count++; sub(/^[^=]*=[ \t]*/, "", line); value = line
    }
    END { if (count != 1) exit 1; print value }
  ' "$GAME/OptiScaler.ini") && unique_path=yes || unique_path=no
  normalized_path=$(printf '%s' "$nvngx_path" | tr '\\' '/' | tr '[:upper:]' '[:lower:]')
  # Also recognize an absolute Wine Z: path to this install's shim (as used
  # by developer launches); do not accept an arbitrary nvngx.dll elsewhere.
  game_path=$(cd "$GAME" 2>/dev/null && pwd -P)
  absolute_path=$(printf 'z:%s/d4r/nvngx.dll' "$game_path" | tr '[:upper:]' '[:lower:]')
  if [ "$unique_path" = yes ] && { [ "$normalized_path" = 'd4r/nvngx.dll' ] ||
       [ "$normalized_path" = './d4r/nvngx.dll' ] || [ "$normalized_path" = "$absolute_path" ]; }; then
    ok "OptiScaler NGX routing: NvngxPath=$nvngx_path (d4r shim)"
    case "$normalized_path" in
      z:*) ;;
      *) [ "$external_backend" = true ] || note "relative NvngxPath requires the game's working directory to be this folder" ;;
    esac
  else
    misconfigured 'OptiScaler NGX routing: set [Libraries] NvngxPath=d4r\nvngx.dll (exactly once); OptiDllPath alone can load system _nvngx.dll'
    note 'd4r/ngx/_nvngx.dll is the NVIDIA core for the shim, not the OptiScaler target'
  fi
fi
[ -f "$D4R/nvngx_dlss.dll" ] && ok "d4r/nvngx_dlss.dll (NVIDIA DLSS library)" || bad "d4r/nvngx_dlss.dll: copy NVIDIA's DLSS library here"
[ -f "$D4R/ngx/_nvngx.dll" ] && ok "d4r/ngx/_nvngx.dll (NVIDIA NGX runtime)" || bad "d4r/ngx/_nvngx.dll: copy NVIDIA's NGX runtime here"
if [ -f "$D4R/nvngx_dlss.dll" ] && command -v strings >/dev/null 2>&1; then
  v=$(strings -el "$D4R/nvngx_dlss.dll" | grep -A1 '^FileVersion$' | sed -n 2p | tr ',' '.')
  case "$v" in
    310.7.*|310.9.*) ok "DLSS version $v (native kernels verified for 310.7 and 310.9)" ;;
    "") note "cannot read the DLSS version" ;;
    *) note "DLSS version $v: kernels whose code changed run without native kernels (slower)" ;;
  esac
fi

found=
for dir in "$ROCM/lib" /opt/rocm/lib /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu; do
  [ -e "$dir/libamdhip64.so.7" ] && { found="$dir"; break; }
done
[ -n "$found" ] && ok "ROCm HIP runtime ($found/libamdhip64.so.7)" || bad "ROCm HIP runtime 7.x (libamdhip64.so.7); re-extract the d4r zip, which includes it in d4r/rocm"
[ -e /dev/kfd ] && ok "/dev/kfd (ROCm compute device)" || bad "/dev/kfd: the amdgpu compute interface is not available"
target="" best_simds=0
for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
  s=$(sed -n 's/^simd_count //p' "$props" 2>/dev/null); t=$(sed -n 's/^gfx_target_version //p' "$props" 2>/dev/null)
  [ -n "$s" ] && [ "$s" -gt "$best_simds" ] && [ -n "$t" ] && [ "$t" != 0 ] && { target="$t"; best_simds="$s"; }
done
if [ -n "$target" ]; then
  arch=$(printf 'gfx%d%d%x' $((target / 10000)) $(((target / 100) % 100)) $((target % 100)))
  if [ -d "$D4R/kernels/$arch" ]; then ok "GPU $arch: native kernels present"
  else note "GPU $arch: no native kernels for it in this release (DLSS runs, much slower)"; fi
fi

if [ "$external_backend" = true ]; then
  note "AllowExternalBackend requires the patched OptiScaler DLL; keep native AMD identity visible for FSR4"
  printf '\nSteam launch options for this patched setup:\n  PROTON_ENABLE_NVAPI=1 DXVK_NVAPI_ALLOW_OTHER_DRIVERS=1 DXVK_NVAPI_GPU_ARCH=AD100 WINE_HIDE_AMD_GPU=0 %%command%%\n'
else
  printf '\nSteam launch options for this game:\n  PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 %%command%%\n'
fi
[ "$problems" -eq 0 ] && printf '\nFiles and OptiScaler NGX routing checks passed (runtime initialization is not tested).\n' || printf '\n%d thing(s) to fix above.\n' "$problems"
[ "$problems" -eq 0 ]
