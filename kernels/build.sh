#!/usr/bin/env bash
# Builds the native RDNA3 replacements for DLSS kernels into one directory that ZLUDA serves them from
# (D4R_ZLUDA_NATIVE_DIR, or NativeKernelDirFast in d4r.ini). See docs/native-kernels.md.
#
# usage: kernels/build.sh [all|k|m|tex] [OUT_DIR]      (default: all, kernels/out/native)
#
#   D4R_ROCM_DIR    ROCm installation with clang and the HIP device libraries (default /opt/rocm)
#   D4R_GPU_ARCH    target GPU (default gfx1101; the kernels need RDNA3 / gfx11 WMMA)
# Texture kernels (tex) are NVIDIA's PTX with parts replaced, so they additionally need:
#   D4R_DLSS_DLL    nvngx_dlss.dll 310.7 (its PTX is extracted into kernels/extracted/, never committed)
#   D4R_ZLUDA_BUILD a ZLUDA build with patches/zluda applied (directory holding libcuda.so)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHAT="${1:-all}"
OUT="$(realpath -m "${2:-$HERE/out/native}")"
ROCM="${D4R_ROCM_DIR:-/opt/rocm}"
ARCH="${D4R_GPU_ARCH:-gfx1101}"
CLANG="$ROCM/lib/llvm/bin/clang++"
[[ -x "$CLANG" ]] || { echo "clang++ not found in $ROCM/lib/llvm/bin (set D4R_ROCM_DIR)" >&2; exit 2; }
mkdir -p "$OUT"

# HIP source -> raw code object (the kernel name inside matches the DLSS kernel it replaces)
build_hip() {
    local src="$1" name
    name="$(basename "$src" .hip)"
    local tmp
    tmp="$(mktemp -d)"
    (cd "$tmp" && "$CLANG" -x hip --offload-arch="$ARCH" --offload-device-only -O3 -I"$ROCM/include" \
        --rocm-path="$ROCM" --rocm-device-lib-path="$ROCM/amdgcn/bitcode" -I"$(dirname "$src")" \
        -o "$OUT/$name.hsaco" "$src" -save-temps=cwd --no-gpu-bundle-output)
    printf '%-52s %s\n' "$name" "$(grep -hE '^; (NumVgprs|ScratchSize|Occupancy)' "$tmp"/*.s | tail -3 | paste -sd' ')"
    rm -rf "$tmp"
}

if [[ "$WHAT" == all || "$WHAT" == k ]]; then
    echo "== DLSS 4 (preset K) transformer layers"
    for src in "$HERE"/k/dltss_pwin_*.hip; do build_hip "$src"; done
fi
if [[ "$WHAT" == all || "$WHAT" == m ]]; then
    echo "== DLSS 4.5 (preset M) Swin layers"
    for src in "$HERE"/m/rrlite_*.hip; do build_hip "$src"; done
fi
if [[ "$WHAT" == all || "$WHAT" == tex ]]; then
    echo "== texture kernels (NVIDIA PTX with native parts)"
    : "${D4R_DLSS_DLL:?set D4R_DLSS_DLL to nvngx_dlss.dll (310.7)}"
    : "${D4R_ZLUDA_BUILD:?set D4R_ZLUDA_BUILD to a ZLUDA build with patches/zluda}"
    PTX_DIR="$HERE/extracted/ptx"
    python3 "$HERE/tools/extract_dlss_ptx.py" "$D4R_DLSS_DLL" "$PTX_DIR"
    PTXLOAD="$HERE/out/tools/ptxload"
    mkdir -p "$(dirname "$PTXLOAD")"
    cc -O2 -o "$PTXLOAD" "$HERE/tools/ptxload.c" -ldl
    # kernel : source of its native part
    for pair in hiluma_engine_output_depthinv_mvhi_hdr_max_v2_rel:sust_only rrlite_post_3_2_mvhi_hdr_folded:sust_only \
        rrlite_downsample_kernel_static_hdr:sust_only rrlite_enc0_4x4_mvhi_hdr_folded:enc0_tail \
        rrlite_dec0_4x4_folded:dec0_head; do
        D4R_ROCM_DIR="$ROCM" D4R_GPU_ARCH="$ARCH" D4R_DLSS_PTX_DIR="$PTX_DIR" D4R_PTXLOAD="$PTXLOAD" \
            "$HERE/tex/build_tex.sh" "${pair%%:*}" "${pair#*:}" "$OUT"
    done
fi
echo "native kernels in $OUT"
