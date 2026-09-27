# Building d4r

These steps produce the pieces the launcher needs: a patched ZLUDA, a patched vkd3d-proton, the NGX shim and CUDA bridge, a staged runtime directory with your NVIDIA files, and the native kernels. The versions below are the ones that were tested; others may need changes.

| Component | Tested version |
|---|---|
| ZLUDA | `ee2f25a` (upstream), plus `patches/zluda/0002`, `0003`, `0004`, `0005` in that order |
| vkd3d-proton | `3dfc6f07` (the base GE-Proton11-3 ships), plus `patches/vkd3d-proton/0001` |
| ROCm | 7.2 (HIP runtime, clang, device libraries) |
| Proton | GE-Proton11-3 (with its OptiScaler integration) |
| DLSS | `nvngx_dlss.dll` 310.7.0 |

## 1. ZLUDA

```sh
git clone https://github.com/vosen/ZLUDA zluda && cd zluda
git checkout ee2f25a
git submodule update --init --recursive
git lfs pull
for p in 0002 0003 0004 0005; do git apply /path/to/d4r/patches/zluda/$p-*.patch; done
# rebuild the device helpers the patches changed (ptx/lib/zluda_ptx_impl*.bc)
ZLUDA_SOURCE_ROOT=$PWD ROCM_ROOT=/opt/rocm /path/to/d4r/scripts/build_zluda_ptx_helpers.sh
LIBRARY_PATH=/opt/rocm/lib cargo build --release -p zluda
mkdir -p ~/.cache/d4r-zluda-current
cp target/release/libnvcuda.so ~/.cache/d4r-zluda-current/
ln -sf libnvcuda.so ~/.cache/d4r-zluda-current/libcuda.so
```

The launcher looks for ZLUDA in `D4R_ZLUDA_DIR` (default `~/.cache/d4r-zluda-current`, or `ZludaDir` in d4r.ini).

What the patches add:

- **0002**: the CUDA driver API surface NGX and DLSS use (arrays, textures, surfaces, 1010102 formats, launch parameter buffers), the PTX features DLSS kernels need, and module dumps for debugging (`D4R_ZLUDA_DUMP_DIR`).
- **0003**: an implicit 256-thread launch bound for kernels without PTX bounds (removes massive register spilling, `D4R_ZLUDA_IMPLICIT_MAX_BLOCK`) and f16 tensor-core MMA on RDNA3 WMMA (`D4R_ZLUDA_WMMA`).
- **0004**: the native kernel override hook (`D4R_ZLUDA_NATIVE_DIR`, see [native-kernels.md](native-kernels.md)), FP8 MMA on WMMA (`D4R_ZLUDA_WMMA_FP8`), optional elision of per-instruction denormal mode switches (`D4R_ZLUDA_IGNORE_DENORMAL`), inlined image helpers, and linking extra bitcode into a PTX module (`D4R_ZLUDA_EXTRA_BC`, used by the texture-kernel build).
- **0005**: a null texture object (CUDA handle 0) reads as zeros, as on NVIDIA GPUs, instead of faulting the GPU. DLSS samples absent optional inputs that way in some configurations (low-resolution motion vectors without HDR, as in Ghost of Tsushima).

## 2. vkd3d-proton

```sh
scripts/build_vkd3d_proton_d4r.sh ~/.cache/d4r-vkd3d-d4r
```

The patch lets the shim split the game's command list at the DLSS call, so the rest of the frame waits (at queue level) for DLSS instead of showing the previous frame's result. Point `D4R_VKD3D_DIR` (or `VkD3DDir` in d4r.ini) at the output. Without it the shim falls back to showing a one-frame-old result.

## 3. Shim, bridge and NVAPI identity

```sh
scripts/build_d4r_nvngx_shim.sh      # build/d4r_nvngx.dll   (needs x86_64-w64-mingw32-g++ and clang-cl)
scripts/build_wine_nvcuda_bridge.sh  # build/wine-nvcuda/    (needs winegcc and winebuild)
```

The launcher builds the small NVAPI identity bridge (`scripts/build_d4r_nvapi_identity.sh`) itself.

## 4. Runtime directory

```sh
scripts/install_d4r_runtime.sh /path/to/_nvngx.dll /path/to/nvngx_dlss.dll
```

This stages the shim, the bridge, the NGX core and the DLSS feature library in `D4R_RUNTIME_DIR` (default `~/.local/share/d4r-dlss`, or `RuntimeDir` in d4r.ini). The NVIDIA files come from your own sources, for example an NVIDIA driver package (NGX core) and a game or the DLSS SDK (feature library).

## 5. Native kernels

```sh
D4R_ROCM_DIR=/opt/rocm \
D4R_DLSS_DLL=/path/to/nvngx_dlss.dll \
D4R_ZLUDA_BUILD=~/.cache/d4r-zluda-current \
kernels/build.sh all kernels/out/native
```

`kernels/out/native` then holds one code object per replaced DLSS kernel; set `NativeKernelDirFast` in d4r.ini (or `D4R_ZLUDA_NATIVE_DIR`) to it. `kernels/build.sh k` or `m` builds only the network layers and needs neither the DLL nor ZLUDA. The texture kernels (`tex`) extract PTX from your DLL into `kernels/extracted/`; that directory and the build output are git-ignored and must not be redistributed.

## 6. Configure and play

The first launch copies `config/d4r.ini.default` to `~/.config/d4r/d4r.ini`. Set the `[Launch]` paths (Proton, the game's compatdata prefix and the game executable), the `[Paths]` above, and the model in `[DLSS]`, then run:

```sh
scripts/d4r_play.sh
```

In the game, pick DLSS as the upscaler (OptiScaler intercepts it). Every setting in the file can also be given as an environment variable, which takes precedence; `D4R_NO_CONFIG=1` ignores the file.

## 7. Package a release

```sh
scripts/fetch_rocm_runtime.sh         # -> ~/.cache/d4r-rocm-runtime (AMD's ROCm 7.2.4 runtime, checksummed)
D4R_OPTISCALER=/path/to/OptiScaler_0.9.4.7z \
D4R_DLSS_DLLS=/path/to/310.7/nvngx_dlss.dll:/path/to/310.9/nvngx_dlss.dll \
D4R_BUNDLE_DLSS=/path/to/nvngx_dlss.dll D4R_BUNDLE_NGX=/path/to/_nvngx.dll D4R_BUNDLE_TEX=kernels/out/native \
D4R_ZLUDA_DIR=~/.cache/d4r-zluda-current D4R_VKD3D_DIR=~/.cache/d4r-vkd3d-d4r D4R_ROCM_DIR=/opt/rocm \
scripts/package_release.sh            # -> dist/d4r-<version>.zip
```

The script:
- builds the shim, the bridge and the network-layer kernels;
- writes the kernel manifest from the DLLs you list (it records hashes of their PTX, nothing else);
- stages OptiScaler as `dxgi.dll` with the settings in `packaging/optiscaler.settings`;
- adds NVIDIA's two DLLs and the texture kernels (built from NVIDIA's PTX by `kernels/build.sh tex`);
- zips the result together with the ZLUDA and vkd3d-proton builds, the ROCm runtime (as `d4r/rocm`), `packaging/d4r.ini`, the licenses and the patches.

The NVIDIA files are not covered by d4r's license; redistributing them is up to whoever publishes the zip. `D4R_BUNDLE_NVIDIA=0` builds `d4r-<version>-nonvidia.zip` without them and without the texture kernels. [architecture.md](architecture.md#portable-installs-the-release-zip) describes how the installed files work together.

## Checks

- `scripts/check_environment.sh` lists the tools, GPU and Proton builds it finds.
- The D3D12 harness (`scripts/run_d3d12_dlss_harness_proton.sh`) drives DLSS outside a game.
- Native kernels have their own validation path; see [native-kernels.md](native-kernels.md).
