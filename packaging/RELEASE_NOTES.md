# d4r 0.1.0

NVIDIA's own DLSS Super Resolution on AMD Radeon RX 7000 (RDNA3) GPUs, in DirectX 12 games under Proton. This is the first release packaged as a drop-in zip, to install extract it next to the game's executable and set one line of launch options.

It is an early release, tested on one GPU (RX 7700 XT) and one game (SILENT HILL Townfall).

## Install

1. Extract `d4r-0.1.0.zip` into the folder that holds the game's main `.exe`. For Unreal Engine games that is `<Project>/Binaries/Win64`.
2. In Steam, force GE-Proton 11 for the game, and set these launch options:
   ```
   PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 %command%
   ```
3. Select DLSS in the game. The first start compiles DLSS's GPU kernels, which takes about a minute and a half; later starts use the cache in `~/.cache/d4r`.

`D4R_README.txt` in the zip has the details, and `sh d4r/d4r-check.sh` checks an install.

## Requirements

- AMD Radeon RX 7000 series. The fast kernels are built for gfx1101 (RX 7700 XT, RX 7800 XT). Other RDNA3 cards run DLSS without them, which is much slower.
- The ROCm HIP runtime 7.x (`libamdhip64.so.7`), from your distribution. ROCm 7.2 was tested.
- GE-Proton 11 (tested: GE-Proton11-3).

## What is in the zip

- **d4r:** the NGX shim, the Wine CUDA bridge, and hand-written RDNA3 kernels for the DLSS 4 (K) and DLSS 4.5 (M) network layers.
  - Each kernel is used only when the DLSS library's code for it matches the code it was written for. Swapping in another DLSS version therefore still works, only slower.
  - `d4r/d4r.ini` holds the settings for that game. The default model is DLSS 4 (K); E (DLSS 3 CNN) and M (DLSS 4.5) are one line away.
- **NVIDIA:** DLSS 310.7.0 (`nvngx_dlss.dll`), the NGX runtime from driver 596.36 (`_nvngx.dll`), and five DLSS kernels built from NVIDIA's code with parts replaced by d4r's.
- **ZLUDA** with d4r's patches: CUDA on ROCm.
- **vkd3d-proton** with d4r's patch, so every frame shows its own DLSS result (no added latency).
- **OptiScaler 0.9.4**, unmodified, set up to hand the game's DLSS calls to d4r.

Nothing in the Proton prefix or the system is changed. Uninstalling means deleting the extracted files.

## Performance

Radeon RX 7700 XT, SILENT HILL Townfall at 2560×1440, average fps over the same 62-second walk:

| Mode | DLSS 3 CNN (E) | DLSS 4 (K) | DLSS 4.5 (M) | FSR 4 |
|---|---|---|---|---|
| Quality | 72 | 69 | 51 | 76 |
| Balanced | 80 | 76 | 59 | 84 |
| Performance | 88 | 82 | 69 | 94 |
| Ultra Performance | 89 | 93 | 90 | 107 |

Native 2560×1440 without upscaling (the game's TSR at 100%) runs at 49 fps, so DLSS 4 at Quality is 40% faster than native.

DLSS 4 in every mode, and E and M at Quality, were measured with this release. The other E and M figures come from the development setup, which runs the same kernels.

## Known limitations

- Only DLSS Super Resolution in DirectX 12 games. No Frame Generation or Ray Reconstruction.
- Tested with one game. Do not use it in games with anti-cheat.
- DLSS 4.5 (M) is much slower than FSR 4 on RDNA3.

## Licenses

d4r is Apache 2.0. The zip also contains ZLUDA (Apache 2.0 or MIT), vkd3d-proton (LGPL 2.1, patched; the patch is in `d4r/source`) and OptiScaler 0.9.4 (GPL 3.0, unmodified; source at https://github.com/optiscaler/OptiScaler/tree/v0.9.4). `d4r/source/SOURCES.txt` lists where every file comes from.

NVIDIA's files in the zip, and the kernels built from NVIDIA's code, belong to NVIDIA and are not covered by any of these licenses. d4r is not affiliated with NVIDIA, AMD or the OptiScaler project.

**SHA-256** `d4r-0.1.0.zip`: `c26670538cdd122f9766bb30c6099c0636e4f8365f6ef82109c3671eea2f8564`
