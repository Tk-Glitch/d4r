# d4r 0.1.1

NVIDIA's own DLSS Super Resolution on AMD Radeon RX 7000 (RDNA3) GPUs, in DirectX 12 games under Proton. This release fixes a VRAM leak and two input-format problems, bundles the ROCm runtime, and adds a dozen tested games.

## What's new

- **No more VRAM growth when changing the DLSS quality setting.** Each switch between Quality, Balanced, Performance and so on used to keep about 130 MB of VRAM until the game closed, because of a reference-counting bug in ROCm 7.2's external-memory mapping. d4r now reuses its shared buffers, and VRAM stays flat over repeated switches. It also no longer leaks a file handle per buffer.
- **Games that pass 4-channel motion vectors now work.** Motion vectors in `R16G16B16A16` formats (including `TYPELESS`, as Dying Light: The Beast uses) used to be rejected, which left the screen black. They are now converted on the GPU and stay on the fast VRAM path.
- **Multi-channel exposure textures stay on the VRAM path.** Games that pass exposure as an RGBA float texture no longer fall back to the slower copy through system memory, which showed up as reflections lagging behind (Ready or Not).
- **ROCm no longer needs to be installed.** The zip includes the ROCm 7.2.4 runtime (HIP, HSA, comgr) in `d4r/rocm`, from AMD's Ubuntu 22.04 packages; it runs under Steam's container runtime on any distribution. `RocmDir` in `d4r/d4r.ini` still selects another installation.

## Tested games

Radeon RX 7700 XT, GE-Proton 11-3. All three DLSS models (E, K, M) run in each game:

SILENT HILL Townfall, Ghost of Tsushima DIRECTOR'S CUT, Ready or Not, Clair Obscur: Expedition 33, Marvel's Spider-Man: Miles Morales, The Last of Us Part I, Control Ultimate Edition, Cyberpunk 2077, Alan Wake 2, SILENT HILL 2 (2024), Subnautica 2, Horizon Zero Dawn Remastered, Ratchet & Clank: Rift Apart, Dying Light: The Beast.

Several need a small setup step: renaming a file, a launch option, or an `OptiScaler.ini` setting. [SUPPORTED_GAMES.md](https://github.com/countervolts/d4r/blob/main/SUPPORTED_GAMES.md) lists these, along with known image-quality issues per model.

## Install

1. Extract `d4r-0.1.1.zip` into the folder that holds the game's main `.exe`. For Unreal Engine games that is `<Project>/Binaries/Win64`. Keep the zip's layout: `dxgi.dll`, `OptiScaler.ini`, `d3d12.dll` and `d3d12core.dll` go next to the `.exe`, beside the `d4r` folder.
2. In Steam, force GE-Proton 11 for the game, and set these launch options:
   ```
   PROTON_FORCE_NVAPI=1 DXVK_NVAPI_GPU_ARCH=AD100 %command%
   ```
3. Select DLSS in the game. The first start compiles DLSS's GPU kernels, which takes about a minute and a half; later starts use the cache in `~/.cache/d4r`.

**Upgrading from 0.1.0:** extract over the old files, then restore your `d4r/d4r.ini` if you changed it. `D4R_README.txt` in the zip has the details, and `sh d4r/d4r-check.sh` checks an install.

## Requirements

- An AMD Radeon RX 7000 series GPU. The fast kernels are built for gfx1101 (RX 7700 XT, RX 7800 XT, RX 7700). Other RDNA3 cards run DLSS without them:
  - DLSS 3 (E) at full speed;
  - DLSS 4.5 (M) at about half speed;
  - DLSS 4 (K) without its transformer, with a worse image.
- The amdgpu kernel driver (`/dev/kfd`).
- GE-Proton 11 (tested: GE-Proton11-3).

## Performance

Radeon RX 7700 XT, SILENT HILL Townfall at 2560×1440, average fps over the same 62-second walk:

| Mode | DLSS 3 CNN (E) | DLSS 4 (K) | DLSS 4.5 (M) | FSR 4 |
|---|---|---|---|---|
| Quality | 72 | 69 | 51 | 76 |
| Balanced | 80 | 76 | 59 | 84 |
| Performance | 88 | 82 | 69 | 94 |
| Ultra Performance | 89 | 93 | 90 | 107 |

Native 2560×1440 without upscaling (the game's TSR at 100%) runs at 49 fps. This release does not change the per-frame cost of DLSS.

## Known limitations

- Only DLSS Super Resolution in DirectX 12 games. There is no Frame Generation or Ray Reconstruction: keep Ray Reconstruction off in games that offer it.
- Use the game's DLSS setting where it exists. Feeding DLSS from FSR or XeSS inputs through OptiScaler can ghost, or show a black screen in some games.
- Saving from the OptiScaler overlay ("Save INI") resets d4r's `OptiScaler.ini` values to `auto`, including frame generation. Restore them afterwards.
- Do not use it in games with anti-cheat.
- DLSS 4.5 (M) is much slower than FSR 4 on RDNA3.

## Licenses

d4r is Apache 2.0. The zip also contains:
- ZLUDA (Apache 2.0 or MIT);
- vkd3d-proton (LGPL 2.1, patched; the patch is in `d4r/source`);
- the ROCm 7.2.4 runtime, unmodified: HIP and rocprofiler-register (MIT), ROCr (NCSA), comgr (Apache 2.0), and Ubuntu's libelf and libnuma (LGPL). The license texts are in `d4r/licenses`;
- OptiScaler 0.9.4 (GPL 3.0, unmodified; source at https://github.com/optiscaler/OptiScaler/tree/v0.9.4).

`d4r/source/SOURCES.txt` lists where every file comes from.

NVIDIA's files in the zip, and the kernels built from NVIDIA's code, belong to NVIDIA and are not covered by any of these licenses. d4r is not affiliated with NVIDIA, AMD or the OptiScaler project.

**SHA-256** `d4r-0.1.1.zip`: `@SHA256@`
