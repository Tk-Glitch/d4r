# Native kernels

The native kernels are RDNA3 (gfx11) code objects that ZLUDA runs in place of specific DLSS kernels. Each one has the same name, launch shape and parameter block as the PTX kernel it replaces, and reads the same buffers, including the network weights inside NVIDIA's DLL. No weights or NVIDIA code are stored here.

## The override hook (patches/zluda/0004)

With `D4R_ZLUDA_NATIVE_DIR=DIR`, whenever NGX loads a PTX module and asks for kernel `NAME`, ZLUDA looks for `DIR/NAME.hsaco` and uses its kernel `NAME` instead. The code object can export:

| Symbol | Meaning |
|---|---|
| `NAME_prep` kernel | launched first on the same stream, with the same parameters (typically re-lays out weights) |
| `u32 d4r_prep_blocks` | grid size of the prep kernel (128 threads per block) |
| `u32 d4r_prep_key_offset` | byte offset of a u64 in the parameter block; the prep is skipped while that value (the weights pointer) repeats |
| `u32 d4r_block_z` | replaces the launch's block z dimension (more waves per window) |
| `u32 d4r_grid_x` | replaces the grid (persistent kernels) |

Kernels without a file in `DIR` are compiled from PTX as usual, so a partial set works.

**Release directories.** A directory that holds `d4r-kernels.txt` (or `<gfx target>/d4r-kernels.txt`) is checked before use:
- `kernels/tools/kernel_manifest.py DIR nvngx_dlss.dll...` writes the manifest. For each `NAME.hsaco` it records the FNV-1a 64 hash of the PTX module that defines `.entry NAME`, taken from each DLL given.
- The CUDA bridge serves a kernel only after NGX has loaded a module with one of the listed hashes.
- It picks the target folder from the GPU's KFD topology entry.

The release is built this way, with hashes from DLSS 310.7.0 and 310.9.1. The PTX of every replaced kernel is identical in those two versions.

## Families

**DLSS 4, preset K (`kernels/k`, `dltss_pwin_*`).**
- **Network:** a Swin-style U-Net of eleven layers over 8×8 token windows. There are encoders enc0–enc4 with 2×2 patch merging, a bottleneck dec5, and decoders dec4–dec0 with patch expansion and skip inputs. The window attention is either learned (query × key plus a bias table) or position-only (a fixed table times the values).
- **Templates:**
  - `pwin_layer.h`: four waves per window, each owning a 16-token tile.
  - `pwin_pos.h`: position-only layers with two token tiles per wave, which halves weight traffic.
  - `pwin_wide.h`: the deep layers, which have only 6–77 windows per frame. It uses 4·NG waves per window, split by token tile and channel group, so the GPU stays busy.
- **Weights:** prep kernels expand the weights into WMMA operand images once.

**DLSS 4.5, preset M (`kernels/m`, `rrlite_*`).**
- **Network:** the Swin blocks of DLSS 4.5, whose weights are FP8. Their prep kernels expand the weights to f16 WMMA operands.
- **Template:** `swin_block.h` covers encoders, the tube-shaped enc3 and decoders.

**Texture kernels (`kernels/tex`).**
- **The problem:** some DLSS kernels are mostly texture sampling and scalar maths that ZLUDA already compiles well, apart from a few slow parts.
- **The approach:** for those, `make_ptx.py` edits NVIDIA's PTX (extracted from your DLL at build time):
  - surface stores become calls into `tex_common.h`;
  - the `roundf` idiom that forces ZLUDA into strict-FP mode is rewritten exactly;
  - for M's enc0 and dec0, a range of the kernel is replaced by a native tail or head.
- **Build:** ZLUDA compiles the edited PTX with the HIP bitcode linked in (`D4R_ZLUDA_EXTRA_BC`), and the resulting code object is saved.
- **Output redirect:** the K output kernel (`hiluma_engine_output_*`) can also write its result straight into the shim's output buffer instead of a CUDA array. See `D4R_SHIM_OUTPUT_DIRECT` in [architecture.md](architecture.md).

## Numerics

NVIDIA's tensor-core code rounds its f16 accumulator after every 16-deep step. The native kernels either do the same (bit-exact against the reference models) or keep f32 accumulators through a chain (`PWIN_F32ACC`, `SWIN_F32ACC`). The f32 version is more accurate than NVIDIA's own rounding, and its replayed output differs from the exact one by about 66 dB PSNR, which is invisible.

The installed K set uses f32 accumulation for enc0–enc2 and dec0–dec2 and exact rounding for the deep layers. The M set uses the fast numerics throughout.

## Building

```sh
kernels/build.sh k          # DLSS 4 layers   (ROCm clang only)
kernels/build.sh m          # DLSS 4.5 layers
kernels/build.sh tex        # texture kernels (needs D4R_DLSS_DLL and D4R_ZLUDA_BUILD)
kernels/build.sh all DIR    # everything into DIR (default kernels/out/native)
```

`D4R_GPU_ARCH` selects the target (default gfx1101). All kernels need gfx11 WMMA.

## Validating a kernel

1. **Capture launches.** Set `D4R_CUDA_REPLAY_DUMP_DIR` (with `D4R_CUDA_REPLAY_DUMP_FILTER=<kernel name>`) while running the D3D12 harness or a game. The bridge then writes each matching launch as `manifest.txt`, `args.bin` and one `alloc-N.bin` per buffer.
2. **Run the native kernel on a capture.** Build `kernels/tools/dump_runner.cpp` with hipcc. Then run:

   ```sh
   PREP_GRID=<d4r_prep_blocks> dump_runner NAME.hsaco NAME <capture> <out> 30
   ```

   It runs the prep and main kernels, saves every buffer after the first launch, and times 30 launches.
3. **Compare against a reference.** `kernels/tools/pwin_model.py` (K) and `swin_model.py` (M) are numpy models of the layers, written stage by stage against `ptxsim.py`, a small vectorised PTX interpreter that runs NVIDIA's kernels on the CPU. A bit-exact variant should match the model within f16 rounding; after any change, compare the new build's buffers with the previous build's.
4. **Check the whole pipeline.** Replay captured frames through the harness with both kernel sets and compare the outputs with `kernels/tools/psnr.py`. `D4R_CUDA_KERNEL_PROFILE=1` makes the bridge time every kernel; `kernels/tools/kprof.py` summarises the log per frame.

## Per-kernel cost

GPU time per frame at Quality (1705×960 → 2560×1440, RX 7700 XT), from `D4R_CUDA_KERNEL_PROFILE`:

| DLSS 4 (K) | ms | DLSS 4.5 (M) | ms |
|---|---|---|---|
| hiluma output (texture kernel) | 0.82 | enc3 tube (6 launches) | 1.42 |
| dec0 | 0.30 | enc1 | 1.27 |
| enc0 | 0.28 | dec1 | 1.18 |
| dec1 | 0.24 | enc2 | 0.79 |
| hiluma input (ZLUDA) | 0.24 | post (texture kernel) | 0.77 |
| enc1 | 0.21 | dec2 | 0.69 |
| enc2, dec2 | 0.10 each | enc0 (texture kernel) | 0.60 |
| enc3, dec3, enc4, dec4, dec5 | 0.05–0.08 each | dec0 (texture kernel) | 0.46 |
| NGX exposure and misc | 0.1 | downsample, NGX misc | 0.36 |
| **total** | **2.8** | **total** | **7.5** |
