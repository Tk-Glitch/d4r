// fast numerics: intermediate activations f16 (no FP8 re-quantisation), f32 accumulation through GEMM chains
// RX 7700 XT: memory-clause scheduling at the 192 VGPR cap; byte-identical to the default build, gfx1101 only
// d4r-build-flags-gfx1101: -mllvm -misched=gcn-max-memory-clause
#define SWIN_NO_Q8
#define SWIN_F32ACC
#ifndef SWIN_MAX_VGPR
#define SWIN_MAX_VGPR 192
#endif
#ifndef SWIN_MLP_R
#define SWIN_MLP_R(NW) 2
#endif
#include "swin_block.h"
#ifndef SWIN_NW
#define SWIN_NW 8
#endif
SWIN_MODULE_W(rrlite_dec2_4x4, 96, 4, SWIN_NW, 0, false, CommonParams, 128)
