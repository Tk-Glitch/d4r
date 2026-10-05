// WGP mode: with the padded LDS rows (PWIN_PAD) this layer is 2-5% faster in WGP mode than in CU mode
// (CU mode was 1-6% faster before the padding)
// accumulator init vectors from the weights (the prepared f32 table is 2-4% slower here)
// Q, K, V chains two at a time (fewer live accumulators, spills 164 -> 68 B; -2 %)
// d4r-build-flags: -DPWIN_VEC_FROM_WEIGHTS -DPWIN_QKV_SPLIT
#define PWIN_F32ACC
#include "pwin_layer.h"
PWIN_DECODER(dltss_pwin_dec1_layer, 2, 64, 64)
