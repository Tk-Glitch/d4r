// fast numerics: intermediate activations f16 (no FP8 re-quantisation), f32 accumulation through GEMM chains
#define SWIN_NO_Q8
#define SWIN_F32ACC
#define SWIN_MAX_VGPR 192
#define SWIN_MLP_R(NW) 2
// NGX can recycle a weights address for another tube block after feature
// recreation. Address-only caching then serves stale packed weights. Prepare
// each launch; the default single image is sufficient on the launch stream.
#define SWIN_PREP_KEY_AT 0
#include "swin_block.h"
SWIN_MODULE_W(rrlite_enc3_tube_4x4, 128, 4, 8, 0, true, TubeParams, 0)
