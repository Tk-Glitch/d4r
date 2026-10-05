// Exact single-buffer enc1: reload the original input for the residual, preserve WGP mode.
// d4r-build-flags: -DPOS_VSHARE1 -DPWIN_VEC_SCALAR -DPWIN_PK_ELEM
#define PWIN_F32ACC
#include "pwin_pos_recompute.h"
POS_RECOMPUTE_ENCODER(dltss_pwin_enc1_layer, 2, 64, 64, 0, 1)
