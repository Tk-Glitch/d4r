// the WMMA half exchange through ds_bpermute instead of v_permlanex16: -1.5% here
// d4r-build-flags: -DWM_HALF_BPERMUTE
#include "pwin_wide.h"
WIDE_DECODER(dltss_pwin_dec3_layer, 4, 96, 128, 2)
