/*
 * warp_inverse.cu - Warp inversion matching Python compositive_warp_inverse
 *
 * Implements a full iterative optimization with InverseConsistencyOperator:
 *   loss = MSE(compose(u, ref), -ref) + MSE(compose(ref, u), -u)
 *
 * Uses the same WarpAdam compositive update as the main registration.
 * Matches Python's scales=[8,4,2,1], iterations=[200,200,100,50].
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define WI_BLOCK 512

extern "C" {

#include "cfireants/tensor.h"
#include "kernels.h"

/* ------------------------------------------------------------------ */
/* Kernel helpers                                                      */
/* ------------------------------------------------------------------ */

__global__ void wi_sub_k(float *a, const float *b, const float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = b[i] - c[i];
}

/*
 * Fixed-point warp inverse: inv = -interp(u, id + inv)
 *
 * Matches WebGPU/CPU implementation for cross-backend parity.
 * compose(ref, inv) = inv + interp(ref, id+inv)
 * inv_new = inv_old - compose(ref, inv_old) = -interp(ref, id+inv_old)
 */
void cuda_warp_inverse_fixedpoint(
    const float *ref_disp,   /* [D,H,W,3] warp to invert */
    float *inv_out,          /* [D,H,W,3] result */
    int D, int H, int W,
    int n_iters)
{
    int n3 = D * H * W * 3;
    int bn3 = (n3 + WI_BLOCK - 1) / WI_BLOCK;

    cudaMemset(inv_out, 0, n3 * sizeof(float));

    float *d_tmp;
    cudaMalloc(&d_tmp, n3 * sizeof(float));

    extern int cfireants_verbose;
    if (cfireants_verbose >= 2)
        fprintf(stderr, "  Warp inverse (fixed-point): %d iters at [%d,%d,%d]\n",
                n_iters, D, H, W);

    for (int it = 0; it < n_iters; it++) {
        /* d_tmp = inv_old + interp(ref, id + inv_old) */
        cudaMemcpy(d_tmp, inv_out, n3 * sizeof(float), cudaMemcpyDeviceToDevice);
        cuda_fused_compositive_update(ref_disp, d_tmp, d_tmp, D, H, W);

        /* inv_new = inv_old - d_tmp = -interp(ref, id + inv_old) */
        wi_sub_k<<<bn3, WI_BLOCK>>>(inv_out, inv_out, d_tmp, n3);
    }

    cudaFree(d_tmp);
}

} /* extern "C" */
