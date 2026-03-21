/*
 * cc_loss.metal - Fused CC loss for Metal
 *
 * 1:1 translation of fused_cc.cu CUDA kernels.
 * Flow:
 *   1. fcc_create_intermediates: pack [I, J, I², J², IJ] into 5 blocks
 *   2. (box filter done externally on intermediates)
 *   3. fcc_fwd: compute per-voxel NCC with threadgroup reduction → partial sums
 *   4. fcc_bwd_modify: compute gradient multipliers into intermediates
 *   5. (box filter adjoint done externally)
 *   6. fcc_bwd_grads: final per-voxel gradient
 */

#include <metal_stdlib>
using namespace metal;

#define FCC_BLOCK 256

/* ------------------------------------------------------------------ */
/* Kernel 1: Create intermediates                                      */
/* Pack: [I, J, I², J², IJ] into 5 contiguous blocks of spatial size   */
/* ------------------------------------------------------------------ */

struct FCCParams {
    uint spatial;  // D*H*W
    uint _pad;
};

kernel void fcc_create_intermediates(
    const device float *input    [[buffer(0)]],
    const device float *target   [[buffer(1)]],
    device float       *interm   [[buffer(2)]],
    constant FCCParams &p        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.spatial) return;

    float I = input[gid];
    float J = target[gid];

    interm[gid]                  = I;
    interm[gid + p.spatial]      = J;
    interm[gid + 2 * p.spatial]  = I * I;
    interm[gid + 3 * p.spatial]  = J * J;
    interm[gid + 4 * p.spatial]  = I * J;
}

/* ------------------------------------------------------------------ */
/* Kernel 2: Forward NCC from filtered intermediates                   */
/* Per-voxel NCC with threadgroup reduction to partial sums            */
/* ------------------------------------------------------------------ */

struct FCCFwdParams {
    uint spatial;
    int  kernel_volume;  // ks*ks*ks
    float nr;
    float dr;
};

kernel void fcc_fwd(
    const device float *interm       [[buffer(0)]],
    device float       *partial_sum  [[buffer(1)]],
    constant FCCFwdParams &p         [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint wid [[threadgroup_position_in_grid]])
{
    threadgroup float sdata[FCC_BLOCK];

    float val = 0.0f;
    if (gid < p.spatial) {
        float mu    = interm[gid];
        float rho   = interm[gid + p.spatial];
        float mu2   = interm[gid + 2 * p.spatial];
        float rho2  = interm[gid + 3 * p.spatial];
        float murho = interm[gid + 4 * p.spatial];

        float kv = float(p.kernel_volume);
        float A = kv * (murho - mu * rho);
        float B = max(kv * (mu2 - mu * mu), p.dr);
        float C = max(kv * (rho2 - rho * rho), p.dr);

        float ncc = (A * A + p.nr) / (B * C + p.dr);
        ncc = clamp(ncc, -1.0f, 1.0f);
        val = ncc;
    }

    sdata[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = FCC_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sum[wid] = sdata[0];
    }
}

/* ------------------------------------------------------------------ */
/* Kernel 3: Backward — modify intermediates with gradient multipliers */
/* Overwrites interm[0..2] (and optionally [3..4] for target grad)     */
/* ------------------------------------------------------------------ */

struct FCCBwdParams {
    uint  spatial;
    int   kernel_volume;
    float nr;
    float dr;
    float grad_output_val;
    int   compute_grad_target;
};

kernel void fcc_bwd_modify(
    device float       *interm  [[buffer(0)]],
    const device float *input   [[buffer(1)]],
    const device float *target  [[buffer(2)]],
    constant FCCBwdParams &p    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.spatial) return;

    float mu    = interm[gid];
    float rho   = interm[gid + p.spatial];
    float mu2   = interm[gid + 2 * p.spatial];
    float rho2  = interm[gid + 3 * p.spatial];
    float murho = interm[gid + 4 * p.spatial];

    float kv = float(p.kernel_volume);
    float A = kv * (murho - mu * rho);
    float B = kv * (mu2 - mu * mu);
    float C = kv * (rho2 - rho * rho);

    float gO = p.grad_output_val;
    float D = 2.0f * gO * A / (B * C + p.dr);

    B += p.dr;
    C += p.dr;

    /* Write gradient multipliers back into intermediates */
    interm[gid]                  = D;                            /* D */
    interm[gid + p.spatial]      = D * A / B;                    /* D*A/B */
    interm[gid + 2 * p.spatial]  = D * (A / B * mu - rho);      /* D*(A/B*mu - rho) */

    if (p.compute_grad_target) {
        interm[gid + 3 * p.spatial] = D * A / C;                /* D*A/C */
        interm[gid + 4 * p.spatial] = D * (A / C * rho - mu);   /* D*(A/C*rho - mu) */
    }
}

/* ------------------------------------------------------------------ */
/* Kernel 4: Backward — compute final gradients from filtered interm   */
/* ------------------------------------------------------------------ */

struct FCCGradParams {
    uint spatial;
    uint has_grad_target;
};

kernel void fcc_bwd_grads(
    const device float *interm       [[buffer(0)]],
    const device float *input        [[buffer(1)]],
    const device float *target       [[buffer(2)]],
    device float       *grad_input   [[buffer(3)]],
    device float       *grad_target  [[buffer(4)]],
    constant FCCGradParams &p        [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.spatial) return;

    float I = input[gid];
    float J = target[gid];

    float gini_a  = interm[gid];
    float gini_b  = interm[gid + p.spatial];
    float gini_mu = interm[gid + 2 * p.spatial];

    grad_input[gid] = gini_a * J - gini_b * I + gini_mu;

    if (p.has_grad_target) {
        float gini_c   = interm[gid + 3 * p.spatial];
        float gini_mu2 = interm[gid + 4 * p.spatial];
        grad_target[gid] = gini_a * I - gini_c * J + gini_mu2;
    }
}

/* ================================================================== */
/* cc_loss.cu-style kernels (v2)                                       */
/* Matching CUDA's cc_loss.cu: separate buffers, 8 box-filter passes   */
/* (5 forward + 3 adjoint) instead of fused_cc's packed intermediates. */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* cc_multiply: c[i] = a[i] * b[i]                                     */
/* ------------------------------------------------------------------ */

struct CCMulParams {
    uint n;
    uint _pad;
};

kernel void cc_multiply(
    const device float *a      [[buffer(0)]],
    const device float *b      [[buffer(1)]],
    device float       *c      [[buffer(2)]],
    constant CCMulParams &p    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    c[gid] = a[gid] * b[gid];
}

/* ------------------------------------------------------------------ */
/* cc_ncc_and_grad_sources: compute per-voxel NCC + gradient sources   */
/* Matches CUDA cc_ncc_and_grad_sources_kernel exactly.                */
/* ------------------------------------------------------------------ */

struct CCNccParams {
    uint  n;
    float nr;
    float dr;
    int   compute_grad;
};

kernel void cc_ncc_and_grad_sources(
    const device float *p_sum   [[buffer(0)]],
    const device float *t_sum   [[buffer(1)]],
    const device float *p2_sum  [[buffer(2)]],
    const device float *t2_sum  [[buffer(3)]],
    const device float *tp_sum  [[buffer(4)]],
    device float       *ncc_out [[buffer(5)]],
    device float       *src_p   [[buffer(6)]],
    device float       *src_p2  [[buffer(7)]],
    device float       *src_tp  [[buffer(8)]],
    constant CCNccParams &params [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.n) return;

    float ps = p_sum[gid], ts = t_sum[gid];
    float cross = tp_sum[gid] - ps * ts;
    float p_var = p2_sum[gid] - ps * ps;
    float t_var = t2_sum[gid] - ts * ts;
    if (p_var < params.dr) p_var = params.dr;
    if (t_var < params.dr) t_var = params.dr;

    float f = cross * cross + params.nr;
    float g = p_var * t_var + params.dr;
    float ncc = f / g;
    if (ncc > 1.0f) ncc = 1.0f;
    if (ncc < -1.0f) ncc = -1.0f;
    ncc_out[gid] = ncc;

    if (params.compute_grad) {
        float g2 = g * g;
        src_tp[gid] = 2.0f * cross * g / g2;
        src_p2[gid] = -f * t_var / g2;
        src_p[gid] = (-2.0f * cross * ts * g + 2.0f * f * ps * t_var) / g2;
    }
}

/* ------------------------------------------------------------------ */
/* cc_ncc_partial_sum: threadgroup reduction for NCC loss value         */
/* ------------------------------------------------------------------ */

#define CC_BLOCK 256

struct CCReduceParams {
    uint n;
    uint _pad;
};

kernel void cc_ncc_partial_sum(
    const device float *ncc_buf     [[buffer(0)]],
    device float       *partial_sum [[buffer(1)]],
    constant CCReduceParams &p      [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint wid [[threadgroup_position_in_grid]])
{
    threadgroup float sdata[CC_BLOCK];

    sdata[tid] = (gid < p.n) ? ncc_buf[gid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = CC_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sum[wid] = sdata[0];
    }
}

/* ------------------------------------------------------------------ */
/* cc_combine_grad: grad = -inv_count * (adj_p + 2*P*adj_p2 + T*adj_tp) */
/* Matches CUDA cc_combine_grad_kernel exactly.                         */
/* ------------------------------------------------------------------ */

struct CCCombineParams {
    uint  n;
    float inv_count;
};

kernel void cc_combine_grad(
    const device float *adj_p   [[buffer(0)]],
    const device float *adj_p2  [[buffer(1)]],
    const device float *adj_tp  [[buffer(2)]],
    const device float *P       [[buffer(3)]],
    const device float *T       [[buffer(4)]],
    device float       *grad    [[buffer(5)]],
    constant CCCombineParams &p [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    grad[gid] = -p.inv_count * (adj_p[gid] + 2.0f * P[gid] * adj_p2[gid] + T[gid] * adj_tp[gid]);
}
