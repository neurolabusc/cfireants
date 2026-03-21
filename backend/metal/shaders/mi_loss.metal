/*
 * mi_loss.metal - Mutual Information loss with Gaussian Parzen windowing
 *
 * Faithful translation of mi_loss.cu (CUDA gold standard).
 *
 * Three kernels:
 *   1. mi_histogram: threadgroup-local histogram using atomic_uint with
 *      CAS-based float add (threadgroup float atomics not supported in MSL).
 *      Global merge uses device atomic<float> (supported on Apple Silicon).
 *   2. mi_compute: reduction of MI from histogram bins.
 *   3. mi_gradient: per-voxel gradient via softmax derivative chain rule.
 */

#include <metal_stdlib>
using namespace metal;

#define MI_BLOCK 256
#define MAX_BINS 64

/* CAS-based float atomic add for threadgroup atomic_uint.
 * Stores floats as bitcast uint, uses compare_exchange for lock-free add. */
inline void threadgroup_atomic_add_float(threadgroup atomic_uint &target, float value) {
    uint old_val = atomic_load_explicit(&target, memory_order_relaxed);
    while (true) {
        float new_float = as_type<float>(old_val) + value;
        uint new_val = as_type<uint>(new_float);
        bool ok = atomic_compare_exchange_weak_explicit(
            &target, &old_val, new_val,
            memory_order_relaxed, memory_order_relaxed);
        if (ok) break;
    }
}

/* ------------------------------------------------------------------ */
/* Kernel 1: Histogram with threadgroup CAS + device float atomics     */
/* ------------------------------------------------------------------ */

struct MIHistParams {
    int N;
    int num_bins;
    float preterm;
    float inv_maxval_p;
    float inv_maxval_t;
    float _pad0;
    int _pad1, _pad2;
};

/* Threadgroup shared memory: [pa(nb) | pb(nb) | pab(nb*nb)] as atomic_uint.
 * For MAX_BINS=64: 64+64+4096 = 4224 uints = 16896 bytes (within 32KB limit). */

kernel void mi_histogram(
    const device float *pred           [[buffer(0)]],
    const device float *target         [[buffer(1)]],
    device atomic<float> *d_pab        [[buffer(2)]],
    device atomic<float> *d_pa         [[buffer(3)]],
    device atomic<float> *d_pb         [[buffer(4)]],
    const device float *bin_centers    [[buffer(5)]],
    constant MIHistParams &p           [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint wid [[threadgroup_position_in_grid]])
{
    const int nb = p.num_bins;
    const int hist_size = nb * nb + 2 * nb;

    /* Threadgroup histogram as atomic_uint (CAS-based float add) */
    threadgroup atomic_uint smem[MAX_BINS * MAX_BINS + 2 * MAX_BINS];

    /* Initialize to zero (bitcast of 0.0f = 0u) */
    for (int i = (int)tid; i < hist_size; i += MI_BLOCK) {
        atomic_store_explicit(&smem[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)gid < p.N) {
        float pn = pred[gid] * p.inv_maxval_p;
        float tn = target[gid] * p.inv_maxval_t;
        pn = clamp(pn, 0.0f, 1.0f);
        tn = clamp(tn, 0.0f, 1.0f);

        /* Compute softmax-normalized Parzen weights */
        float wa[MAX_BINS], wb[MAX_BINS];
        float sum_wa = 0.0f, sum_wb = 0.0f;

        for (int b = 0; b < nb; b++) {
            float dp = pn - bin_centers[b];
            float dt = tn - bin_centers[b];
            wa[b] = exp(-p.preterm * dp * dp);
            wb[b] = exp(-p.preterm * dt * dt);
            sum_wa += wa[b];
            sum_wb += wb[b];
        }

        float inv_wa = 1.0f / sum_wa;
        float inv_wb = 1.0f / sum_wb;
        for (int b = 0; b < nb; b++) { wa[b] *= inv_wa; wb[b] *= inv_wb; }

        /* Accumulate into threadgroup histogram (matching CUDA exactly).
         * Pre-normalize by 1/N so histograms are probabilities. */
        float inv_N = 1.0f / (float)p.N;

        for (int a = 0; a < nb; a++) {
            float wa_inv_N = wa[a] * inv_N;
            threadgroup_atomic_add_float(smem[a], wa_inv_N);
            threadgroup_atomic_add_float(smem[nb + a], wb[a] * inv_N);
            for (int b = 0; b < nb; b++) {
                float val = wa_inv_N * wb[b];
                if (val > 0.0f) {
                    threadgroup_atomic_add_float(smem[2 * nb + a * nb + b], val);
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Merge threadgroup histogram to global using device atomic<float>.
     * (Device float atomics ARE supported on Apple Silicon.) */
    for (int i = (int)tid; i < nb; i += MI_BLOCK) {
        float v_pa = as_type<float>(atomic_load_explicit(&smem[i], memory_order_relaxed));
        if (v_pa != 0.0f)
            atomic_fetch_add_explicit(&d_pa[i], v_pa, memory_order_relaxed);

        float v_pb = as_type<float>(atomic_load_explicit(&smem[nb + i], memory_order_relaxed));
        if (v_pb != 0.0f)
            atomic_fetch_add_explicit(&d_pb[i], v_pb, memory_order_relaxed);
    }
    for (int i = (int)tid; i < nb * nb; i += MI_BLOCK) {
        float v_pab = as_type<float>(atomic_load_explicit(&smem[2 * nb + i], memory_order_relaxed));
        if (v_pab != 0.0f)
            atomic_fetch_add_explicit(&d_pab[i], v_pab, memory_order_relaxed);
    }
}

/* ------------------------------------------------------------------ */
/* Kernel 2: Compute MI from histogram                                 */
/* ------------------------------------------------------------------ */

struct MIComputeParams {
    int num_bins;
    float nr;
    float dr;
    int _pad;
};

kernel void mi_compute(
    const device float *pab         [[buffer(0)]],
    const device float *pa          [[buffer(1)]],
    const device float *pb          [[buffer(2)]],
    device float *mi_partial        [[buffer(3)]],
    constant MIComputeParams &p     [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint wid [[threadgroup_position_in_grid]])
{
    threadgroup float sdata[MI_BLOCK];

    int total = p.num_bins * p.num_bins;
    float val = 0.0f;

    if ((int)gid < total) {
        int a = (int)gid / p.num_bins;
        int b = (int)gid % p.num_bins;
        float pij = pab[gid];
        float pp = pa[a] * pb[b];
        val = pij * log((pij + p.nr) / (pp + p.dr) + p.dr);
    }

    sdata[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = MI_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) mi_partial[wid] = sdata[0];
}

/* ------------------------------------------------------------------ */
/* Kernel 3: Gradient d(MI)/d(pred[n])                                 */
/* ------------------------------------------------------------------ */

struct MIGradParams {
    int N;
    int num_bins;
    float preterm;
    float inv_maxval_p;
    float inv_maxval_t;
    float nr;
    float dr;
    int _pad;
};

kernel void mi_gradient(
    const device float *pred        [[buffer(0)]],
    const device float *target      [[buffer(1)]],
    const device float *pab         [[buffer(2)]],
    const device float *pa          [[buffer(3)]],
    const device float *pb          [[buffer(4)]],
    device float *grad_pred         [[buffer(5)]],
    const device float *bin_centers [[buffer(6)]],
    constant MIGradParams &p        [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.N) return;

    float pn = pred[gid] * p.inv_maxval_p;
    float tn = target[gid] * p.inv_maxval_t;
    pn = clamp(pn, 0.0f, 1.0f);
    tn = clamp(tn, 0.0f, 1.0f);

    const int nb = p.num_bins;

    /* Recompute Parzen weights */
    float wa[MAX_BINS], wb[MAX_BINS];
    float sum_wa = 0.0f, sum_wb = 0.0f;
    for (int b = 0; b < nb; b++) {
        float dp = pn - bin_centers[b];
        float dt = tn - bin_centers[b];
        wa[b] = exp(-p.preterm * dp * dp);
        wb[b] = exp(-p.preterm * dt * dt);
        sum_wa += wa[b];
        sum_wb += wb[b];
    }
    float inv_swa = 1.0f / sum_wa;
    float inv_swb = 1.0f / sum_wb;
    for (int b = 0; b < nb; b++) { wa[b] *= inv_swa; wb[b] *= inv_swb; }

    float inv_N = 1.0f / (float)p.N;

    /* Softmax derivative: weighted_du = sum_a wa[a] * du_a */
    float weighted_du = 0.0f;
    for (int a = 0; a < nb; a++) {
        weighted_du += wa[a] * (-2.0f * p.preterm * (pn - bin_centers[a]));
    }

    float dmi_dpn = 0.0f;
    for (int a = 0; a < nb; a++) {
        float du_a = -2.0f * p.preterm * (pn - bin_centers[a]);
        float dwa_dpn = wa[a] * (du_a - weighted_du);

        /* Through joint histogram */
        for (int b = 0; b < nb; b++) {
            float pij = pab[a * nb + b];
            float pp = pa[a] * pb[b];
            float dmi_dpab = log((pij + p.nr) / (pp + p.dr) + p.dr) + pij / (pij + p.nr);
            dmi_dpn += dmi_dpab * inv_N * wb[b] * dwa_dpn;
        }

        /* Through marginal pa */
        float dmi_dpa = 0.0f;
        for (int b = 0; b < nb; b++) {
            float pp = pa[a] * pb[b];
            dmi_dpa -= pab[a * nb + b] * pb[b] / (pp + p.dr);
        }
        dmi_dpn += dmi_dpa * inv_N * dwa_dpn;
    }

    /* Chain rule + negate for loss = -MI */
    grad_pred[gid] = -dmi_dpn * p.inv_maxval_p;
}
