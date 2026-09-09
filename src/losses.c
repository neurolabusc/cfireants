/*
 * losses.c - CPU cross-correlation loss
 *
 * Matches fireants LocalNormalizedCrossCorrelationLoss:
 *   kernel_type='rectangular', unsigned=True, reduction='mean'
 *
 * Formula (rectangular kernel, kernel_vol=1.0):
 *   t_sum = conv(target, kernel)       # local average
 *   p_sum = conv(pred, kernel)
 *   cross = conv(pred*target, kernel) - p_sum * t_sum
 *   t_var = max(conv(target², kernel) - t_sum², smooth_dr)
 *   p_var = max(conv(pred², kernel)   - p_sum², smooth_dr)
 *   ncc = (cross² + smooth_nr) / (t_var * p_var + smooth_dr)
 *   loss = -mean(ncc)
 */

#include "cfireants/losses.h"
#include "cfireants/threading.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Separable box filter along one axis.
 * in: source buffer, out: destination buffer (can be same pointer as in)
 * axis: 0=D, 1=H, 2=W
 * The rectangular kernel = uniform average with kernel_size=ks */
typedef struct {
    const float *in;
    float *out;
    int D, H, W, ks, axis;
} box_filter_context_t;

static void box_filter_range(size_t begin, size_t end, void *opaque) {
    box_filter_context_t *ctx = (box_filter_context_t *)opaque;
    const float *in = ctx->in;
    float *out = ctx->out;
    int D = ctx->D, H = ctx->H, W = ctx->W, ks = ctx->ks;
    int r = ks / 2;
    float scale = 1.0f / ks;

    if (ctx->axis == 2) { /* one W row per item */
        for (size_t line = begin; line < end; line++) {
                size_t row = line * (size_t)W;
                for (int w = 0; w < W; w++) {
                    float sum = 0.0f;
                    for (int k = -r; k <= r; k++) {
                        int ww = w + k;
                        if (ww >= 0 && ww < W)
                            sum += in[row + ww];
                    }
                    out[row + w] = sum * scale;
                }
        }
    } else if (ctx->axis == 1) { /* one H column per item */
        for (size_t line = begin; line < end; line++) {
                int d = (int)(line / (size_t)W);
                int w = (int)(line % (size_t)W);
                for (int h = 0; h < H; h++) {
                    float sum = 0.0f;
                    for (int k = -r; k <= r; k++) {
                        int hh = h + k;
                        if (hh >= 0 && hh < H)
                            sum += in[(size_t)(d * H + hh) * W + w];
                    }
                    out[(size_t)(d * H + h) * W + w] = sum * scale;
                }
        }
    } else { /* one D column per item */
        for (size_t line = begin; line < end; line++) {
                int h = (int)(line / (size_t)W);
                int w = (int)(line % (size_t)W);
                for (int d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int k = -r; k <= r; k++) {
                        int dd = d + k;
                        if (dd >= 0 && dd < D)
                            sum += in[(size_t)(dd * H + h) * W + w];
                    }
                    out[(size_t)(d * H + h) * W + w] = sum * scale;
                }
        }
    }
}

static void box_filter_axis(const float *in, float *out,
                            int D, int H, int W, int ks, int axis) {
    box_filter_context_t context = {in, out, D, H, W, ks, axis};
    size_t lines = axis == 2 ? (size_t)D * H
                 : axis == 1 ? (size_t)D * W : (size_t)H * W;
    cfireants_parallel_for(lines, 64, box_filter_range, &context);
}

/* Full separable 3D box filter (rectangular kernel, normalized).
 * Uses two temporary buffers for in-place operation. */
static void separable_box_filter_3d(const float *in, float *out,
                                    int D, int H, int W, int ks,
                                    float *tmp) {
    /* axis 0 (D): in -> tmp */
    box_filter_axis(in, tmp, D, H, W, ks, 0);
    /* axis 1 (H): tmp -> out */
    box_filter_axis(tmp, out, D, H, W, ks, 1);
    /* axis 2 (W): out -> tmp, then copy back */
    box_filter_axis(out, tmp, D, H, W, ks, 2);
    memcpy(out, tmp, (size_t)D * H * W * sizeof(float));
}

int cpu_cc_loss_3d(const tensor_t *pred, const tensor_t *target,
                   int kernel_size, float *loss_out, tensor_t *grad_pred) {
    int B = pred->shape[0], C = pred->shape[1];
    int D = pred->shape[2], H = pred->shape[3], W = pred->shape[4];
    size_t spatial = (size_t)D * H * W;
    float smooth_nr = 1e-5f;
    float smooth_dr = 1e-5f;

    if (grad_pred) {
        int shape[5] = {B, C, D, H, W};
        if (tensor_alloc(grad_pred, 5, shape, DTYPE_FLOAT32, DEVICE_CPU) != 0)
            return -1;
    }

    const float *P = tensor_data_f32(pred);
    const float *T = tensor_data_f32(target);

    /* Allocate intermediate buffers */
    float *p_sum  = (float *)malloc(spatial * sizeof(float));
    float *t_sum  = (float *)malloc(spatial * sizeof(float));
    float *p2_sum = (float *)malloc(spatial * sizeof(float));
    float *t2_sum = (float *)malloc(spatial * sizeof(float));
    float *tp_sum = (float *)malloc(spatial * sizeof(float));
    float *work   = (float *)malloc(spatial * sizeof(float));
    float *tmp    = (float *)malloc(spatial * sizeof(float));

    double total_loss = 0.0;
    size_t total_count = (size_t)B * C * D * H * W;

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            const float *Pbc = P + (b * C + c) * spatial;
            const float *Tbc = T + (b * C + c) * spatial;

            /* Compute box-filtered sums */
            separable_box_filter_3d(Pbc, p_sum, D, H, W, kernel_size, tmp);
            separable_box_filter_3d(Tbc, t_sum, D, H, W, kernel_size, tmp);

            for (size_t i = 0; i < spatial; i++) work[i] = Pbc[i] * Pbc[i];
            separable_box_filter_3d(work, p2_sum, D, H, W, kernel_size, tmp);

            for (size_t i = 0; i < spatial; i++) work[i] = Tbc[i] * Tbc[i];
            separable_box_filter_3d(work, t2_sum, D, H, W, kernel_size, tmp);

            for (size_t i = 0; i < spatial; i++) work[i] = Tbc[i] * Pbc[i];
            separable_box_filter_3d(work, tp_sum, D, H, W, kernel_size, tmp);

            /* Compute NCC per voxel.
             * kernel_vol = 1.0 for normalized rectangular kernel */
            float *grad_out = grad_pred
                ? tensor_data_f32(grad_pred) + (b*C + c)*spatial : NULL;

            /*
             * Gradient computation for CC loss w.r.t. pred (P).
             *
             * ncc[v] = (cross[v]^2 + nr) / (pvar[v] * tvar[v] + dr)
             * loss = -mean(ncc)
             *
             * For each voxel v, ncc depends on P through the box-filtered
             * intermediates. Since box filter is linear and self-adjoint:
             *   d(loss)/d(P[i]) = sum_v d(loss)/d(ncc[v]) * d(ncc[v])/d(intermediates) * d(intermediates)/d(P[i])
             *
             * Strategy: compute per-voxel partial derivatives w.r.t. each
             * intermediate, then apply box-filter adjoint.
             *
             * d(ncc)/d(tp_sum) = 2*cross / denom
             * d(ncc)/d(p_sum)  = -2*cross*t_sum / denom - 2*p_sum*cross^2 / denom^2 * t_var
             *                  = 2*cross / denom * (-t_sum - p_sum * ncc_raw / denom * t_var)
             *                    ... this gets complex.
             *
             * Simpler: compute d(ncc)/d(P[i]) via the chain rule directly.
             * For each voxel i, P[i] contributes to the box-filtered sums at
             * all voxels in its neighborhood. The adjoint of box filter handles this.
             *
             * d(-ncc)/d(p_sum)  = -(2*cross*(-t_sum) + 2*p_sum*(cross^2+nr)*(-t_var)) / denom^2
             *                     ... let's use a cleaner formulation.
             *
             * Let f = cross^2 + nr, g = p_var * t_var + dr
             * ncc = f/g
             * d(ncc)/d(p_sum) = (d(f)/d(p_sum)*g - f*d(g)/d(p_sum)) / g^2
             *   d(f)/d(p_sum) = 2*cross * d(cross)/d(p_sum) = 2*cross*(-t_sum)
             *   d(g)/d(p_sum) = d(p_var)/d(p_sum) * t_var = (-2*p_sum) * t_var
             * => d(ncc)/d(p_sum) = (2*cross*(-t_sum)*g - f*(-2*p_sum*t_var)) / g^2
             *                    = (-2*cross*t_sum*g + 2*f*p_sum*t_var) / g^2
             *
             * Similarly for d(ncc)/d(p2_sum) and d(ncc)/d(tp_sum).
             *
             * Then: d(loss)/d(P[i]) = box_adjoint[ d(ncc)/d(p_sum) * d(p_sum)/d(P) + ... ]
             * where d(p_sum)/d(P[i]) = kernel[i] (box filter weight = 1/ks per elem).
             *
             * Since box filter and its adjoint are the same operation (symmetric kernel),
             * we can compute the per-voxel "source terms" and then apply box_filter.
             */

            /* Compute NCC per voxel and accumulate loss */
            /* Also compute per-voxel source terms for gradient if needed */
            float *src_p = NULL, *src_p2 = NULL, *src_tp = NULL;
            if (grad_out) {
                src_p  = (float *)calloc(spatial, sizeof(float));
                src_p2 = (float *)calloc(spatial, sizeof(float));
                src_tp = (float *)calloc(spatial, sizeof(float));
            }

            for (size_t i = 0; i < spatial; i++) {
                float ps = p_sum[i], ts = t_sum[i];
                float cross = tp_sum[i] - ps * ts;
                float p_var = p2_sum[i] - ps * ps;
                float t_var = t2_sum[i] - ts * ts;

                if (p_var < smooth_dr) p_var = smooth_dr;
                if (t_var < smooth_dr) t_var = smooth_dr;

                float f = cross * cross + smooth_nr;
                float g = p_var * t_var + smooth_dr;
                float ncc = f / g;
                if (ncc > 1.0f) ncc = 1.0f;
                if (ncc < -1.0f) ncc = -1.0f;

                total_loss += ncc;

                if (grad_out) {
                    float g2 = g * g;
                    float scale = -1.0f / total_count; /* will adjust below */

                    /* d(ncc)/d(tp_sum): tp_sum enters only through cross */
                    float dncc_dtp = 2.0f * cross * g / g2;

                    /* d(ncc)/d(p2_sum): enters through p_var */
                    float dncc_dp2 = -f * t_var / g2;

                    /* d(ncc)/d(p_sum): enters through cross and p_var */
                    float dncc_dps = (-2.0f * cross * ts * g + 2.0f * f * ps * t_var) / g2;

                    /* These are d(ncc)/d(boxfiltered_quantity).
                     * We need d(loss)/d(P[i]) which requires box-filter adjoint.
                     * d(loss)/d(P[i]) = sum_v box[v-i] * (
                     *    dncc_dps[v] * 1         +  // d(p_sum)/d(P) = box kernel
                     *    dncc_dp2[v] * 2*P[i]    +  // d(p2_sum)/d(P) = box * 2*P
                     *    dncc_dtp[v] * T[i]         // d(tp_sum)/d(P) = box * T
                     * )
                     * = box_adjoint(dncc_dps) + 2*P * box_adjoint(dncc_dp2) + T * box_adjoint(dncc_dtp)
                     * where box_adjoint = box filter (self-adjoint) */
                    src_p[i]  = dncc_dps;
                    src_p2[i] = dncc_dp2;
                    src_tp[i] = dncc_dtp;
                }
            }

            if (grad_out) {
                /* Apply box filter adjoint to each source term */
                float *adj_p  = (float *)malloc(spatial * sizeof(float));
                float *adj_p2 = (float *)malloc(spatial * sizeof(float));
                float *adj_tp = (float *)malloc(spatial * sizeof(float));

                separable_box_filter_3d(src_p,  adj_p,  D, H, W, kernel_size, tmp);
                separable_box_filter_3d(src_p2, adj_p2, D, H, W, kernel_size, tmp);
                separable_box_filter_3d(src_tp, adj_tp, D, H, W, kernel_size, tmp);

                /* Combine: d(loss)/d(P[i]) = -(1/N) * (adj_p[i] + 2*P[i]*adj_p2[i] + T[i]*adj_tp[i]) */
                float inv_count = 1.0f / total_count;
                for (size_t i = 0; i < spatial; i++) {
                    grad_out[i] = -inv_count * (adj_p[i] + 2.0f*Pbc[i]*adj_p2[i] + Tbc[i]*adj_tp[i]);
                }

                free(adj_p); free(adj_p2); free(adj_tp);
                free(src_p); free(src_p2); free(src_tp);
            }
        }
    }

    if (loss_out)
        *loss_out = -(float)(total_loss / total_count);

    free(p_sum); free(t_sum); free(p2_sum); free(t2_sum);
    free(tp_sum); free(work); free(tmp);
    return 0;
}

/* Fused CC loss matching CUDA fused_cc.cu / Metal fcc_* exactly */
int cpu_fused_cc_loss(const tensor_t *pred, const tensor_t *target,
                       int kernel_size, float *loss_out,
                       tensor_t *grad_pred, tensor_t *grad_target) {
    int D = pred->shape[2], H = pred->shape[3], W = pred->shape[4];
    size_t n = (size_t)D * H * W;
    int kv = kernel_size * kernel_size * kernel_size;
    float nr = 1e-5f, dr = 1e-5f;

    const float *I = tensor_data_f32(pred);
    const float *J = tensor_data_f32(target);
    int shape[5] = {1, 1, D, H, W};
    if (grad_pred) tensor_alloc(grad_pred, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);
    if (grad_target) tensor_alloc(grad_target, 5, shape, DTYPE_FLOAT32, DEVICE_CPU);

    /* Step 1: Create 5 intermediates */
    float *b_I   = (float *)malloc(n * sizeof(float));
    float *b_J   = (float *)malloc(n * sizeof(float));
    float *b_I2  = (float *)malloc(n * sizeof(float));
    float *b_J2  = (float *)malloc(n * sizeof(float));
    float *b_IJ  = (float *)malloc(n * sizeof(float));
    float *tmp   = (float *)malloc(n * sizeof(float));

    memcpy(b_I, I, n * sizeof(float));
    memcpy(b_J, J, n * sizeof(float));
    for (size_t i = 0; i < n; i++) { b_I2[i] = I[i]*I[i]; b_J2[i] = J[i]*J[i]; b_IJ[i] = I[i]*J[i]; }

    /* Step 2: Box filter all 5 */
    separable_box_filter_3d(b_I,  b_I,  D, H, W, kernel_size, tmp);
    separable_box_filter_3d(b_J,  b_J,  D, H, W, kernel_size, tmp);
    separable_box_filter_3d(b_I2, b_I2, D, H, W, kernel_size, tmp);
    separable_box_filter_3d(b_J2, b_J2, D, H, W, kernel_size, tmp);
    separable_box_filter_3d(b_IJ, b_IJ, D, H, W, kernel_size, tmp);

    /* Step 3: Forward NCC
     * Box filter produces local means. The kv scaling converts to sum-based
     * quantities, matching the original CUDA fused_cc.cu implementation.
     * This is intentional — lr and optimizer are tuned for this scale. */
    if (loss_out) {
        double ncc_sum = 0;
        for (size_t i = 0; i < n; i++) {
            float mu = b_I[i], rho = b_J[i];
            float A = (float)kv * (b_IJ[i] - mu*rho);
            float B = (float)kv * (b_I2[i] - mu*mu); if (B < dr) B = dr;
            float C = (float)kv * (b_J2[i] - rho*rho); if (C < dr) C = dr;
            float ncc = (A*A + nr) / (B*C + dr);
            if (ncc > 1.0f) ncc = 1.0f; if (ncc < -1.0f) ncc = -1.0f;
            ncc_sum += ncc;
        }
        *loss_out = -(float)(ncc_sum / n);
    }

    /* Steps 4-6: Backward */
    if (grad_pred || grad_target) {
        float gO = -1.0f / (float)n;
        int cgt = (grad_target != NULL) ? 1 : 0;

        /* Step 4: bwd_modify — overwrite intermediates with gradient multipliers */
        for (int i = 0; i < n; i++) {
            float mu = b_I[i], rho = b_J[i];
            float A = (float)kv * (b_IJ[i] - mu*rho);
            float B = (float)kv * (b_I2[i] - mu*mu);
            float C = (float)kv * (b_J2[i] - rho*rho);
            float Dv = 2.0f * gO * A / (B*C + dr);
            B += dr; C += dr;
            b_I[i]  = Dv;                           /* gini_a */
            b_J[i]  = Dv * A / B;                   /* gini_b */
            b_I2[i] = Dv * (A / B * mu - rho);      /* gini_mu */
            if (cgt) {
                b_J2[i] = Dv * A / C;               /* gini_c */
                b_IJ[i] = Dv * (A / C * rho - mu);  /* gini_mu2 */
            }
        }

        /* Step 5: Box filter adjoint */
        separable_box_filter_3d(b_I,  b_I,  D, H, W, kernel_size, tmp);
        separable_box_filter_3d(b_J,  b_J,  D, H, W, kernel_size, tmp);
        separable_box_filter_3d(b_I2, b_I2, D, H, W, kernel_size, tmp);
        if (cgt) {
            separable_box_filter_3d(b_J2, b_J2, D, H, W, kernel_size, tmp);
            separable_box_filter_3d(b_IJ, b_IJ, D, H, W, kernel_size, tmp);
        }

        /* Step 6: Final gradients (divide by ks² to match Python cc.py autograd) */
        float inv_ks2 = 1.0f / (float)(kernel_size * kernel_size);
        if (grad_pred) {
            float *gp = tensor_data_f32(grad_pred);
            for (size_t i = 0; i < n; i++)
                gp[i] = (b_I[i]*J[i] - b_J[i]*I[i] + b_I2[i]) * inv_ks2;
        }
        if (grad_target) {
            float *gt = tensor_data_f32(grad_target);
            for (size_t i = 0; i < n; i++)
                gt[i] = (b_I[i]*I[i] - b_J2[i]*J[i] + b_IJ[i]) * inv_ks2;
        }
    }

    free(b_I); free(b_J); free(b_I2); free(b_J2); free(b_IJ); free(tmp);
    return 0;
}


/* ------------------------------------------------------------------ */
/* Mutual Information loss (Gaussian Parzen windowing)                 */
/* ------------------------------------------------------------------ */

/* Parzen weights for one normalized intensity x: w[i] = G(x - c_i) / sum */
static inline void mi_parzen(float x, const float *centers, int nb, float preterm, float *w) {
    float sum = 0;
    for (int i = 0; i < nb; i++) {
        float d = x - centers[i];
        w[i] = expf(-preterm * d * d);
        sum += w[i];
    }
    for (int i = 0; i < nb; i++) w[i] /= sum;
}

static inline float mi_clamp01(float v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

typedef struct {
    const float *P, *T, *centers;
    double *partials;      /* [nthreads][nb*nb pab | nb pa | nb pb] */
    size_t N, chunk, stride;
    int nb;
    float inv_max, preterm;
} mi_hist_context_t;

/* One item = one thread-sized chunk, so slot == item index. */
static void mi_hist_range(size_t begin, size_t end, void *opaque) {
    mi_hist_context_t *c = (mi_hist_context_t *)opaque;
    int nb = c->nb;
    float wa[nb], wb[nb];
    for (size_t slot = begin; slot < end; slot++) {
        double *pab = c->partials + slot * c->stride;
        double *pa = pab + (size_t)nb * nb, *pb = pa + nb;
        size_t n1 = (slot + 1) * c->chunk;
        if (n1 > c->N) n1 = c->N;
        for (size_t n = slot * c->chunk; n < n1; n++) {
            mi_parzen(mi_clamp01(c->P[n] * c->inv_max), c->centers, nb, c->preterm, wa);
            mi_parzen(mi_clamp01(c->T[n] * c->inv_max), c->centers, nb, c->preterm, wb);
            for (int i = 0; i < nb; i++) {
                pa[i] += wa[i];
                pb[i] += wb[i];
                for (int j = 0; j < nb; j++) pab[i * nb + j] += wa[i] * wb[j];
            }
        }
    }
}

typedef struct {
    const float *P, *T, *centers, *dmi_dpab, *dmi_dpa;
    float *grad_out;
    int nb;
    float inv_max, preterm, inv_N, grad_scale;
} mi_grad_context_t;

static void mi_grad_range(size_t begin, size_t end, void *opaque) {
    mi_grad_context_t *c = (mi_grad_context_t *)opaque;
    int nb = c->nb;
    float wa[nb], wb[nb];
    for (size_t n = begin; n < end; n++) {
        float pn = mi_clamp01(c->P[n] * c->inv_max);
        mi_parzen(pn, c->centers, nb, c->preterm, wa);
        mi_parzen(mi_clamp01(c->T[n] * c->inv_max), c->centers, nb, c->preterm, wb);

        /* Softmax-style derivative: d(wa)/d(pn) = wa * (du_a - sum_a' wa' du_a'),
         * du_a/dpn = -2*preterm*(pn - c_a) */
        float weighted_deriv = 0;
        for (int a = 0; a < nb; a++)
            weighted_deriv += wa[a] * (-2.0f * c->preterm * (pn - c->centers[a]));

        float dmi_dpn = 0;
        for (int a = 0; a < nb; a++) {
            float du_a = -2.0f * c->preterm * (pn - c->centers[a]);
            float dwa_dpn = wa[a] * (du_a - weighted_deriv);
            for (int bb = 0; bb < nb; bb++)
                dmi_dpn += c->dmi_dpab[a * nb + bb] * c->inv_N * wb[bb] * dwa_dpn;
            dmi_dpn += c->dmi_dpa[a] * c->inv_N * dwa_dpn;
        }
        c->grad_out[n] = dmi_dpn * c->grad_scale;
    }
}

int cpu_mi_loss_3d(const tensor_t *pred, const tensor_t *target,
                   int num_bins, float *loss_out, tensor_t *grad_pred) {
    int B = pred->shape[0], C = pred->shape[1];
    int D = pred->shape[2], H = pred->shape[3], W = pred->shape[4];
    size_t spatial = (size_t)D * H * W;
    float smooth_nr = 1e-7f, smooth_dr = 1e-7f;

    if (grad_pred) {
        int shape[5] = {B, C, D, H, W};
        if (tensor_alloc(grad_pred, 5, shape, DTYPE_FLOAT32, DEVICE_CPU) != 0)
            return -1;
    }

    const float *P = tensor_data_f32(pred);
    const float *T = tensor_data_f32(target);

    /* Bin centers and sigma matching Python */
    float *bin_centers = (float *)malloc(num_bins * sizeof(float));
    for (int i = 0; i < num_bins; i++)
        bin_centers[i] = (float)i / num_bins + 0.5f / num_bins;
    float bin_spacing = 1.0f / num_bins;
    float sigma = bin_spacing * 0.5f; /* sigma_ratio=1.0 */
    float preterm = 1.0f / (2.0f * sigma * sigma);

    double total_mi = 0.0;
    int total_channels = 0;

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            const float *Pbc = P + (b * C + c) * spatial;
            const float *Tbc = T + (b * C + c) * spatial;

            /* Find max for normalization */
            float pmax = Pbc[0], tmax = Tbc[0];
            for (size_t i = 1; i < spatial; i++) {
                if (Pbc[i] > pmax) pmax = Pbc[i];
                if (Tbc[i] > tmax) tmax = Tbc[i];
            }
            float maxval = pmax > tmax ? pmax : tmax;
            if (maxval <= 0) maxval = 1.0f;
            float inv_max = (maxval > 1.0f) ? 1.0f / maxval : 1.0f;

            int nb = num_bins;
            size_t N = spatial;
            int nt = cfireants_num_threads();
            size_t stride = (size_t)nb * nb + 2 * nb;
            double *partials = (double *)calloc((size_t)nt * stride, sizeof(double));

            /* Pass 1: per-thread partial histograms (pab, pa, pb), merged in
             * slot order so the result is deterministic for a given thread count. */
            mi_hist_context_t hctx = {Pbc, Tbc, bin_centers, partials, N,
                                      (N + nt - 1) / nt, stride, nb, inv_max, preterm};
            cfireants_parallel_for((size_t)nt, 1, mi_hist_range, &hctx);

            float *pa = (float *)malloc(nb * sizeof(float));
            float *pb = (float *)malloc(nb * sizeof(float));
            float *pab = (float *)malloc(nb * nb * sizeof(float));
            float inv_N = 1.0f / N;
            for (size_t k = 0; k < stride; k++) {
                double acc = 0.0;
                for (int t = 0; t < nt; t++) acc += partials[t * stride + k];
                float v = (float)acc * inv_N;
                if (k < (size_t)nb * nb) pab[k] = v;
                else if (k < (size_t)nb * nb + nb) pa[k - nb * nb] = v;
                else pb[k - nb * nb - nb] = v;
            }
            free(partials);

            /* Compute MI = sum_{a,b} pab * log(pab / (pa*pb)) */
            double mi = 0.0;
            for (int i = 0; i < nb; i++) {
                for (int j = 0; j < nb; j++) {
                    float p = pab[i * nb + j];
                    float pp = pa[i] * pb[j];
                    mi += p * logf((p + smooth_nr) / (pp + smooth_dr) + smooth_dr);
                }
            }

            total_mi += mi;
            total_channels++;

            /* Pass 2: gradient (if requested) */
            if (grad_pred) {
                float *grad_out = tensor_data_f32(grad_pred) + (b*C + c)*spatial;

                /* d(MI)/d(pab[a][b]) = log(pab/(pa*pb)) + 1
                 * d(MI)/d(pa[a]) = -sum_b pab[a][b] / pa[a]  (approximately -1 for each a)
                 * d(pab[a][b])/d(pred[n]) = (1/N) * wb_n[b] * d(wa_n[a])/d(pred[n])
                 * d(wa_n[a])/d(pred[n]) = wa_n[a] * (-2*preterm*(pn-center[a])/maxval)
                 *   - wa_n[a] * sum_a' wa_n[a'] * (-2*preterm*(pn-center[a'])/maxval)
                 * = wa_n[a] * 2*preterm/maxval * (sum_a' wa_n[a']*(pn-center[a']) - (pn-center[a]))
                 */

                /* Precompute d(MI)/d(pab) and d(MI)/d(pa) */
                float *dmi_dpab = (float *)malloc(nb * nb * sizeof(float));
                float *dmi_dpa  = (float *)malloc(nb * sizeof(float));
                for (int i = 0; i < nb; i++) {
                    dmi_dpa[i] = 0;
                    for (int j = 0; j < nb; j++) {
                        float p = pab[i * nb + j];
                        float pp = pa[i] * pb[j];
                        float ratio = (p + smooth_nr) / (pp + smooth_dr) + smooth_dr;
                        dmi_dpab[i * nb + j] = logf(ratio) + p / (p + smooth_nr);
                        /* d(MI)/d(pa[i]) contribution: -pab[i][j] / (pa[i] * pb[j] + dr) * pb[j] */
                        dmi_dpa[i] -= p * pb[j] / (pp + smooth_dr);
                    }
                }

                mi_grad_context_t gctx = {Pbc, Tbc, bin_centers, dmi_dpab, dmi_dpa, grad_out,
                                          nb, inv_max, preterm, inv_N,
                                          -inv_max / total_channels};
                cfireants_parallel_for(N, 4096, mi_grad_range, &gctx);

                free(dmi_dpab);
                free(dmi_dpa);
            }

            free(pa); free(pb); free(pab);
        }
    }

    if (loss_out)
        *loss_out = -(float)(total_mi / total_channels);

    free(bin_centers);
    return 0;
}
