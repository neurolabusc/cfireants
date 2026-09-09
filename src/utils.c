/*
 * utils.c - Gaussian blur, Adam optimizer, trilinear resize (CPU)
 */

#include "cfireants/utils.h"
#include "cfireants/threading.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Gaussian blur                                                       */
/* ------------------------------------------------------------------ */

/* Build 1D Gaussian kernel matching fireants gaussian_1d with approx="erf" */
int make_gaussian_kernel(float sigma, float truncated,
                                float **kernel_out, int *len_out) {
    if (sigma <= 0.0f) {
        *kernel_out = (float *)malloc(sizeof(float));
        (*kernel_out)[0] = 1.0f;
        *len_out = 1;
        return 0;
    }

    int tail = (int)(truncated * sigma + 0.5f);
    int klen = 2 * tail + 1;
    float *k = (float *)malloc(klen * sizeof(float));

    /* "erf" approximation: k[i] = erf((i+0.5)/sigma/sqrt(2)) - erf((i-0.5)/sigma/sqrt(2)) */
    float inv_sigma_sqrt2 = 1.0f / (sigma * sqrtf(2.0f));
    float sum = 0.0f;
    for (int i = 0; i < klen; i++) {
        float x = (float)(i - tail);
        float v = 0.5f * (erff((x + 0.5f) * inv_sigma_sqrt2)
                        - erff((x - 0.5f) * inv_sigma_sqrt2));
        k[i] = v;
        sum += v;
    }
    /* Normalize */
    for (int i = 0; i < klen; i++)
        k[i] /= sum;

    *kernel_out = k;
    *len_out = klen;
    return 0;
}

/* 1D convolution along one axis with padding='same' (zero pad).
 * axis: 0=dim2(D), 1=dim3(H), 2=dim4(W) for a [B,C,D,H,W] tensor.
 * Works on a single (D,H,W) spatial volume. */
typedef struct {
    const float *in, *kernel;
    float *out;
    int D, H, W, klen, axis;
} conv1d_context_t;

static void conv1d_range(size_t begin, size_t end, void *opaque) {
    conv1d_context_t *ctx = (conv1d_context_t *)opaque;
    const float *in = ctx->in, *kernel = ctx->kernel;
    float *out = ctx->out;
    int D = ctx->D, H = ctx->H, W = ctx->W, klen = ctx->klen;
    int r = klen / 2;

    if (ctx->axis == 2) {
        for (size_t line = begin; line < end; line++) {
                size_t row = line * (size_t)W;
                for (int w = 0; w < W; w++) {
                    float sum = 0.0f;
                    for (int k = 0; k < klen; k++) {
                        int ww = w + k - r;
                        if (ww >= 0 && ww < W)
                            sum += in[row + ww] * kernel[k];
                    }
                    out[row + w] = sum;
                }
        }
    } else if (ctx->axis == 1) {
        for (size_t line = begin; line < end; line++) {
                int d = (int)(line / (size_t)W);
                int w = (int)(line % (size_t)W);
                for (int h = 0; h < H; h++) {
                    float sum = 0.0f;
                    for (int k = 0; k < klen; k++) {
                        int hh = h + k - r;
                        if (hh >= 0 && hh < H)
                            sum += in[(size_t)(d * H + hh) * W + w] * kernel[k];
                    }
                    out[(size_t)(d * H + h) * W + w] = sum;
                }
        }
    } else {
        for (size_t line = begin; line < end; line++) {
                int h = (int)(line / (size_t)W);
                int w = (int)(line % (size_t)W);
                for (int d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int k = 0; k < klen; k++) {
                        int dd = d + k - r;
                        if (dd >= 0 && dd < D)
                            sum += in[(size_t)(dd * H + h) * W + w] * kernel[k];
                    }
                    out[(size_t)(d * H + h) * W + w] = sum;
                }
        }
    }
}

static void conv1d_axis(const float *in, float *out,
                        int D, int H, int W,
                        const float *kernel, int klen, int axis) {
    conv1d_context_t context = {in, kernel, out, D, H, W, klen, axis};
    size_t lines = axis == 2 ? (size_t)D * H
                 : axis == 1 ? (size_t)D * W : (size_t)H * W;
    cfireants_parallel_for(lines, 64, conv1d_range, &context);
}

int cpu_gaussian_blur_3d(const tensor_t *input, tensor_t *output,
                         float sigma, float truncated) {
    int B = input->shape[0], C = input->shape[1];
    int D = input->shape[2], H = input->shape[3], W = input->shape[4];
    size_t spatial = (size_t)D * H * W;

    /* Allocate output */
    int shape[5] = {B, C, D, H, W};
    if (tensor_alloc(output, 5, shape, DTYPE_FLOAT32, DEVICE_CPU) != 0)
        return -1;

    /* Build Gaussian kernel */
    float *kernel = NULL;
    int klen = 0;
    make_gaussian_kernel(sigma, truncated, &kernel, &klen);

    const float *inp = tensor_data_f32(input);
    float *out = tensor_data_f32(output);
    float *tmp = (float *)malloc(spatial * sizeof(float));

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            const float *src = inp + (b * C + c) * spatial;
            float *dst = out + (b * C + c) * spatial;

            /* Separable: D -> tmp, H -> dst, W -> tmp, copy back */
            conv1d_axis(src, tmp, D, H, W, kernel, klen, 0);
            conv1d_axis(tmp, dst, D, H, W, kernel, klen, 1);
            conv1d_axis(dst, tmp, D, H, W, kernel, klen, 2);
            memcpy(dst, tmp, spatial * sizeof(float));
        }
    }

    free(tmp);
    free(kernel);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Per-axis Gaussian blur (in-place on single-channel [D,H,W])         */
/* ------------------------------------------------------------------ */

void cpu_blur_volume(float *data, int D, int H, int W,
                      float sigma_d, float sigma_h, float sigma_w,
                      float truncated) {
    size_t sz = (size_t)D * H * W;
    float *tmp = (float *)malloc(sz * sizeof(float));
    float sigmas[3] = { sigma_d, sigma_h, sigma_w };

    for (int axis = 0; axis < 3; axis++) {
        if (sigmas[axis] <= 0) continue;
        float *kernel = NULL;
        int klen = 0;
        make_gaussian_kernel(sigmas[axis], truncated, &kernel, &klen);
        conv1d_axis(data, tmp, D, H, W, kernel, klen, axis);
        memcpy(data, tmp, sz * sizeof(float));
        free(kernel);
    }
    free(tmp);
}

/* ------------------------------------------------------------------ */
/* Gaussian blur + trilinear downsample (matching GPU blur_downsample)  */
/* ------------------------------------------------------------------ */

void cpu_blur_downsample(const float *input, float *output,
                          int iD, int iH, int iW,
                          int oD, int oH, int oW) {
    size_t in_sz = (size_t)iD * iH * iW;

    /* Copy input so we don't modify it */
    float *blurred = (float *)malloc(in_sz * sizeof(float));
    memcpy(blurred, input, in_sz * sizeof(float));

    /* Gaussian blur with per-axis sigma = 0.5 * in_dim / out_dim */
    cpu_blur_volume(blurred, iD, iH, iW,
                     0.5f * (float)iD / (float)oD,
                     0.5f * (float)iH / (float)oH,
                     0.5f * (float)iW / (float)oW,
                     2.0f);

    /* Trilinear resize: blurred [iD,iH,iW] -> output [oD,oH,oW] */
    tensor_t in_t, out_t;
    int is[5] = {1, 1, iD, iH, iW};
    int os[5] = {1, 1, oD, oH, oW};
    memset(&in_t, 0, sizeof(in_t));
    in_t.ndim = 5; memcpy(in_t.shape, is, sizeof(is));
    in_t.dtype = DTYPE_FLOAT32; in_t.device = DEVICE_CPU;
    in_t.data = blurred; in_t.owns_data = 0;
    tensor_alloc(&out_t, 5, os, DTYPE_FLOAT32, DEVICE_CPU);
    cpu_trilinear_resize(&in_t, &out_t, 1);
    memcpy(output, out_t.data, (size_t)oD * oH * oW * sizeof(float));
    tensor_free(&out_t);
    free(blurred);
}

/* ------------------------------------------------------------------ */
/* Adam optimizer step                                                 */
/* ------------------------------------------------------------------ */

int cpu_adam_step(tensor_t *param, const tensor_t *grad,
                 tensor_t *exp_avg, tensor_t *exp_avg_sq,
                 float lr, float beta1, float beta2, float eps, int step) {
    if (param->dtype != DTYPE_FLOAT32) return -1;
    size_t n = param->numel;
    float *p = tensor_data_f32(param);
    const float *g = tensor_data_f32(grad);
    float *m = tensor_data_f32(exp_avg);
    float *v = tensor_data_f32(exp_avg_sq);

    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    float step_size = lr / bc1;

    for (size_t i = 0; i < n; i++) {
        /* Update biased first moment */
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        /* Update biased second raw moment */
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
        /* Compute bias-corrected update */
        float denom = sqrtf(v[i] / bc2) + eps;
        p[i] -= step_size * m[i] / denom;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Trilinear resize                                                    */
/* ------------------------------------------------------------------ */

int cpu_trilinear_resize(const tensor_t *input, tensor_t *output,
                         int align_corners) {
    int B = input->shape[0], C = input->shape[1];
    int iD = input->shape[2], iH = input->shape[3], iW = input->shape[4];
    int oD = output->shape[2], oH = output->shape[3], oW = output->shape[4];

    if (output->dtype != DTYPE_FLOAT32) return -1;

    const float *inp = tensor_data_f32(input);
    float *out = tensor_data_f32(output);

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            const float *src = inp + ((size_t)b * C + c) * iD * iH * iW;
            float *dst = out + ((size_t)b * C + c) * oD * oH * oW;

            for (int od = 0; od < oD; od++) {
                float sd;
                if (align_corners && oD > 1)
                    sd = (float)od * (iD - 1) / (oD - 1);
                else
                    sd = ((float)od + 0.5f) * iD / oD - 0.5f;

                int d0 = (int)floorf(sd);
                int d1 = d0 + 1;
                float fd = sd - d0;
                if (d0 < 0) { d0 = 0; fd = 0; }
                if (d1 >= iD) { d1 = iD - 1; fd = (d0 == d1) ? 0 : fd; }

                for (int oh = 0; oh < oH; oh++) {
                    float sh;
                    if (align_corners && oH > 1)
                        sh = (float)oh * (iH - 1) / (oH - 1);
                    else
                        sh = ((float)oh + 0.5f) * iH / oH - 0.5f;

                    int h0 = (int)floorf(sh);
                    int h1 = h0 + 1;
                    float fh = sh - h0;
                    if (h0 < 0) { h0 = 0; fh = 0; }
                    if (h1 >= iH) { h1 = iH - 1; fh = (h0 == h1) ? 0 : fh; }

                    for (int ow = 0; ow < oW; ow++) {
                        float sw;
                        if (align_corners && oW > 1)
                            sw = (float)ow * (iW - 1) / (oW - 1);
                        else
                            sw = ((float)ow + 0.5f) * iW / oW - 0.5f;

                        int w0 = (int)floorf(sw);
                        int w1 = w0 + 1;
                        float fw = sw - w0;
                        if (w0 < 0) { w0 = 0; fw = 0; }
                        if (w1 >= iW) { w1 = iW - 1; fw = (w0 == w1) ? 0 : fw; }

                        /* Trilinear interpolation */
                        #define SRC(d,h,w) src[(size_t)(d)*iH*iW + (h)*iW + (w)]
                        float val =
                            (1-fd)*(1-fh)*(1-fw)*SRC(d0,h0,w0) +
                            (1-fd)*(1-fh)*fw    *SRC(d0,h0,w1) +
                            (1-fd)*fh    *(1-fw)*SRC(d0,h1,w0) +
                            (1-fd)*fh    *fw    *SRC(d0,h1,w1) +
                            fd    *(1-fh)*(1-fw)*SRC(d1,h0,w0) +
                            fd    *(1-fh)*fw    *SRC(d1,h0,w1) +
                            fd    *fh    *(1-fw)*SRC(d1,h1,w0) +
                            fd    *fh    *fw    *SRC(d1,h1,w1);
                        #undef SRC

                        dst[(size_t)od*oH*oW + oh*oW + ow] = val;
                    }
                }
            }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Fixed-point warp inverse                                            */
/* ------------------------------------------------------------------ */

/* Trilinear sample of a [D,H,W,3] field at normalized coords (zeros padding). */
static inline void sample_disp3(const float *f, int D, int H, int W,
                                float sx, float sy, float sz, float v[3]) {
    float ix = (sx+1.0f)*0.5f*(W-1), iy = (sy+1.0f)*0.5f*(H-1), iz = (sz+1.0f)*0.5f*(D-1);
    int x0 = (int)floorf(ix), y0 = (int)floorf(iy), z0 = (int)floorf(iz);
    float fx = ix-x0, fy = iy-y0, fz = iz-z0;
    float wt[8] = {(1-fx)*(1-fy)*(1-fz),fx*(1-fy)*(1-fz),(1-fx)*fy*(1-fz),fx*fy*(1-fz),
                   (1-fx)*(1-fy)*fz,fx*(1-fy)*fz,(1-fx)*fy*fz,fx*fy*fz};
    int dz[8]={z0,z0,z0,z0,z0+1,z0+1,z0+1,z0+1};
    int dy[8]={y0,y0,y0+1,y0+1,y0,y0,y0+1,y0+1};
    int dx[8]={x0,x0+1,x0,x0+1,x0,x0+1,x0,x0+1};
    v[0] = v[1] = v[2] = 0;
    for (int k = 0; k < 8; k++) {
        if (dz[k]<0||dz[k]>=D||dy[k]<0||dy[k]>=H||dx[k]<0||dx[k]>=W) continue;
        const float *q = f + (((size_t)dz[k]*H+dy[k])*W+dx[k])*3;
        v[0] += wt[k]*q[0]; v[1] += wt[k]*q[1]; v[2] += wt[k]*q[2];
    }
}

static inline float norm_coord(int i, int n) { return n > 1 ? 2.0f * i / (n - 1) - 1.0f : 0.0f; }

typedef struct {
    const float *field;   /* sampled field */
    const float *offset;  /* per-voxel displacement added to identity */
    float *out;
    int D, H, W, mode;    /* mode 0: out = -sample (inverse); 1: out += sample (compose) */
} disp_sample_context_t;

static void disp_sample_range(size_t begin, size_t end, void *opaque) {
    disp_sample_context_t *c = (disp_sample_context_t *)opaque;
    int H = c->H, W = c->W;
    for (size_t s = begin; s < end; s++) {
        int w = (int)(s % W), h = (int)((s / W) % H), d = (int)(s / ((size_t)H * W));
        const float *o = c->offset + s*3;
        float v[3];
        sample_disp3(c->field, c->D, H, W,
                     norm_coord(w, W) + o[0], norm_coord(h, H) + o[1], norm_coord(d, c->D) + o[2], v);
        float *out = c->out + s*3;
        if (c->mode == 0) { out[0] = -v[0]; out[1] = -v[1]; out[2] = -v[2]; }
        else { out[0] += v[0]; out[1] += v[1]; out[2] += v[2]; }
    }
}

typedef struct {
    const float *in, *kern;
    float *out;
    int D, H, W, klen, axis;
} blur_disp_context_t;

static void blur_disp_range(size_t begin, size_t end, void *opaque) {
    blur_disp_context_t *c = (blur_disp_context_t *)opaque;
    int D = c->D, H = c->H, W = c->W, r = c->klen / 2;
    for (size_t s = begin; s < end; s++) {
        int w = (int)(s % W), h = (int)((s / W) % H), d = (int)(s / ((size_t)H * W));
        float s0 = 0, s1 = 0, s2 = 0;
        for (int k = 0; k < c->klen; k++) {
            int dd = d, hh = h, ww = w;
            if (c->axis == 0) dd += k - r; else if (c->axis == 1) hh += k - r; else ww += k - r;
            if (dd < 0 || dd >= D || hh < 0 || hh >= H || ww < 0 || ww >= W) continue;
            const float *q = c->in + (((size_t)dd*H+hh)*W+ww)*3;
            s0 += q[0]*c->kern[k]; s1 += q[1]*c->kern[k]; s2 += q[2]*c->kern[k];
        }
        c->out[s*3] = s0; c->out[s*3+1] = s1; c->out[s*3+2] = s2;
    }
}

void moving_pyramid_size(int scale, const double fixed_spacing[3],
                         const double moving_spacing[3], int mD, int mH, int mW,
                         int *odD, int *odH, int *odW) {
    const int m[3] = { mW, mH, mD };   /* spacing order: [W,H,D] */
    int out[3];
    for (int a = 0; a < 3; a++) {
        double fs = fixed_spacing[a] > 0 ? fixed_spacing[a] : 1.0;
        double ms = moving_spacing[a] > 0 ? moving_spacing[a] : 1.0;
        double eff = (double)scale * fs / ms;
        if (eff < 1.0) eff = 1.0;
        int n = (int)(m[a] / eff);
        out[a] = n < 8 ? 8 : n;
        if (out[a] > m[a]) out[a] = m[a];
    }
    *odW = out[0]; *odH = out[1]; *odD = out[2];
}

void cpu_blur_disp_dhw3(float *data, int D, int H, int W, float sigma) {
    if (sigma <= 0) return;
    size_t spatial = (size_t)D * H * W, n3 = spatial * 3;
    float *scratch = (float *)malloc(n3 * sizeof(float));
    float *kern = NULL; int klen = 0;
    make_gaussian_kernel(sigma, 2.0f, &kern, &klen);
    blur_disp_context_t c = {data, kern, scratch, D, H, W, klen, 0};
    cfireants_parallel_for(spatial, 4096, blur_disp_range, &c);
    c.in = scratch; c.out = data; c.axis = 1;
    cfireants_parallel_for(spatial, 4096, blur_disp_range, &c);
    c.in = data; c.out = scratch; c.axis = 2;
    cfireants_parallel_for(spatial, 4096, blur_disp_range, &c);
    memcpy(data, scratch, n3 * sizeof(float));
    free(scratch); free(kern);
}

typedef struct {
    const float *grad;
    float *exp_avg, *exp_avg_sq, *dir;
    float beta1, beta2, bc1, bc2, eps, scale;
    float *partial_max; size_t chunk, n;
} adam_context_t;

static void adam_moments_range(size_t begin, size_t end, void *opaque) {
    adam_context_t *c = (adam_context_t *)opaque;
    for (size_t i = begin; i < end; i++) {
        float g = c->grad[i];
        c->exp_avg[i] = c->beta1 * c->exp_avg[i] + (1.0f - c->beta1) * g;
        c->exp_avg_sq[i] = c->beta2 * c->exp_avg_sq[i] + (1.0f - c->beta2) * g * g;
        c->dir[i] = (c->exp_avg[i] / c->bc1) / (sqrtf(c->exp_avg_sq[i] / c->bc2) + c->eps);
    }
}

/* One item = one thread-sized chunk of voxels; max is order-independent. */
static void adam_maxnorm_range(size_t begin, size_t end, void *opaque) {
    adam_context_t *c = (adam_context_t *)opaque;
    for (size_t slot = begin; slot < end; slot++) {
        size_t s1 = (slot + 1) * c->chunk;
        if (s1 > c->n) s1 = c->n;
        float m = 0;
        for (size_t s = slot * c->chunk; s < s1; s++) {
            float dx = c->dir[s*3], dy = c->dir[s*3+1], dz = c->dir[s*3+2];
            float l2 = sqrtf(dx*dx + dy*dy + dz*dz);
            if (l2 > m) m = l2;
        }
        c->partial_max[slot] = m;
    }
}

static void adam_scale_range(size_t begin, size_t end, void *opaque) {
    adam_context_t *c = (adam_context_t *)opaque;
    for (size_t i = begin; i < end; i++) c->dir[i] *= c->scale;
}

void cpu_warp_adam_step(float *warp, const float *grad, float *exp_avg, float *exp_avg_sq,
                        float *adam_dir, int D, int H, int W, int *step_t,
                        float lr, float beta1, float beta2, float eps, float smooth_warp_sigma) {
    size_t spatial = (size_t)D * H * W, n3 = spatial * 3;
    int nt = cfireants_num_threads();
    float partial_max[nt];

    (*step_t)++;
    adam_context_t c = {grad, exp_avg, exp_avg_sq, adam_dir, beta1, beta2,
                        1.0f - powf(beta1, (float)*step_t), 1.0f - powf(beta2, (float)*step_t),
                        eps, 0, partial_max, (spatial + nt - 1) / nt, spatial};
    cfireants_parallel_for(n3, 8192, adam_moments_range, &c);

    cfireants_parallel_for((size_t)nt, 1, adam_maxnorm_range, &c);
    float gradmax = eps;
    for (int t = 0; t < nt; t++) if (partial_max[t] > gradmax) gradmax = partial_max[t];
    if (gradmax < 1.0f) gradmax = 1.0f;
    float half_res = 1.0f / (float)((D > H ? (D > W ? D : W) : (H > W ? H : W)) - 1);
    c.scale = half_res / gradmax * (-lr);
    cfireants_parallel_for(n3, 8192, adam_scale_range, &c);

    /* Compositive update: dir += interp(warp, id + dir), then smooth */
    disp_sample_context_t dc = {warp, adam_dir, adam_dir, D, H, W, 1};
    cfireants_parallel_for(spatial, 4096, disp_sample_range, &dc);
    cpu_blur_disp_dhw3(adam_dir, D, H, W, smooth_warp_sigma);
    memcpy(warp, adam_dir, n3 * sizeof(float));
}

void cpu_warp_inverse(const float *u, float *inv, int D, int H, int W, int n_iters) {
    size_t spatial = (size_t)D * H * W, n3 = spatial * 3;
    memset(inv, 0, n3 * sizeof(float));
    float *tmp = (float *)malloc(n3 * sizeof(float));
    disp_sample_context_t c = {u, inv, tmp, D, H, W, 0};
    for (int iter = 0; iter < n_iters; iter++) {
        cfireants_parallel_for(spatial, 4096, disp_sample_range, &c);
        memcpy(inv, tmp, n3 * sizeof(float));
    }
    free(tmp);
}
