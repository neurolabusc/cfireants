/*
 * utils.c - Gaussian blur, Adam optimizer, trilinear resize (CPU)
 */

#include "cfireants/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Gaussian blur                                                       */
/* ------------------------------------------------------------------ */

/* Build 1D Gaussian kernel matching fireants gaussian_1d with approx="erf" */
static int make_gaussian_kernel(float sigma, float truncated,
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
static void conv1d_axis(const float *in, float *out,
                        int D, int H, int W,
                        const float *kernel, int klen, int axis) {
    int r = klen / 2;

    if (axis == 2) { /* W */
        for (int d = 0; d < D; d++)
            for (int h = 0; h < H; h++) {
                size_t row = (size_t)(d * H + h) * W;
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
    } else if (axis == 1) { /* H */
        for (int d = 0; d < D; d++)
            for (int w = 0; w < W; w++)
                for (int h = 0; h < H; h++) {
                    float sum = 0.0f;
                    for (int k = 0; k < klen; k++) {
                        int hh = h + k - r;
                        if (hh >= 0 && hh < H)
                            sum += in[(size_t)(d * H + hh) * W + w] * kernel[k];
                    }
                    out[(size_t)(d * H + h) * W + w] = sum;
                }
    } else { /* axis == 0, D */
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
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
