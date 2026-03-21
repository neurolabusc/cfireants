/*
 * fft_cpu_fallback.c - CPU FFT-based downsampling using kissfft
 *
 * Mirrors cuda_downsample_fft() exactly:
 *   1. Real to complex
 *   2. Forward 3D C2C FFT at input size
 *   3. fftshift
 *   4. Crop centered region (target + 2*padding)
 *   5. Gaussian window * multiplier
 *   6. Remove padding
 *   7. ifftshift
 *   8. Inverse 3D C2C FFT at output size
 *   9. Scale by 1/N (kissfft doesn't normalize inverse)
 *  10. Clamp to original range
 *
 * Used as WebGPU backend fallback since there's no GPU FFT in WGSL.
 * Performance is acceptable — FFT downsampling runs only 2-3 times
 * per registration stage at scale transitions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "kiss_fftnd.h"

void webgpu_downsample_fft(
    const float *input, float *output,
    int B, int C,
    int iD, int iH, int iW,
    int oD, int oH, int oW)
{
    long spatial_in = (long)iD * iH * iW;
    long spatial_out = (long)oD * oH * oW;
    int padding = 1;

    /* Cropped dims (with padding) */
    int cD = oD + 2 * padding;
    int cH = oH + 2 * padding;
    int cW = oW + 2 * padding;
    long spatial_crop = (long)cD * cH * cW;

    /* Start indices for crop (matching Python/CUDA) */
    int d0 = iD / 2 - (oD / 2 + padding);
    int h0 = iH / 2 - (oH / 2 + padding);
    int w0 = iW / 2 - (oW / 2 + padding);

    /* Multiplier = prod(target/source) */
    float multiplier = ((float)oD / iD) * ((float)oH / iH) * ((float)oW / iW);

    /* Frequency-domain start coords for Gaussian window */
    int zs = -(oD / 2 + padding);
    int ys = -(oH / 2 + padding);
    int xs = -(oW / 2 + padding);

    /* Create kissfft plans */
    int dims_fwd[3] = {iD, iH, iW};
    int dims_inv[3] = {oD, oH, oW};
    kiss_fftnd_cfg plan_fwd = kiss_fftnd_alloc(dims_fwd, 3, 0, NULL, NULL);
    kiss_fftnd_cfg plan_inv = kiss_fftnd_alloc(dims_inv, 3, 1, NULL, NULL);

    /* Allocate work buffers */
    kiss_fft_cpx *fft_in      = (kiss_fft_cpx *)malloc(spatial_in * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *fft_out     = (kiss_fft_cpx *)malloc(spatial_in * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *shifted     = (kiss_fft_cpx *)malloc(spatial_in * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *cropped     = (kiss_fft_cpx *)malloc(spatial_crop * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *trimmed     = (kiss_fft_cpx *)malloc(spatial_out * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *unshifted   = (kiss_fft_cpx *)malloc(spatial_out * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *ifft_out    = (kiss_fft_cpx *)malloc(spatial_out * sizeof(kiss_fft_cpx));

    for (int bc = 0; bc < B * C; bc++) {
        const float *src = input + (long)bc * spatial_in;
        float *dst = output + (long)bc * spatial_out;

        /* Get min/max for clamping */
        float vmin = src[0], vmax = src[0];
        for (long i = 1; i < spatial_in; i++) {
            if (src[i] < vmin) vmin = src[i];
            if (src[i] > vmax) vmax = src[i];
        }

        /* Real to complex */
        for (long i = 0; i < spatial_in; i++) {
            fft_in[i].r = src[i];
            fft_in[i].i = 0;
        }

        /* Forward FFT */
        kiss_fftnd(plan_fwd, fft_in, fft_out);

        /* fftshift: out[i] = in[(i + ceil(N/2)) % N] per axis */
        for (int d = 0; d < iD; d++) {
            int sd = (d + (iD + 1) / 2) % iD;
            for (int h = 0; h < iH; h++) {
                int sh = (h + (iH + 1) / 2) % iH;
                for (int w = 0; w < iW; w++) {
                    int sw = (w + (iW + 1) / 2) % iW;
                    long dst_idx = ((long)d * iH + h) * iW + w;
                    long src_idx = ((long)sd * iH + sh) * iW + sw;
                    shifted[dst_idx] = fft_out[src_idx];
                }
            }
        }

        /* Crop centered region */
        for (int d = 0; d < cD; d++) {
            for (int h = 0; h < cH; h++) {
                for (int w = 0; w < cW; w++) {
                    long si = ((long)(d + d0) * iH + (h + h0)) * iW + (w + w0);
                    long di = ((long)d * cH + h) * cW + w;
                    cropped[di] = shifted[si];
                }
            }
        }

        /* Gaussian window in frequency domain + multiplier */
        for (int d = 0; d < cD; d++) {
            float sigma_z = (float)cD / 4.0f;
            float z_freq = (float)(zs + d) / sigma_z;
            for (int h = 0; h < cH; h++) {
                float sigma_y = (float)cH / 4.0f;
                float y_freq = (float)(ys + h) / sigma_y;
                for (int w = 0; w < cW; w++) {
                    float sigma_x = (float)cW / 4.0f;
                    float x_freq = (float)(xs + w) / sigma_x;
                    float exp_val = expf(-0.5f * (z_freq * z_freq + y_freq * y_freq + x_freq * x_freq));
                    float scale = exp_val * multiplier;
                    long idx = ((long)d * cH + h) * cW + w;
                    cropped[idx].r *= scale;
                    cropped[idx].i *= scale;
                }
            }
        }

        /* Remove padding: crop [padding:-padding] from each dim */
        for (int d = 0; d < oD; d++) {
            for (int h = 0; h < oH; h++) {
                for (int w = 0; w < oW; w++) {
                    long si = ((long)(d + padding) * cH + (h + padding)) * cW + (w + padding);
                    long di = ((long)d * oH + h) * oW + w;
                    trimmed[di] = cropped[si];
                }
            }
        }

        /* ifftshift: out[i] = in[(i + N//2) % N] per axis */
        for (int d = 0; d < oD; d++) {
            int sd = (d + oD / 2) % oD;
            for (int h = 0; h < oH; h++) {
                int sh = (h + oH / 2) % oH;
                for (int w = 0; w < oW; w++) {
                    int sw = (w + oW / 2) % oW;
                    long dst_idx = ((long)d * oH + h) * oW + w;
                    long src_idx = ((long)sd * oH + sh) * oW + sw;
                    unshifted[dst_idx] = trimmed[src_idx];
                }
            }
        }

        /* Inverse FFT */
        kiss_fftnd(plan_inv, unshifted, ifft_out);

        /* kissfft inverse doesn't normalize — divide by N
         * (same as cuFFT: neither forward nor inverse normalizes) */
        float inv_N = 1.0f / (float)spatial_out;
        for (long i = 0; i < spatial_out; i++) {
            float val = ifft_out[i].r * inv_N;
            if (val < vmin) val = vmin;
            if (val > vmax) val = vmax;
            dst[i] = val;
        }
    }

    free(fft_in); free(fft_out); free(shifted);
    free(cropped); free(trimmed); free(unshifted); free(ifft_out);
    free(plan_fwd);
    free(plan_inv);
}
