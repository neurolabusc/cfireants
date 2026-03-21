/*
 * metal_fft.m - FFT-based downsampling for Metal backend
 *
 * On Metal with unified memory, the FFT runs on CPU directly on
 * shared-memory buffers (zero-copy). Uses kissfft (vendored in
 * third_party/kissfft/) — the same implementation as the WebGPU backend.
 *
 * The FFT only runs 2-3 times per registration stage (at scale transitions)
 * and takes ~30ms for a 91×109×91 volume. Not a performance bottleneck.
 *
 * Future optimization: replace with MPSGraph FFT for GPU-native transforms.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

#include "metal_context.h"
#include "metal_kernels.h"

/* kissfft (vendored in third_party/kissfft/) */
#include "kiss_fft.h"
#include "kiss_fftnd.h"

/*
 * Faithful clone of Python downsample_fft() via kissfft.
 * Same implementation as webgpu_downsample_fft in fft_cpu_fallback.c.
 *
 * Flow:
 *   1. Forward 3D FFT
 *   2. fftshift
 *   3. Crop to [target_size + 2*padding] centered region
 *   4. Apply Gaussian window: exp(-0.5 * (freq/sigma)^2), sigma = dim/4
 *   5. Scale by prod(target/source)
 *   6. Remove padding
 *   7. ifftshift
 *   8. Inverse 3D FFT, normalize by 1/N
 *   9. Clamp to original range
 */
void metal_downsample_fft(
    const float *input, float *output,
    int B, int C,
    int iD, int iH, int iW,
    int oD, int oH, int oW)
{
    /* Ensure GPU writes are visible to CPU (unified memory) */
    metal_sync();

    long spatial_in  = (long)iD * iH * iW;
    long spatial_out = (long)oD * oH * oW;
    int padding = 1;

    int cD = oD + 2 * padding;
    int cH = oH + 2 * padding;
    int cW = oW + 2 * padding;
    long spatial_crop = (long)cD * cH * cW;

    float multiplier = ((float)oD / iD) * ((float)oH / iH) * ((float)oW / iW);

    /* kissfft setup: forward (iD,iH,iW), inverse (oD,oH,oW) */
    int dims_fwd[3] = { iD, iH, iW };
    int dims_inv[3] = { oD, oH, oW };
    kiss_fftnd_cfg cfg_fwd = kiss_fftnd_alloc(dims_fwd, 3, 0, NULL, NULL);
    kiss_fftnd_cfg cfg_inv = kiss_fftnd_alloc(dims_inv, 3, 1, NULL, NULL);

    kiss_fft_cpx *fft_buf  = (kiss_fft_cpx *)malloc(spatial_in * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *shifted  = (kiss_fft_cpx *)malloc(spatial_in * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *cropped  = (kiss_fft_cpx *)malloc(spatial_crop * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *trimmed  = (kiss_fft_cpx *)malloc(spatial_out * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *unshift  = (kiss_fft_cpx *)malloc(spatial_out * sizeof(kiss_fft_cpx));

    for (int bc = 0; bc < B * C; bc++) {
        const float *src = input + (long)bc * spatial_in;
        float *dst = output + (long)bc * spatial_out;

        /* Find min/max for clamping */
        float vmin = src[0], vmax = src[0];
        for (long i = 1; i < spatial_in; i++) {
            if (src[i] < vmin) vmin = src[i];
            if (src[i] > vmax) vmax = src[i];
        }

        /* Real to complex */
        for (long i = 0; i < spatial_in; i++) {
            fft_buf[i].r = src[i];
            fft_buf[i].i = 0;
        }

        /* Forward 3D FFT */
        kiss_fftnd(cfg_fwd, fft_buf, fft_buf);

        /* fftshift: shift by ceil(N/2) along each dim */
        for (long i = 0; i < spatial_in; i++) {
            int w = i % iW;
            int h = (int)((i / iW) % iH);
            int d = (int)(i / ((long)iH * iW));
            int sd = (d + (iD + 1) / 2) % iD;
            int sh = (h + (iH + 1) / 2) % iH;
            int sw = (w + (iW + 1) / 2) % iW;
            long src_idx = ((long)sd * iH + sh) * iW + sw;
            shifted[i] = fft_buf[src_idx];
        }

        /* Crop centered region */
        int d0 = iD / 2 - (oD / 2 + padding);
        int h0 = iH / 2 - (oH / 2 + padding);
        int w0 = iW / 2 - (oW / 2 + padding);
        for (long i = 0; i < spatial_crop; i++) {
            int dw = i % cW;
            int dh = (int)((i / cW) % cH);
            int dd = (int)(i / ((long)cH * cW));
            long si = ((long)(dd + d0) * iH + (dh + h0)) * iW + (dw + w0);
            cropped[i] = shifted[si];
        }

        /* Gaussian window + scale */
        int zs = -(oD / 2 + padding);
        int ys = -(oH / 2 + padding);
        int xs = -(oW / 2 + padding);
        for (long i = 0; i < spatial_crop; i++) {
            int w = i % cW;
            int h = (int)((i / cW) % cH);
            int d = (int)(i / ((long)cH * cW));
            float sz = (float)cD / 4.0f;
            float sy = (float)cH / 4.0f;
            float sx = (float)cW / 4.0f;
            float zf = (float)(zs + d) / sz;
            float yf = (float)(ys + h) / sy;
            float xf = (float)(xs + w) / sx;
            float scale = expf(-0.5f * (zf*zf + yf*yf + xf*xf)) * multiplier;
            cropped[i].r *= scale;
            cropped[i].i *= scale;
        }

        /* Remove padding: trim [padding:-padding] from each dim */
        for (long i = 0; i < spatial_out; i++) {
            int tw = i % oW;
            int th = (int)((i / oW) % oH);
            int td = (int)(i / ((long)oH * oW));
            long si = ((long)(td + padding) * cH + (th + padding)) * cW + (tw + padding);
            trimmed[i] = cropped[si];
        }

        /* ifftshift: shift by floor(N/2) */
        for (long i = 0; i < spatial_out; i++) {
            int w = i % oW;
            int h = (int)((i / oW) % oH);
            int d = (int)(i / ((long)oH * oW));
            int sd = (d + oD / 2) % oD;
            int sh = (h + oH / 2) % oH;
            int sw = (w + oW / 2) % oW;
            long src_idx = ((long)sd * oH + sh) * oW + sw;
            unshift[i] = trimmed[src_idx];
        }

        /* Inverse 3D FFT */
        kiss_fftnd(cfg_inv, unshift, unshift);

        /* Complex to real + 1/N normalization + clamp */
        float inv_N = 1.0f / (float)spatial_out;
        for (long i = 0; i < spatial_out; i++) {
            float val = unshift[i].r * inv_N;
            if (val < vmin) val = vmin;
            if (val > vmax) val = vmax;
            dst[i] = val;
        }
    }

    free(fft_buf); free(shifted); free(cropped);
    free(trimmed); free(unshift);
    free(cfg_fwd); free(cfg_inv);
}
