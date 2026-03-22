/*
 * metal_fft.m - FFT-based downsampling via MPSGraph
 *
 * Uses MPSGraph's fastFourierTransform for GPU-native 3D FFT on Apple Silicon.
 * Supports non-power-of-2 sizes natively. The intermediate operations (fftshift,
 * crop, Gaussian window) run on CPU via unified memory — these are O(N) and
 * negligible compared to the FFT itself.
 *
 * Flow matching Python downsample_fft() / CUDA downsample_fft.cu:
 *   1. Forward 3D FFT (MPSGraph, GPU)
 *   2. fftshift (CPU, unified memory)
 *   3. Crop to [target_size + 2*padding] centered region (CPU)
 *   4. Apply Gaussian window (CPU)
 *   5. Scale by prod(target/source)
 *   6. Remove padding (CPU)
 *   7. ifftshift (CPU)
 *   8. Inverse 3D FFT (MPSGraph, GPU), normalize by 1/N
 *   9. Clamp to original range (CPU)
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "metal_context.h"
#include "metal_kernels.h"

/* Complex float type matching MPSDataTypeComplexFloat32 layout (real, imag) */
typedef struct { float real, imag; } cfloat_t;

/* Allocate a Metal shared-memory buffer and register it for dispatch. */
static float *fft_alloc_buf(size_t bytes, id<MTLBuffer> *out_buf) {
    id<MTLBuffer> buf = [g_metal.device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    if (!buf) return NULL;
    float *ptr = (float *)buf.contents;
    metal_register_buffer(ptr, (__bridge void *)buf, bytes);
    *out_buf = buf;
    return ptr;
}

static void fft_free_buf(float *ptr, id<MTLBuffer> buf) {
    if (ptr) metal_unregister_buffer(ptr);
    (void)buf;
}

/*
 * Run a 3D FFT via MPSGraph using MTLBuffer for zero-copy I/O.
 *
 * For forward (inverse=0): real float [D*H*W] → complex [D*H*W*2 floats]
 * For inverse (inverse=1): complex [D*H*W*2 floats] → complex [D*H*W*2 floats]
 *   (caller takes real part)
 */
static void mpsgraph_fft_3d(const void *input_data, size_t input_bytes,
                             void *output_data, size_t output_bytes,
                             int D, int H, int W,
                             int inverse, int is_complex_input)
{
    @autoreleasepool {
        MPSGraph *graph = [[MPSGraph alloc] init];

        NSArray<NSNumber *> *shape = @[@(D), @(H), @(W)];
        MPSDataType inType = is_complex_input ? MPSDataTypeComplexFloat32 : MPSDataTypeFloat32;

        MPSGraphTensor *inputTensor = [graph placeholderWithShape:shape
                                                         dataType:inType
                                                             name:@"input"];

        MPSGraphFFTDescriptor *desc = [MPSGraphFFTDescriptor descriptor];
        desc.inverse = inverse ? YES : NO;
        desc.scalingMode = inverse ? MPSGraphFFTScalingModeSize : MPSGraphFFTScalingModeNone;

        NSArray<NSNumber *> *axes = @[@0, @1, @2];
        MPSGraphTensor *fftResult = [graph fastFourierTransformWithTensor:inputTensor
                                                                     axes:axes
                                                               descriptor:desc
                                                                     name:@"fft3d"];

        /* Create input MTLBuffer (shared memory — wraps existing data) */
        id<MTLBuffer> inBuf = [g_metal.device newBufferWithBytes:input_data
                                                          length:input_bytes
                                                         options:MTLResourceStorageModeShared];

        /* Create output MTLBuffer (shared memory) */
        size_t complex_bytes = (size_t)D * H * W * sizeof(cfloat_t);
        id<MTLBuffer> outBuf = [g_metal.device newBufferWithLength:complex_bytes
                                                           options:MTLResourceStorageModeShared];

        MPSGraphTensorData *inputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:inBuf shape:shape dataType:inType];

        /* For the output, we need to tell MPSGraph where to write.
         * Use runWithMTLCommandQueue:feeds:targetTensors:targetOperations:
         * and then read back from the result's underlying buffer. */
        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
            @{ inputTensor : inputData };

        /* Create output tensor data backed by our buffer */
        MPSGraphTensorData *outputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:outBuf shape:shape dataType:MPSDataTypeComplexFloat32];

        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
            [graph runWithMTLCommandQueue:g_metal.queue
                                    feeds:feeds
                            targetTensors:@[fftResult]
                         targetOperations:nil];

        /* Copy result to output buffer.
         * The graph returns new MPSGraphTensorData; read via mpsndarray → readBytes. */
        MPSGraphTensorData *resultData = results[fftResult];
        MPSNDArray *ndarray = [resultData mpsndarray];

        /* Read the ndarray into our output buffer */
        [ndarray readBytes:output_data strideBytes:nil];
    }
}

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

    /* Allocate complex buffers */
    cfloat_t *fft_out  = (cfloat_t *)malloc(spatial_in * sizeof(cfloat_t));
    cfloat_t *shifted  = (cfloat_t *)malloc(spatial_in * sizeof(cfloat_t));
    cfloat_t *cropped  = (cfloat_t *)malloc(spatial_crop * sizeof(cfloat_t));
    cfloat_t *trimmed  = (cfloat_t *)malloc(spatial_out * sizeof(cfloat_t));
    cfloat_t *unshift  = (cfloat_t *)malloc(spatial_out * sizeof(cfloat_t));
    cfloat_t *ifft_out = (cfloat_t *)malloc(spatial_out * sizeof(cfloat_t));

    for (int bc = 0; bc < B * C; bc++) {
        const float *src = input + (long)bc * spatial_in;
        float *dst = output + (long)bc * spatial_out;

        /* Find min/max for clamping */
        float vmin = src[0], vmax = src[0];
        for (long i = 1; i < spatial_in; i++) {
            if (src[i] < vmin) vmin = src[i];
            if (src[i] > vmax) vmax = src[i];
        }

        /* Forward 3D FFT via MPSGraph (real → complex) */
        mpsgraph_fft_3d(src, spatial_in * sizeof(float),
                         fft_out, spatial_in * sizeof(cfloat_t),
                         iD, iH, iW, 0, 0);

        /* fftshift: shift by ceil(N/2) along each dim */
        for (long i = 0; i < spatial_in; i++) {
            int w = i % iW;
            int h = (int)((i / iW) % iH);
            int d = (int)(i / ((long)iH * iW));
            int sd = (d + (iD + 1) / 2) % iD;
            int sh = (h + (iH + 1) / 2) % iH;
            int sw = (w + (iW + 1) / 2) % iW;
            long src_idx = ((long)sd * iH + sh) * iW + sw;
            shifted[i] = fft_out[src_idx];
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
            cropped[i].real *= scale;
            cropped[i].imag *= scale;
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

        /* Inverse 3D FFT via MPSGraph (complex → complex, with 1/N scaling) */
        mpsgraph_fft_3d(unshift, spatial_out * sizeof(cfloat_t),
                         ifft_out, spatial_out * sizeof(cfloat_t),
                         oD, oH, oW, 1, 1);

        /* Take real part + clamp to original range */
        for (long i = 0; i < spatial_out; i++) {
            float val = ifft_out[i].real;
            if (val < vmin) val = vmin;
            if (val > vmax) val = vmax;
            dst[i] = val;
        }
    }

    free(fft_out); free(shifted); free(cropped);
    free(trimmed); free(unshift); free(ifft_out);
}

/* ------------------------------------------------------------------ */
/* Gaussian blur + trilinear resize (GPU-native, no FFT)               */
/* ------------------------------------------------------------------ */

static float *make_gauss_kernel_fft(float sigma, int *out_klen) {
    if (sigma <= 0) {
        float *k = (float *)malloc(sizeof(float));
        k[0] = 1.0f;
        *out_klen = 1;
        return k;
    }
    int tail = (int)(2.0f * sigma + 0.5f);
    int klen = 2 * tail + 1;
    float *h = (float *)malloc(klen * sizeof(float));
    float inv = 1.0f / (sigma * sqrtf(2.0f));
    float ksum = 0;
    for (int i = 0; i < klen; i++) {
        float x = (float)(i - tail);
        h[i] = 0.5f * (erff((x+0.5f)*inv) - erff((x-0.5f)*inv));
        ksum += h[i];
    }
    for (int i = 0; i < klen; i++) h[i] /= ksum;
    *out_klen = klen;
    return h;
}

void metal_blur_volume(float *data, int D, int H, int W,
                        float sigma_d, float sigma_h, float sigma_w)
{
    size_t sz = (size_t)D * H * W * sizeof(float);
    id<MTLBuffer> scratch_buf;
    float *scratch = fft_alloc_buf(sz, &scratch_buf);

    float sigmas[3] = { sigma_d, sigma_h, sigma_w };
    for (int axis = 0; axis < 3; axis++) {
        int klen;
        float *h_k = make_gauss_kernel_fft(sigmas[axis], &klen);

        id<MTLBuffer> kern_buf;
        float *d_k = fft_alloc_buf(klen * sizeof(float), &kern_buf);
        memcpy(d_k, h_k, klen * sizeof(float));
        free(h_k);

        metal_conv1d_axis(data, scratch, D, H, W, d_k, klen, axis);
        metal_sync();
        memcpy(data, scratch, sz);

        fft_free_buf(d_k, kern_buf);
    }
    fft_free_buf(scratch, scratch_buf);
}

void metal_blur_downsample(const float *input, float *output,
                            int B, int C, int iD, int iH, int iW,
                            int oD, int oH, int oW)
{
    metal_sync();

    size_t in_sz = (size_t)iD * iH * iW * sizeof(float);

    /* Allocate temporary for blur (don't modify input) */
    id<MTLBuffer> blur_buf;
    float *blurred = fft_alloc_buf(in_sz, &blur_buf);
    memcpy(blurred, input, in_sz);

    /* Gaussian blur with sigma = 0.5 * (in_dim / out_dim) per axis */
    float sig_d = 0.5f * (float)iD / (float)oD;
    float sig_h = 0.5f * (float)iH / (float)oH;
    float sig_w = 0.5f * (float)iW / (float)oW;
    metal_blur_volume(blurred, iD, iH, iW, sig_d, sig_h, sig_w);

    /* Trilinear resize via hardware 3D texture sampling */
    metal_trilinear_resize_texture(blurred, output, iD, iH, iW, oD, oH, oW, 1);
    metal_sync();

    fft_free_buf(blurred, blur_buf);
}
