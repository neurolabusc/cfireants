/*
 * metal_kernels.m - Metal kernel dispatch wrappers
 *
 * Implements dispatch wrappers for element-wise and reduction operations.
 * Each function looks up the compute pipeline by name, binds buffers and
 * parameters, and dispatches via metal_dispatch().
 *
 * The corresponding Metal shader functions are in shaders/elementwise.metal
 * and shaders/reduction.metal (created separately).
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "metal_context.h"
#include "metal_kernels.h"

/* --- Param structs matching Metal shader buffer layouts --- */

/* Must match shader struct layouts exactly (field order matters!) */
typedef struct {
    uint32_t n;
    float value;
} ew_params_t;  /* matches EWParams in elementwise.metal */

typedef struct {
    uint32_t n;
    float alpha;
} axpy_params_t;  /* matches AXPYParams in elementwise.metal */

typedef struct {
    int n;
} reduce_params_t;

/* Phase 2: grid sampling, affine grid, resize */
typedef struct { uint32_t B, C, iD, iH, iW, oD, oH, oW; } gs_params_t;
typedef struct { uint32_t B, D, H, W; } ag_params_t;
typedef struct { uint32_t B, C, iD, iH, iW, oD, oH, oW, align_corners, _pad; } resize_params_t;
typedef struct { uint32_t D, H, W, klen, axis, _pad0, _pad1, _pad2; } conv1d_params_t;

/* --- Element-wise operations --- */

void metal_tensor_fill(float *data, float value, int n) {
    void *pso = metal_get_pipeline("tensor_fill");
    if (!pso) {
        /* CPU fallback */
        for (int i = 0; i < n; i++) data[i] = value;
        return;
    }

    ew_params_t params = { .n = (uint32_t)n, .value = value };
    const void *bufs[] = { data };
    size_t sizes[] = { (size_t)n * sizeof(float) };

    metal_dispatch(pso, bufs, sizes, 1, &params, sizeof(params), (uint32_t)n);
}

void metal_tensor_scale(float *data, float alpha, int n) {
    void *pso = metal_get_pipeline("tensor_scale");
    if (!pso) {
        /* CPU fallback */
        for (int i = 0; i < n; i++) data[i] *= alpha;
        return;
    }

    ew_params_t params = { .n = (uint32_t)n, .value = alpha };
    const void *bufs[] = { data };
    size_t sizes[] = { (size_t)n * sizeof(float) };

    metal_dispatch(pso, bufs, sizes, 1, &params, sizeof(params), (uint32_t)n);
}

void metal_tensor_axpy(float *y, float alpha, const float *x, int n) {
    void *pso = metal_get_pipeline("tensor_axpy");
    if (!pso) {
        /* CPU fallback */
        for (int i = 0; i < n; i++) y[i] += alpha * x[i];
        return;
    }

    axpy_params_t params = { .n = (uint32_t)n, .alpha = alpha };
    const void *bufs[] = { y, x };
    size_t sizes[] = { (size_t)n * sizeof(float), (size_t)n * sizeof(float) };

    metal_dispatch(pso, bufs, sizes, 2, &params, sizeof(params), (uint32_t)n);
}

/* --- Reductions --- */

/*
 * Sum and mean use a two-pass approach:
 *   Pass 1: GPU partial sums (one per threadgroup) via "tensor_reduce_sum" shader
 *   Pass 2: CPU final sum of partial results
 *
 * For Phase 1, we use a simple CPU fallback since the reduction shader
 * requires careful synchronization. The GPU path will be enabled once
 * shaders/reduction.metal is validated.
 */

float metal_tensor_sum(const float *data, int n) {
    /* Sync to ensure GPU writes are visible */
    metal_sync();

    /* CPU reduction (data is in shared memory, CPU-accessible) */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)data[i];
    }
    return (float)sum;
}

float metal_tensor_mean(const float *data, int n) {
    if (n == 0) return 0.0f;
    return metal_tensor_sum(data, n) / (float)n;
}

/* --- Phase 2: Grid sampling, affine grid, resize --- */

void metal_grid_sample_3d_fwd(const float *input, const float *grid, float *output,
                               int B, int C, int iD, int iH, int iW,
                               int oD, int oH, int oW) {
    void *pso = metal_get_pipeline("grid_sample_3d_fwd");
    if (!pso) {
        fprintf(stderr, "metal_grid_sample_3d_fwd: pipeline not found\n");
        return;
    }

    gs_params_t params = {
        .B = (uint32_t)B, .C = (uint32_t)C,
        .iD = (uint32_t)iD, .iH = (uint32_t)iH, .iW = (uint32_t)iW,
        .oD = (uint32_t)oD, .oH = (uint32_t)oH, .oW = (uint32_t)oW
    };

    const void *bufs[] = { input, grid, output };
    size_t sizes[] = {
        (size_t)B * C * iD * iH * iW * sizeof(float),
        (size_t)B * oD * oH * oW * 3 * sizeof(float),
        (size_t)B * C * oD * oH * oW * sizeof(float)
    };

    uint32_t total_threads = (uint32_t)(B * oD * oH * oW);
    metal_dispatch(pso, bufs, sizes, 3, &params, sizeof(params), total_threads);
}

void metal_grid_sample_3d_bwd(const float *grad_output, const float *input,
                               const float *grid, float *grad_grid,
                               int B, int C, int iD, int iH, int iW,
                               int oD, int oH, int oW) {
    void *pso = metal_get_pipeline("grid_sample_3d_bwd");
    if (!pso) {
        fprintf(stderr, "metal_grid_sample_3d_bwd: pipeline not found\n");
        return;
    }

    gs_params_t params = {
        .B = (uint32_t)B, .C = (uint32_t)C,
        .iD = (uint32_t)iD, .iH = (uint32_t)iH, .iW = (uint32_t)iW,
        .oD = (uint32_t)oD, .oH = (uint32_t)oH, .oW = (uint32_t)oW
    };

    const void *bufs[] = { grad_output, input, grid, grad_grid };
    size_t sizes[] = {
        (size_t)B * C * oD * oH * oW * sizeof(float),
        (size_t)B * C * iD * iH * iW * sizeof(float),
        (size_t)B * oD * oH * oW * 3 * sizeof(float),
        (size_t)B * oD * oH * oW * 3 * sizeof(float)
    };

    uint32_t total_threads = (uint32_t)(B * oD * oH * oW);
    metal_dispatch(pso, bufs, sizes, 4, &params, sizeof(params), total_threads);
}

void metal_affine_grid_3d(const float *affine, float *grid,
                           int B, int D, int H, int W) {
    void *pso = metal_get_pipeline("affine_grid_3d");
    if (!pso) {
        fprintf(stderr, "metal_affine_grid_3d: pipeline not found\n");
        return;
    }

    ag_params_t params = {
        .B = (uint32_t)B, .D = (uint32_t)D,
        .H = (uint32_t)H, .W = (uint32_t)W
    };

    const void *bufs[] = { affine, grid };
    size_t sizes[] = {
        (size_t)B * 3 * 4 * sizeof(float),
        (size_t)B * D * H * W * 3 * sizeof(float)
    };

    uint32_t total_threads = (uint32_t)(B * D * H * W);
    metal_dispatch(pso, bufs, sizes, 2, &params, sizeof(params), total_threads);
}

void metal_trilinear_resize(const float *input, float *output,
                             int B, int C, int iD, int iH, int iW,
                             int oD, int oH, int oW, int align_corners) {
    void *pso = metal_get_pipeline("trilinear_resize");
    if (!pso) {
        fprintf(stderr, "metal_trilinear_resize: pipeline not found\n");
        return;
    }

    resize_params_t params = {
        .B = (uint32_t)B, .C = (uint32_t)C,
        .iD = (uint32_t)iD, .iH = (uint32_t)iH, .iW = (uint32_t)iW,
        .oD = (uint32_t)oD, .oH = (uint32_t)oH, .oW = (uint32_t)oW,
        .align_corners = (uint32_t)align_corners, ._pad = 0
    };

    const void *bufs[] = { input, output };
    size_t sizes[] = {
        (size_t)B * C * iD * iH * iW * sizeof(float),
        (size_t)B * C * oD * oH * oW * sizeof(float)
    };

    uint32_t total_threads = (uint32_t)(B * C * oD * oH * oW);
    metal_dispatch(pso, bufs, sizes, 2, &params, sizeof(params), total_threads);
}

/* Hardware 3D texture trilinear resize — single channel only.
 * Creates a temporary MTLTexture from the input buffer, then dispatches
 * the texture-sampling kernel. The GPU's texture unit performs the
 * 8-point interpolation in hardware. */
void metal_trilinear_resize_texture(const float *input, float *output,
                                     int iD, int iH, int iW,
                                     int oD, int oH, int oW, int align_corners) {
    void *pso = metal_get_pipeline("trilinear_resize_texture");
    if (!pso) {
        fprintf(stderr, "metal_trilinear_resize_texture: pipeline not found, falling back\n");
        metal_trilinear_resize(input, output, 1, 1, iD, iH, iW, oD, oH, oW, align_corners);
        return;
    }

    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)metal_buffer_from_ptr(output);
    if (!out_buf) {
        fprintf(stderr, "metal_trilinear_resize_texture: output buffer not registered, falling back\n");
        metal_trilinear_resize(input, output, 1, 1, iD, iH, iW, oD, oH, oW, align_corners);
        return;
    }

    @autoreleasepool {
        MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
        desc.textureType = MTLTextureType3D;
        desc.pixelFormat = MTLPixelFormatR32Float;
        desc.width = (NSUInteger)iW;
        desc.height = (NSUInteger)iH;
        desc.depth = (NSUInteger)iD;
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;

        id<MTLTexture> tex = [g_metal.device newTextureWithDescriptor:desc];
        if (!tex) {
            fprintf(stderr, "metal_trilinear_resize_texture: failed to create texture\n");
            metal_trilinear_resize(input, output, 1, 1, iD, iH, iW, oD, oH, oW, align_corners);
            return;
        }

        MTLRegion region = MTLRegionMake3D(0, 0, 0, (NSUInteger)iW, (NSUInteger)iH, (NSUInteger)iD);
        [tex replaceRegion:region mipmapLevel:0 slice:0
             withBytes:input bytesPerRow:(size_t)iW * sizeof(float)
             bytesPerImage:(size_t)iW * iH * sizeof(float)];

        typedef struct { uint32_t oD, oH, oW, iD, iH, iW, align_corners, _pad; } tex_resize_params_t;
        tex_resize_params_t params = {
            .oD = (uint32_t)oD, .oH = (uint32_t)oH, .oW = (uint32_t)oW,
            .iD = (uint32_t)iD, .iH = (uint32_t)iH, .iW = (uint32_t)iW,
            .align_corners = (uint32_t)align_corners, ._pad = 0
        };

        id<MTLBuffer> param_buf = [g_metal.device newBufferWithBytes:&params
                                                              length:sizeof(params)
                                                             options:MTLResourceStorageModeShared];
        uint32_t total = (uint32_t)(oD * oH * oW);
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)pso;
        NSUInteger tw = pipeline.maxTotalThreadsPerThreadgroup;
        if (tw > 256) tw = 256;

        id<MTLCommandBuffer> cmd = [g_metal.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        [enc setTexture:tex atIndex:0];
        [enc setBuffer:out_buf offset:0 atIndex:0];
        [enc setBuffer:param_buf offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void metal_conv1d_axis(const float *in, float *out,
                        int D, int H, int W,
                        const float *kernel, int klen, int axis) {
    void *pso = metal_get_pipeline("conv1d_axis");
    if (!pso) {
        fprintf(stderr, "metal_conv1d_axis: pipeline not found\n");
        return;
    }

    conv1d_params_t params = {
        .D = (uint32_t)D, .H = (uint32_t)H, .W = (uint32_t)W,
        .klen = (uint32_t)klen, .axis = (uint32_t)axis,
        ._pad0 = 0, ._pad1 = 0, ._pad2 = 0
    };

    const void *bufs[] = { in, out, kernel };
    size_t sizes[] = {
        (size_t)D * H * W * sizeof(float),
        (size_t)D * H * W * sizeof(float),
        (size_t)klen * sizeof(float)
    };

    uint32_t total_threads = (uint32_t)(D * H * W);
    metal_dispatch(pso, bufs, sizes, 3, &params, sizeof(params), total_threads);
}

/* --- Phase 3 param structs matching Metal shader buffer layouts --- */

typedef struct { uint32_t spatial, _pad; } fcc_params_t;
typedef struct { uint32_t spatial; int32_t kernel_volume; float nr, dr; } fcc_fwd_params_t;
typedef struct { uint32_t spatial; int32_t kernel_volume; float nr, dr, grad_output_val; int32_t compute_grad_target; } fcc_bwd_params_t;
typedef struct { uint32_t spatial, has_grad_target; } fcc_grad_params_t;
typedef struct { uint32_t D, H, W, klen; } blur_params_t;

void metal_box_filter_axis(const float *in, float *out,
                            int D, int H, int W, int ks, int axis, float scale) {
    /* Build a uniform kernel of length ks filled with scale */
    @autoreleasepool {
        id<MTLDevice> device = g_metal.device;
        id<MTLBuffer> kern_buf = [device newBufferWithLength:(NSUInteger)(ks * sizeof(float))
                                                    options:MTLResourceStorageModeShared];
        float *kern_ptr = (float *)[kern_buf contents];
        for (int i = 0; i < ks; i++) {
            kern_ptr[i] = scale;
        }

        /* Register the temporary buffer so metal_dispatch can find it */
        metal_register_buffer(kern_ptr, (__bridge void *)kern_buf, (size_t)(ks * sizeof(float)));

        metal_conv1d_axis(in, out, D, H, W, kern_ptr, ks, axis);

        /* Sync before releasing the temporary buffer */
        metal_sync();

        metal_unregister_buffer(kern_ptr);
    }
}

/* --- Phase 3: CC loss + Gaussian blur --- */

/*
 * box_filter_intermediates_metal — apply 3-pass separable box filter to each
 * of 5 channels packed in interm[5*spatial].  For each channel, filters along
 * axes D, H, W in order (matching CUDA's separable_box_filter_gpu).
 * Uses scratch as temporary.
 */
static void box_filter_intermediates_metal(float *interm, int spatial,
                                            int D, int H, int W, int ks,
                                            float *scratch) {
    float scale = 1.0f / (float)ks;
    for (int ch = 0; ch < 5; ch++) {
        float *chan = interm + (size_t)ch * spatial;
        /* D axis: chan -> scratch */
        metal_box_filter_axis(chan, scratch, D, H, W, ks, 0, scale);
        /* H axis: scratch -> chan */
        metal_box_filter_axis(scratch, chan, D, H, W, ks, 1, scale);
        /* W axis: chan -> scratch */
        metal_box_filter_axis(chan, scratch, D, H, W, ks, 2, scale);
        /* Copy scratch back to chan (unified memory) */
        metal_sync();
        memcpy(chan, scratch, (size_t)spatial * sizeof(float));
    }
}

/*
 * separable_box_filter_metal — apply 3-pass separable box filter to a single
 * buffer. Axis order: D→tmp, H→out, W→tmp, copy back.
 * Matches CUDA's separable_box_filter_gpu exactly.
 */
static void separable_box_filter_metal(const float *in, float *out,
                                        int D, int H, int W, int ks,
                                        float *tmp) {
    float scale = 1.0f / (float)ks;
    /* D axis: in -> tmp */
    metal_box_filter_axis(in, tmp, D, H, W, ks, 0, scale);
    /* H axis: tmp -> out */
    metal_box_filter_axis(tmp, out, D, H, W, ks, 1, scale);
    /* W axis: out -> tmp */
    metal_box_filter_axis(out, tmp, D, H, W, ks, 2, scale);
    /* Copy tmp back to out (unified memory) */
    metal_sync();
    memcpy(out, tmp, (size_t)D * H * W * sizeof(float));
}

void metal_fused_cc_loss(
    const float *pred, const float *target,
    float *grad_pred, float *grad_target,
    int D, int H, int W, int ks,
    float *h_loss_out,
    float *interm, float *scratch)
{
    uint32_t spatial = (uint32_t)(D * H * W);

    /* Step 1: create intermediates (pred*pred, target*target, pred*target, pred, target) */
    {
        void *pso = metal_get_pipeline("fcc_create_intermediates");
        if (!pso) {
            fprintf(stderr, "metal_fused_cc_loss: fcc_create_intermediates pipeline not found\n");
            return;
        }
        fcc_params_t params = { .spatial = spatial, ._pad = 0 };
        const void *bufs[] = { pred, target, interm };
        size_t sizes[] = {
            (size_t)spatial * sizeof(float),
            (size_t)spatial * sizeof(float),
            (size_t)5 * spatial * sizeof(float)
        };
        metal_dispatch(pso, bufs, sizes, 3, &params, sizeof(params), spatial);
    }

    /* Step 2: box filter the 5 intermediate channels */
    box_filter_intermediates_metal(interm, (int)spatial, D, H, W, ks, scratch);

    /* Step 3: forward pass — compute loss if requested */
    if (h_loss_out) {
        void *pso = metal_get_pipeline("fcc_fwd");
        if (!pso) {
            fprintf(stderr, "metal_fused_cc_loss: fcc_fwd pipeline not found\n");
            return;
        }
        int kv = ks * ks * ks;
        fcc_fwd_params_t params = { .spatial = spatial, .kernel_volume = (int32_t)kv,
                                     .nr = 1e-5f, .dr = 1e-5f };
        /* scratch used for partial sums (one per threadgroup) */
        const void *bufs[] = { interm, scratch };
        size_t sizes[] = {
            (size_t)5 * spatial * sizeof(float),
            (size_t)spatial * sizeof(float)
        };
        metal_dispatch(pso, bufs, sizes, 2, &params, sizeof(params), spatial);

        /* Read back partial sums and reduce on CPU.
         * fcc_fwd writes one partial sum per threadgroup. */
        metal_sync();
        uint32_t n_groups = metal_div_ceil(spatial, MTL_THREADGROUP_SIZE);
        double sum = 0.0;
        for (uint32_t i = 0; i < n_groups; i++) {
            sum += (double)scratch[i];
        }
        *h_loss_out = (float)(-sum / (double)spatial);
    }

    /* Step 4: backward pass if gradients requested */
    if (grad_pred) {
        /* Step 4a: modify intermediates for backward */
        {
            void *pso = metal_get_pipeline("fcc_bwd_modify");
            if (!pso) {
                fprintf(stderr, "metal_fused_cc_loss: fcc_bwd_modify pipeline not found\n");
                return;
            }
            int kv = ks * ks * ks;
            fcc_bwd_params_t params = {
                .spatial = spatial,
                .kernel_volume = (int32_t)kv,
                .nr = 1e-5f, .dr = 1e-5f,
                .grad_output_val = -1.0f / (float)spatial,
                .compute_grad_target = (grad_target != NULL) ? 1 : 0
            };
            const void *bufs[] = { interm, pred, target };
            size_t sizes[] = {
                (size_t)5 * spatial * sizeof(float),
                (size_t)spatial * sizeof(float),
                (size_t)spatial * sizeof(float)
            };
            metal_dispatch(pso, bufs, sizes, 3, &params, sizeof(params), spatial);
        }

        /* Step 4b: box filter intermediates again (adjoint) */
        box_filter_intermediates_metal(interm, (int)spatial, D, H, W, ks, scratch);

        /* Step 4c: compute final gradients */
        {
            void *pso = metal_get_pipeline("fcc_bwd_grads");
            if (!pso) {
                fprintf(stderr, "metal_fused_cc_loss: fcc_bwd_grads pipeline not found\n");
                return;
            }
            fcc_grad_params_t params = {
                .spatial = spatial,
                .has_grad_target = (grad_target != NULL) ? 1u : 0u
            };
            const void *bufs[] = { interm, pred, target, grad_pred,
                                   grad_target ? grad_target : grad_pred /* dummy if NULL */ };
            size_t sizes[] = {
                (size_t)spatial * sizeof(float),
                (size_t)spatial * sizeof(float),
                (size_t)5 * spatial * sizeof(float),
                (size_t)spatial * sizeof(float),
                (size_t)spatial * sizeof(float)
            };
            metal_dispatch(pso, bufs, sizes, 5, &params, sizeof(params), spatial);
        }
    }
}

void metal_blur_disp_dhw3(float *data, float *scratch,
                           int D, int H, int W,
                           const float *kernel_data, int klen) {
    uint32_t spatial = (uint32_t)(D * H * W);
    uint32_t total = spatial * 3;

    blur_params_t params = {
        .D = (uint32_t)D, .H = (uint32_t)H,
        .W = (uint32_t)W, .klen = (uint32_t)klen
    };

    /* Axis 0 (D): data -> scratch */
    {
        void *pso = metal_get_pipeline("blur_dhw3_axis0");
        if (!pso) {
            fprintf(stderr, "metal_blur_disp_dhw3: blur_dhw3_axis0 pipeline not found\n");
            return;
        }
        const void *bufs[] = { data, scratch, kernel_data };
        size_t sizes[] = {
            (size_t)total * sizeof(float),
            (size_t)total * sizeof(float),
            (size_t)klen * sizeof(float)
        };
        metal_dispatch(pso, bufs, sizes, 3, &params, sizeof(params), total);
    }

    /* Axis 1 (H): scratch -> data */
    {
        void *pso = metal_get_pipeline("blur_dhw3_axis1");
        if (!pso) {
            fprintf(stderr, "metal_blur_disp_dhw3: blur_dhw3_axis1 pipeline not found\n");
            return;
        }
        const void *bufs[] = { scratch, data, kernel_data };
        size_t sizes[] = {
            (size_t)total * sizeof(float),
            (size_t)total * sizeof(float),
            (size_t)klen * sizeof(float)
        };
        metal_dispatch(pso, bufs, sizes, 3, &params, sizeof(params), total);
    }

    /* Axis 2 (W): data -> scratch */
    {
        void *pso = metal_get_pipeline("blur_dhw3_axis2");
        if (!pso) {
            fprintf(stderr, "metal_blur_disp_dhw3: blur_dhw3_axis2 pipeline not found\n");
            return;
        }
        const void *bufs[] = { data, scratch, kernel_data };
        size_t sizes[] = {
            (size_t)total * sizeof(float),
            (size_t)total * sizeof(float),
            (size_t)klen * sizeof(float)
        };
        metal_dispatch(pso, bufs, sizes, 3, &params, sizeof(params), total);
    }

    /* Copy result back: scratch -> data (unified memory) */
    metal_sync();
    memcpy(data, scratch, (size_t)total * sizeof(float));
}

/*
 * metal_cc_loss_3d_v2 — CC loss matching CUDA's cc_loss.cu algorithm exactly.
 *
 * Uses separate buffers for each of the 5 filtered intermediates, with
 * 8 box-filter passes (5 forward + 3 adjoint) instead of fused_cc's 10
 * (5 forward + 5 adjoint). The gradient formula is different but
 * mathematically equivalent.
 *
 * Algorithm:
 *   1. Box-filter: pred, target, pred², target², pred*target
 *      → p_sum, t_sum, p2_sum, t2_sum, tp_sum
 *   2. Compute per-voxel NCC + gradient sources:
 *      cross = tp_sum - p_sum * t_sum
 *      p_var = max(p2_sum - p_sum², dr)
 *      t_var = max(t2_sum - t_sum², dr)
 *      ncc = (cross² + nr) / (p_var * t_var + dr)
 *      src_tp = 2*cross*g / g²
 *      src_p2 = -f*t_var / g²
 *      src_p = (-2*cross*t_sum*g + 2*f*p_sum*t_var) / g²
 *   3. Adjoint box-filter: src_p, src_p2, src_tp → adj_p, adj_p2, adj_tp
 *   4. Combine: grad = -inv_count * (adj_p + 2*P*adj_p2 + T*adj_tp)
 */
static void metal_cc_loss_3d_v2(const float *pred, const float *target,
                                 float *grad_pred,
                                 int D, int H, int W, int ks, float *h_loss_out) {
    @autoreleasepool {
        uint32_t n = (uint32_t)(D * H * W);
        size_t buf_bytes = (size_t)n * sizeof(float);
        float nr = 1e-5f, dr = 1e-5f;

        /* Declare all ARC variables up front to avoid goto-past-init errors */
        id<MTLBuffer> src_p_buf = nil, src_p2_buf = nil, src_tp_buf = nil;
        float *src_p = NULL, *src_p2 = NULL, *src_tp = NULL;
        int compute_grad = (grad_pred != NULL);

        /* Allocate 7 work buffers: p_sum, t_sum, p2_sum, t2_sum, tp_sum, work, tmp */
        id<MTLBuffer> p_sum_buf  = [g_metal.device newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> t_sum_buf  = [g_metal.device newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> p2_sum_buf = [g_metal.device newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> t2_sum_buf = [g_metal.device newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> tp_sum_buf = [g_metal.device newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> work_buf   = [g_metal.device newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> tmp_buf    = [g_metal.device newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];

        if (!p_sum_buf || !t_sum_buf || !p2_sum_buf || !t2_sum_buf ||
            !tp_sum_buf || !work_buf || !tmp_buf) {
            fprintf(stderr, "metal_cc_loss_3d_v2: failed to allocate workspace\n");
            return;
        }

        float *p_sum  = (float *)[p_sum_buf contents];
        float *t_sum  = (float *)[t_sum_buf contents];
        float *p2_sum = (float *)[p2_sum_buf contents];
        float *t2_sum = (float *)[t2_sum_buf contents];
        float *tp_sum = (float *)[tp_sum_buf contents];
        float *work   = (float *)[work_buf contents];
        float *tmp    = (float *)[tmp_buf contents];

        metal_register_buffer(p_sum,  (__bridge void *)p_sum_buf,  buf_bytes);
        metal_register_buffer(t_sum,  (__bridge void *)t_sum_buf,  buf_bytes);
        metal_register_buffer(p2_sum, (__bridge void *)p2_sum_buf, buf_bytes);
        metal_register_buffer(t2_sum, (__bridge void *)t2_sum_buf, buf_bytes);
        metal_register_buffer(tp_sum, (__bridge void *)tp_sum_buf, buf_bytes);
        metal_register_buffer(work,   (__bridge void *)work_buf,   buf_bytes);
        metal_register_buffer(tmp,    (__bridge void *)tmp_buf,    buf_bytes);

        /* --- Step 1: Box filter the 5 forward intermediates --- */

        /* box(pred) → p_sum */
        separable_box_filter_metal(pred, p_sum, D, H, W, ks, tmp);

        /* box(target) → t_sum */
        separable_box_filter_metal(target, t_sum, D, H, W, ks, tmp);

        /* pred² → work, box(work) → p2_sum */
        {
            void *pso = metal_get_pipeline("cc_multiply");
            if (!pso) { fprintf(stderr, "metal_cc_loss_3d_v2: cc_multiply pipeline not found\n"); goto cleanup; }
            struct { uint32_t n, _pad; } params = { .n = n, ._pad = 0 };
            const void *bufs[] = { pred, pred, work };
            size_t sizes[] = { buf_bytes, buf_bytes, buf_bytes };
            metal_dispatch(pso, bufs, sizes, 3, &params, sizeof(params), n);
        }
        separable_box_filter_metal(work, p2_sum, D, H, W, ks, tmp);

        /* target² → work, box(work) → t2_sum */
        {
            void *pso = metal_get_pipeline("cc_multiply");
            struct { uint32_t n, _pad; } params = { .n = n, ._pad = 0 };
            const void *bufs[] = { target, target, work };
            size_t sizes[] = { buf_bytes, buf_bytes, buf_bytes };
            metal_dispatch(pso, bufs, sizes, 3, &params, sizeof(params), n);
        }
        separable_box_filter_metal(work, t2_sum, D, H, W, ks, tmp);

        /* pred*target → work, box(work) → tp_sum */
        {
            void *pso = metal_get_pipeline("cc_multiply");
            struct { uint32_t n, _pad; } params = { .n = n, ._pad = 0 };
            const void *bufs[] = { pred, target, work };
            size_t sizes[] = { buf_bytes, buf_bytes, buf_bytes };
            metal_dispatch(pso, bufs, sizes, 3, &params, sizeof(params), n);
        }
        separable_box_filter_metal(work, tp_sum, D, H, W, ks, tmp);

        /* --- Step 2: Compute NCC and gradient source terms --- */

        /* Reuse work buffer for NCC values */
        float *ncc_buf = work;

        if (compute_grad) {
            src_p_buf  = [g_metal.device newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];
            src_p2_buf = [g_metal.device newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];
            src_tp_buf = [g_metal.device newBufferWithLength:buf_bytes options:MTLResourceStorageModeShared];
            if (!src_p_buf || !src_p2_buf || !src_tp_buf) {
                fprintf(stderr, "metal_cc_loss_3d_v2: failed to allocate gradient source buffers\n");
                goto cleanup;
            }
            src_p  = (float *)[src_p_buf contents];
            src_p2 = (float *)[src_p2_buf contents];
            src_tp = (float *)[src_tp_buf contents];
            metal_register_buffer(src_p,  (__bridge void *)src_p_buf,  buf_bytes);
            metal_register_buffer(src_p2, (__bridge void *)src_p2_buf, buf_bytes);
            metal_register_buffer(src_tp, (__bridge void *)src_tp_buf, buf_bytes);
        }

        {
            void *pso = metal_get_pipeline("cc_ncc_and_grad_sources");
            if (!pso) { fprintf(stderr, "metal_cc_loss_3d_v2: cc_ncc_and_grad_sources pipeline not found\n"); goto cleanup_grad; }
            struct { uint32_t n; float nr, dr; int32_t compute_grad; } params = {
                .n = n, .nr = nr, .dr = dr, .compute_grad = compute_grad
            };
            /* Use dummy pointers for src buffers when not computing gradient */
            const void *bufs[] = {
                p_sum, t_sum, p2_sum, t2_sum, tp_sum,
                ncc_buf,
                compute_grad ? (const void *)src_p  : (const void *)ncc_buf,
                compute_grad ? (const void *)src_p2 : (const void *)ncc_buf,
                compute_grad ? (const void *)src_tp : (const void *)ncc_buf
            };
            size_t sizes[] = {
                buf_bytes, buf_bytes, buf_bytes, buf_bytes, buf_bytes,
                buf_bytes, buf_bytes, buf_bytes, buf_bytes
            };
            metal_dispatch(pso, bufs, sizes, 9, &params, sizeof(params), n);
        }

        /* --- Step 2b: Reduce NCC to scalar loss --- */
        if (h_loss_out) {
            /* Use partial sum reduction via cc_ncc_partial_sum shader */
            uint32_t n_groups = metal_div_ceil(n, MTL_THREADGROUP_SIZE);
            size_t partial_bytes = (size_t)n_groups * sizeof(float);

            id<MTLBuffer> partial_buf = [g_metal.device newBufferWithLength:partial_bytes
                                                                    options:MTLResourceStorageModeShared];
            if (!partial_buf) {
                fprintf(stderr, "metal_cc_loss_3d_v2: failed to allocate partial sum buffer\n");
                goto cleanup_grad;
            }
            float *partial = (float *)[partial_buf contents];
            metal_register_buffer(partial, (__bridge void *)partial_buf, partial_bytes);

            {
                void *pso = metal_get_pipeline("cc_ncc_partial_sum");
                if (!pso) { fprintf(stderr, "metal_cc_loss_3d_v2: cc_ncc_partial_sum pipeline not found\n");
                    metal_unregister_buffer(partial); goto cleanup_grad; }
                struct { uint32_t n, _pad; } params = { .n = n, ._pad = 0 };
                const void *bufs[] = { ncc_buf, partial };
                size_t sizes[] = { buf_bytes, partial_bytes };
                metal_dispatch(pso, bufs, sizes, 2, &params, sizeof(params), n);
            }

            /* CPU final sum of partial results */
            metal_sync();
            double sum = 0.0;
            for (uint32_t i = 0; i < n_groups; i++) {
                sum += (double)partial[i];
            }
            *h_loss_out = -(float)(sum / (double)n);

            metal_unregister_buffer(partial);
        }

        /* --- Step 3: Compute gradient if requested --- */
        if (compute_grad) {
            /* Adjoint box-filter the 3 gradient source terms */
            /* Reuse p_sum, t_sum, p2_sum as adj_p, adj_p2, adj_tp */
            float *adj_p  = p_sum;   /* reuse */
            float *adj_p2 = t_sum;   /* reuse */
            float *adj_tp = p2_sum;  /* reuse */

            separable_box_filter_metal(src_p,  adj_p,  D, H, W, ks, tmp);
            separable_box_filter_metal(src_p2, adj_p2, D, H, W, ks, tmp);
            separable_box_filter_metal(src_tp, adj_tp, D, H, W, ks, tmp);

            /* Combine: grad = -inv_count * (adj_p + 2*P*adj_p2 + T*adj_tp) */
            {
                void *pso = metal_get_pipeline("cc_combine_grad");
                if (!pso) { fprintf(stderr, "metal_cc_loss_3d_v2: cc_combine_grad pipeline not found\n"); goto cleanup_grad; }
                struct { uint32_t n; float inv_count; } params = {
                    .n = n, .inv_count = 1.0f / (float)n
                };
                const void *bufs[] = { adj_p, adj_p2, adj_tp, pred, target, grad_pred };
                size_t sizes[] = { buf_bytes, buf_bytes, buf_bytes, buf_bytes, buf_bytes, buf_bytes };
                metal_dispatch(pso, bufs, sizes, 6, &params, sizeof(params), n);
            }
        }

    cleanup_grad:
        if (src_tp)  metal_unregister_buffer(src_tp);
        if (src_p2)  metal_unregister_buffer(src_p2);
        if (src_p)   metal_unregister_buffer(src_p);

    cleanup:
        metal_sync();
        metal_unregister_buffer(tmp);
        metal_unregister_buffer(work);
        metal_unregister_buffer(tp_sum);
        metal_unregister_buffer(t2_sum);
        metal_unregister_buffer(p2_sum);
        metal_unregister_buffer(t_sum);
        metal_unregister_buffer(p_sum);
    }
}

void metal_cc_loss_3d(const float *pred, const float *target,
                       float *grad_pred,
                       int D, int H, int W, int ks, float *h_loss_out) {
    metal_cc_loss_3d_v2(pred, target, grad_pred, D, H, W, ks, h_loss_out);
}

/* ================================================================== */
/* Phase 6: MI Loss — faithful translation of cuda_mi_loss_3d         */
/* ================================================================== */

/* Param structs matching mi_loss.metal shader layouts */
typedef struct {
    int32_t N;
    int32_t num_bins;
    float preterm;
    float inv_maxval_p;
    float inv_maxval_t;
    float _pad0;
    int32_t _pad1, _pad2;
} mi_hist_params_t;

typedef struct {
    int32_t num_bins;
    float nr, dr;
    int32_t _pad;
} mi_compute_params_t;

typedef struct {
    int32_t N;
    int32_t num_bins;
    float preterm;
    float inv_maxval_p;
    float inv_maxval_t;
    float nr, dr;
    int32_t _pad;
} mi_grad_params_t;

void metal_mi_loss_3d(const float *pred, const float *target,
                       float *grad_pred,
                       int D, int H, int W,
                       int num_bins, float *h_loss_out)
{
    @autoreleasepool {
        int N = D * H * W;
        int nb = num_bins;
        float nr = 1e-7f, dr = 1e-7f;

        if (nb > 64) {
            fprintf(stderr, "metal_mi_loss: num_bins=%d exceeds MAX_BINS=64\n", nb);
            if (h_loss_out) *h_loss_out = 0;
            return;
        }

        /* Find max values for normalization.
         * With unified memory, just sync and read directly — no download needed. */
        metal_sync();
        float pmax = pred[0], tmax = target[0];
        for (int i = 1; i < N; i++) {
            if (pred[i] > pmax) pmax = pred[i];
            if (target[i] > tmax) tmax = target[i];
        }

        float maxval = (pmax > tmax) ? pmax : tmax;
        float inv_maxval_p = (maxval > 1.0f) ? 1.0f / maxval : 1.0f;
        float inv_maxval_t = inv_maxval_p;

        /* Build bin centers (matching Python/CUDA) */
        float h_bins[64];
        for (int i = 0; i < nb; i++)
            h_bins[i] = (float)i / nb + 0.5f / nb;

        float bin_spacing = 1.0f / nb;
        float sigma = bin_spacing * 0.5f;
        float preterm = 1.0f / (2.0f * sigma * sigma);

        /* Allocate shared-memory GPU buffers for histogram and bin centers */
        size_t pab_sz = (size_t)nb * nb * sizeof(float);
        size_t marg_sz = (size_t)nb * sizeof(float);
        id<MTLBuffer> pab_buf = [g_metal.device newBufferWithLength:pab_sz
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> pa_buf  = [g_metal.device newBufferWithLength:marg_sz
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> pb_buf  = [g_metal.device newBufferWithLength:marg_sz
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bins_buf = [g_metal.device newBufferWithLength:marg_sz
                                                             options:MTLResourceStorageModeShared];
        if (!pab_buf || !pa_buf || !pb_buf || !bins_buf) {
            fprintf(stderr, "metal_mi_loss: buffer allocation failed\n");
            if (h_loss_out) *h_loss_out = 0;
            return;
        }
        float *d_pab  = (float *)pab_buf.contents;
        float *d_pa   = (float *)pa_buf.contents;
        float *d_pb   = (float *)pb_buf.contents;
        float *d_bins = (float *)bins_buf.contents;
        metal_register_buffer(d_pab, (__bridge void *)pab_buf, pab_sz);
        metal_register_buffer(d_pa, (__bridge void *)pa_buf, marg_sz);
        metal_register_buffer(d_pb, (__bridge void *)pb_buf, marg_sz);
        metal_register_buffer(d_bins, (__bridge void *)bins_buf, marg_sz);

        /* Initialize: zero histograms, copy bin centers */
        memset(d_pab, 0, (size_t)nb * nb * sizeof(float));
        memset(d_pa, 0, (size_t)nb * sizeof(float));
        memset(d_pb, 0, (size_t)nb * sizeof(float));
        memcpy(d_bins, h_bins, (size_t)nb * sizeof(float));

        /* Step 1: Accumulate histogram on GPU */
        {
            void *pso = metal_get_pipeline("mi_histogram");
            if (!pso) {
                fprintf(stderr, "metal_mi_loss: mi_histogram pipeline not found\n");
                if (h_loss_out) *h_loss_out = 0;
                goto cleanup;
            }

            mi_hist_params_t params = {
                .N = N, .num_bins = nb, .preterm = preterm,
                .inv_maxval_p = inv_maxval_p, .inv_maxval_t = inv_maxval_t,
                ._pad0 = 0, ._pad1 = 0, ._pad2 = 0
            };

            const void *bufs[] = { pred, target, d_pab, d_pa, d_pb, d_bins };
            size_t sizes[] = {
                (size_t)N * sizeof(float),
                (size_t)N * sizeof(float),
                (size_t)nb * nb * sizeof(float),
                (size_t)nb * sizeof(float),
                (size_t)nb * sizeof(float),
                (size_t)nb * sizeof(float)
            };
            metal_dispatch(pso, bufs, sizes, 6, &params, sizeof(params), (uint32_t)N);
        }

        /* Step 2: Compute MI on CPU from histogram (fast — only nb² iterations).
         * With unified memory, just sync and read the histogram directly. */
        if (h_loss_out) {
            metal_sync();
            double mi = 0.0;
            for (int a = 0; a < nb; a++) {
                for (int b = 0; b < nb; b++) {
                    float p = d_pab[a * nb + b];
                    float pp = d_pa[a] * d_pb[b];
                    mi += p * logf((p + nr) / (pp + dr) + dr);
                }
            }
            *h_loss_out = -(float)mi;
        }

        /* Step 3: Gradient on GPU */
        if (grad_pred) {
            void *pso = metal_get_pipeline("mi_gradient");
            if (!pso) {
                fprintf(stderr, "metal_mi_loss: mi_gradient pipeline not found\n");
                goto cleanup;
            }

            mi_grad_params_t params = {
                .N = N, .num_bins = nb, .preterm = preterm,
                .inv_maxval_p = inv_maxval_p, .inv_maxval_t = inv_maxval_t,
                .nr = nr, .dr = dr, ._pad = 0
            };

            const void *bufs[] = { pred, target, d_pab, d_pa, d_pb, grad_pred, d_bins };
            size_t sizes[] = {
                (size_t)N * sizeof(float),
                (size_t)N * sizeof(float),
                (size_t)nb * nb * sizeof(float),
                (size_t)nb * sizeof(float),
                (size_t)nb * sizeof(float),
                (size_t)N * sizeof(float),
                (size_t)nb * sizeof(float)
            };
            metal_dispatch(pso, bufs, sizes, 7, &params, sizeof(params), (uint32_t)N);
        }

    cleanup:
        metal_unregister_buffer(d_bins);
        metal_unregister_buffer(d_pb);
        metal_unregister_buffer(d_pa);
        metal_unregister_buffer(d_pab);
        /* ARC releases the MTLBuffers when they go out of scope */
    }
}

/* ================================================================== */
/* Phase 7: Deformable registration ops                                */
/* ================================================================== */

/* --- Vector add: out[i] = a[i] + b[i] --- */
void metal_vec_add(float *out, const float *a, const float *b, int n) {
    /* Use tensor_axpy pattern: copy a to out, then axpy with b */
    /* For simplicity and to avoid needing a new shader, do on CPU
       since unified memory makes this fast. */
    metal_sync();
    for (int i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

/* --- Permute dispatches --- */

typedef struct { uint32_t D, H, W, _pad; } permute_params_t;

void metal_permute_3dhw_dhw3(const float *in, float *out, int D, int H, int W) {
    void *pso = metal_get_pipeline("permute_3dhw_dhw3");
    if (!pso) {
        fprintf(stderr, "metal_permute_3dhw_dhw3: pipeline not found\n");
        return;
    }
    permute_params_t params = { .D = (uint32_t)D, .H = (uint32_t)H, .W = (uint32_t)W, ._pad = 0 };
    uint32_t spatial = (uint32_t)D * H * W;
    const void *bufs[] = { in, out };
    size_t sizes[] = { (size_t)spatial * 3 * sizeof(float), (size_t)spatial * 3 * sizeof(float) };
    metal_dispatch(pso, bufs, sizes, 2, &params, sizeof(params), spatial);
}

void metal_permute_dhw3_3dhw(const float *in, float *out, int D, int H, int W) {
    void *pso = metal_get_pipeline("permute_dhw3_3dhw");
    if (!pso) {
        fprintf(stderr, "metal_permute_dhw3_3dhw: pipeline not found\n");
        return;
    }
    permute_params_t params = { .D = (uint32_t)D, .H = (uint32_t)H, .W = (uint32_t)W, ._pad = 0 };
    uint32_t spatial = (uint32_t)D * H * W;
    const void *bufs[] = { in, out };
    size_t sizes[] = { (size_t)spatial * 3 * sizeof(float), (size_t)spatial * 3 * sizeof(float) };
    metal_dispatch(pso, bufs, sizes, 2, &params, sizeof(params), spatial);
}

/* --- Fused compositive warp update --- */

typedef struct { uint32_t D, H, W, _pad; } compose_params_t;

void metal_fused_compositive_update(const float *warp, const float *update,
                                     float *output, int D, int H, int W) {
    void *pso = metal_get_pipeline("fused_compositive_update");
    if (!pso) {
        fprintf(stderr, "metal_fused_compositive_update: pipeline not found\n");
        return;
    }
    compose_params_t params = { .D = (uint32_t)D, .H = (uint32_t)H, .W = (uint32_t)W, ._pad = 0 };
    uint32_t spatial = (uint32_t)D * H * W;
    const void *bufs[] = { warp, update, output };
    size_t sizes[] = {
        (size_t)spatial * 3 * sizeof(float),
        (size_t)spatial * 3 * sizeof(float),
        (size_t)spatial * 3 * sizeof(float)
    };
    metal_dispatch(pso, bufs, sizes, 3, &params, sizeof(params), spatial);
}

/* --- Max L2 norm reduction --- */

float metal_max_l2_norm(const float *data, int spatial) {
    float eps = 1e-8f;
    void *pso = metal_get_pipeline("max_l2_norm");
    if (!pso) {
        /* CPU fallback */
        metal_sync();
        float maxval = 0;
        for (int i = 0; i < spatial; i++) {
            float dx = data[i*3], dy = data[i*3+1], dz = data[i*3+2];
            float v = sqrtf(dx*dx + dy*dy + dz*dz);
            if (v > maxval) maxval = v;
        }
        return eps + maxval;
    }

    @autoreleasepool {
        uint32_t n_groups = metal_div_ceil((uint32_t)spatial, MTL_THREADGROUP_SIZE);

        id<MTLBuffer> partial_buf = [g_metal.device newBufferWithLength:(NSUInteger)(n_groups * sizeof(float))
                                                                options:MTLResourceStorageModeShared];
        if (!partial_buf) {
            /* CPU fallback */
            metal_sync();
            float maxval = 0;
            for (int i = 0; i < spatial; i++) {
                float dx = data[i*3], dy = data[i*3+1], dz = data[i*3+2];
                float v = sqrtf(dx*dx + dy*dy + dz*dz);
                if (v > maxval) maxval = v;
            }
            return eps + maxval;
        }

        float *d_partial = (float *)partial_buf.contents;
        metal_register_buffer(d_partial, (__bridge void *)partial_buf, (size_t)(n_groups * sizeof(float)));

        struct { uint32_t spatial, _pad; float eps, _pad1; } params = {
            .spatial = (uint32_t)spatial, ._pad = 0, .eps = eps, ._pad1 = 0
        };
        const void *bufs[] = { data, d_partial };
        size_t sizes[] = {
            (size_t)spatial * 3 * sizeof(float),
            (size_t)n_groups * sizeof(float)
        };
        metal_dispatch(pso, bufs, sizes, 2, &params, sizeof(params), (uint32_t)spatial);

        /* CPU final max of partial results */
        metal_sync();
        float maxval = 0;
        for (uint32_t i = 0; i < n_groups; i++)
            if (d_partial[i] > maxval) maxval = d_partial[i];

        metal_unregister_buffer(d_partial);
        return eps + maxval;
    }
}

/* --- Warp inversion via iterative fixed-point --- */

typedef struct { uint32_t D, H, W, _pad; } warp_params_t;

void metal_warp_inverse(const float *u, float *inv_u, int D, int H, int W, int n_iters) {
    if (n_iters <= 0) n_iters = 550;  /* default matching Python/CUDA */

    uint32_t spatial = (uint32_t)D * H * W;
    uint32_t n3 = spatial * 3;

    /* Initialize inv_u = -u */
    void *negate_pso = metal_get_pipeline("negate_field");
    if (!negate_pso) {
        /* CPU fallback */
        metal_sync();
        for (uint32_t i = 0; i < n3; i++)
            inv_u[i] = -u[i];
    } else {
        warp_params_t params = { .D = (uint32_t)D, .H = (uint32_t)H, .W = (uint32_t)W, ._pad = 0 };
        const void *bufs[] = { u, inv_u };
        size_t sizes[] = { (size_t)n3 * sizeof(float), (size_t)n3 * sizeof(float) };
        metal_dispatch(negate_pso, bufs, sizes, 2, &params, sizeof(params), n3);
    }

    void *iter_pso = metal_get_pipeline("warp_inverse_iter");
    if (!iter_pso) {
        fprintf(stderr, "metal_warp_inverse: warp_inverse_iter pipeline not found, using negate only\n");
        return;
    }

    @autoreleasepool {
        /* Need a temporary buffer for ping-pong iteration */
        id<MTLBuffer> tmp_buf = [g_metal.device newBufferWithLength:(NSUInteger)(n3 * sizeof(float))
                                                             options:MTLResourceStorageModeShared];
        if (!tmp_buf) {
            fprintf(stderr, "metal_warp_inverse: failed to allocate temp buffer\n");
            return;
        }
        float *d_tmp = (float *)tmp_buf.contents;
        metal_register_buffer(d_tmp, (__bridge void *)tmp_buf, (size_t)(n3 * sizeof(float)));

        warp_params_t params = { .D = (uint32_t)D, .H = (uint32_t)H, .W = (uint32_t)W, ._pad = 0 };

        for (int iter = 0; iter < n_iters; iter++) {
            /* out = -interp(u, identity + inv_u)
             * Alternate between inv_u->tmp and tmp->inv_u */
            float *src = (iter % 2 == 0) ? inv_u : d_tmp;
            float *dst = (iter % 2 == 0) ? d_tmp : inv_u;

            const void *bufs[] = { u, src, dst };
            size_t sizes[] = {
                (size_t)n3 * sizeof(float),
                (size_t)n3 * sizeof(float),
                (size_t)n3 * sizeof(float)
            };
            metal_dispatch(iter_pso, bufs, sizes, 3, &params, sizeof(params), spatial);
        }

        /* If n_iters is odd, result is in d_tmp; copy back to inv_u */
        if (n_iters % 2 != 0) {
            metal_sync();
            memcpy(inv_u, d_tmp, (size_t)n3 * sizeof(float));
        }

        metal_unregister_buffer(d_tmp);
    }
}
