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
 * axes W, H, D in order (matching CUDA).  Uses scratch as temporary.
 */
static void box_filter_intermediates_metal(float *interm, int spatial,
                                            int D, int H, int W, int ks,
                                            float *scratch) {
    float scale = 1.0f / (float)ks;
    for (int ch = 0; ch < 5; ch++) {
        float *chan = interm + (size_t)ch * spatial;
        /* W axis: chan -> scratch */
        metal_box_filter_axis(chan, scratch, D, H, W, ks, 2, scale);
        /* H axis: scratch -> chan */
        metal_box_filter_axis(scratch, chan, D, H, W, ks, 1, scale);
        /* D axis: chan -> scratch */
        metal_box_filter_axis(chan, scratch, D, H, W, ks, 0, scale);
        /* Copy scratch back to chan (unified memory) */
        metal_sync();
        memcpy(chan, scratch, (size_t)spatial * sizeof(float));
    }
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

void metal_cc_loss_3d(const float *pred, const float *target,
                       float *grad_pred,
                       int D, int H, int W, int ks, float *h_loss_out) {
    @autoreleasepool {
        uint32_t spatial = (uint32_t)(D * H * W);
        size_t interm_bytes = (size_t)5 * spatial * sizeof(float);
        size_t scratch_bytes = (size_t)spatial * sizeof(float);

        /* Allocate workspace as shared MTLBuffers */
        id<MTLBuffer> interm_buf = [g_metal.device newBufferWithLength:interm_bytes
                                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> scratch_buf = [g_metal.device newBufferWithLength:scratch_bytes
                                                                options:MTLResourceStorageModeShared];
        if (!interm_buf || !scratch_buf) {
            fprintf(stderr, "metal_cc_loss_3d: failed to allocate workspace\n");
            return;
        }

        float *interm = (float *)[interm_buf contents];
        float *scratch = (float *)[scratch_buf contents];

        metal_register_buffer(interm, (__bridge void *)interm_buf, interm_bytes);
        metal_register_buffer(scratch, (__bridge void *)scratch_buf, scratch_bytes);

        metal_fused_cc_loss(pred, target, grad_pred, NULL,
                            D, H, W, ks, h_loss_out, interm, scratch);

        metal_sync();

        metal_unregister_buffer(scratch);
        metal_unregister_buffer(interm);
    }
}
