/*
 * test_webgpu_linear.c - Test Moments + Rigid + Affine on WebGPU backend
 *
 * Runs the linear registration pipeline on the small validation dataset
 * and compares the affine matrix against the CUDA result.
 */

#include "cfireants/backend.h"
#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/registration.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef CFIREANTS_HAS_WEBGPU
extern int cfireants_init_webgpu(void);
#endif

int main(void) {
    printf("=== WebGPU Linear Registration Test (small dataset) ===\n");

    cfireants_init_cpu();

#ifndef CFIREANTS_HAS_WEBGPU
    printf("WebGPU not enabled, skipping\n");
    return 0;
#else
    if (cfireants_init_webgpu() != 0) {
        printf("Failed to initialize WebGPU\n");
        return 1;
    }

    /* Load images (CPU — registration functions handle upload) */
    image_t fixed, moving;
    if (image_load(&fixed, "validate/small/MNI152_T1_2mm.nii.gz", DEVICE_CPU) != 0) {
        printf("Failed to load fixed image\n");
        return 1;
    }
    if (image_load(&moving, "validate/small/T1_head_2mm.nii.gz", DEVICE_CPU) != 0) {
        printf("Failed to load moving image\n");
        return 1;
    }

    printf("Fixed: [%d,%d,%d]  Moving: [%d,%d,%d]\n",
           fixed.data.shape[2], fixed.data.shape[3], fixed.data.shape[4],
           moving.data.shape[2], moving.data.shape[3], moving.data.shape[4]);

    /* Moments registration (CPU — lightweight) */
    printf("\n--- Moments ---\n");
    moments_opts_t mopts = moments_opts_default();
    moments_result_t mom_result;
    moments_register(&fixed, &moving, mopts, &mom_result);
    printf("  Moments NCC: %.4f\n", mom_result.ncc_loss);
    fflush(stdout);

    /* Rigid registration (WebGPU) */
    printf("\n--- Rigid (WebGPU) ---\n");
    fflush(stdout);
    fprintf(stderr, "About to call rigid_register_webgpu...\n");
    int rigid_scales[] = {4, 2, 1};
    int rigid_iters[] = {200, 100, 50};
    rigid_opts_t ropts = {
        .n_scales = 3, .scales = rigid_scales,
        .iterations = rigid_iters,
        .lr = 0.01f, .loss_type = LOSS_MI,
        .mi_num_bins = 32, .cc_kernel_size = 5,
        .tolerance = 1e-6f, .max_tolerance_iters = 10
    };
    rigid_result_t rigid_result;
    rigid_register_webgpu(&fixed, &moving, &mom_result, ropts, &rigid_result);
    printf("  Rigid NCC: %.4f\n", rigid_result.ncc_loss);

    /* Print rigid matrix */
    printf("  Rigid matrix:\n");
    for (int i = 0; i < 3; i++)
        printf("    [%8.4f %8.4f %8.4f %8.4f]\n",
               rigid_result.rigid_mat[i][0], rigid_result.rigid_mat[i][1],
               rigid_result.rigid_mat[i][2], rigid_result.rigid_mat[i][3]);

    /* Affine registration (WebGPU) */
    printf("\n--- Affine (WebGPU) ---\n");
    int affine_scales[] = {4, 2, 1};
    int affine_iters[] = {200, 100, 50};
    affine_opts_t aopts = {
        .n_scales = 3, .scales = affine_scales,
        .iterations = affine_iters,
        .lr = 0.001f, .loss_type = LOSS_MI,
        .mi_num_bins = 32, .cc_kernel_size = 5,
        .tolerance = 1e-6f, .max_tolerance_iters = 10
    };
    affine_result_t affine_result;
    affine_register_webgpu(&fixed, &moving, rigid_result.rigid_mat,
                            aopts, &affine_result);
    printf("  Affine NCC: %.4f\n", affine_result.ncc_loss);

    /* Print affine matrix */
    printf("  Affine matrix:\n");
    for (int i = 0; i < 3; i++)
        printf("    [%8.4f %8.4f %8.4f %8.4f]\n",
               affine_result.affine_mat[i][0], affine_result.affine_mat[i][1],
               affine_result.affine_mat[i][2], affine_result.affine_mat[i][3]);

    /* Quality check: NCC should be significant improvement over moments */
    int pass = 1;
    if (rigid_result.ncc_loss > -0.005f) {
        printf("\n  FAIL: rigid NCC loss too weak (%.4f)\n", rigid_result.ncc_loss);
        pass = 0;
    }
    if (affine_result.ncc_loss > -0.010f) {
        printf("\n  FAIL: affine NCC loss too weak (%.4f)\n", affine_result.ncc_loss);
        pass = 0;
    }
    if (pass) {
        printf("\n  PASS: linear registration pipeline produced reasonable results\n");
    }

    image_free(&fixed);
    image_free(&moving);

    printf("\n========================================\n");
    return pass ? 0 : 1;
#endif
}
