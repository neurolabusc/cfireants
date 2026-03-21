/*
 * test_metal_linear.c - Test rigid + affine registration on Metal backend
 */

#include "cfireants/backend.h"
#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/registration.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef CFIREANTS_HAS_METAL
extern int cfireants_init_metal(void);
#endif

int main(void) {
    printf("=== Metal Linear Registration Test (small dataset) ===\n");
    cfireants_init_cpu();

#ifndef CFIREANTS_HAS_METAL
    printf("Metal not enabled, skipping\n");
    return 0;
#else
    if (cfireants_init_metal() != 0) {
        printf("Failed to initialize Metal\n");
        return 1;
    }

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

    /* Moments */
    printf("\n--- Moments ---\n");
    moments_opts_t mopts = moments_opts_default();
    moments_result_t mom;
    moments_register(&fixed, &moving, mopts, &mom);
    printf("  Moments NCC: %.4f\n", mom.ncc_loss);

    /* Rigid */
    printf("\n--- Rigid (Metal) ---\n");
    int rigid_scales[] = {4, 2, 1};
    int rigid_iters[] = {200, 100, 50};
    rigid_opts_t ropts = {
        .n_scales = 3, .scales = rigid_scales,
        .iterations = rigid_iters,
        .lr = 0.005f, .loss_type = LOSS_CC,
        .cc_kernel_size = 5,
        .tolerance = 1e-6f, .max_tolerance_iters = 10
    };
    rigid_result_t rigid;
    rigid_register_metal(&fixed, &moving, &mom, ropts, &rigid);
    printf("  Rigid NCC: %.4f\n", rigid.ncc_loss);

    /* Affine */
    printf("\n--- Affine (Metal) ---\n");
    int affine_scales[] = {4, 2, 1};
    int affine_iters[] = {200, 100, 50};
    affine_opts_t aopts = {
        .n_scales = 3, .scales = affine_scales,
        .iterations = affine_iters,
        .lr = 0.001f, .loss_type = LOSS_CC,
        .cc_kernel_size = 5,
        .tolerance = 1e-6f, .max_tolerance_iters = 10
    };
    affine_result_t affine;
    affine_register_metal(&fixed, &moving, rigid.rigid_mat, aopts, &affine);
    printf("  Affine NCC: %.4f\n", affine.ncc_loss);

    int pass = (rigid.ncc_loss < -0.005f && affine.ncc_loss < -0.010f);
    printf("\n  %s: Metal linear registration\n", pass ? "PASS" : "FAIL");

    image_free(&fixed);
    image_free(&moving);
    return pass ? 0 : 1;
#endif
}
