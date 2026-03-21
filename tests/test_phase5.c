/*
 * test_phase5.c - Validate affine registration against Python
 */
#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/backend.h"
#include "cfireants/registration.h"
#include "cfireants/losses.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int load_bin_f32(const char *dir, const char *name, float *buf, size_t count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    size_t n = fread(buf, sizeof(float), count, f);
    fclose(f);
    return (n == count) ? 0 : -1;
}

static int test_affine(const char *data_dir, const char *dataset,
                       const char *fixed_path, const char *moving_path,
                       int loss_type_rigid, int loss_type_affine,
                       int cc_ks, int mi_bins, float lr_rigid, float lr_affine) {
    printf("\n=== Affine: %s ===\n", dataset);
    int failures = 0;

    image_t fixed, moving;
    if (image_load(&fixed, fixed_path, DEVICE_CPU) != 0) return 1;
    if (image_load(&moving, moving_path, DEVICE_CPU) != 0) return 1;

    /* Moments */
    moments_opts_t mom_opts = moments_opts_default();
    moments_result_t mom_result;
    moments_register(&fixed, &moving, mom_opts, &mom_result);

    /* Rigid */
    int scales3[] = {4, 2, 1};
    int iters3[] = {200, 100, 50};
    rigid_opts_t ropts = {
        .n_scales = 3, .scales = scales3, .iterations = iters3,
        .loss_type = loss_type_rigid, .cc_kernel_size = cc_ks,
        .mi_num_bins = mi_bins, .lr = lr_rigid,
        .tolerance = 1e-6f, .max_tolerance_iters = 10
    };
    rigid_result_t rigid_result;
    rigid_register(&fixed, &moving, &mom_result, ropts, &rigid_result);
    printf("  Rigid NCC: %.6f\n", rigid_result.ncc_loss);

    /* Affine */
    affine_opts_t aopts = {
        .n_scales = 3, .scales = scales3, .iterations = iters3,
        .loss_type = loss_type_affine, .cc_kernel_size = cc_ks,
        .mi_num_bins = mi_bins, .lr = lr_affine,
        .tolerance = 1e-6f, .max_tolerance_iters = 10
    };
    affine_result_t affine_result;
    affine_register(&fixed, &moving, rigid_result.rigid_mat, aopts, &affine_result);

    /* Load Python reference */
    char name[128];
    float py_ncc;
    snprintf(name, sizeof(name), "affine_%s_ncc", dataset);
    load_bin_f32(data_dir, name, &py_ncc, 1);

    printf("  C NCC:      %.6f\n", affine_result.ncc_loss);
    printf("  Python NCC: %.6f\n", py_ncc);

    float ncc_diff = fabsf(affine_result.ncc_loss - py_ncc);
    int ncc_pass = (ncc_diff < 0.05f);
    printf("  NCC diff: %.6f  %s\n", ncc_diff, ncc_pass ? "PASS" : "FAIL");
    if (!ncc_pass) failures++;

    /* Print affine matrix */
    printf("  C affine [3,4]:\n");
    for (int i = 0; i < 3; i++)
        printf("    [%10.6f %10.6f %10.6f %10.6f]\n",
               affine_result.affine_mat[i][0], affine_result.affine_mat[i][1],
               affine_result.affine_mat[i][2], affine_result.affine_mat[i][3]);

    image_free(&fixed);
    image_free(&moving);
    return failures;
}

int main(int argc, char **argv) {
    const char *data_dir = "cfireants/tests/test_data";
    if (argc > 1) data_dir = argv[1];
    cfireants_init_cpu();

    int failures = 0;
    failures += test_affine(data_dir, "small",
        "validate/small/MNI152_T1_2mm.nii.gz",
        "validate/small/T1_head_2mm.nii.gz",
        LOSS_MI, LOSS_MI, 5, 32, 3e-3f, 1e-3f);
    failures += test_affine(data_dir, "medium",
        "validate/medium/MNI152_T1_1mm_brain.nii.gz",
        "validate/medium/t1_brain.nii.gz",
        LOSS_CC, LOSS_CC, 5, 0, 3e-3f, 1e-3f);

    printf("\n========================================\n");
    if (failures == 0)
        printf("All Phase 5 tests PASSED\n");
    else
        printf("%d test(s) FAILED\n", failures);
    printf("========================================\n");

    cfireants_cleanup();
    return failures;
}
