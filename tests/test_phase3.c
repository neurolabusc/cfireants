/*
 * test_phase3.c - Validate moments registration against Python
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

/* Helper for det */
static float mat3_det_ext(const float m[3][3]) {
    return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
         - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
         + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
}

static int load_bin_f32(const char *dir, const char *name,
                        float *buf, size_t count) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }
    size_t n = fread(buf, sizeof(float), count, f);
    fclose(f);
    return (n == count) ? 0 : -1;
}

static int test_moments(const char *data_dir,
                        const char *dataset,
                        const char *fixed_path,
                        const char *moving_path) {
    printf("\n=== Moments: %s ===\n", dataset);
    int failures = 0;

    /* Load images */
    image_t fixed, moving;
    if (image_load(&fixed, fixed_path, DEVICE_CPU) != 0) return 1;
    if (image_load(&moving, moving_path, DEVICE_CPU) != 0) return 1;

    /* Run moments registration */
    moments_opts_t opts = moments_opts_default();
    moments_result_t result;
    moments_register(&fixed, &moving, opts, &result);

    /* Load Python reference */
    char name[128];
    float py_Rf[9], py_tf[3], py_aff[12];

    snprintf(name, sizeof(name), "mom_%s_Rf", dataset);
    load_bin_f32(data_dir, name, py_Rf, 9);

    snprintf(name, sizeof(name), "mom_%s_tf", dataset);
    load_bin_f32(data_dir, name, py_tf, 3);

    snprintf(name, sizeof(name), "mom_%s_affine_init", dataset);
    load_bin_f32(data_dir, name, py_aff, 12);

    /* Compare rotation matrices */
    printf("  Rotation matrix comparison:\n");
    printf("  C Rf:     Python Rf:\n");
    float max_rf_err = 0;
    for (int i = 0; i < 3; i++) {
        printf("  [%8.5f %8.5f %8.5f]  [%8.5f %8.5f %8.5f]\n",
               result.Rf[i][0], result.Rf[i][1], result.Rf[i][2],
               py_Rf[i*3+0], py_Rf[i*3+1], py_Rf[i*3+2]);
        for (int j = 0; j < 3; j++) {
            float err = fabsf(result.Rf[i][j] - py_Rf[i*3+j]);
            if (err > max_rf_err) max_rf_err = err;
        }
    }

    /* Compare translations */
    float max_tf_err = 0;
    printf("  C tf:  [%.4f, %.4f, %.4f]\n", result.tf[0], result.tf[1], result.tf[2]);
    printf("  Py tf: [%.4f, %.4f, %.4f]\n", py_tf[0], py_tf[1], py_tf[2]);
    for (int i = 0; i < 3; i++) {
        float err = fabsf(result.tf[i] - py_tf[i]);
        if (err > max_tf_err) max_tf_err = err;
    }

    /* Check det(Rf) */
    float det = mat3_det_ext(result.Rf);
    printf("  det(Rf) = %.6f\n", det);

    /* Evaluate: warp moving to fixed space and compute NCC */
    tensor_t moved;
    apply_affine_transform(&fixed, &moving, result.affine, &moved);

    float ncc;
    cpu_cc_loss_3d(&moved, &fixed.data, 9, &ncc, NULL);
    printf("  NCC loss (k=9): %.6f\n", ncc);

    /* Load Python NCC for comparison */
    float py_ncc;
    snprintf(name, sizeof(name), "mom_%s_ncc", dataset);
    load_bin_f32(data_dir, name, &py_ncc, 1);
    printf("  Python NCC:     %.6f\n", py_ncc);

    /* Tolerances - moments can differ slightly due to SVD ordering/sign */
    int rf_pass = (max_rf_err < 0.05f);
    int tf_pass = (max_tf_err < 5.0f);  /* mm */
    int ncc_pass = (fabsf(ncc - py_ncc) < 0.05f);  /* 5% NCC tolerance */

    printf("  Rf max err: %.6f  %s\n", max_rf_err, rf_pass ? "PASS" : "FAIL");
    printf("  tf max err: %.4f mm  %s\n", max_tf_err, tf_pass ? "PASS" : "FAIL");
    printf("  NCC diff:   %.6f  %s\n", fabsf(ncc - py_ncc), ncc_pass ? "PASS" : "FAIL");

    if (!rf_pass) failures++;
    if (!tf_pass) failures++;
    if (!ncc_pass) failures++;

    tensor_free(&moved);
    image_free(&fixed);
    image_free(&moving);
    return failures;
}

int main(int argc, char **argv) {
    const char *data_dir = "cfireants/tests/test_data";
    if (argc > 1) data_dir = argv[1];

    cfireants_init_cpu();

    int failures = 0;
    failures += test_moments(data_dir, "small",
                            "validate/small/MNI152_T1_2mm.nii.gz",
                            "validate/small/T1_head_2mm.nii.gz");
    failures += test_moments(data_dir, "medium",
                            "validate/medium/MNI152_T1_1mm_brain.nii.gz",
                            "validate/medium/t1_brain.nii.gz");

    printf("\n========================================\n");
    if (failures == 0)
        printf("All Phase 3 tests PASSED\n");
    else
        printf("%d test(s) FAILED\n", failures);
    printf("========================================\n");

    cfireants_cleanup();
    return failures;
}
