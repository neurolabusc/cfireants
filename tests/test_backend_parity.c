/*
 * test_backend_parity.c — CPU/GPU numerical-equivalence regression test.
 *
 * Runs Moments -> Rigid -> Affine -> Greedy, plus an independently seeded SyN
 * stage, on CPU and every compiled GPU backend and asserts they agree.
 *
 * Each stage is seeded from the CPU result of the previous stage, not from the
 * backend's own. Without this the comparison is worthless: a 1e-3 difference in
 * the rigid matrix moves the affine start point, which moves the greedy start
 * point, and the final warped volumes disagree by several percent for reasons
 * that have nothing to do with the stage under test. Seeding each stage
 * identically makes every number below attributable to one stage.
 *
 * Asserts:
 *   1. each backend's rigid / affine matrix matches CPU's within tolerance,
 *   2. each backend's warped volume correlates with every other's above
 *      PARITY_MIN_CORR,
 *   3. each backend's warped volume correlates with the FIXED image above
 *      FIXED_MIN_CORR, so a mutually-consistent-but-wrong set still fails,
 *   4. each backend's final Greedy local NCC and displacement field match CPU,
 *   5. SyN moved images and both displacement fields match CPU.
 *
 * Dataset: medium (1mm brain-extracted), production MI for Rigid/Affine and CC
 * for deformable stages. The small dataset is not useful here: its tiny MI
 * histogram makes backend noise large relative to the transform signal.
 *
 * Iterations are cut to 60x30x15 (from the production 200x100x50) but all
 * three pyramid levels are kept: several past bugs only appeared at a scale
 * transition. 60x30x15 is not arbitrary — at 20x10x5 the deformation is so
 * small that the warped volume is dominated by the affine, and a deliberately
 * broken deformable stage still passed. 60x30x15 is the shortest schedule
 * measured to catch it. Runtime is about 41s wall on an M4 Pro, including the
 * shorter 20x10x5 SyN comparison.
 *
 * Progress chatter from the backends goes to stderr; this test's report goes to
 * stdout, so `test_backend_parity 2>/dev/null` gives a clean run.
 *
 * Exit code 0 = pass, 1 = fail. Backends not compiled in are skipped, not failed.
 *
 * Usage: build/test_backend_parity        (run from repo root)
 */

#include "cfireants/backend.h"
#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/registration.h"
#include "cfireants/losses.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Baseline measured 2026-09-09, Apple M-series, macOS 26.x, Metal + WebGPU
 * build, medium dataset, 60x30x15 iters, common trilinear downsampling, every stage
 * CPU-seeded.
 *
 *   rigid  vs CPU:  Metal  lin 0.006864  trans 0.1528mm
 *                   WebGPU lin 0.007088  trans 0.1562mm
 *   affine vs CPU:  Metal  lin 0.000158  trans 0.0027mm
 *                   WebGPU lin 0.000728  trans 0.0132mm
 *   warped correlation:  all pairs >= 0.99999
 *   warped vs fixed:     all backends 0.9459
 *   Greedy local NCC: CPU -0.846080, Metal -0.846059, WebGPU -0.846089
 *   Greedy displacement: correlations >= 0.99998
 *   SyN moved/fwd/rev: correlations >= 0.99969, loss diff <= 0.000118
 *
 * Three things the baseline records rather than hides:
 *
 *  - The old Metal Greedy path used fused CC while CPU and WebGPU used regular
 *    CC. Its local NCC was -0.812928, 0.035 from CPU. Using regular CC closes
 *    the shared-trilinear gap to 0.000005, so the stage metric is asserted.
 *
 * Thresholds sit just below measured, with enough headroom to survive a
 * different GPU or driver but not enough to survive a real regression. */
#define LINEAR_MAX_DIFF    0.025f   /* measured max about 0.008 */
#define TRANS_MAX_DIFF_MM  0.5f     /* measured max 0.1562 mm */
#define PARITY_MIN_CORR    0.995f   /* measured min 0.99999 */
#define FIXED_MIN_CORR     0.920f   /* measured min 0.9459 (CPU) */
#define DEFORM_MAX_NCC_DIFF 0.0005f /* measured max 0.000021 */
#define DISP_MIN_CORR      0.99f
#define SYN_MIN_CORR       0.98f
#define SYN_MAX_NCC_DIFF   0.005f

#define MAX_BACKENDS 3

typedef enum { BK_CPU, BK_METAL, BK_WEBGPU } backend_id_t;

static const int SCALES[3] = {4, 2, 1};
static const int ITERS[3]  = {60, 30, 15};

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float global_corr(const float *a, const float *b, size_t n) {
    double sa = 0, sb = 0;
    for (size_t i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
    double ma = sa / n, mb = sb / n;
    double cov = 0, va = 0, vb = 0;
    for (size_t i = 0; i < n; i++) {
        double da = a[i] - ma, db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    double d = sqrt(va * vb);
    return (d < 1e-12) ? 0.0f : (float)(cov / d);
}

static float rmse(const float *a, const float *b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) { double d = a[i] - b[i]; s += d * d; }
    return (float)sqrt(s / n);
}

/* Linear part (columns 0-2, unitless) and translation (column 3, mm) have
 * different scales, so they get separate tolerances. */
static void mat34_max_diff(const float a[3][4], const float b[3][4],
                           float *lin, float *trans) {
    *lin = 0; *trans = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++) {
            float d = fabsf(a[i][j] - b[i][j]);
            float *m = (j == 3) ? trans : lin;
            if (d > *m) *m = d;
        }
}

static int run_syn_stage(backend_id_t bk, const image_t *fixed,
                         const image_t *moving, const float affine[4][4],
                         syn_result_t *result) {
    static const int syn_iters[3] = {20, 10, 5};
    syn_opts_t opts = {
        .n_scales = 3, .scales = (int *)SCALES,
        .iterations = (int *)syn_iters,
        .cc_kernel_size = 5, .lr = 0.1f,
        .smooth_warp_sigma = 0.5f, .smooth_grad_sigma = 1.0f,
        .tolerance = 1e-6f, .max_tolerance_iters = 10,
        .downsample_mode = DOWNSAMPLE_TRILINEAR,
    };
    switch (bk) {
    case BK_CPU:
        return syn_register(fixed, moving, affine, opts, result);
#ifdef CFIREANTS_HAS_METAL
    case BK_METAL:
        return syn_register_metal(fixed, moving, affine, opts, result);
#endif
#ifdef CFIREANTS_HAS_WEBGPU
    case BK_WEBGPU:
        return syn_register_webgpu(fixed, moving, affine, opts, result);
#endif
    default:
        return -1;
    }
}

static int check_syn_parity(const char *const *names, const backend_id_t *ids,
                            int n) {
    image_t fixed = {0}, moving = {0};
    if (image_load(&fixed, "validate/small/MNI152_T1_2mm.nii.gz", DEVICE_CPU) != 0 ||
        image_load(&moving, "validate/small/T1_head_2mm.nii.gz", DEVICE_CPU) != 0) {
        fprintf(stderr, "FAIL: cannot load validate/small for SyN parity\n");
        image_free(&fixed);
        image_free(&moving);
        return 1;
    }

    moments_result_t moments;
    if (moments_register(&fixed, &moving, moments_opts_default(), &moments) != 0) {
        image_free(&fixed);
        image_free(&moving);
        return 1;
    }
    float affine[4][4] = {{0}};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++) affine[r][c] = moments.affine[r][c];
    affine[3][3] = 1.0f;

    syn_result_t result[MAX_BACKENDS];
    memset(result, 0, sizeof(result));
    int fail = 0;
    printf("\n  SyN stage agreement (small dataset, 20x10x5):\n");
    for (int i = 0; i < n; i++) {
        double t0 = get_time();
        int rc = run_syn_stage(ids[i], &fixed, &moving, affine, &result[i]);
        if (rc != 0 || !result[i].moved.data ||
            !result[i].fwd_disp.data || !result[i].rev_disp.data) {
            printf("    %-7s returned incomplete result   FAIL\n", names[i]);
            fail = 1;
            continue;
        }
        printf("    %-7s loss %.6f  (%.1fs)\n",
               names[i], result[i].ncc_loss, get_time() - t0);
    }

    if (!fail) {
        for (int i = 1; i < n; i++) {
            float moved_corr = global_corr(tensor_data_f32(&result[0].moved),
                                           tensor_data_f32(&result[i].moved),
                                           result[0].moved.numel);
            float fwd_corr = global_corr(tensor_data_f32(&result[0].fwd_disp),
                                         tensor_data_f32(&result[i].fwd_disp),
                                         result[0].fwd_disp.numel);
            float rev_corr = global_corr(tensor_data_f32(&result[0].rev_disp),
                                         tensor_data_f32(&result[i].rev_disp),
                                         result[0].rev_disp.numel);
            float loss_diff = fabsf(result[i].ncc_loss - result[0].ncc_loss);
            int bad = moved_corr < SYN_MIN_CORR || fwd_corr < SYN_MIN_CORR ||
                      rev_corr < SYN_MIN_CORR || loss_diff > SYN_MAX_NCC_DIFF;
            printf("    CPU vs %-7s moved %.5f  fwd %.5f  rev %.5f  "
                   "loss diff %.6f  %s\n",
                   names[i], moved_corr, fwd_corr, rev_corr, loss_diff,
                   bad ? "FAIL" : "OK");
            if (bad) fail = 1;
        }
    }

    for (int i = 0; i < n; i++) {
        tensor_free(&result[i].moved);
        tensor_free(&result[i].fwd_disp);
        tensor_free(&result[i].rev_disp);
    }
    image_free(&fixed);
    image_free(&moving);
    return fail;
}

/* Run one backend's pipeline. `seed_rigid` / `seed_affine44` are the CPU
 * results (NULL for the CPU run itself, which produces them). */
static int run_pipeline(backend_id_t bk, const image_t *fixed, const image_t *moving,
                        const moments_result_t *mom,
                        const float (*seed_rigid)[4], const float (*seed_affine44)[4],
                        rigid_result_t *rigid, affine_result_t *affine,
                        tensor_t *moved, tensor_t *disp, float *ncc_loss) {
    rigid_opts_t ropts = {
        .n_scales = 3, .scales = (int *)SCALES, .iterations = (int *)ITERS,
        .lr = 3e-3f, .loss_type = LOSS_MI, .mi_num_bins = 32, .cc_kernel_size = 5,
        .tolerance = 1e-6f, .max_tolerance_iters = 10,
        .downsample_mode = DOWNSAMPLE_TRILINEAR };
    affine_opts_t aopts = {
        .n_scales = 3, .scales = (int *)SCALES, .iterations = (int *)ITERS,
        .lr = 1e-3f, .loss_type = LOSS_MI, .mi_num_bins = 32, .cc_kernel_size = 5,
        .tolerance = 1e-6f, .max_tolerance_iters = 10,
        .downsample_mode = DOWNSAMPLE_TRILINEAR };
    greedy_opts_t gopts = {
        .n_scales = 3, .scales = (int *)SCALES, .iterations = (int *)ITERS,
        .cc_kernel_size = 5, .lr = 0.1f,
        .smooth_warp_sigma = 0.5f, .smooth_grad_sigma = 1.0f,
        .tolerance = 1e-6f, .max_tolerance_iters = 10,
        .downsample_mode = DOWNSAMPLE_TRILINEAR };

    /* Backends own result initialization. Poison it first so the test catches
     * any path that leaves an unsupported output partially uninitialized. */
    greedy_result_t greedy;
    memset(&greedy, 0xa5, sizeof(greedy));
    int rc;

    switch (bk) {
    case BK_CPU:
        rc = rigid_register(fixed, moving, mom, ropts, rigid);
        break;
#ifdef CFIREANTS_HAS_METAL
    case BK_METAL:
        rc = rigid_register_metal(fixed, moving, mom, ropts, rigid);
        break;
#endif
#ifdef CFIREANTS_HAS_WEBGPU
    case BK_WEBGPU:
        rc = rigid_register_webgpu(fixed, moving, mom, ropts, rigid);
        break;
#endif
    default: return -1;
    }
    if (rc) return rc;

    float rigid_in[3][4];
    if (seed_rigid) memcpy(rigid_in, seed_rigid, sizeof(rigid_in));
    else memcpy(rigid_in, rigid->rigid_mat, sizeof(rigid_in));

    switch (bk) {
    case BK_CPU:
        rc = affine_register(fixed, moving, rigid_in, aopts, affine);
        break;
#ifdef CFIREANTS_HAS_METAL
    case BK_METAL:
        rc = affine_register_metal(fixed, moving, rigid_in, aopts, affine);
        break;
#endif
#ifdef CFIREANTS_HAS_WEBGPU
    case BK_WEBGPU:
        rc = affine_register_webgpu(fixed, moving, rigid_in, aopts, affine);
        break;
#endif
    default: return -1;
    }
    if (rc) return rc;

    float aff44[4][4] = {{0}};
    if (seed_affine44) {
        memcpy(aff44, seed_affine44, sizeof(aff44));
    } else {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++) aff44[i][j] = affine->affine_mat[i][j];
        aff44[3][3] = 1.0f;
    }

    switch (bk) {
    case BK_CPU:    rc = greedy_register(fixed, moving, aff44, gopts, &greedy); break;
#ifdef CFIREANTS_HAS_METAL
    case BK_METAL:  rc = greedy_register_metal(fixed, moving, aff44, gopts, &greedy); break;
#endif
#ifdef CFIREANTS_HAS_WEBGPU
    case BK_WEBGPU: rc = greedy_register_webgpu(fixed, moving, aff44, gopts, &greedy); break;
#endif
    default: return -1;
    }
    if (rc) return rc;

    *moved = greedy.moved;
    *disp = greedy.disp;
    *ncc_loss = greedy.ncc_loss;
    if (!greedy_result_has_displacement(&greedy)) {
        fprintf(stderr, "Greedy backend %d did not return its displacement field\n", bk);
        tensor_free(moved);
        return -1;
    }
    return 0;
}

int main(void) {
    cfireants_verbose = 0;
    cfireants_init_cpu();

    const char *names[MAX_BACKENDS];
    backend_id_t ids[MAX_BACKENDS];
    int n = 0;
    names[n] = "CPU"; ids[n++] = BK_CPU;   /* must be index 0: it seeds the rest */

#ifdef CFIREANTS_HAS_METAL
    if (cfireants_init_metal() == 0) { names[n] = "Metal"; ids[n++] = BK_METAL; }
    else printf("SKIP: Metal compiled in but device init failed\n");
#else
    printf("SKIP: Metal not compiled in\n");
#endif
#ifdef CFIREANTS_HAS_WEBGPU
    if (cfireants_init_webgpu() == 0) { names[n] = "WebGPU"; ids[n++] = BK_WEBGPU; }
    else printf("SKIP: WebGPU compiled in but device init failed\n");
#else
    printf("SKIP: WebGPU not compiled in\n");
#endif

    if (n < 2) {
        printf("Only %d backend available — nothing to compare. SKIP.\n", n);
        return 77; /* ctest SKIP_RETURN_CODE */
    }

    image_t fixed, moving;
    if (image_load(&fixed, "validate/medium/MNI152_T1_1mm_brain.nii.gz", DEVICE_CPU) != 0 ||
        image_load(&moving, "validate/medium/t1_brain.nii.gz", DEVICE_CPU) != 0) {
        printf("FAIL: cannot load validate/medium — run from repo root\n");
        return 1;
    }
    size_t fN = (size_t)fixed.data.shape[2] * fixed.data.shape[3] * fixed.data.shape[4];
    const float *fdata = tensor_data_f32(&fixed.data);

    moments_result_t mom;
    moments_register(&fixed, &moving, moments_opts_default(), &mom);

    rigid_result_t rigid[MAX_BACKENDS];
    affine_result_t affine[MAX_BACKENDS];
    tensor_t moved[MAX_BACKENDS];
    tensor_t disp[MAX_BACKENDS];
    float ncc_loss[MAX_BACKENDS];
    float cpu_aff44[4][4] = {{0}};
    int fail = 0;

    printf("Backend parity: medium dataset, MI linear + CC deformable, trilinear, %dx%dx%d iters, 3 scales\n",
           ITERS[0], ITERS[1], ITERS[2]);
    printf("Every stage seeded from the CPU result, so each number is per-stage.\n\n");

    for (int i = 0; i < n; i++) {
        double t0 = get_time();
        int rc = run_pipeline(ids[i], &fixed, &moving, &mom,
                              i ? (const float (*)[4])rigid[0].rigid_mat : NULL,
                              i ? (const float (*)[4])cpu_aff44 : NULL,
                              &rigid[i], &affine[i], &moved[i], &disp[i], &ncc_loss[i]);
        if (rc != 0) {
            printf("FAIL: %s pipeline returned %d\n", names[i], rc);
            return 1;
        }
        if (i == 0) {
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++) cpu_aff44[r][c] = affine[0].affine_mat[r][c];
            cpu_aff44[3][3] = 1.0f;
        }
        float cf = global_corr(fdata, tensor_data_f32(&moved[i]), fN);
        printf("  %-7s warped vs fixed: corr %.4f   (%.1fs)\n",
               names[i], cf, get_time() - t0);
        if (cf < FIXED_MIN_CORR) {
            printf("    FAIL: below floor %.4f — this backend did not register\n",
                   FIXED_MIN_CORR);
            fail = 1;
        }
    }

    printf("\n  Rigid / affine matrices vs CPU:\n");
    for (int i = 1; i < n; i++) {
        float rl, rt, al, at;
        mat34_max_diff(rigid[i].rigid_mat, rigid[0].rigid_mat, &rl, &rt);
        mat34_max_diff(affine[i].affine_mat, affine[0].affine_mat, &al, &at);
        int bad = rl > LINEAR_MAX_DIFF || al > LINEAR_MAX_DIFF ||
                  rt > TRANS_MAX_DIFF_MM || at > TRANS_MAX_DIFF_MM;
        printf("    %-7s rigid  lin %.6f  trans %.4f mm\n", names[i], rl, rt);
        printf("    %-7s affine lin %.6f  trans %.4f mm   %s\n",
               names[i], al, at, bad ? "FAIL" : "OK");
        if (bad) fail = 1;
    }

    printf("\n  Warped volume agreement:\n");
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            float c = global_corr(tensor_data_f32(&moved[i]), tensor_data_f32(&moved[j]), fN);
            float e = rmse(tensor_data_f32(&moved[i]), tensor_data_f32(&moved[j]), fN);
            printf("    %-7s vs %-7s: corr %.5f  rmse %.4f  %s\n",
                   names[i], names[j], c, e, c >= PARITY_MIN_CORR ? "OK" : "FAIL");
            if (c < PARITY_MIN_CORR) fail = 1;
        }

    printf("\n  Deformable-stage local NCC loss:\n");
    for (int i = 0; i < n; i++) {
        float d = fabsf(ncc_loss[i] - ncc_loss[0]);
        int bad = d > DEFORM_MAX_NCC_DIFF;
        printf("    %-7s %.6f   diff vs CPU %.6f   %s\n",
               names[i], ncc_loss[i], d, bad ? "FAIL" : "OK");
        if (bad) fail = 1;
    }

    printf("\n  Greedy displacement agreement:\n");
    size_t disp_n = disp[0].numel;
    for (int i = 1; i < n; i++) {
        if (disp[i].numel != disp_n) {
            printf("    %-7s shape mismatch (%zu vs %zu)   FAIL\n",
                   names[i], disp[i].numel, disp_n);
            fail = 1;
            continue;
        }
        float c = global_corr(tensor_data_f32(&disp[0]),
                              tensor_data_f32(&disp[i]), disp_n);
        float e = rmse(tensor_data_f32(&disp[0]),
                       tensor_data_f32(&disp[i]), disp_n);
        printf("    CPU     vs %-7s: corr %.5f  rmse %.7f  %s\n",
               names[i], c, e, c >= DISP_MIN_CORR ? "OK" : "FAIL");
        if (c < DISP_MIN_CORR) fail = 1;
    }

    if (check_syn_parity(names, ids, n) != 0) fail = 1;

    for (int i = 0; i < n; i++) {
        tensor_free(&moved[i]);
        tensor_free(&disp[i]);
    }
    image_free(&fixed);
    image_free(&moving);
    cfireants_cleanup();

    printf("\n%s\n", fail ? "FAILED" : "PASSED");
    return fail;
}
