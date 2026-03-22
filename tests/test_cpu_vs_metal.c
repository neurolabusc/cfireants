/*
 * test_cpu_vs_metal.c — Stage-by-stage comparison of CPU and Metal backends
 *
 * Runs Moments → Rigid → Affine → Greedy on the small dataset using both
 * CPU and Metal, printing matrices and NCC at each stage for comparison.
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
#include <time.h>

#ifdef CFIREANTS_HAS_METAL
extern int cfireants_init_metal(void);
#endif

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void print_mat34(const char *label, const float m[3][4]) {
    printf("  %s:\n", label);
    for (int i = 0; i < 3; i++)
        printf("    [%10.6f %10.6f %10.6f %10.6f]\n",
               m[i][0], m[i][1], m[i][2], m[i][3]);
}

static float mat34_diff(const float a[3][4], const float b[3][4]) {
    float maxd = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++) {
            float d = fabsf(a[i][j] - b[i][j]);
            if (d > maxd) maxd = d;
        }
    return maxd;
}

static float compute_global_ncc(const float *a, const float *b, int n) {
    double sa=0,sb=0,sab=0,sa2=0,sb2=0;
    for(int i=0;i<n;i++){
        sa+=a[i]; sb+=b[i]; sab+=a[i]*b[i];
        sa2+=a[i]*a[i]; sb2+=b[i]*b[i];
    }
    double ma=sa/n, mb=sb/n;
    double cov=sab/n-ma*mb;
    double va=sa2/n-ma*ma, vb=sb2/n-mb*mb;
    if(va<1e-10||vb<1e-10) return 0;
    return (float)(cov/sqrt(va*vb));
}

int main(int argc, char **argv) {
    cfireants_init_cpu();
#ifndef CFIREANTS_HAS_METAL
    printf("Metal not enabled — build with -DCFIREANTS_METAL=ON\n");
    return 1;
#else
    if (cfireants_init_metal() != 0) { printf("Metal init failed\n"); return 1; }

    /* Use CC for all stages so CPU and Metal are directly comparable.
     * MI would also work but CC is simpler to debug differences. */
    int use_mi = 0, use_syn = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mi") == 0) use_mi = 1;
        if (strcmp(argv[i], "--syn") == 0) use_syn = 1;
    }

    int loss_type = use_mi ? LOSS_MI : LOSS_CC;
    const char *loss_name = use_mi ? "MI" : "CC";

    printf("=== CPU vs Metal Stage-by-Stage Comparison ===\n");
    printf("Loss: %s, Downsample: trilinear, Dataset: small\n\n", loss_name);

    image_t fixed, moving;
    if (image_load(&fixed, "validate/small/MNI152_T1_2mm.nii.gz", DEVICE_CPU) != 0 ||
        image_load(&moving, "validate/small/T1_head_2mm.nii.gz", DEVICE_CPU) != 0) {
        printf("Failed to load small dataset\n");
        return 1;
    }
    int fD = fixed.data.shape[2], fH = fixed.data.shape[3], fW = fixed.data.shape[4];
    int fN = fD * fH * fW;

    /* ============================================================ */
    /* Stage 1: Moments (always CPU)                                 */
    /* ============================================================ */
    printf("--- Stage 1: Moments ---\n");
    moments_opts_t mopts = moments_opts_default();
    moments_result_t mom;
    moments_register(&fixed, &moving, mopts, &mom);
    printf("  Moments NCC: %.6f\n", mom.ncc_loss);
    print_mat34("Moments 3x4", mom.affine);

    /* ============================================================ */
    /* Stage 2: Rigid                                                */
    /* ============================================================ */
    printf("\n--- Stage 2: Rigid (%s) ---\n", loss_name);
    int rs[] = {4, 2, 1}, ri[] = {200, 100, 50};
    rigid_opts_t ropts = {
        .n_scales = 3, .scales = rs, .iterations = ri,
        .lr = 3e-3f, .loss_type = loss_type, .mi_num_bins = 32,
        .cc_kernel_size = 5,
        .tolerance = 1e-6f, .max_tolerance_iters = 10,
        .downsample_mode = DOWNSAMPLE_TRILINEAR };

    /* CPU rigid */
    double t0 = get_time();
    rigid_result_t rigid_cpu;
    rigid_register(&fixed, &moving, &mom, ropts, &rigid_cpu);
    double t_rigid_cpu = get_time() - t0;

    /* Metal rigid */
    t0 = get_time();
    rigid_result_t rigid_metal;
    rigid_register_metal(&fixed, &moving, &mom, ropts, &rigid_metal);
    double t_rigid_metal = get_time() - t0;

    print_mat34("CPU rigid", rigid_cpu.rigid_mat);
    print_mat34("Metal rigid", rigid_metal.rigid_mat);
    float rigid_diff = mat34_diff(rigid_cpu.rigid_mat, rigid_metal.rigid_mat);
    printf("  Max element diff: %.6f\n", rigid_diff);
    printf("  CPU NCC: %.6f  Metal NCC: %.6f\n", rigid_cpu.ncc_loss, rigid_metal.ncc_loss);
    printf("  CPU time: %.1fs  Metal time: %.1fs\n", t_rigid_cpu, t_rigid_metal);
    printf("  %s\n", rigid_diff < 0.1f ? "PASS (diff < 0.1)" : "*** LARGE DIFF ***");

    /* ============================================================ */
    /* Stage 3: Affine                                               */
    /* ============================================================ */
    printf("\n--- Stage 3: Affine (%s) ---\n", loss_name);
    int as[] = {4, 2, 1}, ai[] = {200, 100, 50};
    affine_opts_t aopts = {
        .n_scales = 3, .scales = as, .iterations = ai,
        .lr = 1e-3f, .loss_type = loss_type, .mi_num_bins = 32,
        .cc_kernel_size = 5,
        .tolerance = 1e-6f, .max_tolerance_iters = 10,
        .downsample_mode = DOWNSAMPLE_TRILINEAR };

    /* CPU affine (from CPU rigid) */
    t0 = get_time();
    affine_result_t affine_cpu;
    affine_register(&fixed, &moving, rigid_cpu.rigid_mat, aopts, &affine_cpu);
    double t_affine_cpu = get_time() - t0;

    /* Metal affine (from Metal rigid) */
    t0 = get_time();
    affine_result_t affine_metal;
    affine_register_metal(&fixed, &moving, rigid_metal.rigid_mat, aopts, &affine_metal);
    double t_affine_metal = get_time() - t0;

    print_mat34("CPU affine", affine_cpu.affine_mat);
    print_mat34("Metal affine", affine_metal.affine_mat);
    float affine_diff = mat34_diff(affine_cpu.affine_mat, affine_metal.affine_mat);
    printf("  Max element diff: %.6f\n", affine_diff);
    printf("  CPU NCC: %.6f  Metal NCC: %.6f\n", affine_cpu.ncc_loss, affine_metal.ncc_loss);
    printf("  CPU time: %.1fs  Metal time: %.1fs\n", t_affine_cpu, t_affine_metal);
    printf("  %s\n", affine_diff < 0.5f ? "PASS (diff < 0.5)" : "*** LARGE DIFF ***");

    /* ============================================================ */
    /* Stage 4: Greedy deformable                                    */
    /* ============================================================ */
    printf("\n--- Stage 4: Greedy (CC) ---\n");
    int gs[] = {4, 2, 1}, gi[] = {200, 100, 50};
    greedy_opts_t gopts = {
        .n_scales = 3, .scales = gs, .iterations = gi,
        .cc_kernel_size = 5, .lr = 0.1f,
        .smooth_warp_sigma = 0.5f, .smooth_grad_sigma = 1.0f,
        .tolerance = 1e-6f, .max_tolerance_iters = 10,
        .downsample_mode = DOWNSAMPLE_TRILINEAR };

    /* Build 4x4 from each affine result */
    float aff44_cpu[4][4] = {{0}}, aff44_metal[4][4] = {{0}};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++) {
            aff44_cpu[i][j] = affine_cpu.affine_mat[i][j];
            aff44_metal[i][j] = affine_metal.affine_mat[i][j];
        }
    aff44_cpu[3][3] = 1.0f;
    aff44_metal[3][3] = 1.0f;

    /* CPU greedy */
    t0 = get_time();
    greedy_result_t greedy_cpu;
    greedy_register(&fixed, &moving, aff44_cpu, gopts, &greedy_cpu);
    double t_greedy_cpu = get_time() - t0;

    /* Metal greedy */
    t0 = get_time();
    greedy_result_t greedy_metal;
    greedy_register_metal(&fixed, &moving, aff44_metal, gopts, &greedy_metal);
    double t_greedy_metal = get_time() - t0;

    float ncc_cpu = compute_global_ncc((float *)fixed.data.data,
                                        (float *)greedy_cpu.moved.data, fN);
    float ncc_metal = compute_global_ncc((float *)fixed.data.data,
                                          (float *)greedy_metal.moved.data, fN);

    printf("  CPU:   Global NCC = %.4f  Local NCC Loss = %.4f  Time = %.1fs\n",
           ncc_cpu, greedy_cpu.ncc_loss, t_greedy_cpu);
    printf("  Metal: Global NCC = %.4f  Local NCC Loss = %.4f  Time = %.1fs\n",
           ncc_metal, greedy_metal.ncc_loss, t_greedy_metal);
    float ncc_diff = fabsf(ncc_cpu - ncc_metal);
    printf("  Global NCC diff: %.4f\n", ncc_diff);
    printf("  %s\n", ncc_diff < 0.05f ? "PASS (diff < 0.05)" : "*** LARGE DIFF ***");

    /* ============================================================ */
    /* Stage 5: SyN (optional — slow on CPU)                         */
    /* ============================================================ */
    float syn_ncc_cpu = 0, syn_ncc_metal = 0, syn_ncc_diff = 0;
    if (use_syn) {
        printf("\n--- Stage 5: SyN (CC) ---\n");
        int ss[] = {4, 2, 1}, si_arr[] = {200, 100, 50};
        syn_opts_t sopts = {
            .n_scales = 3, .scales = ss, .iterations = si_arr,
            .cc_kernel_size = 5, .lr = 0.1f,
            .smooth_warp_sigma = 0.5f, .smooth_grad_sigma = 1.0f,
            .tolerance = 1e-6f, .max_tolerance_iters = 10,
            .downsample_mode = DOWNSAMPLE_TRILINEAR };

        t0 = get_time();
        syn_result_t syn_cpu;
        syn_register(&fixed, &moving, aff44_cpu, sopts, &syn_cpu);
        double t_syn_cpu = get_time() - t0;

        t0 = get_time();
        syn_result_t syn_metal;
        syn_register_metal(&fixed, &moving, aff44_metal, sopts, &syn_metal);
        double t_syn_metal = get_time() - t0;

        syn_ncc_cpu = compute_global_ncc((float *)fixed.data.data,
                                          (float *)syn_cpu.moved.data, fN);
        syn_ncc_metal = compute_global_ncc((float *)fixed.data.data,
                                            (float *)syn_metal.moved.data, fN);
        syn_ncc_diff = fabsf(syn_ncc_cpu - syn_ncc_metal);

        printf("  CPU:   Global NCC = %.4f  Local NCC Loss = %.4f  Time = %.1fs\n",
               syn_ncc_cpu, syn_cpu.ncc_loss, t_syn_cpu);
        printf("  Metal: Global NCC = %.4f  Local NCC Loss = %.4f  Time = %.1fs\n",
               syn_ncc_metal, syn_metal.ncc_loss, t_syn_metal);
        printf("  Global NCC diff: %.4f\n", syn_ncc_diff);
        printf("  %s\n", syn_ncc_diff < 0.05f ? "PASS (diff < 0.05)" : "*** LARGE DIFF ***");

        tensor_free(&syn_cpu.moved);
        tensor_free(&syn_metal.moved);
    }

    /* ============================================================ */
    /* Summary                                                       */
    /* ============================================================ */
    printf("\n=== Summary ===\n");
    printf("  Rigid:  max diff = %.6f  %s\n", rigid_diff,
           rigid_diff < 0.1f ? "OK" : "FAIL");
    printf("  Affine: max diff = %.6f  %s\n", affine_diff,
           affine_diff < 0.5f ? "OK" : "FAIL");
    printf("  Greedy: NCC diff = %.4f  CPU=%.4f Metal=%.4f  %s\n",
           ncc_diff, ncc_cpu, ncc_metal,
           ncc_diff < 0.05f ? "OK" : "FAIL");
    if (use_syn)
        printf("  SyN:    NCC diff = %.4f  CPU=%.4f Metal=%.4f  %s\n",
               syn_ncc_diff, syn_ncc_cpu, syn_ncc_metal,
               syn_ncc_diff < 0.05f ? "OK" : "FAIL");

    tensor_free(&greedy_cpu.moved);
    tensor_free(&greedy_metal.moved);
    image_free(&fixed);
    image_free(&moving);
    return 0;
#endif
}
