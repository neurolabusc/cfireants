/*
 * registration.h - Registration algorithms for cfireants
 */

#ifndef CFIREANTS_REGISTRATION_H
#define CFIREANTS_REGISTRATION_H

#include "cfireants/tensor.h"
#include "cfireants/image.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Global verbosity: 0=silent, 1=summary, 2=per-iteration */
extern int cfireants_verbose;

/* Result from moments registration */
typedef struct {
    float Rf[3][3];     /* Rotation matrix (physical space) */
    float tf[3];        /* Translation vector (physical space) */
    float affine[3][4]; /* Combined [Rf | tf] */
    float ncc_loss;     /* CC loss after applying the transform */
} moments_result_t;

/* Options for moments registration */
typedef struct {
    int moments;          /* 1 (translation only) or 2 (rotation+translation) */
    int orientation;      /* 0=rot, 1=antirot, 2=both */
    int blur;             /* 1=Gaussian blur before downsampling */
    float scale;          /* Downsampling factor (1.0 = no downsampling) */
    int cc_kernel_size;   /* CC kernel size for orientation selection */
    int try_identity;     /* 1=also evaluate identity+COM and pure identity candidates
                           * (not in Python FireANTs; useful when sforms already align) */
} moments_opts_t;

/* Default moments options */
static inline moments_opts_t moments_opts_default(void) {
    moments_opts_t o;
    o.moments = 2;
    o.orientation = 2; /* both */
    o.blur = 1;
    o.scale = 1.0f;
    o.cc_kernel_size = 5;
    o.try_identity = 0; /* off by default to match Python */
    return o;
}

/* Run moments registration.
 *   fixed:  stationary image
 *   moving: image to be registered
 *   opts:   registration options
 *   result: output rotation/translation
 * Returns 0 on success. */
int moments_register(const image_t *fixed, const image_t *moving,
                     moments_opts_t opts, moments_result_t *result);

/* Apply an affine transform and evaluate an image.
 * Warps moving image into fixed image space using the affine [3, 4].
 *   fixed:  defines the output space
 *   moving: source image
 *   affine: [3, 4] physical-space affine (Rf | tf)
 *   output: warped image tensor [1, 1, D, H, W]
 * Returns 0 on success. */
int apply_affine_transform(const image_t *fixed, const image_t *moving,
                           const float affine[3][4], tensor_t *output);

/* ------------------------------------------------------------------ */
/* Rigid registration                                                  */
/* ------------------------------------------------------------------ */

/* Result from rigid registration */
typedef struct {
    float rigid_mat[3][4]; /* [R | t] physical-space affine */
    float ncc_loss;        /* CC loss after registration */
} rigid_result_t;

/* Loss type codes */
#define LOSS_CC  0
#define LOSS_MI  1

/* Downsample mode codes */
#define DOWNSAMPLE_FFT       0   /* FFT-based (matching Python default) */
#define DOWNSAMPLE_TRILINEAR 1   /* Gaussian blur + trilinear resize (faster, GPU-native) */

/* Options for rigid registration */
typedef struct {
    int n_scales;              /* Number of scales */
    const int *scales;         /* Downsampling factors (descending) */
    const int *iterations;     /* Iterations per scale */
    int loss_type;             /* LOSS_CC or LOSS_MI */
    int cc_kernel_size;        /* CC kernel size (if LOSS_CC) */
    int mi_num_bins;           /* MI histogram bins (if LOSS_MI, default 32) */
    float lr;                  /* Adam learning rate */
    float tolerance;           /* Convergence tolerance */
    int max_tolerance_iters;   /* Max iters at tolerance before stopping */
    int downsample_mode;       /* DOWNSAMPLE_FFT or DOWNSAMPLE_TRILINEAR */
} rigid_opts_t;

/* Run rigid registration.
 *   fixed:        stationary image
 *   moving:       image to be registered
 *   moments_init: optional moments initialization (NULL for identity)
 *   opts:         registration options
 *   result:       output rigid matrix
 * Returns 0 on success. */
int rigid_register(const image_t *fixed, const image_t *moving,
                   const moments_result_t *moments_init,
                   rigid_opts_t opts, rigid_result_t *result);

#ifdef CFIREANTS_HAS_CUDA
int rigid_register_gpu(const image_t *fixed, const image_t *moving,
                       const moments_result_t *moments_init,
                       rigid_opts_t opts, rigid_result_t *result);
#endif

#ifdef CFIREANTS_HAS_WEBGPU
int rigid_register_webgpu(const image_t *fixed, const image_t *moving,
                           const moments_result_t *moments_init,
                           rigid_opts_t opts, rigid_result_t *result);
#endif

#ifdef CFIREANTS_HAS_METAL
int rigid_register_metal(const image_t *fixed, const image_t *moving,
                          const moments_result_t *moments_init,
                          rigid_opts_t opts, rigid_result_t *result);
#endif

/* ------------------------------------------------------------------ */
/* Affine registration                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    float affine_mat[3][4]; /* [A | t] physical-space affine */
    float ncc_loss;
} affine_result_t;

typedef struct {
    int n_scales;
    const int *scales;
    const int *iterations;
    int loss_type;           /* LOSS_CC or LOSS_MI */
    int cc_kernel_size;
    int mi_num_bins;
    float lr;
    float tolerance;
    int max_tolerance_iters;
    int downsample_mode;       /* DOWNSAMPLE_FFT or DOWNSAMPLE_TRILINEAR */
} affine_opts_t;

int affine_register(const image_t *fixed, const image_t *moving,
                    const float init_rigid_34[3][4],
                    affine_opts_t opts, affine_result_t *result);

#ifdef CFIREANTS_HAS_CUDA
int affine_register_gpu(const image_t *fixed, const image_t *moving,
                        const float init_rigid_34[3][4],
                        affine_opts_t opts, affine_result_t *result);
#endif

#ifdef CFIREANTS_HAS_WEBGPU
int affine_register_webgpu(const image_t *fixed, const image_t *moving,
                            const float init_rigid_34[3][4],
                            affine_opts_t opts, affine_result_t *result);
#endif

#ifdef CFIREANTS_HAS_METAL
int affine_register_metal(const image_t *fixed, const image_t *moving,
                           const float init_rigid_34[3][4],
                           affine_opts_t opts, affine_result_t *result);
#endif

/* ------------------------------------------------------------------ */
/* Greedy deformable registration                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    tensor_t disp;          /* displacement field [1, D, H, W, 3] (owned) */
    tensor_t moved;         /* warped image [1, 1, D, H, W] on CPU (owned) */
    float affine_44[4][4];  /* initial affine (physical space) */
    float ncc_loss;         /* local NCC loss (CC k=9) */
} greedy_result_t;

typedef struct {
    int n_scales;
    const int *scales;
    const int *iterations;
    int cc_kernel_size;
    float lr;
    float smooth_warp_sigma;
    float smooth_grad_sigma;
    float tolerance;
    int max_tolerance_iters;
    int downsample_mode;       /* DOWNSAMPLE_FFT or DOWNSAMPLE_TRILINEAR */
} greedy_opts_t;

int greedy_register(const image_t *fixed, const image_t *moving,
                    const float init_affine_44[4][4],
                    greedy_opts_t opts, greedy_result_t *result);

#ifdef CFIREANTS_HAS_CUDA
/* GPU-accelerated greedy (all computation on device) */
int greedy_register_gpu(const image_t *fixed, const image_t *moving,
                        const float init_affine_44[4][4],
                        greedy_opts_t opts, greedy_result_t *result);
#endif

#ifdef CFIREANTS_HAS_WEBGPU
int greedy_register_webgpu(const image_t *fixed, const image_t *moving,
                            const float init_affine_44[4][4],
                            greedy_opts_t opts, greedy_result_t *result);
#endif

#ifdef CFIREANTS_HAS_METAL
int greedy_register_metal(const image_t *fixed, const image_t *moving,
                           const float init_affine_44[4][4],
                           greedy_opts_t opts, greedy_result_t *result);
#endif

/* ------------------------------------------------------------------ */
/* SyN (symmetric) deformable registration                             */
/* ------------------------------------------------------------------ */

typedef struct {
    tensor_t fwd_disp;      /* forward displacement [1, D, H, W, 3] */
    tensor_t rev_disp;      /* reverse displacement [1, D, H, W, 3] */
    tensor_t moved;         /* warped image [1, 1, D, H, W] on CPU */
    float affine_44[4][4];  /* initial affine */
    float ncc_loss;
} syn_result_t;

typedef struct {
    int n_scales;
    const int *scales;
    const int *iterations;
    int cc_kernel_size;
    float lr;
    float smooth_warp_sigma;
    float smooth_grad_sigma;
    float tolerance;
    int max_tolerance_iters;
    int downsample_mode;       /* DOWNSAMPLE_FFT or DOWNSAMPLE_TRILINEAR */
} syn_opts_t;

int syn_register(const image_t *fixed, const image_t *moving,
                 const float init_affine_44[4][4],
                 syn_opts_t opts, syn_result_t *result);

#ifdef CFIREANTS_HAS_CUDA
int syn_register_gpu(const image_t *fixed, const image_t *moving,
                     const float init_affine_44[4][4],
                     syn_opts_t opts, syn_result_t *result);
#endif

#ifdef CFIREANTS_HAS_WEBGPU
int syn_register_webgpu(const image_t *fixed, const image_t *moving,
                         const float init_affine_44[4][4],
                         syn_opts_t opts, syn_result_t *result);
#endif

#ifdef CFIREANTS_HAS_METAL
int syn_register_metal(const image_t *fixed, const image_t *moving,
                        const float init_affine_44[4][4],
                        syn_opts_t opts, syn_result_t *result);
#endif

/* Evaluate SyN result: warp moving into fixed space */
int syn_evaluate(const image_t *fixed, const image_t *moving,
                 const syn_result_t *result, tensor_t *output);

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_REGISTRATION_H */
