/*
 * cfireants — ANTs-style command-line registration tool
 *
 * Usage mirrors antsRegistration:
 *   cfireants -f fixed.nii.gz -m moving.nii.gz \
 *     --transform Rigid[0.003] --metric MI[32] --convergence [200x100x50,1e-6,10] \
 *     --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0 \
 *     --transform Affine[0.001] --metric MI[32] --convergence [200x100x50,1e-6,10] \
 *     --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0 \
 *     --transform SyN[0.1,0.5,1.0] --metric CC[5] --convergence [200x100x50,1e-6,10] \
 *     --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0 \
 *     -o output_
 */

#define CFIREANTS_VERSION "0.1.20260323"

#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/backend.h"
#include "cfireants/registration.h"
#include "cfireants/losses.h"
#include "cfireants/interpolator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

#ifdef CFIREANTS_HAS_METAL
extern int cfireants_init_metal(void);
#endif
#ifdef CFIREANTS_HAS_WEBGPU
extern int cfireants_init_webgpu(void);
#endif
#ifdef CFIREANTS_HAS_CUDA
extern int cfireants_init_cuda(void);
#endif

/* Maximum registration stages */
#define MAX_STAGES 8
#define MAX_LEVELS 8

typedef enum { STAGE_RIGID, STAGE_AFFINE, STAGE_SYN, STAGE_GREEDY } stage_type_t;
typedef enum { BACKEND_CPU, BACKEND_METAL, BACKEND_WEBGPU, BACKEND_CUDA } backend_t;

typedef struct {
    stage_type_t type;
    float lr;
    float syn_warp_sigma;   /* SyN[lr, warp_sigma, grad_sigma] */
    float syn_grad_sigma;

    int metric_type;        /* LOSS_CC or LOSS_MI */
    int metric_param;       /* CC: kernel_size, MI: num_bins */

    int n_levels;
    int iterations[MAX_LEVELS];
    int shrink_factors[MAX_LEVELS];
    float smoothing_sigmas[MAX_LEVELS];

    float tolerance;
    int tolerance_window;
} stage_config_t;

typedef struct {
    const char *fixed_path;
    const char *moving_path;
    const char *output_prefix;
    const char *output_warped;
    const char *skullstrip_mask;   /* mask in moving/template space */
    float skullstrip_threshold;    /* threshold for warped mask (default 0.5) */

    int n_stages;
    stage_config_t stages[MAX_STAGES];

    backend_t backend;
    int downsample_mode;
    int initial_moving_transform;  /* 0=none, 1=moments */
    int verbose;  /* 0=silent (errors only), 1=summary, 2=debug (per-iter) */
} cli_config_t;

/* Verbosity uses cfireants_verbose from backend.h/registration.h */

/* Ensure parent directory of path exists, creating if needed.
 * Returns 0 on success, -1 on failure. */
static int ensure_parent_dir(const char *path) {
    char buf[512];
    strncpy(buf, path, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
    /* Find last / or \ */
    char *sep = strrchr(buf, '/');
    if (!sep) sep = strrchr(buf, '\\');
    if (!sep) return 0; /* no directory component */
    *sep = 0;
    if (buf[0] == 0) return 0; /* root */

    /* Try mkdir -p by walking the path */
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            *p = 0;
            mkdir(buf, 0755);
            *p = '/';
        }
    }
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Error: cannot create directory '%s': %s\n", buf, strerror(errno));
        return -1;
    }
    return 0;
}

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Parse "200x100x50" into int array, return count */
static int parse_xlist_int(const char *s, int *out, int max) {
    int n = 0;
    const char *p = s;
    while (n < max) {
        out[n++] = atoi(p);
        p = strchr(p, 'x');
        if (!p) break;
        p++;
    }
    return n;
}

/* Parse "2.0x1.0x0.0" into float array */
static int parse_xlist_float(const char *s, float *out, int max) {
    int n = 0;
    const char *p = s;
    while (n < max) {
        out[n++] = (float)atof(p);
        p = strchr(p, 'x');
        if (!p) break;
        p++;
    }
    return n;
}

/* Parse --transform Type[params] */
static int parse_transform(const char *arg, stage_config_t *stage) {
    if (strncmp(arg, "Rigid[", 6) == 0) {
        stage->type = STAGE_RIGID;
        stage->lr = (float)atof(arg + 6);
    } else if (strncmp(arg, "Affine[", 7) == 0) {
        stage->type = STAGE_AFFINE;
        stage->lr = (float)atof(arg + 7);
    } else if (strncmp(arg, "SyN[", 4) == 0) {
        stage->type = STAGE_SYN;
        float params[3] = {0.1f, 0.5f, 1.0f};
        sscanf(arg + 4, "%f,%f,%f", &params[0], &params[1], &params[2]);
        stage->lr = params[0];
        stage->syn_warp_sigma = params[1];
        stage->syn_grad_sigma = params[2];
    } else if (strncmp(arg, "Greedy[", 7) == 0) {
        stage->type = STAGE_GREEDY;
        float params[3] = {0.1f, 0.5f, 1.0f};
        sscanf(arg + 7, "%f,%f,%f", &params[0], &params[1], &params[2]);
        stage->lr = params[0];
        stage->syn_warp_sigma = params[1];
        stage->syn_grad_sigma = params[2];
    } else if (strcmp(arg, "Rigid") == 0) {
        stage->type = STAGE_RIGID; stage->lr = 0.003f;
    } else if (strcmp(arg, "Affine") == 0) {
        stage->type = STAGE_AFFINE; stage->lr = 0.001f;
    } else if (strcmp(arg, "SyN") == 0) {
        stage->type = STAGE_SYN; stage->lr = 0.1f;
        stage->syn_warp_sigma = 0.5f; stage->syn_grad_sigma = 1.0f;
    } else if (strcmp(arg, "Greedy") == 0) {
        stage->type = STAGE_GREEDY; stage->lr = 0.1f;
        stage->syn_warp_sigma = 0.5f; stage->syn_grad_sigma = 1.0f;
    } else {
        fprintf(stderr, "Unknown transform: %s\n", arg);
        return -1;
    }
    /* Defaults */
    stage->metric_type = (stage->type <= STAGE_AFFINE) ? LOSS_MI : LOSS_CC;
    stage->metric_param = (stage->metric_type == LOSS_MI) ? 32 : 5;
    stage->tolerance = 1e-6f;
    stage->tolerance_window = 10;
    return 0;
}

/* Parse --metric MI[32] or CC[5] */
static int parse_metric(const char *arg, stage_config_t *stage) {
    if (strncmp(arg, "MI[", 3) == 0) {
        stage->metric_type = LOSS_MI;
        stage->metric_param = atoi(arg + 3);
        if (stage->metric_param <= 0) stage->metric_param = 32;
    } else if (strncmp(arg, "CC[", 3) == 0) {
        stage->metric_type = LOSS_CC;
        stage->metric_param = atoi(arg + 3);
        if (stage->metric_param <= 0) stage->metric_param = 5;
    } else if (strcmp(arg, "MI") == 0) {
        stage->metric_type = LOSS_MI; stage->metric_param = 32;
    } else if (strcmp(arg, "CC") == 0) {
        stage->metric_type = LOSS_CC; stage->metric_param = 5;
    } else {
        fprintf(stderr, "Unknown metric: %s\n", arg);
        return -1;
    }
    return 0;
}

/* Parse --convergence [200x100x50,1e-6,10] */
static int parse_convergence(const char *arg, stage_config_t *stage) {
    const char *p = arg;
    if (*p == '[') p++;
    /* Parse iterations (before first comma) */
    char iter_buf[128];
    const char *comma = strchr(p, ',');
    if (comma) {
        size_t len = comma - p;
        if (len >= sizeof(iter_buf)) len = sizeof(iter_buf) - 1;
        memcpy(iter_buf, p, len); iter_buf[len] = 0;
        stage->n_levels = parse_xlist_int(iter_buf, stage->iterations, MAX_LEVELS);
        p = comma + 1;
        stage->tolerance = (float)atof(p);
        comma = strchr(p, ',');
        if (comma) stage->tolerance_window = atoi(comma + 1);
    } else {
        /* Just iterations, strip trailing ] */
        char buf[128]; strncpy(buf, p, sizeof(buf)-1); buf[sizeof(buf)-1]=0;
        char *br = strchr(buf, ']'); if (br) *br = 0;
        stage->n_levels = parse_xlist_int(buf, stage->iterations, MAX_LEVELS);
    }
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -f <fixed> -m <moving> [options] [stages...]\n"
        "\n"
        "Required:\n"
        "  -f, --fixed <file>          Fixed (stationary) NIfTI image\n"
        "  -m, --moving <file>         Moving image to register\n"
        "\n"
        "Output:\n"
        "  -o, --output <prefix>       Output prefix (default: output_)\n"
        "  -w, --warped <file>         Warped output NIfTI (default: <prefix>Warped.nii.gz)\n"
        "\n"
        "Backend:\n"
        "  --backend <name>            cpu, metal, webgpu, cuda (default: best available)\n"
        "  --trilinear                 Use trilinear downsample (default: FFT)\n"
        "\n"
        "Initial transform:\n"
        "  --moments                   Initialize with center-of-mass + orientation (default)\n"
        "  --no-moments                Skip moments initialization\n"
        "\n"
        "Skull stripping:\n"
        "  --skullstrip <mask.nii.gz>  Brain mask in template/moving space. Warps mask to subject\n"
        "                              space, thresholds at 0.5, applies to fixed image. When used,\n"
        "                              -o is the output filename (not a prefix). Preserves native\n"
        "                              datatype. Non-brain voxels set to darkest intensity.\n"
        "\n"
        "Registration stages (repeat for each stage):\n"
        "  --transform <Type[params]>  Rigid[lr], Affine[lr], SyN[lr,warp_sigma,grad_sigma],\n"
        "                              Greedy[lr,warp_sigma,grad_sigma]\n"
        "  --metric <Type[param]>      MI[bins] or CC[kernel_size] (default: MI for linear, CC for deformable)\n"
        "  --convergence <[itersxiters...,tol,win]>\n"
        "                              Iterations per level, tolerance, window (default: [200x100x50,1e-6,10])\n"
        "  --shrink-factors <NxNx...>  Downsample factors per level (default: 4x2x1)\n"
        "  --smoothing-sigmas <NxNx...> Blur sigmas per level (currently unused, reserved)\n"
        "\n"
        "Presets:\n"
        "  --rigid                     Shorthand for --transform Rigid[0.003] --metric MI[32]\n"
        "  --affine                    Shorthand for Rigid + Affine stages\n"
        "  --syn                       Shorthand for Rigid + Affine + SyN (default)\n"
        "  --greedy                    Shorthand for Rigid + Affine + Greedy\n"
        "\n"
        "  -v, --verbose [level]       0=silent (default), 1=summary, 2=per-iteration\n"
        "  --version                   Print version and exit\n"
        "  -h, --help                  Show this help\n"
        "\n"
        "Examples:\n"
        "  # Default full pipeline (Moments + Rigid MI + Affine MI + SyN CC)\n"
        "  %s -f mni.nii.gz -m subject.nii.gz -o reg_\n"
        "\n"
        "  # Affine only (no deformable)\n"
        "  %s -f mni.nii.gz -m subject.nii.gz --affine -o reg_\n"
        "\n"
        "  # Custom stages\n"
        "  %s -f mni.nii.gz -m subject.nii.gz \\\n"
        "    --transform Rigid[0.003] --metric MI[32] --convergence [200x100x50,1e-6,10] --shrink-factors 4x2x1 \\\n"
        "    --transform Affine[0.001] --metric MI[32] --convergence [200x100x50,1e-6,10] --shrink-factors 4x2x1 \\\n"
        "    --transform SyN[0.1,0.5,1.0] --metric CC[5] --convergence [200x100x50,1e-6,10] --shrink-factors 4x2x1 \\\n"
        "    -o reg_\n",
        prog, prog, prog, prog);
}

/* Set default stages for a preset */
static void set_preset(cli_config_t *cfg, const char *preset) {
    int default_iters[] = {200, 100, 50};
    int default_shrink[] = {4, 2, 1};
    cfg->n_stages = 0;

    if (strcmp(preset, "rigid") == 0 || strcmp(preset, "affine") == 0 ||
        strcmp(preset, "syn") == 0 || strcmp(preset, "greedy") == 0) {
        /* Rigid stage */
        stage_config_t *r = &cfg->stages[cfg->n_stages++];
        memset(r, 0, sizeof(*r));
        r->type = STAGE_RIGID; r->lr = 0.003f;
        r->metric_type = LOSS_MI; r->metric_param = 32;
        r->n_levels = 3;
        memcpy(r->iterations, default_iters, sizeof(default_iters));
        memcpy(r->shrink_factors, default_shrink, sizeof(default_shrink));
        r->tolerance = 1e-6f; r->tolerance_window = 10;
    }

    if (strcmp(preset, "affine") == 0 || strcmp(preset, "syn") == 0 ||
        strcmp(preset, "greedy") == 0) {
        /* Affine stage */
        stage_config_t *a = &cfg->stages[cfg->n_stages++];
        memset(a, 0, sizeof(*a));
        a->type = STAGE_AFFINE; a->lr = 0.001f;
        a->metric_type = LOSS_MI; a->metric_param = 32;
        a->n_levels = 3;
        memcpy(a->iterations, default_iters, sizeof(default_iters));
        memcpy(a->shrink_factors, default_shrink, sizeof(default_shrink));
        a->tolerance = 1e-6f; a->tolerance_window = 10;
    }

    if (strcmp(preset, "syn") == 0) {
        stage_config_t *s = &cfg->stages[cfg->n_stages++];
        memset(s, 0, sizeof(*s));
        s->type = STAGE_SYN; s->lr = 0.1f;
        s->syn_warp_sigma = 0.5f; s->syn_grad_sigma = 1.0f;
        s->metric_type = LOSS_CC; s->metric_param = 5;
        s->n_levels = 3;
        memcpy(s->iterations, default_iters, sizeof(default_iters));
        memcpy(s->shrink_factors, default_shrink, sizeof(default_shrink));
        s->tolerance = 1e-6f; s->tolerance_window = 10;
    }

    if (strcmp(preset, "greedy") == 0) {
        stage_config_t *g = &cfg->stages[cfg->n_stages++];
        memset(g, 0, sizeof(*g));
        g->type = STAGE_GREEDY; g->lr = 0.1f;
        g->syn_warp_sigma = 0.5f; g->syn_grad_sigma = 1.0f;
        g->metric_type = LOSS_CC; g->metric_param = 5;
        g->n_levels = 3;
        memcpy(g->iterations, default_iters, sizeof(default_iters));
        memcpy(g->shrink_factors, default_shrink, sizeof(default_shrink));
        g->tolerance = 1e-6f; g->tolerance_window = 10;
    }
}

static int parse_args(int argc, char **argv, cli_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->output_prefix = "output_";
    cfg->skullstrip_threshold = 0.5f;
    cfg->initial_moving_transform = 1; /* moments by default */
    cfg->backend = BACKEND_CPU;

    /* Auto-select best backend */
#ifdef CFIREANTS_HAS_METAL
    cfg->backend = BACKEND_METAL;
#elif defined(CFIREANTS_HAS_CUDA)
    cfg->backend = BACKEND_CUDA;
#elif defined(CFIREANTS_HAS_WEBGPU)
    cfg->backend = BACKEND_WEBGPU;
#endif

    int current_stage = -1;

    #define NEED_ARG() do { if (i + 1 >= argc) { \
        fprintf(stderr, "Missing value for %s\n", arg); return -1; } } while(0)

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            print_usage(argv[0]); return -1;
        } else if (strcmp(arg, "-f") == 0 || strcmp(arg, "--fixed") == 0) {
            NEED_ARG(); cfg->fixed_path = argv[++i];
        } else if (strcmp(arg, "-m") == 0 || strcmp(arg, "--moving") == 0) {
            NEED_ARG(); cfg->moving_path = argv[++i];
        } else if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            NEED_ARG(); cfg->output_prefix = argv[++i];
        } else if (strcmp(arg, "-w") == 0 || strcmp(arg, "--warped") == 0) {
            NEED_ARG(); cfg->output_warped = argv[++i];
        } else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0) {
            if (i + 1 < argc && argv[i+1][0] >= '0' && argv[i+1][0] <= '9')
                cfg->verbose = atoi(argv[++i]);
            else
                cfg->verbose = 1;  /* -v without number = 1 */
        } else if (strcmp(arg, "--version") == 0) {
            printf("cfireants_reg %s\n", CFIREANTS_VERSION);
            exit(0);
        } else if (strcmp(arg, "--trilinear") == 0) {
            cfg->downsample_mode = DOWNSAMPLE_TRILINEAR;
        } else if (strcmp(arg, "--moments") == 0) {
            cfg->initial_moving_transform = 1;
        } else if (strcmp(arg, "--no-moments") == 0) {
            cfg->initial_moving_transform = 0;
        } else if (strcmp(arg, "--skullstrip") == 0) {
            NEED_ARG(); cfg->skullstrip_mask = argv[++i];
        } else if (strcmp(arg, "--backend") == 0) {
            NEED_ARG(); const char *b = argv[++i];
            if (strcmp(b, "cpu") == 0) cfg->backend = BACKEND_CPU;
            else if (strcmp(b, "metal") == 0) cfg->backend = BACKEND_METAL;
            else if (strcmp(b, "webgpu") == 0) cfg->backend = BACKEND_WEBGPU;
            else if (strcmp(b, "cuda") == 0) cfg->backend = BACKEND_CUDA;
            else { fprintf(stderr, "Unknown backend: %s\n", b); return -1; }
        } else if (strcmp(arg, "--rigid") == 0) {
            set_preset(cfg, "rigid");
        } else if (strcmp(arg, "--affine") == 0) {
            set_preset(cfg, "affine");
        } else if (strcmp(arg, "--syn") == 0) {
            set_preset(cfg, "syn");
        } else if (strcmp(arg, "--greedy") == 0) {
            set_preset(cfg, "greedy");
        } else if (strcmp(arg, "--transform") == 0) {
            if (cfg->n_stages >= MAX_STAGES) { fprintf(stderr, "Too many stages\n"); return -1; }
            current_stage = cfg->n_stages;
            stage_config_t *s = &cfg->stages[cfg->n_stages++];
            memset(s, 0, sizeof(*s));
            /* Default convergence */
            s->n_levels = 3;
            int di[] = {200, 100, 50}; memcpy(s->iterations, di, sizeof(di));
            int ds[] = {4, 2, 1}; memcpy(s->shrink_factors, ds, sizeof(ds));
            s->tolerance = 1e-6f; s->tolerance_window = 10;
            NEED_ARG(); if (parse_transform(argv[++i], s) != 0) return -1;
        } else if (strcmp(arg, "--metric") == 0) {
            if (current_stage < 0) { fprintf(stderr, "--metric before --transform\n"); return -1; }
            NEED_ARG(); if (parse_metric(argv[++i], &cfg->stages[current_stage]) != 0) return -1;
        } else if (strcmp(arg, "--convergence") == 0) {
            if (current_stage < 0) { fprintf(stderr, "--convergence before --transform\n"); return -1; }
            NEED_ARG(); if (parse_convergence(argv[++i], &cfg->stages[current_stage]) != 0) return -1;
        } else if (strcmp(arg, "--shrink-factors") == 0) {
            if (current_stage < 0) { fprintf(stderr, "--shrink-factors before --transform\n"); return -1; }
            stage_config_t *s = &cfg->stages[current_stage];
            NEED_ARG(); int n = parse_xlist_int(argv[++i], s->shrink_factors, MAX_LEVELS);
            if (s->n_levels == 0) s->n_levels = n;
        } else if (strcmp(arg, "--smoothing-sigmas") == 0) {
            if (current_stage < 0) { fprintf(stderr, "--smoothing-sigmas before --transform\n"); return -1; }
            NEED_ARG(); parse_xlist_float(argv[++i], cfg->stages[current_stage].smoothing_sigmas, MAX_LEVELS);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            print_usage(argv[0]);
            return -1;
        }
    }

    if (!cfg->fixed_path || !cfg->moving_path) {
        fprintf(stderr, "Error: -f and -m are required\n");
        print_usage(argv[0]);
        return -1;
    }

    /* Default: full SyN pipeline */
    if (cfg->n_stages == 0)
        set_preset(cfg, "syn");

    return 0;
}

static const char *stage_name(stage_type_t t) {
    switch (t) {
        case STAGE_RIGID: return "Rigid";
        case STAGE_AFFINE: return "Affine";
        case STAGE_SYN: return "SyN";
        case STAGE_GREEDY: return "Greedy";
    }
    return "?";
}

int main(int argc, char **argv) {
    cli_config_t cfg;
    if (parse_args(argc, argv, &cfg) != 0) return 1;
    cfireants_verbose = cfg.verbose;

    /* Initialize backend */
    cfireants_init_cpu();
    if (cfg.backend == BACKEND_METAL) {
#ifdef CFIREANTS_HAS_METAL
        if (cfireants_init_metal() != 0) { fprintf(stderr, "Metal init failed\n"); return 1; }
#else
        fprintf(stderr, "Metal not compiled in\n"); return 1;
#endif
    } else if (cfg.backend == BACKEND_WEBGPU) {
#ifdef CFIREANTS_HAS_WEBGPU
        if (cfireants_init_webgpu() != 0) { fprintf(stderr, "WebGPU init failed\n"); return 1; }
#else
        fprintf(stderr, "WebGPU not compiled in\n"); return 1;
#endif
    } else if (cfg.backend == BACKEND_CUDA) {
#ifdef CFIREANTS_HAS_CUDA
        if (cfireants_init_cuda() != 0) { fprintf(stderr, "CUDA init failed\n"); return 1; }
#else
        fprintf(stderr, "CUDA not compiled in\n"); return 1;
#endif
    }

    /* Load images */
    image_t fixed, moving;
    if (image_load(&fixed, cfg.fixed_path, DEVICE_CPU) != 0) {
        fprintf(stderr, "Failed to load fixed image: %s\n", cfg.fixed_path); return 1;
    }
    if (image_load(&moving, cfg.moving_path, DEVICE_CPU) != 0) {
        fprintf(stderr, "Failed to load moving image: %s\n", cfg.moving_path);
        image_free(&fixed); return 1;
    }

    int fD = fixed.data.shape[2], fH = fixed.data.shape[3], fW = fixed.data.shape[4];
    if (cfireants_verbose >= 1) {
        fprintf(stderr, "Fixed:  %s [%d x %d x %d]\n", cfg.fixed_path, fD, fH, fW);
        fprintf(stderr, "Moving: %s [%d x %d x %d]\n", cfg.moving_path,
                moving.data.shape[2], moving.data.shape[3], moving.data.shape[4]);
        fprintf(stderr, "Backend: %s\n",
                cfg.backend == BACKEND_METAL ? "Metal" :
                cfg.backend == BACKEND_WEBGPU ? "WebGPU" :
                cfg.backend == BACKEND_CUDA ? "CUDA" : "CPU");
        fprintf(stderr, "Stages: %d\n", cfg.n_stages);
    }

    double t_total = get_time();

    /* Moments initialization */
    moments_result_t mom;
    memset(&mom, 0, sizeof(mom));
    mom.Rf[0][0] = mom.Rf[1][1] = mom.Rf[2][2] = 1.0f;
    /* Identity affine [3x4] for --no-moments case */
    mom.affine[0][0] = mom.affine[1][1] = mom.affine[2][2] = 1.0f;

    if (cfg.initial_moving_transform) {
        double t0 = get_time();
        moments_opts_t mopts = moments_opts_default();
        moments_register(&fixed, &moving, mopts, &mom);
        if (cfireants_verbose >= 1)
            fprintf(stderr, "Moments: %.1fs (NCC %.4f)\n", get_time() - t0, mom.ncc_loss);
    }

    /* Run stages */
    float rigid_mat[3][4] = {{0}};
    memcpy(rigid_mat, mom.affine, sizeof(rigid_mat));

    float affine_mat[3][4] = {{0}};
    memcpy(affine_mat, rigid_mat, sizeof(affine_mat));

    float affine_44[4][4] = {{0}};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            affine_44[i][j] = affine_mat[i][j];
    affine_44[3][3] = 1.0f;

    tensor_t final_moved = {0};
    (void)0; /* final_ncc removed — NCC printed per-stage */

    for (int si = 0; si < cfg.n_stages; si++) {
        stage_config_t *s = &cfg.stages[si];
        double t0 = get_time();

        if (cfireants_verbose >= 1) fprintf(stderr, "\n--- Stage %d: %s (lr=%.4f, %s%s%d, %d levels) ---\n",
                si + 1, stage_name(s->type), s->lr,
                s->metric_type == LOSS_MI ? "MI bins=" : "CC k=",
                "", s->metric_param, s->n_levels);

        if (s->type == STAGE_RIGID) {
            rigid_opts_t opts = {
                .n_scales = s->n_levels,
                .scales = s->shrink_factors,
                .iterations = s->iterations,
                .lr = s->lr,
                .loss_type = s->metric_type,
                .cc_kernel_size = (s->metric_type == LOSS_CC) ? s->metric_param : 5,
                .mi_num_bins = (s->metric_type == LOSS_MI) ? s->metric_param : 32,
                .tolerance = s->tolerance,
                .max_tolerance_iters = s->tolerance_window,
                .downsample_mode = cfg.downsample_mode,
            };
            rigid_result_t result;
            int rc;
            switch (cfg.backend) {
#ifdef CFIREANTS_HAS_CUDA
                case BACKEND_CUDA: rc = rigid_register_gpu(&fixed, &moving, &mom, opts, &result); break;
#endif
#ifdef CFIREANTS_HAS_METAL
                case BACKEND_METAL: rc = rigid_register_metal(&fixed, &moving, &mom, opts, &result); break;
#endif
#ifdef CFIREANTS_HAS_WEBGPU
                case BACKEND_WEBGPU: rc = rigid_register_webgpu(&fixed, &moving, &mom, opts, &result); break;
#endif
                default: rc = rigid_register(&fixed, &moving, &mom, opts, &result); break;
            }
            (void)rc;
            memcpy(rigid_mat, result.rigid_mat, sizeof(rigid_mat));
            memcpy(affine_mat, result.rigid_mat, sizeof(affine_mat));
            if (cfireants_verbose >= 1) fprintf(stderr, "  %s: %.1fs (NCC %.4f)\n", stage_name(s->type), get_time() - t0, result.ncc_loss);

        } else if (s->type == STAGE_AFFINE) {
            affine_opts_t opts = {
                .n_scales = s->n_levels,
                .scales = s->shrink_factors,
                .iterations = s->iterations,
                .lr = s->lr,
                .loss_type = s->metric_type,
                .cc_kernel_size = (s->metric_type == LOSS_CC) ? s->metric_param : 5,
                .mi_num_bins = (s->metric_type == LOSS_MI) ? s->metric_param : 32,
                .tolerance = s->tolerance,
                .max_tolerance_iters = s->tolerance_window,
                .downsample_mode = cfg.downsample_mode,
            };
            affine_result_t result;
            int rc;
            switch (cfg.backend) {
#ifdef CFIREANTS_HAS_CUDA
                case BACKEND_CUDA: rc = affine_register_gpu(&fixed, &moving, rigid_mat, opts, &result); break;
#endif
#ifdef CFIREANTS_HAS_METAL
                case BACKEND_METAL: rc = affine_register_metal(&fixed, &moving, rigid_mat, opts, &result); break;
#endif
#ifdef CFIREANTS_HAS_WEBGPU
                case BACKEND_WEBGPU: rc = affine_register_webgpu(&fixed, &moving, rigid_mat, opts, &result); break;
#endif
                default: rc = affine_register(&fixed, &moving, rigid_mat, opts, &result); break;
            }
            (void)rc;
            memcpy(affine_mat, result.affine_mat, sizeof(affine_mat));
            /* Update 4x4 */
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    affine_44[i][j] = affine_mat[i][j];
            affine_44[3][3] = 1.0f;
            if (cfireants_verbose >= 1) fprintf(stderr, "  %s: %.1fs (NCC %.4f)\n", stage_name(s->type), get_time() - t0, result.ncc_loss);

        } else if (s->type == STAGE_SYN) {
            syn_opts_t opts = {
                .n_scales = s->n_levels,
                .scales = s->shrink_factors,
                .iterations = s->iterations,
                .cc_kernel_size = s->metric_param,
                .lr = s->lr,
                .smooth_warp_sigma = s->syn_warp_sigma,
                .smooth_grad_sigma = s->syn_grad_sigma,
                .tolerance = s->tolerance,
                .max_tolerance_iters = s->tolerance_window,
                .downsample_mode = cfg.downsample_mode,
            };
            syn_result_t result;
            int rc;
            switch (cfg.backend) {
#ifdef CFIREANTS_HAS_CUDA
                case BACKEND_CUDA: rc = syn_register_gpu(&fixed, &moving, affine_44, opts, &result); break;
#endif
#ifdef CFIREANTS_HAS_METAL
                case BACKEND_METAL: rc = syn_register_metal(&fixed, &moving, affine_44, opts, &result); break;
#endif
#ifdef CFIREANTS_HAS_WEBGPU
                case BACKEND_WEBGPU: rc = syn_register_webgpu(&fixed, &moving, affine_44, opts, &result); break;
#endif
                default: rc = syn_register(&fixed, &moving, affine_44, opts, &result); break;
            }
            (void)rc;
            if (final_moved.data) tensor_free(&final_moved);
            final_moved = result.moved;
            /* NCC printed per-stage above */
            if (cfireants_verbose >= 1) fprintf(stderr, "  %s: %.1fs (NCC %.4f)\n", stage_name(s->type), get_time() - t0, result.ncc_loss);

        } else if (s->type == STAGE_GREEDY) {
            greedy_opts_t opts = {
                .n_scales = s->n_levels,
                .scales = s->shrink_factors,
                .iterations = s->iterations,
                .cc_kernel_size = s->metric_param,
                .lr = s->lr,
                .smooth_warp_sigma = s->syn_warp_sigma,
                .smooth_grad_sigma = s->syn_grad_sigma,
                .tolerance = s->tolerance,
                .max_tolerance_iters = s->tolerance_window,
                .downsample_mode = cfg.downsample_mode,
            };
            greedy_result_t result;
            int rc;
            switch (cfg.backend) {
#ifdef CFIREANTS_HAS_CUDA
                case BACKEND_CUDA: rc = greedy_register_gpu(&fixed, &moving, affine_44, opts, &result); break;
#endif
#ifdef CFIREANTS_HAS_METAL
                case BACKEND_METAL: rc = greedy_register_metal(&fixed, &moving, affine_44, opts, &result); break;
#endif
#ifdef CFIREANTS_HAS_WEBGPU
                case BACKEND_WEBGPU: rc = greedy_register_webgpu(&fixed, &moving, affine_44, opts, &result); break;
#endif
                default: rc = greedy_register(&fixed, &moving, affine_44, opts, &result); break;
            }
            (void)rc;
            if (final_moved.data) tensor_free(&final_moved);
            final_moved = result.moved;
            /* NCC printed per-stage above */
            if (cfireants_verbose >= 1) fprintf(stderr, "  %s: %.1fs (NCC %.4f)\n", stage_name(s->type), get_time() - t0, result.ncc_loss);
        }
    }

    if (cfireants_verbose >= 1) fprintf(stderr, "\nTotal: %.1fs\n", get_time() - t_total);

    /* Skullstrip mode: -o is the output filename, only produce skull-stripped image */
    if (cfg.skullstrip_mask) {
        image_t mask_img;
        if (image_load(&mask_img, cfg.skullstrip_mask, DEVICE_CPU) != 0) {
            fprintf(stderr, "Failed to load mask: %s\n", cfg.skullstrip_mask);
        } else {
            mat44d pd2, tm2, cb2;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) pd2.m[i][j] = affine_44[i][j];
            mat44d_mul(&tm2, &pd2, &fixed.meta.torch2phy);
            mat44d_mul(&cb2, &mask_img.meta.phy2torch, &tm2);
            float ha2[12]; for (int i=0;i<3;i++) for(int j=0;j<4;j++) ha2[i*4+j]=(float)cb2.m[i][j];

            tensor_t aff2; int as2[3]={1,3,4};
            tensor_alloc(&aff2,3,as2,DTYPE_FLOAT32,DEVICE_CPU);
            memcpy(tensor_data_f32(&aff2), ha2, 48);
            int fH = fixed.data.shape[3], fW = fixed.data.shape[4];
            int gs2[3]={fD,fH,fW};
            tensor_t grid2; affine_grid_3d(&aff2, gs2, &grid2);
            tensor_t warped_mask; cpu_grid_sample_3d_forward(&mask_img.data, &grid2, &warped_mask, 1);

            int fN = fD * fH * fW;
            const float *mask_data = tensor_data_f32(&warped_mask);

            /* -o is the skull-stripped output file */
            const char *out = cfg.output_prefix;
            if (ensure_parent_dir(out) != 0) { image_free(&mask_img); goto cleanup; }
            if (cfireants_verbose >= 1) fprintf(stderr, "Saving: %s (native datatype, threshold=%.2f)\n",
                    out, cfg.skullstrip_threshold);
            image_skullstrip_save(out, cfg.fixed_path,
                                   mask_data, cfg.skullstrip_threshold, fN);

            tensor_free(&aff2); tensor_free(&grid2);
            tensor_free(&warped_mask);
            image_free(&mask_img);
        }
        if (final_moved.data) tensor_free(&final_moved);
    } else if (final_moved.data) {
        /* Deformable output: save warped moving image */
        char warped_path[512];
        if (cfg.output_warped) {
            snprintf(warped_path, sizeof(warped_path), "%s", cfg.output_warped);
        } else {
            snprintf(warped_path, sizeof(warped_path), "%sWarped.nii.gz", cfg.output_prefix);
        }
        if (ensure_parent_dir(warped_path) != 0) { tensor_free(&final_moved); goto cleanup; }
        if (cfireants_verbose >= 1) fprintf(stderr, "Saving: %s\n", warped_path);
        int fN = fD * fixed.data.shape[3] * fixed.data.shape[4];
        image_save_like(warped_path, cfg.fixed_path, tensor_data_f32(&final_moved), fN);
        tensor_free(&final_moved);
    } else {
        /* Affine-only output: resample moving with affine and save */
        mat44d pd, tm, cb;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) pd.m[i][j] = affine_44[i][j];
        mat44d_mul(&tm, &pd, &fixed.meta.torch2phy);
        mat44d_mul(&cb, &moving.meta.phy2torch, &tm);
        float ha[12]; for (int i=0;i<3;i++) for(int j=0;j<4;j++) ha[i*4+j]=(float)cb.m[i][j];

        tensor_t aff_t; int as[3]={1,3,4};
        tensor_alloc(&aff_t,3,as,DTYPE_FLOAT32,DEVICE_CPU);
        memcpy(tensor_data_f32(&aff_t), ha, 48);
        int gs[3]={fD,fixed.data.shape[3],fixed.data.shape[4]};
        tensor_t grid; affine_grid_3d(&aff_t, gs, &grid);
        tensor_t moved; cpu_grid_sample_3d_forward(&moving.data, &grid, &moved, 1);

        char warped_path[512];
        if (cfg.output_warped) snprintf(warped_path,sizeof(warped_path),"%s",cfg.output_warped);
        else snprintf(warped_path,sizeof(warped_path),"%sWarped.nii.gz",cfg.output_prefix);
        if (ensure_parent_dir(warped_path) != 0) {
            tensor_free(&aff_t); tensor_free(&grid); tensor_free(&moved); goto cleanup;
        }
        if (cfireants_verbose >= 1) fprintf(stderr, "Saving: %s\n", warped_path);
        image_save_like(warped_path, cfg.fixed_path, tensor_data_f32(&moved),
                         fD * fixed.data.shape[3] * fixed.data.shape[4]);

        tensor_free(&aff_t); tensor_free(&grid); tensor_free(&moved);
    }

cleanup:
    image_free(&fixed);
    image_free(&moving);
    return 0;
}
