/*
 * test_nifti_load.c - Test NIfTI loading and coordinate transforms
 *
 * Loads each validation dataset and prints shape, spacing, origin, and
 * coordinate transform matrices. Output is compared against Python to
 * ensure exact agreement.
 *
 * Usage: test_nifti_load <nifti_file> [<nifti_file2> ...]
 *    or: test_nifti_load --all   (loads all validation images)
 */

#include "cfireants/tensor.h"
#include "cfireants/image.h"
#include "cfireants/backend.h"
#include <stdio.h>
#include <string.h>

static void print_image_info(const char *path) {
    image_t img;

    printf("=== %s ===\n", path);

    if (image_load(&img, path, DEVICE_CPU) != 0) {
        printf("  FAILED to load\n\n");
        return;
    }

    printf("  dims: %d\n", img.meta.dims);
    printf("  size: [%d, %d, %d]\n",
           img.meta.size[0], img.meta.size[1], img.meta.size[2]);
    printf("  spacing: [%.10g, %.10g, %.10g]\n",
           img.meta.spacing[0], img.meta.spacing[1], img.meta.spacing[2]);
    printf("  origin: [%.10g, %.10g, %.10g]\n",
           img.meta.origin[0], img.meta.origin[1], img.meta.origin[2]);
    printf("  tensor shape: [");
    for (int d = 0; d < img.data.ndim; d++) {
        printf("%d", img.data.shape[d]);
        if (d < img.data.ndim - 1) printf(", ");
    }
    printf("]\n");
    printf("  tensor numel: %zu\n", img.data.numel);

    /* Print intensity range */
    const float *data = tensor_data_f32(&img.data);
    float vmin = data[0], vmax = data[0];
    for (size_t i = 1; i < img.data.numel; i++) {
        if (data[i] < vmin) vmin = data[i];
        if (data[i] > vmax) vmax = data[i];
    }
    printf("  intensity: [%.4f, %.4f]\n", vmin, vmax);

    /* Print coordinate transform matrices */
    printf("  torch2phy:\n");
    for (int i = 0; i < 4; i++)
        printf("    [%12.6f %12.6f %12.6f %12.6f]\n",
               img.meta.torch2phy.m[i][0], img.meta.torch2phy.m[i][1],
               img.meta.torch2phy.m[i][2], img.meta.torch2phy.m[i][3]);

    printf("  phy2torch:\n");
    for (int i = 0; i < 4; i++)
        printf("    [%12.6f %12.6f %12.6f %12.6f]\n",
               img.meta.phy2torch.m[i][0], img.meta.phy2torch.m[i][1],
               img.meta.phy2torch.m[i][2], img.meta.phy2torch.m[i][3]);

    printf("\n");
    image_free(&img);
}

/* Validation image paths (relative to repo root) */
static const char *validation_images[] = {
    "validate/small/MNI152_T1_2mm.nii.gz",
    "validate/small/T1_head_2mm.nii.gz",
    "validate/medium/MNI152_T1_1mm_brain.nii.gz",
    "validate/medium/t1_brain.nii.gz",
    "validate/large/MNI152_T1_1mm.nii.gz",
    "validate/large/chris_t1.nii.gz",
    NULL
};

int main(int argc, char **argv) {
    /* Initialize CPU backend */
    cfireants_init_cpu();

    if (argc > 1 && strcmp(argv[1], "--all") == 0) {
        /* Load all validation images */
        for (int i = 0; validation_images[i]; i++)
            print_image_info(validation_images[i]);
    } else if (argc > 1) {
        /* Load specified files */
        for (int i = 1; i < argc; i++)
            print_image_info(argv[i]);
    } else {
        printf("Usage: %s <nifti_file> [<nifti_file2> ...]\n", argv[0]);
        printf("       %s --all  (loads all validation images)\n", argv[0]);
        return 1;
    }

    cfireants_cleanup();
    return 0;
}
