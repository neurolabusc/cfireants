/*
 * image.h - Medical image loading and coordinate transforms for cfireants
 *
 * Loads NIfTI images into tensors and computes the coordinate transform
 * matrices (torch2phy, phy2torch) that map between normalized [-1,1]
 * coordinates (used by grid_sample) and physical (mm) coordinates.
 *
 * These matrices are critical for multi-image registration where images
 * have different voxel sizes, orientations, and origins.
 */

#ifndef CFIREANTS_IMAGE_H
#define CFIREANTS_IMAGE_H

#include "cfireants/tensor.h"
#include "cfireants/nifti_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 4x4 matrix (row-major, doubles for precision in coordinate transforms) */
typedef struct {
    double m[4][4];
} mat44d;

/* Image metadata extracted from NIfTI header */
typedef struct {
    int    dims;             /* 2 or 3 */
    int    size[3];          /* nx, ny, nz (voxel counts) */
    double spacing[3];       /* dx, dy, dz (mm) */
    double origin[3];        /* physical origin (mm) */
    double direction[3][3];  /* direction cosines */
    mat44d px2phy;           /* pixel index → physical coordinates */
    mat44d torch2px;         /* normalized [-1,1] → pixel index */
    mat44d torch2phy;        /* normalized → physical (px2phy @ torch2px) */
    mat44d phy2torch;        /* physical → normalized (inverse of torch2phy) */
    int    nifti_datatype;   /* original NIfTI datatype (DT_FLOAT32, DT_UINT16, etc.) */
    int    nifti_nbyper;     /* bytes per voxel in original format */
    float  scl_slope;        /* NIfTI intensity scaling: val = slope * raw + inter */
    float  scl_inter;
} image_meta_t;

/* Loaded image: tensor data + spatial metadata */
typedef struct {
    tensor_t data;           /* [1, C, D, H, W] for 3D, [1, C, H, W] for 2D */
    image_meta_t meta;       /* Coordinate transforms and spatial info */
} image_t;

/* --- Loading --- */

/* Load a NIfTI file into an image_t.
   The data tensor will be float32 on the specified device.
   Returns 0 on success. */
int image_load(image_t *img, const char *path, int device);

/* Free image data and reset */
void image_free(image_t *img);

/* --- Coordinate transform utilities --- */

/* Compute image metadata (px2phy, torch2px, torch2phy, phy2torch) from
   NIfTI header fields. Called internally by image_load. */
void image_compute_meta(image_meta_t *meta, const nifti_image *nim);

/* Convert image_meta_t matrices to float32 tensors [1, 4, 4] for use
   in registration. Allocates output tensors on specified device. */
int image_meta_to_tensors(const image_meta_t *meta,
                          tensor_t *torch2phy_out,
                          tensor_t *phy2torch_out,
                          int device);

/* --- Matrix utilities --- */

/* Initialize to identity */
void mat44d_identity(mat44d *m);

/* Matrix multiply: c = a * b */
void mat44d_mul(mat44d *c, const mat44d *a, const mat44d *b);

/* Matrix inverse (full 4x4). Returns 0 on success, -1 if singular. */
int mat44d_inverse(mat44d *inv, const mat44d *m);

/* Print matrix to stderr */
void mat44d_print(const mat44d *m, const char *name);

/* --- Save --- */

/* Save a tensor as a NIfTI file using metadata from an image.
   The data tensor shape should be [1, C, D, H, W].
   Returns 0 on success. */
int image_save(const char *path, const tensor_t *data, const image_meta_t *meta);

/* Save float32 tensor using the NIfTI header from an existing file.
 * Clones sform, qform, pixdim, units, codes from ref_path exactly.
 * Data is written as float32. Dimensions must match. */
int image_save_like(const char *out_path, const char *ref_path,
                     const float *data, int nvox);

/* Apply a float32 mask to a NIfTI image at its native datatype.
 * Loads the source NIfTI, applies mask (threshold at thresh), sets
 * masked voxels to the minimum intensity, saves result.
 * Preserves original datatype (UINT16, INT16, FLOAT32, etc.). */
int image_skullstrip_save(const char *out_path, const char *src_path,
                           const float *mask, float thresh, int nvox);

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_IMAGE_H */
