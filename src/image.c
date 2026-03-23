/*
 * image.c - NIfTI image loading and coordinate transform computation
 *
 * Replicates the coordinate transform logic from fireants/io/image.py:
 *   px2phy = direction * spacing matrix with origin
 *   torch2px = maps [-1,1] normalized coords to pixel indices
 *   torch2phy = px2phy @ torch2px
 *   phy2torch = inverse(torch2phy)
 */

#include "cfireants/image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Matrix utilities                                                    */
/* ------------------------------------------------------------------ */

void mat44d_identity(mat44d *m) {
    memset(m, 0, sizeof(mat44d));
    m->m[0][0] = m->m[1][1] = m->m[2][2] = m->m[3][3] = 1.0;
}

void mat44d_mul(mat44d *c, const mat44d *a, const mat44d *b) {
    mat44d tmp;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            tmp.m[i][j] = 0.0;
            for (int k = 0; k < 4; k++)
                tmp.m[i][j] += a->m[i][k] * b->m[k][j];
        }
    *c = tmp;
}

int mat44d_inverse(mat44d *inv, const mat44d *m) {
    /* Full 4x4 inverse via cofactor expansion */
    double a[16], b[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            a[i * 4 + j] = m->m[i][j];

    /* Compute cofactors */
    b[0]  =  a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
    b[1]  = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
    b[2]  =  a[1]*a[6]*a[15]  - a[1]*a[7]*a[14]  - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7]  - a[13]*a[3]*a[6];
    b[3]  = -a[1]*a[6]*a[11]  + a[1]*a[7]*a[10]  + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7]   + a[9]*a[3]*a[6];
    b[4]  = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
    b[5]  =  a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
    b[6]  = -a[0]*a[6]*a[15]  + a[0]*a[7]*a[14]  + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7]  + a[12]*a[3]*a[6];
    b[7]  =  a[0]*a[6]*a[11]  - a[0]*a[7]*a[10]  - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7]   - a[8]*a[3]*a[6];
    b[8]  =  a[4]*a[9]*a[15]  - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
    b[9]  = -a[0]*a[9]*a[15]  + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
    b[10] =  a[0]*a[5]*a[15]  - a[0]*a[7]*a[13]  - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7]  - a[12]*a[3]*a[5];
    b[11] = -a[0]*a[5]*a[11]  + a[0]*a[7]*a[9]   + a[4]*a[1]*a[11] - a[4]*a[3]*a[9]  - a[8]*a[1]*a[7]   + a[8]*a[3]*a[5];
    b[12] = -a[4]*a[9]*a[14]  + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
    b[13] =  a[0]*a[9]*a[14]  - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
    b[14] = -a[0]*a[5]*a[14]  + a[0]*a[6]*a[13]  + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6]  + a[12]*a[2]*a[5];
    b[15] =  a[0]*a[5]*a[10]  - a[0]*a[6]*a[9]   - a[4]*a[1]*a[10] + a[4]*a[2]*a[9]  + a[8]*a[1]*a[6]   - a[8]*a[2]*a[5];

    double det = a[0]*b[0] + a[1]*b[4] + a[2]*b[8] + a[3]*b[12];
    if (fabs(det) < 1e-15) {
        fprintf(stderr, "mat44d_inverse: singular matrix (det=%g)\n", det);
        return -1;
    }

    double inv_det = 1.0 / det;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            inv->m[i][j] = b[i * 4 + j] * inv_det;

    return 0;
}

void mat44d_print(const mat44d *m, const char *name) {
    fprintf(stderr, "%s:\n", name ? name : "mat44");
    for (int i = 0; i < 4; i++)
        fprintf(stderr, "  [%10.6f %10.6f %10.6f %10.6f]\n",
                m->m[i][0], m->m[i][1], m->m[i][2], m->m[i][3]);
}

/* ------------------------------------------------------------------ */
/* Image metadata computation                                          */
/* ------------------------------------------------------------------ */

void image_compute_meta(image_meta_t *meta, const nifti_image *nim) {
    int dims = (nim->nz <= 1) ? 2 : 3;
    meta->dims = dims;

    /* Image size (SimpleITK uses x,y,z = nx,ny,nz) */
    meta->size[0] = nim->nx;
    meta->size[1] = nim->ny;
    meta->size[2] = (dims == 3) ? nim->nz : 1;

    /* Spacing */
    meta->spacing[0] = nim->dx;
    meta->spacing[1] = nim->dy;
    meta->spacing[2] = (dims == 3) ? nim->dz : 1.0;

    /*
     * Extract origin and direction from the sform (sto_xyz) if available,
     * otherwise from qform (qto_xyz).
     *
     * SimpleITK convention (matching Python FireANTs):
     *   direction = columns of the rotation part of the affine, normalized by spacing
     *   origin = translation column
     *   px2phy[:3,:3] = direction * diag(spacing)
     *   px2phy[:3,3]  = origin
     *
     * NIfTI sform: sto_xyz is a 4x4 that maps (i,j,k) → (x,y,z) in RAS.
     * SimpleITK converts to LPS by negating X and Y:
     *   sitk_affine = diag(-1,-1,1,1) @ nifti_sform
     * Then: origin = sitk_affine[:3,3], direction*spacing = sitk_affine[:3,:3]
     *
     * We must match SimpleITK's convention since Python FireANTs uses it.
     */
    const nifti_dmat44 *aff;
    if (nim->sform_code > 0) {
        aff = &nim->sto_xyz;
    } else {
        aff = &nim->qto_xyz;
    }

    /* Apply RAS→LPS conversion (negate rows 0 and 1) to get SimpleITK coords.
     * lps_affine[i][j] = sign[i] * ras_affine[i][j]
     * where sign = {-1, -1, 1} for LPS
     */
    double lps_aff[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            lps_aff[i][j] = aff->m[i][j];
    /* Negate X row (row 0) and Y row (row 1) */
    for (int j = 0; j < 4; j++) {
        lps_aff[0][j] = -lps_aff[0][j];
        lps_aff[1][j] = -lps_aff[1][j];
    }

    /* Origin (from LPS affine) */
    meta->origin[0] = lps_aff[0][3];
    meta->origin[1] = lps_aff[1][3];
    meta->origin[2] = lps_aff[2][3];

    /* Direction: extract from LPS affine by dividing out spacing.
     * lps_aff[:3,:3] = direction * diag(spacing)
     * direction[:,j] = lps_aff[:,j] / spacing[j]
     */
    for (int j = 0; j < dims; j++) {
        double sp = meta->spacing[j];
        if (sp == 0.0) sp = 1.0;
        for (int i = 0; i < dims; i++)
            meta->direction[i][j] = lps_aff[i][j] / sp;
    }
    /* Fill unused entries for 2D */
    if (dims == 2) {
        meta->direction[2][0] = 0; meta->direction[2][1] = 0; meta->direction[2][2] = 1;
        meta->direction[0][2] = 0; meta->direction[1][2] = 0;
    }

    /*
     * Build px2phy: pixel index → physical coordinates
     * Matches Python: px2phy = eye(4); px2phy[:d,:d] = direction * spacing; px2phy[:d,3] = origin
     */
    mat44d_identity(&meta->px2phy);
    for (int i = 0; i < dims; i++) {
        for (int j = 0; j < dims; j++)
            meta->px2phy.m[i][j] = meta->direction[i][j] * meta->spacing[j];
        meta->px2phy.m[i][3] = meta->origin[i];
    }

    /*
     * Build torch2px: normalized [-1,1] → pixel indices
     * Matches Python:
     *   scaleterm = (np.array(size) - 1) * 0.5
     *   torch2px = eye(4)
     *   torch2px[:d,:d] = diag(scaleterm)
     *   torch2px[:d,3]  = scaleterm
     *
     * This maps: -1 → 0, +1 → size-1  (align_corners=True convention)
     */
    mat44d_identity(&meta->torch2px);
    for (int i = 0; i < dims; i++) {
        double scale = (meta->size[i] - 1) * 0.5;
        meta->torch2px.m[i][i] = scale;
        meta->torch2px.m[i][3] = scale;
    }

    /* torch2phy = px2phy @ torch2px */
    mat44d_mul(&meta->torch2phy, &meta->px2phy, &meta->torch2px);

    /* phy2torch = inverse(torch2phy) */
    if (mat44d_inverse(&meta->phy2torch, &meta->torch2phy) != 0) {
        fprintf(stderr, "image_compute_meta: singular torch2phy matrix\n");
    }
}

/* ------------------------------------------------------------------ */
/* NIfTI data type → float conversion                                  */
/* ------------------------------------------------------------------ */

static float *nii_to_float(const nifti_image *nim, size_t nvox) {
    float *out = (float *)malloc(nvox * sizeof(float));
    if (!out) return NULL;

    float slope = nim->scl_slope;
    float inter = nim->scl_inter;
    int apply_scaling = (slope != 0.0f && !(slope == 1.0f && inter == 0.0f));

    switch (nim->datatype) {
        case DT_UINT8: {
            const uint8_t *src = (const uint8_t *)nim->data;
            for (size_t i = 0; i < nvox; i++) out[i] = (float)src[i];
            break;
        }
        case DT_INT16: {
            const int16_t *src = (const int16_t *)nim->data;
            for (size_t i = 0; i < nvox; i++) out[i] = (float)src[i];
            break;
        }
        case DT_INT32: {
            const int32_t *src = (const int32_t *)nim->data;
            for (size_t i = 0; i < nvox; i++) out[i] = (float)src[i];
            break;
        }
        case DT_FLOAT32: {
            memcpy(out, nim->data, nvox * sizeof(float));
            break;
        }
        case DT_FLOAT64: {
            const double *src = (const double *)nim->data;
            for (size_t i = 0; i < nvox; i++) out[i] = (float)src[i];
            break;
        }
        case DT_INT8: {
            const int8_t *src = (const int8_t *)nim->data;
            for (size_t i = 0; i < nvox; i++) out[i] = (float)src[i];
            break;
        }
        case DT_UINT16: {
            const uint16_t *src = (const uint16_t *)nim->data;
            for (size_t i = 0; i < nvox; i++) out[i] = (float)src[i];
            break;
        }
        case DT_UINT32: {
            const uint32_t *src = (const uint32_t *)nim->data;
            for (size_t i = 0; i < nvox; i++) out[i] = (float)src[i];
            break;
        }
        default:
            fprintf(stderr, "nii_to_float: unsupported datatype %d\n", nim->datatype);
            free(out);
            return NULL;
    }

    if (apply_scaling) {
        for (size_t i = 0; i < nvox; i++)
            out[i] = out[i] * slope + inter;
    }

    return out;
}

/* ------------------------------------------------------------------ */
/* Image loading                                                       */
/* ------------------------------------------------------------------ */

int image_load(image_t *img, const char *path, int device) {
    memset(img, 0, sizeof(image_t));

    /* Read NIfTI */
    nifti_image *nim = nifti_image_read(path, 1);
    if (!nim) {
        fprintf(stderr, "image_load: failed to read '%s'\n", path);
        return -1;
    }

    /* Compute metadata */
    image_compute_meta(&img->meta, nim);

    /* Convert data to float32 */
    size_t nvox = (size_t)nim->nvox;
    float *fdata = nii_to_float(nim, nvox);
    if (!fdata) {
        nifti_image_free(nim);
        return -1;
    }

    /*
     * SimpleITK returns arrays in reverse order: GetArrayFromImage gives [Z, Y, X].
     * NIfTI stores data in [X, Y, Z] order (fastest to slowest: x varies fastest).
     * So the raw NIfTI data is already in the correct memory order for our
     * tensor shape [1, 1, nz, ny, nx] which has nx varying fastest.
     *
     * However, the Python code uses SimpleITK which reverses to [Z,Y,X],
     * then torch unsqueezes to [1, 1, Z, Y, X].
     *
     * Since NIfTI native order is X-fastest (i.e., memory layout is x,y,z),
     * and we want shape [1, 1, Z, Y, X] with X-fastest stride, the data
     * is already correct — we just label the dims accordingly.
     */
    int dims = img->meta.dims;
    int shape[5];
    if (dims == 3) {
        shape[0] = 1;                    /* batch */
        shape[1] = 1;                    /* channels */
        shape[2] = img->meta.size[2];    /* Z (nz) */
        shape[3] = img->meta.size[1];    /* Y (ny) */
        shape[4] = img->meta.size[0];    /* X (nx) */
    } else {
        shape[0] = 1;
        shape[1] = 1;
        shape[2] = img->meta.size[1];    /* Y */
        shape[3] = img->meta.size[0];    /* X */
    }
    int ndim = (dims == 3) ? 5 : 4;

    /* Allocate tensor on CPU first */
    if (tensor_alloc(&img->data, ndim, shape, DTYPE_FLOAT32, DEVICE_CPU) != 0) {
        free(fdata);
        nifti_image_free(nim);
        return -1;
    }

    /* Copy float data into tensor */
    memcpy(img->data.data, fdata, nvox * sizeof(float));
    free(fdata);

    /* If CUDA device requested, transfer */
    if (device == DEVICE_CUDA) {
#ifdef CFIREANTS_HAS_CUDA
        tensor_t gpu_tensor;
        if (tensor_alloc(&gpu_tensor, ndim, shape, DTYPE_FLOAT32, DEVICE_CUDA) != 0) {
            tensor_free(&img->data);
            nifti_image_free(nim);
            return -1;
        }
        if (tensor_copy(&gpu_tensor, &img->data) != 0) {
            tensor_free(&gpu_tensor);
            tensor_free(&img->data);
            nifti_image_free(nim);
            return -1;
        }
        tensor_free(&img->data);
        img->data = gpu_tensor;
#else
        fprintf(stderr, "image_load: CUDA requested but not compiled with CFIREANTS_HAS_CUDA\n");
        tensor_free(&img->data);
        nifti_image_free(nim);
        return -1;
#endif
    }

    nifti_image_free(nim);
    return 0;
}

void image_free(image_t *img) {
    tensor_free(&img->data);
    memset(img, 0, sizeof(image_t));
}

int image_meta_to_tensors(const image_meta_t *meta,
                          tensor_t *torch2phy_out,
                          tensor_t *phy2torch_out,
                          int device) {
    int shape[3] = {1, 4, 4};

    /* torch2phy [1, 4, 4] as float32 */
    if (tensor_alloc(torch2phy_out, 3, shape, DTYPE_FLOAT32, DEVICE_CPU) != 0)
        return -1;
    float *t2p = tensor_data_f32(torch2phy_out);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            t2p[i * 4 + j] = (float)meta->torch2phy.m[i][j];

    /* phy2torch [1, 4, 4] as float32 */
    if (tensor_alloc(phy2torch_out, 3, shape, DTYPE_FLOAT32, DEVICE_CPU) != 0) {
        tensor_free(torch2phy_out);
        return -1;
    }
    float *p2t = tensor_data_f32(phy2torch_out);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            p2t[i * 4 + j] = (float)meta->phy2torch.m[i][j];

    return 0;
}

/* ------------------------------------------------------------------ */
/* Image saving                                                        */
/* ------------------------------------------------------------------ */

int image_save(const char *path, const tensor_t *data, const image_meta_t *meta) {
    if (data->device != DEVICE_CPU) {
        fprintf(stderr, "image_save: tensor must be on CPU\n");
        return -1;
    }
    if (data->dtype != DTYPE_FLOAT32) {
        fprintf(stderr, "image_save: only float32 supported\n");
        return -1;
    }

    /* Create nifti_image */
    nifti_image nim;
    memset(&nim, 0, sizeof(nim));

    int dims = meta->dims;
    nim.ndim = dims;
    nim.nx = meta->size[0]; nim.dim[1] = nim.nx;
    nim.ny = meta->size[1]; nim.dim[2] = nim.ny;
    if (dims == 3) {
        nim.nz = meta->size[2]; nim.dim[3] = nim.nz;
    } else {
        nim.nz = 1; nim.dim[3] = 1;
    }
    nim.nt = 1; nim.dim[4] = 1;
    nim.nu = 1; nim.dim[5] = 1;
    nim.nv = 1; nim.dim[6] = 1;
    nim.nw = 1; nim.dim[7] = 1;
    nim.dim[0] = dims;
    nim.nvox = (size_t)nim.nx * nim.ny * nim.nz;

    nim.dx = (float)meta->spacing[0]; nim.pixdim[1] = nim.dx;
    nim.dy = (float)meta->spacing[1]; nim.pixdim[2] = nim.dy;
    nim.dz = (float)meta->spacing[2]; nim.pixdim[3] = nim.dz;
    nim.dt = 1.0f; nim.pixdim[4] = 1.0f;

    nim.datatype = DT_FLOAT32;
    nim.nbyper = 4;
    nim.scl_slope = 0.0f;
    nim.scl_inter = 0.0f;

    /* Set sform from px2phy */
    nim.sform_code = 1;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            nim.sto_xyz.m[i][j] = meta->px2phy.m[i][j];

    /* Compute sto_ijk = inverse of sto_xyz */
    mat44d sto_ijk;
    if (mat44d_inverse(&sto_ijk, &meta->px2phy) != 0) {
        fprintf(stderr, "image_save: singular px2phy matrix\n");
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            nim.sto_ijk.m[i][j] = sto_ijk.m[i][j];

    /* Set qform too (same as sform) */
    nim.qform_code = 1;
    nim.qto_xyz = nim.sto_xyz;
    nim.qto_ijk = nim.sto_ijk;

    /* Single-file NIfTI-1 format (.nii or .nii.gz) */
    nim.nifti_type = 1; /* NIFTI_FTYPE_NIFTI1_1 */
    nim.iname_offset = 352;

    /* Point data at tensor (we won't free it) */
    nim.data = data->data;
    nim.fname = strdup(path);
    nim.iname = strdup(path);

    nifti_image_write(&nim);

    free(nim.fname);
    free(nim.iname);
    nim.data = NULL; /* Don't let nifti_image_free touch our tensor data */

    return 0;
}
