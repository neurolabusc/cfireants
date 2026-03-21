/*
 * tensor.h - Minimal tensor data structure for cfireants
 *
 * Supports CPU and GPU (CUDA) memory with basic allocation, reshaping,
 * and element access. Designed for N-dimensional float data typical
 * in medical image registration (images, warp fields, affine matrices).
 */

#ifndef CFIREANTS_TENSOR_H
#define CFIREANTS_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of dimensions (B, C, D, H, W + extras) */
#define TENSOR_MAX_DIMS 8

/* Data types */
#define DTYPE_FLOAT32  0
#define DTYPE_FLOAT64  1
#define DTYPE_INT32    2
#define DTYPE_INT64    3
#define DTYPE_UINT8    4
#define DTYPE_INT16    5

/* Devices */
#define DEVICE_CPU    0
#define DEVICE_CUDA   1
#define DEVICE_WEBGPU 2
#define DEVICE_METAL  3

/* Tensor structure */
typedef struct {
    void   *data;                   /* Raw data pointer (CPU or GPU) */
    int     ndim;                   /* Number of dimensions */
    int     shape[TENSOR_MAX_DIMS]; /* Size along each dimension */
    int     strides[TENSOR_MAX_DIMS]; /* Stride (in elements) along each dim */
    int     dtype;                  /* DTYPE_* code */
    int     device;                 /* DEVICE_* code */
    int     owns_data;              /* 1 = tensor owns data and will free it */
    size_t  numel;                  /* Total number of elements */
} tensor_t;

/* --- Lifecycle --- */

/* Initialize a tensor to zeros (does not allocate data) */
void tensor_init(tensor_t *t);

/* Allocate a tensor with given shape on the specified device.
   Returns 0 on success, -1 on error. */
int tensor_alloc(tensor_t *t, int ndim, const int *shape, int dtype, int device);

/* Allocate a float32 tensor on CPU (convenience) */
int tensor_alloc_cpu_f32(tensor_t *t, int ndim, const int *shape);

/* Free tensor data (if owned). Resets to zero state. */
void tensor_free(tensor_t *t);

/* --- Properties --- */

/* Size of one element in bytes for a given dtype */
size_t dtype_size(int dtype);

/* Human-readable dtype name */
const char *dtype_name(int dtype);

/* Total size in bytes */
static inline size_t tensor_nbytes(const tensor_t *t) {
    return t->numel * dtype_size(t->dtype);
}

/* --- Element access (CPU only, bounds-unchecked) --- */

static inline float *tensor_data_f32(const tensor_t *t) {
    return (float *)t->data;
}

static inline double *tensor_data_f64(const tensor_t *t) {
    return (double *)t->data;
}

/* Linear index from N-dimensional indices.
   idx array must have t->ndim elements. */
static inline size_t tensor_offset(const tensor_t *t, const int *idx) {
    size_t off = 0;
    for (int d = 0; d < t->ndim; d++)
        off += (size_t)idx[d] * t->strides[d];
    return off;
}

/* Get float value at N-dimensional index (CPU, float32 only) */
static inline float tensor_get_f32(const tensor_t *t, const int *idx) {
    return ((float *)t->data)[tensor_offset(t, idx)];
}

/* Set float value at N-dimensional index (CPU, float32 only) */
static inline void tensor_set_f32(tensor_t *t, const int *idx, float val) {
    ((float *)t->data)[tensor_offset(t, idx)] = val;
}

/* --- Operations --- */

/* Fill tensor with a constant value (CPU only, float32) */
void tensor_fill_f32(tensor_t *t, float value);

/* Copy data between tensors (must have same numel and dtype).
   Handles CPU↔CPU. GPU transfers require backend. Returns 0 on success. */
int tensor_copy(tensor_t *dst, const tensor_t *src);

/* Create a view (shallow copy) of an existing tensor.
   The view does NOT own the data. */
void tensor_view(tensor_t *view, const tensor_t *src);

/* Reshape tensor in-place (must preserve numel). Returns 0 on success. */
int tensor_reshape(tensor_t *t, int ndim, const int *new_shape);

/* Compute contiguous strides for the current shape (row-major / C order) */
void tensor_compute_strides(tensor_t *t);

/* Print tensor info to stderr for debugging */
void tensor_info(const tensor_t *t, const char *name);

#ifdef __cplusplus
}
#endif

#endif /* CFIREANTS_TENSOR_H */
