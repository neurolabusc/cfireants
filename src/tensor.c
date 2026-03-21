/*
 * tensor.c - Tensor allocation and basic operations
 */

#include "cfireants/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef CFIREANTS_HAS_CUDA
/* Forward declarations for CUDA memory ops (implemented in backend_cuda.cu) */
int cuda_malloc(void **ptr, size_t nbytes);
void cuda_free(void *ptr);
int cuda_memcpy_h2d(void *dst, const void *src, size_t nbytes);
int cuda_memcpy_d2h(void *dst, const void *src, size_t nbytes);
int cuda_memcpy_d2d(void *dst, const void *src, size_t nbytes);
int cuda_memset(void *ptr, int value, size_t nbytes);
#endif

#ifdef CFIREANTS_HAS_WEBGPU
/* Forward declarations for WebGPU memory ops (implemented in backend_webgpu.c) */
typedef struct WGPUBufferImpl* WGPUBuffer;
int  webgpu_tensor_alloc(tensor_t *t, size_t nbytes);
void webgpu_tensor_free(WGPUBuffer buf);
int  webgpu_memcpy_h2d(WGPUBuffer dst, const void *src, size_t nbytes);
int  webgpu_memcpy_d2h(void *dst, WGPUBuffer src, size_t nbytes);
int  webgpu_memcpy_d2d(WGPUBuffer dst, WGPUBuffer src, size_t nbytes);
#endif

size_t dtype_size(int dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return sizeof(float);
        case DTYPE_FLOAT64: return sizeof(double);
        case DTYPE_INT32:   return sizeof(int32_t);
        case DTYPE_INT64:   return sizeof(int64_t);
        case DTYPE_UINT8:   return sizeof(uint8_t);
        case DTYPE_INT16:   return sizeof(int16_t);
        default:            return 0;
    }
}

const char *dtype_name(int dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return "float32";
        case DTYPE_FLOAT64: return "float64";
        case DTYPE_INT32:   return "int32";
        case DTYPE_INT64:   return "int64";
        case DTYPE_UINT8:   return "uint8";
        case DTYPE_INT16:   return "int16";
        default:            return "unknown";
    }
}

void tensor_init(tensor_t *t) {
    memset(t, 0, sizeof(tensor_t));
}

void tensor_compute_strides(tensor_t *t) {
    /* Row-major (C-order) strides */
    if (t->ndim <= 0) return;
    t->strides[t->ndim - 1] = 1;
    for (int d = t->ndim - 2; d >= 0; d--)
        t->strides[d] = t->strides[d + 1] * t->shape[d + 1];
}

int tensor_alloc(tensor_t *t, int ndim, const int *shape, int dtype, int device) {
    if (ndim <= 0 || ndim > TENSOR_MAX_DIMS) {
        fprintf(stderr, "tensor_alloc: invalid ndim=%d\n", ndim);
        return -1;
    }

    tensor_init(t);
    t->ndim = ndim;
    t->dtype = dtype;
    t->device = device;
    t->owns_data = 1;

    t->numel = 1;
    for (int d = 0; d < ndim; d++) {
        if (shape[d] <= 0) {
            fprintf(stderr, "tensor_alloc: invalid shape[%d]=%d\n", d, shape[d]);
            return -1;
        }
        t->shape[d] = shape[d];
        t->numel *= (size_t)shape[d];
    }
    tensor_compute_strides(t);

    size_t nbytes = t->numel * dtype_size(dtype);
    if (nbytes == 0) {
        fprintf(stderr, "tensor_alloc: zero-size tensor\n");
        return -1;
    }

    if (device == DEVICE_CPU) {
        t->data = calloc(t->numel, dtype_size(dtype));
        if (!t->data) {
            fprintf(stderr, "tensor_alloc: failed to allocate %zu bytes\n", nbytes);
            return -1;
        }
    }
#ifdef CFIREANTS_HAS_CUDA
    else if (device == DEVICE_CUDA) {
        if (cuda_malloc(&t->data, nbytes) != 0) {
            fprintf(stderr, "tensor_alloc: CUDA malloc failed for %zu bytes\n", nbytes);
            return -1;
        }
        cuda_memset(t->data, 0, nbytes);
    }
#endif
#ifdef CFIREANTS_HAS_WEBGPU
    else if (device == DEVICE_WEBGPU) {
        if (webgpu_tensor_alloc(t, nbytes) != 0) {
            fprintf(stderr, "tensor_alloc: WebGPU alloc failed for %zu bytes\n", nbytes);
            return -1;
        }
    }
#endif
    else {
        fprintf(stderr, "tensor_alloc: unsupported device=%d\n", device);
        return -1;
    }

    return 0;
}

int tensor_alloc_cpu_f32(tensor_t *t, int ndim, const int *shape) {
    return tensor_alloc(t, ndim, shape, DTYPE_FLOAT32, DEVICE_CPU);
}

void tensor_free(tensor_t *t) {
    if (t->data && t->owns_data) {
        if (t->device == DEVICE_CPU) {
            free(t->data);
        }
#ifdef CFIREANTS_HAS_CUDA
        else if (t->device == DEVICE_CUDA) {
            cuda_free(t->data);
        }
#endif
#ifdef CFIREANTS_HAS_WEBGPU
        else if (t->device == DEVICE_WEBGPU) {
            webgpu_tensor_free((WGPUBuffer)t->data);
        }
#endif
    }
    tensor_init(t);
}

void tensor_fill_f32(tensor_t *t, float value) {
    if (t->dtype != DTYPE_FLOAT32 || t->device != DEVICE_CPU) return;
    float *data = (float *)t->data;
    for (size_t i = 0; i < t->numel; i++)
        data[i] = value;
}

int tensor_copy(tensor_t *dst, const tensor_t *src) {
    if (dst->numel != src->numel || dst->dtype != src->dtype) {
        fprintf(stderr, "tensor_copy: shape/dtype mismatch\n");
        return -1;
    }
    size_t nbytes = src->numel * dtype_size(src->dtype);

    if (src->device == DEVICE_CPU && dst->device == DEVICE_CPU) {
        memcpy(dst->data, src->data, nbytes);
        return 0;
    }
#ifdef CFIREANTS_HAS_CUDA
    if (src->device == DEVICE_CPU && dst->device == DEVICE_CUDA) {
        return cuda_memcpy_h2d(dst->data, src->data, nbytes);
    }
    if (src->device == DEVICE_CUDA && dst->device == DEVICE_CPU) {
        return cuda_memcpy_d2h(dst->data, src->data, nbytes);
    }
    if (src->device == DEVICE_CUDA && dst->device == DEVICE_CUDA) {
        return cuda_memcpy_d2d(dst->data, src->data, nbytes);
    }
#endif
#ifdef CFIREANTS_HAS_WEBGPU
    if (src->device == DEVICE_CPU && dst->device == DEVICE_WEBGPU) {
        return webgpu_memcpy_h2d((WGPUBuffer)dst->data, src->data, nbytes);
    }
    if (src->device == DEVICE_WEBGPU && dst->device == DEVICE_CPU) {
        return webgpu_memcpy_d2h(dst->data, (WGPUBuffer)src->data, nbytes);
    }
    if (src->device == DEVICE_WEBGPU && dst->device == DEVICE_WEBGPU) {
        return webgpu_memcpy_d2d((WGPUBuffer)dst->data, (WGPUBuffer)src->data, nbytes);
    }
#endif
    fprintf(stderr, "tensor_copy: unsupported device combination\n");
    return -1;
}

void tensor_view(tensor_t *view, const tensor_t *src) {
    *view = *src;
    view->owns_data = 0;
}

int tensor_reshape(tensor_t *t, int ndim, const int *new_shape) {
    if (ndim <= 0 || ndim > TENSOR_MAX_DIMS) return -1;

    size_t new_numel = 1;
    for (int d = 0; d < ndim; d++)
        new_numel *= (size_t)new_shape[d];

    if (new_numel != t->numel) {
        fprintf(stderr, "tensor_reshape: numel mismatch (%zu vs %zu)\n",
                new_numel, t->numel);
        return -1;
    }

    t->ndim = ndim;
    for (int d = 0; d < ndim; d++)
        t->shape[d] = new_shape[d];
    tensor_compute_strides(t);
    return 0;
}

void tensor_info(const tensor_t *t, const char *name) {
    const char *dev_name = "unknown";
    if (t->device == DEVICE_CPU) dev_name = "cpu";
    else if (t->device == DEVICE_CUDA) dev_name = "cuda";
    else if (t->device == DEVICE_WEBGPU) dev_name = "webgpu";
    fprintf(stderr, "tensor '%s': dtype=%s device=%s shape=[",
            name ? name : "?",
            dtype_name(t->dtype),
            dev_name);
    for (int d = 0; d < t->ndim; d++) {
        fprintf(stderr, "%d", t->shape[d]);
        if (d < t->ndim - 1) fprintf(stderr, ", ");
    }
    fprintf(stderr, "] numel=%zu owns=%d data=%p\n",
            t->numel, t->owns_data, t->data);
}
