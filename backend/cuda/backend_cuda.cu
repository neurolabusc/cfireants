/*
 * backend_cuda.cu - CUDA backend for cfireants
 *
 * Provides CUDA memory management and will host GPU-accelerated
 * operations in later phases.
 */

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

#include "cfireants/tensor.h"

/* --- CUDA memory management (called from tensor.c) --- */

int cuda_malloc(void **ptr, size_t nbytes) {
    cudaError_t err = cudaMalloc(ptr, nbytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_malloc: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

void cuda_free(void *ptr) {
    if (ptr) cudaFree(ptr);
}

int cuda_memcpy_h2d(void *dst, const void *src, size_t nbytes) {
    cudaError_t err = cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_memcpy_h2d: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int cuda_memcpy_d2h(void *dst, const void *src, size_t nbytes) {
    cudaError_t err = cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_memcpy_d2h: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int cuda_memcpy_d2d(void *dst, const void *src, size_t nbytes) {
    cudaError_t err = cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_memcpy_d2d: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int cuda_memset(void *ptr, int value, size_t nbytes) {
    cudaError_t err = cudaMemset(ptr, value, nbytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_memset: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/* --- Device info --- */

int cuda_get_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

void cuda_print_device_info(void) {
    int count = cuda_get_device_count();
    fprintf(stderr, "CUDA devices: %d\n", count);
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        fprintf(stderr, "  [%d] %s (%.0f MB, compute %d.%d)\n",
                i, prop.name,
                prop.totalGlobalMem / (1024.0 * 1024.0),
                prop.major, prop.minor);
    }
}

} /* extern "C" */
