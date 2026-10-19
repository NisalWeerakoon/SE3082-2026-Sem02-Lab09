#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { \
    cudaError_t result = (call); \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(result)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static void *host_alloc(size_t bytes) {
    void *ptr = malloc(bytes);
    if (ptr == nullptr) {
        fprintf(stderr, "Host allocation failed for %zu bytes\n", bytes);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static void random_ints(int *values, size_t count) {
    for (size_t i = 0; i < count; ++i)
        values[i] = rand() % 100;
}

#endif
