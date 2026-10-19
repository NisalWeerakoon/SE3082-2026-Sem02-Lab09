#include "cuda_utils.cuh"

#define N 10000000
#define THREADS_PER_BLOCK 512

__global__ void vec_mul(const int *a, const int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] * b[i];
}

int main() {
    const size_t bytes = (size_t)N * sizeof(int);
    int *a = (int *)host_alloc(bytes), *b = (int *)host_alloc(bytes);
    int *c = (int *)host_alloc(bytes);
    random_ints(a, N);
    random_ints(b, N);
    int *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, bytes));
    CHECK(cudaMalloc((void **)&d_b, bytes));
    CHECK(cudaMalloc((void **)&d_c, bytes));
    CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching %d blocks x %d threads\n", blocks, THREADS_PER_BLOCK);
    vec_mul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = N - 1000; i < N; ++i)
        printf("%d) %d x %d = %d\n", i, a[i], b[i], c[i]);
    int errors = 0;
    for (int i = 0; i < N; ++i)
        errors += c[i] != a[i] * b[i];
    printf("Verification: %d mismatches out of %d\n", errors, N);

    free(a); free(b); free(c);
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));
    return errors ? 1 : 0;
}
