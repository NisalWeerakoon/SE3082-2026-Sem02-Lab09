#include "cuda_utils.cuh"

#define N 512

__global__ void add_blocks(const int *a, const int *b, int *c) {
    int i = blockIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    const size_t bytes = N * sizeof(int);
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
    add_blocks<<<N, 1>>>(d_a, d_b, d_c);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
        errors += c[i] != a[i] + b[i];
    }
    printf("Verification: %d mismatches out of %d\n", errors, N);
    free(a); free(b); free(c);
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));
    return errors ? 1 : 0;
}
