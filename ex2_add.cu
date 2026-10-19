#include "cuda_utils.cuh"

__global__ void add(const int *a, const int *b, int *c) {
    *c = *a + *b;
}

int main() {
    const int a = 2, b = 7;
    int c = 0;
    int *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, sizeof(int)));
    CHECK(cudaMalloc((void **)&d_b, sizeof(int)));
    CHECK(cudaMalloc((void **)&d_c, sizeof(int)));
    CHECK(cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice));
    add<<<1, 1>>>(d_a, d_b, d_c);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Result is %d\n", c);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    return c == 9 ? 0 : 1;
}
