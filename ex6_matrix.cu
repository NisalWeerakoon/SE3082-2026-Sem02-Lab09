#include "cuda_utils.cuh"

#define ROWS 10000
#define COLS 10000

/* Element-wise product, not matrix multiplication. */
__global__ void mul_elem(const int *a, const int *b, int *c,
                         int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        size_t idx = (size_t)row * cols + col;
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    const size_t count = (size_t)ROWS * COLS;
    const size_t bytes = count * sizeof(int);
    int *a = (int *)host_alloc(bytes), *b = (int *)host_alloc(bytes);
    int *c = (int *)host_alloc(bytes);
    random_ints(a, count);
    random_ints(b, count);
    int *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, bytes));
    CHECK(cudaMalloc((void **)&d_b, bytes));
    CHECK(cudaMalloc((void **)&d_c, bytes));
    CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    dim3 threads_per_block(32, 16);
    dim3 blocks((COLS + threads_per_block.x - 1) / threads_per_block.x,
                (ROWS + threads_per_block.y - 1) / threads_per_block.y);
    printf("Grid: %u x %u blocks\n", blocks.x, blocks.y);
    mul_elem<<<blocks, threads_per_block>>>(d_a, d_b, d_c, ROWS, COLS);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (size_t idx = count - 1000; idx < count; ++idx)
        printf("C[%zu][%zu] = %d x %d = %d\n",
               idx / COLS, idx % COLS, a[idx], b[idx], c[idx]);
    size_t errors = 0;
    for (size_t idx = 0; idx < count; ++idx)
        errors += c[idx] != a[idx] * b[idx];
    printf("Verification: %zu mismatches out of %zu\n", errors, count);

    free(a); free(b); free(c);
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));
    return errors ? 1 : 0;
}
