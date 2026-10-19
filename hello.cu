#include "cuda_utils.cuh"

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 4>>>();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    return 0;
}
