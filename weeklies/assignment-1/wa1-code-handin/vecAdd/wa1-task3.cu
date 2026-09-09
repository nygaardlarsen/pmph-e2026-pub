#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define GPU_RUNS 300

void vecAddCPU(const float* A, const float* B, float* C, unsigned int N) {
    for (unsigned int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vecAddGPU(const float* A, const float* B, float* C, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s N\n", argv[0]);
        return 1;
    }

    unsigned int N = std::strtoul(argv[1], nullptr, 10);
    size_t bytes = (size_t)N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_cpu_C = (float*)malloc(bytes);
    float* h_gpu_C = (float*)malloc(bytes);

    for (unsigned int i = 0; i < N; i++) {
        h_A[i] = (float)(i % 1000) * 0.001f;
        h_B[i] = (float)(i % 500) * 0.002f;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    vecAddCPU(h_A, h_B, h_cpu_C, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_us = std::chrono::duration<double, std::micro>(cpu_end - cpu_start).count();

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    unsigned int block_size = 256;
    unsigned int grid_size = (N + block_size - 1) / block_size;

    auto gpu_start = std::chrono::high_resolution_clock::now();

    for (int r = 0; r < GPU_RUNS; r++) {
        vecAddGPU<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    }

    cudaDeviceSynchronize();

    auto gpu_end = std::chrono::high_resolution_clock::now();

    double gpu_total_us = std::chrono::duration<double, std::micro>(gpu_end - gpu_start).count();
    double gpu_us = gpu_total_us / GPU_RUNS;

    cudaMemcpy(h_gpu_C, d_C, bytes, cudaMemcpyDeviceToHost);

    bool valid = true;
    const float epsilon = 0.000001f;

    for (unsigned int i = 0; i < N; i++) {
        if (fabs(h_cpu_C[i] - h_gpu_C[i]) >= epsilon) {
            valid = false;
            break;
        }
    }

    double speedup = cpu_us / gpu_us;
    double throughput = (3.0 * bytes) / (gpu_us * 1000.0);

    printf("N: %u\n", N);
    printf("Validation: %s\n", valid ? "VALID" : "INVALID");
    printf("CPU runtime: %.3f us\n", cpu_us);
    printf("CUDA runtime: %.3f us\n", gpu_us);
    printf("Speedup: %.2fx\n", speedup);
    printf("Memory throughput: %.2f GB/s\n", throughput);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_cpu_C);
    free(h_gpu_C);

    return valid ? 0 : 1;
}
