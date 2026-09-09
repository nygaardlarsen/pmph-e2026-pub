#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "helper.h"

#define GPU_RUNS 300

void vecAddCPU(float* A, float* B, float *C, unsigned int N) {

    for (unsigned int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vecAddGPU(float* A, float* B, float *C, unsigned int N) {

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    unsigned int N;
    
    { // reading the number of elements 
      if (argc != 2) { 
        printf("Num Args is: %d instead of 1. Exiting!\n", argc); 
        exit(1);
      }

      N = atoi(argv[1]);
      printf("N is: %d\n", N);

      const unsigned int maxN = 500000000;
      if(N > maxN) {
          printf("N is too big; maximal value is %d. Exiting!\n", maxN);
          exit(2);
      }
    }

    // use the first CUDA device:
    cudaSetDevice(0);

    unsigned int mem_size = N*sizeof(float);

    // allocate host memory
    float* h_A = (float*) malloc(mem_size);
    float* h_B = (float*) malloc(mem_size);
    float* h_cpu_C = (float*) malloc(mem_size);
    float* h_gpu_C = (float*) malloc(mem_size);

    // initialize the memory
    for(unsigned int i=0; i<N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    double cpu_elapsed;
    struct timeval cpu_start, cpu_end, cpu_diff;

    gettimeofday(&cpu_start, NULL);
    vecAddCPU(h_A, h_B, h_cpu_C, N);
    gettimeofday(&cpu_end, NULL);

    timeval_subtract(&cpu_diff, &cpu_end, &cpu_start);
    cpu_elapsed = cpu_diff.tv_sec * 1e6 + cpu_diff.tv_usec;

    printf("CPU runtime: %f microseconds\n", cpu_elapsed);

    // allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, mem_size);
    cudaMalloc((void**)&d_B, mem_size);
    cudaMalloc((void**)&d_C, mem_size);

    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);

    // define block size and grid size
    unsigned int block_size = 256;
    unsigned int grid_size = (N + block_size - 1) / block_size;

    // a small number of dry runs
    for(int r = 0; r < 1; r++) {
        vecAddGPU<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    
    double gpu_elapsed;

    { // execute the kernel a number of times;
      // to measure performance use a large N, e.g., 200000000,
      // and increase GPU_RUNS to 100 or more. 
     
        struct timeval gpu_start, gpu_end, gpu_diff;
        gettimeofday(&gpu_start, NULL);

        for(int r = 0; r < GPU_RUNS; r++) {
            vecAddGPU<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
        }

        cudaDeviceSynchronize();
        // ^ `cudaDeviceSynchronize` is needed for runtime
        //     measurements, since CUDA kernels are executed
        //     asynchronously, i.e., the CPU does not wait
        //     for the kernel to finish.
        //   However, `cudaDeviceSynchronize` is expensive
        //     so we need to amortize it across many runs;
        //     hence, when measuring performance use a big
        //     N and increase GPU_RUNS to 100 or more.
        //   Sure, it would be better by using CUDA events, but
        //     the current procedure is simple & works well enough.
        //   Please note that the execution of multiple
        //     kernels in Cuda executes correctly without such
        //     explicit synchronization; we need this only for
        //     runtime measurement.
        
        gettimeofday(&gpu_end, NULL);
        timeval_subtract(&gpu_diff, &gpu_end, &gpu_start);
        gpu_elapsed = (1.0 * (gpu_diff.tv_sec*1e6+gpu_diff.tv_usec)) / GPU_RUNS;
        double gigabytespersec = (3.0 * N * 4.0) / (gpu_elapsed * 1000.0);
        printf("The kernel took on average %f microseconds. GB/sec: %f \n", gpu_elapsed, gigabytespersec);
        
    }

    double speedup = cpu_elapsed / gpu_elapsed;
    printf("Speedup: %fx\n", speedup);
        
    // check for errors
    gpuAssert( cudaPeekAtLastError() );

    // copy result from ddevice to host
    cudaMemcpy(h_gpu_C, d_C, mem_size, cudaMemcpyDeviceToHost);

    // print result
    //for(unsigned int i=0; i<N; ++i) printf("%.6f\n", h_out[i]);

    double epsilon = 1e-6;
    for(unsigned int i=0; i<N; ++i) {
        float res_cpu   = h_cpu_C[i];
        float res_gpu = h_gpu_C[i]; 
        if(fabs(res_cpu - res_gpu) >= epsilon)  {
            printf("Invalid result at index %d, cpu: %f, gpu: %f. \n", i, res_cpu, res_gpu);
            exit(3);
        }
    }
    printf("Successful Validation.\n");

    // clean-up memory
    free(h_A);
    free(h_B);
    free(h_cpu_C);
    free(h_gpu_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
