#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "gpu_integral.h"

// CUDA kernel for float
__global__ void exponentialIntegralFloatKernel(int* ns, float* xs, float* results, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int n = ns[i];
        float x = xs[i];

        float result;
        if (n == 0) {
            result = expf(-x) / x;
        } else {
            float sum = 0.0f;
            for (int k = 1; k <= 100; ++k) {
                sum += powf(x, k - 1) / (tgammaf(k + n));
            }
            result = expf(-x) * sum;
        }

        results[i] = result;
    }
}

// CUDA kernel for double
__global__ void exponentialIntegralDoubleKernel(int* ns, double* xs, double* results, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int n = ns[i];
        double x = xs[i];

        double result;
        if (n == 0) {
            result = exp(-x) / x;
        } else {
            double sum = 0.0;
            for (int k = 1; k <= 100; ++k) {
                sum += pow(x, k - 1) / (tgamma(k + n));
            }
            result = exp(-x) * sum;
        }

        results[i] = result;
    }
}

// Launcher for float
void launchFloatKernel(int* ns_host, float* xs_host, float* results_host, int total) {
    int *ns_dev;
    float *xs_dev, *results_dev;

    cudaMalloc(&ns_dev, total * sizeof(int));
    cudaMalloc(&xs_dev, total * sizeof(float));
    cudaMalloc(&results_dev, total * sizeof(float));

    cudaMemcpy(ns_dev, ns_host, total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(xs_dev, xs_host, total * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((total + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    exponentialIntegralFloatKernel<<<gridSize, blockSize>>>(ns_dev, xs_dev, results_dev, total);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[GPU float] Execution time: " << milliseconds / 1000.0f << " seconds\n";

    cudaMemcpy(results_host, results_dev, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(ns_dev);
    cudaFree(xs_dev);
    cudaFree(results_dev);
}

// Launcher for double
void launchDoubleKernel(int* ns_host, double* xs_host, double* results_host, int total) {
    int *ns_dev;
    double *xs_dev, *results_dev;

    cudaMalloc(&ns_dev, total * sizeof(int));
    cudaMalloc(&xs_dev, total * sizeof(double));
    cudaMalloc(&results_dev, total * sizeof(double));

    cudaMemcpy(ns_dev, ns_host, total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(xs_dev, xs_host, total * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((total + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    exponentialIntegralDoubleKernel<<<gridSize, blockSize>>>(ns_dev, xs_dev, results_dev, total);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[GPU double] Execution time: " << milliseconds / 1000.0f << " seconds\n";

    cudaMemcpy(results_host, results_dev, total * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(ns_dev);
    cudaFree(xs_dev);
    cudaFree(results_dev);
}
