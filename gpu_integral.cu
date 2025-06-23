#include "gpu_integral.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define THREADS_PER_BLOCK 256

__global__ void exponentialIntegralKernelFloat(int n, const float* x, float* results, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float xi = x[idx];
    float sum = 0.0f;
    int maxIterations = 1000000;
    float t_max = 100.0f;
    float dt = (t_max - 1.0f) / maxIterations;

    for (int i = 0; i < maxIterations; ++i) {
        float t1 = 1.0f + i * dt;
        float t2 = t1 + dt;
        float f1 = expf(-xi * t1) / powf(t1, n);
        float f2 = expf(-xi * t2) / powf(t2, n);
        sum += 0.5f * (f1 + f2) * dt;
    }
    results[idx] = sum;
}

__global__ void exponentialIntegralKernelDouble(int n, const double* x, double* results, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    double xi = x[idx];
    double sum = 0.0;
    int maxIterations = 1000000;
    double t_max = 100.0;
    double dt = (t_max - 1.0) / maxIterations;

    for (int i = 0; i < maxIterations; ++i) {
        double t1 = 1.0 + i * dt;
        double t2 = t1 + dt;
        double f1 = exp(-xi * t1) / pow(t1, n);
        double f2 = exp(-xi * t2) / pow(t2, n);
        sum += 0.5 * (f1 + f2) * dt;
    }
    results[idx] = sum;
}

void computeExponentialIntegralFloatGPU(int n, const std::vector<float>& xValues, std::vector<float>& results) {
    int size = xValues.size();
    results.resize(size);

    float *d_x, *d_results;
    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_results, size * sizeof(float));

    cudaMemcpy(d_x, xValues.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    exponentialIntegralKernelFloat<<<blocks, THREADS_PER_BLOCK>>>(n, d_x, d_results, size);

    cudaMemcpy(results.data(), d_results, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_results);
}

void computeExponentialIntegralDoubleGPU(int n, const std::vector<double>& xValues, std::vector<double>& results) {
    int size = xValues.size();
    results.resize(size);

    double *d_x, *d_results;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_results, size * sizeof(double));

    cudaMemcpy(d_x, xValues.data(), size * sizeof(double), cudaMemcpyHostToDevice);

    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    exponentialIntegralKernelDouble<<<blocks, THREADS_PER_BLOCK>>>(n, d_x, d_results, size);

    cudaMemcpy(results.data(), d_results, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_results);
}

