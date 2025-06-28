#include "gpu_launcher.h"
#include "gpu_integral.h"
#include <cuda_runtime.h>
#include <iostream>

void launchFloatKernel(int* h_n_vals, float* h_m_vals, float* h_results, int total) {
    int* d_n_vals;
    float* d_m_vals;
    float* d_results;

    cudaMalloc(&d_n_vals, sizeof(int) * total);
    cudaMalloc(&d_m_vals, sizeof(float) * total);
    cudaMalloc(&d_results, sizeof(float) * total);

    cudaMemcpy(d_n_vals, h_n_vals, sizeof(int) * total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m_vals, h_m_vals, sizeof(float) * total, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    computeExpIntegralKernel<<<gridSize, blockSize>>>(d_results, d_n_vals, d_m_vals, total);

    cudaMemcpy(h_results, d_results, sizeof(float) * total, cudaMemcpyDeviceToHost);

    cudaFree(d_n_vals);
    cudaFree(d_m_vals);
    cudaFree(d_results);
}

