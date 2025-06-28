#include <cuda_runtime.h>
#include <math.h>
#include "gpu_integral.h"

__device__ float expIntegralSeriesFloat(int n, float x) {
    if (n < 0 || x < 0.0f)
        return -1.0f;
    if (x == 0.0f)
        return (n == 0) ? 1.0f : 1.0f / (float)(n - 1);

    float sum = 0.0f;
    float term = 1.0f;
    for (int k = 1; k < 100; k++) {
        term *= -x / (float)(k + n);
        sum += term;
        if (fabs(term) < 1e-6f)
            break;
    }
    return expf(-x) * (1.0f / (float)n + sum);
}

__device__ double expIntegralSeriesDouble(int n, double x) {
    if (n < 0 || x < 0.0)
        return -1.0;
    if (x == 0.0)
        return (n == 0) ? 1.0 : 1.0 / (double)(n - 1);

    double sum = 0.0;
    double term = 1.0;
    for (int k = 1; k < 100; k++) {
        term *= -x / (double)(k + n);
        sum += term;
        if (fabs(term) < 1e-10)
            break;
    }
    return exp(-x) * (1.0 / (double)n + sum);
}

__global__ void computeExpIntegralFloat(int* d_ns, float* d_xs, float* d_results, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        d_results[idx] = expIntegralSeriesFloat(d_ns[idx], d_xs[idx]);
    }
}

__global__ void computeExpIntegralDouble(int* d_ns, double* d_xs, double* d_results, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        d_results[idx] = expIntegralSeriesDouble(d_ns[idx], d_xs[idx]);
    }
}
