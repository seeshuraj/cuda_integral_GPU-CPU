#include "gpu_integral.h"
#include <math.h>

__global__ void computeExpIntegralKernel(float* results, int* n_vals, float* m_vals, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int n = n_vals[idx];
    float x = m_vals[idx];

    // Trapezoidal rule parameters
    int steps = 100000;  // Must match CPU
    float t_min = 1.0f;
    float t_max = 20.0f; // reasonable upper bound
    float h = (t_max - t_min) / steps;

    float sum = 0.5f * (expf(-x * t_min) / powf(t_min, n) + expf(-x * t_max) / powf(t_max, n));

    for (int i = 1; i < steps; ++i) {
        float t = t_min + i * h;
        sum += expf(-x * t) / powf(t, n);
    }

    results[idx] = h * sum;
}

