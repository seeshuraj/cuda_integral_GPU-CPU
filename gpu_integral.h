#ifndef GPU_INTEGRAL_H
#define GPU_INTEGRAL_H

__global__ void computeExpIntegralKernel(float* results, int* n_vals, float* m_vals, int total);

#endif

