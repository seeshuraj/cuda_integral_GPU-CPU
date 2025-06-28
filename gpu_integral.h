#ifndef GPU_INTEGRAL_H
#define GPU_INTEGRAL_H

void launchFloatKernel(int* ns, float* xs, float* results, int total);
void launchDoubleKernel(int* ns, double* xs, double* results, int total);

#endif // GPU_INTEGRAL_H
