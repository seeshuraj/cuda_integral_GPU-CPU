#ifndef GPU_INTEGRAL_H
#define GPU_INTEGRAL_H

#include <vector>

void computeExponentialIntegralFloatGPU(int n, const std::vector<float>& xValues, std::vector<float>& results);
void computeExponentialIntegralDoubleGPU(int n, const std::vector<double>& xValues, std::vector<double>& results);

#endif // GPU_INTEGRAL_H
