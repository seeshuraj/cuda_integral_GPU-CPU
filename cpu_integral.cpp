#include <cmath>
#include "cpu_integral.h"

// CPU implementation: E_n(x) function for float
float exponentialIntegralFloat(int n, float x) {
    if (n == 0) return expf(-x) / x;
    float sum = 0.0f;
    for (int k = 1; k <= 100; ++k) {
        sum += powf(x, k - 1) / (tgammaf(k + n));
    }
    return expf(-x) * sum;
}

// CPU implementation: E_n(x) function for double
double exponentialIntegralDouble(int n, double x) {
    if (n == 0) return exp(-x) / x;
    double sum = 0.0;
    for (int k = 1; k <= 100; ++k) {
        sum += pow(x, k - 1) / (tgamma(k + n));
    }
    return exp(-x) * sum;
}
