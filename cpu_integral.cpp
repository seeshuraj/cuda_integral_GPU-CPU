#include <cmath>
#include "cpu_integral.h"

float exponentialIntegralFloat(int n, float x) {
    float sum = 0.0f;
    for (int k = 0; k < 100; ++k) {
        sum += pow(x, k + n) / (tgamma(k + n + 1));
    }
    return sum * exp(-x);
}

double exponentialIntegralDouble(int n, double x) {
    double sum = 0.0;
    for (int k = 0; k < 100; ++k) {
        sum += pow(x, k + n) / (tgamma(k + n + 1));
    }
    return sum * exp(-x);
}

