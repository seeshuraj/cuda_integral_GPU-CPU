#include "cpu_integral.h"
#include <cmath>

// Compute the exponential integral using numerical integration
// This uses a simple trapezoidal rule with a large number of iterations for convergence

float exponentialIntegralFloat(int n, float x) {
    const int maxIterations = 1000000;
    const float t_max = 100.0f;
    const float dt = (t_max - 1.0f) / maxIterations;

    float sum = 0.0f;
    for (int i = 0; i < maxIterations; ++i) {
        float t1 = 1.0f + i * dt;
        float t2 = t1 + dt;
        float f1 = expf(-x * t1) / powf(t1, n);
        float f2 = expf(-x * t2) / powf(t2, n);
        sum += 0.5f * (f1 + f2) * dt;
    }

    return sum;
}

double exponentialIntegralDouble(int n, double x) {
    const int maxIterations = 1000000;
    const double t_max = 100.0;
    const double dt = (t_max - 1.0) / maxIterations;

    double sum = 0.0;
    for (int i = 0; i < maxIterations; ++i) {
        double t1 = 1.0 + i * dt;
        double t2 = t1 + dt;
        double f1 = exp(-x * t1) / pow(t1, n);
        double f2 = exp(-x * t2) / pow(t2, n);
        sum += 0.5 * (f1 + f2) * dt;
    }

    return sum;
}

