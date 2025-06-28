#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <iomanip>

#include "gpu_integral.h"

// CPU float version
float exponentialIntegralFloat(int n, float x) {
    if (n == 0) return expf(-x) / x;
    float sum = 0.0f;
    for (int k = 1; k <= 100; ++k) {
        sum += powf(x, k - 1) / (tgammaf(k + n));
    }
    return expf(-x) * sum;
}

// CPU double version
double exponentialIntegralDouble(int n, double x) {
    if (n == 0) return exp(-x) / x;
    double sum = 0.0;
    for (int k = 1; k <= 100; ++k) {
        sum += pow(x, k - 1) / (tgamma(k + n));
    }
    return exp(-x) * sum;
}

// Command line parsing
void parseArguments(int argc, char** argv, int& N, int& M, bool& useCPU) {
    N = 8192;
    M = 8192;
    useCPU = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
            M = atoi(argv[++i]);
        else if (strcmp(argv[i], "-g") == 0)
            useCPU = true; // Run CPU only
    }
}

int main(int argc, char** argv) {
    int N, M;
    bool useCPU;
    parseArguments(argc, argv, N, M, useCPU);
    int total = N * M;

    std::vector<int> ns(total);
    std::vector<float> xs_float(total);
    std::vector<double> xs_double(total);

    for (int i = 0; i < total; ++i) {
        ns[i] = i % 5; // keep n small (0-4)
        xs_float[i] = (float)(1.0f + (i % 100) / 10.0f);
        xs_double[i] = (double)(1.0 + (i % 100) / 10.0);
    }

    std::vector<float> result_cpu_float(total), result_gpu_float(total);
    std::vector<double> result_cpu_double(total), result_gpu_double(total);

    if (useCPU) {
        // CPU Execution
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < total; ++i) {
            result_cpu_float[i] = exponentialIntegralFloat(ns[i], xs_float[i]);
        }
        for (int i = 0; i < total; ++i) {
            result_cpu_double[i] = exponentialIntegralDouble(ns[i], xs_double[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "[CPU] Execution time: " << diff.count() << " seconds\n";
    } else {
        // GPU Execution
        launchFloatKernel(ns.data(), xs_float.data(), result_gpu_float.data(), total);
        launchDoubleKernel(ns.data(), xs_double.data(), result_gpu_double.data(), total);

        // For accuracy, run CPU too
        for (int i = 0; i < total; ++i) {
            result_cpu_float[i] = exponentialIntegralFloat(ns[i], xs_float[i]);
        }
        for (int i = 0; i < total; ++i) {
            result_cpu_double[i] = exponentialIntegralDouble(ns[i], xs_double[i]);
        }

        // Accuracy Check
        int float_diff = 0, double_diff = 0;
        for (int i = 0; i < total; ++i) {
            if (fabs(result_cpu_float[i] - result_gpu_float[i]) > 1e-5f) {
                ++float_diff;
                if (float_diff <= 5) {
                    std::cout << std::setprecision(10)
                              << "Mismatch [float] at " << i << ": CPU=" << result_cpu_float[i]
                              << ", GPU=" << result_gpu_float[i] << "\n";
                }
            }
        }

        for (int i = 0; i < total; ++i) {
            if (fabs(result_cpu_double[i] - result_gpu_double[i]) > 1e-5) {
                ++double_diff;
                if (double_diff <= 5) {
                    std::cout << std::setprecision(15)
                              << "Mismatch [double] at " << i << ": CPU=" << result_cpu_double[i]
                              << ", GPU=" << result_gpu_double[i] << "\n";
                }
            }
        }

        std::cout << "[Check] Float mismatches: " << float_diff << " / " << total << "\n";
        std::cout << "[Check] Double mismatches: " << double_diff << " / " << total << "\n";
    }

    return 0;
}
