#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include "cpu_integral.h"
#include "gpu_launcher.h"

bool use_gpu = true;

void parseArguments(int argc, char** argv, int& n, int& m) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-g") == 0) use_gpu = true;
        if (strcmp(argv[i], "-c") == 0) use_gpu = false;
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) m = atoi(argv[++i]);
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int n = 1000, m = 1000;
    parseArguments(argc, argv, n, m);
    int total = n * m;

    if (use_gpu) {
        std::cout << "Running GPU version...\n";
        float* h_results_f = new float[total];
        float* d_results_f = nullptr;
        int* d_n_m = nullptr;

        // Allocate device memory
        checkCudaError(cudaMalloc(&d_results_f, total * sizeof(float)), "cudaMalloc d_results_f");
        checkCudaError(cudaMalloc(&d_n_m, 2 * sizeof(int)), "cudaMalloc d_n_m");

        int h_n_m[2] = {n, m};
        checkCudaError(cudaMemcpy(d_n_m, h_n_m, 2 * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy d_n_m");

        auto start = std::chrono::high_resolution_clock::now();
        launchFloatKernel(d_n_m, d_results_f, h_results_f, total);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "GPU float time: " << std::chrono::duration<double>(end - start).count() << "s\n";

        delete[] h_results_f;
        cudaFree(d_results_f);
        cudaFree(d_n_m);
    } else {
        std::cout << "Running CPU version...\n";
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < total; ++i) {
            int x = (i % m + 1);
            exponentialIntegralFloat(n, (float)x);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "CPU float time: " << std::chrono::duration<double>(end - start).count() << "s\n";
    }
    return 0;
}

