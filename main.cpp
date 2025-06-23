#include <cuda_runtime.h>
#include "cpu_integral.h"
#include "gpu_integral.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>

using namespace std;

void printUsage() {
    cout << "Usage: ./ei_exec [-g] [-c] -n <max_n> -m <num_samples>\n";
}

int main(int argc, char* argv[]) {
    bool useGPU = false, skipCPU = false;
    int n = 10, m = 100;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-g") == 0) useGPU = true;
        else if (strcmp(argv[i], "-c") == 0) skipCPU = true;
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i+1 < argc) m = atoi(argv[++i]);
        else printUsage();
    }

    vector<float> xFloat(m);
    vector<double> xDouble(m);
    for (int i = 0; i < m; ++i) {
        float x = i * 10.0f / (m - 1);
        xFloat[i] = x;
        xDouble[i] = static_cast<double>(x);
    }

    if (!skipCPU) {
        cout << "Running CPU version...\n";
        auto start = chrono::high_resolution_clock::now();
        vector<float> cpuResultsFloat(m);
        vector<double> cpuResultsDouble(m);
        for (int i = 0; i < m; ++i) {
            cpuResultsFloat[i] = exponentialIntegralFloat(n, xFloat[i]);
            cpuResultsDouble[i] = exponentialIntegralDouble(n, xDouble[i]);
        }
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "CPU Time: " << elapsed.count() << " seconds\n";
    }

    if (useGPU) {
        cout << "Running GPU version...\n";
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        vector<float> gpuResultsFloat;
        vector<double> gpuResultsDouble;
        computeExponentialIntegralFloatGPU(n, xFloat, gpuResultsFloat);
        computeExponentialIntegralDoubleGPU(n, xDouble, gpuResultsDouble);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "GPU Time (float+double): " << milliseconds / 1000.0f << " seconds\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}

