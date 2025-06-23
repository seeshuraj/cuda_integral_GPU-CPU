#include <cuda_runtime.h>
#include "cpu_integral.h"
#include "gpu_integral.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>
#include <fstream>

using namespace std;

void printUsage() {
    cout << "Usage: ./ei_exec [-c] [-g] [-n <max_n>] [-m <num_samples>]" << endl;
    cout << "  -c : Run only CPU version (skip GPU)" << endl;
    cout << "  -g : Run only GPU version (skip CPU)" << endl;
    cout << "  -n : Set maximum n for integral" << endl;
    cout << "  -m : Number of x samples" << endl;
}

void checkCUDAError(const string& msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA Error after " << msg << ": " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    bool runCPU = true;
    bool runGPU = true;
    int n = 10, m = 100;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-c") == 0) runGPU = false;
        else if (strcmp(argv[i], "-g") == 0) runCPU = false;
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

    vector<float> cpuResultsFloat(m);
    vector<double> cpuResultsDouble(m);

    double cpuTime = 0;
    if (runCPU) {
        cout << "Running CPU version...\n";
        auto start = chrono::high_resolution_clock::now();
        for (int i = 0; i < m; ++i) {
            cpuResultsFloat[i] = exponentialIntegralFloat(n, xFloat[i]);
            cpuResultsDouble[i] = exponentialIntegralDouble(n, xDouble[i]);
        }
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cpuTime = elapsed.count();
        cout << "CPU Time: " << cpuTime << " seconds\n";
    }

    if (runGPU) {
        cout << "Running GPU version...\n";
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        vector<float> gpuResultsFloat;
        vector<double> gpuResultsDouble;

        cudaEventRecord(start);
        computeExponentialIntegralFloatGPU(n, xFloat, gpuResultsFloat);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float floatTime = 0;
        cudaEventElapsedTime(&floatTime, start, stop);
        checkCUDAError("Float GPU Kernel");

        cudaEventRecord(start);
        computeExponentialIntegralDoubleGPU(n, xDouble, gpuResultsDouble);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float doubleTime = 0;
        cudaEventElapsedTime(&doubleTime, start, stop);
        checkCUDAError("Double GPU Kernel");

        floatTime /= 1000.0f;
        doubleTime /= 1000.0f;

        cout << "GPU Time (float): " << floatTime << " seconds\n";
        cout << "GPU Time (double): " << doubleTime << " seconds\n";

        if (runCPU && cpuTime > 0) {
            cout << "Speedup (float): " << cpuTime / floatTime << "\n";
            cout << "Speedup (double): " << cpuTime / doubleTime << "\n";
        }

        if (runCPU) {
            cout << "\nVerifying results (first 5 samples):\n";
            for (int i = 0; i < min(5, m); ++i) {
                float error = fabs(cpuResultsFloat[i] - gpuResultsFloat[i]);
                double d_error = fabs(cpuResultsDouble[i] - gpuResultsDouble[i]);
                printf("x = %.5f | CPU_f: %.8f | GPU_f: %.8f | Err_f: %.2e\n",
                       xFloat[i], cpuResultsFloat[i], gpuResultsFloat[i], error);
                printf("x = %.5f | CPU_d: %.12f | GPU_d: %.12f | Err_d: %.2e\n\n",
                       xFloat[i], cpuResultsDouble[i], gpuResultsDouble[i], d_error);
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
