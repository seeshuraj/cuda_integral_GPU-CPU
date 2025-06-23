# MAP55616-03 – CUDA Exponential Integral Calculation

**Name:** Seeshuraj Bhoopalan  
**ID:** 24359927   
**Repository:** [https://github.com/seeshuraj/cuda_integral_GPU-CPU](https://github.com/seeshuraj/cuda_integral_GPU-CPU)

---

## Overview

This project implements a high-performance CUDA-based calculator for the Exponential Integral function $E_n(x)$, supporting both single and double precision computations. It benchmarks the GPU implementation against its CPU counterpart and analyzes the performance gains.

## Features

* ✅ Single and double precision CUDA kernels
* ✅ Accurate computation of $E_n(x)$ for arbitrary $n$ and $x$
* ✅ Command-line flags to run CPU-only, GPU-only, or both
* ✅ Speedup measurement and result validation
* ✅ LLM-based comparison (ChatGPT-4)

## Requirements

* NVIDIA GPU with CUDA support
* CUDA Toolkit 11.0 or higher
* `make`, `g++`, `nvcc`

## Build Instructions

```bash
make clean
make
```

## Usage

```bash
./ei_exec [-c] [-g] [-n <max_n>] [-m <num_samples>]

# Examples:
./ei_exec -n 8192 -m 8192        # Run both CPU and GPU
./ei_exec -g -n 8192 -m 8192     # GPU only
./ei_exec -c -n 8192 -m 8192     # CPU only
```

## Output

```
Running CPU version...
CPU Time: 353.038 seconds
Running GPU version...
GPU Time (float): 0.721694 seconds
GPU Time (double): 29.1637 seconds
Speedup (float): 489.179
Speedup (double): 12.1054

Verifying results (first 5 samples):
x = 0.00000 | CPU_f: 0.00012874 | GPU_f: 0.00012874 | Err_f: 0.00e+00
...
```

## Benchmarking Configurations

* `-n 5000 -m 5000`
* `-n 8192 -m 8192`
* `-n 16384 -m 16384`
* `-n 20000 -m 20000`

Speedup graphs are available in `speedup_plot.png`.

## Directory Structure

```
├── main.cpp
├── cpu_integral.cpp / .h
├── gpu_integral.cu / .h
├── Makefile
├── plot_speedup.py
├── speedup_plot.png
└── Report.md
```

---

## ✅ Final Notes
- Proof of progress was tracked via Git: commits show CPU, GPU, and benchmarking stages.
- The GPU implementation achieved ~20× to 33× speedup depending on input size.
- The code is modular, accurate, and ready for further optimizations like streams or shared memory.

---
