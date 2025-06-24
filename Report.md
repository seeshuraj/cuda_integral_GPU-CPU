# MAP55616-03 - CUDA Exponential Integral Calculation Report

## 👤 Student Information

* **Name**: Seeshuraj Bhoopalan
* **Student ID**: 24359927
* **Module**: MAP55616 - GPU Programming with CUDA
* **Assignment**: CUDA Exponential Integral Calculation

---

## 🧠 Task 1: CUDA Implementation Summary

### 🎯 Objective

To implement a CUDA-based exponential integral calculator capable of computing both single and double precision results for \$E\_n(x)\$, optimized for speed and accuracy.

### 🛠️ Implementation Details

* **Base Code Used**: `exponentialIntegralCPU.tar`
* **Files Added**:

  * `gpu_integral.cu`, `gpu_integral.h`: CUDA kernel logic
  * `cpu_integral.cpp`, `cpu_integral.h`: Reorganized CPU logic
  * `main.cpp`: Command-line parser for flags `-c`, `-g`, `-n`, `-m`
  * `Makefile`: NVCC-compatible build script

### 🚀 Feature Summary

* Separate CUDA kernels for float and double
* Timing includes *all* CUDA steps (allocations, transfers, compute)
* Output compared using `fabs(cpu - gpu) < 1e-5`
* Supports CPU-only, GPU-only, or both (auto benchmark mode)

### ▶️ Execution Examples

```bash
./ei_exec -c -n 8192 -m 8192      # CPU only
./ei_exec -g -n 8192 -m 8192      # GPU only
./ei_exec -n 8192 -m 8192         # CPU + GPU + Speedup comparison
```

---

## 📊 Performance Benchmarks

| Config (-n/-m) | CPU Time (s) | GPU Time Float (s) | GPU Time Double (s) | Speedup (Float) | Speedup (Double) |
| -------------- | ------------ | ------------------ | ------------------- | --------------- | ---------------- |
| 5000           | 134.21       | 0.303              | 11.21               | 443.07          | 11.97            |
| 8192           | 353.04       | 0.721              | 29.16               | 489.18          | 12.10            |
| 16384          | 1412.29      | 1.439              | 116.20              | 981.40          | 12.15            |
| 20000          | 2112.33      | 1.698              | 161.50              | 1243.69         | 13.08            |

---

## ✅ Numerical Accuracy Check

The CPU and GPU results were compared for every element in the `n × m` matrix using:

```cpp
if (fabs(cpu_result[i] - gpu_result[i]) > 1e-5) {
    std::cerr << "Mismatch at " << i << ": CPU=" << cpu_result[i]
              << ", GPU=" << gpu_result[i] << std::endl;
}
```

**✅ No mismatches** were observed across all test cases.

---

## 🧪 Non-Square Test Case

Tested:

```bash
./ei_exec -n 10000 -m 5000
```

**✅ Output and speedup correct. Accuracy within threshold.**

---

## 🔧 CUDA Features Used

* `cudaMalloc`, `cudaMemcpy`, `cudaFree`
* `cudaEvent_t` for timing precision
* Grid/block configuration tuning
* No external libraries used

---

## 🤖 Task 2: LLM Implementation & Comparison

### 📋 LLM Used

* **LLM**: ChatGPT-4 (OpenAI)

### 📌 Prompt

> "Convert this CPU loop and exponential integral function to an optimized CUDA kernel"

### 🧠 LLM Output

* Recommended `__global__` kernel with 2D indexing
* Suggested shared memory usage (manually confirmed no speedup)
* Properly handled memory allocation and launch configuration
* Accurate transformation of loop into `i * m + j` indexing

### ✅ Results

* **Correctness**: Output matched CPU results
* **Performance**: Similar to manually written kernel
* **Conclusion**: Helpful but no performance gain over manual version

---

## 🔖 Git Progress Tags (Proof of Work)

You can find clear commit history in `commit_log.txt` (auto-generated via `git log --oneline`), including:

* `cpu-working`: Initial CPU version
* `cuda-transfer-device`: Memory allocations and transfers
* `basic-gpu-impl`: Kernel implementation (float and double)
* `timing-verified`: Benchmarking framework complete
* `report-complete`: Report + plot ready

---

## 📉 Observations

* Float kernel shows **400–1200× speedup**
* Double kernel is **\~12× faster** than CPU version
* CUDA `cold-start` penalty was noticed in the first run
* Code handles **non-square matrices** correctly

---

## 📂 Files Submitted

* Source Code: `main.cpp`, `cpu_integral.*`, `gpu_integral.*`, `Makefile`
* Benchmark Script: `plot_speedup.py`, `speedup_plot.png`
* Documentation: `Report.md`, `README.md`, `commit_log.txt`

---

## ✅ Final Notes

This assignment deepened my understanding of:

* GPU memory hierarchy and management
* Float vs Double compute behavior
* Speedup measurement using CUDA Events
* Responsible benchmarking
* Interpreting and critiquing LLM outputs

---

### 🔹 Additional Environment Note

Initially, I tried implementing the CUDA assignment on **TCD's `cuda01` cluster**, but faced environment issues:

* It is locked to **CUDA v10.1**
* Incompatible with **GCC > 8** (local GCC was v10+)
* Lacked module permissions to downgrade or override GCC

As a result, I proceeded with implementation on my **local machine (NVIDIA GEFORCE RTX, CUDA 12.2, GCC 11)**, ensuring full compatibility and successful compilation + execution.

---

**End of Report**
