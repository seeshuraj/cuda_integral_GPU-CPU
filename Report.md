# MAP55616-03 - CUDA Exponential Integral Calculation Report

## Student Information

* **Name**: Seeshuraj Bhoopalan
* **Student ID**: 24359927 
* **Module**: MAP55616 - GPU Programming with CUDA
* **Assignment**: CUDA Exponential Integral Calculation

---

## Task 1: CUDA Implementation Summary

### Objective

To implement a CUDA-based exponential integral calculator capable of computing both single and double precision results for $E_n(x)$, optimized for speed and accuracy.

### Implementation Details

* **Base Code Used**: exponentialIntegralCPU.tar
* **Files Added**:

  * `gpu_integral.cu`, `gpu_integral.h`: GPU kernel logic
  * `cpu_integral.cpp`, `cpu_integral.h`: Reorganized CPU logic
  * `main.cpp`: Parses `-c`, `-g`, `-n`, `-m` flags, runs CPU/GPU
  * `Makefile`: Builds the program with `nvcc`

### Feature Summary

* Float and double CUDA kernels
* Timing includes **all** CUDA activities (allocations + transfers)
* Command-line parsing via `-c`, `-g`, `-n`, `-m`
* Output comparison and error check: `fabs(cpu - gpu) < 1e-5`

### Execution Examples

* `./ei_exec -c -n 8192 -m 8192` (CPU only)
* `./ei_exec -g -n 8192 -m 8192` (GPU only)
* `./ei_exec -n 8192 -m 8192` (both + speedup calc)

---

## Performance Benchmarks

### Timings and Speedup

| Config (-n/-m) | CPU Time (s) | GPU Time Float (s) | GPU Time Double (s) | Speedup (float) | Speedup (double) |
| -------------- | ------------ | ------------------ | ------------------- | --------------- | ---------------- |
| 5000           | 134.21       | 0.303              | 11.21               | 443.07          | 11.97            |
| 8192           | 353.04       | 0.721              | 29.16               | 489.18          | 12.10            |
| 16384          | 1412.29      | 1.439              | 116.20              | 981.40          | 12.15            |
| 20000          | 2112.33      | 1.698              | 161.50              | 1243.69         | 13.08            |

### Non-Square Case Tested

* `./ei_exec -n 10000 -m 5000` passed ✅

### Result Accuracy

* GPU and CPU outputs differ < `1e-5`
* Verified for 5 samples; all passed

---

## CUDA Features Used

* `cudaMalloc`, `cudaMemcpy`, `cudaFree`
* `cudaEvent_t` for timing
* Parallel kernel launches with `<<<grid, block>>>`
* No external libraries used (as per guidelines)

---

## Optional Work (Task 2: LLM Assistance)

* **LLM Tested**: ChatGPT-4 (OpenAI)
* **Prompt**: "Optimize this CUDA exponential integral kernel"
* **LLM Suggestion**:

  * Recommended memory coalescing
  * Suggested using shared memory (did not significantly improve speed)
* **Outcome**:

  * Code executed correctly
  * Performance same as original optimized version
* **Conclusion**: LLM suggestions were mostly accurate, but not superior to hand-optimized version.

---

## Git Progress Tags (Proof of Work)

```
cpu-working:          Initial CPU code tested
cuda-transfer-device: Data moved to GPU
basic-gpu-impl:       GPU kernels implemented
final-benchmark:      Benchmarks & plots completed
report-complete:      Final report added
```

---

## Observations

* CUDA shows 400–1200x speedup in float mode
* Double precision is \~12x faster than CPU
* Initial GPU kernel call may be slow on first launch (cold-start)
* Code works for large values and non-square configurations

---

## Files Submitted

* `main.cpp`, `cpu_integral.cpp`, `cpu_integral.h`
* `gpu_integral.cu`, `gpu_integral.h`, `Makefile`
* `plot_speedup.py`, `speedup_plot.png`
* `report.md`, `README.md`

---

## Final Note

This assignment served as a strong exercise in understanding CUDA memory management, parallel kernel execution, and benchmarking with respect to CPU execution. All benchmarks and accuracy tests confirm the validity and speed of the CUDA implementation.

