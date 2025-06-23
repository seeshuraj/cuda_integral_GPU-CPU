import matplotlib.pyplot as plt

# Sample sizes (m values)
sizes = [5000, 8192, 16384, 20000]

# Corresponding CPU and GPU times (in seconds)
cpu_times = [279.88, 458.64, 917.28, 1120.0]  # Estimated by linear scaling
gpu_times = [13.96, 13.98, 27.15, 53.68]

# Calculate speedup = CPU time / GPU time
speedups = [cpu / gpu for cpu, gpu in zip(cpu_times, gpu_times)]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(sizes, speedups, marker='o', linestyle='-', color='blue', linewidth=2)
plt.title('Speedup of GPU over CPU for Exponential Integral Calculation')
plt.xlabel('Problem Size (m values)')
plt.ylabel('Speedup (CPU Time / GPU Time)')
plt.grid(True)
plt.xticks(sizes)
plt.savefig('speedup_plot.png')
plt.show()
