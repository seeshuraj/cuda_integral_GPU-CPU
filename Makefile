CUDA_PATH=/usr/local/cuda-12.8
NVCC=$(CUDA_PATH)/bin/nvcc
CXX=g++
CXXFLAGS=-O3 -std=c++11 -I$(CUDA_PATH)/include
NVCCFLAGS=-O3 --expt-relaxed-constexpr

all: ei_exec

ei_exec: main.o cpu_integral.o gpu_integral.o gpu_launcher.o
	$(NVCC) -o ei_exec main.o cpu_integral.o gpu_integral.o gpu_launcher.o -lcuda -lcudart

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

cpu_integral.o: cpu_integral.cpp
	$(CXX) $(CXXFLAGS) -c cpu_integral.cpp

gpu_integral.o: gpu_integral.cu
	$(NVCC) $(NVCCFLAGS) -c gpu_integral.cu

gpu_launcher.o: gpu_launcher.cu
	$(NVCC) $(NVCCFLAGS) -c gpu_launcher.cu

clean:
	rm -f *.o ei_exec

