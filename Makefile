# Compiler settings
NVCC = /usr/local/cuda-10.1/bin/nvcc
CXX = g++
CXXFLAGS = -O3 -std=c++11
NVCCFLAGS = -O3

# Targets and files
EXEC = exp_integral
OBJS = main.o gpu_integral.o cpu_integral.o

# Rules
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

main.o: main.cpp gpu_integral.h cpu_integral.h
	$(CXX) $(CXXFLAGS) -c main.cpp

gpu_integral.o: gpu_integral.cu gpu_integral.h
	$(NVCC) $(NVCCFLAGS) -c gpu_integral.cu

cpu_integral.o: cpu_integral.cpp cpu_integral.h
	$(CXX) $(CXXFLAGS) -c cpu_integral.cpp

clean:
	rm -f *.o $(EXEC)
