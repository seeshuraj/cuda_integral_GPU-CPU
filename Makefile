NVCC = nvcc
NVFLAGS = -O2 -std=c++11

EXEC = ei_exec

OBJS = main.o cpu_integral.o gpu_integral.o

all: $(EXEC)

main.o: main.cpp cpu_integral.h gpu_integral.h
	$(NVCC) $(NVFLAGS) -c main.cpp

cpu_integral.o: cpu_integral.cpp cpu_integral.h
	$(NVCC) $(NVFLAGS) -c cpu_integral.cpp

gpu_integral.o: gpu_integral.cu gpu_integral.h
	$(NVCC) $(NVFLAGS) -c gpu_integral.cu

$(EXEC): $(OBJS)
	$(NVCC) -o $(EXEC) $(OBJS)

clean:
	rm -f *.o $(EXEC)
