CC = clang
CXX = clang++
CFLAGS = -O0 -g -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
CXXFLAGS = -O3 -g -std=c++11 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
MACROS ?= -DPARALLEL=1 -DOFFLOADING=1 -DSHADOW_MEMORY=1 -DEXECUTION_TIME=1 -DNTIMES=5 -DTEAM_NUM=32768 -DTHREAD_LIMIT=1024 -DSTREAM_ARRAY_SIZE=33554432
CUDA_CXX = nvcc
CUDA_CXX_FLAGS = -O3 -std=c++11 -arch=sm_60
CUDA_CXX_MACROS = -DCUDA=1

FF = g77
FFLAGS = -O0

.PHONY: default
default: stream_cpp.exe stream_cu.exe

all: stream_f.exe stream_c.exe

stream_f.exe: stream.f mysecond.o
	$(CC) $(CFLAGS) -c mysecond.c
	$(FF) $(FFLAGS) -c stream.f
	$(FF) $(FFLAGS) stream.o mysecond.o -o stream_f.exe

stream_c.exe: stream.c
	$(CC) $(CFLAGS) $(MACROS) $? -o $@

stream_cpp.exe: stream.cpp
	$(CXX) $(CXXFLAGS) $(MACROS) $? -o $@

stream_cu.exe: stream.cu
	$(CUDA_CXX) $(CUDA_CXX_FLAGS) $(MACROS) $? -o $@

stream_cu2.exe: main.cpp CUDAStream.cu
	$(CUDA_CXX) $(CUDA_CXX_FLAGS) ${CUDA_CXX_MACROS} $? -o $@

clean:
	rm -f stream_f.exe stream_c.exe stream_cpp.exe stream_cu.exe stream_cu2.exe *.o

# an example of a more complex build line for the Intel icc compiler
stream.icc: stream.c
	icc -O3 -xCORE-AVX2 -ffreestanding -qopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 stream.c -o stream.omp.AVX2.80M.20x.icc
