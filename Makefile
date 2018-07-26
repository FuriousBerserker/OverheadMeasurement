CC = clang
CXX = clang++
CFLAGS = -O0 -g -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
CXXFLAGS = -O0 -g -std=c++11 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
MACROS ?= -DPARALLEL=1 -DOFFLOADING=1 -DSHADOW_MEMORY=1 -DEXECUTION_TIME=1 -DNTIMES=5

FF = g77
FFLAGS = -O0

.PHONY: default
default: stream_cpp.exe

all: stream_f.exe stream_c.exe

stream_f.exe: stream.f mysecond.o
	$(CC) $(CFLAGS) -c mysecond.c
	$(FF) $(FFLAGS) -c stream.f
	$(FF) $(FFLAGS) stream.o mysecond.o -o stream_f.exe

stream_c.exe: stream.c
	$(CC) $(CFLAGS) $(MACROS) $? -o $@

stream_cpp.exe: stream.cpp
	$(CXX) $(CXXFLAGS) $(MACROS) $? -o $@

clean:
	rm -f stream_f.exe stream_c.exe stream_cpp.exe *.o

# an example of a more complex build line for the Intel icc compiler
stream.icc: stream.c
	icc -O3 -xCORE-AVX2 -ffreestanding -qopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 stream.c -o stream.omp.AVX2.80M.20x.icc
