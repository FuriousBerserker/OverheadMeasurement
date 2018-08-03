#!/bin/bash

#set -e -x
set -e

macro_original='-DEXECUTION_TIME=1 -DNTIMES=5'
macro_parallel="${macro_original} -DPARALLEL=1"
macro_offloading="${macro_parallel} -DOFFLOADING=1 -DDEVICE_ID=0 -DTEAM_NUM=4 -DTHREAD_LIMIT=512"
macro_shadow_memory="${macro_offloading} -DSHADOW_MEMORY=1"  
macro_cas_shadow_memory="${macro_shadow_memory} -DUSE_CAS=1"

export LIBOMPTARGET_DEBUG=0
export OMP_NUM_THREADS=48

echo 'original'
make clean >/dev/null 2>&1
MACROS=${macro_original} make >/dev/null 2>&1  
#./stream_cpp.exe | grep "overall execution time"
./stream_cpp.exe | grep -A4 "Copy:"

echo 'parallel'
make clean >/dev/null 2>&1
MACROS=${macro_parallel} make >/dev/null 2>&1
#./stream_cpp.exe | grep "overall execution time"
echo 'OpenMP:'
./stream_cpp.exe | grep -A4 "Copy:"

echo 'offloading'
make clean >/dev/null 2>&1
MACROS=${macro_offloading} make >/dev/null 2>&1
#./stream_cpp.exe | grep "overall execution time"
echo 'OpenMP:'
./stream_cpp.exe | grep -A4 "Copy:"
echo 'CUDA:'
./stream_cu.exe | grep -A4 "Copy:"

echo 'shadow memory'
make clean >/dev/null 2>&1
MACROS=${macro_shadow_memory} make >/dev/null 2>&1
#./stream_cpp.exe | grep "overall execution time"
echo 'OpenMP:'
./stream_cpp.exe | grep -A4 "Copy:"
echo 'CUDA-threadfence:'
./stream_cu.exe | grep -A4 "Copy:"
make clean >/dev/null 2>&1
MACROS=${macro_cas_shadow_memory} make >/dev/null 2>&1
echo 'CUDA-CAS:'
./stream_cu.exe | grep -A4 "Copy:"
