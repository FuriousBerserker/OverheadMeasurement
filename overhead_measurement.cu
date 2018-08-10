#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cfloat>
#include <iostream>
#include <string>
#include <vector>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 100
#endif

#ifndef BLOCK_NUM
#define BLOCK_NUM 10
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

#ifndef LOOP_TIME
#define LOOP_TIME 100000
#endif

#ifndef NTIMES
#define NTIMES 10
#endif

#define cudaErrorCheck(call)                                                                                                            \
    do {                                                                                                                                \
        cudaError_t cuErr = call;                                                                                                       \
        if (cudaSuccess != cuErr) {                                                                                                     \
            printf("CUDA Error - %s:%d: '%s, %s'\n", __FILE__, __LINE__, cudaGetErrorName(cuErr), cudaGetErrorString(cuErr));           \
            exit(1);                                                                                                                    \
        }                                                                                                                               \
    } while (0)

__global__ void testLDVolatile(unsigned long long int *array, unsigned size, unsigned long long int *array2) {
    for (unsigned i = 0; i < LOOP_TIME; i++) {
        unsigned long long int element = *(volatile unsigned long long int*)(&array[size - 2]);
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        array2[threadID] = element;
    }
}

__global__ void testSTVolatile(unsigned long long int *array, unsigned size) {
    for (unsigned i = 0; i < LOOP_TIME; i++) {
        *(volatile unsigned long long int*)(&array[size -1]) = threadIdx.x; 
    }
}

__global__ void testAtomicExch(unsigned long long int *array, unsigned size) {
    for (unsigned i = 0; i < LOOP_TIME; i++) {
        atomicExch(&array[size -1], threadIdx.x); 
    }
}

__global__ void testLDSTVolatile(unsigned long long int *array, unsigned size, unsigned long long int *array2) {

    for (unsigned i = 0; i < LOOP_TIME; i++) {
        unsigned long long int element = *(volatile unsigned long long int*)(&array[size - 1]);
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        array2[threadID] = element;
        *(volatile unsigned long long int*)(&array[size -1]) = threadIdx.x; 
    }
}

__global__ void testLDAtomic(unsigned long long int *array, unsigned size, unsigned long long int *array2) {

    for (unsigned i = 0; i < LOOP_TIME; i++) {
        unsigned long long int element = *(volatile unsigned long long int*)(&array[size - 1]);
        unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
        array2[threadID] = element;
        atomicExch(&array[size -1], threadIdx.x); 
    }
}

__global__ void testSTRelaxed(unsigned long long int *array, unsigned size) {
#if __CUDA_ARCH__ < 700
    asm(
    "{\n\t"
    " .reg .pred 	p<2>;\n\t"
    " .reg .b32 	r<8>;\n\t"
    " .reg .b64 	rd<6>;\n\t"
    " mov.u64 	rd3, %0;\n\t"
    " mov.u32 	r4, %1;\n\t"
    " cvta.to.global.u64 	rd4, rd3;\n\t"
    " mov.u32 	r5, %tid.x;\n\t"
    " cvt.u64.u32	rd1, r5;\n\t"
    " add.s32 	r6, r4, -1;\n\t"
    " mul.wide.u32 	rd5, r6, 8;\n\t"
    " add.s64 	rd2, rd4, rd5;\n\t"
    " mov.u32 	r7, -100000;\n\t"
    " BB1_1:\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " st.volatile.global.u64 	[rd2], rd1;\n\t"
    " add.s32 	r7, r7, 50;\n\t"
    " setp.ne.s32	p1, r7, 0;\n\t"
    " @p1 bra 	BB1_1;\n\t"
    " ret;\n\t"
    "}"
    :: "l"(array), "r"(size) : "memory");
#else
    asm(
    "{\n\t"
    " .reg .pred 	p<2>;\n\t"
    " .reg .b32 	r<8>;\n\t"
    " .reg .b64 	rd<6>;\n\t"
    " mov.u64 	rd3, %0;\n\t"
    " mov.u32 	r4, %1;\n\t"
    " cvta.to.global.u64 	rd4, rd3;\n\t"
    " mov.u32 	r5, %tid.x;\n\t"
    " cvt.u64.u32	rd1, r5;\n\t"
    " add.s32 	r6, r4, -1;\n\t"
    " mul.wide.u32 	rd5, r6, 8;\n\t"
    " add.s64 	rd2, rd4, rd5;\n\t"
    " mov.u32 	r7, -100000;\n\t"
    " BB1_1:\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " st.global.relaxed.gpu.u64 	[rd2], rd1;\n\t"
    " add.s32 	r7, r7, 50;\n\t"
    " setp.ne.s32	p1, r7, 0;\n\t"
    " @p1 bra 	BB1_1;\n\t"
    " ret;\n\t"
    "}"
    :: "l"(array), "r"(size) : "memory");
#endif
}

__global__ void testLDSTRelaxed(unsigned long long int *array, unsigned size, unsigned long long int *array2) {
#if __CUDA_ARCH__ < 700
    asm(
    "{\n\t"
    " .reg .pred 	p<2>;\n\t"
    " .reg .b32 	r<11>;\n\t"
    " .reg .b64 	rd<42>;\n\t"
    " mov.u64 	rd4, %0;\n\t"
    " mov.u32 	r4, %1;\n\t"
    " mov.u64 	rd5, %2;\n\t"
    " cvta.to.global.u64 	rd6, rd5;\n\t"
    " add.s32 	r5, r4, -1;\n\t"
    " cvta.to.global.u64 	rd7, rd4;\n\t"
    " mul.wide.u32 	rd8, r5, 8;\n\t"
    " add.s64 	rd1, rd7, rd8;\n\t"
    " mov.u32 	r6, %ntid.x;\n\t"
    " mov.u32 	r7, %ctaid.x;\n\t"
    " mov.u32 	r8, %tid.x;\n\t"
    " mad.lo.s32 	r9, r6, r7, r8;\n\t"
    " mul.wide.u32 	rd9, r9, 8;\n\t"
    " add.s64 	rd2, rd6, rd9;\n\t"
    " cvt.u64.u32	rd3, r8;\n\t"
    " mov.u32 	r10, -100000;\n\t"
    " BB3_1:\n\t"
    " ld.volatile.global.u64 	rd10, [rd1];\n\t"
    " st.global.u64 	[rd2], rd10;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd11, [rd1];\n\t"
    " st.global.u64 	[rd2], rd11;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd12, [rd1];\n\t"
    " st.global.u64 	[rd2], rd12;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd13, [rd1];\n\t"
    " st.global.u64 	[rd2], rd13;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd14, [rd1];\n\t"
    " st.global.u64 	[rd2], rd14;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd15, [rd1];\n\t"
    " st.global.u64 	[rd2], rd15;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd16, [rd1];\n\t"
    " st.global.u64 	[rd2], rd16;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd17, [rd1];\n\t"
    " st.global.u64 	[rd2], rd17;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd18, [rd1];\n\t"
    " st.global.u64 	[rd2], rd18;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd19, [rd1];\n\t"
    " st.global.u64 	[rd2], rd19;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd20, [rd1];\n\t"
    " st.global.u64 	[rd2], rd20;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd21, [rd1];\n\t"
    " st.global.u64 	[rd2], rd21;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd22, [rd1];\n\t"
    " st.global.u64 	[rd2], rd22;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd23, [rd1];\n\t"
    " st.global.u64 	[rd2], rd23;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd24, [rd1];\n\t"
    " st.global.u64 	[rd2], rd24;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd25, [rd1];\n\t"
    " st.global.u64 	[rd2], rd25;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd26, [rd1];\n\t"
    " st.global.u64 	[rd2], rd26;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd27, [rd1];\n\t"
    " st.global.u64 	[rd2], rd27;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd28, [rd1];\n\t"
    " st.global.u64 	[rd2], rd28;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd29, [rd1];\n\t"
    " st.global.u64 	[rd2], rd29;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd30, [rd1];\n\t"
    " st.global.u64 	[rd2], rd30;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd31, [rd1];\n\t"
    " st.global.u64 	[rd2], rd31;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd32, [rd1];\n\t"
    " st.global.u64 	[rd2], rd32;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd33, [rd1];\n\t"
    " st.global.u64 	[rd2], rd33;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd34, [rd1];\n\t"
    " st.global.u64 	[rd2], rd34;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd35, [rd1];\n\t"
    " st.global.u64 	[rd2], rd35;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd36, [rd1];\n\t"
    " st.global.u64 	[rd2], rd36;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd37, [rd1];\n\t"
    " st.global.u64 	[rd2], rd37;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd38, [rd1];\n\t"
    " st.global.u64 	[rd2], rd38;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd39, [rd1];\n\t"
    " st.global.u64 	[rd2], rd39;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd40, [rd1];\n\t"
    " st.global.u64 	[rd2], rd40;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " ld.volatile.global.u64 	rd41, [rd1];\n\t"
    " st.global.u64 	[rd2], rd41;\n\t"
    " st.volatile.global.u64 	[rd1], rd3;\n\t"
    " add.s32 	r10, r10, 32;\n\t"
    " setp.ne.s32	p1, r10, 0;\n\t"
    " @p1 bra 	BB3_1;\n\t"
    " ret;\n\t"
    "}"
    :: "l"(array), "r"(size), "l"(array2) : "memory");
#else
    asm(
    "{\n\t"
    " .reg .pred 	p<2>;\n\t"
    " .reg .b32 	r<11>;\n\t"
    " .reg .b64 	rd<42>;\n\t"
    " mov.u64 	rd4, %0;\n\t"
    " mov.u32 	r4, %1;\n\t"
    " mov.u64 	rd5, %2;\n\t"
    " cvta.to.global.u64 	rd6, rd5;\n\t"
    " add.s32 	r5, r4, -1;\n\t"
    " cvta.to.global.u64 	rd7, rd4;\n\t"
    " mul.wide.u32 	rd8, r5, 8;\n\t"
    " add.s64 	rd1, rd7, rd8;\n\t"
    " mov.u32 	r6, %ntid.x;\n\t"
    " mov.u32 	r7, %ctaid.x;\n\t"
    " mov.u32 	r8, %tid.x;\n\t"
    " mad.lo.s32 	r9, r6, r7, r8;\n\t"
    " mul.wide.u32 	rd9, r9, 8;\n\t"
    " add.s64 	rd2, rd6, rd9;\n\t"
    " cvt.u64.u32	rd3, r8;\n\t"
    " mov.u32 	r10, -100000;\n\t"
    " BB3_1:\n\t"
    " ld.global.relaxed.gpu.u64 	rd10, [rd1];\n\t"
    " st.global.u64 	[rd2], rd10;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd11, [rd1];\n\t"
    " st.global.u64 	[rd2], rd11;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd12, [rd1];\n\t"
    " st.global.u64 	[rd2], rd12;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd13, [rd1];\n\t"
    " st.global.u64 	[rd2], rd13;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd14, [rd1];\n\t"
    " st.global.u64 	[rd2], rd14;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd15, [rd1];\n\t"
    " st.global.u64 	[rd2], rd15;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd16, [rd1];\n\t"
    " st.global.u64 	[rd2], rd16;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd17, [rd1];\n\t"
    " st.global.u64 	[rd2], rd17;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd18, [rd1];\n\t"
    " st.global.u64 	[rd2], rd18;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd19, [rd1];\n\t"
    " st.global.u64 	[rd2], rd19;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd20, [rd1];\n\t"
    " st.global.u64 	[rd2], rd20;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd21, [rd1];\n\t"
    " st.global.u64 	[rd2], rd21;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd22, [rd1];\n\t"
    " st.global.u64 	[rd2], rd22;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd23, [rd1];\n\t"
    " st.global.u64 	[rd2], rd23;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd24, [rd1];\n\t"
    " st.global.u64 	[rd2], rd24;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd25, [rd1];\n\t"
    " st.global.u64 	[rd2], rd25;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd26, [rd1];\n\t"
    " st.global.u64 	[rd2], rd26;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd27, [rd1];\n\t"
    " st.global.u64 	[rd2], rd27;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd28, [rd1];\n\t"
    " st.global.u64 	[rd2], rd28;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd29, [rd1];\n\t"
    " st.global.u64 	[rd2], rd29;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd30, [rd1];\n\t"
    " st.global.u64 	[rd2], rd30;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd31, [rd1];\n\t"
    " st.global.u64 	[rd2], rd31;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd32, [rd1];\n\t"
    " st.global.u64 	[rd2], rd32;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd33, [rd1];\n\t"
    " st.global.u64 	[rd2], rd33;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd34, [rd1];\n\t"
    " st.global.u64 	[rd2], rd34;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd35, [rd1];\n\t"
    " st.global.u64 	[rd2], rd35;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd36, [rd1];\n\t"
    " st.global.u64 	[rd2], rd36;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd37, [rd1];\n\t"
    " st.global.u64 	[rd2], rd37;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd38, [rd1];\n\t"
    " st.global.u64 	[rd2], rd38;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd39, [rd1];\n\t"
    " st.global.u64 	[rd2], rd39;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd40, [rd1];\n\t"
    " st.global.u64 	[rd2], rd40;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " ld.global.relaxed.gpu.u64 	rd41, [rd1];\n\t"
    " st.global.u64 	[rd2], rd41;\n\t"
    " st.global.relaxed.gpu.u64 	[rd1], rd3;\n\t"
    " add.s32 	r10, r10, 32;\n\t"
    " setp.ne.s32	p1, r10, 0;\n\t"
    " @p1 bra 	BB3_1;\n\t"
    " ret;\n\t"
    "}"
    :: "l"(array), "r"(size), "l"(array2) : "memory");
#endif
}


int main() {
    std::vector<std::string> names = {"ld.volatile", "st.volatile", "atomicExch", "ldst.volatile", "ldatomic"};
#if __CUDA_ARCH__ >= 700
    names.push_back("st.relaxed");
    names.push_back("ldst.relaxed");
#endif
    double** execution_times = new double*[names.size()]();
    for (unsigned i = 0; i < names.size(); i++) {
        execution_times[i] = new double[NTIMES + 1]();
    }
    unsigned long long int *array1_host, *array2_host, *array1_device, *array2_device;
    unsigned thread_num = BLOCK_NUM * BLOCK_SIZE;
    unsigned array1_size_in_byte = sizeof(unsigned long long int) * ARRAY_SIZE;
    unsigned array2_size_in_byte = sizeof(unsigned long long int) * thread_num;
    array1_host = (unsigned long long int*)malloc(array1_size_in_byte);
    array2_host = (unsigned long long int*)malloc(array2_size_in_byte);
    for (unsigned i = 0; i < ARRAY_SIZE; i++) {
        array1_host[i] = i;
    }
    cudaErrorCheck(cudaMalloc(&array1_device, array1_size_in_byte));
    cudaErrorCheck(cudaMalloc(&array2_device, array2_size_in_byte));
    cudaErrorCheck(cudaMemcpy(array1_device, array1_host, array1_size_in_byte, cudaMemcpyHostToDevice));
    for (unsigned i = 0; i <= NTIMES; i++) {
        auto start_ld_volatile = std::chrono::high_resolution_clock::now();
        testLDVolatile<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE, array2_device);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_ld_volatile = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_ld_volatile = end_ld_volatile - start_ld_volatile;
        execution_times[0][i] = elapsed_time_ld_volatile.count();

        auto start_st_volatile = std::chrono::high_resolution_clock::now();
        testSTVolatile<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_st_volatile = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_st_volatile = end_st_volatile - start_st_volatile;
        execution_times[1][i] = elapsed_time_st_volatile.count();

        auto start_atomic = std::chrono::high_resolution_clock::now();
        testAtomicExch<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_atomic = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_atomic = end_atomic - start_atomic;
        execution_times[2][i] = elapsed_time_atomic.count();
        
        auto start_ldst_volatile = std::chrono::high_resolution_clock::now();
        testLDSTVolatile<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE, array2_device);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_ldst_volatile = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_ldst_volatile = end_ldst_volatile - start_ldst_volatile;
        execution_times[3][i] = elapsed_time_ldst_volatile.count();

        auto start_ldatomic = std::chrono::high_resolution_clock::now();
        testLDAtomic<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE, array2_device);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_ldatomic = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_ldatomic = end_ldatomic - start_ldatomic;
        execution_times[4][i] = elapsed_time_ldatomic.count();
#if __CUDA_ARCH__ >= 700
        auto start_st_relaxed = std::chrono::high_resolution_clock::now();
        testSTRelaxed<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_st_relaxed = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_st_relaxed = end_st_relaxed - start_st_relaxed;
        execution_times[5][i] = elapsed_time_st_relaxed.count();

        auto start_ldst_relaxed = std::chrono::high_resolution_clock::now();
        testLDSTRelaxed<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE, array2_device);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_ldst_relaxed = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_ldst_relaxed = end_ldst_relaxed - start_ldst_relaxed;
        execution_times[6][i] = elapsed_time_ldst_relaxed.count();
#endif
    }

    cudaErrorCheck(cudaMemcpy(array1_host, array1_device, array1_size_in_byte, cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(array2_host, array2_device, array2_size_in_byte, cudaMemcpyDeviceToHost));
    //verify
    //for (unsigned i = 0; i < thread_num; i++) {
        //if (array2_host[i] != array1_host[ARRAY_SIZE - 2]) {
            //std::cout << "Mismatch at " << i << ", value is " << array2_host[i] << " " << array1_host[ARRAY_SIZE - 2] << std::endl;
        //}
    //}
    for (unsigned i = 0; i < names.size(); i++) {
        std::cout << names[i];
        double average = 0;
        double min = DBL_MAX;
        double max = 0;
        for (unsigned j = 1; j <= NTIMES; j++) {
            if (min > execution_times[i][j]) {
                min = execution_times[i][j];
            } else if (max < execution_times[i][j]) {
                max = execution_times[i][j];
            }
            average += execution_times[i][j];
        }
        average /= NTIMES;
        std::cout << " Average " << average << " Min " << min << " Max " << max << std::endl;
    }
    
    return 0;
}
