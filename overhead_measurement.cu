#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cfloat>
#include <iostream>

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

__global__ void testAtomicExch(unsigned long long int* array, unsigned size) {
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


std::string names[] = {"ld.volatile", "st.volatile", "atomicExch", "ldst", "ldatomic"};

int main() {
    double execution_times[5][NTIMES + 1];
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
        auto start_ld = std::chrono::high_resolution_clock::now();
        testLDVolatile<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE, array2_device);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_ld = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_ld = end_ld - start_ld;
        execution_times[0][i] = elapsed_time_ld.count();

        auto start_st = std::chrono::high_resolution_clock::now();
        testSTVolatile<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_st = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_st = end_st - start_st;
        execution_times[1][i] = elapsed_time_st.count();

        auto start_atomic = std::chrono::high_resolution_clock::now();
        testAtomicExch<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_atomic = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_atomic = end_atomic - start_atomic;
        execution_times[2][i] = elapsed_time_atomic.count();
        
        auto start_ldst = std::chrono::high_resolution_clock::now();
        testLDSTVolatile<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE, array2_device);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_ldst = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_ldst = end_ldst - start_ldst;
        execution_times[3][i] = elapsed_time_ldst.count();

        auto start_ldatomic = std::chrono::high_resolution_clock::now();
        testLDAtomic<<<BLOCK_NUM, BLOCK_SIZE>>>(array1_device, ARRAY_SIZE, array2_device);
        cudaErrorCheck(cudaGetLastError());
        cudaErrorCheck(cudaDeviceSynchronize());
        auto end_ldatomic = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time_ldatomic = end_ldatomic - start_ldatomic;
        execution_times[4][i] = elapsed_time_ldatomic.count();
    }

    cudaErrorCheck(cudaMemcpy(array1_host, array1_device, array1_size_in_byte, cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(array2_host, array2_device, array2_size_in_byte, cudaMemcpyDeviceToHost));
    //verify
    //for (unsigned i = 0; i < thread_num; i++) {
        //if (array2_host[i] != array1_host[ARRAY_SIZE - 2]) {
            //std::cout << "Mismatch at " << i << ", value is " << array2_host[i] << " " << array1_host[ARRAY_SIZE - 2] << std::endl;
        //}
    //}
    for (unsigned i = 0; i < 5; i++) {
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
