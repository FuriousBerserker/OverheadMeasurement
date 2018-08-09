#include <cstdio>
#include <cstddef>
#include <cfloat>
#include <chrono>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 100
#endif

#ifndef ARRAY_TYPE
#define ARRAY_TYPE double
#endif

#ifndef BLOCK_NUM
#define BLOCK_NUM 100
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

#ifndef WINDOW_SIZE
#define WINDOW_SIZE 4
#endif

#ifndef NTIMES
#define NTIMES 10
#endif

#ifndef LOOP_TIME
#define LOOP_TIME 1000000
#endif

#define cudaErrorCheck(call)                                               \
    do {                                                                   \
        cudaError_t cuErr = call;                                          \
        if (cudaSuccess != cuErr) {                                        \
            printf("CUDA Error - %s:%d: '%s, %s'\n", __FILE__, __LINE__,   \
                cudaGetErrorName(cuErr), cudaGetErrorString(cuErr));       \
            exit(0);                                                       \
        }                                                                  \
    } while(0)

#ifdef SHADOW_MEMORY
class ShadowMemory {
   private:
    static unsigned offsetPatterns[4];
    unsigned long long int bits[WINDOW_SIZE];

   public:
    static const unsigned long long int EMPTY = 0;
    ShadowMemory() {
        for (unsigned i = 0; i < WINDOW_SIZE; i++) {
            bits[i] = EMPTY;
        }
    }
    friend __device__ __noinline__ void insertSM(ptrdiff_t address,
                                    unsigned threadID, bool isWrite,
                                    unsigned size);
    unsigned getThreadID(unsigned index) {
        return (unsigned)(this->bits[index] >> 48);
    }

    unsigned long long int getClock(unsigned index) {
        return (this->bits[index] >> 6) & 0x000003FFFFFFFFFF;
    }

    bool isWrite(unsigned index) {
        return ((this->bits[index] >> 5) & 0x0000000000000001) ==
                       0x0000000000000001
                   ? true
                   : false;
    }

    unsigned getAccessSize(unsigned index) {
        unsigned patternIndex =
            (this->bits[index] >> 3) & 0x0000000000000003;
        return offsetPatterns[patternIndex];
    }

    unsigned getAddressOffset(unsigned index) {
        return (unsigned)(this->bits[index] & 0x0000000000000007);
    }

    void outputSM() {
        for (unsigned i = 0; i < WINDOW_SIZE; i++) {
            printf(
                "Cell ID = %d, Thread ID = %d, Clock = %lld, Access mode = %s, Access size = "
                "%d, Offset = %d\n",
                i, getThreadID(i), getClock(i), isWrite(i) ? "write" : "read",
                getAccessSize(i), getAddressOffset(i));
        }
    }
};

unsigned ShadowMemory::offsetPatterns[] = {1, 2, 4, 8};

unsigned smSize =
    ((unsigned)ARRAY_SIZE * sizeof(ARRAY_TYPE) + 7) / 8;

__device__ ShadowMemory *sm;

__device__ __noinline__ void insertSM(ptrdiff_t address,
                         unsigned threadID, bool isWrite,
                         unsigned size) {
    unsigned index = address / 8;
    unsigned offset = address % 8;
    unsigned clock = 0xC0DA;
    unsigned encodedSize = 0;
    while (!(size & 0x0000000000000001)) {
        encodedSize++;
        size >>= 1;
    }

    unsigned long long int bit = 0x0000000000000000;
    bit |= (threadID & 0x000000000000FFFF);
    bit <<= 42;
    bit |= (clock & 0x000003FFFFFFFFFF);
    bit <<= 1;
    bit |= (isWrite ? 0x0000000000000001 : 0x0000000000000000);
    bit <<= 2;
    bit |= encodedSize;
    bit <<= 3;
    bit |= (offset & 0x0000000000000007);

    unsigned nextIndex = WINDOW_SIZE;
    for (unsigned i = 0; i < WINDOW_SIZE; i++) {
        unsigned long long int temp;
        temp = *(volatile unsigned long long int*)(&sm[index].bits[i]);
        if (temp == ShadowMemory::EMPTY && nextIndex == WINDOW_SIZE) {
            nextIndex = i;
        }
    }
    if (nextIndex == WINDOW_SIZE) {
        nextIndex = (address >> 3) % WINDOW_SIZE;
    }
#ifdef USE_CAS
    atomicExch(&sm[index].bits[nextIndex], bit);
#else
    *(volatile unsigned long long int*)(&sm[index].bits[nextIndex]) = bit;
#endif
}

void printShadowMemory(ShadowMemory* sm, unsigned size, unsigned limit = 10, unsigned stride = 1) {
    for (unsigned i = 0; i < limit && i * stride < size; i++) {
        sm[i * stride].outputSM();
    }
}

#endif

__global__ void initialize(double *array1, double *array2, unsigned size1, unsigned size2) {
    unsigned stride = gridDim.x * blockDim.x;
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    while (index < size1) {
        array1[index] = (double)index;
        index += stride;
    }
    index = blockDim.x * blockIdx.x + threadIdx.x;
    while (index < size2) {
        array2[index] = 0;
        index += stride;
    }

}

__global__ void conflictAccess(double *array1, double* array2, unsigned size1) {
    unsigned threadID = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned sum = 0;
    for (int i = 0; i < LOOP_TIME; i++) {
        sum += array1[5];
#ifdef SHADOW_MEMORY
        insertSM(5 * sizeof(ARRAY_TYPE), threadID, false, sizeof(ARRAY_TYPE));
#endif
    }
    array2[threadID] = sum;
}


int main() {
    ARRAY_TYPE *array1OnDevice, *array2OnDevice;
#ifdef SHADOW_MEMORY
    ShadowMemory *smOnDevice;
#endif
    double executionTime[NTIMES + 1];
    unsigned array2Size = BLOCK_NUM * BLOCK_SIZE;
    cudaErrorCheck(cudaMalloc(&array1OnDevice, sizeof(ARRAY_TYPE) * ARRAY_SIZE));
    cudaErrorCheck(cudaMalloc(&array2OnDevice, sizeof(ARRAY_TYPE) * array2Size));
#ifdef SHADOW_MEMORY
    cudaErrorCheck(cudaMalloc(&smOnDevice, sizeof(ShadowMemory) * smSize));
    cudaErrorCheck(cudaMemcpyToSymbol(sm, &smOnDevice, sizeof(ShadowMemory*), 0, cudaMemcpyHostToDevice));
#endif
    initialize<<<BLOCK_NUM, BLOCK_SIZE>>>(array1OnDevice, array2OnDevice, ARRAY_SIZE, array2Size);
    cudaErrorCheck(cudaGetLastError());
#ifdef SHADOW_MEMORY
    ShadowMemory* smOnHost = new ShadowMemory[smSize]();
    cudaErrorCheck(cudaMemcpy(smOnDevice, smOnHost, sizeof(ShadowMemory) * smSize, cudaMemcpyHostToDevice));
#endif
    for (int i = 0; i <= NTIMES; i++) {
        auto startTime = std::chrono::high_resolution_clock::now();
        conflictAccess<<<BLOCK_NUM, BLOCK_SIZE>>>(array1OnDevice, array2OnDevice, ARRAY_SIZE);
        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaGetLastError());
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = endTime - startTime;
        executionTime[i] = elapsedTime.count();
    }
   
    ARRAY_TYPE* array2OnHost = new ARRAY_TYPE[array2Size];
    cudaErrorCheck(cudaMemcpy(array2OnHost, array2OnDevice, sizeof(ARRAY_TYPE) * array2Size, cudaMemcpyDeviceToHost));

    ARRAY_TYPE expectedResult = 0;
    for (unsigned i = 0; i < LOOP_TIME; i++) {
        expectedResult += 5;
    }
    
    unsigned errorNum = 0;
    for (unsigned i = 0; i < array2Size; i++) {
        //printf("array2[%d] = %f\n", i, array2OnHost[i]);
        if (array2OnHost[i] != expectedResult) {
            printf("Mismatch at %d, expected = %f, actual = %f\n", i, expectedResult, array2OnHost[i]);
            errorNum++;
        }
    }
    if (errorNum == 0) {
        printf("The calculation on the device is correct\n");
    } else {
        printf("Total amount of erroenous results = %d\n", errorNum);
    }

    cudaFree(array1OnDevice);
    cudaFree(array2OnDevice);
#ifdef SHADOW_MEMORY
    cudaErrorCheck(cudaMemcpy(smOnHost, smOnDevice, sizeof(ShadowMemory) * smSize, cudaMemcpyDeviceToHost));
    printShadowMemory(smOnHost, smSize);
    cudaFree(smOnDevice);
#endif

    double minTime = DBL_MAX, maxTime = 0, averageTime = 0;
    for (unsigned i = 1; i <= NTIMES; i++) {
        if (executionTime[i] < minTime) {
            minTime = executionTime[i];
        } else if (executionTime[i] > maxTime) {
            maxTime = executionTime[i];
        }
        averageTime += executionTime[i];
    }
    averageTime /= NTIMES;
    printf("Evaluation Result: \n");
    printf("Min: %f Max: %f Average: %f\n", minTime, maxTime, averageTime);
    return 0;
}
