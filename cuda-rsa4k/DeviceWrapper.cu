#include "DeviceWrapper.h"
#include "BigInteger.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

typedef struct
{
	unsigned int value;	
	unsigned int padding[31];
	// padding to match with 32 byte memory line

} memory32byte;

typedef struct
{
	memory32byte result[DeviceWrapper::ADDITION_CELLS_PER_THREAD];
	unsigned int carry;	
	// 4 byte carry offsets to another memory bank, which eliminates bank conflicts

} additionSharedMemory;

__host__ __device__ inline int isXodd(int config)
{
	return ((0xFFFFFFFD | config) == 0xFFFFFFFF) ? 1 : 0;
}

__host__ __device__ inline int isYodd(int config)
{
	return ((0xFFFFFFFE | config) == 0xFFFFFFFF) ? 1 : 0;
}

extern "C" __global__ void device_get_clock(unsigned long long* result)
{
	// todo	
}

// x and y must (128 + 1) unsigned ints allocated to account for overflow
// result return in x
extern "C" __global__ void device_add_partial(unsigned int* x, unsigned int* y)
{
	x = x + blockIdx.x * 2 * (BigInteger::ARRAY_SIZE + 1);
	y = y + blockIdx.x * 2 * (BigInteger::ARRAY_SIZE + 1);

	register const int resultIndex = threadIdx.x;
	register const int startIndex = resultIndex * DeviceWrapper::ADDITION_CELLS_PER_THREAD;

	__shared__ additionSharedMemory shared[BigInteger::ARRAY_SIZE / DeviceWrapper::ADDITION_CELLS_PER_THREAD + 1];

	register int index;

#pragma unroll
	for (index = 0; index < DeviceWrapper::ADDITION_CELLS_PER_THREAD - 1; index++)
	{
		asm volatile (
			"addc.cc.u32 %0, %1, %2; \n\t"	// genarate and propagate carry
			: "=r"(shared[resultIndex].result[index].value)
			: "r"(x[startIndex + index]), "r"(y[startIndex + index]));
	}

	// last iteration generates and stores carry in the array
	asm volatile (
		"addc.cc.u32 %0, %2, %3; \n\t"
		"addc.u32 %1, 0, 0; \n\t"
		: "=r"(shared[resultIndex].result[index].value), "=r"(shared[resultIndex + 1].carry)
		: "r"(x[startIndex + index]), "r"(y[startIndex + index]));

	__syncthreads();

	register unsigned int carry;
#pragma unroll
	for (register int i = 0; i < DeviceWrapper::ADDITION_THREAD_COUNT; i++)
	{
		index = 0;
		carry = shared[resultIndex].carry;

		// first iteration propagates carry from array
		asm volatile (
			"add.cc.u32 %0, %0, %1; \n\t"	//  
			: "+r"(shared[resultIndex].result[index].value)
			: "r"(carry));

#pragma unroll
		for (index = 1; index < DeviceWrapper::ADDITION_CELLS_PER_THREAD - 1; index++)
		{
			asm volatile (
				"addc.cc.u32 %0, %0, 0; \n\t"	//propagate generated carries
				: "+r"(shared[resultIndex].result[index].value));
		}

		// last iteration generates and stores carry in the array
		asm volatile (
			"addc.cc.u32 %0, %0, 0; \n\t"
			"addc.u32 %1, 0, 0; \n\t"
			: "+r"(shared[resultIndex].result[index].value), "=r"(shared[resultIndex + 1].carry));

		__syncthreads();
	}

#pragma unroll
	for (index = 0; index < DeviceWrapper::ADDITION_CELLS_PER_THREAD; index++)
	{
		// store result in x
		x[startIndex + index] = shared[resultIndex].result[index].value;
	}

	__syncthreads();
}



__constant__ unsigned int deviceIndexFixupTable[129];


extern "C" __global__ void device_multiply_partial(unsigned int* result, const unsigned int* x, const unsigned int* y)
{
	register const int arraySize = BigInteger::ARRAY_SIZE + 1;
	register const int sharedMemoryLines = DeviceWrapper::MULTIPLICATION_THREAD_COUNT + 2;
	register const int memoryBanksCount = 32;

	__shared__ unsigned int sharedResult[memoryBanksCount * sharedMemoryLines];
	__shared__ unsigned int carries[memoryBanksCount * sharedMemoryLines];

	// offesets to proper result array index
	result = result + blockIdx.x * (BigInteger::ARRAY_SIZE + 1);	

	register const int xIndex = threadIdx.x * 2 + isXodd(blockIdx.x);

	sharedResult[deviceIndexFixupTable[xIndex]] = 0;
	sharedResult[deviceIndexFixupTable[xIndex + 1]] = 0;
	carries[deviceIndexFixupTable[xIndex]] = 0;
	carries[deviceIndexFixupTable[xIndex + 1]] = 0;

#pragma unroll
	for (register int yIndex = isYodd(blockIdx.x); yIndex < arraySize; yIndex = yIndex + 2)
	{
		if (xIndex + yIndex >= arraySize)
			break;

		register unsigned int carry = carries[deviceIndexFixupTable[xIndex + yIndex]];
		carries[deviceIndexFixupTable[xIndex + yIndex]] = 0;

		asm volatile (
			"add.cc.u32 %0, %0, %5; \n\t"
			"mad.lo.cc.u32 %0, %3, %4, %0; \n\t"
			"madc.hi.cc.u32 %1, %3, %4, %1; \n\t"
			"addc.u32 %2, %2, 0; \n\t"
			: "+r"(sharedResult[deviceIndexFixupTable[xIndex + yIndex]]), "+r"(sharedResult[deviceIndexFixupTable[xIndex + yIndex + 1]]), "+r"(carries[deviceIndexFixupTable[xIndex + yIndex + 2]])
			: "r"(x[xIndex]), "r"(y[yIndex]), "r"(carry));

		__syncthreads();
	}
			
	result[xIndex] = sharedResult[deviceIndexFixupTable[xIndex]];
	result[xIndex + 1] = sharedResult[deviceIndexFixupTable[xIndex + 1]];

	__syncthreads();
}


inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
	}
	return result;
}

DeviceWrapper::DeviceWrapper()
{
	checkCuda(cudaMemcpyToSymbol(deviceIndexFixupTable, indexFixupTable, sizeof(unsigned int) * 129));
}

DeviceWrapper::~DeviceWrapper()
{
	delete[] indexFixupTable;
}

unsigned long long DeviceWrapper::getClock(void)
{
	unsigned long long clock;
	unsigned long long* deviceClock;
	checkCuda(cudaMalloc(&deviceClock, sizeof(unsigned long long)));
	
	device_get_clock << <1, 1 >> > (deviceClock);

	checkCuda(cudaMemcpy(&clock, deviceClock, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	checkCuda(cudaFree(deviceClock));
	
	return clock;
}

unsigned int* DeviceWrapper::addParallel(const BigInteger& x, const BigInteger& y)
{
	unsigned int* resultArray = new unsigned int[BigInteger::ARRAY_SIZE];	

	int size = sizeof(unsigned int) * BigInteger::ARRAY_SIZE;	

	unsigned int* device_x;
	unsigned int* device_y;

	checkCuda(cudaMalloc(&device_x, size + sizeof(unsigned int)));	// + 1 to check for overflow
	checkCuda(cudaMalloc(&device_y, size + sizeof(unsigned int)));

	checkCuda(cudaMemcpy(device_x, x.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(device_y, y.getMagnitudeArray(), size, cudaMemcpyHostToDevice));

	device_add_partial << <1, DeviceWrapper::ADDITION_THREAD_COUNT >> > (device_x, device_y);

	checkCuda(cudaMemcpy(resultArray, device_x, size, cudaMemcpyDeviceToHost));

	unsigned int overflow;
	checkCuda(cudaMemcpy(&overflow, device_x + BigInteger::ARRAY_SIZE, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	if (overflow != 0UL)
	{
		std::cerr << "ERROR: BigInteger::add overflow!" << endl;
		throw std::overflow_error("BigInteger::add overflow");
	}

	checkCuda(cudaFree(device_x));
	checkCuda(cudaFree(device_y));

	return resultArray;
}

unsigned int* DeviceWrapper::multiplyParallel(const BigInteger& x, const BigInteger& y)
{
	unsigned int* resultArray = new unsigned int[BigInteger::ARRAY_SIZE];

	int size = sizeof(unsigned int) * BigInteger::ARRAY_SIZE;
	int deviceResultArraySize = size + sizeof(unsigned int);

	unsigned int* device_result;
	unsigned int* device_x;
	unsigned int* device_y;

	// device memory allocations
	checkCuda(cudaMalloc(&device_result, deviceResultArraySize * 4));	// 4 times for every block
	checkCuda(cudaMalloc(&device_x, size));
	checkCuda(cudaMalloc(&device_y, size));

	checkCuda(cudaMemcpy(device_x, x.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(device_y, y.getMagnitudeArray(), size, cudaMemcpyHostToDevice));

	dim3 blocks(DeviceWrapper::MULTIPLICATION_BLOCKS_COUNT);
	dim3 threads(DeviceWrapper::MULTIPLICATION_THREAD_COUNT);

	device_multiply_partial << <blocks, threads>> > (device_result, device_x, device_y);

	// reduction
	blocks.x = 2;
	threads.x = DeviceWrapper::ADDITION_THREAD_COUNT;
	device_add_partial << <blocks, threads >> > (device_result, device_result + 129);

	// reduction
	blocks.x = 1;
	device_add_partial << <blocks, threads >> > (device_result, device_result + 258);
	
	// copy result to the host
	checkCuda(cudaMemcpy(resultArray, device_result, size, cudaMemcpyDeviceToHost));
	
	unsigned int overflow;
	checkCuda(cudaMemcpy(&overflow, device_result + BigInteger::ARRAY_SIZE, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	if (overflow != 0UL)
	{
		std::cerr << "ERROR: BigInteger::multiply overflow!" << endl;
		//throw std::overflow_error("BigInteger::multiply overflow");
	}

	// clear memory
	checkCuda(cudaFree(device_result));
	checkCuda(cudaFree(device_x));
	checkCuda(cudaFree(device_y));

	// todo overflow?

	return resultArray;
}


