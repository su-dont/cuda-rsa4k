#include "DeviceWrapper.h"
#include "BigInteger.h"
#include "BuildConfig.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// shared memory stuctures
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

//Mapping to sepcific indices of shared memory in order to eliminate bank conflicts in device_multiply_partial
//Dependency: 
// return index % 64 * 32 + (index % 64 & 0xfffffffe) / 2 + index / 64 * 64;
__constant__ unsigned int deviceIndexFixupTable[] { 0, 32, 65, 97, 130, 162, 195, 227, 260, 292, 325, 357,390, 422, 455,
487, 520, 552, 585, 617, 650, 682, 715, 747, 780, 812, 845, 877, 910, 942, 975, 1007, 1040, 1072, 1105, 1137,1170, 1202,
1235, 1267, 1300, 1332, 1365, 1397, 1430, 1462, 1495, 1527, 1560, 1592, 1625, 1657, 1690, 1722, 1755, 1787,1820, 1852,
1885, 1917, 1950, 1982, 2015, 2047, 64, 96, 129, 161, 194, 226, 259, 291, 324, 356, 389, 421, 454, 486, 519,551, 584,
616, 649, 681, 714, 746, 779, 811, 844, 876, 909, 941, 974, 1006, 1039, 1071, 1104, 1136, 1169, 1201, 1234, 1266,1299,
1331, 1364, 1396, 1429, 1461, 1494, 1526, 1559, 1591, 1624, 1656, 1689, 1721, 1754, 1786, 1819, 1851, 1884, 1916, 1949,
1981, 2014, 2046, 2079, 2111, 128 };

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

// x and y must 128 unsigned ints allocated
// result return in x
extern "C" __global__ void device_add_partial(unsigned int* x, unsigned int* y)
{
	// offsets to next 'row' of flatten array
	x = x + blockIdx.x * 256;
	y = y + blockIdx.x * 256;

	register const int resultIndex = threadIdx.x;
	register const int startIndex = resultIndex * DeviceWrapper::ADDITION_CELLS_PER_THREAD;

	// 32 threads + 1 to avoid out of bounds exception
	__shared__ additionSharedMemory shared[33];

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

extern "C" __global__ void device_multiply_partial(unsigned int* result, const unsigned int* x, const unsigned int* y)
{
	register const int arraySize = BigInteger::ARRAY_SIZE;

	// memory banks(32) * (threads(64) + padding(2)) = 32 * 66 = 2112
	__shared__ unsigned int sharedResult[2112];
	__shared__ unsigned int carries[2112];

	// offesets to proper result array index
	result = result + blockIdx.x * arraySize;

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
	if (xIndex + 1 < 128)	
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
	cudaStreamCreate(&mainStream);
	cudaStreamCreate(&memoryCopyStream);
}

DeviceWrapper::~DeviceWrapper()
{
	checkCuda(cudaStreamDestroy(mainStream));
	checkCuda(cudaStreamDestroy(memoryCopyStream));
}

unsigned long long DeviceWrapper::getClock(void)
{
	unsigned long long clock;
	unsigned long long* deviceClock;
	checkCuda(cudaMalloc(&deviceClock, sizeof(unsigned long long)));
	
	device_get_clock << <1, 1>> > (deviceClock);

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

	checkCuda(cudaMalloc(&device_x, size));
	checkCuda(cudaMalloc(&device_y, size));

	cudaEvent_t event;
	checkCuda(cudaEventCreate(&event));

	// async memory copy
	checkCuda(cudaMemcpyAsync(device_x, x.getMagnitudeArray(), size, cudaMemcpyHostToDevice, memoryCopyStream));
	checkCuda(cudaEventRecord(event, memoryCopyStream));	// record x copy finish
	checkCuda(cudaMemcpyAsync(device_y, y.getMagnitudeArray(), size, cudaMemcpyHostToDevice, mainStream));

	// launch config
	dim3 blocks(1);
	dim3 threads(DeviceWrapper::ADDITION_THREAD_COUNT);

	checkCuda(cudaStreamWaitEvent(mainStream, event, 0));	// wait for x,y to finish
	device_add_partial << <blocks, threads, 0, mainStream >> > (device_x, device_y);

	checkCuda(cudaEventDestroy(event));

	checkCuda(cudaMemcpyAsync(resultArray, device_x, size, cudaMemcpyDeviceToHost, mainStream));	
	checkCuda(cudaFree(device_y));	

	cudaStreamSynchronize(mainStream);
	checkCuda(cudaFree(device_x));	

	if (DEBUG)
	{
		// analizing result's length with inputs' lengths
		// to detect possible overflow
		int resultLength = 128, xLength = 128, yLength = 128;
		bool resultSet = false, xSet = false, ySet = false;
		for (int i = 127; i >= 0; i--)
		{
			if (x.getMagnitudeArray()[i] == 0UL && !xSet)
				xLength--;
			else
				xSet = true;

			if (y.getMagnitudeArray()[i] == 0UL && !ySet)
				yLength--;
			else
				ySet = true;

			if (resultArray[i] == 0UL && !resultSet)
				resultLength--;
			else
				resultSet = true;
		}

		if (resultLength < xLength || resultLength < yLength)
		{
			std::cerr << "ERROR: BigInteger::add overflow! -- length difference" << endl;
		}
	}

	return resultArray;
}

unsigned int* DeviceWrapper::multiplyParallel(const BigInteger& x, const BigInteger& y)
{
	unsigned int* resultArray = new unsigned int[BigInteger::ARRAY_SIZE];

	int size = sizeof(unsigned int) * BigInteger::ARRAY_SIZE;

	unsigned int* device_result;
	unsigned int* device_x;
	unsigned int* device_y;

	// device memory allocations
	checkCuda(cudaMalloc(&device_result, size * 4));	// 4 times for every block
	checkCuda(cudaMalloc(&device_x, size));
	checkCuda(cudaMalloc(&device_y, size));

	cudaEvent_t event;
	checkCuda(cudaEventCreate(&event));

	// async memory copy
	checkCuda(cudaMemcpy(device_x, x.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
	checkCuda(cudaEventRecord(event, memoryCopyStream));	// record x copy finish
	checkCuda(cudaMemcpy(device_y, y.getMagnitudeArray(), size, cudaMemcpyHostToDevice));

	// launch config
	dim3 blocks(DeviceWrapper::MULTIPLICATION_BLOCKS_COUNT);
	dim3 threads(DeviceWrapper::MULTIPLICATION_THREAD_COUNT);

	checkCuda(cudaStreamWaitEvent(mainStream, event, 0));	// wait for x,y to finish
	device_multiply_partial << <blocks, threads, 0, mainStream>> > (device_result, device_x, device_y);
	
	checkCuda(cudaEventDestroy(event));

	// reduction
	blocks.x = 2;
	threads.x = DeviceWrapper::ADDITION_THREAD_COUNT;
	device_add_partial << <blocks, threads, 0, mainStream >> > (device_result, device_result + 128);

	// reduction
	blocks.x = 1;
	device_add_partial << <blocks, threads, 0, mainStream >> > (device_result, device_result + 256);
	
	// copy result to the host
	checkCuda(cudaMemcpyAsync(resultArray, device_result, size, cudaMemcpyDeviceToHost, mainStream));
	
	// clear memory
	checkCuda(cudaFree(device_x));
	checkCuda(cudaFree(device_y));

	cudaStreamSynchronize(mainStream);
	checkCuda(cudaFree(device_result));	

	if (DEBUG)
	{
		// analizing result's length with inputs' lengths
		// to detect possible overflow
		int resultLength = 128, xLength = 128, yLength = 128;
		bool resultSet = false, xSet = false, ySet = false;
		for (int i = 127; i >= 0; i--)
		{
			if (x.getMagnitudeArray()[i] == 0UL && !xSet)
				xLength--;
			else
				xSet = true;

			if (y.getMagnitudeArray()[i] == 0UL && !ySet)
				yLength--;
			else
				ySet = true;

			if (resultArray[i] == 0UL && !resultSet)
				resultLength--;
			else
				resultSet = true;
		}

		if (resultLength < xLength || resultLength < yLength)
		{
			std::cerr << "ERROR: BigInteger::multiply overflow! -- length difference" << endl;
		}
	}

	return resultArray;
}


