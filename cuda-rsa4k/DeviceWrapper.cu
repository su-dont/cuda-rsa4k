#include "DeviceWrapper.h"

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

/*
Mapping to sepcific index value in order to eliminate bank conflicts
This function adds complexity which result in efficiency loss.
Index values from 0 to 129 have been tableized in indexFixupTable for each thread separately. 
Unused*/
__host__ __device__ inline unsigned int indexFixup(const unsigned int index)
{
	return index % 64 * 32 + (index % 64 & 0xfffffffe) / 2 + index / 64 * 64;	
}

//parallel multiplication config
static const int EVEN_EVEN = 0x0;
static const int EVEN_ODD = 0x1;
static const int ODD_EVEN = 0x2;
static const int ODD_ODD = 0x3;

__host__ __device__ inline int isXodd(int config)
{
	return ((0xFFFFFFFD | config) == 0xFFFFFFFF) ? 1 : 0;
}

__host__ __device__ inline int isYodd(int config)
{
	return ((0xFFFFFFFE | config) == 0xFFFFFFFF) ? 1 : 0;
}

extern "C" __global__ void device_add(unsigned int* result, const unsigned int* x, const unsigned int* y)
{
	// implementation in DeviceWrapper.ptx	
}

extern "C" __global__ void device_multiply(unsigned int* result, const unsigned int* x, const unsigned int* y)
{
	// implementation in DeviceWrapper.ptx	
}

extern "C" __global__ void device_get_clock(unsigned long long* result)
{
	// implementation in DeviceWrapper.ptx	
}

extern "C" __global__ void device_multiply_partial(unsigned int* result, const unsigned int* x, const unsigned int* y, const int config)
{
	const unsigned int indexFixupTable[] {0, 32, 65, 97, 130, 162, 195, 227, 260, 292, 325, 357, 390, 422, 455, 487, 520, 552, 585, 617, 650, 682, 715, 747, 780, 812, 845, 877, 910, 942, 975, 1007, 1040, 1072, 1105, 1137, 1170, 1202, 1235, 1267, 1300, 1332, 1365, 1397, 1430, 1462, 1495, 1527, 1560, 1592, 1625, 1657, 1690, 1722, 1755, 1787, 1820, 1852, 1885, 1917, 1950, 1982, 2015, 2047, 64, 96, 129, 161, 194, 226, 259, 291, 324, 356, 389, 421, 454, 486, 519, 551, 584, 616, 649, 681, 714, 746, 779, 811, 844, 876, 909, 941, 974, 1006, 1039, 1071, 1104, 1136, 1169, 1201, 1234, 1266, 1299, 1331, 1364, 1396, 1429, 1461, 1494, 1526, 1559, 1591, 1624, 1656, 1689, 1721, 1754, 1786, 1819, 1851, 1884, 1916, 1949, 1981, 2014, 2046, 2079, 2111, 128};

	register const int arraySize = BigInteger::ARRAY_SIZE + 1;
	register const int sharedMemoryLines = DeviceWrapper::MULTIPLICATION_THREAD_COUNT + 2;
	register const int memoryBanksCount = 32;

	__shared__ unsigned int sharedResult[memoryBanksCount * sharedMemoryLines];
	__shared__ unsigned int carries[memoryBanksCount * sharedMemoryLines];

	register const int xIndex = threadIdx.x * 2 + isXodd(config);

	sharedResult[indexFixupTable[xIndex]] = 0;
	sharedResult[indexFixupTable[xIndex + 1]] = 0;
	carries[indexFixupTable[xIndex]] = 0;
	carries[indexFixupTable[xIndex + 1]] = 0;

#pragma unroll
	for (register int yIndex = isYodd(config); yIndex < arraySize; yIndex = yIndex + 2)
	{
		if (xIndex + yIndex >= arraySize)
			break;

		register unsigned int carry = carries[indexFixupTable[xIndex + yIndex]];
		carries[indexFixupTable[xIndex + yIndex]] = 0;

		asm volatile (
			"add.cc.u32 %0, %0, %5; \n\t"
			"mad.lo.cc.u32 %0, %3, %4, %0; \n\t"
			"madc.hi.cc.u32 %1, %3, %4, %1; \n\t"
			"addc.u32 %2, %2, 0; \n\t"
			: "+r"(sharedResult[indexFixupTable[xIndex + yIndex]]), "+r"(sharedResult[indexFixupTable[xIndex + yIndex + 1]]), "+r"(carries[indexFixupTable[xIndex + yIndex + 2]])
			: "r"(x[xIndex]), "r"(y[yIndex]), "r"(carry));

		__syncthreads();
	}
			
	result[xIndex] = sharedResult[indexFixupTable[xIndex]];
	result[xIndex + 1] = sharedResult[indexFixupTable[xIndex + 1]];

	__syncthreads();
}

extern "C" __global__ void device_add_partial(unsigned int* result, const unsigned int* x, const unsigned int* y)
{
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
		result[startIndex + index] = shared[resultIndex].result[index].value;
	}

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
}

DeviceWrapper::~DeviceWrapper()
{
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

BigInteger* DeviceWrapper::add(BigInteger& x, BigInteger& y)
{
	//todo: vaildate x,y

	unsigned int* resultArray = new unsigned int[BigInteger::ARRAY_SIZE + 1];	// + 1 to check for overflow

	int size = sizeof(unsigned int) * BigInteger::ARRAY_SIZE;
	
	unsigned int* device_result;
	unsigned int* device_x;
	unsigned int* device_y;

	checkCuda(cudaMalloc(&device_result, size + sizeof(unsigned int)));
	checkCuda(cudaMalloc(&device_x, size));
	checkCuda(cudaMalloc(&device_y, size));
	
	checkCuda(cudaMemcpy(device_x, x.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(device_y, y.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
	
	device_add << <1, 1 >> > (device_result, device_x, device_y);
	
	checkCuda(cudaMemcpy(resultArray, device_result, size + sizeof(unsigned int), cudaMemcpyDeviceToHost));

	unsigned int overflow = resultArray[128];
	if (overflow != 0UL)
	{
		std::cerr << "ERROR: BigInteger::add overflow!" << endl;
		throw std::overflow_error("BigInteger::add overflow");
	}

	checkCuda(cudaFree(device_result));
	checkCuda(cudaFree(device_x));
	checkCuda(cudaFree(device_y));
	
	return new BigInteger(resultArray);;
}

BigInteger* DeviceWrapper::addParallel(BigInteger& x, BigInteger& y)
{
	//todo: vaildate x,y

	unsigned int* resultArray = new unsigned int[BigInteger::ARRAY_SIZE + 1];	// + 1 to check for overflow

	int size = sizeof(unsigned int) * BigInteger::ARRAY_SIZE;

	unsigned int* device_result;
	unsigned int* device_x;
	unsigned int* device_y;

	checkCuda(cudaMalloc(&device_result, size + sizeof(unsigned int)));
	checkCuda(cudaMalloc(&device_x, size));
	checkCuda(cudaMalloc(&device_y, size));

	checkCuda(cudaMemcpy(device_x, x.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(device_y, y.getMagnitudeArray(), size, cudaMemcpyHostToDevice));

	device_add_partial << <1, DeviceWrapper::ADDITION_THREAD_COUNT >> > (device_result, device_x, device_y);

	checkCuda(cudaMemcpy(resultArray, device_result, size + sizeof(unsigned int), cudaMemcpyDeviceToHost));

	unsigned int overflow = resultArray[128];
	if (overflow != 0UL)
	{
		std::cerr << "ERROR: BigInteger::add overflow!" << endl;
		throw std::overflow_error("BigInteger::add overflow");
	}

	checkCuda(cudaFree(device_result));
	checkCuda(cudaFree(device_x));
	checkCuda(cudaFree(device_y));

	return new BigInteger(resultArray);;
}

BigInteger* DeviceWrapper::multiply(BigInteger& x, BigInteger& y)
{
	//todo: vaildate x,y

	// resultArray twice as long to account for overflow 
	int resultArraySize = BigInteger::ARRAY_SIZE * 2;
	unsigned int* resultArray = new unsigned int[resultArraySize];
	
	int size = sizeof(unsigned int) * BigInteger::ARRAY_SIZE;

	unsigned int* device_result;
	unsigned int* device_x;
	unsigned int* device_y;

	checkCuda(cudaMalloc(&device_result, size * 2));	// resultArray twice as long to account for overflow 
	checkCuda(cudaMalloc(&device_x, size));
	checkCuda(cudaMalloc(&device_y, size));

	checkCuda(cudaMemcpy(device_x, x.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(device_y, y.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
		
	device_multiply << <1, 1 >> > (device_result, device_x, device_y);

	checkCuda(cudaMemcpy(resultArray, device_result, size * 2, cudaMemcpyDeviceToHost));

	unsigned int overflow = 0UL;
	for (int i = resultArraySize - 1; i >= BigInteger::ARRAY_SIZE; i--)
	{
		overflow = overflow | resultArray[i];
	}

	if (overflow != 0UL)
	{
		std::cerr << "ERROR: BigInteger::multiply overflow!" << endl;
		throw std::overflow_error("BigInteger::multiply overflow");
	}

	checkCuda(cudaFree(device_result));
	checkCuda(cudaFree(device_x));
	checkCuda(cudaFree(device_y));

	return new BigInteger(resultArray);;
}

BigInteger* DeviceWrapper::multiplyParallel(BigInteger& x, BigInteger& y)
{
	unsigned int* resultArray = new unsigned int[BigInteger::ARRAY_SIZE + 1]; // + 1 to check for overflow
	
	int size = sizeof(unsigned int) * BigInteger::ARRAY_SIZE;
	int resultArraySize = sizeof(unsigned int) * (BigInteger::ARRAY_SIZE + 1);

	unsigned int* device_result_even_even;
	unsigned int* device_result_even_odd;
	unsigned int* device_result_odd_even;
	unsigned int* device_result_odd_odd;

	unsigned int* device_result_even;
	unsigned int* device_result_odd;

	unsigned int* device_result;

	unsigned int* device_x;
	unsigned int* device_y;
	
	// device memory allocations
	checkCuda(cudaMalloc(&device_result_even_even, resultArraySize));
	checkCuda(cudaMalloc(&device_result_even_odd, resultArraySize));
	checkCuda(cudaMalloc(&device_result_odd_even, resultArraySize));
	checkCuda(cudaMalloc(&device_result_odd_odd, resultArraySize));

	checkCuda(cudaMalloc(&device_result_even, resultArraySize));
	checkCuda(cudaMalloc(&device_result_odd, resultArraySize));
	
	checkCuda(cudaMalloc(&device_result, resultArraySize));

	checkCuda(cudaMalloc(&device_x, size));
	checkCuda(cudaMalloc(&device_y, size));

	checkCuda(cudaMemcpy(device_x, x.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(device_y, y.getMagnitudeArray(), size, cudaMemcpyHostToDevice));

	// two parallel streams for indepentent computations
	cudaStream_t evenStream, oddStream;
	checkCuda(cudaStreamCreate(&evenStream));
	checkCuda(cudaStreamCreate(&oddStream));

	// kernel launches
	device_multiply_partial << <1, DeviceWrapper::MULTIPLICATION_THREAD_COUNT, 0, evenStream >> > (device_result_even_even, device_x, device_y, EVEN_EVEN);
	device_multiply_partial << <1, DeviceWrapper::MULTIPLICATION_THREAD_COUNT, 0, evenStream >> > (device_result_even_odd, device_x, device_y, EVEN_ODD);
	device_multiply_partial << <1, DeviceWrapper::MULTIPLICATION_THREAD_COUNT, 0, oddStream >> > (device_result_odd_even, device_x, device_y, ODD_EVEN);
	device_multiply_partial << <1, DeviceWrapper::MULTIPLICATION_THREAD_COUNT, 0, oddStream >> > (device_result_odd_odd, device_x, device_y, ODD_ODD);

	// reduction
	device_add_partial << <1, DeviceWrapper::ADDITION_THREAD_COUNT, 0, evenStream >> > (device_result_even, device_result_even_even, device_result_even_odd);
	device_add_partial << <1, DeviceWrapper::ADDITION_THREAD_COUNT, 0, oddStream >> > (device_result_odd, device_result_odd_even, device_result_odd_odd);

	checkCuda(cudaStreamSynchronize(evenStream));
	checkCuda(cudaStreamSynchronize(oddStream));

	// reduction
	device_add_partial << <1, DeviceWrapper::ADDITION_THREAD_COUNT >> > (device_result, device_result_even, device_result_odd);
	
	// copy result to the host
	checkCuda(cudaMemcpy(resultArray, device_result, resultArraySize, cudaMemcpyDeviceToHost));

	// kill streams
	checkCuda(cudaStreamDestroy(evenStream));
	checkCuda(cudaStreamDestroy(oddStream));
	
	unsigned int overflow = resultArray[128];
	if (overflow != 0UL)
	{
		std::cerr << "ERROR: BigInteger::multiply overflow!" << endl;
		//throw std::overflow_error("BigInteger::multiply overflow");
	}

	// clear memory
	checkCuda(cudaFree(device_result_even_even));
	checkCuda(cudaFree(device_result_even_odd));
	checkCuda(cudaFree(device_result_odd_even));
	checkCuda(cudaFree(device_result_odd_odd));

	checkCuda(cudaFree(device_result_even));
	checkCuda(cudaFree(device_result_odd));

	checkCuda(cudaFree(device_result));

	checkCuda(cudaFree(device_x));
	checkCuda(cudaFree(device_y));

	BigInteger* result = new BigInteger(resultArray);
	if (result->getLength() < x.getLength() || result->getLength() < y.getLength())
	{
		std::cerr << "ERROR: BigInteger::multiply overflow!" << endl;
		//throw std::overflow_error("BigInteger::multiply overflow");
	}

	return result;
}


