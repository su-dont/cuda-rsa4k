#include "DeviceWrapper.h"
#include "BuildConfig.h"
#include "BigInteger.h"

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

typedef struct
{
	memory32byte result[DeviceWrapper::ADDITION_CELLS_PER_THREAD];
	unsigned int borrow;
	// 4 byte borrow offsets to another memory bank, which eliminates bank conflicts

} subtractionSharedMemory;

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

extern "C" __global__ void device_get_clock(int* result)
{
	// todo	
}

extern "C" __global__ void device_clone_partial(int* x, const int* y)
{
	register int index = threadIdx.x;
	x[index] = y[index];
}

extern "C" __global__ void device_clear_partial(int* x)
{
	register int index = threadIdx.x;
	x[index] = x[index] ^ x[index];
}

extern "C" __global__ void device_equals_partial(int* result, const int* x, const int* y)
{
	register int index = threadIdx.x;
	register int globalIndex = index << 1;

	__shared__ int shared[7][256];

	shared[0][index] = x[index] ^ y[index];
	__syncthreads();
	
#pragma unroll
	for (int i = 1; i < 7; i++)
	{
		if (1 << (7 - i) > index)	// todo: 6 bank conflicts left
			shared[i][index] = shared[i - 1][globalIndex + 1] | shared[i - 1][globalIndex];
		__syncthreads();
	}
	if (index == 0)
	{
		*result = shared[6][globalIndex + 1] | shared[6][globalIndex];
	}
}

__device__ inline int merge(int x, int y)
{
	return x == 0 ? y : x;
}

// returns:
// 0 if  x == y
// 1 if x > y
// -1 if x < y
extern "C" __global__ void device_compare_partial(int* result, const int* x, const int* y)
{
	register int index = threadIdx.x;
	register int globalIndex = index << 1;

	register unsigned int xValue = x[index];
	register unsigned int yValue = y[index];

	__shared__ int shared[7][256];

	shared[0][index] = xValue == yValue ? 0 : (xValue > yValue ? 1 : -1);
	__syncthreads();

#pragma unroll
	for (int i = 1; i < 7; i++)
	{
		if (1 << (7 - i) > index)	// todo: 6 bank conflicts left
			shared[i][index] = merge(shared[i - 1][globalIndex + 1], shared[i - 1][globalIndex]);
		__syncthreads();
	}
	if (index == 0)
	{
		*result = merge(shared[6][globalIndex + 1], shared[6][globalIndex]);		
	}
}

extern "C" __global__ void device_getbitlength_partial(int* result, const int* x)
{
	register int index = threadIdx.x;
	register unsigned int value = x[index];
	register int bits = 0;

	__shared__ int shared[128];

#pragma unroll
	for (int i = 0; i < 32; i++)
	{
		if (value >> i == 1)
			bits = i + 1;
	}

	shared[index] = bits;

	__syncthreads();
		
	if (index == 0)
	{
		*result = 0;
		register int set = 0;
#pragma unroll
		for (int i = 127; i >= 0; i--)
		{
			if (shared[i] != 0 && set == 0)
			{
				*result = (i << 5) + shared[i];
				set = 1;
			}
		}		
	}
}

__device__ inline bool inBounds128(int index)
{
	return index >= 0 && index <= 127;
}

extern "C" __global__ void device_shift_left_partial(int* x, int n)
{
	register int index = threadIdx.x;
	register int ints = n >> 5;		// n / 32
	register int bits = n & 0x1f;	// n mod 32

	__shared__ unsigned int sharedX[BigInteger::ARRAY_SIZE];
	__shared__ unsigned int sharedResult[BigInteger::ARRAY_SIZE + 1];

	sharedX[index] = inBounds128(index - ints) ? x[index - ints] : 0;
	sharedResult[index] = 0UL;

	__syncthreads();

	register int remainingBits = 32 - bits;
	sharedResult[index + 1] = sharedX[index] >> remainingBits;
	__syncthreads();
	sharedResult[index] = sharedResult[index] | sharedX[index] << bits;
	__syncthreads();

	if (bits > 0)
		x[index] = sharedResult[index];
	else
		x[index] = sharedX[index];	// dummy store - constant time execution
	
}

extern "C" __global__ void device_shift_right_partial(int* x, int n)
{
	register int index = threadIdx.x;
	register int ints = n >> 5;		// n / 32
	register int bits = n & 0x1f;	// n mod 32

	__shared__ unsigned int sharedX[BigInteger::ARRAY_SIZE + 1];
	__shared__ unsigned int sharedResult[BigInteger::ARRAY_SIZE + 1];

	sharedX[index] = inBounds128(index + ints) ? x[index + ints] : 0UL;	
	sharedResult[index] = 0UL;

	__syncthreads();

	register int remainingBits = 32 - bits;
	if (index - 1 >= 0)
		sharedResult[index - 1] = sharedX[index] << remainingBits;
	else
		sharedResult[127] = 0UL;
	__syncthreads();
	sharedResult[index] = sharedResult[index] | (sharedX[index] >> bits);
	__syncthreads();

	if (bits > 0)
		x[index] = sharedResult[index];
	else
		x[index] = sharedX[index];	// dummy store - constant time execution
	
}

// x and y must be 128 unsigned ints allocated
// result return in x
extern "C" __global__ void device_add_partial(int* x_arg, const int* y_arg)
{
	// offsets to next 'row' of flatten array
	register int* x = x_arg + (blockIdx.x << 8);
	register const int* y = y_arg + (blockIdx.x << 8);

	register const int resultIndex = threadIdx.x;
	register const int startIndex = resultIndex << 2;	// * DeviceWrapper::ADDITION_CELLS_PER_THREAD;

	// 32 threads + 1 to avoid out of bounds exception
	__shared__ additionSharedMemory shared[33];
	shared[resultIndex].carry = 0UL;

	register int index = 0;

	asm volatile (
		"add.cc.u32 %0, %1, %2; \n\t"	// first iteration - only genarate carry
		: "=r"(shared[resultIndex].result[index].value)
		: "r"(x[startIndex + index]), "r"(y[startIndex + index]) : "memory");

	index++;

	asm volatile (
		"addc.cc.u32 %0, %1, %2; \n\t"	// propagate and genarate carry
		: "=r"(shared[resultIndex].result[index].value)
		: "r"(x[startIndex + index]), "r"(y[startIndex + index]) : "memory");

	index++;

	asm volatile (
		"addc.cc.u32 %0, %1, %2; \n\t"	// propagate and genarate carry
		: "=r"(shared[resultIndex].result[index].value)
		: "r"(x[startIndex + index]), "r"(y[startIndex + index]) : "memory");

	index++;

	// last iteration generates and stores carry in the array
	asm volatile (
		"addc.cc.u32 %0, %2, %3; \n\t"
		"addc.u32 %1, 0, 0; \n\t"
		: "=r"(shared[resultIndex].result[index].value), "=r"(shared[resultIndex + 1].carry)
		: "r"(x[startIndex + index]), "r"(y[startIndex + index]) : "memory");

	__syncthreads();

	register unsigned int carry;

#pragma unroll
	for (register int i = 0; i < DeviceWrapper::ONE_WARP; i++)
	{
		index = 0;
		carry = shared[resultIndex].carry;

		// first iteration propagates carry from array
		asm volatile (
			"add.cc.u32 %0, %0, %1; \n\t"	//  
			: "+r"(shared[resultIndex].result[index].value)
			: "r"(carry) : "memory");

#pragma unroll
		for (index = 1; index < DeviceWrapper::ADDITION_CELLS_PER_THREAD - 1; index++)
		{
			asm volatile (
				"addc.cc.u32 %0, %0, 0; \n\t"	//propagate generated carries
				: "+r"(shared[resultIndex].result[index].value) :: "memory");
		}

		// last iteration generates and stores carry in the array
		asm volatile (
			"addc.cc.u32 %0, %0, 0; \n\t"
			"addc.u32 %1, 0, 0; \n\t"
			: "+r"(shared[resultIndex].result[index].value), "=r"(shared[resultIndex + 1].carry) :: "memory");

		__syncthreads();
	}

#pragma unroll
	for (index = 0; index < DeviceWrapper::ADDITION_CELLS_PER_THREAD; index++)
	{
		// store result in x		
		x[startIndex + index] = shared[resultIndex].result[index].value;
	}
}

// x and y must 128 unsigned ints allocated
// result return in x
extern "C" __global__ void device_subtract_partial(int* x_arg, const int* y_arg)
{
	// offsets to next 'row' of flatten array
	register int* x = x_arg + (blockIdx.x << 8);	// * 128 * 2
	register const int* y = y_arg + (blockIdx.x << 8);

	register const int resultIndex = threadIdx.x;
	register const int startIndex = resultIndex << 2; // * DeviceWrapper::ADDITION_CELLS_PER_THREAD;

	// 32 threads + 1 to avoid out of bounds exception
	__shared__ subtractionSharedMemory shared[33];

	shared[resultIndex].borrow = 0UL;
	__syncthreads();

	register int index = 0;

	asm volatile (
		"sub.cc.u32 %0, %1, %2; \n\t"	//first interation - only genarate borrow out
		: "=r"(shared[resultIndex].result[index].value)
		: "r"(x[startIndex + index]), "r"(y[startIndex + index]) : "memory");

	index++;
		
	asm volatile (
		"subc.cc.u32 %0, %1, %2; \n\t"	// genarate and propagate borrow out
		: "=r"(shared[resultIndex].result[index].value)
		: "r"(x[startIndex + index]), "r"(y[startIndex + index]) : "memory");

	index++;

	asm volatile (
		"subc.cc.u32 %0, %1, %2; \n\t"	// genarate and propagate borrow out
		: "=r"(shared[resultIndex].result[index].value)
		: "r"(x[startIndex + index]), "r"(y[startIndex + index]) : "memory");
		
	index++;

	// last iteration generates and stores borrow in the array
	asm volatile (
		"subc.cc.u32 %0, %2, %3; \n\t"
		"subc.u32 %1, 1, 0; \n\t"	// if borrow out than %1 has 0 (1-0-1=0), else %1 has 1 (1-0-0=1)
		"xor.b32 %1, %1, 1; \n\t"	// invert 1-->0 and 0-->1
		: "=r"(shared[resultIndex].result[index].value), "+r"(shared[resultIndex + 1].borrow)
		: "r"(x[startIndex + index]), "r"(y[startIndex + index]) : "memory");

	__syncthreads();

	register unsigned int borrow;
#pragma unroll
	for (register int i = 0; i < DeviceWrapper::ONE_WARP; i++)
	{
		index = 0;
		borrow = shared[resultIndex].borrow;

		// first iteration propagates borrow from array
		asm volatile (
			"sub.cc.u32 %0, %0, %1; \n\t"
			: "+r"(shared[resultIndex].result[index].value)
			: "r"(borrow) : "memory");

#pragma unroll
		for (index = 1; index < DeviceWrapper::ADDITION_CELLS_PER_THREAD - 1; index++)
		{
			asm volatile (
				"subc.cc.u32 %0, %0, 0; \n\t"	//propagate generated borrows
				: "+r"(shared[resultIndex].result[index].value) :: "memory");
		}

		__syncthreads();

		// last iteration generates and stores borrow in the array
		asm volatile (
			"subc.cc.u32 %0, %0, 0; \n\t"
			"subc.u32 %1, 1, 0; \n\t"
			"xor.b32 %1, %1, 1; \n\t"	// invert 1-->0 and 0-->1
			: "+r"(shared[resultIndex].result[index].value), "+r"(shared[resultIndex + 1].borrow) :: "memory");

		__syncthreads();
	}

	
#pragma unroll
	for (index = 0; index < DeviceWrapper::ADDITION_CELLS_PER_THREAD; index++)
	{
		// store result in x
		x[startIndex + index] = shared[resultIndex].result[index].value;
	}

}

__device__ inline int isXodd(int config)
{
	return ((0xFFFFFFFD | config) == 0xFFFFFFFF) ? 1 : 0;
}

__device__ inline int isYodd(int config)
{
	return ((0xFFFFFFFE | config) == 0xFFFFFFFF) ? 1 : 0;
}

extern "C" __global__ void device_multiply_partial(int* result_arg, const int* x, const int* y)
{
	register const int arraySize = BigInteger::ARRAY_SIZE;
	register int* result = result_arg;

	// memory banks(32) * (threads(64) + padding(2)) = 32 * 66 = 2112
	__shared__ unsigned int sharedResult[2112];
	__shared__ unsigned int carries[2112];

	// offesets to proper result array index
	result = result + (blockIdx.x << 7); // * arraySize;

	register const int xIndex = (threadIdx.x << 1) + isXodd(blockIdx.x);

	sharedResult[deviceIndexFixupTable[xIndex]] = 0;
	sharedResult[deviceIndexFixupTable[xIndex + 1]] = 0;
	carries[deviceIndexFixupTable[xIndex]] = 0;
	carries[deviceIndexFixupTable[xIndex + 1]] = 0;
	
#pragma unroll
	for (register int yIndex = isYodd(blockIdx.x); yIndex < arraySize; yIndex = yIndex + 2)
	{
		if (xIndex + yIndex > arraySize + 1)
			break;

		register unsigned int carry = carries[deviceIndexFixupTable[xIndex + yIndex]];
		carries[deviceIndexFixupTable[xIndex + yIndex]] = 0;

		asm volatile (
			"add.cc.u32 %0, %0, %5; \n\t"
			"mad.lo.cc.u32 %0, %3, %4, %0; \n\t"
			"madc.hi.cc.u32 %1, %3, %4, %1; \n\t"
			"addc.u32 %2, %2, 0; \n\t"
			: "+r"(sharedResult[deviceIndexFixupTable[xIndex + yIndex]]), "+r"(sharedResult[deviceIndexFixupTable[xIndex + yIndex + 1]]), "+r"(carries[deviceIndexFixupTable[xIndex + yIndex + 2]])
			: "r"(x[xIndex]), "r"(y[yIndex]), "r"(carry) : "memory");

		__syncthreads();
	}
			 
	result[xIndex] = sharedResult[deviceIndexFixupTable[xIndex]];
	if (xIndex + 1 < 128)	
		result[xIndex + 1] = sharedResult[deviceIndexFixupTable[xIndex + 1]];

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
	checkCuda(cudaStreamCreate(&mainStream));
	checkCuda(cudaMalloc(&deviceOneWord, sizeof(int)));
	checkCuda(cudaMalloc(&device4arrays, sizeof(int) * 128 * 4));
}

DeviceWrapper::~DeviceWrapper()
{		
	checkCuda(cudaFree(deviceOneWord));
	checkCuda(cudaFree(device4arrays));

	checkCuda(cudaStreamSynchronize(mainStream));
	checkCuda(cudaStreamDestroy(mainStream));
}

int* DeviceWrapper::init(int size) const
{	
	int* result;
	checkCuda(cudaMalloc(&result, sizeof(unsigned int) * size));
	
	// launch config
	dim3 blocks(1);
	dim3 threads(size);

	device_clear_partial << <blocks, threads, 0, mainStream >> > (result);
	checkCuda(cudaStreamSynchronize(mainStream));	

	return result;
}

int* DeviceWrapper::init(int size, const int* initial) const
{
	int* result;
	checkCuda(cudaMalloc(&result, sizeof(unsigned int) * size));

	// launch config
	dim3 blocks(1);
	dim3 threads(size);

	device_clone_partial << <blocks, threads, 0, mainStream >> > (result, initial);
	checkCuda(cudaStreamSynchronize(mainStream));

	return result;
}

void DeviceWrapper::updateDevice(int* device_array, const unsigned int* host_array, int size) const
{	
	int bytes = sizeof(unsigned int) * size;
	checkCuda(cudaMemcpyAsync(device_array, host_array, bytes, cudaMemcpyHostToDevice, mainStream));
	checkCuda(cudaStreamSynchronize(mainStream));	
}

void DeviceWrapper::updateHost(unsigned int* host_array, const int* device_array, int size) const
{	
	int bytes = sizeof(unsigned int) * size;
	checkCuda(cudaMemcpyAsync(host_array, device_array, bytes, cudaMemcpyDeviceToHost, mainStream));
	checkCuda(cudaStreamSynchronize(mainStream));	
}

void DeviceWrapper::free(int* device_x) const
{
	checkCuda(cudaFree(device_x));
}

unsigned long long DeviceWrapper::getClock(void)
{
	unsigned long long clock;
	unsigned long long* deviceClock;
	checkCuda(cudaMalloc(&deviceClock, sizeof(unsigned long long)));
	
//	device_get_clock << <1, 1>> > (deviceClock);

	checkCuda(cudaMemcpy(&clock, deviceClock, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	checkCuda(cudaFree(deviceClock));
	
	return clock;
}

void DeviceWrapper::clearParallel(int* device_x) const
{	
	// launch config
	dim3 blocks(1);
	dim3 threads(BigInteger::ARRAY_SIZE);	// 128

	device_clear_partial << <blocks, threads, 0, mainStream >> > (device_x);
	
	checkCuda(cudaStreamSynchronize(mainStream));	
}

// x := y
void DeviceWrapper::cloneParallel(int* device_x, const int* device_y) const
{
	// launch config
	dim3 blocks(1);
	dim3 threads(BigInteger::ARRAY_SIZE);	//128
	
	device_clone_partial << <blocks, threads, 0, mainStream >> > (device_x, device_y);	

	checkCuda(cudaStreamSynchronize(mainStream));	
}

// returns:
// 0 if  x == y
// 1 if x > y
// -1 if x < y
int DeviceWrapper::compareParallel(const int* device_x, const int* device_y) const
{
	int result;		
	
	// launch config
	dim3 blocks(1);
	dim3 threads(BigInteger::ARRAY_SIZE);	//128

	device_compare_partial << <blocks, threads, 0, mainStream >> > (deviceOneWord, device_x, device_y);

	checkCuda(cudaMemcpyAsync(&result, deviceOneWord, sizeof(int), cudaMemcpyDeviceToHost, mainStream));
	
	checkCuda(cudaStreamSynchronize(mainStream));

	return result;
}

bool DeviceWrapper::equalsParallel(const int* device_x, const int* device_y) const
{
	int result;

	// launch config
	dim3 blocks(1);
	dim3 threads(BigInteger::ARRAY_SIZE);	//128

	device_equals_partial << <blocks, threads, 0, mainStream >> > (deviceOneWord, device_x, device_y);

	checkCuda(cudaMemcpyAsync(&result, deviceOneWord, sizeof(int), cudaMemcpyDeviceToHost, mainStream));

	checkCuda(cudaStreamSynchronize(mainStream));

	return result == 0;
}

int DeviceWrapper::getLSB(const int* device_x) const
{
	int result;
	checkCuda(cudaMemcpyAsync(&result, device_x, sizeof(int), cudaMemcpyDeviceToHost, mainStream));
	checkCuda(cudaStreamSynchronize(mainStream));
	
	return result & 0x01;
}

int DeviceWrapper::getBitLength(const int* device_x) const
{
	int result;

	// launch config
	dim3 blocks(1);
	dim3 threads(BigInteger::ARRAY_SIZE);	//128

	device_getbitlength_partial << <blocks, threads, 0, mainStream >> > (deviceOneWord, device_x);

	checkCuda(cudaMemcpyAsync(&result, deviceOneWord, sizeof(int), cudaMemcpyDeviceToHost, mainStream));

	checkCuda(cudaStreamSynchronize(mainStream));

	return result;
}

void DeviceWrapper::shiftLeftParallel(int* device_x, int bits) const
{	
	if (bits == 0)
		return;

	if (DEBUG)
	{
		int bitLength = getBitLength(device_x);
		if (bits > 4096 - bitLength)
		{
			std::cerr << "ERROR: DeviceWrapper::shiftLeftParallel: overflow - trying to shift left by " << bits;
			std::cerr << " whereas this bit length is already" << bitLength << endl;
		}
	}
	
	// launch config
	dim3 blocks(1);
	dim3 threads(BigInteger::ARRAY_SIZE);	// 128

	device_shift_left_partial << <blocks, threads, 0, mainStream >> > (device_x, bits);

	checkCuda(cudaStreamSynchronize(mainStream));	
}

void DeviceWrapper::shiftRightParallel(int* device_x, int bits) const
{	
	if (bits == 0)
		return;
	
	// launch config
	dim3 blocks(1);
	dim3 threads(BigInteger::ARRAY_SIZE);	// 128

	device_shift_right_partial << <blocks, threads, 0, mainStream >> > (device_x, bits);

	checkCuda(cudaStreamSynchronize(mainStream));		
}

void DeviceWrapper::addParallel(int* device_x, const int* device_y) const
{
	// launch config
	dim3 blocks(1);
	dim3 threads(DeviceWrapper::ONE_WARP);
	addParallel(device_x, device_y, blocks, threads);
}

inline void DeviceWrapper::addParallel(int * device_x, const int * device_y, dim3 blocks, dim3 threads) const
{
	int* xBitLength;
	int arrays = blocks.x;
	int** aux_device_x;
	if (DEBUG)
	{	
		xBitLength = new int[arrays];
		for (int i = 0; i < arrays; i++)
		{
			xBitLength[i] = getBitLength(device_x + i * 256);
		}		
		if (ERROR_CHECKING)
		{
			aux_device_x = new int*[arrays];
			int size = sizeof(unsigned int) * 128;
			for (int i = 0; i < arrays; i++)
			{
				checkCuda(cudaMalloc(&aux_device_x[i], size));
				checkCuda(cudaMemcpyAsync(aux_device_x[i], device_x + i * 256, size, cudaMemcpyDeviceToDevice, mainStream));
				checkCuda(cudaStreamSynchronize(mainStream));
			}
		}
	}

	device_add_partial << <blocks, threads, 0, mainStream >> > (device_x, device_y);
	checkCuda(cudaStreamSynchronize(mainStream));

	if (DEBUG)
	{
		int* yBitLength;	
		int* resultBitLength;
		yBitLength = new int[arrays];
		resultBitLength = new int[arrays];
		for (int i = 0; i < arrays; i++)
		{
			yBitLength[i] = getBitLength(device_y + i * 256);
			resultBitLength[i] = getBitLength(device_x + i * 256);
		}

		for (int i = 0; i < arrays; i++)
		{
			// comparing bitwise lengths of the result and parameters to quickly detect overflows
			if (resultBitLength[i] < xBitLength[i] || resultBitLength[i] < yBitLength[i])
			{
				std::cerr << "ERROR: DeviceWrapper::addParallel: Overflow: bit length difference block.x=" << i;
				std::cerr << " x=" << xBitLength[i] << " y=" << yBitLength[i] << " result=" << resultBitLength[i] << endl;
			}
		}

		if (ERROR_CHECKING)
		{
			bool equals;
			int compare;
			for (int i = 0; i < arrays; i++)
			{
				// after addition x equals x + y
				// subtracting initial x from result x should give initial y
				compare = compareParallel(device_x + i * 256, aux_device_x[i]);
				if (compare == -1)
				{
					std::cerr << "ERROR: DeviceWrapper::addParallel: possible overflow x + y < y in block.x:" << i << endl;
				}
				else {
					// WARNING: in this checking result (device_x) is modified.
					// Use ERROR_CHECKING only for debugging
					subtractParallel(device_x + i * 256, aux_device_x[i]);
					equals = equalsParallel(device_x + i * 256, device_y + i * 256);
					if (!equals)
					{
						std::cerr << "ERROR: DeviceWrapper::addParallel: y != (x + y) - x in block.x:" << i << endl;
					}
				}				
			}			
		}
	}
}

void DeviceWrapper::subtractParallel(int* device_x, const int* device_y) const
{
	if (DEBUG)
	{
		if (compareParallel(device_x, device_y) == -1) // x < y
		{
			std::cerr << "ERROR: DeviceWrapper::subtractParallel: x < y - negative output" << endl;
		}
	}

	// launch config
	dim3 blocks(1);
	dim3 threads(DeviceWrapper::ONE_WARP);

	device_subtract_partial << <blocks, threads, 0, mainStream >> > (device_x, device_y);
		
	checkCuda(cudaStreamSynchronize(mainStream));	
}

void DeviceWrapper::multiplyParallel(int* device_x, const int* device_y) const
{	
	// launch config
	dim3 blocks(DeviceWrapper::MULTIPLICATION_BLOCKS_COUNT);
	dim3 threads(DeviceWrapper::TWO_WARPS);

	// parallel multiplication
	device_multiply_partial << <blocks, threads, 0, mainStream>> > (device4arrays, device_x, device_y);

	// reduction
	blocks.x = 2;
	threads.x = DeviceWrapper::ONE_WARP;
	addParallel(device4arrays, device4arrays + 128, blocks, threads);	

	// reduction
	blocks.x = 1;
	addParallel(device4arrays, device4arrays + 256 , blocks, threads);

	// set x := result
	threads.x = BigInteger::ARRAY_SIZE;
	device_clone_partial << <blocks, threads, 0, mainStream >> > (device_x, device4arrays);
	
	checkCuda(cudaStreamSynchronize(mainStream));
}

