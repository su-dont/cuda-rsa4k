#include "DeviceWrapper.h"
#include "BuildConfig.h"
#include "BigInteger.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

const int ADDITION_THREADS = 32;	// ONE_WARP
const int ADDITION_CELLS_PER_THREAD = 4;	// BigInteger::ARRAY_SIZE / ONE_WARP

// shared memory stuctures
typedef struct
{
	unsigned int value;
	unsigned int padding[31];
	// padding to match with 32 byte memory line

} memory32byte;

typedef struct
{
	memory32byte result[ADDITION_CELLS_PER_THREAD];
	unsigned int carry;
	// 4 byte carry offsets to another memory bank, which eliminates bank conflicts

} additionSharedMemory;

typedef struct
{
	memory32byte result[ADDITION_CELLS_PER_THREAD];
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


////////////////////////////////////
// DEVICE FUNCTIONS
///////////////////////////////////


extern "C" __device__ inline void device_clone_partial(unsigned int* x, const unsigned int* y)
{
	register int index = threadIdx.x;
	x[index] = y[index];
}

extern "C" __device__ inline void device_clear_partial(unsigned int* x)
{
	register int index = threadIdx.x;
	x[index] = x[index] ^ x[index];
}

extern "C" __device__ inline int device_equals_partial(const unsigned int* x, const unsigned int* y)
{
	register int index = threadIdx.x;
	register int globalIndex = index << 1;

	__shared__ unsigned int shared[7][256];

	shared[0][index] = x[index] ^ y[index];
	__syncthreads();
		
#pragma unroll
	for (int i = 1; i < 7; i++)
	{
		if (1 << (7 - i) > index)	// todo: 6 bank conflicts left
			shared[i][index] = shared[i - 1][globalIndex + 1] | shared[i - 1][globalIndex];
		__syncthreads();
	}
	return shared[6][1] | shared[6][0];
}

extern "C" __device__ inline int merge(int x, int y)
{
	return x == 0 ? y : x;
}

// returns:
// 0 if  x == y
// 1 if x > y
// -1 if x < y
extern "C" __device__  int inline device_compare_partial(const unsigned int* x, const unsigned int* y)
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
	return merge(shared[6][1], shared[6][0]);
}

extern "C" __device__  inline int device_getbitlength_partial(const unsigned int* x)
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
		
	register int resultIndex = -1;

#pragma unroll
	for (int i = 0; i < 128; i++)
	{
		if (shared[i] != 0)
		{
			resultIndex = i;			
		}
	}

	if (resultIndex > -1)
		return (resultIndex << 5) + shared[resultIndex];
	else
		return 0;	
}

extern "C" __device__ inline bool inBounds128(int index)
{
	return index >= 0 && index <= 127;
}

extern "C" __device__ inline void device_shift_left_partial(unsigned int* x, int n)
{
	register int index = threadIdx.x;
	register int ints = n >> 5;		// n / 32
	register int bits = n & 0x1f;	// n mod 32

	__shared__ unsigned int sharedX[BigInteger::ARRAY_SIZE];
	__shared__ unsigned int sharedResult[BigInteger::ARRAY_SIZE + 1];

	sharedX[index] = inBounds128(index - ints) ? x[index - ints] : 0UL;
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
		x[index] = sharedX[index];
	
}

extern "C" __device__ void inline device_shift_right_partial(unsigned int* x, int n)
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
		x[index] = sharedX[index];	
}

// x and y must be 128 unsigned ints allocated
// result return in x
extern "C" __device__ inline void device_add_partial(unsigned int* x, const  unsigned int* y)
{
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
	for (register int i = 0; i < ADDITION_THREADS; i++)
	{
		index = 0;
		carry = shared[resultIndex].carry;

		// first iteration propagates carry from array
		asm volatile (
			"add.cc.u32 %0, %0, %1; \n\t"	//  
			: "+r"(shared[resultIndex].result[index].value)
			: "r"(carry) : "memory");

#pragma unroll
		for (index = 1; index < ADDITION_CELLS_PER_THREAD - 1; index++)
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
	for (index = 0; index < ADDITION_CELLS_PER_THREAD; index++)
	{
		// store result in x		
		x[startIndex + index] = shared[resultIndex].result[index].value;
	}
}

// x and y must 128 unsigned ints allocated
// result return in x
extern "C" __device__ void inline device_subtract_partial(unsigned int* x, const unsigned int* y)
{
	register const int resultIndex = threadIdx.x & 0x1f;
	register const int startIndex = resultIndex << 2; // * DeviceWrapper::ADDITION_CELLS_PER_THREAD;

	// 32 threads + 1 to avoid out of bounds exception
	__shared__ subtractionSharedMemory shared[33];
	shared[resultIndex].borrow = 0UL;

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
	for (register int i = 0; i < ADDITION_THREADS; i++)
	{
		index = 0;
		borrow = shared[resultIndex].borrow;

		// first iteration propagates borrow from array
		asm volatile (
			"sub.cc.u32 %0, %0, %1; \n\t"
			: "+r"(shared[resultIndex].result[index].value)
			: "r"(borrow) : "memory");

#pragma unroll
		for (index = 1; index < ADDITION_CELLS_PER_THREAD - 1; index++)
		{
			asm volatile (
				"subc.cc.u32 %0, %0, 0; \n\t"	//propagate generated borrows
				: "+r"(shared[resultIndex].result[index].value) :: "memory");
		}

		// last iteration generates and stores borrow in the array
		asm volatile (
			"subc.cc.u32 %0, %0, 0; \n\t"
			"subc.u32 %1, 1, 0; \n\t"
			"xor.b32 %1, %1, 1; \n\t"	// invert 1-->0 and 0-->1
			: "+r"(shared[resultIndex].result[index].value), "+r"(shared[resultIndex + 1].borrow) :: "memory");

		__syncthreads();
	}

	
#pragma unroll
	for (index = 0; index < ADDITION_CELLS_PER_THREAD; index++)
	{
		// store result in x
		x[startIndex + index] = shared[resultIndex].result[index].value;
	}
}

extern "C" __device__ inline int isXodd(int config)
{
	return ((0xFFFFFFFD | config) == 0xFFFFFFFF) ? 1 : 0;
}

extern "C" __device__ inline int isYodd(int config)
{
	return ((0xFFFFFFFE | config) == 0xFFFFFFFF) ? 1 : 0;
}

extern "C" __device__ void inline device_multiply_partial(unsigned int* result, const unsigned int* x, const unsigned int* y)
{
	register int block = blockIdx.x;
	register const int arraySize = BigInteger::ARRAY_SIZE;

	// memory banks(32) * (threads(64) + padding(2)) = 32 * 66 = 2112
	__shared__ unsigned int sharedResult[2112];
	__shared__ unsigned int carries[2112];
	
	register const int xIndex = (threadIdx.x << 1) + isXodd(block);

	sharedResult[deviceIndexFixupTable[xIndex]] = 0UL;
	sharedResult[deviceIndexFixupTable[xIndex + 1]] = 0UL;
	carries[deviceIndexFixupTable[xIndex]] = 0UL;
	carries[deviceIndexFixupTable[xIndex + 1]] = 0UL;

#pragma unroll
	for (register int yIndex = isYodd(block); yIndex < arraySize; yIndex = yIndex + 2)
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

extern "C" __device__ void inline device_reduce_modulo_partial(unsigned int* x, const unsigned int* m)
{
	register int index = threadIdx.x;

	__shared__ unsigned int modulus[128];
	device_clone_partial(modulus, m);

	register int compare = device_compare_partial(x, modulus);

	if (compare == -1)
	{
		// Trying to reduce modulo a greater integer		
		return;
	}
	if (compare == 0)
	{
		// Reducing modulo same integer
		device_clear_partial(x);
		return;
	}

	register int bitwiseDifference;

	register int xLength;
	register int mLength;

	xLength = device_getbitlength_partial(x);
	mLength = device_getbitlength_partial(modulus);

	bitwiseDifference = xLength - mLength;

	device_shift_left_partial(modulus, bitwiseDifference);
	while (bitwiseDifference >= 0) // TODO: side channel vulnerability ?
	{
		compare = device_compare_partial(x, modulus);
		if (compare == 1)	// x > m
		{
			if (index < 32)
			{
				device_subtract_partial(x, modulus);
			}
			else
			{
				__syncthreads();
				for (int i = 0; i < 32; i++)
				{
					__syncthreads();
				}
			}
		}
		else if (compare == 0)  // x == m	
		{
			device_clear_partial(x);
			__syncthreads();
			return;
		}
		else // x <= m
		{
			device_shift_right_partial(modulus, 1);
			bitwiseDifference--;
		}
		__syncthreads();
	}
}


////////////////////////////////////
// KERNEL LAUNCHES
///////////////////////////////////


extern "C" __global__ void device_get_clock(unsigned long long* time)
{
	asm volatile (
		"mov.u64 %0, %%clock64; \n\t"
		: "=l"(*time) :: "memory");
}

__global__ void device_clone_partial_1(unsigned int* x, const unsigned int* y)
{
	device_clone_partial(x, y);
}

__global__ void device_clear_partial_1(unsigned int* x)
{
	device_clear_partial(x);
}

__global__ void device_equals_partial_1(int* result, const unsigned int* x, const unsigned int* y)
{
	*result = device_equals_partial(x, y);
}

// 0 if  x == y
// 1 if x > y
// -1 if x < y
// output stored in result
__global__ void device_compare_partial_1(int* result, const unsigned int* x, const unsigned int* y)
{
	*result = device_compare_partial(x, y);
}

//// aligned arrays
//__global__ void device_compare_partial_2(int* result, const int* x, const int* y)
//{
//	register int block = blockIdx.x;
//	device_compare_partial(result + block, x + block * 128, y + block * 128);
//}
//
//// not aligned arrays
//__global__ void device_compare_partial_2(int* result, const int* x, const int* y, const int* x1, const int* y1)
//{
//	register int block = blockIdx.x;
//	if (block == 0)
//		device_compare_partial(result + block, x, y);
//	else
//		device_compare_partial(result + block, x1, y1);
//}

//// aligned arrays
//__global__ void device_compare_partial_4(int* result, const int* x, const int* y)
//{
//	register int block = blockIdx.x;
//	device_compare_partial(result + block, x + block * 128, y + block * 128);
//}

__global__ void device_getbitlength_partial_1(int* result, const unsigned int* x)
{
	*result = device_getbitlength_partial(x);
}

// not aligned arrays
__global__ void device_getbitlength_partial_2(int* result, const unsigned int* x, const unsigned int* y)
{
	register int block = blockIdx.x;
	result[block] = device_getbitlength_partial(block == 0 ? x : y);
}

// aligned arrays
__global__ void device_getbitlength_partial_4(int* result, const unsigned int* x)
{
	register int block = blockIdx.x;
	result[block] = device_getbitlength_partial(x + block * 128);
}

__global__ void device_shift_left_partial_1(unsigned int* x, int n)
{
	device_shift_left_partial(x, n);
}

// aligned arrays
__global__ void device_shift_left_partial_2(unsigned int* x, int n)
{
	register int block = blockIdx.x;
	device_shift_left_partial(x + block * 128, n);
}

// aligned arrays
__global__ void device_shift_left_partial_4(unsigned int* x, int n)
{
	register int block = blockIdx.x;
	device_shift_left_partial(x + block * 128, n);
}

__global__ void device_shift_right_partial_1(unsigned int* x, int n)
{
	device_shift_right_partial(x, n);
}

// aligned arrays
__global__ void device_shift_right_partial_2(unsigned int* x, int n)
{
	register int block = blockIdx.x;
	device_shift_right_partial(x + block * 128, n);
}

// aligned arrays
__global__ void device_shift_right_partial_4(unsigned int* x, int n)
{
	register int block = blockIdx.x;
	device_shift_right_partial(x + block * 128, n);
}

__global__ void device_add_partial_1(unsigned int* x, const unsigned int* y)
{
	device_add_partial(x, y);
}

// aligned arrays
__global__ void device_add_partial_2(unsigned int* x, const unsigned int* y)
{
	register int block = blockIdx.x;
	device_add_partial(x + block * 128, y + block * 128);
}

__global__ void device_subtract_partial_1(unsigned int* x, const unsigned int* y)
{
	device_subtract_partial(x, y);
}

// not aligned arrays
__global__ void device_subtract_partial_2(unsigned int* x, const unsigned int* y, unsigned int* x1, const unsigned int* y1)
{
	register int block = blockIdx.x;
	if (block == 0)
		device_subtract_partial(x, y);
	else
		device_subtract_partial(x1, y1);
}

// aligned arrays
__global__ void device_subtract_partial_4(unsigned int* x, const unsigned int* y)
{
	register int block = blockIdx.x;
	device_subtract_partial(x + block * 128, y + block * 128);
}

 __global__ void device_multiply_partial_4(unsigned int* result, const unsigned int* x, const unsigned int* y)
{
	register int block = blockIdx.x;
	// offesets to proper result array index	
	device_multiply_partial(result + block * 128, x, y);
}

 __global__ void device_reduce_modulo_partial_1(unsigned int* x, const unsigned int* m)
 {	 	 
	 device_reduce_modulo_partial(x, m);
 }

 // not aligned arrays
 __global__ void device_reduce_modulo_partial_2(unsigned int* x, unsigned int* y,  const unsigned int* m)
 {
	 register int block = blockIdx.x;
	 if (block == 0)
		device_reduce_modulo_partial(x, m);
	 else
		device_reduce_modulo_partial(y, m);
 }

 // aligned arrays, modulus the same in global memory
 __global__ void device_reduce_modulo_partial_4(unsigned int* x, const unsigned int* m)
 {
	 register int block = blockIdx.x;
	 device_reduce_modulo_partial(x + block * 128, m);
 }

 ////////////////////////////////////
 // HOST CODE
 ///////////////////////////////////


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
	block_1.x = 1;
	block_2.x = 2;
	block_4.x = 4;
	thread_warp.x = 32;
	thread_2_warp.x = 64;
	thread_4_warp.x = 128;

	checkCuda(cudaStreamCreate(&mainStream));
	checkCuda(cudaMalloc(&deviceWords, sizeof(int) * 4));
	checkCuda(cudaMalloc(&device4arrays, sizeof(unsigned int) * 128 * 4));
	checkCuda(cudaMalloc(&deviceArray, sizeof(unsigned int) * 128));
	checkCuda(cudaMalloc(&deviceStartTime, sizeof(unsigned long long)));
	checkCuda(cudaMalloc(&deviceStopTime, sizeof(unsigned long long)));
}

DeviceWrapper::~DeviceWrapper()
{	
	checkCuda(cudaFree(deviceWords));
	checkCuda(cudaFree(device4arrays));
	checkCuda(cudaFree(deviceArray));
	checkCuda(cudaFree(deviceStartTime));
	checkCuda(cudaFree(deviceStopTime));
	
	checkCuda(cudaStreamSynchronize(mainStream));
	checkCuda(cudaStreamDestroy(mainStream));
}

unsigned int* DeviceWrapper::init(int size) const
{	
	unsigned int* result;
	checkCuda(cudaMalloc(&result, sizeof(unsigned int) * size));	

	device_clear_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (result);
	checkCuda(cudaStreamSynchronize(mainStream));	

	return result;
}

unsigned int* DeviceWrapper::init(int size, const unsigned int* initial) const
{
	unsigned int* result;
	checkCuda(cudaMalloc(&result, sizeof(unsigned int) * size));
	
	device_clone_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (result, initial);
	checkCuda(cudaStreamSynchronize(mainStream));

	return result;
}

void DeviceWrapper::updateDevice(unsigned int* device_array, const unsigned int* host_array, int size) const
{	
	int bytes = sizeof(unsigned int) * size;
	checkCuda(cudaMemcpyAsync(device_array, host_array, bytes, cudaMemcpyHostToDevice, mainStream));
	checkCuda(cudaStreamSynchronize(mainStream));	
}

void DeviceWrapper::updateHost(unsigned int* host_array, const unsigned int* device_array, int size) const
{	
	int bytes = sizeof(unsigned int) * size;
	checkCuda(cudaMemcpyAsync(host_array, device_array, bytes, cudaMemcpyDeviceToHost, mainStream));
	checkCuda(cudaStreamSynchronize(mainStream));	
}

void DeviceWrapper::free(unsigned int* device_x) const
{
	checkCuda(cudaFree(device_x));
}

void DeviceWrapper::clearParallel(unsigned int* device_x) const
{
	device_clear_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (device_x);	
	checkCuda(cudaStreamSynchronize(mainStream));	
}

// x := y
void DeviceWrapper::cloneParallel(unsigned int* device_x, const unsigned int* device_y) const
{
	device_clone_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (device_x, device_y);
	checkCuda(cudaStreamSynchronize(mainStream));	
}

// returns:
// 0 if  x == y
// 1 if x > y
// -1 if x < y
int DeviceWrapper::compareParallel(const unsigned int* device_x, const unsigned int* device_y) const
{
	int result;		
	
	device_compare_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (deviceWords, device_x, device_y);
	checkCuda(cudaMemcpyAsync(&result, deviceWords, sizeof(int), cudaMemcpyDeviceToHost, mainStream));	
	checkCuda(cudaStreamSynchronize(mainStream));

	return result;
}

bool DeviceWrapper::equalsParallel(const unsigned int* device_x, const unsigned int* device_y) const
{
	int result;

	device_equals_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (deviceWords, device_x, device_y);
	checkCuda(cudaMemcpyAsync(&result, deviceWords, sizeof(int), cudaMemcpyDeviceToHost, mainStream));
	checkCuda(cudaStreamSynchronize(mainStream));

	return result == 0;
}

int DeviceWrapper::getLSB(const unsigned int* device_x) const
{
	int result;
	checkCuda(cudaMemcpyAsync(&result, device_x, sizeof(int), cudaMemcpyDeviceToHost, mainStream));
	checkCuda(cudaStreamSynchronize(mainStream));
	
	return result & 0x01;
}

int DeviceWrapper::getBitLength(const unsigned int* device_x) const
{
	int result;
		
	device_getbitlength_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (deviceWords, device_x);
	checkCuda(cudaMemcpyAsync(&result, deviceWords, sizeof(int), cudaMemcpyDeviceToHost, mainStream));
	checkCuda(cudaStreamSynchronize(mainStream));

	return result;
}

void DeviceWrapper::startClock(void)
{
	device_get_clock << <1, 1, 0, mainStream >> > (deviceStartTime);
	checkCuda(cudaStreamSynchronize(mainStream));
}

unsigned long long DeviceWrapper::stopClock(void)
{
	device_get_clock << <1, 1, 0, mainStream >> > (deviceStopTime);
	checkCuda(cudaStreamSynchronize(mainStream));
	unsigned long long start;
	unsigned long long stop;
	checkCuda(cudaMemcpyAsync(&start, deviceStartTime, sizeof(unsigned long long), cudaMemcpyDeviceToHost, mainStream));
	checkCuda(cudaMemcpyAsync(&stop, deviceStopTime, sizeof(unsigned long long), cudaMemcpyDeviceToHost, mainStream));
	checkCuda(cudaStreamSynchronize(mainStream));

	return stop - start;
}

void DeviceWrapper::shiftLeftParallel(unsigned int* device_x, int bits) const
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
	
	device_shift_left_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (device_x, bits);
	checkCuda(cudaStreamSynchronize(mainStream));	
}

void DeviceWrapper::shiftRightParallel(unsigned int* device_x, int bits) const
{	
	if (bits == 0)
		return;	

	device_shift_right_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (device_x, bits);
	checkCuda(cudaStreamSynchronize(mainStream));		
}

void DeviceWrapper::addParallelWithOverflow(unsigned int * device_x, const unsigned int * device_y, int blocks) const
{
	int* xBitLength;
	int arrays = blocks;
	unsigned int** aux_device_x;
	if (DEBUG)
	{	
		xBitLength = new int[arrays];
		for (int i = 0; i < arrays; i++)
		{
			xBitLength[i] = getBitLength(device_x + i * 256);
		}		
		if (ERROR_CHECKING)
		{
			aux_device_x = new unsigned int*[arrays];
			int size = sizeof(unsigned int) * 128;
			for (int i = 0; i < arrays; i++)
			{
				checkCuda(cudaMalloc(&aux_device_x[i], size));
				checkCuda(cudaMemcpyAsync(aux_device_x[i], device_x + i * 256, size, cudaMemcpyDeviceToDevice, mainStream));
				checkCuda(cudaStreamSynchronize(mainStream));
			}
		}
	}

	if (blocks == 1)
		device_add_partial_1 << <blocks, thread_warp, 0, mainStream >> > (device_x, device_y);
	else // blocks == 2
		device_add_partial_2 << <blocks, thread_warp, 0, mainStream >> > (device_x, device_y);

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

void DeviceWrapper::addParallel(unsigned int * device_x, const unsigned int * device_y) const
{
	if (DEBUG)
	{
		addParallelWithOverflow(device_x, device_y, block_1.x);
	}
	else
	{
		device_add_partial_1 << <block_1, thread_warp, 0, mainStream >> > (device_x, device_y);
	}
}

void DeviceWrapper::subtractParallel(unsigned int* device_x, const unsigned int* device_y) const
{
	if (DEBUG)
	{
		if (compareParallel(device_x, device_y) == -1) // x < y
		{
			std::cerr << "ERROR: DeviceWrapper::subtractParallel: x < y - negative output" << endl;
		}
	}
	
	device_subtract_partial_1 << <block_1, thread_warp, 0, mainStream >> > (device_x, device_y);		
	checkCuda(cudaStreamSynchronize(mainStream));	
}

void DeviceWrapper::multiplyParallel(unsigned int* device_x, const unsigned int* device_y) const
{	
	// parallel multiplication
	device_multiply_partial_4 << <block_4, thread_2_warp, 0, mainStream>> > (device4arrays, device_x, device_y);
	
	if (DEBUG)
	{
		addParallelWithOverflow(device4arrays, device4arrays + 256, block_2.x);
		addParallelWithOverflow(device4arrays, device4arrays + 128, block_1.x);
	}
	else
	{
		// reduction
		device_add_partial_2 << <block_2, thread_warp, 0, mainStream >> > (device4arrays, device4arrays + 256);

		// reduction	
		device_add_partial_1 << <block_1, thread_warp, 0, mainStream >> > (device4arrays, device4arrays + 128);
	}

	// set x := result
	device_clone_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (device_x, device4arrays);
	
	checkCuda(cudaStreamSynchronize(mainStream));
}

void DeviceWrapper::modParallel(unsigned int * device_x, unsigned int * device_m) const
{
	device_reduce_modulo_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (device_x, device_m);
	checkCuda(cudaStreamSynchronize(mainStream));
}

void DeviceWrapper::multiplyModParallel(unsigned int * device_x, const unsigned int * device_y, const unsigned int * device_m) const
{		
	device_clone_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (deviceArray, device_y);

	// reduce mod first
	device_reduce_modulo_partial_2 << <block_2, thread_4_warp, 0, mainStream >> > (device_x, deviceArray, device_m);

	// parallel multiplication
	device_multiply_partial_4 << <block_4, thread_2_warp, 0, mainStream >> > (device4arrays, device_x, deviceArray);

	if (DEBUG)
	{
		// modular reduction of part-results
		device_reduce_modulo_partial_4 << <block_4, thread_4_warp, 0, mainStream >> > (device4arrays, device_m);

		// reduction		
		addParallelWithOverflow(device4arrays, device4arrays + 256, block_2.x);

		// modular reduction
		device_reduce_modulo_partial_2 << <block_2, thread_4_warp, 0, mainStream >> > (device4arrays, device4arrays + 128, device_m);

		// reduction		
		addParallelWithOverflow(device4arrays, device4arrays + 128, block_1.x);
	}
	else
	{
		// modular reduction of part-results
		device_reduce_modulo_partial_4 << <block_4, thread_4_warp, 0, mainStream >> > (device4arrays, device_m);

		// reduction
		device_add_partial_2 << <block_2, thread_warp, 0, mainStream >> > (device4arrays, device4arrays + 256);

		// modular reduction
		device_reduce_modulo_partial_2 << <block_2, thread_4_warp, 0, mainStream >> > (device4arrays, device4arrays + 128, device_m);

		// reduction
		device_add_partial_1 << <block_1, thread_warp, 0, mainStream >> > (device4arrays, device4arrays + 128);
	}

	// final modular reduction
	device_reduce_modulo_partial_1 << < block_1, thread_4_warp, 0, mainStream >> > (device4arrays, device_m);
		
	// set x := result
	device_clone_partial_1 << <block_1, thread_4_warp, 0, mainStream >> > (device_x, device4arrays);

	checkCuda(cudaStreamSynchronize(mainStream));
}

