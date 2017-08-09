#include "DeviceWrapper.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

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

extern "C" __global__ void device_multiply_partial(unsigned int* result, const unsigned int* x, const unsigned int* y)
{
	int mod = (threadIdx.x % 2);
	__shared__ unsigned int evenResult[290][DeviceWrapper::MULTIPLICATION_THREAD_COUNT];
	__shared__ unsigned int oddResult[290][DeviceWrapper::MULTIPLICATION_THREAD_COUNT];
		
	// threads: 4 * 4 = 16
	// 0, 32, 64, 96
	const int startIndexX = (threadIdx.x % DeviceWrapper::MULTIPLICATION_THREAD_COUNT) * DeviceWrapper::CELLS_PER_THREAD;	// 32
	// 0, 1, 2, 3
	const int yArray = DeviceWrapper::div(threadIdx.x, DeviceWrapper::MULTIPLICATION_THREAD_COUNT);
	// 0, 32, 64, 96
	const int startIndexY = yArray * DeviceWrapper::CELLS_PER_THREAD;	// 32	 	
	const int length = DeviceWrapper::CELLS_PER_THREAD;	// 32
		
	unsigned int(*resultArray)[DeviceWrapper::MULTIPLICATION_THREAD_COUNT];

	if (mod == 0) // even
	{
		resultArray = evenResult;
	}
	else // odd
	{
		resultArray = oddResult;
	}

	for (int i = startIndexX; i < startIndexX + length; i++)
	{
		for (int j = startIndexY; j < startIndexY + length; j++)
		{
			asm volatile (
				"mad.lo.cc.u32 %0, %3, %4, %0; \n\t"
				"madc.hi.cc.u32 %1, %3, %4, %1; \n\t"
				"addc.u32 %2, %2, 0; \n\t"
				: "+r"(resultArray[i + j][yArray]), "+r"(resultArray[i + j + 1][yArray]), "+r"(resultArray[i + j + 2][yArray])
				: "r"(x[i]), "r"(y[j]));
		}
	}
	

	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 0; i < 290; i++)
		{
			asm volatile (
				"addc.cc.u32 %0, %1, %2; \n\t"
				: "=r"(resultArray[i][0])
				: "r"(resultArray[i][0]), "r"(resultArray[i][1]));
		}

		for (int i = 0; i < 290; i++)
		{
			asm volatile (
				"addc.cc.u32 %0, %1, %2; \n\t"
				: "=r"(resultArray[i][0])
				: "r"(resultArray[i][0]), "r"(resultArray[i][2]));
		}

		for (int i = 0; i < 290; i++)
		{
			asm volatile (
				"addc.cc.u32 %0, %1, %2; \n\t"
				: "=r"(resultArray[i][0])
				: "r"(resultArray[i][0]), "r"(resultArray[i][3]));
		}
	}

	if (threadIdx.x == 1)
	{
		for (int i = 0; i < 290; i++)
		{
			asm volatile (
				"addc.cc.u32 %0, %1, %2; \n\t"
				: "=r"(resultArray[i][0])
				: "r"(resultArray[i][0]), "r"(resultArray[i][1]));
		}

		for (int i = 0; i < 290; i++)
		{
			asm volatile (
				"addc.cc.u32 %0, %1, %2; \n\t"
				: "=r"(resultArray[i][0])
				: "r"(resultArray[i][0]), "r"(resultArray[i][2]));
		}

		for (int i = 0; i < 290; i++)
		{
			asm volatile (
				"addc.cc.u32 %0, %1, %2; \n\t"
				: "=r"(resultArray[i][0])
				: "r"(resultArray[i][0]), "r"(resultArray[i][3]));
		}
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i = 0; i < 256; i++)
		{
			asm volatile (
				"addc.cc.u32 %0, %1, %2; \n\t"
				: "=r"(result[i])
				: "r"(evenResult[i][0]), "r"(oddResult[i][0]));
		}
	}
	
}

extern "C" __global__ void device_add_partial(unsigned int* result, const unsigned int* x, const unsigned int* y)
{
	register const int startIndex = threadIdx.x * DeviceWrapper::ADDITION_CELLS_PER_THREAD;
	__shared__ unsigned short carries[BigInteger::ARRAY_SIZE + 1];

	register int index;
#pragma unroll
	for (index = 0; index < DeviceWrapper::ADDITION_CELLS_PER_THREAD - 1; index++)
	{
		asm volatile (
			"addc.cc.u32 %0, %1, %2; \n\t"	// genarate and propagate carry
			: "=r"(result[startIndex + index])
			: "r"(x[startIndex + index]), "r"(y[startIndex + index]));		
	}

	// last iteration generates and stores carry in the array
	asm volatile (
		"addc.cc.u32 %0, %2, %3; \n\t"
		"addc.u16 %1, 0, 0; \n\t"
		: "=r"(result[startIndex + index]), "=h"(carries[startIndex + 1 + index])
		: "r"(x[startIndex + index]), "r"(y[startIndex + index]));

	__syncthreads();	

	register unsigned int carry;
#pragma unroll
	for (register int i = 0; i < DeviceWrapper::ADDITION_THREAD_COUNT; i++)
	{
		index = 0;
		carry = carries[startIndex + index];

		// first iteration propagates carry from array
		asm volatile (
			"add.cc.u32 %0, %0, %1; \n\t"	//  
			: "+r"(result[startIndex + index])
			: "r"(carry));

#pragma unroll
		for (index = 1; index < DeviceWrapper::ADDITION_CELLS_PER_THREAD - 1; index++)
		{
			asm volatile (
				"addc.cc.u32 %0, %0, 0; \n\t"	//propagate generated carries
				: "+r"(result[startIndex + index]));
		}

		// last iteration generates and stores carry in the array
		asm volatile (
			"addc.cc.u32 %0, %0, 0; \n\t"
			"addc.u16 %1, 0, 0; \n\t"
			: "+r"(result[startIndex + index]), "=h"(carries[startIndex + 1 + index]));

		__syncthreads();
	}	
}

// returns x div y
extern "C" __host__ __device__ inline int DeviceWrapper::div(const int x, const int y) 
{
	return x / y;
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

	device_multiply_partial<<<1,16>>>(device_result, device_x, device_y);

	checkCuda(cudaMemcpy(resultArray, device_result, size * 2, cudaMemcpyDeviceToHost));

	unsigned int overflow = 0UL;
	for (int i = resultArraySize - 1; i >= BigInteger::ARRAY_SIZE; i--)
	{
		overflow = overflow | resultArray[i];
	}

	if (overflow != 0UL)
	{
		std::cerr << "ERROR: BigInteger::multiply overflow!" << endl;
		//throw std::overflow_error("BigInteger::multiply overflow");
	}

	checkCuda(cudaFree(device_result));
	checkCuda(cudaFree(device_x));
	checkCuda(cudaFree(device_y));
	
	//for (int i = 255; i >= 0; i--)
	//{
	//	cout << dec << resultArray[i];

	//}
	//cout << endl;

	return new BigInteger(resultArray);;
}

