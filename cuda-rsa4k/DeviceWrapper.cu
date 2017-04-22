#include "DeviceWrapper.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void device_add(unsigned int* result, const unsigned int* x, const unsigned int* y)
{
	// implementation in DeviceWrapper.ptx	
}

__global__ void device_multiply(unsigned int* result, const unsigned int* x, const unsigned int* y)
{
	// implementation in DeviceWrapper.ptx	
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

BigInteger* DeviceWrapper::add(BigInteger& x, BigInteger& y)
{
	//todo: vaildate x,y
	//todo: overflow?

	unsigned int* resultArray = new unsigned int[BigInteger::ARRAY_SIZE];

	int size = sizeof(unsigned int*) * BigInteger::ARRAY_SIZE;
	
	unsigned int* device_result;
	unsigned int* device_x;
	unsigned int* device_y;

	checkCuda(cudaMalloc(&device_result, size));
	checkCuda(cudaMalloc(&device_x, size));
	checkCuda(cudaMalloc(&device_y, size));
	
	checkCuda(cudaMemcpy(device_x, x.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(device_y, y.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
	
	device_add << <1, 1 >> > (device_result, device_x, device_y);
	
	checkCuda(cudaMemcpy(resultArray, device_result, size, cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(device_result));
	checkCuda(cudaFree(device_x));
	checkCuda(cudaFree(device_y));
	
	return new BigInteger(resultArray);;
}

BigInteger* DeviceWrapper::multiply(BigInteger& x, BigInteger& y)
{
	//todo: vaildate x,y
	//todo: overflow?

	unsigned int* resultArray = new unsigned int[BigInteger::ARRAY_SIZE];
	
	int size = sizeof(unsigned int*) * BigInteger::ARRAY_SIZE;

	unsigned int* device_result;
	unsigned int* device_x;
	unsigned int* device_y;

	checkCuda(cudaMalloc(&device_result, size));
	checkCuda(cudaMalloc(&device_x, size));
	checkCuda(cudaMalloc(&device_y, size));

	checkCuda(cudaMemcpy(device_x, x.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(device_y, y.getMagnitudeArray(), size, cudaMemcpyHostToDevice));
		
	device_multiply << <1, 1 >> > (device_result, device_x, device_y);

	checkCuda(cudaMemcpy(resultArray, device_result, size, cudaMemcpyDeviceToHost));

	checkCuda(cudaFree(device_result));
	checkCuda(cudaFree(device_x));
	checkCuda(cudaFree(device_y));

	return new BigInteger(resultArray);;
}



