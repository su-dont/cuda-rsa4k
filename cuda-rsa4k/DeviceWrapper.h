#pragma once
#include <cuda_runtime.h>

class BigInteger;	// forward declaration

class DeviceWrapper
{

private:

	// main stream for kernel launches
	cudaStream_t mainStream;
	
	// addtional stream for async memory copy
	cudaStream_t memoryCopyStream;

public:

	// two warps
	static const int MULTIPLICATION_THREAD_COUNT = 64;
	static const int MULTIPLICATION_BLOCKS_COUNT = 4;

	// one warp
	static const int ADDITION_THREAD_COUNT = 32;
	static const int ADDITION_CELLS_PER_THREAD = 4;	// BigInteger::ARRAY_SIZE / ADDITION_THREAD_COUNT

	DeviceWrapper();
	~DeviceWrapper();

	static unsigned long long getClock(void);

	unsigned int* addParallel(const BigInteger& x, const BigInteger& y);
	unsigned int* multiplyParallel(const BigInteger& x, const BigInteger& y);
};
