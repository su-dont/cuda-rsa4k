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

	static const int TWO_WARPS = 64;
	static const int MULTIPLICATION_BLOCKS_COUNT = 4;
		
	static const int ONE_WARP = 32;
	// both addition and subtraction
	static const int ADDITION_CELLS_PER_THREAD = 4;	// BigInteger::ARRAY_SIZE / ONE_WARP

	DeviceWrapper();
	~DeviceWrapper();

	static unsigned long long getClock(void);

	unsigned int* addParallel(const BigInteger& x, const BigInteger& y) const;
	unsigned int* subtractParallel(const BigInteger& x, const BigInteger& y) const;
	unsigned int* multiplyParallel(const BigInteger& x, const BigInteger& y) const;	
};
