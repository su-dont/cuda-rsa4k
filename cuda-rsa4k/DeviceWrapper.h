#pragma once
#include "BigInteger.h"

class DeviceWrapper
{
public:

	// two warps
	static const int MULTIPLICATION_THREAD_COUNT = 64;

	// one warp
	static const int ADDITION_THREAD_COUNT = 32;
	static const int ADDITION_CELLS_PER_THREAD = BigInteger::ARRAY_SIZE / ADDITION_THREAD_COUNT;

	DeviceWrapper();
	~DeviceWrapper();

	static unsigned long long getClock(void);

	static BigInteger* add(BigInteger& x, BigInteger& y);
	static BigInteger* addParallel(BigInteger& x, BigInteger& y);	
	static BigInteger* multiply(BigInteger& x, BigInteger& y);
	static BigInteger* multiplyParallel(BigInteger& x, BigInteger& y);
};

