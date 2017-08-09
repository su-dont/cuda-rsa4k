#pragma once
#include "BigInteger.h"

class DeviceWrapper
{
public:

	static const int MULTIPLICATION_THREAD_COUNT = 4;
	static const int CELLS_PER_THREAD = BigInteger::ARRAY_SIZE / MULTIPLICATION_THREAD_COUNT; // 32

	static const int ADDITION_THREAD_COUNT = 16;
	static const int ADDITION_CELLS_PER_THREAD = BigInteger::ARRAY_SIZE / ADDITION_THREAD_COUNT;

	DeviceWrapper();
	~DeviceWrapper();

	static unsigned long long getClock(void);
	static inline int div(const int x, const int y);

	static BigInteger* add(BigInteger& x, BigInteger& y);
	static BigInteger* addParallel(BigInteger& x, BigInteger& y);	
	static BigInteger* multiply(BigInteger& x, BigInteger& y);
	static BigInteger* multiplyParallel(BigInteger& x, BigInteger& y);

};

