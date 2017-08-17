#pragma once

class BigInteger;	// forward declaration

class DeviceWrapper
{
public:

	// two warps
	static const int MULTIPLICATION_THREAD_COUNT = 64;

	// one warp
	static const int ADDITION_THREAD_COUNT = 32;
	static const int ADDITION_CELLS_PER_THREAD = 4;	// BigInteger::ARRAY_SIZE / ADDITION_THREAD_COUNT

	DeviceWrapper();
	~DeviceWrapper();

	static unsigned long long getClock(void);

	unsigned int* add(const BigInteger& x, const BigInteger& y);
	unsigned int* addParallel(const BigInteger& x, const BigInteger& y);

	unsigned int* multiply(const BigInteger& x, const BigInteger& y);
	unsigned int* multiplyParallel(const BigInteger& x, const BigInteger& y);
};
