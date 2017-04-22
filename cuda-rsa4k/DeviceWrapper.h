#pragma once
#include "BigInteger.h"

class DeviceWrapper
{
public:
	DeviceWrapper();
	~DeviceWrapper();

	static BigInteger* add(BigInteger& x, BigInteger& y);
	static BigInteger* multiply(BigInteger& x, BigInteger& y);

};

