#pragma once
#include "BigInteger.h"

class RSA
{
private:

	static const unsigned int FERMATS_FOUR = 0x10001;

	BigInteger* exponent;

public:	
	
	RSA();
	~RSA();

	// return time of the encryption
	unsigned long long encrypt(BigInteger& value, const BigInteger& modulus) const;

};

