#include "RSA.h"
#include <iostream>

RSA::RSA()
{
	exponent = new BigInteger(FERMATS_FOUR);
}

RSA::~RSA()
{
	delete exponent;
}

// using Fermat's four - 0x10001 for encryption
unsigned long long RSA::encrypt(BigInteger& value, const BigInteger& modulus) const
{		
	value.startTimer();
	value.powerMod(*exponent, modulus);
	return value.stopTimer();
}

