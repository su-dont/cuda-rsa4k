#pragma once
#include "DeviceWrapper.h"

class BigInteger
{
	//fields
public:

	// 4096 bits
	static const int ARRAY_SIZE = 128;	


	// the magnitude array in little endian order
	// most-significant int is mag[length-1]
	// least-significant int is mag[0]
	unsigned int* magnitude;
	
private:
	// device wrapper instance diffrent for every integer
	// to provide parallel execution
	DeviceWrapper* deviceWrapper;

	// methods
public:
	
	BigInteger();
	BigInteger(const BigInteger& x);
	~BigInteger();
	
	// factory
	static BigInteger* fromHexString(const char* string);
	static BigInteger* ONE(void);

	// arithmetics
	void add(const BigInteger& x);
	void subtract(const BigInteger& x);
	void multiply(const BigInteger& x);
	void mod(const BigInteger& modulus);
	void multiplyMod(const BigInteger& x, const BigInteger& modulus);
	void powerMod(const BigInteger& exponent, const BigInteger& modulus);

	// logics
	void shiftLeft(int bits);
	void shiftRight(int bits);

	// extras
	bool equals(const BigInteger& value) const;
	int compare(const BigInteger& value) const;
	int getBitwiseLengthDiffrence(const BigInteger& value) const;
	int getBitwiseLength(void) const;
	int getLSB(void) const;
	char* toHexString(void) const;
	void print(const char* title) const;
	void clear(void);

private:

	/*
	Parses hex string to unsigned int type.
	Accepts both upper and lower case, no "0x" at the beginning.
	E.g.: 314Da43F 
	*/	
	static unsigned int parseUnsignedInt(const char* hexString);
	
};

