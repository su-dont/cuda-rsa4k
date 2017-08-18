#pragma once
#include "DeviceWrapper.h"

class BigInteger
{
	//fields
public:

	// 4096 bits
	static const int ARRAY_SIZE = 128;	

private:

	// the magnitude array in little endian order
	// most-significant int is mag[length-1]
	// least-significant int is mag[0]
	unsigned int* magnitude;

	// device wrapper instance diffrent for every integer
	// to provide parallel execution
	DeviceWrapper* deviceWrapper;

	// methods
public:
	
	BigInteger();
	~BigInteger();
	
	static BigInteger* fromHexString(const char* string);

	void add(const BigInteger* x);
	void multiply(const BigInteger* x);

	unsigned int* getMagnitudeArray(void) const;

	bool equals(const BigInteger& value) const;
	int compare(const BigInteger& value) const;
	char* toHexString(void) const;
	void print(const char* title) const;

private:

	/*
	Parses hex string to unsigned int type.
	Accepts both upper and lower case, no "0x" at the beginning.
	E.g.: 314Da43F 
	*/	
	static unsigned int parseUnsignedInt(const char* hexString);
	
};

