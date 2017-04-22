#pragma once

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
	unsigned int magnitude[ARRAY_SIZE];

	// actual magnitude array's lenght <= ARRAY_SIZE
	int length;

	// methods
public:
	
	BigInteger();
	BigInteger(const unsigned int* magnitude);
	~BigInteger();
	
	static BigInteger fromHexString(const char* string);

	void leftShift(int bits);
	void add(BigInteger x);
	void multiply(BigInteger x);

	unsigned int* getMagnitudeArray(void);
	int getLength(void);

	bool equals(BigInteger& value) const;
	char* toHexString(void);
	void print(const char* title);

private:

	/*
	Parses hex string to unsigned int type.
	Accepts both upper and lower case, no "0x" at the beginning.
	E.g.: 314Da43F 
	*/	
	static unsigned int parseUnsignedInt(const char* hexString);
	
};

