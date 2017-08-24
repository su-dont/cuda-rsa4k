#pragma once
#include "DeviceWrapper.h"

// mutable!
class BigInteger
{
	//fields
public:

	// 4096 bits
	static const int ARRAY_SIZE = 128;	

private:
	// Magnitude array in little endian order.
	// Most-significant int is mag[length-1].
	// Least-significant int is mag[0].
	// Allocated on the device.
	unsigned int* deviceMagnitude;

	// same array allocated on the host 
	// provides faster access if nothing was changed
	unsigned int* hostMagnitude;

	// flag indicating if hostMagnitude matches deviceMagnitude
	bool upToDate;

	// Device wrapper instance diffrent for every integer
	// to provide parallel execution
	DeviceWrapper* deviceWrapper;

	// methods
public:
	
	BigInteger();
	BigInteger(const BigInteger& x);
	BigInteger(unsigned int value);
	~BigInteger();

	const unsigned int& operator[](int index);

	// factory
	static BigInteger* fromHexString(const char* string);
	static BigInteger* createRandom(int bitLength);

	// setters, getters
	void set(const BigInteger& x);	
	unsigned int* getDeviceMagnitude(void) const;

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
	bool testBit(int bit);
	char* toHexString(void);
	void print(const char* title);

	//timer
	void startTimer(void);
	unsigned long long stopTimer(void);
	
private:

	void setMagnitude(const unsigned int* magnitude);
	void clear(void);
	void updateDeviceMagnitiude(void);
	void updateHostMagnitiude(void);
	static unsigned int random32(void);

	/*
	Parses hex string to unsigned int type.
	Accepts both upper and lower case, no "0x" at the beginning.
	E.g.: 314Da43F 
	*/	
	static unsigned int parseUnsignedInt(const char* hexString);
	
};

