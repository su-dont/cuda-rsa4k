#include "BigInteger.h"
#include "BuildConfig.h"
#include <iostream>

using namespace std;

BigInteger::BigInteger()
{
	magnitude = new unsigned int[ARRAY_SIZE + 1];
	dummyMagnitude = new unsigned int[ARRAY_SIZE + 1];
	deviceWrapper = new DeviceWrapper();
	
	for (int i = 0; i <= ARRAY_SIZE; i++)
	{
		magnitude[i] ^= magnitude[i];	// clear
	}
}

BigInteger::~BigInteger()
{
	delete[] magnitude;
	delete[] dummyMagnitude;
	delete deviceWrapper;
}

// public
BigInteger* BigInteger::fromHexString(const char* string)
{
	BigInteger* integer = new BigInteger();	
	int length = strlen(string);
	char temp[9];
	int i = length - 8;
	int j = 0;
	for (; i >= 0; i -= 8)
	{
		strncpy(temp, string + i, 8);
		temp[8] = '\0';
		integer->magnitude[j++] = parseUnsignedInt(temp);
	}
	if (i < 0)
	{
		int index = 8 + i;
		char* temp = (char*) malloc(index + 1);
		strncpy(temp, string, 8 + i);
		temp[index] = '\0';
		unsigned int value = parseUnsignedInt(temp);
		integer->magnitude[j] = value;		
		if (value > 0UL) 
			j++;
	}
	return integer;
}

void BigInteger::add(const BigInteger* x)
{
	unsigned int* result = deviceWrapper->addParallel(*this, *x);
	delete[] magnitude;
	magnitude = result;
}

void BigInteger::subtract(const BigInteger* x)
{
	unsigned int* result = deviceWrapper->subtractParallel(*this, *x);
	delete[] magnitude;
	magnitude = result;
}

void BigInteger::multiply(const BigInteger* x)
{
	unsigned int* result = deviceWrapper->multiplyParallel(*this, *x);
	delete[] magnitude;
	magnitude = result;
}

void BigInteger::shiftLeft(int n)
{
	unsigned int* result = deviceWrapper->shiftLeft(*this, n);
	delete[] magnitude;
	magnitude = result;
}

// constant time execution resistant to timing attacks
void BigInteger::shiftRight(int n)
{
	if (n == 0)
		return;

	int ints = n >> 5;
	int bits = n & 0x1f;

	int index;
	for (int i = 0; i < 128; i++)
	{
		index = i + ints;
		if (index < 128)
			magnitude[i] = magnitude[index];
		else
			magnitude[i] = 0UL;
	}

	unsigned int* array;
	if (bits != 0)
		array = magnitude;
	else
		array = dummyMagnitude;
	
	int remainingBits = 32 - bits;
	int highBits = 0;
	int lowBits;

	for (int i = 127; i >= 0; i--)
	{
		lowBits = array[i] << remainingBits;
		array[i] = array[i] >> bits | highBits;
		highBits = lowBits;
	}
}

void BigInteger::mod(BigInteger* x)
{	
	int compareValue = compare(*x);
	if (compareValue != -1)
	{
		if (DEBUG)
		{
			cout << "Trying to reduce modulo a greater or the same integer" << endl;
		}
		return;
	}

	unsigned int* result;

	int bitwiseDifference = getBitwiseLengthDiffrence(*x);
	x->shiftLeft(bitwiseDifference);

	while (bitwiseDifference >= 0) // TODO: side channel vulnerability
	{
		if (compare(*x) == -1)	// this > x
		{			
			result = deviceWrapper->subtractParallel(*this, *x);
			delete[] magnitude;
			magnitude = result;
		}
		else // this <= x
		{				
			x->shiftRight(1);
			bitwiseDifference--;
		}
	}	

	if (bitwiseDifference > 0)
		x->shiftRight(bitwiseDifference);	// restore x to previous value	
}

unsigned int* BigInteger::getMagnitudeArray(void) const
{
	return magnitude;
}

// constant time execution resistant to timing attacks
bool BigInteger::equals(const BigInteger& value) const
{
	bool equals = true;
	bool dummy = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
	{		
		if (magnitude[i] != value.getMagnitudeArray()[i])
		{			
			if (equals)
				equals = false;
			else
				dummy = false;
		}
	}
	return equals;
}

// constant time execution resistant to timing attacks
// returns:
// 0 if value is the same with this
// 1 if value is greater than this
// -1 if value is lower than this
int BigInteger::compare(const BigInteger& value) const
{
	bool equals = true;
	bool greater = true;
	bool dummy = true;

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		if (magnitude[i] != value.getMagnitudeArray()[i])
		{
			if (equals)
				equals = false;
			else
				dummy = false;

			greater = magnitude[i] < value.getMagnitudeArray()[i];
		}
	}
	return equals ? 0 : greater ? 1 : -1;
}

// value must not be greater than this
int BigInteger::getBitwiseLengthDiffrence(const BigInteger& value) const
{
	int thisInts;
	int valueInts;
	bool thisSet = false;
	bool valueSet = false;
	for (int i = 127; i >= 0; i--)
	{
		if (magnitude[i] != 0UL && !thisSet)
		{
			thisInts = i;		
			thisSet = true;
		}

		if (value.getMagnitudeArray()[i] != 0UL && !valueSet)
		{
			valueInts = i;
			valueSet = true;
		}
	}	
	if (valueInts > thisInts)
	{
		cerr << "ERROR: BigInteger::getBitwiseLengthDiffrence - provided value is greater than this!" << endl;
	}

	int thisBits = 0;
	int valueBits = 0;
	unsigned int thisValue = magnitude[thisInts];
	unsigned int valueValue = value.getMagnitudeArray()[valueInts];
	for (int i = 1; i < 32; i++)
	{
		if (thisValue >> i == 1)
			thisBits = i + 1;
		if (valueValue >> i == 1)
			valueBits = i + 1;
	}
	
	if (valueInts == thisInts && valueBits > thisBits)
	{
		cerr << "ERROR: BigInteger::getBitwiseLengthDiffrence - provided value is greater than this!" << endl;
	}

	return 32 * (thisInts - valueInts) + thisBits - valueBits;
}

char* BigInteger::toHexString(void) const
{
	char* buffer = (char*) malloc(ARRAY_SIZE * 8 + 1);
	for (int i = ARRAY_SIZE - 1, j = 0; i >= 0; i--)
	{
		sprintf(buffer + 8 * j++, "%08x", magnitude[i]);
	}

	// clear leading zeros
	int i = 0;
	for (; i < ARRAY_SIZE * 8 - 1; i++)
		if (buffer[i] != '0')
			break;

	return buffer + i;
}

void BigInteger::print(const char* title) const
{
	cout << title << endl;
	cout << "Mag: " << toHexString() << endl;
	cout << "MagLength: " << strlen(toHexString()) << endl;
}

// private
unsigned int BigInteger::parseUnsignedInt(const char* hexString)
{
	int length = strlen(hexString);
	char* padding = new char[9];
	if (length < 8)
	{
		int zeros = 8 - length;
		for (int i = 0, j = 0; i < 8; i++)
		{
			if (i < zeros)
				padding[i] = '0';
			else
				padding[i] = hexString[j++];
		}
		padding[8] = '\0';
	}
	else
	{
		strcpy(padding, hexString);
	}
	unsigned int chunk = 0UL;
	sscanf(padding, "%x", &chunk);
	return chunk;
}


