#include "BigInteger.h"
#include <iostream>

using namespace std;

BigInteger::BigInteger()
{
	magnitude = new unsigned int[ARRAY_SIZE + 1];
	deviceWrapper = new DeviceWrapper();
	
	// todo: necessary?
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		magnitude[i] ^= magnitude[i];	// clear
	}
}

BigInteger::~BigInteger()
{
	delete[] magnitude;
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

void BigInteger::leftShift(int n)
{
	// to be done

	/*if (n == 0) 
		return;

	int ints = n >> 5;
	int bits = n & 0x1f;
	int newLength = length + ints;

	if (newLength >= ARRAY_SIZE)
	{
		throw std::overflow_error("BigInteger::leftShift newLength >= ARRAY_SIZE");
		return;
	}

	for (int i = newLength - 1; i >= ints; i--)		
		magnitude[i] = magnitude[i-ints];
		
	for (int i = 0; i < ints; i++)
		magnitude[i] = 0UL;
		
	if (bits != 0)
	{		
		newLength++;
        int remainingBits = 32 - bits;
        int highBits;
		int lowBits = 0;

		for (int i = ints; i < newLength; i++)
		{
			highBits = magnitude[i] >> remainingBits;
			magnitude[i] = magnitude[i] << bits | lowBits;
			lowBits = highBits;
		}			
	}	
	length = newLength;*/
}

void BigInteger::add(const BigInteger* x)
{
	unsigned int* result = deviceWrapper->add(*this, *x);
	delete[] magnitude;
	magnitude = result;
}

void BigInteger::addParallel(const BigInteger* x)
{
	unsigned int* result = deviceWrapper->addParallel(*this, *x);
	delete[] magnitude;
	magnitude = result;
}

void BigInteger::multiply(const BigInteger* x)
{
	unsigned int* result = deviceWrapper->multiply(*this, *x);
	delete[] magnitude;
	magnitude = result;
}

void BigInteger::multiplyParallel(const BigInteger* x)
{
	unsigned int* result = deviceWrapper->multiplyParallel(*this, *x);
	delete[] magnitude;
	magnitude = result;
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


