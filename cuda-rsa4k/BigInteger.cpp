#include "BigInteger.h"
#include "DeviceWrapper.h"
#include <iostream>

using namespace std;


// constructor
BigInteger::BigInteger()
{
	length = 0;
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		magnitude[i] ^= magnitude[i];	// clear
	}
}

// magnitude must be ARRAY_SIZE allocated
BigInteger::BigInteger(const unsigned int* magnitude)
{
	int i = ARRAY_SIZE - 1;
	for (; i >= 0; i--)
	{
		this->magnitude[i] = magnitude[i];
		if (magnitude[i] != 0UL)
		{
			length = i + 1;
			break;
		}
	}
	for (; i >= 0; i--)
		this->magnitude[i] = magnitude[i];
}

// destuctor
BigInteger::~BigInteger()
{
}

// public
BigInteger BigInteger::fromHexString(const char* string)
{
	BigInteger integer;	
	int length = strlen(string);
	char temp[9];
	int i = length - 8;
	int j = 0;
	for (; i >= 0; i -= 8)
	{
		strncpy(temp, string + i, 8);
		temp[8] = '\0';
		integer.magnitude[j++] = parseUnsignedInt(temp);
	}
	if (i < 0)
	{
		int index = 8 + i;
		char* temp = (char*) malloc(index + 1);
		strncpy(temp, string, 8 + i);
		temp[index] = '\0';
		unsigned int value = parseUnsignedInt(temp);
		integer.magnitude[j] = value;		
		if (value > 0UL) 
			j++;
	}
	integer.length = j;
	return integer;
}

void BigInteger::leftShift(int n)
{
	if (n == 0) 
		return;

	int ints = n >> 5;
	int bits = n & 0x1f;
	int newLength = length + ints;

	if(newLength >= ARRAY_SIZE)
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
	length = newLength;
}

void BigInteger::add(BigInteger x)
{	
	BigInteger* result = DeviceWrapper::add(*this, x);
	
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		magnitude[i] = result->getMagnitudeArray()[i];
	}
	length = result->getLength();
}

void BigInteger::multiply(BigInteger x)
{
	BigInteger* result = DeviceWrapper::multiply(*this, x);

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		magnitude[i] = result->getMagnitudeArray()[i];
	}
	length = result->getLength();
}

unsigned int* BigInteger::getMagnitudeArray(void)
{
	return magnitude;
}

int BigInteger::getLength(void)
{
	return length;
}

bool BigInteger::equals(BigInteger& value) const
{
	if (length != value.getLength())
		return false;
	for (int i = 0; i < length; i++)
	{
		if (magnitude[i] != value.getMagnitudeArray()[i])
			return false;
	}
	return true;
}

char* BigInteger::toHexString(void)
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

void BigInteger::print(const char* title)
{
	cout << title << endl;
	cout << "Length: " << length << endl;
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


