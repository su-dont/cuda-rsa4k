#include "BigInteger.h"
#include "BuildConfig.h"
#include <iostream>

using namespace std;

BigInteger::BigInteger()
{
	magnitude = new unsigned int[ARRAY_SIZE + 1];
	deviceWrapper = new DeviceWrapper();
	clear(); // set magnitude to 0	
}

BigInteger::BigInteger(const BigInteger & x)
{
	magnitude = new unsigned int[ARRAY_SIZE + 1];
	deviceWrapper = new DeviceWrapper();

	for (int i = 0; i <= ARRAY_SIZE; i++)
	{
		magnitude[i] = x.magnitude[i];
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

BigInteger* BigInteger::ONE(void)
{
	BigInteger* one = new BigInteger();
	one->magnitude[0] = 1;
	return one;
}

void BigInteger::add(const BigInteger& x)
{
	deviceWrapper->addParallel(*this, x);	
}

void BigInteger::subtract(const BigInteger& x)
{
	deviceWrapper->subtractParallel(*this, x);
}

void BigInteger::multiply(const BigInteger& x)
{
	deviceWrapper->multiplyParallel(*this, x);	
}

void BigInteger::shiftLeft(int n)
{
	if (n == 0)
		return;

	deviceWrapper->shiftLeftParallel(*this, n);
}

void BigInteger::shiftRight(int n)
{
	if (n == 0)
		return;	

	deviceWrapper->shiftRightParallel(*this, n);
}

void BigInteger::mod(const BigInteger& modulus)
{	
	BigInteger* mod = new BigInteger(modulus);

	int compareValue = compare(*mod);
	if (compareValue == 1)
	{
		 // Trying to reduce modulo a greater integer		
		return;
	}
	if (compareValue == 0)
	{
		// Reducing modulo same integer		
		clear();
		return;
	}

	int bitwiseDifference = getBitwiseLengthDiffrence(*mod);
	mod->shiftLeft(bitwiseDifference);
	
	while (bitwiseDifference >= 0) // TODO: side channel vulnerability
	{
		if (compare(*mod) == -1)	// this > x
		{		
			subtract(*mod);
		}
		else // this <= x
		{				
			mod->shiftRight(1);
			bitwiseDifference--;
		}
	}		

	delete mod;
}

void BigInteger::multiplyMod(const BigInteger& x, const BigInteger& modulus)
{	
	multiply(x);
	mod(modulus);
}

void BigInteger::powerMod(const BigInteger& exponent, const BigInteger& modulus)
{
	// Assert :: (modulus - 1) * (modulus - 1) does not overflow base

	BigInteger* base = new BigInteger(*this);
	
	// set this to 1
	clear();
	magnitude[0] = 1;
	
	BigInteger* exp = new BigInteger(exponent);
	base->mod(modulus);

	int bits = exp->getBitwiseLength();
	
	for (int i = 0; i < bits; i++)
	{		
		if (exp->getLSB() == 1)
		{			
			multiplyMod(*base, modulus);
		}		
		exp->shiftRight(1);		
		base->multiplyMod(*base, modulus);		
	}	
}

// constant time execution resistant to timing attacks
bool BigInteger::equals(const BigInteger& value) const
{
	bool equals = true;
	bool dummy = true;
	for (int i = 0; i < ARRAY_SIZE; i++)
	{		
		if (magnitude[i] != value.magnitude[i])
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
		if (magnitude[i] != value.magnitude[i])
		{
			if (equals)
				equals = false;
			else
				dummy = false;

			greater = magnitude[i] < value.magnitude[i];
		}
	}
	return equals ? 0 : greater ? 1 : -1;
}

// value must not be greater than this
int BigInteger::getBitwiseLengthDiffrence(const BigInteger& value) const
{
	if (DEBUG)
	{
		if (compare(value) != -1)
		{
			cerr << "ERROR: BigInteger::getBitwiseLengthDiffrence - provided value is greater than this!" << endl;
		}
	}

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

		if (value.magnitude[i] != 0UL && !valueSet)
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
	unsigned int valueValue = value.magnitude[valueInts];
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

int BigInteger::getBitwiseLength(void) const
{
	int ints = 0;	
	bool set = false;
	for (int i = 127; i >= 0; i--)
	{
		if (magnitude[i] != 0UL && !set)
		{
			ints = i;
			set = true;
		}
	}

	int bits = 0;	
	unsigned int value = magnitude[ints];
	
	for (int i = 1; i < 32; i++)
	{
		if (value >> i == 1)
			bits = i + 1;
	}
	
	return 32 * ints + bits;
}

int BigInteger::getLSB(void) const
{
	return magnitude[0] & 0x01;
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

void BigInteger::clear(void)
{
	deviceWrapper->clear(*this);	
}
