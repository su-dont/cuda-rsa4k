#include "BigInteger.h"
#include "BuildConfig.h"
#include <iostream>

using namespace std;

static const BigInteger* ONE = new BigInteger(1);

BigInteger::BigInteger()
{
	deviceWrapper = new DeviceWrapper();
	deviceMagnitude = deviceWrapper->init(ARRAY_SIZE);
}

BigInteger::BigInteger(const BigInteger & x)
{
	deviceWrapper = new DeviceWrapper();
	deviceMagnitude = deviceWrapper->init(ARRAY_SIZE, x.getDeviceMagnitude());
}

BigInteger::BigInteger(unsigned int value)
{
	deviceWrapper = new DeviceWrapper();
	deviceMagnitude = deviceWrapper->init(ARRAY_SIZE);
	unsigned int* magnitude = new unsigned int[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
		magnitude[i] ^= magnitude[i];
	magnitude[0] = value;
	setMagnitude(magnitude);
	delete[] magnitude;
}

BigInteger::~BigInteger()
{
	deviceWrapper->free(deviceMagnitude);
	delete deviceWrapper;	
}

BigInteger* BigInteger::fromHexString(const char* string)
{
	// todo: fix +1
	unsigned int* magnitude = new unsigned int[ARRAY_SIZE + 1];
	for (int i = 0; i < 128; i++)
		magnitude[i] ^= magnitude[i];
	BigInteger* integer = new BigInteger();	
	int length = strlen(string);
	char temp[9];
	int i = length - 8;
	int j = 0;
	for (; i >= 0; i -= 8)
	{
		strncpy(temp, string + i, 8);
		temp[8] = '\0';
		magnitude[j++] = parseUnsignedInt(temp);
	}
	if (i < 0)
	{
		int index = 8 + i;
		char* temp = (char*) malloc(index + 1);
		strncpy(temp, string, 8 + i);
		temp[index] = '\0';
		unsigned int value = parseUnsignedInt(temp);
		magnitude[j] = value;		
		if (value > 0UL) 
			j++;
	}

	integer->setMagnitude(magnitude);

	delete[] magnitude;

	return integer;
}

void BigInteger::set(const BigInteger& x)
{
	deviceWrapper->cloneParallel(deviceMagnitude, x.getDeviceMagnitude());
}

int* BigInteger::getDeviceMagnitude(void) const
{
	return deviceMagnitude;
}

void BigInteger::add(const BigInteger& x)
{
	deviceWrapper->addParallel(deviceMagnitude, x.getDeviceMagnitude());	
}

void BigInteger::subtract(const BigInteger& x)
{
	deviceWrapper->subtractParallel(deviceMagnitude, x.getDeviceMagnitude());
}

void BigInteger::multiply(const BigInteger& x)
{
	deviceWrapper->multiplyParallel(deviceMagnitude, x.getDeviceMagnitude());	
}

void BigInteger::shiftLeft(int n)
{
	deviceWrapper->shiftLeftParallel(deviceMagnitude, n);
}

void BigInteger::shiftRight(int n)
{
	deviceWrapper->shiftRightParallel(deviceMagnitude, n);
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
	set(*ONE);
	
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

bool BigInteger::equals(const BigInteger& value) const
{	
	return deviceWrapper->equalsParallel(deviceMagnitude, value.getDeviceMagnitude());
}

// returns:
// 0 if value is the same with this
// 1 if value is greater than this
// -1 if value is lower than this
int BigInteger::compare(const BigInteger& value) const
{
	return deviceWrapper->compareParallel(deviceMagnitude, value.getDeviceMagnitude());	
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

	int thisLength = getBitwiseLength();
	int valueLength = value.getBitwiseLength();

	if (thisLength < valueLength)
	{
		cerr << "ERROR: BigInteger::getBitwiseLengthDiffrence - provided value is greater than this!" << endl;
	}

	return thisLength - valueLength;

	/*int thisInts;
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

	return 32 * (thisInts - valueInts) + thisBits - valueBits;*/
}

int BigInteger::getBitwiseLength(void) const
{
	// TODO: parallel

	unsigned int* magnitude = new unsigned int[ARRAY_SIZE];
	deviceWrapper->updateHost(magnitude, deviceMagnitude, ARRAY_SIZE);

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
	return deviceWrapper->getLSB(deviceMagnitude);
}

char* BigInteger::toHexString(void) const
{
	unsigned int* magnitude = new unsigned int[ARRAY_SIZE];
	deviceWrapper->updateHost(magnitude, deviceMagnitude, ARRAY_SIZE);

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

void BigInteger::setMagnitude(const unsigned int* magnitude)
{
	deviceWrapper->updateDevice(deviceMagnitude, magnitude, ARRAY_SIZE);
}

void BigInteger::clear(void)
{
	deviceWrapper->clearParallel(deviceMagnitude);	
}
