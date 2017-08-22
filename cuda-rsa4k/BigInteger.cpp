#include "BigInteger.h"
#include "BuildConfig.h"
#include <iostream>
#include <cmath> 

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
	unsigned int* magnitude = new unsigned int[ARRAY_SIZE];
	for (int i = 0; i < 128; i++)
		magnitude[i] ^= magnitude[i];
	BigInteger* integer = new BigInteger();	
	int length = strlen(string);
	if (length == 0)
	{
		cerr << "ERROR: Nothing to parse: string length == 0" << endl;
		return integer;
	}
	if (length > 1024)
	{
		cerr << "ERROR: Overflow: string length > 1024" << endl;
		return integer;
	}
	char temp[9];
	int i = length - 8;
	int j = 0;
	for (; i >= 0; i -= 8)
	{
		strncpy(temp, string + i, 8);
		temp[8] = '\0';		
		magnitude[j] = parseUnsignedInt(temp);
		j++;		
	}	
	if (i < 0 && j < 127)
	{
		int index = 8 + i;
		char* temp = (char*) malloc(index + 1);
		strncpy(temp, string, 8 + i);
		temp[index] = '\0';
		unsigned int value = parseUnsignedInt(temp);		
		magnitude[j] = value;				
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
	if (compareValue == -1)
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
		if (compare(*mod) == 1)	// this > x
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
// 0 if this == value
// 1 if this > value
// -1 if this < value
int BigInteger::compare(const BigInteger& value) const
{
	return deviceWrapper->compareParallel(deviceMagnitude, value.getDeviceMagnitude());	
}

int BigInteger::getBitwiseLengthDiffrence(const BigInteger& value) const
{
	int thisLength = getBitwiseLength();
	int valueLength = value.getBitwiseLength();

	return abs(thisLength - valueLength);	
}

int BigInteger::getBitwiseLength(void) const
{
	return deviceWrapper->getBitLength(deviceMagnitude);
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
