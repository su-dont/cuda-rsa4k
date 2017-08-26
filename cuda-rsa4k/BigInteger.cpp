#include "BigInteger.h"
#include "BuildConfig.h"
#include <iostream>
#include <cmath> 
#include <time.h>

using namespace std;

BigInteger::BigInteger()
{
	deviceWrapper = new DeviceWrapper();
	deviceMagnitude = deviceWrapper->init(ARRAY_SIZE);
	hostMagnitude = new unsigned int[ARRAY_SIZE];
	upToDate = false;
}

BigInteger::BigInteger(const BigInteger & x)
{
	deviceWrapper = new DeviceWrapper();
	deviceMagnitude = deviceWrapper->init(ARRAY_SIZE, x.getDeviceMagnitude());
	hostMagnitude = new unsigned int[ARRAY_SIZE];
	upToDate = false;
}

BigInteger::BigInteger(unsigned int value)
{
	deviceWrapper = new DeviceWrapper();
	deviceMagnitude = deviceWrapper->init(ARRAY_SIZE);
	hostMagnitude = new unsigned int[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++)
		hostMagnitude[i] ^= hostMagnitude[i];
	hostMagnitude[0] = value;
	updateDeviceMagnitiude();
}

BigInteger::~BigInteger()
{
	deviceWrapper->free(deviceMagnitude);
	delete[] hostMagnitude;
	delete deviceWrapper;	
}

const unsigned int& BigInteger::operator[](int index)
{
	if (!upToDate)
	{
		updateHostMagnitiude();
	}
	return hostMagnitude[index];
}

BigInteger* BigInteger::fromHexString(const char* string)
{
	BigInteger* integer = new BigInteger();
	for (int i = 0; i < 128; i++)
		integer->hostMagnitude[i] ^= integer->hostMagnitude[i];
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
		integer->hostMagnitude[j] = parseUnsignedInt(temp);
		j++;		
	}	
	if (i < 0 && j < 127)
	{
		int index = 8 + i;
		char* temp = (char*) malloc(index + 1);
		strncpy(temp, string, 8 + i);
		temp[index] = '\0';
		unsigned int value = parseUnsignedInt(temp);		
		integer->hostMagnitude[j] = value;
	}

	integer->setMagnitude(integer->hostMagnitude);
	integer->upToDate = true;

	return integer;
}

BigInteger* BigInteger::createRandom(int bitLength)
{
	if (bitLength > 4096)
	{
		cout << "ERROR: BigInteger::createRandom Too many bits!" << endl;
		return nullptr;
	}
	if (bitLength < 0)
	{
		cout << "ERROR: BigInteger::createRandom No bits!: bitLength:" << bitLength << endl;
		return nullptr;
	}
	if (bitLength == 0)
	{
		return new BigInteger();
	}

	srand(time(NULL));
	int ints = bitLength >> 5;
	int bits = bitLength & 0x1f;

	BigInteger* integer = new BigInteger();

	int i = 0;
	for (; i < ints; i++)
	{
		integer->hostMagnitude[i] = random32();
	}
	for (; i < 128; i++)
	{
		integer->hostMagnitude[i] ^= integer->hostMagnitude[i];
	}

	if (bits > 0)
	{
		int msb = 1 << (bits - 1);
		int mask = 0xffffffff << bits;
		mask = ~mask;
		int partial = random32();
		partial = partial & mask;
		partial = partial | msb;
		integer->hostMagnitude[ints] = partial;
	}
	else
	{
		int msb = 1 << 31;
		integer->hostMagnitude[ints - 1] |= msb;
	}
	
	integer->setMagnitude(integer->hostMagnitude);
	integer->upToDate = true;

	return integer;
}

void BigInteger::set(const BigInteger& x)
{
	upToDate = false;
	deviceWrapper->cloneParallel(deviceMagnitude, x.getDeviceMagnitude());
}

unsigned int* BigInteger::getDeviceMagnitude(void) const
{
	return deviceMagnitude;
}

void BigInteger::add(const BigInteger& x)
{
	upToDate = false;
	deviceWrapper->addParallel(deviceMagnitude, x.getDeviceMagnitude());	
}

void BigInteger::subtract(const BigInteger& x)
{
	upToDate = false;
	deviceWrapper->subtractParallel(deviceMagnitude, x.getDeviceMagnitude());
}

void BigInteger::multiply(const BigInteger& x)
{
	upToDate = false;
	deviceWrapper->multiplyParallel(deviceMagnitude, x.getDeviceMagnitude());	
}

void BigInteger::square(void)
{
	upToDate = false;
	deviceWrapper->squareParallel(deviceMagnitude);
}

void BigInteger::shiftLeft(int n)
{
	upToDate = false;
	deviceWrapper->shiftLeftParallel(deviceMagnitude, n);
}

void BigInteger::shiftRight(int n)
{
	upToDate = false;
	deviceWrapper->shiftRightParallel(deviceMagnitude, n);
}

void BigInteger::mod(const BigInteger& modulus)
{
	upToDate = false;
	deviceWrapper->modParallel(deviceMagnitude, modulus.getDeviceMagnitude());	
}

void BigInteger::multiplyMod(const BigInteger& x, const BigInteger& modulus)
{	
	upToDate = false;
	deviceWrapper->multiplyModParallel(deviceMagnitude, x.getDeviceMagnitude(), modulus.getDeviceMagnitude());
}

void BigInteger::squareMod(const BigInteger & modulus)
{
	upToDate = false;
	deviceWrapper->squareModParallel(deviceMagnitude, modulus.getDeviceMagnitude());
}

void BigInteger::modAsync(const BigInteger & modulus)
{
	upToDate = false;
	deviceWrapper->modParallelAsync(deviceMagnitude, modulus.getDeviceMagnitude());
}

void BigInteger::multiplyModAsync(const BigInteger& x, const BigInteger& modulus)
{
	upToDate = false;
	deviceWrapper->multiplyModParallelAsync(deviceMagnitude, x.getDeviceMagnitude(), modulus.getDeviceMagnitude());
}

void BigInteger::squareModAsync(const BigInteger & modulus)
{
	upToDate = false;
	deviceWrapper->squareModParallelAsync(deviceMagnitude, modulus.getDeviceMagnitude());
}

void BigInteger::powerMod(BigInteger& exponent, const BigInteger& modulus)
{
	upToDate = false;

	BigInteger x0(1);
	BigInteger x1(*this);	

	for (int bits = exponent.getBitwiseLength() - 1; bits >= 0; bits--)
	{
		if (exponent.testBit(bits))
		{
			x0.multiplyModAsync(x1, modulus);
			x1.multiplyModAsync(x1, modulus);
		}
		else
		{
			x1.multiplyModAsync(x0, modulus);
			x0.multiplyModAsync(x0, modulus);
		}			
		x0.synchronize();
		x1.synchronize();		
	}
	set(x0);
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

bool BigInteger::testBit(int bit)
{
	if (bit < 0 || bit > 4095)
	{
		cerr << "ERROR: BigInteger::getBit: trying to get bit: " << bit << endl;
		return -1;
	}
	return ((*this)[bit >> 5] & (1 << (bit & 0x1f))) != 0 ;
}

void BigInteger::synchronize(void)
{
	deviceWrapper->synchronize();
}

char* BigInteger::toHexString(void)
{
	updateHostMagnitiude();

	char* buffer = (char*) malloc(ARRAY_SIZE * 8 + 1);
	for (int i = ARRAY_SIZE - 1, j = 0; i >= 0; i--)
	{
		sprintf(buffer + 8 * j++, "%08x", hostMagnitude[i]);
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
	cout << "Mag: " << toHexString() << endl;
	cout << "MagLength: " << strlen(toHexString()) << endl;
}

void BigInteger::startTimer(void)
{
	deviceWrapper->startClock();
}

unsigned long long BigInteger::stopTimer(void)
{
	return deviceWrapper->stopClock();
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
	upToDate = false;
	deviceWrapper->updateDevice(deviceMagnitude, magnitude, ARRAY_SIZE);	
}

void BigInteger::clear(void)
{
	upToDate = false;
	deviceWrapper->clearParallel(deviceMagnitude);	
}

void BigInteger::updateDeviceMagnitiude(void)
{
	deviceWrapper->updateDevice(deviceMagnitude, hostMagnitude, ARRAY_SIZE);
	upToDate = true;
}

void BigInteger::updateHostMagnitiude(void)
{
	deviceWrapper->updateHost(hostMagnitude, deviceMagnitude, ARRAY_SIZE);
	upToDate = true;
}

unsigned int BigInteger::random32(void)
{
	unsigned random = (rand() & 0x7FFF);
	random <<= 15;
	random |= (rand() & 0x7FFF);
	return random;
}
