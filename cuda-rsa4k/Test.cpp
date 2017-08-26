#include "Test.h"
#include "BigInteger.h"
#include "BuildConfig.h"
#include <iostream>

using namespace std;

Test::Test()
{
}

Test::~Test()
{
}

void Test::runAll(bool print, int minBits, int maxBits, int step, int repeats)
{
	testBigIntegerCorrectness(print);	
	testBigIntegerTimes(minBits, maxBits, step, repeats);

	cout << "\nFINISHED!" << endl;
}

void Test::testBigIntegerCorrectness(bool print)
{
	cout << "Testing BigInteger correctnes" << endl;
		
	testParsing(print);
	testCreateRandom(print);
	testTestBit(print);
	testEquals(print);
	testCompare(print);
	testBitwiseLengthDiffrence(print);

	cout << "logics..." << endl;
	testShiftLeft(print);
	testShiftRight(print);

	cout << "arithmetics..." << endl;
	testAdd(print);
	testSubtract(print);
	testMultiply(print);
	testSquare(print);
	testMod(print);	
	testMultiplyMod(print);
	testSquareMod(print);
	testPowerMod(print);

}

void Test::testBigIntegerTimes(int minBits, int maxBits, int step, int repeats)
{
	cout << "\n\nTesting BigInteger execution times..." << endl;

	testEqualsTimings(minBits, maxBits, step, repeats);
	testCompareTimings(minBits, maxBits, step, repeats);

	cout << "logics..." << endl;
	testShiftLeftTimings(minBits, maxBits, step, repeats);
	testShiftRightTimings(minBits, maxBits, step, repeats);

	cout << "arithmetics..." << endl;
	testAddTimings(minBits, maxBits, step, repeats);
	testSubtractTimings(minBits, maxBits, step, repeats);
	testMultiplyTimings(minBits, maxBits, step, repeats);
	testSquareTimings(minBits, maxBits, step, repeats);
	testModTimings(minBits, maxBits, step, repeats);
	testMultiplyModTimings(minBits, maxBits, step, repeats);
	testSquareModTimings(minBits, maxBits, step, repeats);
	testPowerModTimings(minBits, maxBits, step, repeats);
}

void Test::testEqualsTimings(int minBits, int maxBits, int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += testEqualsTime(bits);
		}
		cout << "Test equals: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

void Test::testCompareTimings(int minBits, int maxBits, int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += testCompareTime(bits);
		}
		cout << "Test compare: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

void Test::testShiftLeftTimings(int minBits, int maxBits, int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += testShiftLeftTime(bits, 5);
		}
		cout << "Test shift left: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

void Test::testShiftRightTimings(int minBits, int maxBits, int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += testShiftRightTime(bits, 3);
		}
		cout << "Test shift right: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

void Test::testAddTimings(int minBits, int maxBits, int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += testAddTime(bits);
		}
		cout << "Test add: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

void Test::testSubtractTimings(int minBits, int maxBits,  int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += testSubtractTime(bits);
		}
		cout << "Test subtract: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

void Test::testMultiplyTimings(int minBits, int maxBits, int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += testMultiplyTime(bits);
		}
		cout << "Test multiply: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

void Test::testSquareTimings(int minBits, int maxBits, int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += testSquareTime(bits);
		}
		cout << "Test square: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

void Test::testModTimings(int minBits, int maxBits, int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += testModTime(bits);
		}
		cout << "Test mod: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

void Test::testMultiplyModTimings(int minBits, int maxBits, int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += tesMultiplytModTime(bits);
		}
		cout << "Test multiply mod: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

void Test::testSquareModTimings(int minBits, int maxBits, int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += testSquareModTime(bits);
		}
		cout << "Test square mod: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

void Test::testPowerModTimings(int minBits, int maxBits, int step, int repeats)
{
	unsigned long long sum;
	for (int bits = minBits; bits <= maxBits; bits = bits + step)
	{
		sum = 0ULL;
		for (int i = 0; i < repeats; i++)
		{
			sum += testPowerModTime(bits);
		}
		cout << "Test power mod: bits: " << bits << " avg time: " << sum / (unsigned long long) repeats << endl;
	}
}

unsigned long long Test::testBitwiseLengthDiffrenceTime(int bits)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	BigInteger* bigInteger2 = BigInteger::createRandom(bits);
	if (bigInteger == nullptr || bigInteger2 == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->getBitwiseLengthDiffrence(*bigInteger2);
	unsigned long long time = bigInteger->stopTimer();	
	delete bigInteger;
	delete bigInteger2;
	return time;
}

unsigned long long Test::testEqualsTime(int bits)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	BigInteger* bigInteger2 = BigInteger::createRandom(bits);
	if (bigInteger == nullptr || bigInteger2 == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->equals(*bigInteger2);
	unsigned long long time = bigInteger->stopTimer();
	delete bigInteger;
	delete bigInteger2;
	return time;
}

unsigned long long Test::testCompareTime(int bits)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	BigInteger* bigInteger2 = BigInteger::createRandom(bits);
	if (bigInteger == nullptr || bigInteger2 == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->compare(*bigInteger2);
	unsigned long long time = bigInteger->stopTimer();
	delete bigInteger;
	delete bigInteger2;
	return time;
}

unsigned long long Test::testAddTime(int bits)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	BigInteger* bigInteger2 = BigInteger::createRandom(bits);
	if (bigInteger == nullptr || bigInteger2 == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->add(*bigInteger2);
	unsigned long long time = bigInteger->stopTimer();
	delete bigInteger;
	delete bigInteger2;
	return time;
}

unsigned long long Test::testSubtractTime(int bits)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	BigInteger* bigInteger2 = BigInteger::createRandom(bits);	
	if (bigInteger == nullptr || bigInteger2 == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	int compare = bigInteger->compare(*bigInteger2);
	unsigned long long time;
	if (compare == 1)
	{
		bigInteger->startTimer();
		bigInteger->subtract(*bigInteger2);
		time = bigInteger->stopTimer();
	}
	else
	{
		bigInteger2->startTimer();
		bigInteger2->subtract(*bigInteger);
		time = bigInteger2->stopTimer();
	}
	delete bigInteger;
	delete bigInteger2;
	return time;
}

unsigned long long Test::testMultiplyTime(int bits)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	BigInteger* bigInteger2 = BigInteger::createRandom(bits);
	if (bigInteger == nullptr || bigInteger2 == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->multiply(*bigInteger2);
	unsigned long long time = bigInteger->stopTimer();
	delete bigInteger;
	delete bigInteger2;
	return time;
}

unsigned long long Test::testSquareTime(int bits)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	if (bigInteger == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->square();
	unsigned long long time = bigInteger->stopTimer();
	delete bigInteger;
	return time;
}

unsigned long long Test::testShiftLeftTime(int bits, int n)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	if (bigInteger == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->shiftLeft(n);
	unsigned long long time = bigInteger->stopTimer();
	delete bigInteger;
	return time;
}

unsigned long long Test::testShiftRightTime(int bits, int n)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	if (bigInteger == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->shiftRight(n);
	unsigned long long time = bigInteger->stopTimer();
	delete bigInteger;
	return time;
}

unsigned long long Test::testModTime(int bits)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	BigInteger* bigInteger2 = BigInteger::createRandom(bits);
	if (bigInteger == nullptr || bigInteger2 == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->mod(*bigInteger2);
	unsigned long long time = bigInteger->stopTimer();
	delete bigInteger;
	delete bigInteger2;
	return time;
}

unsigned long long Test::tesMultiplytModTime(int bits)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	BigInteger* bigInteger2 = BigInteger::createRandom(bits);
	BigInteger* bigInteger3 = BigInteger::createRandom(bits);
	if (bigInteger == nullptr || bigInteger2 == nullptr || bigInteger3 == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->multiplyMod(*bigInteger2, *bigInteger3);
	unsigned long long time = bigInteger->stopTimer();
	delete bigInteger;
	delete bigInteger2;
	delete bigInteger3;
	return time;
}

unsigned long long Test::testSquareModTime(int bits)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	BigInteger* bigInteger2 = BigInteger::createRandom(bits);
	if (bigInteger == nullptr || bigInteger2 == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->squareMod(*bigInteger2);
	unsigned long long time = bigInteger->stopTimer();
	delete bigInteger;
	delete bigInteger2;
	return time;
}


unsigned long long Test::testPowerModTime(int bits)
{
	BigInteger* bigInteger = BigInteger::createRandom(bits);
	BigInteger* bigInteger2 = new BigInteger(0x10001);
	BigInteger* bigInteger3 = BigInteger::createRandom(bits);
	if (bigInteger == nullptr || bigInteger2 == nullptr || bigInteger3 == nullptr)
	{
		cout << "BitInteger is null" << endl;
		return 0;
	}
	bigInteger->startTimer();
	bigInteger->powerMod(*bigInteger2, *bigInteger3);
	unsigned long long time = bigInteger->stopTimer();
	delete bigInteger;
	delete bigInteger2;
	delete bigInteger3;
	return time;
}


void Test::testParsing(bool print)
{
	// test parsing
	const char* string = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

	BigInteger* bigInteger = BigInteger::fromHexString(string);
	char* string2 = bigInteger->toHexString();	
	bool ok = _stricmp(string, string2) == 0;
	if (print || !ok)
	{
		if (ok)
		{
			cout << "BigInteger::fromHexString and BigInteger::toHexString... SUCCESS" << endl;
		}
		else
		{
			cout << "BigInteger::fromHexString or BigInteger::toHexString... FAILED" << endl;
		}
	}	

	delete bigInteger;
}

void Test::testCreateRandom(bool print)
{
	int bits = 3194;
	bool ok = false;
	BigInteger* bigInteger = BigInteger::createRandom(bits);	
	if (bigInteger != nullptr)
	{
		ok = bits == bigInteger->getBitwiseLength();
	}
	if (print || !ok)
	{
		if (ok)
		{
			cout << "BigInteger::createRandom... SUCCESS" << endl;
		}
		else
		{
			cout << "BigInteger::createRandom... FAILED" << endl;
		}
	}

	delete bigInteger;	
}

void Test::testTestBit(bool print)
{
	BigInteger* bigInteger = BigInteger::fromHexString("76766766327663766766326766766326f9fa32766326f9fa6f9f766326f9faaf9fa3766326f9fa26f9fa766326f9fa"
		"26f9fa6f9766326f9fafa326f9766326f9fafa63766326f766326f9fa97663766326f9fa26f9fafa21c1766326f9fa7324766326f9fa766766326f9fa37634766326f9fa9326f"
		"9f96cec15c67");
	
	int bits = bigInteger->getBitwiseLength();

	char* buffer = (char*) malloc(bits + 1);
	int j = 0;
	for (int i = bits; i >= 0; i--)
	{
		sprintf(buffer + j++, "%d", bigInteger->testBit(i));
	}
	
	bool ok = _stricmp(buffer, "0111011001110110011001110110011000110010011101100110001101110110011001110110011000110010011001110110011001110110011"
		"000110010011011111001111110100011001001110110011000110010011011111001111110100110111110011111011101100110001100100110111110011111101010101"
		"111100111111010001101110110011000110010011011111001111110100010011011111001111110100111011001100011001001101111100111111010001001101111100"
		"111111010011011111001011101100110001100100110111110011111101011111010001100100110111110010111011001100011001001101111100111111010111110100"
		"110001101110110011000110010011011110111011001100011001001101111100111111010100101110110011000110111011001100011001001101111100111111010001"
		"001101111100111111010111110100010000111000001011101100110001100100110111110011111101001110011001001000111011001100011001001101111100111111"
		"010011101100110011101100110001100100110111110011111101000110111011000110100011101100110001100100110111110011111101010010011001001101111100"
		"111111001011011001110110000010101110001100111") == 0;

	if (print || !ok)
	{
		if (ok)
		{
			cout << "BigInteger::testBit... SUCCESS" << endl;
		}
		else
		{
			cout << "BigInteger::testBit...... FAILED" << endl;
		}
	}

	delete bigInteger;
}

unsigned long long Test::testBitwiseLengthDiffrence(bool print)
{
	// test bitwise length diffrence
	BigInteger* bigInteger = BigInteger::fromHexString("8000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

	BigInteger* bigInteger2 = BigInteger::fromHexString("1");
	
	bigInteger->startTimer();
	bool ok = bigInteger->getBitwiseLengthDiffrence(*bigInteger2) == 4095;
	unsigned long long time = bigInteger->stopTimer();
	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			bigInteger2->print("result");
		}
		if (ok)
		{
			cout << "BigInteger::getBitwiseLengthDiffrence... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::getBitwiseLengthDiffrence... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}

	delete bigInteger;
	delete bigInteger2;

	return time;
}

unsigned long long Test::testEquals(bool print)
{
	// test equals
	BigInteger* bigInteger = BigInteger::fromHexString("3e2bf9b23b2e1be1891e7fe032a06c2752abdfbd947a7edddbd9b980d26547f16ef4068e"
													"493873573d8ff96ae2a730ecc737a7bd82f27fa482b8346c98e2bacb8f93cf748eb56909"
													"0c17c3c4d8bc6edc4033d320a45dbfbf14ed8b6fa67c9d45f1368fd272b5b7395c0fed4b"
													"dd7a61f377c9073a8705b500d8a173cc00085468eb76ba67b5aafc516bd6d71ef2381a4c"
													"6882398dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee16"
													  "493873573d8ff96ae2a730ecc737a7bd82f27fa482b8346c98e2bacb8f93cf748eb56909"
													  "0c17c3c4d8bc6edc4033d320a45dbfbf14ed8b6fa67c9d45f1368fd272b5b7395c0fed4b"
													  "dd7a61f377c9073a8705b500d8a173cc00085468eb76ba67b5aafc516bd6d71ef2381a4c"
													  "6882398dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee16"
													  "82b3cdb0be24af1bb42de45906956ce0c4cca117a20a519df37e0a02ed1ed20a0f2ba8ca"
													  "1a2ad347cabcdaeefee446f07d6753964e8e37394442fc0a9a3e7dea75d64aad8298ace7");

	BigInteger* bigInteger2 = BigInteger::fromHexString("3e2bf9b23b2e1be1891e7fe032a06c2752abdfbd947a7edddbd9b980d26547f16ef4068e"
														"493873573d8ff96ae2a730ecc737a7bd82f27fa482b8346c98e2bacb8f93cf748eb56909"
														"0c17c3c4d8bc6edc4033d320a45dbfbf14ed8b6fa67c9d45f1368fd272b5b7395c0fed4b"
														"dd7a61f377c9073a8705b500d8a173cc00085468eb76ba67b5aafc516bd6d71ef2381a4c"
														"6882398dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee16"
														"493873573d8ff96ae2a730ecc737a7bd82f27fa482b8346c98e2bacb8f93cf748eb56909"
														"0c17c3c4d8bc6edc4033d320a45dbfbf14ed8b6fa67c9d45f1368fd272b5b7395c0fed4b"
														"dd7a61f377c9073a8705b500d8a173cc00085468eb76ba67b5aafc516bd6d71ef2381a4c"
														"6882398dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee16"
														"82b3cdb0be24af1bb42de45906956ce0c4cca117a20a519df37e0a02ed1ed20a0f2ba8ca"
														"1a2ad347cabcdaeefee446f07d6753964e8e37394442fc0a9a3e7dea75d64aad8298ace7");	

	bigInteger->startTimer();
	bool ok = bigInteger->equals(*bigInteger2);
	unsigned long long time = bigInteger->stopTimer();
	
	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			bigInteger2->print("result");
		}
		if (ok)
		{
			cout << "BigInteger::equals... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::equals... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}

	delete bigInteger;
	delete bigInteger2;

	return time;
}

unsigned long long Test::testCompare(bool print)
{
	// test equals
	BigInteger* bigInteger = BigInteger::fromHexString("8000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

	BigInteger* same = BigInteger::fromHexString("8000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

	BigInteger* greater = BigInteger::fromHexString("8000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001");

	BigInteger* lower = BigInteger::fromHexString("1");

	bigInteger->startTimer();
	bool ok = bigInteger->compare(*same) == 0;
	ok = ok && bigInteger->compare(*lower) == 1;
	ok = ok && bigInteger->compare(*greater) == -1;
	unsigned long long time = bigInteger->stopTimer();

	if (print || !ok)
	{

		if (ok)
		{
			cout << "BigInteger::compare... SUCCESS elapsed time:  " << time / 3 << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::compare... FAILED elapsed time:  " << time / 3 << " cycles" << endl;
		}
	}

	delete bigInteger;
	delete same;
	delete greater;
	delete lower;

	return time / 3;
}

unsigned long long Test::testAdd(bool print)
{
	// test add parallel
	BigInteger* bigInteger = BigInteger::fromHexString("7ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");

	BigInteger* added = BigInteger::fromHexString("1");

	BigInteger* result = BigInteger::fromHexString("8000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

	bigInteger->startTimer();
	bigInteger->add(*added);
	unsigned long long time = bigInteger->stopTimer();

	bool ok;
	if (ERROR_CHECKING)
	{
		ok= bigInteger->equals(*added);
	}
	else
	{
		ok = bigInteger->equals(*result);
	}

	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			result->print("result");
		}
		if (ok)
		{
			cout << "BigInteger::add... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::add... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}	


	delete bigInteger;
	delete added;
	delete result;

	return time;
}

unsigned long long Test::testSubtract(bool print)
{
	// test subtract parallel
	BigInteger* bigInteger = BigInteger::fromHexString("800000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000");

	BigInteger* subtracted = BigInteger::fromHexString("1");

	BigInteger* result = BigInteger::fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffff");

	bigInteger->startTimer();
	bigInteger->subtract(*subtracted);
	unsigned long long time = bigInteger->stopTimer();

	bool ok = bigInteger->equals(*result);

	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			result->print("result");
		}
		if (ok)
		{
			cout << "BigInteger::subtract... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::subtract... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}


	delete bigInteger;
	delete subtracted;
	delete result;

	return time;
}

unsigned long long Test::testMultiply(bool print)
{
	// test multiply parallel
	BigInteger* bigInteger = BigInteger::fromHexString("2c0e41230c57595c64c958967841e13167bc85f4ff6311edfff0bb1fa595f8f7efe6da5354ca2dd4fdf06a"
		"a1a6da1d3229fe430c945a53abbccaf807c8cdaea43d8790fcfb86647389fc4a96c27e02ee70b3b0a170b7d8ea13e072c71c444096c21308ef5b31635013af224e444"
		"9675f93e3ac4d13e3fca652fe4e7a5279b33f6a849c294c138e830a0cba3debe66c741d5436ae4cde8dbfd07b154a24a4cba3827bba8e144985a74b0bd73222cfeea4"
		"9069b1700542e2f0863138faccd5cb241fe9d5f883a13c9a6bb21aad4741004e7c298ada8dbc13cb8c6389c0220deaddd3e6689e1b3aca68700ad560e97c20bd85f60"
		"2da1355adae3e373139226ddbb3");
	BigInteger* multiplied = BigInteger::fromHexString("20150673f581db09a09d08f8ea0f0af0555a7cc42aa415001d1385cb634a8d6eb582add30b4a7ff82c9ddc"
		"ef1ec9a2c5955e593208d3fe8e06b615559b3a88812a350408a6bded1a80911b76cb3fd2d45e126d259fec8c9bef221e847e1f9f5da30143e34905b9866bcf6a46165"
		"a9eb27c3df5cff9f0e41bbfd3d9249c13f2d705c3dd95e142d2d11a4c9041cfc2051501f687f57106b5b76e2d4ed870bb21363d425363628e8d013c61f3542c8f469a"
		"80bc322452eec6a1ca98e8e18947af8cba631231e3f86149b8880f06dcd03d35fdd4e984348b25ac0824297a5932fb7bbcc5fc736dbe6926ab644bec0b1cfaf5c6cfb"
		"bd3520bf2106a0d31e8a69220db3");
	BigInteger* result = BigInteger::fromHexString("585666c03990095afd8c3c7d0c30fb0a6aa4a782663acb74e6b66fa5ea831018c8bab2924e7e6881f69d8effc5"
		"9b237a480ea56b26748526311fc692c7b011b91ce5efc3e87d2d0af3a174f4299ec5a4358dbd7e6db0494c60c0580601c2e25f40341c026bd1f4444e9e82eb793ab7b"
		"1ddf6186f8b63d3a5d5f782033a2f6ea6c8f61954bcf14a75294c7e23ccfd92e3d655501fca1ca5dc2df47739fdf6306e52f1feceea93d130b512232a1ef019054536"
		"69591dc40a40022e665937780327d785ca628a6fac350c909f940b9a3f767bd6c814da77eb9c09ee7cba650e3e9ee68ba5aef0a535cd338789882928f1f183e3f5f10"
		"c027931187b0c1098e99bccc5318058355757f833758e611e67a964eb57c7675702dee7f230cddf1b8cc3793c5ed022b210510e6e5f58ebc918852ce4aff954af2add"
		"453667cb22cee1e3778ee5e78eaa4033c57c99d3452f1d369a968e84bfa818a97678b4f728a7fd8b894dceb344bb4e3f78fa4f41cadecf062e3d4d21b3655e0b9dc3d"
		"af1b72a224cfdb49b7d85a6d2e32bf50e77dcee10a3b5b47e2b0c3f816a69135d7f51931b84a9c5aef7da5945759f7f5d9b26dd88c7bacd043b44372e949be026b6df"
		"9a7981022488acada7d08527a8d3c069524c86799ba7ebba98ffe0fd9b03807243ae25d151641a42c4c2c5107c856f1f113c6e90191c2bd0f2f6ae66aa4d5be43fbeb529");


	bigInteger->startTimer();
	bigInteger->multiply(*multiplied);
	unsigned long long time = bigInteger->stopTimer();

	bool ok = bigInteger->equals(*result);
	
	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			result->print("result");
		}
		
		if (ok)
		{
			cout << "BigInteger::multiply... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::multiply... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}	

	delete bigInteger;
	delete multiplied;
	delete result;

	return time;
}

unsigned long long Test::testSquare(bool print)
{
	// test square parallel
	BigInteger* bigInteger = BigInteger::fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffff");
	BigInteger* result = BigInteger::fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffe000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
		"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001");


	bigInteger->startTimer();
	bigInteger->square();
	unsigned long long time = bigInteger->stopTimer();

	bool ok = bigInteger->equals(*result);

	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			result->print("result");
		}

		if (ok)
		{
			cout << "BigInteger::square... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::square... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}

	delete bigInteger;
	delete result;

	return time;
}


unsigned long long Test::testShiftLeft(bool print)
{
	// test left shift
	BigInteger* bigInteger = BigInteger::fromHexString("7ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");

	BigInteger* result = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe");
			
	bigInteger->startTimer();
	bigInteger->shiftLeft(1);
	unsigned long long time = bigInteger->stopTimer();

	bool ok = bigInteger->equals(*result);

	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			result->print("result");
		}
		if (ok)
		{
			cout << "BigInteger::shiftLeft... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::shiftLeft... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}
	
	delete bigInteger;
	delete result;

	return time;
}


unsigned long long Test::testShiftRight(bool print)
{
	// test right shift
	BigInteger* bigInteger = BigInteger::fromHexString("1243abc312def391acd89897ad987f789868091243b45ac0"
														"7773FFAA83d19a3b3549937cfcF8a3dB9931254639186109"
														"ba3e70688a2f49CC67ff1dcfFF254639DF186BC109ADDE8c"
														"43ce74e249f9924093cebb70034a32a33ed731574ab50c3b");

	BigInteger* result = BigInteger::fromHexString("921d5e1896f79c8d66c4c4bd6cc3fbc4c34048921da2d603bb9ffd541");
	

	bigInteger->startTimer();
	bigInteger->shiftRight(537);
	unsigned long long time = bigInteger->stopTimer();

	bool ok = bigInteger->equals(*result);

	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			result->print("result");
		}
		if (ok)
		{
			cout << "BigInteger::shiftRight... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::shiftRight... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}

	delete bigInteger;
	delete result;

	return time;
}

unsigned long long Test::testMod(bool print)
{
	// test reduction modulo
	BigInteger* bigInteger = BigInteger::fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
	BigInteger* mod = BigInteger::fromHexString("1c1732493873573d8ff96ae2507a334766a9326f9f9f147b732624a8c7230c2137a507a334766326f9fa9326f9f9f14"
												"7326f9fb32624326f9fa8c7230c2137730ecc737bd2f27fa482b8346c907a334766a9326f9f9f147b732624a8c7230c"
												"21378e2bacb8cf748eb569091c17c3c4d8bc6edc033d320a45dbfbf14ed8b6fa67c91368fd272b5b7395c81a4c68823"
												"981a4c6882398dca52c7372f8dcc7372f0fed4bdda61f377c9073a8705b500d8173cc00085468eb76ba67b5aafc516b"
												"d6d71ef2381a4c688239dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee1682b3cdb0be"
												"24afb42de45906956ce0c4cca117a20a51370a02ed1d20a0f2ba8ca6f9f6ec4033d3326f9326f9fffbf14ed8507a334"												
												"256ce0906956ce0c4cca117a206956ce0c4326fcca1326f9f17a24e72ba344ee16");
	BigInteger* result = BigInteger::fromHexString("113268dc7c302b9cd85c2ad2ce15c4dfceaf21c765301af6c06369e4ae7063bc188310089b7d012c94eb91bec251"
		"74b5f4508dd7e6300d336db817d788728a916f76598b84749d2ffd9f0f12490d94fcc099cd8f6b2579b1dc650d7ade24f2c97febff048fbe4c3515c5517bd2bebbda7c6"
		"15ee3066e6a34d45b522e397cb4aa9e26d449cf5587cac7795085ff49a9a60ada0fb9d2d4431d38117f9bfdc3eb7b4433c54d8686749bcb3b10deab2cea84da322c9d40"
		"ffa03c0609604e66bde510e2b33161f0bc355bcb51cb68261ffc5f24e645e1a73b3af4d40354e61d5955018ec3c51abc0449131def3344aaacbcd427bcc88ace3e587a8"
		"71d3c0dc0de14b7bf61de744d053018883fe2f8c450bb4a6f95094a574a4df0036f18dd9fe7268c6d255f9f680cc8a3e8b666a5bd8f966177983efd771b32e577af4757b7fb");

	bigInteger->startTimer();
	bigInteger->mod(*mod);
	unsigned long long time = bigInteger->stopTimer();

	bool ok = bigInteger->equals(*result);

	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			result->print("result");
		}
		if (ok)
		{
			cout << "BigInteger::mod... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::mod... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}

	delete bigInteger;
	delete mod;
	delete result;

	return time;
}

unsigned long long Test::testMultiplyMod(bool print)
{
	// test modular multiplication 
	BigInteger* bigInteger = BigInteger::fromHexString("7a334766a93261c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9fa9"
		"326f9f9f146f9fa9326f9f9f14f9f9f1471c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9fa9326f9f9f146f9fa9326f9f9f14"
		"b731c173249387351c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9fa9326f9f9f146f9fa9326f9f9f1473d8ff96ae257a3347"
		"66321c1732493873573d8ff96ae257a334766326f9fa9326f9f9f146f9fa9326f9f9f142624a11c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae2"
		"57a334766326f9fa9326f9f9f146f9fa9326f9f9f14c1732493873573d8ff96ae257a334766326f9fa9326f9f9f148c727a334766a9326f9f9f147b732624a8c7230c213"
		"7a507a334766326f9fa9326f9f9f130c2137a507a334766326f9fa9326f9f9f1");
	BigInteger* times = BigInteger::fromHexString("7");
	BigInteger* mod = BigInteger::fromHexString("1c1732493873573d8ff96a1c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9f"
		"a9326f9f9f146f9fa9326f9f9f14e257a33476631c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9fa9326f9f9f146f9fa9326f"
		"9f9f1421c173249381c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9fa9326f9f9f146f9fa9326f9f9f1473573d8ff96ae257a"
		"334766326f9fa9326f9f9f146f9fa9326f9f9f14");

	BigInteger* result = BigInteger::fromHexString("46c127f892523f2c2c37eb44f9145bc5e06be1511016449f2d08647d719c3cc584133acd073346ea9be365445e291106954713c9c9fa4bbfe9bd430c9e3a240dc7dcbc1092db49e2eca8cc416babeae66e73ca11eeeb70b0e9685a005ff9bd1c6d08183a7f497c77e39d9b47ccf79c5f33731b4e04fdc2c42f3ac5bcb12dc618a268bf6e2f30a3fc08dba0aa1cc063bc5177718fa003ab7ee5ca86e54e4733b70706731ea5a640b817d6f86fd6311054f5f1d647c42359862da9d1ca8b057419c207840c1a0b373293295fb");

	bigInteger->startTimer();
	bigInteger->multiplyMod(*times, *mod);
	unsigned long long time = bigInteger->stopTimer();

	bool ok = bigInteger->equals(*result);

	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			result->print("result");
		}
		if (ok)
		{
			cout << "BigInteger::multiplyMod... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::multiplyMod... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}

	delete bigInteger;
	delete times;
	delete mod;
	delete result;

	return time;
}

unsigned long long Test::testSquareMod(bool print)
{
	// test modular square 
	BigInteger* bigInteger = BigInteger::fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffffffffffffffff");	
	BigInteger* mod = BigInteger::fromHexString("9fa32766326f9fa6f9f766326f9faaf9fa3766326f9fa26f9fa766326f9fa26f");

	BigInteger* result = BigInteger::fromHexString("350d329b02f5d57d05e747d539d6ed434e38b8be1f8eb142c13840482843fa1");

	bigInteger->startTimer();
	bigInteger->squareMod(*mod);
	unsigned long long time = bigInteger->stopTimer();

	bool ok = bigInteger->equals(*result);

	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			result->print("result");
		}
		if (ok)
		{
			cout << "BigInteger::squareMod... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::squareMod... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}

	delete bigInteger;
	delete mod;
	delete result;

	return time;
}


unsigned long long Test::testPowerMod(bool print)
{
	// test modular exponentiation 
	BigInteger* bigInteger = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
		"fffffffffffffff");
	BigInteger* exponent = BigInteger::fromHexString("10001");
	BigInteger* mod = BigInteger::fromHexString("76766766327663766766326766766326f9fa32766326f9fa6f9f766326f9faaf9fa3766326f9fa26f9fa766326f9fa26"
		"f9fa6f9766326f9fafa326f9766326f9fafa63766326f766326f9fa97663766326f9fa26f9fafa21c1766326f9fa7324766326f9fa766766326f76766766327663766766"
		"326766766326f9fa32766326f9fa6f9f766326f9faaf9fa3766326f9fa26f9fa766326f9fa26f9fa6f9766326f9fafa326f9766326f9fafa63766326f766326634766326"
		"f9fa9326f9f96cec15c67");	

	BigInteger* result = BigInteger::fromHexString("3526b7970e87462c90837b07a6af105122dbd34dcbbd6df757528e82c8ebccd26ee354415617a14d158951ff6fe77"
		"7ad623facbc211c6b4d3cb3688c949da188a1900c5b57a1b6fb8d55fd5d0b4b867fb86c152e53a5994e664bb14c8f97458f07aef731e47ddb718dc4c45d2be4a30deabe8"
		"0bc8e4b05b2c04d1e8f3297f49246212d791fcb1a2d303b3ea1da74c8cacaadef772489e2535a0fbc0ebd7a4ebc962ebfabf9c791a60955f147929ec1be9559f05243986"
		"7b982fa78faf78c7050ddef5");

	bigInteger->startTimer();
	bigInteger->powerMod(*exponent, *mod);
	unsigned long long time = bigInteger->stopTimer();

	bool ok = bigInteger->equals(*result);

	if (print || !ok)
	{
		if (!ok)
		{
			bigInteger->print("bigInteger:");
			result->print("result");
		}
		if (ok)
		{
			cout << "BigInteger::powerMod... SUCCESS elapsed time:  " << time << " cycles" << endl;
		}
		else
		{
			cout << "BigInteger::powerMod... FAILED elapsed time:  " << time << " cycles" << endl;
		}
	}

	delete bigInteger;
	delete exponent;
	delete mod;
	delete result;

	return time;
}