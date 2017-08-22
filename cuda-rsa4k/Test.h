#pragma once
class Test
{

public:

	Test();
	~Test();

	void runAll(bool print, int bits);
	
	void testBigInteger(bool print);
	void testBigIntegerTimes(int bits);

	unsigned long long testBitwiseLengthDiffrenceTime(int bits);
	unsigned long long testEqualsTime(int bits);
	unsigned long long testCompareTime(int bits);
	unsigned long long testAddTime(int bits);
	unsigned long long testSubtractTime(int bits);
	unsigned long long testMultiplyTime(int bits);
	unsigned long long testShiftLeftTime(int bits, int n);
	unsigned long long testShiftRightTime(int bits, int n);
	unsigned long long testModTime(int bits);
	unsigned long long testPowerModTime(int bits);


	void testParsing(bool print);
	void testCreateRandom(bool print);
	unsigned long long testBitwiseLengthDiffrence(bool print);
	unsigned long long testEquals(bool print);
	unsigned long long testCompare(bool print);
	unsigned long long testAdd(bool print);
	unsigned long long testSubtract(bool print);
	unsigned long long testMultiply(bool print);
	unsigned long long testShiftLeft(bool print);
	unsigned long long testShiftRight(bool print);
	unsigned long long testMod(bool print);
	unsigned long long testPowerMod(bool print);
};

