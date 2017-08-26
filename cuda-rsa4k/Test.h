#pragma once
class Test
{

public:

	Test();
	~Test();

	void runAll(bool print, int minBits, int maxBits, int step, int repeats);
	void testBigIntegerCorrectness(bool print);
	void testRsaCorrectness(bool print);
	void testBigIntegerTimes(int minBits, int maxBits, int step, int repeats);
	void testRsaTimes(int minBits, int maxBits, int step, int repeats);

	void testEqualsTimings(int minBits, int maxBits, int step, int repeats);
	void testCompareTimings(int minBits, int maxBits, int step, int repeats);
	void testShiftLeftTimings(int minBits, int maxBits, int step, int repeats);
	void testShiftRightTimings(int minBits, int maxBits, int step, int repeats);
	void testAddTimings(int minBits, int maxBits, int step, int repeats);
	void testSubtractTimings(int minBits, int maxBits, int step, int repeats);
	void testMultiplyTimings(int minBits, int maxBits, int step, int repeats);
	void testSquareTimings(int minBits, int maxBits, int step, int repeats);
	void testModTimings(int minBits, int maxBits, int step, int repeats);
	void testMultiplyModTimings(int minBits, int maxBits, int step, int repeats);
	void testSquareModTimings(int minBits, int maxBits, int step, int repeats);
	void testPowerModTimings(int minBits, int maxBits, int step, int repeats);
	void testRsaTimings(int minBits, int maxBits, int step, int repeats);

	void testParsing(bool print);
	void testCreateRandom(bool print);
	void testTestBit(bool print);
	unsigned long long testBitwiseLengthDiffrence(bool print);
	unsigned long long testEquals(bool print);
	unsigned long long testCompare(bool print);
	unsigned long long testAdd(bool print);
	unsigned long long testSubtract(bool print);
	unsigned long long testMultiply(bool print);
	unsigned long long testSquare(bool print);
	unsigned long long testShiftLeft(bool print);
	unsigned long long testShiftRight(bool print);
	unsigned long long testMod(bool print);
	unsigned long long testMultiplyMod(bool print);
	unsigned long long testSquareMod(bool print);
	unsigned long long testPowerMod(bool print);
	unsigned long long testRsa(bool print);

private:
	unsigned long long testBitwiseLengthDiffrenceTime(int bits);
	unsigned long long testEqualsTime(int bits);
	unsigned long long testCompareTime(int bits);
	unsigned long long testAddTime(int bits);
	unsigned long long testSubtractTime(int bits);
	unsigned long long testMultiplyTime(int bits);
	unsigned long long testSquareTime(int bits);
	unsigned long long testShiftLeftTime(int bits, int n);
	unsigned long long testShiftRightTime(int bits, int n);
	unsigned long long testModTime(int bits);
	unsigned long long tesMultiplytModTime(int bits);
	unsigned long long testSquareModTime(int bits);
	unsigned long long testPowerModTime(int bits);
	unsigned long long testRsaTime(int bits);
};

