#pragma once
class Test
{

public:

	Test();
	~Test();

	void runAll(bool print);
	void testBigInteger(bool print);
	void testParsing(bool print);
	unsigned long long testbitwiseLengthDiffrence(bool print);
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

