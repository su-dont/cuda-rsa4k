#pragma once
class Test
{

public:

	Test();
	~Test();

	void runAll(bool print);
	void testBigInteger(bool print);
	void testParsing(bool print);
	long long  testbitwiseLengthDiffrence(bool print);
	long long testEquals(bool print);	
	long long testCompare(bool print);
	long long testAdd(bool print);
	long long testSubtract(bool print);
	long long testMultiply(bool print);
	long long  testShiftLeft(bool print);
	long long  testShiftRight(bool print);
	long long testMod(bool print);
};

