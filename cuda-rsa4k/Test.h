#pragma once
class Test
{

public:

	Test();
	~Test();

	void runAll(bool print);
	void testBigInteger(bool print);
	void testParsing(bool print);
	void testLeftShift(bool print);
	long long testAddParallel(bool print);
	long long testMultiplyParallel(bool print);
};

