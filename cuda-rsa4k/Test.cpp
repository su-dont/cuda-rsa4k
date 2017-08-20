#include "Test.h"
#include "BigInteger.h"
#include <iostream>
#include <chrono>

using namespace std;

using ns = chrono::nanoseconds;
using get_time = chrono::steady_clock;

Test::Test()
{
}

Test::~Test()
{
}

void Test::runAll(bool print)
{
	testBigInteger(print);
}

void Test::testBigInteger(bool print)
{
	cout << "Testing BigInteger..." << endl;
		
	testParsing(print);
	testEquals(print);
	testCompare(print);
	testbitwiseLengthDiffrence(print);

	cout << "logics..." << endl;
	testShiftLeft(print);
	testShiftRight(print);

	cout << "arithmetics..." << endl;
	testAdd(print);
	testSubtract(print);
	testMultiply(print);
	testMod(print);	
}

void Test::testParsing(bool print)
{
	// test parsing
	const char* string = "1243abc312def391acd89897ad987f789868091243b45ac0"
							"7773FFAA83d19a3b3549937cfcF8a3dB9931254639186109"
							"ba3e70688a2f49CC67ff1dcfFF254639DF186BC109ADDE8c"
							"43ce74e249f9924093cebb70034a32a33ed731574ab50c3b";
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

long long Test::testbitwiseLengthDiffrence(bool print)
{
	// test bitwise length diffrence
	BigInteger* bigInteger = BigInteger::fromHexString("ffffffffffffffffffffffffffffffff");

	BigInteger* bigInteger2 = BigInteger::fromHexString("2000000");
	
	auto start = get_time::now();
	bool ok = bigInteger->getBitwiseLengthDiffrence(*bigInteger2) == 102;
	auto end = get_time::now();
	auto diff = end - start;

	if (print || !ok)
	{
		if (ok)
		{
			cout << "BigInteger::getBitwiseLengthDiffrence... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::equgetBitwiseLengthDiffrenceals... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}

	delete bigInteger;
	delete bigInteger2;

	return chrono::duration_cast<ns>(diff).count();
}

long long Test::testEquals(bool print)
{
	// test equals
	BigInteger* bigInteger = BigInteger::fromHexString("3e2bf9b23b2e1be1891e7fe032a06c2752abdfbd947a7edddbd9b980d26547f16ef4068e"
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
														"82b3cdb0be24af1bb42de45906956ce0c4cca117a20a519df37e0a02ed1ed20a0f2ba8ca"
														"1a2ad347cabcdaeefee446f07d6753964e8e37394442fc0a9a3e7dea75d64aad8298ace7");	

	auto start = get_time::now();
	bool ok = bigInteger->equals(*bigInteger2);
	auto end = get_time::now();
	auto diff = end - start;
	
	if (print || !ok)
	{
		if (ok)
		{
			cout << "BigInteger::equals... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::equals... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}

	delete bigInteger;
	delete bigInteger2;

	return chrono::duration_cast<ns>(diff).count();
}

long long Test::testCompare(bool print)
{
	// test equals
	BigInteger* bigInteger = BigInteger::fromHexString("3e2bf9b23b2e1be1891e7fe032a06c2752abdfbd947a7edddbd9b980d26547f16ef4068e"
														"493873573d8ff96ae2a730ecc737a7bd82f27fa482b8346c98e2bacb8f93cf748eb56909"
														"0c17c3c4d8bc6edc4033d320a45dbfbf14ed8b6fa67c9d45f1368fd272b5b7395c0fed4b"
														"dd7a61f377c9073a8705b500d8a173cc00085468eb76ba67b5aafc516bd6d71ef2381a4c"
														"6882398dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee16"
														"82b3cdb0be24af1bb42de45906956ce0c4cca117a20a519df37e0a02ed1ed20a0f2ba8ca"
														"1a2ad347cabcdaeefee446f07d6753964e8e37394442fc0a9a3e7dea75d64aad8298ace7");

	BigInteger* same = BigInteger::fromHexString("3e2bf9b23b2e1be1891e7fe032a06c2752abdfbd947a7edddbd9b980d26547f16ef4068e"
													"493873573d8ff96ae2a730ecc737a7bd82f27fa482b8346c98e2bacb8f93cf748eb56909"
													"0c17c3c4d8bc6edc4033d320a45dbfbf14ed8b6fa67c9d45f1368fd272b5b7395c0fed4b"
													"dd7a61f377c9073a8705b500d8a173cc00085468eb76ba67b5aafc516bd6d71ef2381a4c"
													"6882398dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee16"
													"82b3cdb0be24af1bb42de45906956ce0c4cca117a20a519df37e0a02ed1ed20a0f2ba8ca"
													"1a2ad347cabcdaeefee446f07d6753964e8e37394442fc0a9a3e7dea75d64aad8298ace7");

	BigInteger* greater = BigInteger::fromHexString("3e2bf9b23b2e1be1891e7fe032a06c2752abdfbd947a7edddbd9b980d26547f16ef4068e"
													"493873573d8ff96ae2a730ecc737a7bd82f27fa482b8346c98e2bacb8f93cf748eb56909"
													"1c17c3c4d8bc6edc4033d320a45dbfbf14ed8b6fa67c9d45f1368fd272b5b7395c0fed4b"
													"dd7a61f377c9073a8705b500d8a173cc00085468eb76ba67b5aafc516bd6d71ef2381a4c"
													"6882398dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee16"
													"82b3cdb0be24af1bb42de45906956ce0c4cca117a20a519df37e0a02ed1ed20a0f2ba8ca"
													"1a2ad347cabcdaeefee446f07d6753964e8e37394442fc0a9a3e7dea75d64aad8298ace7");

	BigInteger* lower = BigInteger::fromHexString("3e2bf9b23b2e1be1891e7fe032a06c2752abdfbd947a7edddbd9b980d26547f16ef4068e"
													"493873573d8ff96ae2a730ecc737a7bd82f27fa482b8346c98e2bacb8f93cf748eb56909"
													"0c17c3c4d8bc6edc4033d320a45dbfbf14ed8b6fa67c9d45f1368fd272b5b7395c0fed4b"
													"dd7a61f377c9073a8705b500d8a173cc00085468eb76ba67b5aafc516bd6d71ef2381a4c"
													"6882398dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee16"
													"82b3cdb0be24af1bb42de45906956ce0c4cca117a20a519df37e0a02ed1ed20a0f2ba8ca"
													"1a2ad347caacdaeefee446f07d6753964e8e37394442fc0a9a3e7dea75d64aad8298ace7");

	auto start = get_time::now();
	bool ok = bigInteger->compare(*same) == 0;
	ok = ok && bigInteger->compare(*greater) == 1;
	ok = ok && bigInteger->compare(*lower) == -1;
	auto end = get_time::now();
	auto diff = end - start;	

	if (print || !ok)
	{
		if (ok)
		{
			cout << "BigInteger::compare... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() / 3 << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::compare... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() / 3 << " ns" << endl;
		}
	}

	delete bigInteger;
	delete same;
	delete greater;
	delete lower;

	return chrono::duration_cast<ns>(diff).count() / 3;
}

long long Test::testAdd(bool print)
{
	// test add parallel
	BigInteger* bigInteger = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffff");

	BigInteger* added = BigInteger::fromHexString("1");

	BigInteger* result = BigInteger::fromHexString("100000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
													"000000000000000000000000000000000000000000000000000000000000000");

	auto start = get_time::now();
	bigInteger->add(*added);
	auto end = get_time::now();
	auto diff = end - start;
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
			cout << "BigInteger::add... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::add... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}	


	delete bigInteger;
	delete added;
	delete result;

	return chrono::duration_cast<ns>(diff).count();
}

long long Test::testSubtract(bool print)
{
	// test subtract parallel
	BigInteger* bigInteger = BigInteger::fromHexString("100000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"00000000000000000000000000000000000000000000000000000000000000000000000000000000"
														"000000000000000000000000000000000000000000000000000000000000000");

	BigInteger* subtracted = BigInteger::fromHexString("1");

	BigInteger* result = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffff");

	auto start = get_time::now();
	bigInteger->subtract(*subtracted);
	auto end = get_time::now();
	auto diff = end - start;
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
			cout << "BigInteger::subtract... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::subtract... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}


	delete bigInteger;
	delete subtracted;
	delete result;

	return chrono::duration_cast<ns>(diff).count();
}

long long Test::testMultiply(bool print)
{
	// test multiply parallel
	BigInteger* bigInteger = BigInteger::fromHexString("3e2bf9b23b2e1be1891e7fe032a06c2752abdfbd947a7edddbd9b980d26547f16ef4068e"
													  "493873573d8ff96ae2a730ecc737a7bd82f27fa482b8346c98e2bacb8f93cf748eb56909"
													  "0c17c3c4d8bc6edc4033d320a45dbfbf14ed8b6fa67c9d45f1368fd272b5b7395c0fed4b"
													  "dd7a61f377c9073a8705b500d8a173cc00085468eb76ba67b5aafc516bd6d71ef2381a4c"
													  "6882398dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee16"
													  "82b3cdb0be24af1bb42de45906956ce0c4cca117a20a519df37e0a02ed1ed20a0f2ba8ca"
													  "1a2ad347cabcdaeefee446f07d6753964e8e37394442fc0a9a3e7dea75d64aad8298ace7"
													  "6e26f9c3650255a6dc80d273b07cfb13ccac940011188624a029c9b0fdda9b36132b78ca"
													  "a82ed0d00cb287a0ec2a9ddf52c67dec2edde28f59172c82ad68b5d59600e8ef2f19e9fff");
	BigInteger* multiplied = BigInteger::fromHexString("41e1c98f1fbf9dce616092207503df412b6abb4e5c351dfe33ce4d4b9382309a2aa9f931"
													"afbe10435ee11009d1a0d1d78da36539926c10d4ff33ca664c5cc13c6909396c24498bd6"
													"78f45a5a2de167ea082e71d64ee4a8e39859b9b312bd7859f2616d6ea46cc95623fd5315"
													"f90edb9ac5e4b4069a3647b5613f606bde716b34faeca5fa147d127359f5dfdf627b455f"
													"dc4e556d160d8ff72cc800f80592d3809d82ea2f83bae2131702f7a4799a52756870b9a2"
													"37ebb93e2cdcbd9");
	BigInteger* result = BigInteger::fromHexString("100000000000000000000000000000000000000000000000000000000000000000000000000"
												  "000000000000000000000000000000000000000000000000000000000000000000000000000"
												  "000000000000000000000000000000000000000000000000000000000000000000000000000"
												  "000000000000000000000000000000000000000000000000000000000000000000000000000"
												  "000000000000000000000000000000000000000000000000000000000000000000000000000"
												  "000000000000000000000000000000000000000000000000000000000000000000000000000"
												  "000000000000000000000000000000000000000000000000000000000000000000000000000"
												  "000000000000000000000000000000000000000000000000000000000000000000000000000"
												  "00000000000000000000000000000000000000000000001fa0904b664e9676d28e9fd80bd13"
												  "12290fb37fa15238e039460194b98ffbd57ecec30e873e2d1469f3c25be3805b50c51c666c9"
												  "45b76058e87a6093ec72fe3c2575d130af2d730390cf448be26e845a38f33f82668981d1c85"
												  "92401ad7fe12738a6262ac27a345eebe1b397179522e34749d172c595ed79682c28e76ed90f"
												  "35d9c1ef0d95d1216efc566123271ec86892dd715ca9f6362f078c6121b2e7788a01a725aea"
												  "8e67768c755ba7fd360f0806b97af222123027b39d587d427");

	auto start = get_time::now();
	bigInteger->multiply(*multiplied);
	auto end = get_time::now();
	auto diff = end - start;
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
			cout << "BigInteger::multiply... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::multiply... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}	

	delete bigInteger;
	delete multiplied;
	delete result;

	return chrono::duration_cast<ns>(diff).count();
}


long long Test::testShiftLeft(bool print)
{
	// test left shift
	BigInteger* bigInteger = BigInteger::fromHexString("1243abc312def391acd89897ad987f789868091243b45ac0"
														"7773FFAA83d19a3b3549937cfcF8a3dB9931254639186109"
														"ba3e70688a2f49CC67ff1dcfFF254639DF186BC109ADDE8c"
														"43ce74e249f9924093cebb70034a32a33ed731574ab50c3b");

	BigInteger* result = BigInteger::fromHexString("2487578625bde72359b1312f5b30fef130d012248768b580eee7ff"
													"5507a334766a9326f9f9f147b732624a8c7230c213747ce0d1145"
													"e9398cffe3b9ffe4a8c73be30d782135bbd18879ce9c493f32481"
													"279d76e0069465467dae62ae956a1876000000000000000000000"
													"00000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000"
													"00000000000000000000000000000000000000000000000000000"
													"0000000000000000000000000000000000000000000000000000000000000");
			
	auto start = get_time::now();
	bigInteger->shiftLeft(1389);
	auto end = get_time::now();
	auto diff = end - start;
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
			cout << "BigInteger::shiftLeft... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::shiftLeft... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}
	
	delete bigInteger;
	delete result;

	return chrono::duration_cast<ns>(diff).count();
}


long long Test::testShiftRight(bool print)
{
	// test reight shift
	BigInteger* bigInteger = BigInteger::fromHexString("1243abc312def391acd89897ad987f789868091243b45ac0"
														"7773FFAA83d19a3b3549937cfcF8a3dB9931254639186109"
														"ba3e70688a2f49CC67ff1dcfFF254639DF186BC109ADDE8c"
														"43ce74e249f9924093cebb70034a32a33ed731574ab50c3b");

	BigInteger* result = BigInteger::fromHexString("921d5e1896f79c8d66c4c4bd6cc3fbc4c34048921da2d603bb9ffd541");
	

	auto start = get_time::now();
	bigInteger->shiftRight(537);
	auto end = get_time::now();
	auto diff = end - start;

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
			cout << "BigInteger::shiftRight... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::shiftRight... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}

	delete bigInteger;
	delete result;

	return chrono::duration_cast<ns>(diff).count();
}

long long Test::testMod(bool print)
{
	// test reduction modulo
	BigInteger* bigInteger = BigInteger::fromHexString("493873573d8ff96ae2507a334766a9326f9f9f147b732624a8c7230c2137a507a334766326f9fa9326f9f9f1"
														"47326f9fb732624f9fa8c72c2137730ecc737a7bdf27fa482b8346c9507a334766a9326f9f9f147b732624a"
														"8c7230c21378e2bacb8f93cf748eb569091c17c3c4d8bc6edc4033d320a45dbfbf14ed8b6fa67c9df1368fd"
														"272b5b7395c81a4c68823981a4c6882398dca52c732f8dca5c7372f0fed4bdd7a61f3c9073a870b500d8a17"
														"3cc00085468eb76ba67b5aafc516bd6d71ef2381a4c63981c17326f9f6edc4033d3326f9326f9fffbf14ed8"
														"507a334766a9326f9f9f326f9f147b732507a33326f9f4766a326f9f9326f9f9f147b73262a8c7232137624"
														"a8c7230c2137b6fa67c45f1368fd272b5b7395c0fed4bdd7a61f377c9073a8705b50d8a173cc00085468eb7"
														"6ba67b5aafc5d6d71ef2381a4c6882398dca52c7372f700fa6fbabc0aecc68823981a4c8823906c881a4c68"
														"823998dca52c73c68823981a4c68823972f881a4c688239c68823981a4c6326f9f882398dca52c732f9f372"
														"fa52c73706326f9f956c326f9fe0c4cca117a256ce0906956ce0c4cca117a206956ce0c4326f9fcca1326f9"
														"f17a24e72ba344ee16dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee1682b3"
														"cdb0be24af1bb42de45906956ce0c4cca117a20a519df37e0a02ed1ed20a2ba8ca");
	BigInteger* mod = BigInteger::fromHexString("1c1732493873573d8ff96ae2507a334766a9326f9f9f147b732624a8c7230c2137a507a334766326f9fa9326f9f9f14"
												"7326f9fb32624326f9fa8c7230c2137730ecc737bd2f27fa482b8346c907a334766a9326f9f9f147b732624a8c7230c"
												"21378e2bacb8cf748eb569091c17c3c4d8bc6edc033d320a45dbfbf14ed8b6fa67c91368fd272b5b7395c81a4c68823"
												"981a4c6882398dca52c7372f8dcc7372f0fed4bdda61f377c9073a8705b500d8173cc00085468eb76ba67b5aafc516b"
												"d6d71ef2381a4c688239dca52c7372f700fa6fbabc0aec06c8f03389c6be6499d6fed5a84e72ba344ee1682b3cdb0be"
												"24afb42de45906956ce0c4cca117a20a51370a02ed1d20a0f2ba8ca6f9f6ec4033d3326f9326f9fffbf14ed8507a334"
												"766a9326f9f9f3f9f147b732507a3326f9f476a326f9f9326f9f9f147b732624a8c7230c2137624a8c7230c2137b6fa"
												"67c45f1368fd272b5b7395c0fed4bdd7a61f377c9073a87050d8a173cc0008546eb76ba67b5aafc516bd6d71ef2381a"
												"4c6882398dca52c7372f700fa6fbbc0aecc688281a4c82906c88a4c6823981a4c6882398dca52c73c68823981a4c688"
												"972f881a4c68239c68823981a4c6326f9f882398dca52c7326ff372fdca52c737906326f9f956c326f9fe0c4cca117a"
												"256ce0906956ce0c4cca117a206956ce0c4326fcca1326f9f17a24e72ba344ee16");
	BigInteger* result = BigInteger::fromHexString("1318404bc43b47ce2f9c6c14142a0038963413f75d3ce7fbc266439b21dc6f551a1456923081c2df92be7e7e3c87"
													"0e803dbf46c5151919a2a506f9a619b2ad823cd5f8fbab0049d71003b840c429759f8de7127b7b98d110d8bcca4"
													"037dccb89bcd09f795949d6bf43a13728c375ed6a5a61fb1292b379e2de6df11477912324a702a14c1770cad88a"
													"6ae83400e83caf439e3dbe06f3b7ac26695ba7e8412ddca48d7f5f924702317160431f6a72b659c5d1a0def0a9d"
													"793d0d798576830788869d66d7cd476009442325bbc6de5e0d2c45d0212e856893248dcec7c858b77bed1c89d30"
													"8d9f60532a17ee43530ff21c5a6001c53ae984d899ec54c962b42b46df6bcb9920681f589cea8ef9da34b8a908d"
													"aacfb19e1029392d4aa998a0ab3425d0ec2e4d1a1faa796ae4ba5af76ee6a3b0066f7c4f1a6addf3c09b0d82149"
													"a8b5dd0859f8dee33ec580f31e0a4d6c9f3b5d9dfd58d2221330ab84c7791164a13d40479c2799ea42a3e3759aa"
													"b675d7c2b88e4d09044c1630036d39ffd5b6774a95915f18e0a019bada9a7d82dddc7f698c845a8e67aa3b52e22"
													"1cfc65f20fa32d36ce418912e242c4369740524cefece5f85d666043d68daef5f28ca8441d9266628fa41720e62"
													"465a61577dd4b3e12a8ae80324fd6d5c45a844a4a4e26c2f00b47bec7aeae3beb3ad306d7b0d2b88cbda28cd023"
													"3d0094819f1efe");

	auto start = get_time::now();
	bigInteger->mod(*mod);
	auto end = get_time::now();
	auto diff = end - start;
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
			cout << "BigInteger::mod... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::mod... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}

	delete bigInteger;
	delete mod;
	delete result;

	return chrono::duration_cast<ns>(diff).count();
}
