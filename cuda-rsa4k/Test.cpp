#include "Test.h"
#include "BigInteger.h"
#include "BuildConfig.h"
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
	testPowerMod(print);
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

long long Test::testbitwiseLengthDiffrence(bool print)
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
	
	auto start = get_time::now();
	bool ok = bigInteger->getBitwiseLengthDiffrence(*bigInteger2) == 4095;
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
			cout << "BigInteger::getBitwiseLengthDiffrence... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
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

	auto start = get_time::now();
	bool ok = bigInteger->compare(*same) == 0;
	ok = ok && bigInteger->compare(*lower) == 1;
	ok = ok && bigInteger->compare(*greater) == -1;	
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

	auto start = get_time::now();
	bigInteger->add(*added);
	auto end = get_time::now();
	auto diff = end - start;
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
			
	auto start = get_time::now();
	bigInteger->shiftLeft(1);
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
	// test right shift
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


long long Test::testPowerMod(bool print)
{
	// test modular exponentiation 
	BigInteger* bigInteger = BigInteger::fromHexString("7a334766a93261c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9fa9"
		"326f9f9f146f9fa9326f9f9f14f9f9f1471c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9fa9326f9f9f146f9fa9326f9f9f14"
		"b731c173249387351c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9fa9326f9f9f146f9fa9326f9f9f1473d8ff96ae257a3347"
		"66321c1732493873573d8ff96ae257a334766326f9fa9326f9f9f146f9fa9326f9f9f142624a11c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae2"
		"57a334766326f9fa9326f9f9f146f9fa9326f9f9f14c1732493873573d8ff96ae257a334766326f9fa9326f9f9f148c727a334766a9326f9f9f147b732624a8c7230c213"
		"7a507a334766326f9fa9326f9f9f130c2137a507a334766326f9fa9326f9f9f1");
	BigInteger* exponent = BigInteger::fromHexString("7");
	BigInteger* mod = BigInteger::fromHexString("1c1732493873573d8ff96a1c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9f"
		"a9326f9f9f146f9fa9326f9f9f14e257a33476631c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9fa9326f9f9f146f9fa9326f"
		"9f9f1421c173249381c1732493873573d8ff96ae257a334766321c1732493873573d8ff96ae257a334766326f9fa9326f9f9f146f9fa9326f9f9f1473573d8ff96ae257a"
		"334766326f9fa9326f9f9f146f9fa9326f9f9f14");	

	BigInteger* result = BigInteger::fromHexString("179804bc47fd5bb216e37871d9f6fdcb91921a9e2f4b634c5a5190c2e0a238dfdfb6b4c61a75c9be8697c3531d2a5"
		"c17f1b46bb2c1d2f8ea924113debed4529d9d59ea67465098cda1dd06b6feba5eb16fba7fcfaff50dc12b7b390c5355a89a193904ca7d1e63891c86f0f1cfc1aa87de040"
		"5d6bddb06c5b667dfc610459a4b363d3a436ccf94c93ccdd7d52456b1984d08898c025009ebb3a1b392d114b9be80d3f1ba6451cc9bb22363a66f57d6ccf01412371dea1"
		"0bf25929231736f1464635a15ed6a4bf1e87ea5970d");

	auto start = get_time::now();
	bigInteger->powerMod(*exponent, *mod);
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
			cout << "BigInteger::powerMod... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::powerMod... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}

	delete bigInteger;
	delete exponent;
	delete mod;
	delete result;

	return chrono::duration_cast<ns>(diff).count();
}