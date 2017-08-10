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
	testLeftShift(print);
	testAdd(print);
	testAddParallel(print);
	testMultiply(print);
	testMultiplyParallel(print);
}

void Test::testParsing(bool print)
{
	// test parsing
	const char* string = "1243abc312def391acd89897ad987f789868091243b45ac0"
							"7773FFAA83d19a3b3549937cfcF8a3dB9931254639186109"
							"ba3e70688a2f49CC67ff1dcfFF254639DF186BC109ADDE8c"
							"43ce74e249f9924093cebb70034a32a33ed731574ab50c3b";
	BigInteger bigInteger = BigInteger::fromHexString(string);
	char* string2 = bigInteger.toHexString();
	bool ok = _stricmp(string, string2) == 0;
	if (print) 
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
}

void Test::testLeftShift(bool print)
{
	// test left shift
	BigInteger bigInteger = BigInteger::fromHexString("1243abc312def391acd89897ad987f789868091243abc312def391acd8989"
														"7ad987f75436486afcdf232f6f4ffd9012340198354abcdef");

	bigInteger.leftShift(492);
	bool ok = _stricmp(bigInteger.toHexString(), "1243abc312def391acd89897ad987f789868091243abc312def391acd"
							"89897ad987f75436486afcdf232f6f4ffd9012340198354abcdef00000"
							"00000000000000000000000000000000000000000000000000000000000"
							"00000000000000000000000000000000000000000000000000000000000") == 0;

	if (print) 
	{
		if (ok)
		{
			cout << "BigInteger::leftShift... SUCCESS" << endl;
		}
		else
		{
			cout << "BigInteger::leftShift... FAILED" << endl;
		}
	}	
}

long long Test::testAdd(bool print)
{
	// test add
	BigInteger bigInteger = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
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

	BigInteger added = BigInteger::fromHexString("1");

	BigInteger result = BigInteger::fromHexString("100000000000000000000000000000000000000000000000000000000000000000000000000000000"
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
	bigInteger.add(added);
	auto end = get_time::now();
	auto diff = end - start;
	bool ok = bigInteger.equals(result);

	if (print) 
	{
		if (ok)
		{
			cout << "BigInteger::add... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::add... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}	

	return chrono::duration_cast<ns>(diff).count();
}

long long Test::testAddParallel(bool print)
{
	// test add parallel
	BigInteger bigInteger = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
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

	BigInteger added = BigInteger::fromHexString("1");

	BigInteger result = BigInteger::fromHexString("100000000000000000000000000000000000000000000000000000000000000000000000000000000"
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
	bigInteger.addParallel(added);
	auto end = get_time::now();
	auto diff = end - start;
	bool ok = bigInteger.equals(result);

	if (print) 
	{
		if (ok)
		{
			cout << "BigInteger::addParallel... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::addParallel... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}	

	return chrono::duration_cast<ns>(diff).count();
}

long long Test::testMultiply(bool print)
{
	// test multiply
	BigInteger bigInteger = BigInteger::fromHexString("419994ba5751fdef22fa8f9aece028df0c83101cdf669edbf583ee30239f3da0d52b0"
														"2f18b1646de3a678517c8516a87f5d38ad5cc8a7d496fb7844d2e4a6e1f6d1cd27546"
														"156dcdc45ff6baaa089e3262412a1e0db224cf9cb8af4b73436421fbb8b4f23d303bf"
														"c002b524eaf7309d322403c49fe097156a4e91005b4bee5b952ab0b7788e19904274f"
														"d3004ac7834a75847a851b984ba5909d9d24e400243d0292dafdd574225b1ec13c7c4"
														"3385a3aeff2be9e50d80606154735d3f50a030c88c7dcde4a7c487a4b0a6091988733"
														"e276a7eb6b66c90dca629649fa4a1433ef0ed229515272bb1c5611355074287083806"
														"7c41f878b67909ca952af998da8025246986699aa9cb72616c8de9");
	BigInteger multiplied = BigInteger::fromHexString("3e70688a2f49a25cdeb55c76f5b2d209999b0b2d209999b0d8ba1c3d1d"
														"2024209999b0d8ba1c3d1d2023152348fd9012340198354abcdefd8ba1c3d1d202b2"
														"d209999b0d8ba1c3d1d2024209999b0db2d209999b0d8ba1c3d1d2024209999b0d8b"
														"a1c3d1d2023152348fd9012340198354abcdef8ba1c3d1d2023152348fd901234019"
														"8354abcdef4209999b0d8ba1c3d1d2023152348fd9012340198354abcdefabc312de"
														"f391acd89897ad987f789868091243abc312def391acd89897ad987f78986809e08f"
														"890c0988d98a098e87152348fd9012340198354abcdefe08f890c0988d98a099999b"
														"0d8ba1c3d1d2024208e871");
	BigInteger result = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffff2081879e48183d3310d0760892fff163e4db24dc"
													"e1374a0ce0b9adf91fc31ab3c219d8ec31cd9c1f0f15fe74bbf5eb292f1535a981fcbc42"
													"41e1c98f1fbf9dce616092207503df412b6abb4e5c351dfe33ce4d4b9382309a2aa9f931"
													"afbe10435ee11009d1a0d1d78da36539926c10d4ff33ca664c5cc13c6909396c24498bd6"
													"78f45a5a2de167ea082e71d64ee4a8e39859b9b312bd7859f2616d6ea46cc95623fd5315"
													"f90edb9ac5e4b4069a3647b5613f606bde716b34faeca5fa147d127359f5dfdf627b455f"
													"dc4e556d160d8ff72cc800f80592d3809d82ea2f83bae2131702f7a4799a52756870b9a2"
													"37ebb93e2cdcbd9");

	auto start = get_time::now();
	bigInteger.multiply(multiplied);
	auto end = get_time::now();
	auto diff = end - start;
	bool ok = bigInteger.equals(result);

	if (print) 
	{
		if (ok)
		{
			cout << "BigInteger::multiply... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::multiply... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}

	return chrono::duration_cast<ns>(diff).count();
}

long long Test::testMultiplyParallel(bool print)
{
	// test multiply parallel
	BigInteger bigInteger = BigInteger::fromHexString("419994ba5751fdef22fa8f9aece028df0c83101cdf669edbf583ee30239f3da0d52b0"
											"2f18b1646de3a678517c8516a87f5d38ad5cc8a7d496fb7844d2e4a6e1f6d1cd27546"
											"156dcdc45ff6baaa089e3262412a1e0db224cf9cb8af4b73436421fbb8b4f23d303bf"
											"c002b524eaf7309d322403c49fe097156a4e91005b4bee5b952ab0b7788e19904274f"
											"d3004ac7834a75847a851b984ba5909d9d24e400243d0292dafdd574225b1ec13c7c4"
											"3385a3aeff2be9e50d80606154735d3f50a030c88c7dcde4a7c487a4b0a6091988733"
											"e276a7eb6b66c90dca629649fa4a1433ef0ed229515272bb1c5611355074287083806"
											"7c41f878b67909ca952af998da8025246986699aa9cb72616c8de9");
	BigInteger multiplied = BigInteger::fromHexString("3e70688a2f49a25cdeb55c76f5b2d209999b0b2d209999b0d8ba1c3d1d"
											"2024209999b0d8ba1c3d1d2023152348fd9012340198354abcdefd8ba1c3d1d202b2"
											"d209999b0d8ba1c3d1d2024209999b0db2d209999b0d8ba1c3d1d2024209999b0d8b"
											"a1c3d1d2023152348fd9012340198354abcdef8ba1c3d1d2023152348fd901234019"
											"8354abcdef4209999b0d8ba1c3d1d2023152348fd9012340198354abcdefabc312de"
											"f391acd89897ad987f789868091243abc312def391acd89897ad987f78986809e08f"
											"890c0988d98a098e87152348fd9012340198354abcdefe08f890c0988d98a099999b"
											"0d8ba1c3d1d2024208e871");
	BigInteger result = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
													"ffffffffffffffffffffffffffffffff2081879e48183d3310d0760892fff163e4db24dc"
													"e1374a0ce0b9adf91fc31ab3c219d8ec31cd9c1f0f15fe74bbf5eb292f1535a981fcbc42"
													"41e1c98f1fbf9dce616092207503df412b6abb4e5c351dfe33ce4d4b9382309a2aa9f931"
													"afbe10435ee11009d1a0d1d78da36539926c10d4ff33ca664c5cc13c6909396c24498bd6"
													"78f45a5a2de167ea082e71d64ee4a8e39859b9b312bd7859f2616d6ea46cc95623fd5315"
													"f90edb9ac5e4b4069a3647b5613f606bde716b34faeca5fa147d127359f5dfdf627b455f"
													"dc4e556d160d8ff72cc800f80592d3809d82ea2f83bae2131702f7a4799a52756870b9a2"
													"37ebb93e2cdcbd9");

	auto start = get_time::now();
	bigInteger.multiplyParallel(multiplied);
	auto end = get_time::now();
	auto diff = end - start;
	bool ok = bigInteger.equals(result);

	if (print)
	{
		if (ok)
		{
			cout << "BigInteger::multiplyParallel... SUCCESS elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
		else
		{
			cout << "BigInteger::multiplyParallel... FAILED elapsed time:  " << chrono::duration_cast<ns>(diff).count() << " ns" << endl;
		}
	}	

	return chrono::duration_cast<ns>(diff).count();

}
