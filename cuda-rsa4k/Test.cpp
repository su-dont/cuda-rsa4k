#include "Test.h"
#include "BigInteger.h"
#include <iostream>

using namespace std;

Test::Test()
{
}


Test::~Test()
{
}

void Test::runAll(void)
{
	testBigInteger();
}

void Test::testBigInteger(void)
{
	cout << "Testing BigInteger..." << endl;
	
	// test parsing
	const char* string = "1243abc312def391acd89897ad987f789868091243b45ac0"
						 "7773FFAA83d19a3b3549937cfcF8a3dB9931254639186109"
						 "ba3e70688a2f49CC67ff1dcfFF254639DF186BC109ADDE8c"
						 "43ce74e249f9924093cebb70034a32a33ed731574ab50c3b";
	BigInteger bigInteger = BigInteger::fromHexString(string);
	char* string2 = bigInteger.toHexString();
	bool ok = _stricmp(string, string2) == 0;
	if (ok)
	{
		cout << "BigInteger::fromHexString and BigInteger::toHexString... SUCCESS" << endl;
	}
	else
	{
		cout << "BigInteger::fromHexString or BigInteger::toHexString... FAILED" << endl;
	}

	// test left shift
	bigInteger = BigInteger::fromHexString("1243abc312def391acd89897ad987f789868091243abc312def391acd8989"
									       "7ad987f75436486afcdf232f6f4ffd9012340198354abcdef");

	bigInteger.leftShift(492);
	ok = _stricmp(bigInteger.toHexString(), "1243abc312def391acd89897ad987f789868091243abc312def391acd"
										  "89897ad987f75436486afcdf232f6f4ffd9012340198354abcdef00000"
										  "00000000000000000000000000000000000000000000000000000000000"
										  "00000000000000000000000000000000000000000000000000000000000") == 0;
	if (ok)
	{
		cout << "BigInteger::leftShift... SUCCESS" << endl;
	}
	else
	{
		cout << "BigInteger::leftShift... FAILED" << endl;
	}
	
	// test add
	bigInteger = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffff543ced210c6e532767"
											"68526780876797f6edbc543ced210c6e53276768526780876797f61f7076f3f677"
											"2675f67178eadcb7026fedcbfe67cab5432101f7076f3f6772675f67178edbc543"
											"ced210c6e53276768526780876797f6edbc4ba53f888cf5babf5d87c2e65c4cab6"
											"6c830375c266cedab9c6e79ef645c18f9775d0b65da3214aa3890a4d2df66664f2"
											"745e3c2e2dfdbdf66664f2745e3c2e2dfdceadcb7026fedcbfe67cab543210543c"
											"ed210c6e53276768526780876797f6edbc543ced210c6e53276768526780876797"
											"f61f7076f3f6772675f67178eadcb7026fedcbfe67cab5432101f7076f3f677267"
											"5f67178edbc543ced210c6e53276768526780876797f6edbc4ba53f888cf5babf5"
											"d87c2e65c4cab66c830375c266cedab9c6e79ef645c18f9775d0b65da3214aa389"
											"0a4d2df66664f2745e3c2e2dfdbdf66664f2745e3c2e2dfdceadcb7026fedcbfe6"
											"7cab543210543ced210c6e53276768526780876797f6edbc543ced210c6e532767"
											"68526780876797f61f7076f3f6772675f67178eadcb7026fedcbfe67cab5432101"
											"f7076f3f6772675f67178edbc543ced210c6e53276768526780876797f6edbc4ba"
											"53f888cf5babf5d87c2e65c4cab66c830375c266cedab9c6e79ef645c18f9775d0"
											"b65da3214aa3890a4d2df66664f2745e3");
	BigInteger added = BigInteger::fromHexString("abc312def391acd89897ad987f789868091243abc312def391acd89897ad9"
												 "87f78986809e08f890c0988d98a098e87152348fd9012340198354abcdefe"
												 "08f890c0988d98a098e871243abc312def391acd89897ad987f7898680912"
												 "43b45ac077730a4540a2783d19a3b3549937cfc8a3d9931254639186109ba"
												 "3e70688a2f49a25cdeb55c76f5b2d209999b0d8ba1c3d1d2024209999b0d8"
												 "ba1c3d1d2023152348fd9012340198354abcdefabc312def391acd89897ad"
												 "987f789868091243abc312def391acd89897ad987f78986809e08f890c098"
												 "8d98a098e87152348fd9012340198354abcdefe08f890c0988d98a098e871"
												 "243abc312def391acd89897ad987f789868091243b45ac077730a4540a278"
												 "3d19a3b3549937cfc8a3d9931254639186109ba3e70688a2f49a25cdeb55c"
												 "76f5b2d209999b0d8ba1c3d1d2024209999b0d8ba1c3d1d2023152348fd90"
												 "12340198354abcdefabc312def391acd89897ad987f789868091243abc312"
												 "def391acd89897ad987f78986809e08f890c0988d98a098e87152348fd901"
												 "2340198354abcdefe08f890c0988d98a098e871243abc312def391acd8989"
												 "7ad987f789868091243b45ac077730a4540a2783d19a3b3549937cfc8a3d9"
												 "931254639186109ba3e70688a2f49a25cdeb55c76f5b2d209999b0d8ba1c");
	BigInteger result = BigInteger::fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
												  "ffff");
		
	bigInteger.add(added);
	ok = bigInteger.equals(result);
	if (ok)
	{
		cout << "BigInteger::add... SUCCESS" << endl;
	}
	else
	{
		cout << "BigInteger::add... FAILED" << endl;
	}

	// test add partial
	bigInteger = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
	added = BigInteger::fromHexString("1");
	result = BigInteger::fromHexString("1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

	bigInteger.addParallel(added);
	ok = bigInteger.equals(result);
	if (ok)
	{
		cout << "BigInteger::addParallel... SUCCESS" << endl;
	}
	else
	{
		cout << "BigInteger::addParallel... FAILED" << endl;
	}

	
	// test multiply
	bigInteger = BigInteger::fromHexString("419994ba5751fdef22fa8f9aece028df0c83101cdf669edbf583ee30239f3da0d52b0"
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
	result = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
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

	bigInteger.multiply(multiplied);
	ok = bigInteger.equals(result);
	if (ok)
	{
		cout << "BigInteger::multiply... SUCCESS" << endl;
	}
	else
	{
		cout << "BigInteger::multiply... FAILED" << endl;
	}	

	// test multiply parallel
	bigInteger = BigInteger::fromHexString("419994ba5751fdef22fa8f9aece028df0c83101cdf669edbf583ee30239f3da0d52b0"
										   "2f18b1646de3a678517c8516a87f5d38ad5cc8a7d496fb7844d2e4a6e1f6d1cd27546"
										   "156dcdc45ff6baaa089e3262412a1e0db224cf9cb8af4b73436421fbb8b4f23d303bf"
										   "c002b524eaf7309d322403c49fe097156a4e91005b4bee5b952ab0b7788e19904274f"
										   "d3004ac7834a75847a851b984ba5909d9d24e400243d0292dafdd574225b1ec13c7c4"
										   "3385a3aeff2be9e50d80606154735d3f50a030c88c7dcde4a7c487a4b0a6091988733"
										   "e276a7eb6b66c90dca629649fa4a1433ef0ed229515272bb1c5611355074287083806"
										   "7c41f878b67909ca952af998da8025246986699aa9cb72616c8de9");
	multiplied = BigInteger::fromHexString("3e70688a2f49a25cdeb55c76f5b2d209999b0b2d209999b0d8ba1c3d1d"
											"2024209999b0d8ba1c3d1d2023152348fd9012340198354abcdefd8ba1c3d1d202b2"
											"d209999b0d8ba1c3d1d2024209999b0db2d209999b0d8ba1c3d1d2024209999b0d8b"
											"a1c3d1d2023152348fd9012340198354abcdef8ba1c3d1d2023152348fd901234019"
											"8354abcdef4209999b0d8ba1c3d1d2023152348fd9012340198354abcdefabc312de"
											"f391acd89897ad987f789868091243abc312def391acd89897ad987f78986809e08f"
											"890c0988d98a098e87152348fd9012340198354abcdefe08f890c0988d98a099999b"
											"0d8ba1c3d1d2024208e871");
	result = BigInteger::fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
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

	bigInteger.multiplyParallel(multiplied);
	ok = bigInteger.equals(result);
	if (ok)
	{
		cout << "BigInteger::multiplyParallel... SUCCESS" << endl;
	}
	else
	{
		cout << "BigInteger::multiplyParallel... FAILED" << endl;
	}
}
