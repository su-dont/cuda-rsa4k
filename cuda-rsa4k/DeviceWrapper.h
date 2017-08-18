#pragma once

class BigInteger;	// forward declaration

class DeviceWrapper
{

private:

	//Mapping to sepcific indices of shared memory in order to eliminate bank conflicts in device_multiply_partial
	//Dependency: 
	// return index % 64 * 32 + (index % 64 & 0xfffffffe) / 2 + index / 64 * 64;
	const unsigned int* indexFixupTable = new unsigned int[129] { 0, 32, 65, 97, 130, 162, 195, 227, 260, 292, 325, 357,
		390, 422, 455, 487, 520, 552, 585, 617, 650, 682, 715, 747, 780, 812, 845, 877, 910, 942, 975, 1007, 1040, 1072,
		1105, 1137, 1170, 1202, 1235, 1267, 1300, 1332, 1365, 1397, 1430, 1462, 1495, 1527, 1560, 1592, 1625, 1657, 1690,
		1722, 1755, 1787, 1820, 1852, 1885, 1917, 1950, 1982, 2015, 2047, 64, 96, 129, 161, 194, 226, 259, 291, 324, 356,
		389, 421, 454, 486, 519, 551, 584, 616, 649, 681, 714, 746, 779, 811, 844, 876, 909, 941, 974, 1006, 1039, 1071,
		1104, 1136, 1169, 1201, 1234, 1266, 1299, 1331, 1364, 1396, 1429, 1461, 1494, 1526, 1559, 1591, 1624, 1656, 1689,
		1721, 1754, 1786, 1819, 1851, 1884, 1916, 1949, 1981, 2014, 2046, 2079, 2111, 128 };

public:

	// two warps
	static const int MULTIPLICATION_THREAD_COUNT = 64;
	static const int MULTIPLICATION_BLOCKS_COUNT = 4;

	// one warp
	static const int ADDITION_THREAD_COUNT = 32;
	static const int ADDITION_CELLS_PER_THREAD = 4;	// BigInteger::ARRAY_SIZE / ADDITION_THREAD_COUNT

	DeviceWrapper();
	~DeviceWrapper();

	static unsigned long long getClock(void);

	unsigned int* addParallel(const BigInteger& x, const BigInteger& y);
	unsigned int* multiplyParallel(const BigInteger& x, const BigInteger& y);
};
