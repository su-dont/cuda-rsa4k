#pragma once
#include <cuda_runtime.h>

class DeviceWrapper
{

public:

	// statics
	static const int TWO_WARPS = 64;
	static const int MULTIPLICATION_BLOCKS_COUNT = 4;

	static const int ONE_WARP = 32;
	// both addition and subtraction
	static const int ADDITION_CELLS_PER_THREAD = 4;	// BigInteger::ARRAY_SIZE / ONE_WARP


private:

	// main stream for kernel launches
	cudaStream_t mainStream;
	
public:

	DeviceWrapper();
	~DeviceWrapper();

	// statics
	static unsigned long long getClock(void);	

	// sync
	int* init(int size) const;
	int* init(int size, const int* initial) const;
	void updateDevice(int* device_array, const unsigned int* host_array, int size) const;
	void updateHost(unsigned int* host_array, const int* device_array, int size) const;
	void free(int* device_x) const;

	// extras
	void clearParallel(int* device_x) const;
	void cloneParallel(int* device_x, const int* device_y) const;
	int compareParallel(const int* device_x, const int* device_y) const;
	int getLSB(const int* device_x) const;

	// logics
	void shiftLeftParallel(int* device_x, int bits) const;
	void shiftRightParallel(int* device_x, int bits) const;

	// arithmetics
	void addParallel(int* device_x, const int* device_y) const;
	void subtractParallel(int* device_x, const int* device_y) const;
	void multiplyParallel(int* device_x, const int* device_y) const;

};
