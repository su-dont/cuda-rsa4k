#pragma once
#include <cuda_runtime.h>

class DeviceWrapper
{

private:

	// main stream for kernel launches
	cudaStream_t mainStream;

	// lauch config
	dim3 block_1, block_2, block_4;
	dim3 thread_warp, thread_2_warp, thread_4_warp;

	// 4 ints to help store results
	int* deviceWords;

	// auxiliary arrays
	unsigned int* device4arrays;
	unsigned int* deviceArray;

	unsigned long long* deviceStartTime;
	unsigned long long* deviceStopTime;

public:

	DeviceWrapper();
	~DeviceWrapper();
	
	// sync
	unsigned int* init(int size) const;
	unsigned int* init(int size, const unsigned int* initial) const;
	void updateDevice(unsigned int* device_array, const unsigned int* host_array, int size) const;
	void updateHost(unsigned int* host_array, const unsigned int* device_array, int size) const;
	void free(unsigned int* device_x) const;

	// extras
	void clearParallel(unsigned int* device_x) const;
	void cloneParallel(unsigned int* device_x, const unsigned int* device_y) const;
	int compareParallel(const unsigned int* device_x, const unsigned int* device_y) const;
	bool equalsParallel(const unsigned int* device_x, const unsigned int* device_y) const;
	int getLSB(const unsigned int* device_x) const;
	int getBitLength(const unsigned int* device_x) const;

	// measure time
	void startClock(void);
	unsigned long long stopClock(void);

	// logics
	void shiftLeftParallel(unsigned int* device_x, int bits) const;
	void shiftRightParallel(unsigned int* device_x, int bits) const;

	// arithmetics
	void addParallel(unsigned int* device_x, const unsigned int* device_y) const;
	void subtractParallel(unsigned int* device_x, const unsigned int* device_y) const;
	void multiplyParallel(unsigned int* device_x, const unsigned int* device_y) const;
	void modParallel(unsigned int* device_x, unsigned int* device_m) const;
	void multiplyModParallel(unsigned int* device_x, const unsigned int* device_y, const unsigned int* device_m) const;

private:
	void inline addParallelWithOverflow(unsigned int* device_x, const unsigned int* device_y, int blocks) const;

};
