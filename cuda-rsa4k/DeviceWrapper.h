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

	// 4 ints to help storing results
	int* deviceWords;	
	int* device4arrays;

	unsigned long long* deviceStartTime;
	unsigned long long* deviceStopTime;

public:

	DeviceWrapper();
	~DeviceWrapper();
	
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
	bool equalsParallel(const int* device_x, const int* device_y) const;
	int getLSB(const int* device_x) const;
	int getBitLength(const int* device_x) const;

	// measure time
	void startClock(void);
	unsigned long long stopClock(void);

	// logics
	void shiftLeftParallel(int* device_x, int bits) const;
	void shiftRightParallel(int* device_x, int bits) const;

	// arithmetics
	void addParallel(int* device_x, const int* device_y) const;
	void subtractParallel(int* device_x, const int* device_y) const;
	void multiplyParallel(int* device_x, const int* device_y) const;
	void modParallel(int* device_x, int* device_m) const;
	void multiplyModParallel(int* device_x, const int* device_y, const int* device_m) const;

private:
	void inline addParallelWithOverflow(int* device_x, const int* device_y, int blocks) const;

};
