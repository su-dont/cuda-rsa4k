#include "BigInteger.h"
#include "Test.h"
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main()
{
	cudaDeviceReset();

	Test test;
	test.runAll(true);

	cudaDeviceSynchronize();	
	cudaDeviceReset();

	int exit;
	cin >> exit;
    return 0;
}
