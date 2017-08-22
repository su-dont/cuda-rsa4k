#include "BigInteger.h"
#include "Test.h"
#include <conio.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main()
{
	cudaDeviceReset();
	
	Test test;	
	test.runAll(true, 2048);

	cudaDeviceSynchronize();	
	cudaDeviceReset();

	_getch();	
    return 0;
}
