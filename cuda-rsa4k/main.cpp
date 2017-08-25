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
	test.testPowerMod(true);
	//test.testMultiplyMod(true);
	

	cudaDeviceSynchronize();	
	cudaDeviceReset();

	_getch();	
    return 0;
}
