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
	test.runAll(true);

	cudaDeviceSynchronize();	
	cudaDeviceReset();

	getch();	
    return 0;
}
