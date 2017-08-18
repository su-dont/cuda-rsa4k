#include "BigInteger.h"
#include "Test.h"
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main()
{
	Test test;
	test.runAll(true);
	
	cudaDeviceSynchronize();	
	
	int exit;
	cin >> exit;
    return 0;
}
