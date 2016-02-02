#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <device_launch_parameters.h>

#include "Integer4K.cu"


#define MAX_NUMBER_OF_THREADS 512	// 2^9
#define MAX_NUMBER_OF_BLOCKS 65535	// 2^16 - 1



inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	}
	return result;
}

void testAdding()
{
	printf("\ntesting adding...");

	Integer4K *x = fromHexString("1243abc312def391acd89897ad987f789868091243abc312def391acd89897ad"
		"987f78986809e08f890c0988d98a098e87152348fd9012340198354abcdefe08f890c0988d98"
								 "a098e871243abc312def391acd89897ad987f789868091243b45ac"
								 "077730a4540a2783d19a3b3549937cfc8a3d9931254639186109ba3"
								 "e70688a2f49a25cdeb55c76f5b2d209999b0d8ba1c3d1d202420999"
								 "9b0d8ba1c3d1d2023152348fd9012340198354abcdef");
	Integer4K *y = fromHexString("897ad987f78986809e08f81243abc312def391acd89897ad987f7"
								 "8986809e08f890c0981243abc312def391acd89897ad987f789"
								 "868091243b45ac077730a4540a2783d19a3b3549937cfc8a3d9"
								 "931254639186109ba3e70688a2f49a25cdeb55c76f5b2d20999"
								 "9b0d8ba1c3d1d2024209999b0d8ba1c3d1d20238d98a098e871"
								 "52348fd9012340198354abcdef90c0988d98a098e8715234");
	Integer4K *z = (Integer4K*)malloc(sizeof(Integer4K));

	Integer4K *d_x, *d_y, *d_z;
	checkCuda(cudaMalloc(&d_z, sizeof(Integer4K)));
	checkCuda(cudaMalloc(&d_x, sizeof(Integer4K)));
	checkCuda(cudaMalloc(&d_y, sizeof(Integer4K)));
	
	checkCuda(cudaMemcpy(d_x, x, sizeof(Integer4K), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_y, y, sizeof(Integer4K), cudaMemcpyHostToDevice));

	add << <1, 1 >> >(d_z, d_x, d_y);

	checkCuda(cudaMemcpy(z, d_z, sizeof(Integer4K), cudaMemcpyDeviceToHost));
		
	//printf(toHexString(z));

	int ok = strcmp(toHexString(z), "1243abc312def391acd89897ad987f789868091243b45ac0"
								 "77730a4540a2783d19a3b3549937cfc8a3d9931254639186109"
								 "ba3e70688a2f499aa78f67ff1dcfc8a3d99312546391861098c"
								 "43ce74e249f9924093cebb7c0034a32a33e5bd731574ab50c3b"
								 "6a86db909e235df1c501c1ff12463671c7b0da8738e2a53d821"
								 "41444b86bf7d02d5f610a68b8c25d6b82daf2f5c726aeab35bc"
								 "e2ae68cc503eb5556dd2024209999b0d8ba1c3d1d2023");

	if (ok == 0)
		printf("ok!");
	else
		printf("not ok!");

	free(x);
	free(y);
	free(z);

	checkCuda(cudaFree(d_x));
	checkCuda(cudaFree(d_y));
	checkCuda(cudaFree(d_z));

}

void testSubtract()
{
	printf("\ntesting subtracting...");

	Integer4K *x = fromHexString("1243abc312def391acd89897ad987f789868091243abc312def3"
								 "91acd89897ad987f78986809e08f890c0988d98a098e87152348"
								 "fd9012340198354abcdefe08f890c0988d98a098e871243abc31"
								 "2def391acd89897ad987f789868091243b45ac077730a4540a27"
								 "83d19a3b3549937cfc8a3d9931254639186109ba3e70688a2f49"
								 "a25cdeb55c76f5b2d209999b0d8ba1c3d1d2024209999b0d8ba1"
								 "c3d1d2023152348fd9012340198354abcdef");
	Integer4K *y = fromHexString("897ad987f78986809e08f81243abc312def391acd89897ad987f"
								 "78986809e08f890c0981243abc312def391acd89897ad987f789"
								 "868091243b45ac077730a4540a2783d19a3b3549937cfc8a3d99"
								 "31254639186109ba3e70688a2f49a25cdeb55c76f5b2d209999b"
								 "0d8ba1c3d1d2024209999b0d8ba1c3d1d20238d98a098e871523"
								 "48fd9012340198354abcdef90c0988d98a098e8715234");
	Integer4K *z = (Integer4K*)malloc(sizeof(Integer4K));

	Integer4K *d_x, *d_y, *d_z;
	checkCuda(cudaMalloc(&d_z, sizeof(Integer4K)));
	checkCuda(cudaMalloc(&d_x, sizeof(Integer4K)));
	checkCuda(cudaMalloc(&d_y, sizeof(Integer4K)));

	checkCuda(cudaMemcpy(d_x, x, sizeof(Integer4K), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_y, y, sizeof(Integer4K), cudaMemcpyHostToDevice));

	subtract << <1, 1 >> >(d_z, d_x, d_y);

	checkCuda(cudaMemcpy(z, d_z, sizeof(Integer4K), cudaMemcpyDeviceToHost));

	//printf(toHexString(z));

	int ok = strcmp(toHexString(z), "1243abc312def391acd89897ad987f789868091243a32b6546"
								 "741914708eb71e175b3ddc36dbf1566e3e7fff5eb08196fd8ea2a"
								 "af49781736985f19ef9cc1f1566e3e7fff5eb08196fd8bc31a9ed"
								 "799478a35a7f443a370fba6fe2cd3c8b0375e3639d9d91ffa695f"
								 "dc0fe974e430ada07f017cb45cf7ec9bd33e920a4bf8fd0130c85"
								 "3cba94c2dd44da17ed5c7ded68142b313919d0b7650b348a7e83b"
								 "84ead858444ff186895a778ea6c3a7bbb");

	if (ok == 0)
		printf("ok!");
	else
		printf("not ok!");

	free(x);
	free(y);
	free(z);

	checkCuda(cudaFree(d_x));
	checkCuda(cudaFree(d_y));
	checkCuda(cudaFree(d_z));

}

void testLeftShift()
{
	printf("\ntesting left shifting...");

	Integer4K *x = fromHexString("1243abc312def391acd89897ad987f789868091243abc312def391acd8989"
								  "7ad987f75436486afcdf232f6f4ffd9012340198354abcdef");
		
	leftShift(x, 492);	

	//printf(toHexString(x));

	int ok = strcmp(toHexString(x), "1243abc312def391acd89897ad987f789868091243abc312def391acd"
									"89897ad987f75436486afcdf232f6f4ffd9012340198354abcdef00000"
									"00000000000000000000000000000000000000000000000000000000000"
									"00000000000000000000000000000000000000000000000000000000000");

	if (ok == 0)
		printf("ok!");
	else
		printf("not ok!");

	free(x);
}

void testMultiply()
{
	printf("\ntesting multipling...");

	Integer4K *x = fromHexString("2");
	Integer4K *y = fromHexString("2");
	Integer4K *z = (Integer4K*)malloc(sizeof(Integer4K));

	Integer4K *d_x, *d_y, *d_z;
	checkCuda(cudaMalloc(&d_z, sizeof(Integer4K)));
	checkCuda(cudaMalloc(&d_x, sizeof(Integer4K)));
	checkCuda(cudaMalloc(&d_y, sizeof(Integer4K)));

	checkCuda(cudaMemcpy(d_x, x, sizeof(Integer4K), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_y, y, sizeof(Integer4K), cudaMemcpyHostToDevice));

	multiply << <1, 1 >> >(d_z, d_x, d_y);

	checkCuda(cudaMemcpy(z, d_z, sizeof(Integer4K), cudaMemcpyDeviceToHost));

	printf("\ngot: %s", toHexString(z)); 

	char* result = "9bbb";

	printf("\nis:  %s\n", result);

	int ok = strcmp(toHexString(z), result);


	if (ok == 0)
		printf("ok!");
	else
		printf("not ok!");

	free(x);
	free(y);
	free(z);

	checkCuda(cudaFree(d_x));
	checkCuda(cudaFree(d_y));
	checkCuda(cudaFree(d_z));

}

int main()
{	
	checkCuda(cudaDeviceReset());
	clock_t start = clock();
	
	testAdding();
	testSubtract();
	testLeftShift();
	testMultiply();

	checkCuda(cudaDeviceReset());
	checkCuda(cudaDeviceSynchronize()); //Blocks the CPU until all preceding CUDA calls have completed
	clock_t end = clock();	
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("\n\nTOTAL EXE TIME: %f\n", seconds);
	
    return 0;

}
