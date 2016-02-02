// cuda libs -------------------------------------
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// -----------------------------------------------
// c libs ----------------------------------------
#include <string.h>
#include <stdint.h>
// -----------------------------------------------
// statics ---------------------------------------
#define ARRAY_SIZE 128
//#define NUMBER_OF_REGISTERS_PER_THREAD 32
// -----------------------------------------------
// data model ------------------------------------
typedef uint32_t Int32;

typedef struct
{	
	// the magnitude array in little endian order
	// most-significant int is mag[length-1]
	// least-significant int is mag[0]
	Int32 mag[ARRAY_SIZE];

	// actual magnitude array's lenght
	int length;
	
} Integer4K;
// -----------------------------------------------
// functions prototypes---------------------------
__host__ Int32 parseInt32(const char* hexString);
__device__ void deviceClearInteger4k(Integer4K* integer);
__host__ void hostClearInteger4k(Integer4K* integer);
// -----------------------------------------------
// device implementations ------------------------
__global__ void add(Integer4K* device_result, Integer4K* device_x, Integer4K* device_y) 
{

	asm volatile("\n\t"
			"add.cc.u32 %0, %1, %2; \n\t" :
			 "=r"(device_result->mag[0]) :
			 "r"(device_x->mag[0]),
			 "r"(device_y->mag[0])); 

	for (int i = 1; i < ARRAY_SIZE - 1; i++)
	{
		asm volatile("\n\t"
			"addc.cc.u32 %0, %1, %2; \n\t" :
			 "=r"(device_result->mag[i]) :
			 "r"(device_x->mag[i]),
			 "r"(device_y->mag[i])); 
	}

	asm volatile("\n\t"
			"addc.u32 %0, %1, %2; \n\t" :
			 "=r"(device_result->mag[ARRAY_SIZE - 1]) :
			 "r"(device_x->mag[ARRAY_SIZE - 1]),
			 "r"(device_y->mag[ARRAY_SIZE - 1]));

	int i = ARRAY_SIZE - 1;
	for (; i >= 0; i--)
	{
		if (device_result->mag[i] != 0UL)
			break;
	}
	device_result->length = i + 1;

}

__global__ void subtract(Integer4K* device_result, Integer4K* device_x, Integer4K* device_y) 
{

	asm volatile("\n\t"
			"sub.cc.u32 %0, %1, %2; \n\t" :
			 "=r"(device_result->mag[0]) :
			 "r"(device_x->mag[0]),
			 "r"(device_y->mag[0])); 

	for (int i = 1; i < ARRAY_SIZE - 1; i++)
	{
		asm volatile("\n\t"
			"subc.cc.u32 %0, %1, %2; \n\t" :
			 "=r"(device_result->mag[i]) :
			 "r"(device_x->mag[i]),
			 "r"(device_y->mag[i])); 
	}

	asm volatile("\n\t"
			"subc.u32 %0, %1, %2; \n\t" :
			 "=r"(device_result->mag[ARRAY_SIZE - 1]) :
			 "r"(device_x->mag[ARRAY_SIZE - 1]),
			 "r"(device_y->mag[ARRAY_SIZE - 1]));

	int i = ARRAY_SIZE - 1;
	for (; i >= 0; i--)
	{
		if (device_result->mag[i] != 0UL)
			break;
	}
	device_result->length = i + 1;

}

__device__ void deviceClearInteger4k(Integer4K* integer)
{
	for (int i = 0; i < ARRAY_SIZE; i++)
		integer->mag[i] ^= integer->mag[i];	
}
// -----------------------------------------------
// host implementations --------------------------
__host__ void hostClearInteger4k(Integer4K* integer)
{
	for (int i = 0; i < ARRAY_SIZE; i++)
		integer->mag[i] ^= integer->mag[i];	
}

__host__ void leftShift(Integer4K* integer, int n)
{	
	if (n == 0) 
		return;

	int ints = n >> 5;
    int bits = n & 0x1f;
	int newLength = integer->length + ints;

	if(newLength >= ARRAY_SIZE)
	{
		// overflow
		return;
	}

	for (int i = newLength - 1; i >= ints; i--)		
		integer->mag[i] = integer->mag[i-ints];		
		
	for (int i = 0; i < ints; i++)
		integer->mag[i] = 0UL;
		
	if (bits != 0)
	{		
		newLength++;
        int remainingBits = 32 - bits;
        int highBits;
		int lowBits = 0;

		for (int i = ints; i < newLength; i++)
		{
			highBits = integer->mag[i] >> remainingBits;
			integer->mag[i] = integer->mag[i] << bits | lowBits;
			lowBits = highBits;
		}			
	}	

	integer->length = newLength;
}

__host__ Integer4K* fromHexString(const char* string)
{
	Integer4K* integer = (Integer4K*) malloc(sizeof(Integer4K));

	//clear
	hostClearInteger4k(integer);

	int length = strlen(string);

	char temp[9];

	int i = length - 8;
	int j = 0;
	for (; i >= 0; i -= 8)
	{
		strncpy(temp, string + i, 8);
		temp[8] = '\0';
		integer->mag[j++] = parseInt32(temp);
	}

	if (i < 0)
	{
		int index = 8 + i;
		char* temp = (char*) malloc(index + 1);
		strncpy(temp, string, 8 + i);
		temp[index] = '\0';
		Int32 value = parseInt32(temp);
		integer->mag[j] = value;
		if (value > 0UL) j++;
	}
	integer->length = j;
	
	return integer;
}

__host__ char* toHexString(const Integer4K* number)
{	
	char* buffer = (char*) malloc(ARRAY_SIZE * 8 + 1);
		
	for (int i = ARRAY_SIZE - 1, j=0; i >= 0; i--)
	{		
		sprintf(buffer + 8 * j++, "%08lx", number->mag[i]);	
	}

	// clear leading zeros
	int i = 0;
	for (; i < ARRAY_SIZE * 8 - 1; i++)
		if (buffer[i] != '0')
			break;
	
	return buffer + i;	
}


/*
Parses hex string to Int32 type.
Accepts both upper and lower case, no "0x" at the beginning.
E.g.: 314Da43F
*/
__host__ Int32 parseInt32(const char* hexString)
{	
	
	int length = strlen(hexString);
	
	if ( length < 8) 
	{
		char padding[9];			
		
		int zeros = 8 - length;
		for (int i = 0, j = 0; i < 8; i++) 
		{
			if (i < zeros)
				padding[i] = '0';
			else
				padding[i] = hexString[j++];
		}
		
		padding[8] = '\0'; 
		hexString = padding;		
	}

	Int32 chunk = 0UL;			
	sscanf(hexString, "%x", &chunk);	
	return chunk;	
}