#include "DeviceWrapper.h"
#include <iostream>  

int main()
{

	long long time = 0, newTime = 0;
		
	while (true)
	{
		newTime = DeviceWrapper::getClock();
		std::cout << newTime - time << std::endl;
		time = newTime;
	}
    return 0;
}
