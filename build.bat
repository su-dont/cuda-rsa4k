@echo off

REM set yours enviroment
set NVCCBIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin
set CCBIN=C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin
set CSC=C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe

set nvcc=%NVCCBIN%\nvcc.exe

set flags=-arch=sm_30 -m 32
set objects=device.obj BigInteger.obj Test.obj main.obj

cls  
echo ::CLEAN::
if exist "rsa.exe" del app.exe
cd cuda-rsa4k
del /Q bin\*
rd bin
md bin
echo ::BUILD::
cd bin
"%CSC%" /out:WindowsBatchConverter.exe .\..\WindowsBatchConverter.cs
"%nvcc%" %flags% --keep -v -dc .\..\DeviceWrapper.cu -o device.obj > device_warpper_run
REM or edit
copy .\..\DeviceWrapper.ptx .\DeviceWrapper.ptx
call .\WindowsBatchConverter.exe device_warpper_run > .\windows_device_build.bat
call .\windows_device_build.bat
"%nvcc%" %flags% -I.\..\ -ccbin "%CCBIN%" -dc .\..\BigInteger.cpp -o .\BigInteger.obj
"%nvcc%" %flags% -I.\..\ -ccbin "%CCBIN%" -dc .\..\Test.cpp -o .\Test.obj
"%nvcc%" %flags% -I.\..\ -ccbin "%CCBIN%" -dc .\..\main.cpp -o .\main.obj
echo ::LINK::
"%nvcc%" %flags% -ccbin "%CCBIN%" %objects% -o rsa.exe
copy .\rsa.exe ..\..\rsa.exe
cd ..\..
echo ::RUN::
rsa.exe
echo ::QUIT::
