@echo off

REM set your nvcc path
set NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\bin\nvcc.exe
set C_LIB=C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\lib
set C_INCLUDE=C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\include
set CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\lib
set CUDA_INCLUDE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\include

 setlocal

 if "%NVCC%" == "" goto warning 
 if "%C_LIB%" == "" goto warning 
 if "%C_INCLUDE%" == "" goto warning 
 if "%CUDA_LIB%" == "" goto warning 
 if "%CUDA_INCLUDE%" == "" goto warning 

:: Help

 if "%1" == "help" goto help
 if "%1" == "-help" goto help
 if "%1" == "--help" goto help
 
:: Build 

 if "%1" == "build" goto build
 if "%1" == "-build" goto build
 if "%1" == "--build" goto build
 
 :: RUN 

 if "%1" == "run" goto run
 if "%1" == "-run" goto run
 if "%1" == "--run" goto run
 
 :clean
 echo ::CLEAN::
 rmdir /s/q bin
 mkdir bin
 
 if "%1" == "clean" goto quit
 if "%1" == "-clean" goto quit
 if "%1" == "--clean" goto quit
 
 :build
 echo ::BUILD::  
 cd bin
 call "%NVCC%" -G --machine 64 ../src/main.cu -o main.exe
 cd ..
 
  if "%1" == "build" goto quit
  if "%1" == "-build" goto quit
  if "%1" == "--build" goto quit

 :run
 echo ::RUN::
 if not exist bin\main.exe (echo ERROR: Build First) ELSE call bin\main.exe
 goto quit 
  
 :warning
 echo Set environment variables
 goto quit
 
 :help
 echo Usage: build.bat [options]
 echo Where options include:
 echo        no options = clean build run
 echo        -help     print out this message
 echo        -build    build
 echo        -clean    clean bin folder
 echo        -run      run
 goto quit
  
:quit
 endlocal

