@echo off
setlocal enabledelayedexpansion

:: 基础配置
set CC=gcc
set OUT_CPU=train_gpt2_cpu.exe
set OUT_CUDA=train_gpt2_cuda.exe
set SRC_DIR=.
set DEV_DIR=dev
set LLMC_DIR=llmc
set BUILD_DIR=.

:: 创建构建目录
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

:: 检测OpenMP
set OPENMP_FLAG=
echo Detecting OpenMP...
%CC% -dM -E - <nul 2>nul | findstr "_OPENMP" >nul
if not errorlevel 1 (
    set OPENMP_FLAG=-fopenmp -DOMP
    echo OpenMP enabled
) else (
    echo OpenMP not found, building single-threaded
)

:: 检测CUDA
set CUDA_ENABLED=0
where nvcc >nul 2>nul
if not errorlevel 1 (
    set CUDA_ENABLED=1
    echo CUDA toolkit found
) else (
    echo CUDA not found, skipping GPU builds
)

:: 编译CPU版本
echo.
echo Building CPU version...
set CPU_SRCS=%SRC_DIR%\train_gpt2.c
set CPU_OBJS=%BUILD_DIR%\train_gpt2.o

:: 编译主源文件
%CC% -g -O3 %OPENMP_FLAG% -I%LLMC_DIR% -I%DEV_DIR% -c %CPU_SRCS% -o %CPU_OBJS%
if errorlevel 1 goto :error

:: 链接CPU版本
%CC% -O3 %OPENMP_FLAG% %CPU_OBJS% -lm -lws2_32 -o %OUT_CPU%
if errorlevel 1 goto :error
echo CPU build successful: %OUT_CPU%

:: 编译CUDA版本（如果检测到）
if %CUDA_ENABLED%==1 (
    echo.
    echo Building CUDA version...
    set CUDA_SRCS=%SRC_DIR%\train_gpt2.cu
    set CUDA_OBJS=%BUILD_DIR%\train_gpt2_cuda.o
    
    :: 编译CUDA源文件
    nvcc -O3 --use_fast_math -std=c++17 -c %CUDA_SRCS% -o %CUDA_OBJS%
    if errorlevel 1 goto :error
    
    :: 链接CUDA版本
    nvcc -O3 %CUDA_OBJS% -lcublas -lcublasLt -o %OUT_CUDA%
    if errorlevel 1 goto :error
    echo CUDA build successful: %OUT_CUDA%
)

:: 编译测试程序
echo.
echo Building test programs...
for %%f in (%DEV_DIR%\test\test_*.c) do (
    set TEST_NAME=%%~nf
    set TEST_SRC=%%f
    set TEST_OUT=%BUILD_DIR%\!TEST_NAME!.exe
    
    echo Building !TEST_NAME!...
    %CC% -O3 -I%LLMC_DIR% -I%DEV_DIR% !TEST_SRC! -lm -lws2_32 -o !TEST_OUT!
    if errorlevel 1 (
        echo Warning: !TEST_NAME! build failed
    ) else (
        echo !TEST_NAME! built successfully
    )
)

echo.
echo All builds completed successfully!
exit /b 0

:error
echo Build failed!
exit /b 1