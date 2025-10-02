@echo off
setlocal enabledelayedexpansion

:: 基础配置
set CC=gcc
set OUT_CPU=test_gpt2_cpu.exe
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
set CPU_SRCS=%SRC_DIR%\test_gpt2.c
set CPU_OBJS=%BUILD_DIR%\test_gpt2.o

:: 编译主源文件
%CC% -g -O3 %OPENMP_FLAG% -I%LLMC_DIR% -I%DEV_DIR% -c %CPU_SRCS% -o %CPU_OBJS%
if errorlevel 1 goto :error

:: 链接CPU版本
%CC% -O3 %OPENMP_FLAG% %CPU_OBJS% -lm -lws2_32 -o %OUT_CPU%
if errorlevel 1 goto :error
echo CPU build successful: %OUT_CPU%


echo.
echo All builds completed successfully!
exit /b 0

:error
echo Build failed!
exit /b 1