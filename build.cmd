@echo off
rem Buduje bin\zer0.ngin.exe. Wymaga tylko Zig 0.16.0; kernele CUDA kompiluje NVRTC przy pierwszym starcie.
setlocal
set "ZIG=%~1"
if "%ZIG%"=="" set "ZIG=C:\src\tools\zig-x86_64-windows-0.16.0\zig.exe"
cd /d "%~dp0src"
"%ZIG%" build-exe main.zig -O ReleaseFast -mcpu=znver5 -femit-bin=..\bin\zer0.ngin.exe
