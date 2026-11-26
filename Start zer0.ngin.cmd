@echo off
setlocal
title zer0.ngin - Inkling 2 Mini (Zig/CUDA)
echo zer0.ngin, Inkling 2 Mini: http://127.0.0.1:8001/   (API: http://127.0.0.1:8001/v1)
echo Ladowanie trwa ok. 20 s. Ctrl+C zatrzymuje serwer.
echo.
"%~dp0bin\zer0.ngin.exe" -m "%LOCALAPPDATA%\Programs\llama.cpp-Inkling\models\Inkling-2-Mini-UD-Q4_K_M.gguf" --port 8001 -c 32768
if not "%ERRORLEVEL%"=="0" pause
