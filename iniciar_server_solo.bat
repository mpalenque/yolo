@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"

cd /d "%PROJECT_DIR%"

if not exist "%PYTHON_EXE%" (
    echo No encontre el entorno virtual en "%PYTHON_EXE%"
    pause
    exit /b 1
)

echo Iniciando server.py...
"%PYTHON_EXE%" server.py

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo server.py termino con codigo %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
