@echo off
setlocal

echo [1/4] Reconstructing zip file...
copy /b superconducting-env-split.zip.001 + superconducting-env-split.zip.002 + superconducting-env-split.zip.003 + superconducting-env-split.zip.004 superconducting-env.zip

echo [2/4] Extracting conda environment using 7-Zip...

:: Try common 7-Zip paths
set ZIP="C:\Program Files\7-Zip\7z.exe"
if not exist %ZIP% (
    set ZIP="C:\Program Files (x86)\7-Zip\7z.exe"
)
if not exist %ZIP% (
    echo [ERROR] 7-Zip not found. Please install from https://www.7-zip.org/
    pause
    exit /b
)

:: Extract to "env" folder
%ZIP% x superconducting-env.zip -oenv -y

echo [3/4] Running script...

:: Replace with your actual entry point
call env\python.exe your_script.py

echo [4/4] Done.
pause