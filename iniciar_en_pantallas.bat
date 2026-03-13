@echo off
setlocal
set "PROJECT_DIR=C:\Users\001\Desktop\YOLO\yolo-main"
start "" /min powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%PROJECT_DIR%\iniciar_en_pantallas.ps1"
endlocal