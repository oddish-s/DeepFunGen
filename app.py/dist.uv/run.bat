@echo off
setlocal

set "UV_CMD=uv"
where %UV_CMD% >nul 2>nul
if %errorlevel% neq 0 (
  set "UV_CMD=%~dp0\bin\uv.exe"
)

"%UV_CMD%" sync
if %errorlevel% neq 0 exit /b %errorlevel%

"%UV_CMD%" run main.py %*
exit /b %errorlevel%
