@echo off
rem Wrapper for build_exe.ps1 in this folder. Every argument is passed on, e.g.: build_exe.bat -SkipInstall
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_exe.ps1" %*
exit /b %ERRORLEVEL%
