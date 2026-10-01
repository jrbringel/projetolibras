@echo off
cd /d "%~dp0"
title IVY LIBRAS - acesso remoto
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0INICIAR_ONLINE.ps1"
echo.
echo Tunel encerrado.
pause
