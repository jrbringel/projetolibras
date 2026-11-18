@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY="
if exist "%~dp0runtime\python.exe" set "PY=%~dp0runtime\python.exe"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"

if not defined PY (
  echo.
  echo  Nao achei o Python desta pasta.
  echo  Neste computador de desenvolvimento: use a pasta .venv
  echo  No pendrive: rode primeiro preparar_pendrive.ps1
  echo.
  pause
  exit /b 1
)

echo.
echo   IVY LIBRAS
echo   Nao feche esta janela durante a apresentacao.
echo   O navegador abre sozinho quando o servidor estiver pronto.
echo.

start "ivy-browser" /min powershell -NoProfile -WindowStyle Hidden -Command "foreach ($i in 1..45) { Start-Sleep -Seconds 1; try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7860/api/health -TimeoutSec 1 | Out-Null; Start-Process 'http://127.0.0.1:7860'; break } catch {} }"

set "PATH=%~dp0runtime;%~dp0runtime\DLLs;%~dp0runtime\Scripts;%PATH%"
"%PY%" -u app.py
if errorlevel 1 (
  echo.
  echo  O servidor encerrou com erro. Leia a mensagem acima.
  pause
)
