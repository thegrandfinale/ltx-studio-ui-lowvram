@echo off
cd /d "%~dp0"

echo ======================================================
echo   LTX Studio — Text-to-Video
echo   LTX-2.3 22B Distilled GGUF Q4_K_M
echo ======================================================
echo.

REM ── Check installation ─────────────────────────────────
if not exist "ComfyUI\.venv\Scripts\python.exe" (
    echo ERROR: Not installed yet. Run install.bat first.
    pause
    exit /b 1
)

REM ── Kill any previous instances ────────────────────────
taskkill /f /fi "WINDOWTITLE eq LTX-ComfyUI" >nul 2>&1

REM ── Start ComfyUI in its own window ────────────────────
echo [1/2] Starting ComfyUI (port 8188)...
start "LTX-ComfyUI" ComfyUI\.venv\Scripts\python.exe ComfyUI\main.py --listen 127.0.0.1 --port 8188 --lowvram

REM ── Wait for ComfyUI to be ready ───────────────────────
echo       Waiting for ComfyUI to load (this may take 30-60 seconds)...
:wait_loop
timeout /t 5 /nobreak >nul
ComfyUI\.venv\Scripts\python.exe -c "import httpx; r = httpx.get('http://127.0.0.1:8188/system_stats', timeout=3); exit(0 if r.status_code == 200 else 1)" >nul 2>&1
if errorlevel 1 goto wait_loop
echo       ComfyUI ready.
echo.

REM ── Start FastAPI wrapper (port 8000) ──────────────────
echo [2/2] Starting LTX Studio UI (port 8000)...
echo.
echo   ======================================
echo     Open http://localhost:8000
echo   ======================================
echo.
ComfyUI\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
