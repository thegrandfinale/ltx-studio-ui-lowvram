@echo off
setlocal enabledelayedexpansion

echo ======================================================
echo   LTX Studio — Installer
echo   Text-to-Video with LTX-2.3 22B (GGUF Q4_K_M)
echo ======================================================
echo.

cd /d "%~dp0"

REM ── Check Python ───────────────────────────────────────
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo       Found Python %PYVER%
echo.

REM ── Check Git ──────────────────────────────────────────
echo [2/6] Checking Git...
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git not found. Install Git from https://git-scm.com
    pause
    exit /b 1
)
echo       Git OK
echo.

REM ── Clone ComfyUI ──────────────────────────────────────
echo [3/6] Setting up ComfyUI...
if exist "ComfyUI\main.py" (
    echo       ComfyUI already present, skipping clone
) else (
    echo       Cloning ComfyUI...
    git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI
    if errorlevel 1 (
        echo ERROR: Failed to clone ComfyUI
        pause
        exit /b 1
    )
)
echo.

REM ── Create venv + install dependencies ─────────────────
echo [4/6] Creating Python environment and installing dependencies...
echo       This may take 10-15 minutes on first run.
echo.

if not exist "ComfyUI\.venv\Scripts\python.exe" (
    echo       Creating virtual environment...
    python -m venv ComfyUI\.venv
)

REM Install PyTorch with CUDA 12.8
echo       Installing PyTorch (CUDA 12.8)...
ComfyUI\.venv\Scripts\pip.exe install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

REM Install ComfyUI requirements
echo       Installing ComfyUI dependencies...
ComfyUI\.venv\Scripts\pip.exe install --quiet -r ComfyUI\requirements.txt

REM Install wrapper dependencies
echo       Installing LTX Studio dependencies...
ComfyUI\.venv\Scripts\pip.exe install --quiet -r requirements.txt

echo       Dependencies installed.
echo.

REM ── Clone custom nodes ─────────────────────────────────
echo [5/6] Installing custom nodes...

if not exist "ComfyUI\custom_nodes" mkdir "ComfyUI\custom_nodes"

REM ComfyUI-GGUF (GGUF model loader)
if exist "ComfyUI\custom_nodes\ComfyUI-GGUF" (
    echo       ComfyUI-GGUF already installed
) else (
    echo       Cloning ComfyUI-GGUF...
    git clone https://github.com/city96/ComfyUI-GGUF.git ComfyUI\custom_nodes\ComfyUI-GGUF
    ComfyUI\.venv\Scripts\pip.exe install --quiet -r ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt 2>nul
)

REM ComfyUI-LTXVideo (LTX video nodes)
if exist "ComfyUI\custom_nodes\ComfyUI-LTXVideo" (
    echo       ComfyUI-LTXVideo already installed
) else (
    echo       Cloning ComfyUI-LTXVideo...
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git ComfyUI\custom_nodes\ComfyUI-LTXVideo
    ComfyUI\.venv\Scripts\pip.exe install --quiet -r ComfyUI\custom_nodes\ComfyUI-LTXVideo\requirements.txt 2>nul
)

REM comfyui-kjnodes (VAELoaderKJ, SageAttention, ChunkFeedForward)
if exist "ComfyUI\custom_nodes\comfyui-kjnodes" (
    echo       comfyui-kjnodes already installed
) else (
    echo       Cloning comfyui-kjnodes...
    git clone https://github.com/kijai/ComfyUI-KJNodes.git ComfyUI\custom_nodes\comfyui-kjnodes
    ComfyUI\.venv\Scripts\pip.exe install --quiet -r ComfyUI\custom_nodes\comfyui-kjnodes\requirements.txt 2>nul
)

echo       Custom nodes ready.
echo.

REM ── Download models ────────────────────────────────────
echo [6/6] Downloading models (~20 GB)...
echo       If this is your first run, this will take a while.
echo.
ComfyUI\.venv\Scripts\python.exe scripts\download_models.py
echo.

REM ── Done ───────────────────────────────────────────────
echo ======================================================
echo   Installation complete!
echo.
echo   To start LTX Studio, run: start.bat
echo   Then open http://localhost:8000 in your browser.
echo ======================================================
pause
