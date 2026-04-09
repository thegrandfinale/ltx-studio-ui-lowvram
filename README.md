# LTX Studio

Text-to-Video generation powered by [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (22B parameters).
Runs on consumer GPUs with 8 GB VRAM via GGUF quantization.

## Requirements

- **OS**: Windows 10/11
- **GPU**: NVIDIA with 8+ GB VRAM (tested on RTX 4070)
- **Disk**: ~25 GB free space (for models)
- **Software**: [Python 3.10+](https://python.org), [Git](https://git-scm.com), [CUDA 12.x](https://developer.nvidia.com/cuda-downloads)

## Installation

```
git clone https://github.com/YOUR_USER/ltx-studio.git
cd ltx-studio
install.bat
```

The installer will:
1. Clone ComfyUI
2. Create a Python virtual environment
3. Install all dependencies (PyTorch, ComfyUI, custom nodes)
4. Download models from HuggingFace (~20 GB)

First install takes 15-30 minutes depending on your connection.

## Usage

```
start.bat
```

Then open **http://localhost:8000** in your browser.

## How it works

```
Browser (UI)  -->  FastAPI wrapper (:8000)  -->  ComfyUI API (:8188)
                                                      |
                                              T2V workflow (GGUF)
                                                      |
                                                  Video output
```

1. The web UI sends your prompt to the FastAPI wrapper
2. The wrapper injects parameters into the optimized ComfyUI workflow
3. ComfyUI runs the two-stage pipeline (generate at half-res, upscale 2x)
4. The generated video is returned to the browser

## Project structure

```
ltx-studio/
├── install.bat              <- one-click installer
├── start.bat                <- launcher
├── requirements.txt         <- FastAPI wrapper dependencies
├── app/
│   ├── main.py              <- FastAPI wrapper (T2V only)
│   ├── workflows/
│   │   └── text_to_video.json   <- optimized GGUF workflow
│   └── templates/
│       └── index.html       <- web UI
├── scripts/
│   └── download_models.py   <- model downloader
└── ComfyUI/                 <- installed by install.bat (gitignored)
    ├── models/              <- ~20 GB of model files
    └── custom_nodes/        <- ComfyUI-GGUF, ComfyUI-LTXVideo, comfyui-kjnodes
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | ComfyUI + model status |
| POST | `/generate` | Start T2V generation |
| GET | `/status/{id}` | Poll job status |
| GET | `/outputs/{file}` | Download result |

### POST /generate

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | string | required | Text description of the video |
| negative_prompt | string | "blurry, low quality..." | What to avoid |
| width | int | 768 | Output width (px) |
| height | int | 512 | Output height (px) |
| num_frames | int | 97 | Number of frames |
| frame_rate | float | 24.0 | FPS |
| cfg | float | 1.0 | Guidance scale |
| seed | int | -1 | Random seed (-1 = random) |

## VRAM tips for 8 GB GPUs

| Setting | Recommended |
|---------|-------------|
| Resolution | 768x512 or 900x512 |
| Duration | 4-8 seconds |
| ComfyUI flag | `--lowvram` (set in start.bat) |

## License

Model weights are subject to [Lightricks LTX-Video license](https://huggingface.co/Lightricks/LTX-2.3).
