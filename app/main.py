"""
LTX Studio — FastAPI wrapper over ComfyUI
LTX-2.3 22B Distilled (GGUF Q4_K_M) — Two-stage T2V pipeline with latent upscaling
Optimised for 8 GB VRAM via GGUF + FP4 text encoder + ChunkFeedForward
"""
import json
import uuid
import asyncio
import logging
import random
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
OUTPUT_DIR = APP_DIR / "outputs"
TEMPLATE_DIR = APP_DIR / "templates"
WORKFLOW_DIR = APP_DIR / "workflows"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── ComfyUI connection ───────────────────────────────────────────────────────
COMFYUI_URL = "http://127.0.0.1:8188"

# ── Workflow ─────────────────────────────────────────────────────────────────
WORKFLOW_PATH = WORKFLOW_DIR / "text_to_video.json"

# Default ManualSigmas schedules (optimized for distilled model)
DEFAULT_SIGMAS_STAGE1 = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
DEFAULT_SIGMAS_STAGE2 = "0.85, 0.7250, 0.4219, 0.0"

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="LTX Studio", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Job tracker
jobs: dict[str, dict] = {}


def load_workflow(**kwargs) -> dict:
    """Load the T2V workflow template and fill in variables."""
    if not WORKFLOW_PATH.exists():
        raise ValueError(f"Workflow not found: {WORKFLOW_PATH}")

    raw = WORKFLOW_PATH.read_text(encoding="utf-8")

    # Two-stage pipeline: Stage 1 runs at half resolution, Stage 2 upscales 2x
    width = int(kwargs.get("width", 768))
    height = int(kwargs.get("height", 512))
    half_width = (width // 2 // 32) * 32   # Must be divisible by 32
    half_height = (height // 2 // 32) * 32

    # Replace template variables
    replacements = {
        "{{PROMPT}}":          str(kwargs.get("prompt", "")),
        "{{NEGATIVE_PROMPT}}": str(kwargs.get("negative_prompt", "")),
        "{{WIDTH}}":           str(width),
        "{{HEIGHT}}":          str(height),
        "{{HALF_WIDTH}}":      str(half_width),
        "{{HALF_HEIGHT}}":     str(half_height),
        "{{NUM_FRAMES}}":      str(kwargs.get("num_frames", 97)),
        "{{SEED}}":            str(kwargs.get("seed", 42)),
        "{{CFG}}":             str(kwargs.get("cfg", 1.0)),
        "{{FRAME_RATE}}":      str(kwargs.get("frame_rate", 24.0)),
        "{{SIGMAS_STAGE1}}":   str(kwargs.get("sigmas_stage1", DEFAULT_SIGMAS_STAGE1)),
        "{{SIGMAS_STAGE2}}":   str(kwargs.get("sigmas_stage2", DEFAULT_SIGMAS_STAGE2)),
    }

    for key, val in replacements.items():
        raw = raw.replace(key, val)

    # Parse and fix types (JSON template has strings for numbers)
    workflow = json.loads(raw)

    # Convert numeric strings back to numbers in input fields
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        for k, v in inputs.items():
            if isinstance(v, str) and v.replace(".", "").replace("-", "").isdigit():
                inputs[k] = float(v) if "." in v else int(v)

    return workflow


async def _run_comfyui_generation(job_id: str, params: dict):
    """Send the T2V workflow to ComfyUI and poll for completion."""
    try:
        jobs[job_id]["status"] = "generating"

        workflow = load_workflow(**params)

        async with httpx.AsyncClient(timeout=600) as client:
            # Submit workflow
            resp = await client.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": workflow},
            )
            if resp.status_code != 200:
                raise RuntimeError(f"ComfyUI rejected workflow: {resp.text}")

            comfy_data = resp.json()
            prompt_id = comfy_data["prompt_id"]
            logger.info(f"Job {job_id} [T2V]: ComfyUI prompt_id={prompt_id}")

            # Poll for completion
            while True:
                await asyncio.sleep(2)
                hist_resp = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
                if hist_resp.status_code != 200:
                    continue

                history = hist_resp.json()
                if prompt_id not in history:
                    continue

                # Check for errors
                status_info = history[prompt_id].get("status", {})
                if status_info.get("status_str") == "error":
                    msgs = status_info.get("messages", [])
                    raise RuntimeError(f"ComfyUI error: {msgs}")

                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_output in outputs.items():
                    for key in ("videos", "images", "gifs"):
                        if key not in node_output:
                            continue
                        for item in node_output[key]:
                            filename = item["filename"]
                            subfolder = item.get("subfolder", "")
                            img_type = item.get("type", "output")

                            view_resp = await client.get(
                                f"{COMFYUI_URL}/view",
                                params={
                                    "filename": filename,
                                    "subfolder": subfolder,
                                    "type": img_type,
                                },
                            )
                            if view_resp.status_code == 200:
                                ext = Path(filename).suffix or ".mp4"
                                out_path = OUTPUT_DIR / f"{job_id}{ext}"
                                out_path.write_bytes(view_resp.content)
                                jobs[job_id].update({
                                    "status": "done",
                                    "result": f"/outputs/{job_id}{ext}",
                                })
                                logger.info(f"Job {job_id} complete -> {out_path}")
                                return

                if status_info.get("completed"):
                    break

            jobs[job_id].update({"status": "error", "error": "No output found in ComfyUI response"})

    except Exception as exc:
        logger.exception(f"Job {job_id} failed")
        jobs[job_id].update({"status": "error", "error": str(exc)})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(TEMPLATE_DIR / "index.html")


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{COMFYUI_URL}/system_stats")
            comfy_stats = resp.json() if resp.status_code == 200 else None
    except Exception:
        comfy_stats = None

    return {
        "status": "ok" if comfy_stats else "comfyui_offline",
        "comfyui": comfy_stats,
        "model": "LTX-2.3 22B Distilled GGUF Q4_K_M — Two-Stage T2V",
    }


@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    prompt:           str   = Form(...),
    negative_prompt:  str   = Form("blurry, low quality, still frame, watermark, overlay, titles"),
    width:            int   = Form(768),
    height:           int   = Form(512),
    num_frames:       int   = Form(97),
    frame_rate:       float = Form(24.0),
    cfg:              float = Form(1.0),
    seed:             int   = Form(-1),
):
    """Generate a video from text prompt (two-stage T2V pipeline)."""
    job_id = str(uuid.uuid4())[:8]
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    jobs[job_id] = {"status": "queued", "result": None}

    params = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        cfg=cfg,
        seed=actual_seed,
    )

    background_tasks.add_task(_run_comfyui_generation, job_id, params)
    return JSONResponse({"job_id": job_id, "seed": actual_seed})


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JSONResponse(job)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
