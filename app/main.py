"""
LTX Studio — FastAPI wrapper over ComfyUI
LTX-2.3 22B Distilled (GGUF Q4_K_M) — Two-stage T2V pipeline with latent upscaling
Flux.2 Klein 4B (GGUF) — Text-to-Image + Image Edit pipeline
Optimised for 8 GB VRAM via GGUF + FP4 text encoder + ChunkFeedForward
"""
import json
import uuid
import asyncio
import logging
import random
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, BackgroundTasks, UploadFile
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

# ── Video Workflow ───────────────────────────────────────────────────────────
WORKFLOW_T2V_PATH = WORKFLOW_DIR / "text_to_video.json"
WORKFLOW_T2V_NOAUDIO_PATH = WORKFLOW_DIR / "text_to_video_noaudio.json"

# Default ManualSigmas schedules (optimized for distilled model)
DEFAULT_SIGMAS_STAGE1 = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
DEFAULT_SIGMAS_STAGE2 = "0.85, 0.7250, 0.4219, 0.0"

# ── Image Workflows ──────────────────────────────────────────────────────────
WORKFLOW_T2I_PATH = WORKFLOW_DIR / "text_to_image.json"
WORKFLOW_EDIT_PATH = WORKFLOW_DIR / "image_edit.json"
WORKFLOW_REF1_PATH = WORKFLOW_DIR / "text_to_image_ref1.json"
WORKFLOW_REF2_PATH = WORKFLOW_DIR / "text_to_image_ref2.json"

# Flux Klein model presets: name → (gguf_file, clip_file)
FLUX_PRESETS = {
    "klein-4b": {
        "model": "flux-2-klein-4b-Q4_K_M.gguf",
        "clip":  "qwen_3_4b.safetensors",
    },
    "klein-9b": {
        "model": "flux-2-klein-9b-Q4_K_M.gguf",
        "clip":  "qwen_3_8b_fp4mixed.safetensors",
    },
}
DEFAULT_FLUX_PRESET = "klein-4b"
DEFAULT_FLUX_VAE = "flux2-vae.safetensors"

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="LTX Studio", version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Job tracker
jobs: dict[str, dict] = {}


# ── Workflow loaders ──────────────────────────────────────────────────────────

def load_video_workflow(**kwargs) -> dict:
    """Load the T2V workflow template and fill in variables."""
    enable_audio = kwargs.get("enable_audio", False)
    wf_path = WORKFLOW_T2V_PATH if enable_audio else WORKFLOW_T2V_NOAUDIO_PATH
    if not wf_path.exists():
        raise ValueError(f"Workflow not found: {wf_path}")

    raw = wf_path.read_text(encoding="utf-8")
    logger.info(f"Using workflow: {wf_path.name} (audio={'on' if enable_audio else 'off'})")

    width = int(kwargs.get("width", 768))
    height = int(kwargs.get("height", 512))
    half_width = (width // 2 // 32) * 32
    half_height = (height // 2 // 32) * 32

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

    workflow = json.loads(raw)
    _fix_numeric_strings(workflow)
    return workflow


def _resolve_flux_preset(preset_name: str) -> dict:
    """Resolve a preset name to model/clip filenames."""
    preset = FLUX_PRESETS.get(preset_name, FLUX_PRESETS[DEFAULT_FLUX_PRESET])
    return preset["model"], preset["clip"]


def load_image_workflow(**kwargs) -> dict:
    """Load the T2I workflow template and fill in variables."""
    if not WORKFLOW_T2I_PATH.exists():
        raise ValueError(f"Workflow not found: {WORKFLOW_T2I_PATH}")

    raw = WORKFLOW_T2I_PATH.read_text(encoding="utf-8")
    model_file, clip_file = _resolve_flux_preset(kwargs.get("flux_model", DEFAULT_FLUX_PRESET))
    logger.info(f"Using workflow: text_to_image.json (Flux Klein: {model_file})")

    replacements = {
        "{{PROMPT}}":      str(kwargs.get("prompt", "")),
        "{{WIDTH}}":       str(kwargs.get("width", 1024)),
        "{{HEIGHT}}":      str(kwargs.get("height", 1024)),
        "{{SEED}}":        str(kwargs.get("seed", 42)),
        "{{STEPS}}":       str(kwargs.get("steps", 4)),
        "{{MODEL_NAME}}":  model_file,
        "{{CLIP_NAME}}":   clip_file,
        "{{VAE_NAME}}":    str(kwargs.get("vae_name", DEFAULT_FLUX_VAE)),
    }

    for key, val in replacements.items():
        raw = raw.replace(key, val)

    workflow = json.loads(raw)
    _fix_numeric_strings(workflow)
    return workflow


def load_edit_workflow(**kwargs) -> dict:
    """Load the Image Edit workflow template and fill in variables."""
    if not WORKFLOW_EDIT_PATH.exists():
        raise ValueError(f"Workflow not found: {WORKFLOW_EDIT_PATH}")

    raw = WORKFLOW_EDIT_PATH.read_text(encoding="utf-8")
    model_file, clip_file = _resolve_flux_preset(kwargs.get("flux_model", DEFAULT_FLUX_PRESET))
    logger.info(f"Using workflow: image_edit.json (Flux Klein I2I: {model_file})")

    replacements = {
        "{{PROMPT}}":       str(kwargs.get("prompt", "")),
        "{{SEED}}":         str(kwargs.get("seed", 42)),
        "{{STEPS}}":        str(kwargs.get("steps", 8)),
        "{{DENOISE}}":      str(kwargs.get("denoise", 0.7)),
        "{{INPUT_IMAGE}}":  str(kwargs.get("input_image", "")),
        "{{MODEL_NAME}}":   model_file,
        "{{CLIP_NAME}}":    clip_file,
        "{{VAE_NAME}}":     str(kwargs.get("vae_name", DEFAULT_FLUX_VAE)),
    }

    for key, val in replacements.items():
        raw = raw.replace(key, val)

    workflow = json.loads(raw)
    _fix_numeric_strings(workflow)
    return workflow


def load_image_ref_workflow(**kwargs) -> dict:
    """Load the T2I+Reference workflow template and fill in variables."""
    ref_images = kwargs.get("ref_images", [])
    num_refs = len(ref_images)
    if num_refs == 0:
        return load_image_workflow(**kwargs)

    wf_path = WORKFLOW_REF2_PATH if num_refs >= 2 else WORKFLOW_REF1_PATH
    if not wf_path.exists():
        raise ValueError(f"Workflow not found: {wf_path}")

    raw = wf_path.read_text(encoding="utf-8")
    model_file, clip_file = _resolve_flux_preset(kwargs.get("flux_model", DEFAULT_FLUX_PRESET))
    logger.info(f"Using workflow: {wf_path.name} ({num_refs} refs, {model_file})")

    replacements = {
        "{{PROMPT}}":        str(kwargs.get("prompt", "")),
        "{{WIDTH}}":         str(kwargs.get("width", 1024)),
        "{{HEIGHT}}":        str(kwargs.get("height", 1024)),
        "{{SEED}}":          str(kwargs.get("seed", 42)),
        "{{STEPS}}":         str(kwargs.get("steps", 4)),
        "{{REF_STRENGTH}}":  str(kwargs.get("ref_strength", 0.5)),
        "{{REF_IMAGE_1}}":   str(ref_images[0]) if num_refs >= 1 else "",
        "{{REF_IMAGE_2}}":   str(ref_images[1]) if num_refs >= 2 else "",
        "{{MODEL_NAME}}":    model_file,
        "{{CLIP_NAME}}":     clip_file,
        "{{VAE_NAME}}":      str(kwargs.get("vae_name", DEFAULT_FLUX_VAE)),
    }

    for key, val in replacements.items():
        raw = raw.replace(key, val)

    workflow = json.loads(raw)
    _fix_numeric_strings(workflow)
    return workflow


def _fix_numeric_strings(workflow: dict):
    """Convert numeric strings back to numbers in input fields."""
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        for k, v in inputs.items():
            if isinstance(v, str) and v.replace(".", "").replace("-", "").isdigit():
                inputs[k] = float(v) if "." in v else int(v)


async def _upload_image_to_comfyui(image_bytes: bytes, filename: str) -> str:
    """Upload an image to ComfyUI's input directory and return the filename."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{COMFYUI_URL}/upload/image",
            files={"image": (filename, image_bytes, "image/png")},
            data={"overwrite": "true"},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to upload image to ComfyUI: {resp.text}")

        result = resp.json()
        comfy_filename = result.get("name", filename)
        subfolder = result.get("subfolder", "")
        if subfolder:
            comfy_filename = f"{subfolder}/{comfy_filename}"
        logger.info(f"Uploaded image to ComfyUI: {comfy_filename}")
        return comfy_filename


async def _run_comfyui_generation(job_id: str, params: dict, mode: str = "video"):
    """Send a workflow to ComfyUI and poll for completion."""
    try:
        jobs[job_id]["status"] = "generating"

        if mode == "image":
            workflow = load_image_workflow(**params)
        elif mode == "image_ref":
            workflow = load_image_ref_workflow(**params)
        elif mode == "edit":
            workflow = load_edit_workflow(**params)
        else:
            workflow = load_video_workflow(**params)

        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": workflow},
            )
            if resp.status_code != 200:
                raise RuntimeError(f"ComfyUI rejected workflow: {resp.text}")

            comfy_data = resp.json()
            prompt_id = comfy_data["prompt_id"]
            logger.info(f"Job {job_id} [{mode.upper()}]: ComfyUI prompt_id={prompt_id}")

            while True:
                await asyncio.sleep(2)
                try:
                    hist_resp = await client.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10.0)
                    if hist_resp.status_code != 200:
                        continue
                except (httpx.TimeoutException, httpx.ReadError, httpx.ConnectError):
                    logger.warning(f"Job {job_id}: ComfyUI HTTP exception (GPU busy). Retrying...")
                    continue

                history = hist_resp.json()
                if prompt_id not in history:
                    continue

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
                                ext = Path(filename).suffix or (".mp4" if mode == "video" else ".png")
                                out_path = OUTPUT_DIR / f"{job_id}{ext}"
                                out_path.write_bytes(view_resp.content)
                                jobs[job_id].update({
                                    "status": "done",
                                    "result": f"/outputs/{job_id}{ext}",
                                    "mode": mode,
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
        "model": "LTX-2.3 22B + Flux Klein 4B/9B",
        "flux_presets": list(FLUX_PRESETS.keys()),
    }


@app.post("/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    prompt:           str   = Form(...),
    negative_prompt:  str   = Form("blurry, low quality, still frame, watermark, overlay, titles"),
    width:            int   = Form(768),
    height:           int   = Form(512),
    num_frames:       int   = Form(97),
    frame_rate:       float = Form(24.0),
    cfg:              float = Form(1.0),
    seed:             int   = Form(-1),
    enable_audio:     bool  = Form(False),
):
    """Generate a video from text prompt (two-stage T2V pipeline)."""
    job_id = str(uuid.uuid4())[:8]
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    jobs[job_id] = {"status": "queued", "result": None, "mode": "video"}

    params = dict(
        prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, num_frames=num_frames,
        frame_rate=frame_rate, cfg=cfg, seed=actual_seed,
        enable_audio=enable_audio,
    )

    background_tasks.add_task(_run_comfyui_generation, job_id, params, "video")
    return JSONResponse({"job_id": job_id, "seed": actual_seed, "mode": "video"})


@app.post("/generate-image")
async def generate_image(
    background_tasks: BackgroundTasks,
    prompt:      str = Form(...),
    width:       int = Form(1024),
    height:      int = Form(1024),
    steps:       int = Form(4),
    seed:        int = Form(-1),
    flux_model:  str = Form("klein-4b"),
):
    """Generate an image from text prompt (Flux.2 Klein)."""
    job_id = str(uuid.uuid4())[:8]
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    jobs[job_id] = {"status": "queued", "result": None, "mode": "image"}

    params = dict(
        prompt=prompt, width=width, height=height,
        steps=steps, seed=actual_seed, flux_model=flux_model,
    )

    background_tasks.add_task(_run_comfyui_generation, job_id, params, "image")
    return JSONResponse({"job_id": job_id, "seed": actual_seed, "mode": "image", "flux_model": flux_model})


@app.post("/generate-image-ref")
async def generate_image_ref(
    background_tasks: BackgroundTasks,
    prompt:        str   = Form(...),
    width:         int   = Form(1024),
    height:        int   = Form(1024),
    steps:         int   = Form(4),
    seed:          int   = Form(-1),
    ref_strength:  float = Form(0.5),
    flux_model:    str   = Form("klein-4b"),
    ref_images:    list[UploadFile] = File(...),
):
    """Generate an image with reference style conditioning (Flux Redux)."""
    job_id = str(uuid.uuid4())[:8]
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    # Upload reference images to ComfyUI
    comfy_ref_filenames = []
    for i, ref in enumerate(ref_images[:2]):  # max 2 refs
        ref_bytes = await ref.read()
        fname = await _upload_image_to_comfyui(ref_bytes, f"ref_{job_id}_{i}.png")
        comfy_ref_filenames.append(fname)

    jobs[job_id] = {"status": "queued", "result": None, "mode": "image_ref"}

    params = dict(
        prompt=prompt, width=width, height=height,
        steps=steps, seed=actual_seed, flux_model=flux_model,
        ref_images=comfy_ref_filenames, ref_strength=ref_strength,
    )

    background_tasks.add_task(_run_comfyui_generation, job_id, params, "image_ref")
    return JSONResponse({"job_id": job_id, "seed": actual_seed, "mode": "image_ref", "refs": len(comfy_ref_filenames)})


@app.post("/generate-edit")
async def generate_edit(
    background_tasks: BackgroundTasks,
    prompt:      str        = Form(...),
    denoise:     float      = Form(0.7),
    steps:       int        = Form(8),
    seed:        int        = Form(-1),
    flux_model:  str        = Form("klein-4b"),
    image:       UploadFile = File(...),
):
    """Edit an image using prompt + denoise (Flux.2 Klein I2I)."""
    job_id = str(uuid.uuid4())[:8]
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    # Upload input image to ComfyUI
    image_bytes = await image.read()
    comfy_filename = await _upload_image_to_comfyui(image_bytes, f"edit_{job_id}.png")

    jobs[job_id] = {"status": "queued", "result": None, "mode": "edit"}

    params = dict(
        prompt=prompt, steps=steps, denoise=denoise,
        seed=actual_seed, input_image=comfy_filename, flux_model=flux_model,
    )

    background_tasks.add_task(_run_comfyui_generation, job_id, params, "edit")
    return JSONResponse({"job_id": job_id, "seed": actual_seed, "mode": "edit", "flux_model": flux_model})


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JSONResponse(job)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
