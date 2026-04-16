"""
Download all models required by LTX Studio (T2V + T2I workflows).
Run this after install.bat has set up ComfyUI.

Models are downloaded from HuggingFace Hub (~28 GB total).
Some repos are gated — run `huggingface-cli login` first if needed.
"""
import os
import sys
import shutil
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
COMFYUI_DIR = ROOT_DIR / "ComfyUI"
MODELS_DIR = COMFYUI_DIR / "models"

if not COMFYUI_DIR.exists():
    print(f"ERROR: ComfyUI not found at {COMFYUI_DIR}")
    print("Run install.bat first to set up ComfyUI.")
    sys.exit(1)

# ── Model definitions ────────────────────────────────────────────────────────
# Each entry: (repo_id, filename_in_repo, local_subdir, local_filename)
MODELS = [
    # ━━━ LTX-2.3 Text-to-Video ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # GGUF transformer — distilled 22B quantized Q4_K_M (~15 GB)
    (
        "unsloth/LTX-2.3-GGUF",
        "ltx-2.3-22b-distilled-UD-Q4_K_M.gguf",
        "diffusion_models",
        "ltx-2.3-22b-distilled-UD-Q4_K_M.gguf",
    ),
    # Video VAE (~1.4 GB)
    (
        "unsloth/LTX-2.3-GGUF",
        "vae/ltx-2.3-22b-dev_video_vae.safetensors",
        "vae",
        "ltx-2.3-22b-dev_video_vae.safetensors",
    ),
    # Audio VAE (~348 MB)
    (
        "unsloth/LTX-2.3-GGUF",
        "vae/ltx-2.3-22b-dev_audio_vae.safetensors",
        "vae",
        "ltx-2.3-22b-dev_audio_vae.safetensors",
    ),
    # Text projection for LTX 2.3 (small) — DualCLIPLoader looks in clip/
    (
        "unsloth/LTX-2.3-GGUF",
        "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        "clip",
        "ltx-2.3_text_projection_bf16.safetensors",
    ),
    # Gemma 3 12B text encoder — FP4 mixed quantized (~800 MB)
    (
        "unsloth/LTX-2.3-GGUF",
        "text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
        "clip",
        "gemma_3_12B_it_fp4_mixed.safetensors",
    ),
    # Spatial upscaler x2 (~500 MB)
    (
        "Lightricks/LTX-2.3",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "latent_upscale_models",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    ),

    # ━━━ Flux.2 Klein 4B Text-to-Image ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Flux Klein 4B GGUF Q4_K_M (~2.6 GB)
    (
        "unsloth/FLUX.2-klein-4B-GGUF",
        "flux-2-klein-4b-Q4_K_M.gguf",
        "diffusion_models",
        "flux-2-klein-4b-Q4_K_M.gguf",
    ),
    # Qwen3-4B text encoder for Klein 4B (~8 GB)
    (
        "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
        "split_files/text_encoders/qwen_3_4b.safetensors",
        "clip",
        "qwen_3_4b.safetensors",
    ),
    # ━━━ Flux.2 Klein 9B Text-to-Image (HQ mode) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Flux Klein 9B GGUF Q4_K_M (~5.3 GB)
    (
        "unsloth/FLUX.2-klein-9B-GGUF",
        "flux-2-klein-9b-Q4_K_M.gguf",
        "diffusion_models",
        "flux-2-klein-9b-Q4_K_M.gguf",
    ),
    # Qwen3-8B FP4 text encoder for Klein 9B (~6.8 GB)
    (
        "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
        "split_files/text_encoders/qwen_3_8b_fp4mixed.safetensors",
        "clip",
        "qwen_3_8b_fp4mixed.safetensors",
    ),

    # ━━━ Shared: Flux2 VAE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Flux2 VAE (~320 MB) — shared by Klein 4B and 9B
    (
        "Comfy-Org/vae-text-encorder-for-flux-klein-4b",
        "split_files/vae/flux2-vae.safetensors",
        "vae",
        "flux2-vae.safetensors",
    ),

    # ━━━ Flux Redux — Multi-Reference Style Conditioning ━━━━━━━━━━━━━━━━━━━━

    # Flux Redux style model (~320 MB)
    (
        "black-forest-labs/FLUX.1-Redux-dev",
        "flux1-redux-dev.safetensors",
        "style_models",
        "flux1-redux-dev.safetensors",
    ),
    # SigCLIP Vision encoder (~360 MB)
    (
        "Comfy-Org/sigclip_vision_384",
        "sigclip_vision_patch14_384.safetensors",
        "clip_vision",
        "sigclip_vision_patch14_384.safetensors",
    ),
]


def download_all():
    total = len(MODELS)
    for i, (repo, repo_file, subdir, local_name) in enumerate(MODELS, 1):
        dest_dir = MODELS_DIR / subdir
        dest_path = dest_dir / local_name

        if dest_path.exists():
            print(f"  [{i}/{total}] {local_name} — already exists, skipping")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{i}/{total}] Downloading {local_name} from {repo}...")

        try:
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=repo_file,
                local_dir=str(dest_dir),
                local_dir_use_symlinks=False,
            )
            # hf_hub_download may place the file in a subfolder matching the repo path
            downloaded = Path(downloaded)
            if downloaded != dest_path and downloaded.exists():
                shutil.move(str(downloaded), str(dest_path))
                # Clean up empty subdirectories left behind
                for parent in downloaded.parents:
                    if parent == dest_dir:
                        break
                    try:
                        parent.rmdir()
                    except OSError:
                        break

            print(f"  [{i}/{total}] {local_name} — done")

        except Exception as e:
            print(f"  [{i}/{total}] ERROR downloading {local_name}: {e}")
            print(f"           You may need to run: huggingface-cli login")
            continue

    print()
    print("Model download complete!")
    print(f"Models location: {MODELS_DIR}")


if __name__ == "__main__":
    print("=" * 60)
    print("  LTX Studio — Model Downloader")
    print("  LTX-2.3 T2V + Flux Klein T2I (~28 GB total)")
    print("=" * 60)
    print()
    download_all()
