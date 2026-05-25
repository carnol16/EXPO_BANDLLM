"""Install dependencies, clone ACE-Step 1.5, and download models for EXPO_BANDLLM.

Usage:
    python setup.py                     # install deps + clone ACE-Step + download all models
    python setup.py --skip-models       # install deps + clone ACE-Step only
    python setup.py --coordinator-only  # download only the coordinator model (~2.2 GB)
    python setup.py --skip-acestep      # skip ACE-Step clone (already present)
"""

import argparse
import os
import platform
import subprocess
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

# ---------------------------------------------------------------------------
# Model definitions — edit URLs here if you want to swap in different models.
# All models are GGUF format. See docs/STARTUP_GUIDE.md for details.
# ---------------------------------------------------------------------------
MODELS = [
    {
        "role": "Coordinator (moderator LLM)",
        "file": "Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf",
        "url": "https://huggingface.co/cognitivecomputations/Dolphin3.0-Llama3.2-3B-GGUF/resolve/main/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf",
        "size_gb": 2.2,
        "coordinator_only": True,
    },
    {
        "role": "Rex — Singer (cynical, confrontational)",
        "file": "meta-llama-3.1-8b-instruct-abliterated.Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-abliterated-v3-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-abliterated-v3-Q4_K_M.gguf",
        "size_gb": 4.6,
        "coordinator_only": False,
    },
    {
        "role": "Charles — Guitarist (abrasive, noise-committed)",
        "file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        "size_gb": 4.1,
        "coordinator_only": False,
    },
    {
        "role": "George — Bassist (contrarian, holds a grudge)",
        "file": "Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat.gguf",
        "url": "https://huggingface.co/mradermacher/Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored-i1-GGUF/resolve/main/Qwen3-8B-Hivemind-Inst-Hrtic-Ablit-Uncensored.i1-Q4_K_M.gguf",
        "size_gb": 4.3,
        "coordinator_only": False,
    },
    {
        "role": "Johnathan — Drummer (precise, nihilistic)",
        "file": "dolphin-2.6-mistral-7b.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-GGUF/resolve/main/dolphin-2.6-mistral-7b.Q4_K_M.gguf",
        "size_gb": 3.8,
        "coordinator_only": False,
    },
]


def download(url, dest):
    if os.path.exists(dest):
        print(f"  Already exists: {os.path.basename(dest)}")
        return
    print(f"  Downloading {os.path.basename(dest)}...")

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            print(f"\r  {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\r  Done ({os.path.getsize(dest) / 1e9:.2f} GB)")


def has_nvidia_gpu():
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def install_deps():
    print("\n=== Installing dependencies ===")

    cuda = has_nvidia_gpu()
    system = platform.system()

    if cuda:
        print("NVIDIA GPU detected — building llama-cpp-python with CUDA support...")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.2.0",
             "--force-reinstall", "--no-cache-dir"],
            env=env, check=True,
        )
    elif system == "Darwin":
        print("Apple Silicon detected — building llama-cpp-python with Metal support...")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_METAL=on"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.2.0",
             "--force-reinstall", "--no-cache-dir"],
            env=env, check=True,
        )
    else:
        print("No GPU detected — installing llama-cpp-python (CPU only)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.2.0"],
            check=True,
        )

    print("Installing remaining dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r",
         os.path.join(SCRIPT_DIR, "requirements.txt")],
        check=True,
    )
    print("Dependencies installed.")


def download_models(coordinator_only=False):
    os.makedirs(MODELS_DIR, exist_ok=True)

    to_download = [m for m in MODELS if not coordinator_only or m["coordinator_only"]]
    total_gb = sum(m["size_gb"] for m in to_download)

    print(f"\n=== Downloading models ({len(to_download)} files, ~{total_gb:.1f} GB total) ===")
    print(f"Destination: {MODELS_DIR}\n")

    for model in to_download:
        print(f"[{model['role']}]")
        dest = os.path.join(MODELS_DIR, model["file"])
        download(model["url"], dest)

    print("\nAll models ready.")


_ACESTEP_DIR = os.path.join(SCRIPT_DIR, "ACE-Step-1.5")
_ACESTEP_URL = "https://github.com/ace-step/ACE-Step"


def clone_acestep():
    if os.path.isdir(_ACESTEP_DIR):
        print(f"\n=== ACE-Step 1.5 already present at {_ACESTEP_DIR} — skipping clone ===")
        return
    print(f"\n=== Cloning ACE-Step 1.5 into {_ACESTEP_DIR} ===")
    try:
        subprocess.run(
            ["git", "clone", _ACESTEP_URL, _ACESTEP_DIR],
            check=True,
        )
        print("ACE-Step 1.5 cloned.")
    except FileNotFoundError:
        print("ERROR: git not found — install git and re-run, or clone manually:")
        print(f"  git clone {_ACESTEP_URL} ACE-Step-1.5")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: git clone failed (exit {e.returncode})")
        print("Clone manually and re-run with --skip-acestep.")


def main():
    parser = argparse.ArgumentParser(description="Set up EXPO_BANDLLM.")
    parser.add_argument("--skip-models", action="store_true",
                        help="Install dependencies only, skip model downloads")
    parser.add_argument("--coordinator-only", action="store_true",
                        help="Download only the coordinator model (~2.2 GB)")
    parser.add_argument("--skip-acestep", action="store_true",
                        help="Skip ACE-Step 1.5 clone (use if already present)")
    args = parser.parse_args()

    print("=== EXPO_BANDLLM — Setup ===")

    install_deps()

    if not args.skip_acestep:
        clone_acestep()
    else:
        print("\nSkipping ACE-Step clone (--skip-acestep).")

    if not args.skip_models:
        download_models(coordinator_only=args.coordinator_only)
    else:
        print("\nSkipping model downloads (--skip-models).")
        print("Drop your .gguf files into models/ before running.")

    print("\n=== Setup complete! ===")
    print("Run: python band_coordinator.py --moderator-model models/Dolphin3.0-Llama3.2-3B-Q5_K_M.gguf --bot-model models/<bot-model>.gguf")


if __name__ == "__main__":
    main()
