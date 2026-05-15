"""Download models and install dependencies for two_llm_chat."""

import os
import platform
import subprocess
import sys
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(SCRIPT_DIR, "voices")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

GGUF_MODEL = "dolphin-2.6-mistral-7b.Q4_K_M.gguf"
GGUF_URL = f"https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-GGUF/resolve/main/{GGUF_MODEL}"

PIPER_VOICES = {
    "en_US-lessac-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium",
    "en_GB-alan-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium",
}


def download(url, dest):
    """Download a file with progress."""
    if os.path.exists(dest):
        print(f"  Already exists: {os.path.basename(dest)}")
        return
    print(f"  Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Done ({os.path.getsize(dest) / 1e6:.1f} MB)")


def has_nvidia_gpu():
    """Check if an NVIDIA GPU is available."""
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def install_deps():
    """Install Python dependencies, with CUDA support if available."""
    cuda = has_nvidia_gpu()

    if cuda:
        print("\nNVIDIA GPU detected — installing llama-cpp-python with CUDA support...")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.2.0", "--force-reinstall", "--no-cache-dir"],
            env=env, check=True,
        )
    else:
        system = platform.system()
        if system == "Darwin":
            print("\nApple system — installing llama-cpp-python with Metal support...")
            env = os.environ.copy()
            env["CMAKE_ARGS"] = "-DGGML_METAL=on"
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.2.0", "--force-reinstall", "--no-cache-dir"],
                env=env, check=True,
            )
        else:
            print("\nNo GPU detected — installing llama-cpp-python (CPU only)...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.2.0"],
                check=True,
            )

    print("Installing other dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "piper-tts>=1.4.0"],
        check=True,
    )


def download_model():
    """Download the GGUF model."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, GGUF_MODEL)
    print(f"\nGGUF model -> {MODELS_DIR}/")
    download(GGUF_URL, dest)
    return dest


def download_voices():
    """Download piper voice models."""
    os.makedirs(VOICES_DIR, exist_ok=True)
    print(f"\nPiper voices -> {VOICES_DIR}/")
    for name, base_url in PIPER_VOICES.items():
        download(f"{base_url}/{name}.onnx", os.path.join(VOICES_DIR, f"{name}.onnx"))
        download(f"{base_url}/{name}.onnx.json", os.path.join(VOICES_DIR, f"{name}.onnx.json"))


def main():
    print("=== Two LLM Chat Setup ===")

    install_deps()
    model_path = download_model()
    download_voices()

    print(f"\n=== Setup complete! ===")
    print(f"Run with:")
    print(f"  python3 two_llm_chat.py {model_path}")


if __name__ == "__main__":
    main()
