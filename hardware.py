"""Hardware auto-detection for optimal llama-cpp settings."""

import os
import platform
import subprocess
from collections import namedtuple

HardwareProfile = namedtuple("HardwareProfile", ["n_gpu_layers", "n_ctx", "name"])


def register_cuda_dll_dirs():
    """Register CUDA DLL directories so ctypes can find them.

    Python 3.8+ on Windows no longer searches PATH for DLLs.
    This must be called BEFORE importing llama_cpp.
    """
    if platform.system() != "Windows" or not hasattr(os, "add_dll_directory"):
        return

    def _add(path):
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)
            except OSError:
                pass

    # CUDA toolkit (any version installed on the system)
    cuda_base = os.environ.get(
        "CUDA_PATH",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
    )
    for subdir in ("bin\\x64", "bin"):
        _add(os.path.join(cuda_base, subdir))

    # nvidia pip packages (nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, etc.)
    # These bundle CUDA 12.x DLLs in site-packages/nvidia/*/bin/
    import site
    for site_dir in site.getsitepackages():
        nvidia_root = os.path.join(site_dir, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for pkg in os.listdir(nvidia_root):
            for subdir in ("bin", os.path.join("lib", "x64")):
                _add(os.path.join(nvidia_root, pkg, subdir))


def patch_jinja_loopcontrols():
    """Enable {% break %}/{% continue %} in Jinja2 for llama-cpp-python chat templates.

    Some GGUF models (e.g. c4ai-command-r) embed chat templates that use
    {% break %} inside loops. llama-cpp-python's Jinja2 environment omits the
    loopcontrols extension, causing a TemplateSyntaxError at load time.
    Must be called before 'from llama_cpp import Llama'.
    """
    try:
        import jinja2
        _orig = jinja2.Environment.__init__

        def _patched(self, *args, **kwargs):
            exts = list(kwargs.get("extensions", []))
            if "jinja2.ext.loopcontrols" not in exts:
                exts.append("jinja2.ext.loopcontrols")
            kwargs["extensions"] = exts
            _orig(self, *args, **kwargs)

        jinja2.Environment.__init__ = _patched
    except Exception:
        pass


def _cuda_available() -> bool:
    """Return True if an NVIDIA GPU is accessible via nvidia-smi or torch."""
    # Primary: nvidia-smi (works without torch, just needs the NVIDIA driver)
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Fallback: torch (if installed)
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except (ImportError, AttributeError):
        pass

    return False


def query_free_vram_mb() -> int | None:
    """Query free GPU VRAM in MB via nvidia-smi. Returns None on any failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        # Multi-GPU: take first line (GPU 0)
        first_line = result.stdout.decode().strip().splitlines()[0].strip()
        return int(first_line)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError,
            ValueError, IndexError):
        return None


_ESTIMATED_LAYERS = 32  # default for 7B models
_KV_BUFFER_MB = 256     # reserve for KV cache
_CONSERVATIVE_GPU_LAYERS = 10  # fallback when nvidia-smi fails on CUDA systems


def calculate_gpu_layers(model_path: str) -> int:
    """Calculate optimal n_gpu_layers based on free VRAM and model size.

    Returns -1 for full offload, 0 for CPU-only, or a positive int for partial.
    Never returns -1 blindly — always checks available VRAM first.
    """
    free_vram = query_free_vram_mb()
    if free_vram is None:
        # nvidia-smi failed — use conservative fallback
        if _cuda_available():
            print(f"  VRAM query failed — using conservative {_CONSERVATIVE_GPU_LAYERS} GPU layers")
            return _CONSERVATIVE_GPU_LAYERS
        # Non-CUDA systems (Metal, CPU, RPi) — delegate to detect_hardware_profile
        return detect_hardware_profile().n_gpu_layers

    try:
        model_size_bytes = os.path.getsize(model_path)
    except OSError:
        print(f"  WARNING: cannot read model file size, falling back to CPU")
        return 0

    model_size_mb = model_size_bytes / (1024 * 1024)
    if model_size_mb <= 0:
        return 0

    per_layer_mb = model_size_mb / _ESTIMATED_LAYERS
    if per_layer_mb <= 0:
        return 0

    usable = max(0, free_vram - _KV_BUFFER_MB)
    gpu_layers = int(usable // per_layer_mb)
    gpu_layers = min(gpu_layers, _ESTIMATED_LAYERS)

    if gpu_layers >= _ESTIMATED_LAYERS:
        print(f"  VRAM: {free_vram}MB free → full offload (all {_ESTIMATED_LAYERS} layers)")
        return -1

    print(f"  VRAM: {free_vram}MB free → {gpu_layers} GPU layers "
          f"(model={int(model_size_mb)}MB, {int(per_layer_mb)}MB/layer)")
    return gpu_layers


def plan_gpu_allocation(model_sizes_mb: list[int], free_vram_mb: int | None = None) -> list[int]:
    """Plan GPU layer allocation for multiple models sharing one GPU.

    Distributes available VRAM across models. Each model gets the same
    number of GPU layers (equal-layer allocation), because the proportional
    share formula simplifies: model_size cancels out.

    Returns a list of n_gpu_layers values (one per model, same order as input).
    Returns -1 for full offload, 0 for CPU-only.
    """
    free_vram = free_vram_mb if free_vram_mb is not None else query_free_vram_mb()
    if free_vram is None:
        if _cuda_available():
            print(f"  VRAM query failed — using conservative {_CONSERVATIVE_GPU_LAYERS} GPU layers each")
            return [_CONSERVATIVE_GPU_LAYERS] * len(model_sizes_mb)
        return [detect_hardware_profile().n_gpu_layers] * len(model_sizes_mb)

    total_model_size = sum(model_sizes_mb)
    if total_model_size <= 0:
        return [0] * len(model_sizes_mb)

    kv_reserved = _KV_BUFFER_MB * len(model_sizes_mb)
    usable = max(0, free_vram - kv_reserved)

    gpu_layers = int(usable * _ESTIMATED_LAYERS // total_model_size)
    gpu_layers = min(gpu_layers, _ESTIMATED_LAYERS)

    if gpu_layers >= _ESTIMATED_LAYERS:
        print(f"  VRAM: {free_vram}MB free → full offload for all {len(model_sizes_mb)} models")
        return [-1] * len(model_sizes_mb)

    print(f"  VRAM: {free_vram}MB free → {gpu_layers} GPU layers each "
          f"({len(model_sizes_mb)} models, total={total_model_size}MB)")
    return [gpu_layers] * len(model_sizes_mb)


def detect_hardware_profile() -> HardwareProfile:
    """Detect runtime hardware and return optimal llama-cpp settings.

    Priority order:
      1. CUDA  -> SPLIT (10 layers to GPU, rest to DDR5)
      2. Metal -> METAL (all layers, unified memory)
      3. RPi   -> RPI   (CPU + LPDDR5, reduced context)
      4. CPU   -> CPU   (pure CPU fallback)
    """
    # 1. CUDA
    if _cuda_available():
        try:
            from llama_cpp.llama_cpp import llama_supports_gpu_offload
            if llama_supports_gpu_offload():
                print("Hardware: CUDA detected — GPU profile (layers calculated dynamically)")
                return HardwareProfile(n_gpu_layers=0, n_ctx=4096, name="GPU")
            else:
                print(
                    "WARNING: CUDA GPU detected but llama-cpp-python lacks GPU support.\n"
                    "Models will run on CPU only. To fix, reinstall with CUDA:\n"
                    "  pip install llama-cpp-python --force-reinstall "
                    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
                )
        except (ImportError, AttributeError):
            print(
                "WARNING: CUDA GPU detected but llama_cpp not available for GPU check.\n"
                "Falling back to non-GPU detection."
            )

    # 2. Metal via torch
    try:
        import torch
        if torch.backends.mps.is_available():
            print("Hardware: Apple Metal detected — METAL profile (all layers)")
            return HardwareProfile(n_gpu_layers=-1, n_ctx=4096, name="METAL")
    except (ImportError, AttributeError):
        pass

    # 2b. Metal without torch (arm64 Darwin)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("Hardware: Apple Silicon detected — METAL profile (all layers)")
        return HardwareProfile(n_gpu_layers=-1, n_ctx=4096, name="METAL")

    # 3. Raspberry Pi
    if platform.machine() == "aarch64":
        try:
            with open("/proc/device-tree/model") as f:
                if "Raspberry Pi" in f.read():
                    print("Hardware: Raspberry Pi detected — RPI profile (CPU only)")
                    return HardwareProfile(n_gpu_layers=0, n_ctx=2048, name="RPI")
        except OSError:
            pass

    # 4. CPU fallback
    print("Hardware: CPU fallback — CPU profile")
    return HardwareProfile(n_gpu_layers=0, n_ctx=4096, name="CPU")


def _whisper_cuda_dlls_available() -> bool:
    """Return True if the CUDA DLLs required by faster-whisper/CTranslate2 are loadable.

    Checks both cublas and cudnn — ctranslate2 requires both, and cuDNN version
    mismatches (e.g. torch+cu124 ships cuDNN 9.x while ctranslate2 expects 8.x)
    cause a hard native crash (0xC0000409) rather than a catchable Python exception.
    """
    import ctypes
    for dll in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_8.dll", "cudnn_ops_infer64_8.dll"):
        try:
            ctypes.CDLL(dll)
        except OSError:
            return False
    return True


def detect_whisper_device() -> tuple[str, str]:
    """Return (device, compute_type) for faster-whisper WhisperModel.

    faster-whisper uses CTranslate2 which supports CUDA but not Apple MPS.
    Apple Silicon falls back to CPU with int8 (still fast on M-series).

    Returns:
        (device, compute_type) e.g. ("cuda", "float16") or ("cpu", "int8")
    """
    # 1. CUDA (NVIDIA GPU via CTranslate2)
    # faster-whisper defers DLL loading until first inference, so probe now.
    # cuDNN version must match what ctranslate2 was compiled against (8.x).
    if _cuda_available() and _whisper_cuda_dlls_available():
        return ("cuda", "float16")

    # 2. Apple Silicon — CTranslate2 has no MPS backend; CPU int8 is fast on M-series
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return ("cpu", "int8")

    # 3. Raspberry Pi — confirm with /proc/device-tree/model (same check as detect_hardware_profile)
    if platform.machine() == "aarch64" and platform.system() == "Linux":
        try:
            with open("/proc/device-tree/model") as f:
                if "Raspberry Pi" in f.read():
                    return ("cpu", "int8")
        except OSError:
            pass

    # 4. CPU fallback (Windows, x86 Linux, unrecognized aarch64 boards, etc.)
    return ("cpu", "int8")
