"""Tests for hardware auto-detection profiles."""
import sys
import unittest.mock
from unittest.mock import patch, MagicMock
import importlib


def _make_nvidia_smi_result(found: bool):
    """Return a mock CompletedProcess mimicking nvidia-smi -L output."""
    result = MagicMock()
    result.returncode = 0 if found else 1
    result.stdout = b"GPU 0: NVIDIA GeForce RTX 3080\n" if found else b""
    return result


def _mock_gpu_offload(supported: bool):
    """Return a mock for llama_cpp.llama_cpp with llama_supports_gpu_offload."""
    mock_module = MagicMock()
    mock_module.llama_supports_gpu_offload.return_value = supported
    return mock_module


def test_cuda_profile_via_nvidia_smi():
    """GPU profile returned when nvidia-smi detects a GPU and offload supported."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch("subprocess.run", return_value=_make_nvidia_smi_result(True)), \
         patch.dict(sys.modules, {"torch": None, "llama_cpp": MagicMock(), "llama_cpp.llama_cpp": _mock_gpu_offload(True)}):
        profile = hw_module.detect_hardware_profile()
    assert profile.name == "GPU"
    assert profile.n_gpu_layers == 0  # layers now calculated dynamically
    assert profile.n_ctx == 4096


def test_cuda_profile_via_torch_fallback():
    """GPU profile returned via torch when nvidia-smi is absent and offload supported."""
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = True
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch("subprocess.run", side_effect=FileNotFoundError), \
         patch.dict(sys.modules, {"torch": torch_mock, "llama_cpp": MagicMock(), "llama_cpp.llama_cpp": _mock_gpu_offload(True)}):
        profile = hw_module.detect_hardware_profile()
    assert profile.name == "GPU"
    assert profile.n_gpu_layers == 0  # layers now calculated dynamically


def test_cuda_without_gpu_offload_falls_back_to_cpu():
    """CUDA detected but llama-cpp-python lacks GPU support -> CPU profile + warning."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch("subprocess.run", return_value=_make_nvidia_smi_result(True)), \
         patch.dict(sys.modules, {"torch": None, "llama_cpp": MagicMock(), "llama_cpp.llama_cpp": _mock_gpu_offload(False)}), \
         patch("builtins.print") as mock_print:
        profile = hw_module.detect_hardware_profile()
    assert profile.name == "CPU"
    assert profile.n_gpu_layers == 0
    # Verify warning was printed
    printed = " ".join(str(c) for c in mock_print.call_args_list)
    assert "WARNING" in printed


def test_cuda_with_gpu_offload_returns_gpu():
    """CUDA detected and llama-cpp-python has GPU support -> GPU profile."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch("subprocess.run", return_value=_make_nvidia_smi_result(True)), \
         patch.dict(sys.modules, {"torch": None, "llama_cpp": MagicMock(), "llama_cpp.llama_cpp": _mock_gpu_offload(True)}):
        profile = hw_module.detect_hardware_profile()
    assert profile.name == "GPU"
    assert profile.n_gpu_layers == 0  # layers now calculated dynamically


def test_cuda_with_llama_cpp_not_installed_falls_back_to_cpu():
    """CUDA detected but llama_cpp not installed -> CPU profile."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch("subprocess.run", return_value=_make_nvidia_smi_result(True)), \
         patch.dict(sys.modules, {"torch": None, "llama_cpp": None, "llama_cpp.llama_cpp": None}):
        profile = hw_module.detect_hardware_profile()
    assert profile.name == "CPU"
    assert profile.n_gpu_layers == 0


def test_metal_profile_returns_metal():
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.backends.mps.is_available.return_value = True
    import hardware as hw_module
    with patch("subprocess.run", return_value=_make_nvidia_smi_result(False)), \
         patch.dict(sys.modules, {"torch": torch_mock}):
        importlib.reload(hw_module)
        profile = hw_module.detect_hardware_profile()
    assert profile.name == "METAL"
    assert profile.n_gpu_layers == -1
    assert profile.n_ctx == 4096


def test_rpi_profile_when_aarch64_and_rpi_model():
    import hardware as hw_module
    with patch("subprocess.run", return_value=_make_nvidia_smi_result(False)), \
         patch("platform.machine", return_value="aarch64"), \
         patch("platform.system", return_value="Linux"), \
         patch("builtins.open", unittest.mock.mock_open(read_data="Raspberry Pi 5 Model B")), \
         patch.dict(sys.modules, {"torch": None}):
        importlib.reload(hw_module)
        profile = hw_module.detect_hardware_profile()
    assert profile.name == "RPI"
    assert profile.n_gpu_layers == 0
    assert profile.n_ctx == 2048


def test_cpu_fallback():
    import hardware as hw_module
    with patch("subprocess.run", return_value=_make_nvidia_smi_result(False)), \
         patch.dict(sys.modules, {"torch": None}), \
         patch("platform.machine", return_value="x86_64"), \
         patch("platform.system", return_value="Linux"):
        importlib.reload(hw_module)
        profile = hw_module.detect_hardware_profile()
    assert profile.name == "CPU"
    assert profile.n_gpu_layers == 0
    assert profile.n_ctx == 4096


def _make_nvidia_smi_free_result(free_mb: int):
    """Return a mock CompletedProcess for nvidia-smi memory.free query."""
    result = MagicMock()
    result.returncode = 0
    result.stdout = f"{free_mb}\n".encode()
    return result


def test_query_free_vram_returns_int():
    """query_free_vram_mb returns free VRAM in MB."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch("subprocess.run", return_value=_make_nvidia_smi_free_result(4096)):
        assert hw_module.query_free_vram_mb() == 4096


def test_query_free_vram_returns_none_on_missing_nvidia_smi():
    """query_free_vram_mb returns None when nvidia-smi not found."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert hw_module.query_free_vram_mb() is None


def test_query_free_vram_returns_none_on_timeout():
    """query_free_vram_mb returns None when nvidia-smi times out."""
    import hardware as hw_module
    importlib.reload(hw_module)
    import subprocess
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
        assert hw_module.query_free_vram_mb() is None


def test_query_free_vram_returns_none_on_bad_exit():
    """query_free_vram_mb returns None on non-zero exit code."""
    import hardware as hw_module
    importlib.reload(hw_module)
    bad_result = MagicMock()
    bad_result.returncode = 1
    bad_result.stdout = b""
    with patch("subprocess.run", return_value=bad_result):
        assert hw_module.query_free_vram_mb() is None


def test_query_free_vram_returns_none_on_parse_error():
    """query_free_vram_mb returns None when output is not numeric."""
    import hardware as hw_module
    importlib.reload(hw_module)
    bad_result = MagicMock()
    bad_result.returncode = 0
    bad_result.stdout = b"[not a number]\n"
    with patch("subprocess.run", return_value=bad_result):
        assert hw_module.query_free_vram_mb() is None


def test_query_free_vram_multi_gpu_takes_first():
    """query_free_vram_mb takes first GPU line on multi-GPU systems."""
    import hardware as hw_module
    importlib.reload(hw_module)
    multi_result = MagicMock()
    multi_result.returncode = 0
    multi_result.stdout = b"8192\n4096\n"
    with patch("subprocess.run", return_value=multi_result):
        assert hw_module.query_free_vram_mb() == 8192


def test_calculate_gpu_layers_full_offload():
    """When enough VRAM for all layers, returns -1."""
    import hardware as hw_module
    importlib.reload(hw_module)
    # 4096 MB model, 32 layers = 128 MB/layer
    # 8000 MB free - 256 buffer = 7744 usable → 60 layers → clamp to 32 → return -1
    with patch.object(hw_module, "query_free_vram_mb", return_value=8000), \
         patch.object(hw_module, "_cuda_available", return_value=True), \
         patch("os.path.getsize", return_value=4096 * 1024 * 1024):
        result = hw_module.calculate_gpu_layers("/fake/model.gguf")
    assert result == -1


def test_calculate_gpu_layers_partial_offload():
    """When VRAM fits some but not all layers, returns the count."""
    import hardware as hw_module
    importlib.reload(hw_module)
    # 4096 MB model, 32 layers = 128 MB/layer
    # 2000 MB free - 256 buffer = 1744 usable → floor(1744/128) = 13
    with patch.object(hw_module, "query_free_vram_mb", return_value=2000), \
         patch.object(hw_module, "_cuda_available", return_value=True), \
         patch("os.path.getsize", return_value=4096 * 1024 * 1024):
        result = hw_module.calculate_gpu_layers("/fake/model.gguf")
    assert result == 13


def test_calculate_gpu_layers_no_vram_returns_zero():
    """When free VRAM is less than KV buffer, returns 0."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch.object(hw_module, "query_free_vram_mb", return_value=100), \
         patch.object(hw_module, "_cuda_available", return_value=True), \
         patch("os.path.getsize", return_value=4096 * 1024 * 1024):
        result = hw_module.calculate_gpu_layers("/fake/model.gguf")
    assert result == 0


def test_calculate_gpu_layers_nvidia_smi_fails_cuda_fallback():
    """When nvidia-smi fails but CUDA is available, returns conservative 10."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch.object(hw_module, "query_free_vram_mb", return_value=None), \
         patch.object(hw_module, "_cuda_available", return_value=True), \
         patch("os.path.getsize", return_value=4096 * 1024 * 1024):
        result = hw_module.calculate_gpu_layers("/fake/model.gguf")
    assert result == 10


def test_calculate_gpu_layers_no_cuda_returns_zero():
    """When no CUDA GPU, returns 0."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch.object(hw_module, "query_free_vram_mb", return_value=None), \
         patch.object(hw_module, "_cuda_available", return_value=False), \
         patch.object(hw_module, "detect_hardware_profile",
                      return_value=hw_module.HardwareProfile(n_gpu_layers=0, n_ctx=4096, name="CPU")), \
         patch("os.path.getsize", return_value=1024 * 1024):
        result = hw_module.calculate_gpu_layers("/fake/model.gguf")
    assert result == 0


def test_calculate_gpu_layers_missing_file_returns_zero():
    """When model file doesn't exist, returns 0."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch.object(hw_module, "query_free_vram_mb", return_value=8000), \
         patch.object(hw_module, "_cuda_available", return_value=True), \
         patch("os.path.getsize", side_effect=OSError("No such file")):
        result = hw_module.calculate_gpu_layers("/nonexistent/model.gguf")
    assert result == 0


def test_calculate_gpu_layers_zero_byte_file_returns_zero():
    """When model file is 0 bytes, returns 0 (avoids division by zero)."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch.object(hw_module, "query_free_vram_mb", return_value=8000), \
         patch.object(hw_module, "_cuda_available", return_value=True), \
         patch("os.path.getsize", return_value=0):
        result = hw_module.calculate_gpu_layers("/fake/model.gguf")
    assert result == 0


def test_calculate_gpu_layers_metal_returns_all_layers():
    """On Metal (no CUDA, no nvidia-smi), falls back to detect_hardware_profile."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch.object(hw_module, "query_free_vram_mb", return_value=None), \
         patch.object(hw_module, "_cuda_available", return_value=False), \
         patch.object(hw_module, "detect_hardware_profile",
                      return_value=hw_module.HardwareProfile(n_gpu_layers=-1, n_ctx=4096, name="METAL")), \
         patch("os.path.getsize", return_value=4096 * 1024 * 1024):
        result = hw_module.calculate_gpu_layers("/fake/model.gguf")
    assert result == -1


# --- plan_gpu_allocation tests ---

def test_plan_gpu_allocation_equal_models():
    """Four equal-sized models split VRAM equally."""
    import hardware as hw_module
    importlib.reload(hw_module)
    result = hw_module.plan_gpu_allocation([4096, 4096, 4096, 4096], free_vram_mb=9000)
    assert result == [15, 15, 15, 15]


def test_plan_gpu_allocation_different_models():
    """Different-sized models still get equal layer count."""
    import hardware as hw_module
    importlib.reload(hw_module)
    result = hw_module.plan_gpu_allocation([2048, 4096, 3072, 5120], free_vram_mb=9000)
    assert result == [17, 17, 17, 17]


def test_plan_gpu_allocation_full_offload():
    """When all models fit entirely, returns [-1, -1, ...]."""
    import hardware as hw_module
    importlib.reload(hw_module)
    result = hw_module.plan_gpu_allocation([500, 500], free_vram_mb=9000)
    assert result == [-1, -1]


def test_plan_gpu_allocation_vram_too_small():
    """When VRAM is less than KV reserve, returns all zeros."""
    import hardware as hw_module
    importlib.reload(hw_module)
    result = hw_module.plan_gpu_allocation([4096, 4096, 4096, 4096], free_vram_mb=500)
    assert result == [0, 0, 0, 0]


def test_plan_gpu_allocation_nvidia_smi_failure_cuda():
    """When nvidia-smi fails and CUDA available, returns conservative layers."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch.object(hw_module, "query_free_vram_mb", return_value=None), \
         patch.object(hw_module, "_cuda_available", return_value=True):
        result = hw_module.plan_gpu_allocation([4096, 4096])
    assert result == [10, 10]


def test_plan_gpu_allocation_nvidia_smi_failure_no_cuda():
    """When nvidia-smi fails and no CUDA, delegates to detect_hardware_profile."""
    import hardware as hw_module
    importlib.reload(hw_module)
    with patch.object(hw_module, "query_free_vram_mb", return_value=None), \
         patch.object(hw_module, "_cuda_available", return_value=False), \
         patch.object(hw_module, "detect_hardware_profile",
                      return_value=hw_module.HardwareProfile(n_gpu_layers=0, n_ctx=4096, name="CPU")):
        result = hw_module.plan_gpu_allocation([4096, 4096])
    assert result == [0, 0]


def test_plan_gpu_allocation_single_bot():
    """Single bot gets all available VRAM."""
    import hardware as hw_module
    importlib.reload(hw_module)
    result = hw_module.plan_gpu_allocation([4096], free_vram_mb=9000)
    assert result == [-1]


def test_plan_gpu_allocation_zero_total_size():
    """Zero-byte models return all zeros."""
    import hardware as hw_module
    importlib.reload(hw_module)
    result = hw_module.plan_gpu_allocation([0, 0], free_vram_mb=9000)
    assert result == [0, 0]


# --- detect_whisper_device tests ---


def test_cuda_returns_cuda_float16():
    import hardware as hw_module
    with patch("hardware._cuda_available", return_value=True):
        device, compute = hw_module.detect_whisper_device()
    assert device == "cuda"
    assert compute == "float16"


def test_apple_silicon_returns_cpu_int8():
    import hardware as hw_module
    with (
        patch("hardware._cuda_available", return_value=False),
        patch("hardware.platform") as mock_platform,
    ):
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"
        device, compute = hw_module.detect_whisper_device()
    assert device == "cpu"
    assert compute == "int8"


def test_raspberry_pi_returns_cpu_int8():
    import hardware as hw_module
    mock_file_content = "Raspberry Pi 4 Model B\x00"
    with (
        patch("hardware._cuda_available", return_value=False),
        patch("hardware.platform") as mock_platform,
        patch("builtins.open", MagicMock(
            return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=mock_file_content))),
                __exit__=MagicMock(return_value=False),
            )
        )),
    ):
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "aarch64"
        device, compute = hw_module.detect_whisper_device()
    assert device == "cpu"
    assert compute == "int8"


def test_cpu_fallback_returns_cpu_int8():
    import hardware as hw_module
    with (
        patch("hardware._cuda_available", return_value=False),
        patch("hardware.platform") as mock_platform,
    ):
        mock_platform.system.return_value = "Windows"
        mock_platform.machine.return_value = "AMD64"
        device, compute = hw_module.detect_whisper_device()
    assert device == "cpu"
    assert compute == "int8"


def test_returns_tuple_of_two_strings():
    import hardware as hw_module
    with patch("hardware._cuda_available", return_value=False):
        with patch("hardware.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "x86_64"
            result = hw_module.detect_whisper_device()
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(v, str) for v in result)
