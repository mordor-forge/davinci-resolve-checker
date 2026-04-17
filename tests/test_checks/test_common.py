from __future__ import annotations

import pytest

from davinci_resolve_checker.checks.common import (
    check_distro,
    check_gl_vendor,
    check_gpu_conflict,
    check_gpu_presence,
    check_opencl_mesa,
    check_opencl_platforms,
)
from davinci_resolve_checker.models import CheckStatus
from tests.conftest import (
    AMD_NAVI_GPU,
    CLOVER_PLATFORM,
    INTEL_GPU,
    NVIDIA_CL_PLATFORM,
    NVIDIA_GPU,
    ROC_PLATFORM,
    _make_state,
)


class TestCheckDistro:
    @pytest.mark.parametrize("distro_id", ["arch", "manjaro", "endeavouros", "garuda", "cachyos"])
    def test_supported_distros(self, distro_id: str):
        state = _make_state(distro_id=distro_id)
        results = check_distro(state)
        assert all(r.status != CheckStatus.FAIL for r in results)

    def test_unsupported_distro(self):
        state = _make_state(distro_id="ubuntu", distro_name="Ubuntu 24.04")
        results = check_distro(state)
        assert any(r.status == CheckStatus.FAIL for r in results)
        assert any("ubuntu" in r.message.lower() or "Ubuntu" in r.message for r in results)


class TestCheckOpenCLMesa:
    def test_opencl_mesa_installed(self):
        state = _make_state(opencl_drivers=["opencl-mesa", "opencl-nvidia"])
        results = check_opencl_mesa(state)
        assert any(r.status == CheckStatus.FAIL for r in results)

    def test_opencl_mesa_not_installed(self):
        state = _make_state(opencl_drivers=["opencl-nvidia"])
        results = check_opencl_mesa(state)
        assert all(r.status != CheckStatus.FAIL for r in results)


class TestCheckGPUPresence:
    def test_nvidia_gpu_present(self):
        state = _make_state(gpus=[NVIDIA_GPU])
        results = check_gpu_presence(state)
        assert all(r.status != CheckStatus.FAIL for r in results)

    def test_amd_gpu_present(self):
        state = _make_state(gpus=[AMD_NAVI_GPU])
        results = check_gpu_presence(state)
        assert all(r.status != CheckStatus.FAIL for r in results)

    def test_intel_only(self):
        state = _make_state(gpus=[INTEL_GPU])
        results = check_gpu_presence(state)
        assert any(r.status == CheckStatus.FAIL for r in results)

    def test_no_gpus(self):
        state = _make_state(gpus=[])
        results = check_gpu_presence(state)
        assert any(r.status == CheckStatus.FAIL for r in results)

    def test_multiple_intel_gpus(self):
        intel2 = INTEL_GPU.model_copy(update={"pci_slot": "0000:00:12.0"})
        state = _make_state(gpus=[INTEL_GPU, intel2])
        results = check_gpu_presence(state)
        assert any(r.status == CheckStatus.FAIL and "Intel" in r.message for r in results)

    def test_multiple_nvidia_gpus(self):
        nvidia2 = NVIDIA_GPU.model_copy(update={"pci_slot": "0000:02:00.0"})
        state = _make_state(gpus=[NVIDIA_GPU, nvidia2])
        results = check_gpu_presence(state)
        assert any(r.status == CheckStatus.FAIL and "NVIDIA" in r.message for r in results)

    def test_intel_plus_nvidia_ok(self):
        state = _make_state(gpus=[INTEL_GPU, NVIDIA_GPU])
        results = check_gpu_presence(state)
        assert all(r.status != CheckStatus.FAIL for r in results)

    def test_multiple_amd_gpus_warns(self):
        amd2 = AMD_NAVI_GPU.model_copy(update={"pci_slot": "0000:02:00.0"})
        state = _make_state(gpus=[AMD_NAVI_GPU, amd2])
        results = check_gpu_presence(state)
        assert any(r.status == CheckStatus.WARNING and "AMD" in r.message for r in results)


class TestCheckGPUConflict:
    def test_amd_and_nvidia(self):
        state = _make_state(gpus=[AMD_NAVI_GPU, NVIDIA_GPU])
        results = check_gpu_conflict(state)
        assert any(r.status == CheckStatus.FAIL for r in results)

    def test_amd_only(self):
        state = _make_state(gpus=[AMD_NAVI_GPU])
        results = check_gpu_conflict(state)
        assert len(results) == 0

    def test_nvidia_only(self):
        state = _make_state(gpus=[NVIDIA_GPU])
        results = check_gpu_conflict(state)
        assert len(results) == 0


class TestCheckGLVendor:
    def test_gl_vendor_present(self):
        state = _make_state(gl_vendor="NVIDIA Corporation")
        results = check_gl_vendor(state)
        assert len(results) == 0

    def test_gl_vendor_empty(self):
        state = _make_state(gl_vendor="")
        results = check_gl_vendor(state)
        assert any(r.status == CheckStatus.FAIL for r in results)


class TestCheckOpenCLPlatforms:
    def test_valid_platform_with_devices(self):
        state = _make_state(opencl_platforms=[ROC_PLATFORM])
        results = check_opencl_platforms(state)
        assert all(r.status != CheckStatus.FAIL for r in results)

    def test_clover_only(self):
        state = _make_state(opencl_platforms=[CLOVER_PLATFORM])
        results = check_opencl_platforms(state)
        assert any(r.status == CheckStatus.FAIL for r in results)

    def test_no_platforms(self):
        state = _make_state(opencl_platforms=[])
        results = check_opencl_platforms(state)
        assert any(r.status == CheckStatus.FAIL for r in results)

    def test_platform_with_no_devices(self):
        empty_platform = NVIDIA_CL_PLATFORM.model_copy(update={"devices": []})
        state = _make_state(opencl_platforms=[empty_platform])
        results = check_opencl_platforms(state)
        assert any(r.status == CheckStatus.FAIL for r in results)
