from __future__ import annotations

from davinci_resolve_checker.checks.nvidia import check_nvidia
from davinci_resolve_checker.models import CheckStatus
from tests.conftest import INTEL_GPU, NVIDIA_GPU, POCL_PLATFORM, _make_state


class TestCheckNvidia:
    def test_all_good(self, nvidia_desktop):
        results = check_nvidia(nvidia_desktop)
        assert any(r.status == CheckStatus.PASS for r in results)
        assert all(r.status != CheckStatus.FAIL for r in results)

    def test_missing_opencl_nvidia(self):
        state = _make_state(
            gpus=[NVIDIA_GPU],
            opencl_nvidia_installed=False,
            gl_vendor="NVIDIA Corporation",
            gl_renderer="NVIDIA GeForce RTX 2070 SUPER",
        )
        results = check_nvidia(state)
        assert any(r.status == CheckStatus.FAIL and "opencl-nvidia" in r.message for r in results)

    def test_wrong_kernel_driver(self):
        bad_gpu = NVIDIA_GPU.model_copy(update={"driver": "nouveau"})
        state = _make_state(
            gpus=[bad_gpu],
            opencl_nvidia_installed=True,
            gl_vendor="NVIDIA Corporation",
            gl_renderer="NVIDIA GeForce RTX 2070 SUPER",
        )
        results = check_nvidia(state)
        assert any(r.status == CheckStatus.FAIL and "nvidia" in r.message.lower() for r in results)

    def test_wrong_gl_vendor(self):
        state = _make_state(
            gpus=[INTEL_GPU, NVIDIA_GPU],
            opencl_nvidia_installed=True,
            gl_vendor="Intel",
            gl_renderer="Mesa Intel(R) UHD Graphics 630",
        )
        results = check_nvidia(state)
        assert any(
            r.status == CheckStatus.FAIL
            and ("GL_VENDOR" in r.message or "renderer" in r.message.lower())
            for r in results
        )

    def test_missing_nvidia_opencl_platform(self):
        state = _make_state(
            gpus=[NVIDIA_GPU],
            opencl_drivers=["opencl-nvidia"],
            opencl_platforms=[POCL_PLATFORM],
            opencl_nvidia_installed=True,
            gl_vendor="NVIDIA Corporation",
            gl_renderer="NVIDIA GeForce RTX 2070 SUPER",
        )
        results = check_nvidia(state)
        assert any(r.status == CheckStatus.FAIL and "OpenCL platform" in r.message for r in results)

    def test_emit_pass_false_suppresses_success_message(self, nvidia_desktop):
        results = check_nvidia(nvidia_desktop, emit_pass=False)
        assert all(r.status != CheckStatus.PASS for r in results)

    def test_no_nvidia_gpu_returns_empty(self):
        state = _make_state(gpus=[INTEL_GPU])
        results = check_nvidia(state)
        assert results == []
