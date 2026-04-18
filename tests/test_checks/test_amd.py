from __future__ import annotations

from davinci_resolve_checker.checks.amd import check_amd
from davinci_resolve_checker.models import ChassisType, CheckStatus, GPUVendor
from tests.conftest import (
    AMD_ELLESMERE_GPU,
    AMD_NAVI_GPU,
    AMD_RADEON_DRIVER_GPU,
    INTEL_GPU,
    POCL_PLATFORM,
    ROC_PLATFORM,
    _make_gpu,
    _make_state,
)


class TestCheckAmdDriver:
    def test_amdgpu_driver(self, amd_desktop):
        results = check_amd(amd_desktop)
        assert all(r.status != CheckStatus.FAIL for r in results)

    def test_radeon_driver_with_amdgpu_available(self):
        state = _make_state(
            gpus=[AMD_RADEON_DRIVER_GPU],
            gl_vendor="AMD",
            gl_renderer="AMD Radeon RX 580",
        )
        results = check_amd(state)
        assert any(r.status == CheckStatus.FAIL and "radeon" in r.message.lower() for r in results)

    def test_no_amdgpu_driver(self):
        gpu = _make_gpu(
            name="Some Old AMD GPU",
            vendor=GPUVendor.AMD,
            driver="radeon",
            kernel_modules=["radeon"],
        )
        state = _make_state(gpus=[gpu], gl_vendor="AMD", gl_renderer="AMD GPU")
        results = check_amd(state)
        assert any(r.status == CheckStatus.FAIL for r in results)

    def test_no_driver_at_all(self):
        gpu = _make_gpu(
            name="Navi 23",
            vendor=GPUVendor.AMD,
            driver=None,
            kernel_modules=[],
        )
        state = _make_state(gpus=[gpu], gl_vendor="AMD", gl_renderer="AMD GPU")
        results = check_amd(state)
        assert any(r.status == CheckStatus.FAIL for r in results)


class TestCheckAmdMixedIntel:
    def test_intel_amd_mobile_rejected(self):
        state = _make_state(
            chassis=ChassisType.LAPTOP,
            gpus=[INTEL_GPU, AMD_NAVI_GPU],
            gl_vendor="AMD",
            gl_renderer="AMD Radeon RX 6600",
        )
        results = check_amd(state)
        assert any(r.status == CheckStatus.FAIL and "Intel" in r.message for r in results)

    def test_intel_amd_desktop_intel_primary(self):
        state = _make_state(
            chassis=ChassisType.DESKTOP,
            gpus=[INTEL_GPU, AMD_NAVI_GPU],
            gl_vendor="Intel",
            gl_renderer="Mesa Intel(R) UHD Graphics 630",
        )
        results = check_amd(state)
        assert any(r.status == CheckStatus.FAIL and "primary" in r.message.lower() for r in results)

    def test_intel_amd_desktop_amd_primary_ok(self):
        state = _make_state(
            chassis=ChassisType.DESKTOP,
            gpus=[INTEL_GPU, AMD_NAVI_GPU],
            opencl_drivers=["rocm-opencl-runtime"],
            opencl_platforms=[ROC_PLATFORM],
            gl_vendor="AMD",
            gl_renderer="AMD Radeon RX 6600",
        )
        results = check_amd(state)
        assert all(r.status != CheckStatus.FAIL or "Intel" not in r.message for r in results)


class TestCheckAmdPro:
    def test_pro_stack_all_good(self):
        state = _make_state(
            gpus=[AMD_NAVI_GPU],
            opencl_drivers=["opencl-amd"],
            opencl_platforms=[ROC_PLATFORM],
            gl_vendor="Advanced Micro Devices, Inc.",
            gl_renderer="AMD Radeon RX 6600",
            package_versions={"opencl-amd": "6.1.0", "amdgpu-pro-libgl": "6.1.0"},
        )
        results = check_amd(state, pro_stack=True)
        assert all(r.status != CheckStatus.FAIL for r in results)

    def test_pro_stack_missing_opengl(self):
        state = _make_state(
            gpus=[AMD_NAVI_GPU],
            opencl_drivers=["opencl-amd"],
            opencl_platforms=[ROC_PLATFORM],
            gl_vendor="AMD",
            gl_renderer="AMD Radeon RX 6600",
        )
        results = check_amd(state, pro_stack=True)
        assert any(r.status == CheckStatus.FAIL and "OpenGL" in r.message for r in results)

    def test_pro_stack_missing_opencl(self):
        state = _make_state(
            gpus=[AMD_NAVI_GPU],
            opencl_drivers=[],
            opencl_platforms=[ROC_PLATFORM],
            gl_vendor="Advanced Micro Devices, Inc.",
            gl_renderer="AMD Radeon RX 6600",
        )
        results = check_amd(state, pro_stack=True)
        assert any(r.status == CheckStatus.FAIL and "OpenCL" in r.message for r in results)

    def test_pro_stack_version_mismatch(self):
        state = _make_state(
            gpus=[AMD_NAVI_GPU],
            opencl_drivers=["opencl-amd"],
            opencl_platforms=[ROC_PLATFORM],
            gl_vendor="Advanced Micro Devices, Inc.",
            gl_renderer="AMD Radeon RX 6600",
            package_versions={"opencl-amd": "6.1.0", "amdgpu-pro-libgl": "6.0.0"},
        )
        results = check_amd(state, pro_stack=True)
        assert any(
            r.status == CheckStatus.WARNING and "mismatch" in r.message.lower() for r in results
        )

    def test_pro_stack_pre_vega_needs_legacy_opencl(self):
        state = _make_state(
            gpus=[AMD_ELLESMERE_GPU],
            opencl_drivers=["opencl-legacy-amdgpu-pro"],
            opencl_platforms=[ROC_PLATFORM],
            gl_vendor="Advanced Micro Devices, Inc.",
            gl_renderer="AMD Radeon RX 580",
        )
        results = check_amd(state, pro_stack=True)
        assert all(r.status != CheckStatus.FAIL or "OpenCL" not in r.message for r in results)

    def test_pro_stack_pre_vega_opencl_amd_alone_fails(self):
        state = _make_state(
            gpus=[AMD_ELLESMERE_GPU],
            opencl_drivers=["opencl-amd"],
            opencl_platforms=[ROC_PLATFORM],
            gl_vendor="Advanced Micro Devices, Inc.",
            gl_renderer="AMD Radeon RX 580",
        )
        results = check_amd(state, pro_stack=True)
        assert any(r.status == CheckStatus.FAIL and "OpenCL" in r.message for r in results)


class TestCheckAmdOpen:
    def test_open_stack_rocm_good(self, amd_desktop):
        results = check_amd(amd_desktop)
        assert any(r.status == CheckStatus.PASS for r in results)

    def test_open_stack_missing_rocm(self):
        state = _make_state(
            gpus=[AMD_NAVI_GPU],
            opencl_drivers=[],
            gl_vendor="AMD",
            gl_renderer="AMD Radeon RX 6600",
        )
        results = check_amd(state)
        assert any(r.status == CheckStatus.FAIL for r in results)

    def test_open_stack_opencl_amd_warns(self):
        state = _make_state(
            gpus=[AMD_NAVI_GPU],
            opencl_drivers=["opencl-amd"],
            gl_vendor="AMD",
            gl_renderer="AMD Radeon RX 6600",
        )
        results = check_amd(state)
        assert any(r.status == CheckStatus.FAIL and "rocm" in r.message.lower() for r in results)

    def test_pre_vega_needs_env_var(self):
        state = _make_state(
            gpus=[AMD_ELLESMERE_GPU],
            opencl_drivers=["rocm-opencl-runtime"],
            gl_vendor="AMD",
            gl_renderer="AMD Radeon RX 580",
            roc_enable_pre_vega=False,
        )
        results = check_amd(state)
        assert any(
            r.status == CheckStatus.FAIL and "ROC_ENABLE_PRE_VEGA" in r.message for r in results
        )

    def test_pre_vega_with_env_var(self, amd_pre_vega):
        results = check_amd(amd_pre_vega)
        assert all(
            r.status != CheckStatus.FAIL or "ROC_ENABLE_PRE_VEGA" not in r.message for r in results
        )

    def test_open_stack_unknown_codename_warns(self):
        gpu = _make_gpu(
            name="Some Unknown AMD GPU",
            vendor=GPUVendor.AMD,
            driver="amdgpu",
            kernel_modules=["amdgpu"],
        )
        state = _make_state(
            gpus=[gpu],
            opencl_drivers=["rocm-opencl-runtime"],
            opencl_platforms=[ROC_PLATFORM],
            gl_vendor="AMD",
            gl_renderer="AMD Unknown GPU",
        )
        results = check_amd(state)
        assert any(
            r.status == CheckStatus.WARNING and "undetectable" in r.message.lower() for r in results
        )
        assert all(
            r.status != CheckStatus.FAIL or "ROC_ENABLE_PRE_VEGA" not in r.message for r in results
        )

    def test_missing_amd_opencl_platform_fails(self):
        state = _make_state(
            gpus=[AMD_NAVI_GPU],
            opencl_drivers=["rocm-opencl-runtime"],
            opencl_platforms=[POCL_PLATFORM],
            gl_vendor="AMD",
            gl_renderer="AMD Radeon RX 6600",
        )
        results = check_amd(state)
        assert any(r.status == CheckStatus.FAIL and "OpenCL platform" in r.message for r in results)

    def test_emit_pass_false_suppresses_success_message(self, amd_desktop):
        results = check_amd(amd_desktop, emit_pass=False)
        assert all(r.status != CheckStatus.PASS for r in results)

    def test_no_amd_gpu_returns_empty(self):
        state = _make_state(gpus=[INTEL_GPU])
        results = check_amd(state)
        assert results == []
