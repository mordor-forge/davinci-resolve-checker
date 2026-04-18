from __future__ import annotations

from unittest.mock import patch

from davinci_resolve_checker.checks import run_all_checks
from davinci_resolve_checker.models import CheckStatus
from tests.conftest import (
    AMD_NAVI_GPU,
    INTEL_GPU,
    NVIDIA_CL_PLATFORM,
    NVIDIA_GPU,
    ROC_PLATFORM,
    _make_state,
)


class TestRunAllChecks:
    def test_nvidia_desktop_passes(self, nvidia_desktop):
        results = run_all_checks(nvidia_desktop)
        assert any(r.status == CheckStatus.PASS for r in results)
        assert all(r.status != CheckStatus.FAIL for r in results)

    def test_amd_desktop_passes(self, amd_desktop):
        results = run_all_checks(amd_desktop)
        assert any(r.status == CheckStatus.PASS for r in results)

    def test_intel_only_fails(self, intel_only):
        results = run_all_checks(intel_only)
        assert any(r.status == CheckStatus.FAIL for r in results)

    def test_unsupported_distro_fails(self):
        state = _make_state(distro_id="ubuntu", distro_name="Ubuntu")
        results = run_all_checks(state)
        assert any(r.status == CheckStatus.FAIL for r in results)

    def test_fail_fast_stops_early(self):
        intel2 = INTEL_GPU.model_copy(update={"pci_slot": "0000:00:12.0"})
        state = _make_state(gpus=[INTEL_GPU, intel2])
        results_normal = run_all_checks(state, fail_fast=False)
        results_fast = run_all_checks(state, fail_fast=True)
        assert len(results_normal) > 1
        assert len(results_fast) == 1
        assert results_fast[0].status == CheckStatus.FAIL

    def test_pro_stack_flag_forwarded(self, amd_desktop):
        with patch("davinci_resolve_checker.checks.check_amd", return_value=[]) as mock_check_amd:
            run_all_checks(amd_desktop, pro_stack=True)
        _, kwargs = mock_check_amd.call_args
        assert kwargs.get("pro_stack") is True

    def test_global_failures_suppress_vendor_passes(self):
        state = _make_state(
            gpus=[AMD_NAVI_GPU, NVIDIA_GPU],
            opencl_drivers=["rocm-opencl-runtime", "opencl-nvidia"],
            opencl_platforms=[ROC_PLATFORM, NVIDIA_CL_PLATFORM],
            opencl_nvidia_installed=True,
            gl_vendor="NVIDIA Corporation",
            gl_renderer="NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2",
        )
        results = run_all_checks(state)
        assert any(r.status == CheckStatus.FAIL for r in results)
        assert all(r.status != CheckStatus.PASS for r in results)
