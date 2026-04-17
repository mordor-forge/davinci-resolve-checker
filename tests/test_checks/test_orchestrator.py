from __future__ import annotations

from davinci_resolve_checker.checks import run_all_checks
from davinci_resolve_checker.models import CheckStatus
from tests.conftest import INTEL_GPU, _make_state


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
        state = _make_state(distro_id="ubuntu", distro_name="Ubuntu", gpus=[INTEL_GPU])
        results_normal = run_all_checks(state, fail_fast=False)
        results_fast = run_all_checks(state, fail_fast=True)
        assert len(results_fast) <= len(results_normal)
        assert any(r.status == CheckStatus.FAIL for r in results_fast)

    def test_pro_stack_flag_forwarded(self):
        results = run_all_checks(_make_state(), pro_stack=True)
        assert isinstance(results, list)
