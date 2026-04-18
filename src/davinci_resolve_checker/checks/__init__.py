from __future__ import annotations

from collections.abc import Callable

from davinci_resolve_checker.checks.amd import check_amd
from davinci_resolve_checker.checks.common import (
    check_distro,
    check_gl_renderer,
    check_gl_vendor,
    check_gpu_conflict,
    check_gpu_presence,
    check_opencl_mesa,
    check_opencl_platforms,
)
from davinci_resolve_checker.checks.nvidia import check_nvidia
from davinci_resolve_checker.models import CheckResult, CheckStatus, GPUVendor, SystemState


def run_all_checks(
    state: SystemState,
    pro_stack: bool = False,
    fail_fast: bool = False,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    for check_fn in _common_check_sequence(state):
        if _append_check_results(results, check_fn(), fail_fast):
            return results

    emit_pass = not any(result.status == CheckStatus.FAIL for result in results)

    has_amd = any(g.vendor == GPUVendor.AMD for g in state.gpus)
    has_nvidia = any(g.vendor == GPUVendor.NVIDIA for g in state.gpus)

    if has_amd and _append_check_results(
        results,
        check_amd(state, pro_stack=pro_stack, emit_pass=emit_pass),
        fail_fast,
    ):
        return results

    if has_nvidia and _append_check_results(
        results,
        check_nvidia(state, emit_pass=emit_pass),
        fail_fast,
    ):
        return results

    return results


def _append_check_results(
    results: list[CheckResult],
    check_results: list[CheckResult],
    fail_fast: bool,
) -> bool:
    for result in check_results:
        results.append(result)
        if fail_fast and result.status == CheckStatus.FAIL:
            return True

    return False


def _common_check_sequence(state: SystemState) -> list[Callable[[], list[CheckResult]]]:
    return [
        lambda: check_distro(state),
        lambda: check_opencl_mesa(state),
        lambda: check_gpu_presence(state),
        lambda: check_gpu_conflict(state),
        lambda: check_gl_vendor(state),
        lambda: check_gl_renderer(state),
        lambda: check_opencl_platforms(state),
    ]
