from __future__ import annotations

from collections.abc import Callable

from davinci_resolve_checker.checks.amd import check_amd
from davinci_resolve_checker.checks.common import (
    check_distro,
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

    for check_fn in _check_sequence(state, pro_stack):
        results.extend(check_fn())
        if fail_fast and any(r.status == CheckStatus.FAIL for r in results):
            break

    return results


def _check_sequence(
    state: SystemState,
    pro_stack: bool,
) -> list[Callable[[], list[CheckResult]]]:
    checks: list[Callable[[], list[CheckResult]]] = [
        lambda: check_distro(state),
        lambda: check_opencl_mesa(state),
        lambda: check_gpu_presence(state),
        lambda: check_gpu_conflict(state),
        lambda: check_gl_vendor(state),
        lambda: check_opencl_platforms(state),
    ]

    has_amd = any(g.vendor == GPUVendor.AMD for g in state.gpus)
    has_nvidia = any(g.vendor == GPUVendor.NVIDIA for g in state.gpus)

    if has_amd:
        checks.append(lambda: check_amd(state, pro_stack=pro_stack))
    if has_nvidia:
        checks.append(lambda: check_nvidia(state))

    return checks
