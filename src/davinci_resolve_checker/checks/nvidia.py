from __future__ import annotations

from davinci_resolve_checker.models import CheckResult, CheckStatus, GPUVendor, SystemState


def check_nvidia(state: SystemState) -> list[CheckResult]:
    nvidia_gpus = [g for g in state.gpus if g.vendor == GPUVendor.NVIDIA]
    if not nvidia_gpus:
        return []

    gpu = nvidia_gpus[0]
    results: list[CheckResult] = []

    if not state.opencl_nvidia_installed:
        results.append(
            CheckResult(
                status=CheckStatus.FAIL,
                message="opencl-nvidia package is not installed.",
                suggestion="Install it: sudo pacman -S opencl-nvidia",
            )
        )

    if gpu.driver != "nvidia":
        results.append(
            CheckResult(
                status=CheckStatus.FAIL,
                message=f"NVIDIA GPU is using '{gpu.driver}' driver instead of 'nvidia'.",
                suggestion="Install the proprietary NVIDIA driver.",
            )
        )

    if state.gl_vendor != "NVIDIA Corporation":
        results.append(
            CheckResult(
                status=CheckStatus.FAIL,
                message=f"OpenGL renderer is not NVIDIA (GL_VENDOR: '{state.gl_vendor}').",
                suggestion="Ensure the NVIDIA GPU is the primary renderer.",
            )
        )

    if not results:
        results.append(
            CheckResult(
                status=CheckStatus.PASS,
                message="NVIDIA GPU configuration looks good for DaVinci Resolve.",
            )
        )

    return results
