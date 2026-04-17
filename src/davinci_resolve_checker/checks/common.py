from __future__ import annotations

from davinci_resolve_checker.models import CheckResult, CheckStatus, GPUVendor, SystemState

SUPPORTED_DISTROS = {"arch", "manjaro", "endeavouros", "garuda", "cachyos"}


def check_distro(state: SystemState) -> list[CheckResult]:
    if state.distro_id in SUPPORTED_DISTROS:
        return []
    return [
        CheckResult(
            status=CheckStatus.FAIL,
            message=f"Running {state.distro_name} ({state.distro_id}), which is not tested.",
            suggestion="This tool is designed for Arch-based distributions only.",
        )
    ]


def check_opencl_mesa(state: SystemState) -> list[CheckResult]:
    if "opencl-mesa" in state.opencl_drivers:
        return [
            CheckResult(
                status=CheckStatus.FAIL,
                message="opencl-mesa is installed, which conflicts with DaVinci Resolve.",
                suggestion="Remove opencl-mesa: sudo pacman -R opencl-mesa",
            )
        ]
    return []


def check_gpu_presence(state: SystemState) -> list[CheckResult]:
    if not state.gpus:
        return [
            CheckResult(
                status=CheckStatus.FAIL,
                message="No GPUs detected.",
            )
        ]

    intel_gpus = [g for g in state.gpus if g.vendor == GPUVendor.INTEL]
    nvidia_gpus = [g for g in state.gpus if g.vendor == GPUVendor.NVIDIA]
    amd_gpus = [g for g in state.gpus if g.vendor == GPUVendor.AMD]

    results: list[CheckResult] = []

    if len(intel_gpus) > 1:
        results.append(
            CheckResult(
                status=CheckStatus.FAIL,
                message="Multiple Intel GPUs detected. Configuration not supported.",
            )
        )

    if len(nvidia_gpus) > 1:
        results.append(
            CheckResult(
                status=CheckStatus.FAIL,
                message="Multiple NVIDIA GPUs detected. Configuration not supported.",
            )
        )

    if not amd_gpus and not nvidia_gpus:
        results.append(
            CheckResult(
                status=CheckStatus.FAIL,
                message="Only Intel GPU(s) found. DaVinci Resolve requires an AMD or NVIDIA GPU.",
            )
        )

    return results


def check_gpu_conflict(state: SystemState) -> list[CheckResult]:
    has_amd = any(g.vendor == GPUVendor.AMD for g in state.gpus)
    has_nvidia = any(g.vendor == GPUVendor.NVIDIA for g in state.gpus)
    if has_amd and has_nvidia:
        return [
            CheckResult(
                status=CheckStatus.FAIL,
                message="Both AMD and NVIDIA GPUs detected. This configuration is not supported.",
                suggestion="Remove one GPU, or pass through one via VFIO for VM use.",
            )
        ]
    return []


def check_gl_vendor(state: SystemState) -> list[CheckResult]:
    if not state.gl_vendor:
        return [
            CheckResult(
                status=CheckStatus.FAIL,
                message="OpenGL vendor string is empty.",
                suggestion="Install mesa-utils and verify GPU drivers are loaded.",
            )
        ]
    return []


def check_opencl_platforms(state: SystemState) -> list[CheckResult]:
    usable = [p for p in state.opencl_platforms if p.name != "Clover" and len(p.devices) > 0]
    if not usable:
        return [
            CheckResult(
                status=CheckStatus.FAIL,
                message="No usable OpenCL platforms found.",
                suggestion="Install the appropriate OpenCL driver for your GPU.",
            )
        ]
    return []
