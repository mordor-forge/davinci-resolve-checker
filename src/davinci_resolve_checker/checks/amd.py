from __future__ import annotations

from davinci_resolve_checker.models import (
    CheckResult,
    CheckStatus,
    GPUDevice,
    GPUVendor,
    SystemState,
)

PRO_GL_VENDOR = "Advanced Micro Devices, Inc."


def check_amd(state: SystemState, pro_stack: bool = False) -> list[CheckResult]:
    amd_gpus = [g for g in state.gpus if g.vendor == GPUVendor.AMD]
    if not amd_gpus:
        return []

    gpu = amd_gpus[0]
    results: list[CheckResult] = []

    results.extend(_check_amd_driver(gpu))
    if any(r.status == CheckStatus.FAIL for r in results):
        return results

    results.extend(_check_amd_mixed_intel(state))
    if any(r.status == CheckStatus.FAIL for r in results):
        return results

    if pro_stack:
        results.extend(_check_amd_pro(state, gpu))
    else:
        results.extend(_check_amd_open(state, gpu))

    if not any(r.status == CheckStatus.FAIL for r in results):
        results.append(
            CheckResult(
                status=CheckStatus.PASS,
                message="AMD GPU configuration looks good for DaVinci Resolve.",
            )
        )

    return results


def _check_amd_driver(gpu: GPUDevice) -> list[CheckResult]:
    if gpu.driver == "amdgpu":
        return []

    if gpu.driver == "radeon":
        if "amdgpu" in gpu.kernel_modules:
            return [
                CheckResult(
                    status=CheckStatus.FAIL,
                    message="GPU is using 'radeon' driver but 'amdgpu' is available.",
                    suggestion=(
                        "Switch to amdgpu by adding 'amdgpu.si_support=1' or "
                        "'amdgpu.cik_support=1' to kernel parameters."
                    ),
                )
            ]
        return [
            CheckResult(
                status=CheckStatus.FAIL,
                message="GPU uses 'radeon' driver which does not support amdgpu. "
                "Cannot run DaVinci Resolve.",
            )
        ]

    return [
        CheckResult(
            status=CheckStatus.FAIL,
            message=f"GPU is not using 'amdgpu' driver (current: '{gpu.driver}'). "
            "Cannot run DaVinci Resolve.",
        )
    ]


def _check_amd_mixed_intel(state: SystemState) -> list[CheckResult]:
    has_intel = any(g.vendor == GPUVendor.INTEL for g in state.gpus)
    if not has_intel:
        return []

    if state.chassis.is_mobile:
        return [
            CheckResult(
                status=CheckStatus.FAIL,
                message="Mixed Intel + AMD GPU on a mobile device is not supported "
                "for DaVinci Resolve.",
            )
        ]

    if state.gl_vendor == "Intel":
        return [
            CheckResult(
                status=CheckStatus.FAIL,
                message="Primary GPU is Intel. Set primary display to PCIe in UEFI settings.",
                suggestion="Go to UEFI/BIOS settings and set primary display to PCIE.",
            )
        ]

    return []


def _check_amd_pro(state: SystemState, gpu: GPUDevice) -> list[CheckResult]:
    results: list[CheckResult] = []

    if gpu.is_pre_vega is False:
        results.append(
            CheckResult(
                status=CheckStatus.WARNING,
                message="Using AMD Pro stack on a modern (Vega+) GPU. "
                "Consider using the open ROCm stack instead.",
            )
        )

    if state.gl_vendor != PRO_GL_VENDOR:
        results.append(
            CheckResult(
                status=CheckStatus.FAIL,
                message="Not using Pro OpenGL implementation.",
                suggestion="Install amdgpu-pro-libgl and run DaVinci Resolve with 'progl' prefix.",
            )
        )

    is_pre_vega = gpu.is_pre_vega
    valid_opencl = ["opencl-amd"]
    if is_pre_vega is True:
        valid_opencl.append("opencl-legacy-amdgpu-pro")

    if not any(drv in state.opencl_drivers for drv in valid_opencl):
        results.append(
            CheckResult(
                status=CheckStatus.FAIL,
                message="Missing required OpenCL driver for AMD Pro stack.",
                suggestion=f"Install one of: {', '.join(valid_opencl)}",
            )
        )

    opencl_ver = state.package_versions.get("opencl-amd", "")
    libgl_ver = state.package_versions.get("amdgpu-pro-libgl", "")
    if opencl_ver and libgl_ver and opencl_ver != libgl_ver:
        results.append(
            CheckResult(
                status=CheckStatus.WARNING,
                message=(
                    f"Version mismatch: opencl-amd ({opencl_ver}) vs "
                    f"amdgpu-pro-libgl ({libgl_ver})."
                ),
                suggestion="Ensure both packages are the same version.",
            )
        )

    return results


def _check_amd_open(state: SystemState, gpu: GPUDevice) -> list[CheckResult]:
    results: list[CheckResult] = []

    is_pre_vega = gpu.is_pre_vega
    if is_pre_vega is None:
        results.append(
            CheckResult(
                status=CheckStatus.WARNING,
                message="AMD GPU codename undetectable. Assuming pre-Vega for safety.",
            )
        )
        is_pre_vega = True

    if is_pre_vega and not state.roc_enable_pre_vega:
        results.append(
            CheckResult(
                status=CheckStatus.FAIL,
                message="Pre-Vega AMD GPU requires ROC_ENABLE_PRE_VEGA=1 environment variable.",
                suggestion="Run: ROC_ENABLE_PRE_VEGA=1 davinci-resolve",
            )
        )

    if not any(drv in state.opencl_drivers for drv in ["rocm-opencl-runtime", "opencl-amd"]):
        results.append(
            CheckResult(
                status=CheckStatus.FAIL,
                message="Missing OpenCL driver for AMD open stack.",
                suggestion="Install rocm-opencl-runtime.",
            )
        )
    elif "opencl-amd" in state.opencl_drivers and "rocm-opencl-runtime" not in state.opencl_drivers:
        results.append(
            CheckResult(
                status=CheckStatus.FAIL,
                message="Using opencl-amd instead of rocm-opencl-runtime.",
                suggestion="Use rocm-opencl-runtime for the open stack.",
            )
        )

    return results
